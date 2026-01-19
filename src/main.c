// trm_trading.c
// Build:   gcc -O2 -std=c11 trm_trading.c -lm -o trm_trading
// Run:     ./trm_trading [ohlcv_csv] [mlp_layers] [--load model.bin] [--save model.bin] [--eval]
//
// Trains a tiny "recursive" model (TRM-like) to predict buy/sell/hold
// using real OHLCV data from a CSV file (see scripts/fetch_binance_ohlcv.py).
//
// Notes:
// - Uses tanh activation (more stable than ReLU for recursion)
// - Correct gradients (uses "old" weights when computing dh for Wo/Wh)
// - Optional class weights + grad clipping for stability

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include "gemm.h"

// ---------- utils ----------
static unsigned int g_rng = 123456789u;

static inline unsigned int xorshift32(void) {
    unsigned int x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

static inline double urand01(void) {
    return (xorshift32() / (double)UINT_MAX);
}

static inline float clipf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static void softmax3(const float z[3], float p[3]) {
    float m = z[0];
    if (z[1] > m) m = z[1];
    if (z[2] > m) m = z[2];
    float e0 = expf(z[0] - m);
    float e1 = expf(z[1] - m);
    float e2 = expf(z[2] - m);
    float s = e0 + e1 + e2;
    p[0] = e0 / s;
    p[1] = e1 / s;
    p[2] = e2 / s;
}

static inline float act(float x) { return tanhf(x); }
static inline float dact_from_pre(float pre) {
    float t = tanhf(pre);
    return 1.0f - t * t;
}

static int argmax3(const float p[3]) {
    int a = 0;
    if (p[1] > p[a]) a = 1;
    if (p[2] > p[a]) a = 2;
    return a;
}

typedef struct {
    double o, h, l, c, v;
} Bar;

static int parse_csv_doubles(const char *line, double *vals, int max_vals) {
    const char *p = line;
    int n = 0;
    while (n < max_vals) {
        char *end = NULL;
        double v = strtod(p, &end);
        if (end == p) break;
        vals[n++] = v;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
    }
    return n;
}

static int load_ohlcv_csv(const char *path, Bar **out_bars) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    size_t cap = 4096;
    size_t n = 0;
    Bar *bars = (Bar*)calloc(cap, sizeof(Bar));
    if (!bars) {
        fclose(f);
        return 0;
    }

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        double vals[6];
        int nvals = parse_csv_doubles(line, vals, 6);
        if (nvals < 6) continue;
        if (n >= cap) {
            cap *= 2;
            Bar *tmp = (Bar*)realloc(bars, cap * sizeof(Bar));
            if (!tmp) {
                free(bars);
                fclose(f);
                return 0;
            }
            bars = tmp;
        }
        bars[n].o = vals[1];
        bars[n].h = vals[2];
        bars[n].l = vals[3];
        bars[n].c = vals[4];
        bars[n].v = vals[5];
        n++;
    }
    fclose(f);

    *out_bars = bars;
    return (int)n;
}

// ---------- features ----------
typedef struct {
    // 8 features per sample
    // [0] ret1 scaled
    // [1] ret5 scaled
    // [2] mom10 (close/sma10 - 1) scaled
    // [3] vol10 (std of ret1 over 10) scaled
    // [4] range_pct ((h-l)/c) scaled
    // [5] vol_norm (v/ema_v - 1) scaled
    // [6] rsi_14 scaled to [-1..1]
    // [7] bias 1
    float x[8];
    int y; // 0=SELL, 1=HOLD, 2=BUY
} Sample;

static double mean(const double *a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i];
    return s / (double)n;
}

static double stddev(const double *a, int n) {
    double m = mean(a, n);
    double s2 = 0.0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - m;
        s2 += d * d;
    }
    return sqrt(s2 / (double)(n > 1 ? (n - 1) : 1));
}

static double sma_close(const Bar *b, int i, int win) {
    double s = 0.0;
    for (int k = i - win + 1; k <= i; k++) s += b[k].c;
    return s / (double)win;
}

static double ema_update(double prev, double x, double alpha) {
    return alpha * x + (1.0 - alpha) * prev;
}

static double rsi14(const Bar *b, int i) {
    // simple RSI over last 14 deltas
    double gain = 0.0, loss = 0.0;
    for (int k = i - 13; k <= i; k++) {
        double ch = b[k].c - b[k-1].c;
        if (ch >= 0) gain += ch;
        else loss += -ch;
    }
    gain /= 14.0;
    loss /= 14.0;
    if (loss < 1e-12) return 100.0;
    double rs = gain / loss;
    double rsi = 100.0 - (100.0 / (1.0 + rs));
    return rsi;
}

static int make_samples(const Bar *bars, int n, Sample *out, int horizon, double thr) {
    // label by future return at i+horizon:
    // BUY if >= +thr, SELL if <= -thr, else HOLD
    int idx = 0;
    double ema_v = bars[0].v;
    const double alpha_v = 2.0 / (30.0 + 1.0); // EMA ~30

    // precompute 1-step returns
    double *r1 = (double*)calloc((size_t)n, sizeof(double));
    for (int i = 1; i < n; i++) r1[i] = (bars[i].c / bars[i-1].c) - 1.0;

    for (int i = 50; i < n - horizon; i++) {
        ema_v = ema_update(ema_v, bars[i].v, alpha_v);

        double ret1 = r1[i];
        double ret5 = (bars[i].c / bars[i-5].c) - 1.0;

        double sma10 = sma_close(bars, i, 10);
        double mom10 = (bars[i].c / sma10) - 1.0;

        double window_r[10];
        for (int k = 0; k < 10; k++) window_r[k] = r1[i-k];
        double vol10 = stddev(window_r, 10);

        double range_pct = (bars[i].h - bars[i].l) / bars[i].c;
        double vol_norm = (bars[i].v / (ema_v + 1e-9)) - 1.0;

        double rsi = rsi14(bars, i);
        double rsi_scaled = (rsi - 50.0) / 50.0; // [-1..+1]

        double fut = (bars[i + horizon].c / bars[i].c) - 1.0;
        int y = 1; // HOLD
        if (fut >= thr) y = 2;       // BUY
        else if (fut <= -thr) y = 0; // SELL

        // light scaling (keeps values in nicer ranges)
        out[idx].x[0] = (float)(ret1 * 50.0);
        out[idx].x[1] = (float)(ret5 * 20.0);
        out[idx].x[2] = (float)(mom10 * 30.0);
        out[idx].x[3] = (float)(vol10 * 200.0);
        out[idx].x[4] = (float)(range_pct * 50.0);
        out[idx].x[5] = (float)(vol_norm * 3.0);
        out[idx].x[6] = (float)rsi_scaled;
        out[idx].x[7] = 1.0f;
        out[idx].y = y;
        idx++;
    }

    free(r1);
    return idx;
}

// ---------- tiny recursive model (TRM-like) ----------
#define IN_DIM 8
#define H_DIM 96
#define OUT_DIM 3
#define MAX_K 63
#define MAX_MLP_LAYERS 4

typedef struct {
    // Shared recursive block: h_t = act(b + Wx*x + Wh*h_{t-1})
    float Wx[H_DIM][IN_DIM];
    float Wh[H_DIM][H_DIM];
    float b[H_DIM];

    // MLP layers on top of h_K
    int mlp_layers;
    float Wmlp[MAX_MLP_LAYERS][H_DIM][H_DIM];
    float bmlp[MAX_MLP_LAYERS][H_DIM];

    // Output: logits = Wo*h + bo
    float Wo[OUT_DIM][H_DIM];
    float bo[OUT_DIM];
} Model;

typedef struct {
    float pre_rec[MAX_K+1][H_DIM];
    float h_rec[MAX_K+1][H_DIM];
    float pre_mlp[MAX_MLP_LAYERS][H_DIM];
    float h_mlp[MAX_MLP_LAYERS][H_DIM];
    int K;
    int mlp_layers;
} Cache;

typedef struct {
    char magic[4];
    int version;
    int model_bytes;
} ModelHeader;

static float frand(float scale) {
    return (float)((urand01() * 2.0 - 1.0) * scale);
}

static void model_init(Model *m, int mlp_layers) {
    // small init is important for stable recursion
    float sWx = 0.08f, sWh = 0.05f, sWo = 0.08f;
    for (int i = 0; i < H_DIM; i++) {
        for (int j = 0; j < IN_DIM; j++) m->Wx[i][j] = frand(sWx);
        for (int j = 0; j < H_DIM; j++) m->Wh[i][j] = frand(sWh);
        m->b[i] = 0.01f + frand(0.01f); // small positive bias helps
    }

    m->mlp_layers = mlp_layers;
    for (int l = 0; l < mlp_layers; l++) {
        for (int i = 0; i < H_DIM; i++) {
            for (int j = 0; j < H_DIM; j++) m->Wmlp[l][i][j] = frand(sWx);
            m->bmlp[l][i] = frand(0.02f);
        }
    }

    for (int o = 0; o < OUT_DIM; o++) {
        for (int j = 0; j < H_DIM; j++) m->Wo[o][j] = frand(sWo);
        m->bo[o] = frand(0.02f);
    }
}

static void forward(const Model *m, const float x[IN_DIM], int K, float logits[OUT_DIM], Cache *cache) {
    if (K > MAX_K) K = MAX_K;
    if (K < 1) K = 1;
    cache->K = K;
    cache->mlp_layers = m->mlp_layers;

    // h0 = 0
    zero_buf(&cache->h_rec[0][0], H_DIM);
    zero_buf(&cache->pre_rec[0][0], H_DIM);

    float tmp_wx[H_DIM];
    float tmp_wh[H_DIM];

    // recursion
    for (int t = 1; t <= K; t++) {
        matmul_nn(&m->Wx[0][0], x, tmp_wx, H_DIM, 1, IN_DIM);
        matmul_nn(&m->Wh[0][0], cache->h_rec[t-1], tmp_wh, H_DIM, 1, H_DIM);
        for (int i = 0; i < H_DIM; i++) {
            float pre = m->b[i] + tmp_wx[i] + tmp_wh[i];
            cache->pre_rec[t][i] = pre;
            cache->h_rec[t][i] = act(pre);
        }
    }

    const float *mlp_in = cache->h_rec[K];
    for (int l = 0; l < m->mlp_layers; l++) {
        matmul_nn(&m->Wmlp[l][0][0], mlp_in, cache->pre_mlp[l], H_DIM, 1, H_DIM);
        for (int i = 0; i < H_DIM; i++) {
            float pre = cache->pre_mlp[l][i] + m->bmlp[l][i];
            cache->pre_mlp[l][i] = pre;
            cache->h_mlp[l][i] = act(pre);
        }
        mlp_in = cache->h_mlp[l];
    }

    const float *out_in = (m->mlp_layers > 0) ? cache->h_mlp[m->mlp_layers - 1] : cache->h_rec[K];
    matmul_nn(&m->Wo[0][0], out_in, logits, OUT_DIM, 1, H_DIM);
    for (int o = 0; o < OUT_DIM; o++) {
        logits[o] += m->bo[o];
    }
}

static double train_step(Model *m, const Sample *s, int K, double lr) {
    Cache cache;
    float logits[3], p[3];
    forward(m, s->x, K, logits, &cache);
    softmax3(logits, p);

    // cross-entropy
    double loss = -log((double)p[s->y] + 1e-12);

    // class weights (tune if HOLD dominates)
    const float class_w[3] = {1.0f, 0.6f, 1.0f};
    float w = class_w[s->y];

    // dL/dlogits = (p - y_onehot) * w
    float dlogits[3] = { p[0], p[1], p[2] };
    dlogits[s->y] -= 1.0f;
    dlogits[0] *= w; dlogits[1] *= w; dlogits[2] *= w;

    // grad buffers
    float dWo[OUT_DIM][H_DIM];
    float dbo[OUT_DIM];
    float dWx[H_DIM][IN_DIM];
    float dWh[H_DIM][H_DIM];
    float db[H_DIM];
    float dWmlp[MAX_MLP_LAYERS][H_DIM][H_DIM];
    float dbmlp[MAX_MLP_LAYERS][H_DIM];

    zero_buf(&dWo[0][0], OUT_DIM * H_DIM);
    zero_buf(dbo, OUT_DIM);
    zero_buf(&dWx[0][0], H_DIM * IN_DIM);
    zero_buf(&dWh[0][0], H_DIM * H_DIM);
    zero_buf(db, H_DIM);
    zero_buf(&dWmlp[0][0][0], MAX_MLP_LAYERS * H_DIM * H_DIM);
    zero_buf(&dbmlp[0][0], MAX_MLP_LAYERS * H_DIM);

    // output grads and dh using OLD Wo
    for (int o = 0; o < OUT_DIM; o++) {
        dbo[o] += dlogits[o];
    }

    const float *out_in = (m->mlp_layers > 0) ? cache.h_mlp[m->mlp_layers - 1] : cache.h_rec[cache.K];
    for (int o = 0; o < OUT_DIM; o++) {
        for (int j = 0; j < H_DIM; j++) {
            dWo[o][j] += dlogits[o] * out_in[j];
        }
    }

    float dh[H_DIM];
    matmul_tn(&m->Wo[0][0], dlogits, dh, H_DIM, 1, OUT_DIM);

    // backprop through MLP layers
    for (int l = m->mlp_layers - 1; l >= 0; l--) {
        float dpre[H_DIM];
        for (int i = 0; i < H_DIM; i++) {
            dpre[i] = dh[i] * dact_from_pre(cache.pre_mlp[l][i]);
            dbmlp[l][i] += dpre[i];
        }

        const float *mlp_in = (l == 0) ? cache.h_rec[cache.K] : cache.h_mlp[l - 1];
        matmul_nt(dpre, mlp_in, &dWmlp[l][0][0], H_DIM, H_DIM, 1);
        matmul_tn(&m->Wmlp[l][0][0], dpre, dh, H_DIM, 1, H_DIM);
    }

    // BPTT through recursion using OLD Wh
    float dh_next[H_DIM];
    for (int j = 0; j < H_DIM; j++) dh_next[j] = dh[j];

    for (int t = cache.K; t >= 1; t--) {
        float dpre[H_DIM];
        for (int i = 0; i < H_DIM; i++) {
            dpre[i] = dh_next[i] * dact_from_pre(cache.pre_rec[t][i]);
            db[i] += dpre[i];
        }

        float tmp_wx[H_DIM * IN_DIM];
        float tmp_wh[H_DIM * H_DIM];
        zero_buf(tmp_wx, H_DIM * IN_DIM);
        zero_buf(tmp_wh, H_DIM * H_DIM);
        matmul_nt(dpre, s->x, tmp_wx, H_DIM, IN_DIM, 1);
        matmul_nt(dpre, cache.h_rec[t-1], tmp_wh, H_DIM, H_DIM, 1);
        add_inplace(&dWx[0][0], tmp_wx, H_DIM * IN_DIM);
        add_inplace(&dWh[0][0], tmp_wh, H_DIM * H_DIM);

        matmul_tn(&m->Wh[0][0], dpre, dh_next, H_DIM, 1, H_DIM);
    }

    // grad clipping (elementwise)
    const float clipv = 1.0f;
    for (int o = 0; o < OUT_DIM; o++) {
        dbo[o] = clipf(dbo[o], -clipv, clipv);
        for (int j = 0; j < H_DIM; j++) dWo[o][j] = clipf(dWo[o][j], -clipv, clipv);
    }
    for (int i = 0; i < H_DIM; i++) {
        db[i] = clipf(db[i], -clipv, clipv);
        for (int j = 0; j < IN_DIM; j++) dWx[i][j] = clipf(dWx[i][j], -clipv, clipv);
        for (int j = 0; j < H_DIM; j++) dWh[i][j] = clipf(dWh[i][j], -clipv, clipv);
    }
    for (int l = 0; l < m->mlp_layers; l++) {
        for (int i = 0; i < H_DIM; i++) {
            dbmlp[l][i] = clipf(dbmlp[l][i], -clipv, clipv);
            for (int j = 0; j < H_DIM; j++) dWmlp[l][i][j] = clipf(dWmlp[l][i][j], -clipv, clipv);
        }
    }

    // SGD update
    for (int o = 0; o < OUT_DIM; o++) {
        m->bo[o] -= (float)(lr * dbo[o]);
        for (int j = 0; j < H_DIM; j++) {
            m->Wo[o][j] -= (float)(lr * dWo[o][j]);
        }
    }
    for (int i = 0; i < H_DIM; i++) {
        m->b[i] -= (float)(lr * db[i]);
        for (int j = 0; j < IN_DIM; j++) {
            m->Wx[i][j] -= (float)(lr * dWx[i][j]);
        }
        for (int j = 0; j < H_DIM; j++) {
            m->Wh[i][j] -= (float)(lr * dWh[i][j]);
        }
    }
    for (int l = 0; l < m->mlp_layers; l++) {
        for (int i = 0; i < H_DIM; i++) {
            m->bmlp[l][i] -= (float)(lr * dbmlp[l][i]);
            for (int j = 0; j < H_DIM; j++) {
                m->Wmlp[l][i][j] -= (float)(lr * dWmlp[l][i][j]);
            }
        }
    }

    return loss;
}

static double eval_accuracy(const Model *m, const Sample *S, int n, int K) {
    int correct = 0;
    Cache cache;
    for (int i = 0; i < n; i++) {
        float logits[3], p[3];
        forward(m, S[i].x, K, logits, &cache);
        softmax3(logits, p);
        int pred = argmax3(p);
        if (pred == S[i].y) correct++;
    }
    return (double)correct / (double)n;
}

static void evaluate_model(const Model *m, const Sample *S, int n, int K) {
    int conf[3][3] = {0};
    Cache cache;
    for (int i = 0; i < n; i++) {
        float logits[3], p[3];
        forward(m, S[i].x, K, logits, &cache);
        softmax3(logits, p);
        int pred = argmax3(p);
        conf[S[i].y][pred]++;
    }

    printf("\nFinal evaluation:\n");
    printf("Confusion matrix (rows=true, cols=pred):\n");
    for (int r = 0; r < 3; r++) {
        printf("%d %d %d\n", conf[r][0], conf[r][1], conf[r][2]);
    }

    int correct = conf[0][0] + conf[1][1] + conf[2][2];
    printf("Accuracy: %.3f\n", (double)correct / (double)n);
}

static int save_model(const char *path, const Model *m) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return 0;
    }
    ModelHeader header = { {'T', 'R', 'M', '1'}, 1, (int)sizeof(Model) };
    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return 0;
    }
    if (fwrite(m, sizeof(Model), 1, f) != 1) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static int load_model(const char *path, Model *m) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return 0;
    }
    ModelHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return 0;
    }
    if (memcmp(header.magic, "TRM1", 4) != 0 || header.version != 1 ||
        header.model_bytes != (int)sizeof(Model)) {
        fclose(f);
        return 0;
    }
    if (fread(m, sizeof(Model), 1, f) != 1) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

// ---------- main ----------
int main(int argc, char **argv) {
    const char *csv_path = "binance_ohlcv.csv";
    int mlp_layers = 1;
    const char *save_path = NULL;
    const char *load_path = NULL;
    int eval_only = 0;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            load_path = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--eval") == 0) {
            eval_only = 1;
            continue;
        }
        if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
        if (positional == 0) {
            csv_path = argv[i];
            positional++;
        } else if (positional == 1) {
            mlp_layers = atoi(argv[i]);
            positional++;
        }
    }
    if (mlp_layers < 0) mlp_layers = 0;
    if (mlp_layers > MAX_MLP_LAYERS) mlp_layers = MAX_MLP_LAYERS;

    Bar *bars = NULL;
    int nBars = load_ohlcv_csv(csv_path, &bars);
    if (nBars <= 0) {
        fprintf(stderr, "Failed to load OHLCV data from %s\n", csv_path);
        return 1;
    }
    printf("Loaded %d bars from %s\n", nBars, csv_path);

    // 2) samples
    const int HORIZON = 8;      // label based on close at t+8
    const double THR = 0.004;   // +/-0.4% threshold for BUY/SELL (try 0.002 if HOLD dominates)
    Sample *samples = (Sample*)calloc((size_t)nBars, sizeof(Sample));
    int nS = make_samples(bars, nBars, samples, HORIZON, THR);

    // label distribution
    int cnt[3] = {0, 0, 0};
    for (int i = 0; i < nS; i++) cnt[samples[i].y]++;
    printf("Samples=%d | Label dist: SELL=%d HOLD=%d BUY=%d\n", nS, cnt[0], cnt[1], cnt[2]);

    // 3) split train/test (time-based)
    int nTrain = (int)(nS * 0.8);
    int nTest = nS - nTrain;
    Sample *trainS = samples;
    Sample *testS = samples + nTrain;

    // 4) init model
    Model m;
    if (load_path) {
        if (!load_model(load_path, &m)) {
            fprintf(stderr, "Failed to load model from %s\n", load_path);
            return 1;
        }
        mlp_layers = m.mlp_layers;
        printf("Loaded model from %s (mlp_layers=%d)\n", load_path, mlp_layers);
    } else {
        model_init(&m, mlp_layers);
    }

    if (eval_only) {
        evaluate_model(&m, testS, nTest, K);
        free(samples);
        free(bars);
        return 0;
    }

    // 5) train
    const int K = 4;            // start with 4 (stable), try 8 later
    const int EPOCHS = 200;
    double lr = 0.001;

    for (int e = 1; e <= EPOCHS; e++) {
        double loss_sum = 0.0;
        for (int i = 0; i < nTrain; i++) {
            loss_sum += train_step(&m, &trainS[i], K, lr);
        }
        double train_acc = eval_accuracy(&m, trainS, nTrain, K);
        double test_acc  = eval_accuracy(&m, testS, nTest, K);

        printf("Epoch %d | loss=%.4f | train_acc=%.3f | test_acc=%.3f | lr=%.6f\n",
               e, loss_sum / (double)nTrain, train_acc, test_acc, lr);
    }

    evaluate_model(&m, testS, nTest, K);

    if (save_path) {
        if (save_model(save_path, &m)) {
            printf("Saved model to %s\n", save_path);
        } else {
            fprintf(stderr, "Failed to save model to %s\n", save_path);
            return 1;
        }
    }

    // 6) show a few predictions
    printf("\nSample predictions (0=SELL,1=HOLD,2=BUY):\n");
    for (int i = 0; i < 10 && i < nTest; i++) {
        Cache cache;
        float logits[3], p[3];
        forward(&m, testS[i].x, K, logits, &cache);
        softmax3(logits, p);
        int pred = argmax3(p);
        printf("y=%d pred=%d probs=[%.2f %.2f %.2f]\n", testS[i].y, pred, p[0], p[1], p[2]);
    }

    free(samples);
    free(bars);
    return 0;
}
