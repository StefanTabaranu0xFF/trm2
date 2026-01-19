// trm_trading.c
// Build:   gcc -O2 -std=c11 trm_trading.c -lm -o trm_trading
// Run:     ./trm_trading [ohlcv_csv]
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

static inline double clip(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static void softmax3(const double z[3], double p[3]) {
    double m = z[0];
    if (z[1] > m) m = z[1];
    if (z[2] > m) m = z[2];
    double e0 = exp(z[0] - m);
    double e1 = exp(z[1] - m);
    double e2 = exp(z[2] - m);
    double s = e0 + e1 + e2;
    p[0] = e0 / s;
    p[1] = e1 / s;
    p[2] = e2 / s;
}

static inline double act(double x) { return tanh(x); }
static inline double dact_from_pre(double pre) {
    double t = tanh(pre);
    return 1.0 - t * t;
}

static int argmax3(const double p[3]) {
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
    double x[8];
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
        out[idx].x[0] = ret1 * 50.0;
        out[idx].x[1] = ret5 * 20.0;
        out[idx].x[2] = mom10 * 30.0;
        out[idx].x[3] = vol10 * 200.0;
        out[idx].x[4] = range_pct * 50.0;
        out[idx].x[5] = vol_norm * 3.0;
        out[idx].x[6] = rsi_scaled;
        out[idx].x[7] = 1.0;
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

typedef struct {
    // Shared recursive block: h_t = act(b + Wx*x + Wh*h_{t-1})
    double Wx[H_DIM][IN_DIM];
    double Wh[H_DIM][H_DIM];
    double b[H_DIM];

    // Output: logits = Wo*hK + bo
    double Wo[OUT_DIM][H_DIM];
    double bo[OUT_DIM];
} Model;

typedef struct {
    double pre[MAX_K+1][H_DIM];
    double h[MAX_K+1][H_DIM];
    int K;
} Cache;

static double frand(double scale) {
    return (urand01() * 2.0 - 1.0) * scale;
}

static void model_init(Model *m) {
    // small init is important for stable recursion
    double sWx = 0.08, sWh = 0.05, sWo = 0.08;
    for (int i = 0; i < H_DIM; i++) {
        for (int j = 0; j < IN_DIM; j++) m->Wx[i][j] = frand(sWx);
        for (int j = 0; j < H_DIM; j++) m->Wh[i][j] = frand(sWh);
        m->b[i] = 0.01 + frand(0.01); // small positive bias helps
    }
    for (int o = 0; o < OUT_DIM; o++) {
        for (int j = 0; j < H_DIM; j++) m->Wo[o][j] = frand(sWo);
        m->bo[o] = frand(0.02);
    }
}

static void forward(const Model *m, const double x[IN_DIM], int K, double logits[OUT_DIM], Cache *cache) {
    if (K > MAX_K) K = MAX_K;
    if (K < 1) K = 1;
    cache->K = K;

    // h0 = 0
    for (int j = 0; j < H_DIM; j++) {
        cache->h[0][j] = 0.0;
        cache->pre[0][j] = 0.0;
    }

    // recursion
    for (int t = 1; t <= K; t++) {
        for (int i = 0; i < H_DIM; i++) {
            double s = m->b[i];
            for (int j = 0; j < IN_DIM; j++) s += m->Wx[i][j] * x[j];
            for (int j = 0; j < H_DIM; j++) s += m->Wh[i][j] * cache->h[t-1][j];
            cache->pre[t][i] = s;
            cache->h[t][i] = act(s);
        }
    }

    // output
    for (int o = 0; o < OUT_DIM; o++) {
        double s = m->bo[o];
        for (int j = 0; j < H_DIM; j++) s += m->Wo[o][j] * cache->h[K][j];
        logits[o] = s;
    }
}

static double train_step(Model *m, const Sample *s, int K, double lr) {
    Cache cache;
    double logits[3], p[3];
    forward(m, s->x, K, logits, &cache);
    softmax3(logits, p);

    // cross-entropy
    double loss = -log(p[s->y] + 1e-12);

    // class weights (tune if HOLD dominates)
    const double class_w[3] = {1.0, 0.6, 1.0};
    const double w = class_w[s->y];

    // dL/dlogits = (p - y_onehot) * w
    double dlogits[3] = { p[0], p[1], p[2] };
    dlogits[s->y] -= 1.0;
    dlogits[0] *= w; dlogits[1] *= w; dlogits[2] *= w;

    // grad buffers
    double dWo[OUT_DIM][H_DIM]; memset(dWo, 0, sizeof(dWo));
    double dbo[OUT_DIM];        memset(dbo, 0, sizeof(dbo));
    double dWx[H_DIM][IN_DIM];  memset(dWx, 0, sizeof(dWx));
    double dWh[H_DIM][H_DIM];   memset(dWh, 0, sizeof(dWh));
    double db[H_DIM];           memset(db,  0, sizeof(db));

    // output grads and dhK using OLD Wo
    double dh_next[H_DIM];
    for (int j = 0; j < H_DIM; j++) dh_next[j] = 0.0;

    for (int o = 0; o < OUT_DIM; o++) {
        dbo[o] += dlogits[o];
        for (int j = 0; j < H_DIM; j++) {
            dWo[o][j] += dlogits[o] * cache.h[cache.K][j];
            dh_next[j] += dlogits[o] * m->Wo[o][j]; // old Wo
        }
    }

    // BPTT through recursion using OLD Wh
    for (int t = cache.K; t >= 1; t--) {
        double dpre[H_DIM];
        for (int i = 0; i < H_DIM; i++) {
            dpre[i] = dh_next[i] * dact_from_pre(cache.pre[t][i]);
        }

        // accumulate grads for params
        for (int i = 0; i < H_DIM; i++) {
            db[i] += dpre[i];
            for (int j = 0; j < IN_DIM; j++) {
                dWx[i][j] += dpre[i] * s->x[j];
            }
            for (int j = 0; j < H_DIM; j++) {
                dWh[i][j] += dpre[i] * cache.h[t-1][j];
            }
        }

        // propagate to previous h using old Wh
        double dh_prev[H_DIM];
        for (int j = 0; j < H_DIM; j++) {
            double sum = 0.0;
            for (int i = 0; i < H_DIM; i++) {
                sum += dpre[i] * m->Wh[i][j]; // old Wh
            }
            dh_prev[j] = sum;
        }
        for (int j = 0; j < H_DIM; j++) dh_next[j] = dh_prev[j];
    }

    // grad clipping (elementwise)
    const double clipv = 1.0;
    for (int o = 0; o < OUT_DIM; o++) {
        dbo[o] = clip(dbo[o], -clipv, clipv);
        for (int j = 0; j < H_DIM; j++) dWo[o][j] = clip(dWo[o][j], -clipv, clipv);
    }
    for (int i = 0; i < H_DIM; i++) {
        db[i] = clip(db[i], -clipv, clipv);
        for (int j = 0; j < IN_DIM; j++) dWx[i][j] = clip(dWx[i][j], -clipv, clipv);
        for (int j = 0; j < H_DIM; j++) dWh[i][j] = clip(dWh[i][j], -clipv, clipv);
    }

    // SGD update
    for (int o = 0; o < OUT_DIM; o++) {
        m->bo[o] -= lr * dbo[o];
        for (int j = 0; j < H_DIM; j++) {
            m->Wo[o][j] -= lr * dWo[o][j];
        }
    }
    for (int i = 0; i < H_DIM; i++) {
        m->b[i] -= lr * db[i];
        for (int j = 0; j < IN_DIM; j++) {
            m->Wx[i][j] -= lr * dWx[i][j];
        }
        for (int j = 0; j < H_DIM; j++) {
            m->Wh[i][j] -= lr * dWh[i][j];
        }
    }

    return loss;
}

static double eval_accuracy(const Model *m, const Sample *S, int n, int K) {
    int correct = 0;
    Cache cache;
    for (int i = 0; i < n; i++) {
        double logits[3], p[3];
        forward(m, S[i].x, K, logits, &cache);
        softmax3(logits, p);
        int pred = argmax3(p);
        if (pred == S[i].y) correct++;
    }
    return (double)correct / (double)n;
}

// ---------- main ----------
int main(int argc, char **argv) {
    const char *csv_path = (argc > 1) ? argv[1] : "binance_ohlcv.csv";
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
    model_init(&m);

    // 5) train
    const int K = 4;            // start with 4 (stable), try 8 later
    const int EPOCHS = 550;
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

        // mild LR decay
        // lr *= 0.98;
    }

    // 6) show a few predictions
    printf("\nSample predictions (0=SELL,1=HOLD,2=BUY):\n");
    for (int i = 0; i < 10 && i < nTest; i++) {
        Cache cache;
        double logits[3], p[3];
        forward(&m, testS[i].x, K, logits, &cache);
        softmax3(logits, p);
        int pred = argmax3(p);
        printf("y=%d pred=%d probs=[%.2f %.2f %.2f]\n", testS[i].y, pred, p[0], p[1], p[2]);
    }

    free(samples);
    free(bars);
    return 0;
}
