// trm_toy.c
// Minimal TRM-like implementation in pure C (no external deps).
// Task: classify candle-feature vectors into {SHORT, WAIT, LONG} using recursive latent refinement.
//
// Build:
//   gcc -O2 -std=c11 -lm trm_toy.c -o trm_toy
//
// Run:
//   ./trm_toy
//
// Notes:
// - This is a compact "toy" TRM. It demonstrates the algorithmic loop (z updates, then y update, deep supervision).
// - Uses AdamW-ish optimizer (simplified).
// - Generates mock candle data (random walk). Label from future return.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------------- utils / rng -------------------------

static uint64_t g_rng = 0x123456789abcdef0ULL;

static uint64_t xorshift64star(void) {
    uint64_t x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float frand_uniform(void) {
    // [0,1)
    uint32_t r = (uint32_t)(xorshift64star() >> 32);
    return (r / 4294967296.0f);
}

static float frand_normal(void) {
    // Box-Muller
    float u1 = fmaxf(frand_uniform(), 1e-7f);
    float u2 = frand_uniform();
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return z0;
}

static float clampf(float x, float lo, float hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

// ------------------------- hyperparams -------------------------

enum { CLS_SHORT = 0, CLS_WAIT = 1, CLS_LONG = 2 };

#define D   32        // latent dimension for x_emb, y, z
#define F   16        // raw feature dimension from candles
#define H   64        // hidden width of tiny MLP (2-layer)
#define N_SUP 16      // deep supervision steps
#define T_REC 3       // T in paper (deep recursion cycles)
#define N_LAT 6       // n in paper (latent z updates per cycle)

#define TRAIN_SAMPLES 4000
#define TEST_SAMPLES  1000

#define LR     1e-3f
#define WD     1e-2f
#define BETA1  0.9f
#define BETA2  0.95f
#define EPS    1e-8f

#define EPOCHS  10
#define BATCH   64

// ------------------------- tiny MLP "net" -------------------------
//
// net(v_in) -> v_out of size D
// two-layer: Linear(in_dim -> H), SiLU, Linear(H -> D)
//
// We'll use *one* shared net with fixed input size D by construction:
// - x_emb, y, z are all D.
// - z-update input: v = x_emb + y + z  (D)
// - y-update input: v = y + z          (D)
//
// So net input dim = D, output dim = D.

typedef struct {
    // W1: [H, D], b1: [H]
    float W1[H][D];
    float b1[H];

    // W2: [D, H], b2: [D]
    float W2[D][H];
    float b2[D];

    // Adam moments
    float mW1[H][D], vW1[H][D];
    float mb1[H], vb1[H];
    float mW2[D][H], vW2[D][H];
    float mb2[D], vb2[D];

    int64_t step;
} Net;

// SiLU: x * sigmoid(x)
static inline float sigmoidf_fast(float x) {
    // stable-ish sigmoid
    if (x >= 0) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static inline float siluf(float x) {
    return x * sigmoidf_fast(x);
}

static inline float dsiluf(float x) {
    // derivative: sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    float s = sigmoidf_fast(x);
    return s + x * s * (1.0f - s);
}

static void net_init(Net *net, float scale) {
    memset(net, 0, sizeof(*net));
    net->step = 0;
    for (int i = 0; i < H; i++) {
        net->b1[i] = 0.0f;
        for (int j = 0; j < D; j++) {
            net->W1[i][j] = frand_normal() * scale;
        }
    }
    for (int i = 0; i < D; i++) {
        net->b2[i] = 0.0f;
        for (int j = 0; j < H; j++) {
            net->W2[i][j] = frand_normal() * scale;
        }
    }
}

typedef struct {
    float pre1[H];   // a1 = W1*x + b1
    float act1[H];   // h = SiLU(a1)
    float out[D];    // y = W2*h + b2
    float in[D];     // cached input
} NetCache;

static void net_forward(const Net *net, const float in[D], NetCache *c) {
    memcpy(c->in, in, sizeof(float) * D);
    for (int i = 0; i < H; i++) {
        float s = net->b1[i];
        for (int j = 0; j < D; j++) s += net->W1[i][j] * in[j];
        c->pre1[i] = s;
        c->act1[i] = siluf(s);
    }
    for (int i = 0; i < D; i++) {
        float s = net->b2[i];
        for (int j = 0; j < H; j++) s += net->W2[i][j] * c->act1[j];
        c->out[i] = s;
    }
}

typedef struct {
    float dW1[H][D];
    float db1[H];
    float dW2[D][H];
    float db2[D];
} NetGrad;

static void netgrad_zero(NetGrad *g) {
    memset(g, 0, sizeof(*g));
}

static void net_backward(const Net *net, const NetCache *c, const float dout[D], NetGrad *g, float din[D]) {
    // dout is grad wrt output (size D)
    // Compute grads for W2,b2, then backprop to act1, pre1, then W1,b1, then input.

    // dW2[i][j] += dout[i] * act1[j]
    // db2[i] += dout[i]
    float dact1[H] = {0};

    for (int i = 0; i < D; i++) {
        g->db2[i] += dout[i];
        for (int j = 0; j < H; j++) {
            g->dW2[i][j] += dout[i] * c->act1[j];
            dact1[j] += net->W2[i][j] * dout[i];
        }
    }

    // back through SiLU
    float dpre1[H];
    for (int i = 0; i < H; i++) {
        dpre1[i] = dact1[i] * dsiluf(c->pre1[i]);
        g->db1[i] += dpre1[i];
    }

    // dW1[i][j] += dpre1[i] * in[j]
    // din[j] = sum_i W1[i][j]*dpre1[i]
    for (int j = 0; j < D; j++) din[j] = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < D; j++) {
            g->dW1[i][j] += dpre1[i] * c->in[j];
            din[j] += net->W1[i][j] * dpre1[i];
        }
    }
}

static void adamw_step(Net *net, const NetGrad *g) {
    net->step += 1;
    float t = (float)net->step;

    // bias correction
    float b1t = 1.0f - powf(BETA1, t);
    float b2t = 1.0f - powf(BETA2, t);

    // W1
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < D; j++) {
            float grad = g->dW1[i][j] + WD * net->W1[i][j];
            net->mW1[i][j] = BETA1 * net->mW1[i][j] + (1.0f - BETA1) * grad;
            net->vW1[i][j] = BETA2 * net->vW1[i][j] + (1.0f - BETA2) * grad * grad;

            float mh = net->mW1[i][j] / b1t;
            float vh = net->vW1[i][j] / b2t;
            net->W1[i][j] -= LR * mh / (sqrtf(vh) + EPS);
        }
        float gradb = g->db1[i];
        net->mb1[i] = BETA1 * net->mb1[i] + (1.0f - BETA1) * gradb;
        net->vb1[i] = BETA2 * net->vb1[i] + (1.0f - BETA2) * gradb * gradb;

        float mhb = net->mb1[i] / b1t;
        float vhb = net->vb1[i] / b2t;
        net->b1[i] -= LR * mhb / (sqrtf(vhb) + EPS);
    }

    // W2
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < H; j++) {
            float grad = g->dW2[i][j] + WD * net->W2[i][j];
            net->mW2[i][j] = BETA1 * net->mW2[i][j] + (1.0f - BETA1) * grad;
            net->vW2[i][j] = BETA2 * net->vW2[i][j] + (1.0f - BETA2) * grad * grad;

            float mh = net->mW2[i][j] / b1t;
            float vh = net->vW2[i][j] / b2t;
            net->W2[i][j] -= LR * mh / (sqrtf(vh) + EPS);
        }
        float gradb = g->db2[i];
        net->mb2[i] = BETA1 * net->mb2[i] + (1.0f - BETA1) * gradb;
        net->vb2[i] = BETA2 * net->vb2[i] + (1.0f - BETA2) * gradb * gradb;

        float mhb = net->mb2[i] / b1t;
        float vhb = net->vb2[i] / b2t;
        net->b2[i] -= LR * mhb / (sqrtf(vhb) + EPS);
    }
}

// ------------------------- x embedding + output head -------------------------
//
// x_raw: F -> x_emb: D (linear)
// output_head: y_vec(D) -> logits(3) (linear)

typedef struct {
    float Wx[D][F];
    float bx[D];

    float Wo[3][D];
    float bo[3];

    // Adam moments
    float mWx[D][F], vWx[D][F];
    float mbx[D], vbx[D];

    float mWo[3][D], vWo[3][D];
    float mbo[3], vbo[3];

    int64_t step;
} Heads;

static void heads_init(Heads *h, float scale) {
    memset(h, 0, sizeof(*h));
    h->step = 0;
    for (int i = 0; i < D; i++) {
        h->bx[i] = 0.0f;
        for (int j = 0; j < F; j++) h->Wx[i][j] = frand_normal() * scale;
    }
    for (int i = 0; i < 3; i++) {
        h->bo[i] = 0.0f;
        for (int j = 0; j < D; j++) h->Wo[i][j] = frand_normal() * scale;
    }
}

typedef struct {
    float x_raw[F];
    float x_emb[D];
    float y_vec[D];
    float logits[3];
} ForwardCache;

static void embed_x(const Heads *h, const float x_raw[F], float x_emb[D]) {
    for (int i = 0; i < D; i++) {
        float s = h->bx[i];
        for (int j = 0; j < F; j++) s += h->Wx[i][j] * x_raw[j];
        x_emb[i] = s;
    }
}

static void output_head(const Heads *h, const float y_vec[D], float logits[3]) {
    for (int i = 0; i < 3; i++) {
        float s = h->bo[i];
        for (int j = 0; j < D; j++) s += h->Wo[i][j] * y_vec[j];
        logits[i] = s;
    }
}

static float softmax_xent(const float logits[3], int y_true, float dlogits[3]) {
    float m = fmaxf(logits[0], fmaxf(logits[1], logits[2]));
    float ex0 = expf(logits[0] - m);
    float ex1 = expf(logits[1] - m);
    float ex2 = expf(logits[2] - m);
    float Z = ex0 + ex1 + ex2;
    float p0 = ex0 / Z, p1 = ex1 / Z, p2 = ex2 / Z;

    float loss = -logf((y_true == 0 ? p0 : (y_true == 1 ? p1 : p2)) + 1e-12f);

    dlogits[0] = p0;
    dlogits[1] = p1;
    dlogits[2] = p2;
    dlogits[y_true] -= 1.0f;
    return loss;
}

typedef struct {
    float dWx[D][F];
    float dbx[D];
    float dWo[3][D];
    float dbo[3];
} HeadsGrad;

static void headsgrad_zero(HeadsGrad *g) { memset(g, 0, sizeof(*g)); }

static void heads_backward_embed(const Heads *h, const float x_raw[F], const float dx_emb[D],
                                 HeadsGrad *g) {
    // x_emb = Wx*x_raw + bx
    for (int i = 0; i < D; i++) {
        g->dbx[i] += dx_emb[i];
        for (int j = 0; j < F; j++) g->dWx[i][j] += dx_emb[i] * x_raw[j];
    }
    (void)h;
}

static void heads_backward_out(const Heads *h, const float y_vec[D], const float dlogits[3],
                               HeadsGrad *g, float dy_vec[D]) {
    // logits = Wo*y + bo
    for (int j = 0; j < D; j++) dy_vec[j] = 0.0f;

    for (int i = 0; i < 3; i++) {
        g->dbo[i] += dlogits[i];
        for (int j = 0; j < D; j++) {
            g->dWo[i][j] += dlogits[i] * y_vec[j];
            dy_vec[j] += h->Wo[i][j] * dlogits[i];
        }
    }
}

static void heads_adamw_step(Heads *h, const HeadsGrad *g) {
    h->step += 1;
    float t = (float)h->step;

    float b1t = 1.0f - powf(BETA1, t);
    float b2t = 1.0f - powf(BETA2, t);

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < F; j++) {
            float grad = g->dWx[i][j] + WD * h->Wx[i][j];
            h->mWx[i][j] = BETA1 * h->mWx[i][j] + (1 - BETA1) * grad;
            h->vWx[i][j] = BETA2 * h->vWx[i][j] + (1 - BETA2) * grad * grad;

            float mh = h->mWx[i][j] / b1t;
            float vh = h->vWx[i][j] / b2t;
            h->Wx[i][j] -= LR * mh / (sqrtf(vh) + EPS);
        }
        float gradb = g->dbx[i];
        h->mbx[i] = BETA1 * h->mbx[i] + (1 - BETA1) * gradb;
        h->vbx[i] = BETA2 * h->vbx[i] + (1 - BETA2) * gradb * gradb;

        float mhb = h->mbx[i] / b1t;
        float vhb = h->vbx[i] / b2t;
        h->bx[i] -= LR * mhb / (sqrtf(vhb) + EPS);
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < D; j++) {
            float grad = g->dWo[i][j] + WD * h->Wo[i][j];
            h->mWo[i][j] = BETA1 * h->mWo[i][j] + (1 - BETA1) * grad;
            h->vWo[i][j] = BETA2 * h->vWo[i][j] + (1 - BETA2) * grad * grad;

            float mh = h->mWo[i][j] / b1t;
            float vh = h->vWo[i][j] / b2t;
            h->Wo[i][j] -= LR * mh / (sqrtf(vh) + EPS);
        }
        float gradb = g->dbo[i];
        h->mbo[i] = BETA1 * h->mbo[i] + (1 - BETA1) * gradb;
        h->vbo[i] = BETA2 * h->vbo[i] + (1 - BETA2) * gradb * gradb;

        float mhb = h->mbo[i] / b1t;
        float vhb = h->vbo[i] / b2t;
        h->bo[i] -= LR * mhb / (sqrtf(vhb) + EPS);
    }
}

// ------------------------- mock candle dataset -------------------------

typedef struct {
    float x[F];
    int y;
} Sample;

static void make_features_from_series(const float *p, int t, float out[F]) {
    // p[t] is current price, we look back a few points
    // Features (simple):
    // 0..7  : returns over last 8 steps
    // 8..11 : moving average diffs (short vs long)
    // 12    : volatility proxy (std of last 8 returns)
    // 13    : momentum (p[t]-p[t-8])
    // 14    : normalized price (tanh)
    // 15    : bias term (=1)
    float rets[8];
    for (int i = 0; i < 8; i++) {
        float p0 = p[t - i - 1];
        float p1 = p[t - i];
        rets[i] = (p1 - p0) / fmaxf(p0, 1e-6f);
        out[i] = rets[i];
    }

    // MA diffs
    float ma2 = 0, ma4 = 0, ma8 = 0, ma16 = 0;
    for (int i = 0; i < 2; i++)  ma2  += p[t - i];
    for (int i = 0; i < 4; i++)  ma4  += p[t - i];
    for (int i = 0; i < 8; i++)  ma8  += p[t - i];
    for (int i = 0; i < 16; i++) ma16 += p[t - i];
    ma2  /= 2.0f; ma4 /= 4.0f; ma8 /= 8.0f; ma16 /= 16.0f;

    out[8]  = (ma2 - ma8)  / fmaxf(ma8,  1e-6f);
    out[9]  = (ma4 - ma16) / fmaxf(ma16, 1e-6f);
    out[10] = (ma2 - ma16) / fmaxf(ma16, 1e-6f);
    out[11] = (ma8 - ma16) / fmaxf(ma16, 1e-6f);

    // volatility
    float mean = 0;
    for (int i = 0; i < 8; i++) mean += rets[i];
    mean /= 8.0f;
    float var = 0;
    for (int i = 0; i < 8; i++) {
        float d = rets[i] - mean;
        var += d * d;
    }
    var /= 8.0f;
    out[12] = sqrtf(var);

    // momentum
    out[13] = (p[t] - p[t - 8]) / fmaxf(p[t - 8], 1e-6f);

    // normalized price
    out[14] = tanhf(0.01f * (p[t] - 100.0f));

    out[15] = 1.0f;
}

static void gen_dataset(Sample *dst, int n_samples, int seed_offset) {
    // Random-walk series, then slice samples from it
    // Need at least 16 lookback + horizon for labeling.
    g_rng ^= (uint64_t)(0x9e3779b97f4a7c15ULL + (uint64_t)seed_offset);

    const int series_len = n_samples + 64;
    float *p = (float*)malloc(sizeof(float) * series_len);
    if (!p) exit(1);

    float price = 100.0f;
    for (int t = 0; t < series_len; t++) {
        float step = 0.001f * frand_normal(); // small driftless noise
        price = fmaxf(1.0f, price * (1.0f + step));
        p[t] = price;
    }

    const int lookback = 16;
    const int horizon = 8;
    const float thr = 0.006f; // 0.6%

    for (int i = 0; i < n_samples; i++) {
        int t = i + lookback; // ensure history exists
        make_features_from_series(p, t, dst[i].x);

        float future = p[t + horizon];
        float cur = p[t];
        float fut_ret = (future - cur) / fmaxf(cur, 1e-6f);

        int y = CLS_WAIT;
        if (fut_ret > thr) y = CLS_LONG;
        else if (fut_ret < -thr) y = CLS_SHORT;
        dst[i].y = y;
    }

    free(p);
}

// ------------------------- TRM forward (no grad) -------------------------

static void vec_zero(float v[D]) { for (int i = 0; i < D; i++) v[i] = 0.0f; }

static void vec_add3(float out[D], const float a[D], const float b[D], const float c[D]) {
    for (int i = 0; i < D; i++) out[i] = a[i] + b[i] + c[i];
}

static void vec_add2(float out[D], const float a[D], const float b[D]) {
    for (int i = 0; i < D; i++) out[i] = a[i] + b[i];
}

// One latent recursion cycle (no grad): update z n times using x+y+z, then update y once using y+z
static void latent_recursion_nograd(const Net *net, const float x_emb[D], float y[D], float z[D]) {
    float inp[D];
    NetCache c;

    for (int i = 0; i < N_LAT; i++) {
        vec_add3(inp, x_emb, y, z);
        net_forward(net, inp, &c);
        memcpy(z, c.out, sizeof(float) * D);
    }
    vec_add2(inp, y, z);
    net_forward(net, inp, &c);
    memcpy(y, c.out, sizeof(float) * D);
}

// Deep recursion: do T-1 cycles without grad, then one "with grad" (handled separately in train)
static void deep_recursion_nograd(const Net *net, const float x_emb[D], float y[D], float z[D]) {
    for (int j = 0; j < T_REC; j++) latent_recursion_nograd(net, x_emb, y, z);
}

// ------------------------- TRM train: backprop through last recursion only -------------------------
//
// We mimic paper's: do (T-1) cycles no-grad, then last cycle with grad.
// And within that last cycle, we backprop through ALL (N_LAT z-updates + 1 y-update).

typedef struct {
    NetCache caches_z[N_LAT];
    NetCache cache_y;
    float z_in[N_LAT][D];  // store z input vectors per step (not strictly necessary since in cache)
    float y_in[D];         // input to y-update
    float x_emb[D];
    float y_before[D];
    float z_before[D];
    float y_after[D];
    float z_after[D];
} RecGradCache;

static void latent_recursion_withgrad_forward(const Net *net, const float x_emb[D],
                                             float y[D], float z[D],
                                             RecGradCache *rc) {
    memcpy(rc->x_emb, x_emb, sizeof(float)*D);
    memcpy(rc->y_before, y, sizeof(float)*D);
    memcpy(rc->z_before, z, sizeof(float)*D);

    float inp[D];

    for (int i = 0; i < N_LAT; i++) {
        vec_add3(inp, x_emb, y, z);
        net_forward(net, inp, &rc->caches_z[i]);
        memcpy(z, rc->caches_z[i].out, sizeof(float)*D);
    }

    vec_add2(inp, y, z);
    memcpy(rc->y_in, inp, sizeof(float)*D);
    net_forward(net, inp, &rc->cache_y);
    memcpy(y, rc->cache_y.out, sizeof(float)*D);

    memcpy(rc->y_after, y, sizeof(float)*D);
    memcpy(rc->z_after, z, sizeof(float)*D);
}

static void latent_recursion_withgrad_backward(const Net *net, const RecGradCache *rc,
                                              const float dy_after[D], // grad wrt final y (from loss)
                                              NetGrad *g_net,
                                              float dx_emb[D]) {
    // Backprop through:
    // y_after = net(y_in = y_before + z_after)
    // z_after = z_NLAT after N_LAT steps, each z_{k+1} = net(x_emb + y_before + z_k)
    //
    // y_before is treated as constant w.r.t this recursion (in paper, y carried; in principle gradients flow,
    // but this toy keeps it simple: gradients only through nets + x_emb).
    // We'll let gradients flow into x_emb and z.

    float dy_in[D];
    float din[D];

    // 1) y-update net backward
    net_backward(net, &rc->cache_y, dy_after, g_net, din); // din = d(y_in)
    memcpy(dy_in, din, sizeof(float)*D);

    // y_in = y_before + z_after
    // so grad splits: d z_after += dy_in ; (and d y_before += dy_in, ignored)
    float dz[D];
    memcpy(dz, dy_in, sizeof(float)*D);

    // 2) backprop through z updates in reverse
    float dx_acc[D]; for (int i=0;i<D;i++) dx_acc[i]=0.0f;

    for (int step = N_LAT - 1; step >= 0; step--) {
        // z_{step+1} = net( x_emb + y_before + z_step )
        // rc->caches_z[step] corresponds to that net forward.
        float din2[D];
        net_backward(net, &rc->caches_z[step], dz, g_net, din2); // din2 = d(input_to_net)

        // input_to_net = x_emb + y_before + z_step
        // so: dx_emb += din2; dz = din2 (for z_step); (and dy_before += din2, ignored)
        for (int i = 0; i < D; i++) {
            dx_acc[i] += din2[i];
            dz[i] = din2[i];
        }
    }

    memcpy(dx_emb, dx_acc, sizeof(float)*D);
}

// ------------------------- training / eval -------------------------

static int argmax3(const float a[3]) {
    int m = 0;
    if (a[1] > a[m]) m = 1;
    if (a[2] > a[m]) m = 2;
    return m;
}

static float eval_accuracy(const Net *net, const Heads *heads, const Sample *data, int n) {
    int correct = 0;
    float y[D], z[D], x_emb[D];
    for (int i = 0; i < n; i++) {
        embed_x(heads, data[i].x, x_emb);
        vec_zero(y);
        vec_zero(z);

        // test-time: run full N_SUP steps (like paper does)
        for (int step = 0; step < N_SUP; step++) {
            // deep recursion T times (all no-grad)
            for (int j = 0; j < T_REC; j++) latent_recursion_nograd(net, x_emb, y, z);
        }

        float logits[3];
        output_head(heads, y, logits);
        int pred = argmax3(logits);
        if (pred == data[i].y) correct++;
    }
    return (float)correct / (float)n;
}

int main(void) {
    printf("Toy TRM (C) - candle mock classification: SHORT/WAIT/LONG\n");

    Sample *train = (Sample*)malloc(sizeof(Sample) * TRAIN_SAMPLES);
    Sample *test  = (Sample*)malloc(sizeof(Sample) * TEST_SAMPLES);
    if (!train || !test) { fprintf(stderr, "alloc failed\n"); return 1; }

    gen_dataset(train, TRAIN_SAMPLES, 1);
    gen_dataset(test,  TEST_SAMPLES,  2);

    Net net;
    Heads heads;
    net_init(&net, 0.02f);
    heads_init(&heads, 0.02f);

    // training
    int steps_per_epoch = TRAIN_SAMPLES / BATCH;

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        // simple shuffle
        for (int i = 0; i < TRAIN_SAMPLES; i++) {
            int j = (int)(frand_uniform() * TRAIN_SAMPLES);
            Sample tmp = train[i];
            train[i] = train[j];
            train[j] = tmp;
        }

        for (int it = 0; it < steps_per_epoch; it++) {
            NetGrad gnet; netgrad_zero(&gnet);
            HeadsGrad gh; headsgrad_zero(&gh);

            float batch_loss = 0.0f;

            for (int b = 0; b < BATCH; b++) {
                const Sample *s = &train[it * BATCH + b];

                float x_emb[D];
                embed_x(&heads, s->x, x_emb);

                float y[D], z[D];
                vec_zero(y);
                vec_zero(z);

                // deep supervision steps
                // Each step: do (T-1) cycles no-grad, then 1 cycle with grad, compute loss on y, update y,z (detached)
                for (int sup = 0; sup < N_SUP; sup++) {
                    // (T-1) cycles no-grad
                    for (int j = 0; j < T_REC - 1; j++) latent_recursion_nograd(&net, x_emb, y, z);

                    // last cycle with grad
                    RecGradCache rc;
                    latent_recursion_withgrad_forward(&net, x_emb, y, z, &rc);

                    // output head + xent
                    float logits[3], dlogits[3];
                    output_head(&heads, y, logits);
                    float loss = softmax_xent(logits, s->y, dlogits);
                    batch_loss += loss;

                    // backprop through output head into y
                    float dy_vec[D];
                    heads_backward_out(&heads, y, dlogits, &gh, dy_vec);

                    // backprop through the last recursion into net params and into x_emb (then into Wx)
                    float dx_emb[D];
                    latent_recursion_withgrad_backward(&net, &rc, dy_vec, &gnet, dx_emb);

                    // backprop dx_emb into embed Wx
                    heads_backward_embed(&heads, s->x, dx_emb, &gh);

                    // detach y,z (already values are in y,z; we just keep them as new init)
                    // Optional early stop (paper uses halting head); here we skip for simplicity.
                }
            }

            // average grads over batch
            float invB = 1.0f / (float)BATCH;
            for (int i = 0; i < H; i++) {
                gnet.db1[i] *= invB;
                for (int j = 0; j < D; j++) gnet.dW1[i][j] *= invB;
            }
            for (int i = 0; i < D; i++) {
                gnet.db2[i] *= invB;
                for (int j = 0; j < H; j++) gnet.dW2[i][j] *= invB;
            }
            for (int i = 0; i < D; i++) {
                gh.dbx[i] *= invB;
                for (int j = 0; j < F; j++) gh.dWx[i][j] *= invB;
            }
            for (int i = 0; i < 3; i++) {
                gh.dbo[i] *= invB;
                for (int j = 0; j < D; j++) gh.dWo[i][j] *= invB;
            }

            adamw_step(&net, &gnet);
            heads_adamw_step(&heads, &gh);

            batch_loss *= invB;
            epoch_loss += batch_loss;
        }

        epoch_loss /= (float)steps_per_epoch;

        float acc_train = eval_accuracy(&net, &heads, train, 500); // quick probe
        float acc_test  = eval_accuracy(&net, &heads, test, TEST_SAMPLES);

        printf("Epoch %2d | loss=%.4f | train_acc~%.3f | test_acc=%.3f\n",
               epoch, epoch_loss, acc_train, acc_test);
    }

    free(train);
    free(test);
    return 0;
}
