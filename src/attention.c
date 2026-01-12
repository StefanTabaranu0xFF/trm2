#include "attention.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void attention_cache_alloc(AttentionCache *cache, int seq_len, int d_model) {
    int sd = seq_len * d_model;
    cache->q = (float *)calloc((size_t)sd, sizeof(float));
    cache->k = (float *)calloc((size_t)sd, sizeof(float));
    cache->v = (float *)calloc((size_t)sd, sizeof(float));
    cache->scores = (float *)calloc((size_t)seq_len * seq_len, sizeof(float));
    cache->attn = (float *)calloc((size_t)seq_len * seq_len, sizeof(float));
    cache->context = (float *)calloc((size_t)sd, sizeof(float));
}

void attention_cache_free(AttentionCache *cache) {
    free(cache->q);
    free(cache->k);
    free(cache->v);
    free(cache->scores);
    free(cache->attn);
    free(cache->context);
}

void attention_forward(const float *x, int seq_len, int d_model,
                       const float *wq, const float *wk, const float *wv, const float *wo,
                       const float *bq, const float *bk, const float *bv, const float *bo,
                       float *out, AttentionCache *cache) {
    int sd = seq_len * d_model;
    matmul_nn(x, wq, cache->q, seq_len, d_model, d_model);
    matmul_nn(x, wk, cache->k, seq_len, d_model, d_model);
    matmul_nn(x, wv, cache->v, seq_len, d_model, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            int idx = i * d_model + j;
            cache->q[idx] += bq[j];
            cache->k[idx] += bk[j];
            cache->v[idx] += bv[j];
        }
    }

    float scale = 1.0f / sqrtf((float)d_model);
    matmul_nt(cache->q, cache->k, cache->scores, seq_len, seq_len, d_model);
    scale_inplace(cache->scores, scale, seq_len * seq_len);
    softmax_rows(cache->attn, cache->scores, seq_len, seq_len);

    matmul_nn(cache->attn, cache->v, cache->context, seq_len, d_model, seq_len);
    matmul_nn(cache->context, wo, out, seq_len, d_model, d_model);
    for (int i = 0; i < sd; ++i) {
        out[i] += bo[i % d_model];
    }
}

void attention_backward(const float *x, int seq_len, int d_model,
                        const float *wq, const float *wk, const float *wv, const float *wo,
                        const float *bq, const float *bk, const float *bv, const float *bo,
                        const float *dout,
                        float *dx,
                        float *dwq, float *dwk, float *dwv, float *dwo,
                        float *dbq, float *dbk, float *dbv, float *dbo,
                        AttentionCache *cache) {
    (void)bq;
    (void)bk;
    (void)bv;
    (void)bo;
    int sd = seq_len * d_model;
    float *dcontext = (float *)calloc((size_t)sd, sizeof(float));
    float *dattn = (float *)calloc((size_t)seq_len * seq_len, sizeof(float));
    float *dscores = (float *)calloc((size_t)seq_len * seq_len, sizeof(float));
    float *dq = (float *)calloc((size_t)sd, sizeof(float));
    float *dk = (float *)calloc((size_t)sd, sizeof(float));
    float *dv = (float *)calloc((size_t)sd, sizeof(float));

    matmul_tn(cache->context, dout, dwo, d_model, d_model, seq_len);
    for (int i = 0; i < sd; ++i) {
        dbo[i % d_model] += dout[i];
    }
    matmul_nn(dout, wo, dcontext, seq_len, d_model, d_model);

    matmul_nt(dcontext, cache->v, dattn, seq_len, seq_len, d_model);
    matmul_tn(cache->attn, dcontext, dv, seq_len, d_model, seq_len);

    for (int i = 0; i < seq_len; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            int idx = i * seq_len + j;
            row_sum += dattn[idx] * cache->attn[idx];
        }
        for (int j = 0; j < seq_len; ++j) {
            int idx = i * seq_len + j;
            dscores[idx] = cache->attn[idx] * (dattn[idx] - row_sum);
        }
    }

    float scale = 1.0f / sqrtf((float)d_model);
    scale_inplace(dscores, scale, seq_len * seq_len);
    matmul_nn(dscores, cache->k, dq, seq_len, d_model, seq_len);
    matmul_tn(dscores, cache->q, dk, seq_len, d_model, seq_len);

    matmul_tn(x, dq, dwq, d_model, d_model, seq_len);
    matmul_tn(x, dk, dwk, d_model, d_model, seq_len);
    matmul_tn(x, dv, dwv, d_model, d_model, seq_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            int idx = i * d_model + j;
            dbq[j] += dq[idx];
            dbk[j] += dk[idx];
            dbv[j] += dv[idx];
        }
    }

    float *dxq = (float *)calloc((size_t)sd, sizeof(float));
    float *dxk = (float *)calloc((size_t)sd, sizeof(float));
    float *dxv = (float *)calloc((size_t)sd, sizeof(float));
    matmul_nt(dq, wq, dxq, seq_len, d_model, d_model);
    matmul_nt(dk, wk, dxk, seq_len, d_model, d_model);
    matmul_nt(dv, wv, dxv, seq_len, d_model, d_model);
    for (int i = 0; i < sd; ++i) {
        dx[i] = dxq[i] + dxk[i] + dxv[i];
    }

    free(dcontext);
    free(dattn);
    free(dscores);
    free(dq);
    free(dk);
    free(dv);
    free(dxq);
    free(dxk);
    free(dxv);
}
