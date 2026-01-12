#ifndef TRM_ATTENTION_H
#define TRM_ATTENTION_H

#include "gemm.h"

typedef struct {
    float *q;
    float *k;
    float *v;
    float *scores;
    float *attn;
    float *context;
} AttentionCache;

void attention_forward(const float *x, int seq_len, int d_model,
                       const float *wq, const float *wk, const float *wv, const float *wo,
                       const float *bq, const float *bk, const float *bv, const float *bo,
                       float *out, AttentionCache *cache);

void attention_backward(const float *x, int seq_len, int d_model,
                        const float *wq, const float *wk, const float *wv, const float *wo,
                        const float *bq, const float *bk, const float *bv, const float *bo,
                        const float *dout,
                        float *dx,
                        float *dwq, float *dwk, float *dwv, float *dwo,
                        float *dbq, float *dbk, float *dbv, float *dbo,
                        AttentionCache *cache);

void attention_cache_alloc(AttentionCache *cache, int seq_len, int d_model);
void attention_cache_free(AttentionCache *cache);

#endif
