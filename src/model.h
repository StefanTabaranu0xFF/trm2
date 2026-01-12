#ifndef TRM_MODEL_H
#define TRM_MODEL_H

#include "attention.h"

typedef struct {
    int vocab_size;
    int max_seq;
    int d_model;
    int n_layers;
    int n_recursions;
} TRMConfig;

typedef struct {
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *bq;
    float *bk;
    float *bv;
    float *bo;
    float *w1;
    float *b1;
    float *w2;
    float *b2;
    float *ln1_g;
    float *ln1_b;
    float *ln2_g;
    float *ln2_b;
} TRMLayer;

typedef struct {
    float *tok_embed;
    float *pos_embed;
    float *out_proj;
    float *out_bias;
    TRMLayer *layers;
    TRMConfig config;
} TRMModel;

typedef struct {
    float *ln1_mean;
    float *ln1_var;
    float *ln2_mean;
    float *ln2_var;
    AttentionCache attn;
    float *mlp_hidden;
    float *mlp_act;
} LayerCache;

typedef struct {
    float *input;
    float *ln1_out;
    float *attn_out;
    float *ln2_in;
    float *ln2_out;
    float *mlp_out;
    LayerCache *layer_caches;
} BlockCache;

typedef struct {
    BlockCache *blocks;
} RecursionCache;

typedef struct {
    RecursionCache *recursions;
    int total_steps;
} TRMCache;

TRMModel *trm_model_create(TRMConfig config);
void trm_model_free(TRMModel *model);
void trm_model_zero(TRMModel *model);

void trm_cache_alloc(TRMCache *cache, const TRMConfig *config, int total_steps, int seq_len);
void trm_cache_free(TRMCache *cache, const TRMConfig *config, int total_steps);

void trm_forward(TRMModel *model, const int *tokens, int seq_len,
                 float *logits, TRMCache *cache);

void trm_backward(TRMModel *model, const int *tokens, int seq_len,
                  const int *targets, float *loss_out, float lr, TRMCache *cache);

void trm_save(const TRMModel *model, const char *path);
TRMModel *trm_load(const char *path);

#endif
