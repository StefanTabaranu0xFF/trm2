#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float frand(void) {
    return ((float)rand() / (float)RAND_MAX) - 0.5f;
}

static void init_array(float *buf, int n, float scale) {
    for (int i = 0; i < n; ++i) {
        buf[i] = frand() * scale;
    }
}

static void layernorm_forward(const float *x, const float *gamma, const float *beta,
                              float *out, float *mean, float *var,
                              int seq_len, int d_model) {
    const float eps = 1e-5f;
    for (int i = 0; i < seq_len; ++i) {
        const float *row = x + i * d_model;
        float m = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            m += row[j];
        }
        m /= (float)d_model;
        float v = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            float diff = row[j] - m;
            v += diff * diff;
        }
        v /= (float)d_model;
        mean[i] = m;
        var[i] = v;
        float inv = 1.0f / sqrtf(v + eps);
        float *out_row = out + i * d_model;
        for (int j = 0; j < d_model; ++j) {
            float norm = (row[j] - m) * inv;
            out_row[j] = norm * gamma[j] + beta[j];
        }
    }
}

static void layernorm_backward(const float *dout, const float *x, const float *gamma,
                               const float *mean, const float *var,
                               float *dx, float *dgamma, float *dbeta,
                               int seq_len, int d_model) {
    const float eps = 1e-5f;
    for (int i = 0; i < seq_len; ++i) {
        const float *row = x + i * d_model;
        const float *dout_row = dout + i * d_model;
        float m = mean[i];
        float v = var[i];
        float inv = 1.0f / sqrtf(v + eps);
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            float norm = (row[j] - m) * inv;
            sum1 += dout_row[j] * gamma[j];
            sum2 += dout_row[j] * gamma[j] * norm;
            dgamma[j] += dout_row[j] * norm;
            dbeta[j] += dout_row[j];
        }
        float *dx_row = dx + i * d_model;
        for (int j = 0; j < d_model; ++j) {
            float norm = (row[j] - m) * inv;
            float d = dout_row[j] * gamma[j];
            dx_row[j] = (d - sum1 / d_model - norm * sum2 / d_model) * inv;
        }
    }
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978846f * (x + 0.044715f * x * x * x)));
}

static float gelu_grad(float x) {
    float t = 0.7978846f * (x + 0.044715f * x * x * x);
    float th = tanhf(t);
    float sech2 = 1.0f - th * th;
    return 0.5f * (1.0f + th) + 0.5f * x * sech2 * 0.7978846f * (1.0f + 3.0f * 0.044715f * x * x);
}

static void linear_forward(const float *x, const float *w, const float *b,
                           float *out, int m, int n, int k) {
    matmul_nn(x, w, out, m, n, k);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            out[i * n + j] += b[j];
        }
    }
}

static void linear_backward(const float *x, const float *w, const float *dout,
                            float *dx, float *dw, float *db,
                            int m, int n, int k) {
    matmul_tn(x, dout, dw, k, n, m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            db[j] += dout[i * n + j];
        }
    }
    matmul_nt(dout, w, dx, m, k, n);
}

static void sgd_update(float *w, const float *dw, int n, float lr) {
    for (int i = 0; i < n; ++i) {
        w[i] -= lr * dw[i];
    }
}

static void block_forward(const float *x, const TRMLayer *layer, int seq_len, int d_model,
                          float *out, BlockCache *cache) {
    int sd = seq_len * d_model;
    layernorm_forward(x, layer->ln1_g, layer->ln1_b, cache->ln1_out,
                      cache->layer_caches[0].ln1_mean, cache->layer_caches[0].ln1_var,
                      seq_len, d_model);
    attention_forward(cache->ln1_out, seq_len, d_model,
                      layer->wq, layer->wk, layer->wv, layer->wo,
                      layer->bq, layer->bk, layer->bv, layer->bo,
                      cache->attn_out, &cache->layer_caches[0].attn);
    memcpy(out, x, (size_t)sd * sizeof(float));
    add_inplace(out, cache->attn_out, sd);
    memcpy(cache->ln2_in, out, (size_t)sd * sizeof(float));
    layernorm_forward(cache->ln2_in, layer->ln2_g, layer->ln2_b, cache->ln2_out,
                      cache->layer_caches[0].ln2_mean, cache->layer_caches[0].ln2_var,
                      seq_len, d_model);
    linear_forward(cache->ln2_out, layer->w1, layer->b1, cache->layer_caches[0].mlp_hidden,
                   seq_len, d_model * 4, d_model);
    for (int i = 0; i < seq_len * d_model * 4; ++i) {
        cache->layer_caches[0].mlp_act[i] = gelu(cache->layer_caches[0].mlp_hidden[i]);
    }
    linear_forward(cache->layer_caches[0].mlp_act, layer->w2, layer->b2, cache->mlp_out,
                   seq_len, d_model, d_model * 4);
    add_inplace(out, cache->mlp_out, sd);
}

static void block_backward(const TRMLayer *layer, int seq_len, int d_model,
                           const float *dout, float *dx, BlockCache *cache, float lr) {
    int sd = seq_len * d_model;
    int hidden = d_model * 4;
    float *dres1 = (float *)calloc((size_t)sd, sizeof(float));
    float *dmlp = (float *)calloc((size_t)sd, sizeof(float));
    float *dln2 = (float *)calloc((size_t)sd, sizeof(float));
    float *dln1 = (float *)calloc((size_t)sd, sizeof(float));
    float *dhidden = (float *)calloc((size_t)seq_len * hidden, sizeof(float));
    float *dact = (float *)calloc((size_t)seq_len * hidden, sizeof(float));

    float *dw1 = (float *)calloc((size_t)d_model * hidden, sizeof(float));
    float *db1 = (float *)calloc((size_t)hidden, sizeof(float));
    float *dw2 = (float *)calloc((size_t)hidden * d_model, sizeof(float));
    float *db2 = (float *)calloc((size_t)d_model, sizeof(float));
    float *dln1_g = (float *)calloc((size_t)d_model, sizeof(float));
    float *dln1_b = (float *)calloc((size_t)d_model, sizeof(float));
    float *dln2_g = (float *)calloc((size_t)d_model, sizeof(float));
    float *dln2_b = (float *)calloc((size_t)d_model, sizeof(float));
    float *dwq = (float *)calloc((size_t)d_model * d_model, sizeof(float));
    float *dwk = (float *)calloc((size_t)d_model * d_model, sizeof(float));
    float *dwv = (float *)calloc((size_t)d_model * d_model, sizeof(float));
    float *dwo = (float *)calloc((size_t)d_model * d_model, sizeof(float));
    float *dbq = (float *)calloc((size_t)d_model, sizeof(float));
    float *dbk = (float *)calloc((size_t)d_model, sizeof(float));
    float *dbv = (float *)calloc((size_t)d_model, sizeof(float));
    float *dbo = (float *)calloc((size_t)d_model, sizeof(float));

    memcpy(dres1, dout, (size_t)sd * sizeof(float));
    memcpy(dmlp, dout, (size_t)sd * sizeof(float));

    linear_backward(cache->layer_caches[0].mlp_act, layer->w2, dmlp, dact, dw2, db2,
                    seq_len, d_model, hidden);
    for (int i = 0; i < seq_len * hidden; ++i) {
        dhidden[i] = dact[i] * gelu_grad(cache->layer_caches[0].mlp_hidden[i]);
    }
    linear_backward(cache->ln2_out, layer->w1, dhidden, dln2, dw1, db1,
                    seq_len, hidden, d_model);

    layernorm_backward(dln2, cache->ln2_in, layer->ln2_g,
                       cache->layer_caches[0].ln2_mean, cache->layer_caches[0].ln2_var,
                       dres1, dln2_g, dln2_b, seq_len, d_model);

    float *dx_attn = (float *)calloc((size_t)sd, sizeof(float));
    attention_backward(cache->ln1_out, seq_len, d_model,
                       layer->wq, layer->wk, layer->wv, layer->wo,
                       layer->bq, layer->bk, layer->bv, layer->bo,
                       dres1, dx_attn,
                       dwq, dwk, dwv, dwo,
                       dbq, dbk, dbv, dbo,
                       &cache->layer_caches[0].attn);

    layernorm_backward(dx_attn, cache->input, layer->ln1_g,
                       cache->layer_caches[0].ln1_mean, cache->layer_caches[0].ln1_var,
                       dln1, dln1_g, dln1_b, seq_len, d_model);

    memcpy(dx, dln1, (size_t)sd * sizeof(float));
    add_inplace(dx, dres1, sd);

    sgd_update(layer->w1, dw1, d_model * hidden, lr);
    sgd_update(layer->b1, db1, hidden, lr);
    sgd_update(layer->w2, dw2, hidden * d_model, lr);
    sgd_update(layer->b2, db2, d_model, lr);
    sgd_update(layer->ln1_g, dln1_g, d_model, lr);
    sgd_update(layer->ln1_b, dln1_b, d_model, lr);
    sgd_update(layer->ln2_g, dln2_g, d_model, lr);
    sgd_update(layer->ln2_b, dln2_b, d_model, lr);
    sgd_update(layer->wq, dwq, d_model * d_model, lr);
    sgd_update(layer->wk, dwk, d_model * d_model, lr);
    sgd_update(layer->wv, dwv, d_model * d_model, lr);
    sgd_update(layer->wo, dwo, d_model * d_model, lr);
    sgd_update(layer->bq, dbq, d_model, lr);
    sgd_update(layer->bk, dbk, d_model, lr);
    sgd_update(layer->bv, dbv, d_model, lr);
    sgd_update(layer->bo, dbo, d_model, lr);

    free(dres1);
    free(dmlp);
    free(dln2);
    free(dln1);
    free(dhidden);
    free(dact);
    free(dw1);
    free(db1);
    free(dw2);
    free(db2);
    free(dln1_g);
    free(dln1_b);
    free(dln2_g);
    free(dln2_b);
    free(dwq);
    free(dwk);
    free(dwv);
    free(dwo);
    free(dbq);
    free(dbk);
    free(dbv);
    free(dbo);
    free(dx_attn);
}

TRMModel *trm_model_create(TRMConfig config) {
    TRMModel *model = (TRMModel *)calloc(1, sizeof(TRMModel));
    model->config = config;
    model->tok_embed = (float *)calloc((size_t)config.vocab_size * config.d_model, sizeof(float));
    model->pos_embed = (float *)calloc((size_t)config.max_seq * config.d_model, sizeof(float));
    model->out_proj = (float *)calloc((size_t)config.d_model * config.vocab_size, sizeof(float));
    model->out_bias = (float *)calloc((size_t)config.vocab_size, sizeof(float));
    model->layers = (TRMLayer *)calloc((size_t)config.n_layers, sizeof(TRMLayer));

    float scale = 0.02f;
    init_array(model->tok_embed, config.vocab_size * config.d_model, scale);
    init_array(model->pos_embed, config.max_seq * config.d_model, scale);
    init_array(model->out_proj, config.d_model * config.vocab_size, scale);

    for (int i = 0; i < config.n_layers; ++i) {
        TRMLayer *layer = &model->layers[i];
        layer->wq = (float *)calloc((size_t)config.d_model * config.d_model, sizeof(float));
        layer->wk = (float *)calloc((size_t)config.d_model * config.d_model, sizeof(float));
        layer->wv = (float *)calloc((size_t)config.d_model * config.d_model, sizeof(float));
        layer->wo = (float *)calloc((size_t)config.d_model * config.d_model, sizeof(float));
        layer->bq = (float *)calloc((size_t)config.d_model, sizeof(float));
        layer->bk = (float *)calloc((size_t)config.d_model, sizeof(float));
        layer->bv = (float *)calloc((size_t)config.d_model, sizeof(float));
        layer->bo = (float *)calloc((size_t)config.d_model, sizeof(float));
        layer->w1 = (float *)calloc((size_t)config.d_model * config.d_model * 4, sizeof(float));
        layer->b1 = (float *)calloc((size_t)config.d_model * 4, sizeof(float));
        layer->w2 = (float *)calloc((size_t)config.d_model * 4 * config.d_model, sizeof(float));
        layer->b2 = (float *)calloc((size_t)config.d_model, sizeof(float));
        layer->ln1_g = (float *)calloc((size_t)config.d_model, sizeof(float));
        layer->ln1_b = (float *)calloc((size_t)config.d_model, sizeof(float));
        layer->ln2_g = (float *)calloc((size_t)config.d_model, sizeof(float));
        layer->ln2_b = (float *)calloc((size_t)config.d_model, sizeof(float));

        init_array(layer->wq, config.d_model * config.d_model, scale);
        init_array(layer->wk, config.d_model * config.d_model, scale);
        init_array(layer->wv, config.d_model * config.d_model, scale);
        init_array(layer->wo, config.d_model * config.d_model, scale);
        init_array(layer->w1, config.d_model * config.d_model * 4, scale);
        init_array(layer->w2, config.d_model * config.d_model * 4, scale);
        for (int j = 0; j < config.d_model; ++j) {
            layer->ln1_g[j] = 1.0f;
            layer->ln2_g[j] = 1.0f;
        }
    }

    return model;
}

void trm_model_free(TRMModel *model) {
    if (!model) {
        return;
    }
    free(model->tok_embed);
    free(model->pos_embed);
    free(model->out_proj);
    free(model->out_bias);
    for (int i = 0; i < model->config.n_layers; ++i) {
        TRMLayer *layer = &model->layers[i];
        free(layer->wq);
        free(layer->wk);
        free(layer->wv);
        free(layer->wo);
        free(layer->bq);
        free(layer->bk);
        free(layer->bv);
        free(layer->bo);
        free(layer->w1);
        free(layer->b1);
        free(layer->w2);
        free(layer->b2);
        free(layer->ln1_g);
        free(layer->ln1_b);
        free(layer->ln2_g);
        free(layer->ln2_b);
    }
    free(model->layers);
    free(model);
}

void trm_model_zero(TRMModel *model) {
    zero_buf(model->tok_embed, model->config.vocab_size * model->config.d_model);
}

void trm_cache_alloc(TRMCache *cache, const TRMConfig *config, int total_steps, int seq_len) {
    cache->total_steps = total_steps;
    cache->recursions = (RecursionCache *)calloc((size_t)total_steps, sizeof(RecursionCache));
    int sd = seq_len * config->d_model;
    for (int i = 0; i < total_steps; ++i) {
        cache->recursions[i].blocks = (BlockCache *)calloc((size_t)config->n_layers, sizeof(BlockCache));
        for (int l = 0; l < config->n_layers; ++l) {
            BlockCache *block = &cache->recursions[i].blocks[l];
            block->input = (float *)calloc((size_t)sd, sizeof(float));
            block->ln1_out = (float *)calloc((size_t)sd, sizeof(float));
            block->attn_out = (float *)calloc((size_t)sd, sizeof(float));
            block->ln2_in = (float *)calloc((size_t)sd, sizeof(float));
            block->ln2_out = (float *)calloc((size_t)sd, sizeof(float));
            block->mlp_out = (float *)calloc((size_t)sd, sizeof(float));
            block->layer_caches = (LayerCache *)calloc(1, sizeof(LayerCache));
            block->layer_caches[0].ln1_mean = (float *)calloc((size_t)seq_len, sizeof(float));
            block->layer_caches[0].ln1_var = (float *)calloc((size_t)seq_len, sizeof(float));
            block->layer_caches[0].ln2_mean = (float *)calloc((size_t)seq_len, sizeof(float));
            block->layer_caches[0].ln2_var = (float *)calloc((size_t)seq_len, sizeof(float));
            attention_cache_alloc(&block->layer_caches[0].attn, seq_len, config->d_model);
            block->layer_caches[0].mlp_hidden = (float *)calloc((size_t)seq_len * config->d_model * 4, sizeof(float));
            block->layer_caches[0].mlp_act = (float *)calloc((size_t)seq_len * config->d_model * 4, sizeof(float));
        }
    }
}

void trm_cache_free(TRMCache *cache, const TRMConfig *config, int total_steps) {
    for (int i = 0; i < total_steps; ++i) {
        for (int l = 0; l < config->n_layers; ++l) {
            BlockCache *block = &cache->recursions[i].blocks[l];
            free(block->input);
            free(block->ln1_out);
            free(block->attn_out);
            free(block->ln2_in);
            free(block->ln2_out);
            free(block->mlp_out);
            free(block->layer_caches[0].ln1_mean);
            free(block->layer_caches[0].ln1_var);
            free(block->layer_caches[0].ln2_mean);
            free(block->layer_caches[0].ln2_var);
            attention_cache_free(&block->layer_caches[0].attn);
            free(block->layer_caches[0].mlp_hidden);
            free(block->layer_caches[0].mlp_act);
            free(block->layer_caches);
        }
        free(cache->recursions[i].blocks);
    }
    free(cache->recursions);
}

static void net_forward(TRMModel *model, const float *x, int seq_len, float *out, BlockCache *caches) {
    int sd = seq_len * model->config.d_model;
    memcpy(out, x, (size_t)sd * sizeof(float));
    for (int l = 0; l < model->config.n_layers; ++l) {
        BlockCache *block = &caches[l];
        memcpy(block->input, out, (size_t)sd * sizeof(float));
        block_forward(out, &model->layers[l], seq_len, model->config.d_model, out, block);
    }
}

static void net_backward(TRMModel *model, const float *dout, int seq_len, float *dx,
                         BlockCache *caches, float lr) {
    int sd = seq_len * model->config.d_model;
    float *dtemp = (float *)calloc((size_t)sd, sizeof(float));
    memcpy(dtemp, dout, (size_t)sd * sizeof(float));
    for (int l = model->config.n_layers - 1; l >= 0; --l) {
        float *dprev = (float *)calloc((size_t)sd, sizeof(float));
        block_backward(&model->layers[l], seq_len, model->config.d_model,
                       dtemp, dprev, &caches[l], lr);
        free(dtemp);
        dtemp = dprev;
    }
    memcpy(dx, dtemp, (size_t)sd * sizeof(float));
    free(dtemp);
}

static void trm_run(TRMModel *model, const int *tokens, int seq_len,
                    float *x, float *y, float *z, float *logits, TRMCache *cache) {
    int d = model->config.d_model;
    int sd = seq_len * d;
    for (int i = 0; i < seq_len; ++i) {
        int tok = tokens[i];
        for (int j = 0; j < d; ++j) {
            x[i * d + j] = model->tok_embed[tok * d + j] + model->pos_embed[i * d + j];
        }
    }
    memcpy(y, x, (size_t)sd * sizeof(float));
    zero_buf(z, sd);

    for (int step = 0; step < model->config.n_recursions; ++step) {
        float *input = (float *)calloc((size_t)sd, sizeof(float));
        memcpy(input, x, (size_t)sd * sizeof(float));
        add_inplace(input, y, sd);
        add_inplace(input, z, sd);
        net_forward(model, input, seq_len, z, cache->recursions[step].blocks);
        free(input);
    }

    float *y_input = (float *)calloc((size_t)sd, sizeof(float));
    memcpy(y_input, y, (size_t)sd * sizeof(float));
    add_inplace(y_input, z, sd);
    net_forward(model, y_input, seq_len, y, cache->recursions[model->config.n_recursions].blocks);

    if (logits) {
        matmul_nn(y, model->out_proj, logits, seq_len, model->config.vocab_size, d);
        for (int i = 0; i < seq_len; ++i) {
            for (int v = 0; v < model->config.vocab_size; ++v) {
                logits[i * model->config.vocab_size + v] += model->out_bias[v];
            }
        }
    }

    free(y_input);
}

void trm_forward(TRMModel *model, const int *tokens, int seq_len,
                 float *logits, TRMCache *cache) {
    int d = model->config.d_model;
    int sd = seq_len * d;
    float *x = (float *)calloc((size_t)sd, sizeof(float));
    float *y = (float *)calloc((size_t)sd, sizeof(float));
    float *z = (float *)calloc((size_t)sd, sizeof(float));
    trm_run(model, tokens, seq_len, x, y, z, logits, cache);

    free(x);
    free(y);
    free(z);
}

static float cross_entropy(const float *logits, int target, int vocab_size, float *dlogits) {
    float max_val = logits[0];
    for (int v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_val) {
            max_val = logits[v];
        }
    }
    float sum = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        dlogits[v] = expf(logits[v] - max_val);
        sum += dlogits[v];
    }
    float inv = 1.0f / sum;
    float loss = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        float p = dlogits[v] * inv;
        dlogits[v] = p;
    }
    loss = -logf(dlogits[target] + 1e-9f);
    dlogits[target] -= 1.0f;
    return loss;
}

void trm_backward(TRMModel *model, const int *tokens, int seq_len,
                  const int *targets, float *loss_out, float lr, TRMCache *cache) {
    int d = model->config.d_model;
    int vocab = model->config.vocab_size;
    int sd = seq_len * d;
    float *logits = (float *)calloc((size_t)seq_len * vocab, sizeof(float));
    float *x = (float *)calloc((size_t)sd, sizeof(float));
    float *y = (float *)calloc((size_t)sd, sizeof(float));
    float *z = (float *)calloc((size_t)sd, sizeof(float));

    trm_run(model, tokens, seq_len, x, y, z, logits, cache);

    float *dlogits = (float *)calloc((size_t)seq_len * vocab, sizeof(float));
    float loss = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        loss += cross_entropy(logits + i * vocab, targets[i], vocab, dlogits + i * vocab);
    }
    loss /= (float)seq_len;
    *loss_out = loss;

    float *dy = (float *)calloc((size_t)sd, sizeof(float));
    float *dout_proj = (float *)calloc((size_t)d * vocab, sizeof(float));
    float *dout_bias = (float *)calloc((size_t)vocab, sizeof(float));

    matmul_tn(y, dlogits, dout_proj, d, vocab, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int v = 0; v < vocab; ++v) {
            dout_bias[v] += dlogits[i * vocab + v];
        }
    }
    matmul_nt(dlogits, model->out_proj, dy, seq_len, d, vocab);

    float *dy_input = (float *)calloc((size_t)sd, sizeof(float));
    float *dz = (float *)calloc((size_t)sd, sizeof(float));
    net_backward(model, dy, seq_len, dy_input,
                 cache->recursions[model->config.n_recursions].blocks, lr);
    add_inplace(dz, dy_input, sd);
    add_inplace(dy, dy_input, sd);

    float *dx = (float *)calloc((size_t)sd, sizeof(float));

    for (int step = model->config.n_recursions - 1; step >= 0; --step) {
        float *dinput = (float *)calloc((size_t)sd, sizeof(float));
        net_backward(model, dz, seq_len, dinput, cache->recursions[step].blocks, lr);
        add_inplace(dx, dinput, sd);
        add_inplace(dy, dinput, sd);
        memcpy(dz, dinput, (size_t)sd * sizeof(float));
        free(dinput);
    }

    add_inplace(dx, dy, sd);

    for (int i = 0; i < seq_len; ++i) {
        int tok = tokens[i];
        for (int j = 0; j < d; ++j) {
            int idx = i * d + j;
            model->tok_embed[tok * d + j] -= lr * dx[idx];
            model->pos_embed[i * d + j] -= lr * dx[idx];
        }
    }

    sgd_update(model->out_proj, dout_proj, d * vocab, lr);
    sgd_update(model->out_bias, dout_bias, vocab, lr);

    free(logits);
    free(dlogits);
    free(dy);
    free(dout_proj);
    free(dout_bias);
    free(dy_input);
    free(dz);
    free(dx);
    free(y);
    free(x);
    free(z);
}

void trm_save(const TRMModel *model, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return;
    }
    fwrite("TRM2", 1, 4, f);
    fwrite(&model->config, sizeof(TRMConfig), 1, f);
    fwrite(model->tok_embed, sizeof(float), (size_t)model->config.vocab_size * model->config.d_model, f);
    fwrite(model->pos_embed, sizeof(float), (size_t)model->config.max_seq * model->config.d_model, f);
    fwrite(model->out_proj, sizeof(float), (size_t)model->config.d_model * model->config.vocab_size, f);
    fwrite(model->out_bias, sizeof(float), (size_t)model->config.vocab_size, f);
    for (int i = 0; i < model->config.n_layers; ++i) {
        TRMLayer *layer = &model->layers[i];
        fwrite(layer->wq, sizeof(float), (size_t)model->config.d_model * model->config.d_model, f);
        fwrite(layer->wk, sizeof(float), (size_t)model->config.d_model * model->config.d_model, f);
        fwrite(layer->wv, sizeof(float), (size_t)model->config.d_model * model->config.d_model, f);
        fwrite(layer->wo, sizeof(float), (size_t)model->config.d_model * model->config.d_model, f);
        fwrite(layer->bq, sizeof(float), (size_t)model->config.d_model, f);
        fwrite(layer->bk, sizeof(float), (size_t)model->config.d_model, f);
        fwrite(layer->bv, sizeof(float), (size_t)model->config.d_model, f);
        fwrite(layer->bo, sizeof(float), (size_t)model->config.d_model, f);
        fwrite(layer->w1, sizeof(float), (size_t)model->config.d_model * model->config.d_model * 4, f);
        fwrite(layer->b1, sizeof(float), (size_t)model->config.d_model * 4, f);
        fwrite(layer->w2, sizeof(float), (size_t)model->config.d_model * model->config.d_model * 4, f);
        fwrite(layer->b2, sizeof(float), (size_t)model->config.d_model, f);
        fwrite(layer->ln1_g, sizeof(float), (size_t)model->config.d_model, f);
        fwrite(layer->ln1_b, sizeof(float), (size_t)model->config.d_model, f);
        fwrite(layer->ln2_g, sizeof(float), (size_t)model->config.d_model, f);
        fwrite(layer->ln2_b, sizeof(float), (size_t)model->config.d_model, f);
    }
    fclose(f);
}

TRMModel *trm_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TRM2", 4) != 0) {
        fclose(f);
        return NULL;
    }
    TRMConfig config;
    if (fread(&config, sizeof(TRMConfig), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    TRMModel *model = trm_model_create(config);
    size_t tok_count = (size_t)config.vocab_size * config.d_model;
    size_t pos_count = (size_t)config.max_seq * config.d_model;
    size_t out_count = (size_t)config.d_model * config.vocab_size;
    if (fread(model->tok_embed, sizeof(float), tok_count, f) != tok_count ||
        fread(model->pos_embed, sizeof(float), pos_count, f) != pos_count ||
        fread(model->out_proj, sizeof(float), out_count, f) != out_count ||
        fread(model->out_bias, sizeof(float), (size_t)config.vocab_size, f) != (size_t)config.vocab_size) {
        fclose(f);
        trm_model_free(model);
        return NULL;
    }
    for (int i = 0; i < config.n_layers; ++i) {
        TRMLayer *layer = &model->layers[i];
        size_t dm = (size_t)config.d_model;
        size_t dm2 = dm * dm;
        size_t dm4 = dm * 4;
        if (fread(layer->wq, sizeof(float), dm2, f) != dm2 ||
            fread(layer->wk, sizeof(float), dm2, f) != dm2 ||
            fread(layer->wv, sizeof(float), dm2, f) != dm2 ||
            fread(layer->wo, sizeof(float), dm2, f) != dm2 ||
            fread(layer->bq, sizeof(float), dm, f) != dm ||
            fread(layer->bk, sizeof(float), dm, f) != dm ||
            fread(layer->bv, sizeof(float), dm, f) != dm ||
            fread(layer->bo, sizeof(float), dm, f) != dm ||
            fread(layer->w1, sizeof(float), dm2 * 4, f) != dm2 * 4 ||
            fread(layer->b1, sizeof(float), dm4, f) != dm4 ||
            fread(layer->w2, sizeof(float), dm2 * 4, f) != dm2 * 4 ||
            fread(layer->b2, sizeof(float), dm, f) != dm ||
            fread(layer->ln1_g, sizeof(float), dm, f) != dm ||
            fread(layer->ln1_b, sizeof(float), dm, f) != dm ||
            fread(layer->ln2_g, sizeof(float), dm, f) != dm ||
            fread(layer->ln2_b, sizeof(float), dm, f) != dm) {
            fclose(f);
            trm_model_free(model);
            return NULL;
        }
    }
    fclose(f);
    return model;
}
