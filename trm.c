#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_MAGIC "TRM1"

typedef struct {
    int vocab_size;
    int seq_len;
    int hidden_size;
    int l_cycles;
    float rms_eps;
} Config;

typedef struct {
    float *w1;  // hidden_size x (2 * hidden_size)
    float *b1;  // 2 * hidden_size
    float *w2;  // hidden_size x hidden_size
    float *b2;  // hidden_size
    float *rms_weight; // hidden_size

    float *gw1;
    float *gb1;
    float *gw2;
    float *gb2;
    float *grms_weight;
} Block;

typedef struct {
    Config cfg;
    Block block;

    float *token_embed; // vocab_size x hidden_size
    float *pos_embed;   // seq_len x hidden_size
    float *lm_head;     // hidden_size x vocab_size

    float *g_token_embed;
    float *g_pos_embed;
    float *g_lm_head;

    float *h_init; // hidden_size
    float *l_init; // hidden_size
    float *g_h_init;
    float *g_l_init;
} Model;

typedef struct {
    float *x;      // input to block (seq_len x hidden_size)
    float *u;      // pre-activation (seq_len x 2*hidden)
    float *swish;  // swish(b) (seq_len x hidden)
    float *v;      // after swiglu (seq_len x hidden)
    float *y;      // after w2 (seq_len x hidden)
    float *rms_norm; // rms per token (seq_len)
    float *x_norm; // normalized x (seq_len x hidden)
} BlockCache;

static float randf() {
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void zero_array(float *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = 0.0f;
    }
}

static void init_block(Block *block, int hidden) {
    size_t w1_size = (size_t)hidden * (size_t)hidden * 2;
    size_t w2_size = (size_t)hidden * (size_t)hidden;
    size_t b1_size = (size_t)hidden * 2;
    size_t b2_size = (size_t)hidden;

    block->w1 = malloc(sizeof(float) * w1_size);
    block->b1 = malloc(sizeof(float) * b1_size);
    block->w2 = malloc(sizeof(float) * w2_size);
    block->b2 = malloc(sizeof(float) * b2_size);
    block->rms_weight = malloc(sizeof(float) * hidden);

    block->gw1 = malloc(sizeof(float) * w1_size);
    block->gb1 = malloc(sizeof(float) * b1_size);
    block->gw2 = malloc(sizeof(float) * w2_size);
    block->gb2 = malloc(sizeof(float) * b2_size);
    block->grms_weight = malloc(sizeof(float) * hidden);

    for (size_t i = 0; i < w1_size; i++) {
        block->w1[i] = randf() * 0.1f;
    }
    for (size_t i = 0; i < b1_size; i++) {
        block->b1[i] = 0.0f;
    }
    for (size_t i = 0; i < w2_size; i++) {
        block->w2[i] = randf() * 0.1f;
    }
    for (size_t i = 0; i < b2_size; i++) {
        block->b2[i] = 0.0f;
        block->rms_weight[i] = 1.0f;
    }
}

static void init_model(Model *model, Config cfg) {
    model->cfg = cfg;
    int vocab = cfg.vocab_size;
    int hidden = cfg.hidden_size;
    int seq_len = cfg.seq_len;

    init_block(&model->block, hidden);

    model->token_embed = malloc(sizeof(float) * (size_t)vocab * (size_t)hidden);
    model->pos_embed = malloc(sizeof(float) * (size_t)seq_len * (size_t)hidden);
    model->lm_head = malloc(sizeof(float) * (size_t)hidden * (size_t)vocab);
    model->h_init = malloc(sizeof(float) * hidden);
    model->l_init = malloc(sizeof(float) * hidden);

    model->g_token_embed = malloc(sizeof(float) * (size_t)vocab * (size_t)hidden);
    model->g_pos_embed = malloc(sizeof(float) * (size_t)seq_len * (size_t)hidden);
    model->g_lm_head = malloc(sizeof(float) * (size_t)hidden * (size_t)vocab);
    model->g_h_init = malloc(sizeof(float) * hidden);
    model->g_l_init = malloc(sizeof(float) * hidden);

    for (size_t i = 0; i < (size_t)vocab * (size_t)hidden; i++) {
        model->token_embed[i] = randf() * 0.1f;
    }
    for (size_t i = 0; i < (size_t)seq_len * (size_t)hidden; i++) {
        model->pos_embed[i] = randf() * 0.1f;
    }
    for (size_t i = 0; i < (size_t)hidden * (size_t)vocab; i++) {
        model->lm_head[i] = randf() * 0.1f;
    }
    for (int i = 0; i < hidden; i++) {
        model->h_init[i] = randf() * 0.1f;
        model->l_init[i] = randf() * 0.1f;
    }
}

static void zero_grad(Model *model) {
    Config cfg = model->cfg;
    zero_array(model->block.gw1, (size_t)cfg.hidden_size * (size_t)cfg.hidden_size * 2);
    zero_array(model->block.gb1, (size_t)cfg.hidden_size * 2);
    zero_array(model->block.gw2, (size_t)cfg.hidden_size * (size_t)cfg.hidden_size);
    zero_array(model->block.gb2, (size_t)cfg.hidden_size);
    zero_array(model->block.grms_weight, (size_t)cfg.hidden_size);
    zero_array(model->g_token_embed, (size_t)cfg.vocab_size * (size_t)cfg.hidden_size);
    zero_array(model->g_pos_embed, (size_t)cfg.seq_len * (size_t)cfg.hidden_size);
    zero_array(model->g_lm_head, (size_t)cfg.hidden_size * (size_t)cfg.vocab_size);
    zero_array(model->g_h_init, (size_t)cfg.hidden_size);
    zero_array(model->g_l_init, (size_t)cfg.hidden_size);
}

static void block_forward(const Model *model, const float *input, float *output, BlockCache *cache) {
    int hidden = model->cfg.hidden_size;
    int seq_len = model->cfg.seq_len;
    const Block *block = &model->block;

    memcpy(cache->x, input, sizeof(float) * (size_t)seq_len * (size_t)hidden);

    for (int t = 0; t < seq_len; t++) {
        const float *x = input + t * hidden;
        float *u = cache->u + t * hidden * 2;
        for (int j = 0; j < hidden * 2; j++) {
            float acc = block->b1[j];
            for (int i = 0; i < hidden; i++) {
                acc += x[i] * block->w1[i * hidden * 2 + j];
            }
            u[j] = acc;
        }
        float *swish = cache->swish + t * hidden;
        float *v = cache->v + t * hidden;
        for (int i = 0; i < hidden; i++) {
            float a = u[i];
            float b = u[i + hidden];
            float s = b * sigmoid(b);
            swish[i] = s;
            v[i] = a * s;
        }
        float *y = cache->y + t * hidden;
        for (int j = 0; j < hidden; j++) {
            float acc = block->b2[j];
            for (int i = 0; i < hidden; i++) {
                acc += v[i] * block->w2[i * hidden + j];
            }
            y[j] = acc;
        }
        float *x_norm = cache->x_norm + t * hidden;
        float rms = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float val = x[i] + y[i];
            rms += val * val;
            x_norm[i] = val;
        }
        rms = sqrtf(rms / (float)hidden + model->cfg.rms_eps);
        cache->rms_norm[t] = rms;
        for (int i = 0; i < hidden; i++) {
            output[t * hidden + i] = (x_norm[i] / rms) * block->rms_weight[i];
        }
    }
}

static void block_backward(Model *model, const float *d_output, BlockCache *cache, float *d_input) {
    int hidden = model->cfg.hidden_size;
    int seq_len = model->cfg.seq_len;
    Block *block = &model->block;

    for (int t = 0; t < seq_len; t++) {
        float *x = cache->x + t * hidden;
        float *u = cache->u + t * hidden * 2;
        float *swish = cache->swish + t * hidden;
        float *v = cache->v + t * hidden;
        float *y = cache->y + t * hidden;
        float *x_norm = cache->x_norm + t * hidden;
        float rms = cache->rms_norm[t];
        const float *d_out = d_output + t * hidden;

        float *d_xnorm = malloc(sizeof(float) * hidden);
        float sum_d = 0.0f;
        for (int i = 0; i < hidden; i++) {
            block->grms_weight[i] += d_out[i] * (x_norm[i] / rms);
            d_xnorm[i] = d_out[i] * block->rms_weight[i] / rms;
            sum_d += d_out[i] * block->rms_weight[i] * x_norm[i];
        }
        float coeff = sum_d / (rms * rms * (float)hidden);
        for (int i = 0; i < hidden; i++) {
            d_xnorm[i] -= coeff * x_norm[i];
        }

        float *d_y = malloc(sizeof(float) * hidden);
        for (int i = 0; i < hidden; i++) {
            d_y[i] = d_xnorm[i];
            d_input[t * hidden + i] = d_xnorm[i];
        }

        for (int i = 0; i < hidden; i++) {
            for (int j = 0; j < hidden; j++) {
                block->gw2[i * hidden + j] += v[i] * d_y[j];
            }
        }
        for (int j = 0; j < hidden; j++) {
            block->gb2[j] += d_y[j];
        }

        float *d_v = malloc(sizeof(float) * hidden);
        for (int i = 0; i < hidden; i++) {
            float acc = 0.0f;
            for (int j = 0; j < hidden; j++) {
                acc += d_y[j] * block->w2[i * hidden + j];
            }
            d_v[i] = acc;
        }

        float *d_u = malloc(sizeof(float) * hidden * 2);
        for (int i = 0; i < hidden; i++) {
            float a = u[i];
            float b = u[i + hidden];
            float s = swish[i];
            float sig = sigmoid(b);
            float ds = sig + b * sig * (1.0f - sig);
            d_u[i] = d_v[i] * s;
            d_u[i + hidden] = d_v[i] * a * ds;
        }

        for (int i = 0; i < hidden; i++) {
            for (int j = 0; j < hidden * 2; j++) {
                block->gw1[i * hidden * 2 + j] += x[i] * d_u[j];
            }
        }
        for (int j = 0; j < hidden * 2; j++) {
            block->gb1[j] += d_u[j];
        }
        for (int i = 0; i < hidden; i++) {
            float acc = 0.0f;
            for (int j = 0; j < hidden * 2; j++) {
                acc += d_u[j] * block->w1[i * hidden * 2 + j];
            }
            d_input[t * hidden + i] += acc;
        }

        free(d_xnorm);
        free(d_y);
        free(d_v);
        free(d_u);
    }
}

static void update_params(Model *model, float lr) {
    Config cfg = model->cfg;
    size_t w1_size = (size_t)cfg.hidden_size * (size_t)cfg.hidden_size * 2;
    size_t w2_size = (size_t)cfg.hidden_size * (size_t)cfg.hidden_size;
    size_t b1_size = (size_t)cfg.hidden_size * 2;
    size_t b2_size = (size_t)cfg.hidden_size;
    size_t embed_size = (size_t)cfg.vocab_size * (size_t)cfg.hidden_size;
    size_t pos_size = (size_t)cfg.seq_len * (size_t)cfg.hidden_size;
    size_t lm_size = (size_t)cfg.hidden_size * (size_t)cfg.vocab_size;

    for (size_t i = 0; i < w1_size; i++) {
        model->block.w1[i] -= lr * model->block.gw1[i];
    }
    for (size_t i = 0; i < w2_size; i++) {
        model->block.w2[i] -= lr * model->block.gw2[i];
    }
    for (size_t i = 0; i < b1_size; i++) {
        model->block.b1[i] -= lr * model->block.gb1[i];
    }
    for (size_t i = 0; i < b2_size; i++) {
        model->block.b2[i] -= lr * model->block.gb2[i];
        model->block.rms_weight[i] -= lr * model->block.grms_weight[i];
        model->h_init[i] -= lr * model->g_h_init[i];
        model->l_init[i] -= lr * model->g_l_init[i];
    }
    for (size_t i = 0; i < embed_size; i++) {
        model->token_embed[i] -= lr * model->g_token_embed[i];
    }
    for (size_t i = 0; i < pos_size; i++) {
        model->pos_embed[i] -= lr * model->g_pos_embed[i];
    }
    for (size_t i = 0; i < lm_size; i++) {
        model->lm_head[i] -= lr * model->g_lm_head[i];
    }
}

static int save_model(const char *path, const Model *model, const char *vocab) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return 0;
    }
    fwrite(MODEL_MAGIC, 1, 4, f);
    fwrite(&model->cfg, sizeof(Config), 1, f);
    fwrite(vocab, 1, (size_t)model->cfg.vocab_size, f);
    fwrite(model->token_embed, sizeof(float),
           (size_t)model->cfg.vocab_size * (size_t)model->cfg.hidden_size, f);
    fwrite(model->pos_embed, sizeof(float),
           (size_t)model->cfg.seq_len * (size_t)model->cfg.hidden_size, f);
    fwrite(model->lm_head, sizeof(float),
           (size_t)model->cfg.hidden_size * (size_t)model->cfg.vocab_size, f);
    fwrite(model->h_init, sizeof(float), (size_t)model->cfg.hidden_size, f);
    fwrite(model->l_init, sizeof(float), (size_t)model->cfg.hidden_size, f);

    fwrite(model->block.w1, sizeof(float),
           (size_t)model->cfg.hidden_size * (size_t)model->cfg.hidden_size * 2, f);
    fwrite(model->block.b1, sizeof(float),
           (size_t)model->cfg.hidden_size * 2, f);
    fwrite(model->block.w2, sizeof(float),
           (size_t)model->cfg.hidden_size * (size_t)model->cfg.hidden_size, f);
    fwrite(model->block.b2, sizeof(float),
           (size_t)model->cfg.hidden_size, f);
    fwrite(model->block.rms_weight, sizeof(float),
           (size_t)model->cfg.hidden_size, f);

    fclose(f);
    return 1;
}

static int load_model(const char *path, Model *model, char *vocab) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return 0;
    }
    char magic[5] = {0};
    fread(magic, 1, 4, f);
    if (strncmp(magic, MODEL_MAGIC, 4) != 0) {
        fclose(f);
        return 0;
    }
    Config cfg;
    fread(&cfg, sizeof(Config), 1, f);
    init_model(model, cfg);
    fread(vocab, 1, (size_t)cfg.vocab_size, f);
    vocab[cfg.vocab_size] = '\0';

    fread(model->token_embed, sizeof(float),
          (size_t)cfg.vocab_size * (size_t)cfg.hidden_size, f);
    fread(model->pos_embed, sizeof(float),
          (size_t)cfg.seq_len * (size_t)cfg.hidden_size, f);
    fread(model->lm_head, sizeof(float),
          (size_t)cfg.hidden_size * (size_t)cfg.vocab_size, f);
    fread(model->h_init, sizeof(float), (size_t)cfg.hidden_size, f);
    fread(model->l_init, sizeof(float), (size_t)cfg.hidden_size, f);

    fread(model->block.w1, sizeof(float),
          (size_t)cfg.hidden_size * (size_t)cfg.hidden_size * 2, f);
    fread(model->block.b1, sizeof(float),
          (size_t)cfg.hidden_size * 2, f);
    fread(model->block.w2, sizeof(float),
          (size_t)cfg.hidden_size * (size_t)cfg.hidden_size, f);
    fread(model->block.b2, sizeof(float),
          (size_t)cfg.hidden_size, f);
    fread(model->block.rms_weight, sizeof(float),
          (size_t)cfg.hidden_size, f);

    fclose(f);
    return 1;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t read = fread(buf, 1, (size_t)len, f);
    buf[read] = '\0';
    fclose(f);
    if (len_out) {
        *len_out = read;
    }
    return buf;
}

static int build_vocab(const char *text, size_t len, char *vocab) {
    int used[256] = {0};
    int count = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (!used[c]) {
            used[c] = 1;
            vocab[count++] = (char)c;
        }
    }
    return count;
}

static int char_to_id(const char *vocab, int vocab_size, char c) {
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i] == c) {
            return i;
        }
    }
    return 0;
}

static char id_to_char(const char *vocab, int vocab_size, int id) {
    if (id < 0 || id >= vocab_size) {
        return '?';
    }
    return vocab[id];
}

static void forward(Model *model, const int *tokens, float *logits,
                    float *z_h, float *z_l, BlockCache *l_cache, BlockCache *h_cache,
                    float *input_embed) {
    Config cfg = model->cfg;
    int hidden = cfg.hidden_size;
    int seq_len = cfg.seq_len;

    float scale = sqrtf((float)hidden);
    for (int t = 0; t < seq_len; t++) {
        int token = tokens[t];
        for (int i = 0; i < hidden; i++) {
            float val = model->token_embed[token * hidden + i] + model->pos_embed[t * hidden + i];
            input_embed[t * hidden + i] = val * scale;
        }
    }

    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < hidden; i++) {
            z_h[t * hidden + i] = model->h_init[i];
            z_l[t * hidden + i] = model->l_init[i];
        }
    }

    float *temp = malloc(sizeof(float) * (size_t)seq_len * (size_t)hidden);
    for (int step = 0; step < cfg.l_cycles; step++) {
        for (int t = 0; t < seq_len; t++) {
            for (int i = 0; i < hidden; i++) {
                temp[t * hidden + i] = z_l[t * hidden + i] + z_h[t * hidden + i] + input_embed[t * hidden + i];
            }
        }
        block_forward(model, temp, z_l, l_cache);
    }
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < hidden; i++) {
            temp[t * hidden + i] = z_h[t * hidden + i] + z_l[t * hidden + i];
        }
    }
    block_forward(model, temp, z_h, h_cache);
    free(temp);

    for (int t = 0; t < seq_len; t++) {
        for (int v = 0; v < cfg.vocab_size; v++) {
            float acc = 0.0f;
            for (int i = 0; i < hidden; i++) {
                acc += z_h[t * hidden + i] * model->lm_head[i * cfg.vocab_size + v];
            }
            logits[t * cfg.vocab_size + v] = acc;
        }
    }
}

static float softmax_loss(const float *logits, const int *targets, int seq_len, int vocab_size, float *d_logits) {
    float loss = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const float *row = logits + t * vocab_size;
        float max = row[0];
        for (int v = 1; v < vocab_size; v++) {
            if (row[v] > max) {
                max = row[v];
            }
        }
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum += expf(row[v] - max);
        }
        float logsum = logf(sum) + max;
        int target = targets[t];
        loss += logsum - row[target];
        for (int v = 0; v < vocab_size; v++) {
            float prob = expf(row[v] - logsum);
            d_logits[t * vocab_size + v] = prob;
        }
        d_logits[t * vocab_size + target] -= 1.0f;
    }
    return loss / (float)seq_len;
}

static void backward(Model *model, const int *tokens, const float *d_logits,
                     float *z_h, float *z_l,
                     BlockCache *l_cache, BlockCache *h_cache) {
    Config cfg = model->cfg;
    int hidden = cfg.hidden_size;
    int seq_len = cfg.seq_len;

    float *d_z_h = malloc(sizeof(float) * (size_t)seq_len * (size_t)hidden);
    float *d_z_l = malloc(sizeof(float) * (size_t)seq_len * (size_t)hidden);
    zero_array(d_z_h, (size_t)seq_len * (size_t)hidden);
    zero_array(d_z_l, (size_t)seq_len * (size_t)hidden);

    for (int t = 0; t < seq_len; t++) {
        for (int v = 0; v < cfg.vocab_size; v++) {
            float d = d_logits[t * cfg.vocab_size + v];
            for (int i = 0; i < hidden; i++) {
                model->g_lm_head[i * cfg.vocab_size + v] += z_h[t * hidden + i] * d;
                d_z_h[t * hidden + i] += model->lm_head[i * cfg.vocab_size + v] * d;
            }
        }
    }

    float *d_input_h = malloc(sizeof(float) * (size_t)seq_len * (size_t)hidden);
    zero_array(d_input_h, (size_t)seq_len * (size_t)hidden);
    block_backward(model, d_z_h, h_cache, d_input_h);

    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < hidden; i++) {
            d_z_l[t * hidden + i] += d_input_h[t * hidden + i];
            model->g_h_init[i] += d_input_h[t * hidden + i];
        }
    }

    float *d_input_l = malloc(sizeof(float) * (size_t)seq_len * (size_t)hidden);
    zero_array(d_input_l, (size_t)seq_len * (size_t)hidden);
    block_backward(model, d_z_l, l_cache, d_input_l);

    float scale = sqrtf((float)hidden);
    for (int t = 0; t < seq_len; t++) {
        int token = tokens[t];
        for (int i = 0; i < hidden; i++) {
            float grad = d_input_l[t * hidden + i];
            model->g_l_init[i] += grad;
            model->g_token_embed[token * hidden + i] += grad * scale;
            model->g_pos_embed[t * hidden + i] += grad * scale;
        }
    }

    free(d_z_h);
    free(d_z_l);
    free(d_input_h);
    free(d_input_l);
}

static void generate(Model *model, const char *vocab, const char *prompt, int steps) {
    Config cfg = model->cfg;
    int seq_len = cfg.seq_len;
    int prompt_len = (int)strlen(prompt);
    int *tokens = malloc(sizeof(int) * seq_len);

    for (int t = 0; t < seq_len; t++) {
        char c = prompt_len > 0 ? prompt[t % prompt_len] : vocab[0];
        tokens[t] = char_to_id(vocab, cfg.vocab_size, c);
    }

    float *logits = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.vocab_size);
    float *z_h = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size);
    float *z_l = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size);
    float *input_embed = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size);

    BlockCache l_cache = {
        .x = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
        .u = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size * 2),
        .swish = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
        .v = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
        .y = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
        .rms_norm = malloc(sizeof(float) * (size_t)seq_len),
        .x_norm = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
    };
    BlockCache h_cache = {
        .x = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
        .u = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size * 2),
        .swish = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
        .v = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
        .y = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
        .rms_norm = malloc(sizeof(float) * (size_t)seq_len),
        .x_norm = malloc(sizeof(float) * (size_t)seq_len * (size_t)cfg.hidden_size),
    };

    printf("\nGenerated:\n");
    for (int step = 0; step < steps; step++) {
        forward(model, tokens, logits, z_h, z_l, &l_cache, &h_cache, input_embed);
        int last = seq_len - 1;
        int best = 0;
        float best_val = logits[last * cfg.vocab_size];
        for (int v = 1; v < cfg.vocab_size; v++) {
            float val = logits[last * cfg.vocab_size + v];
            if (val > best_val) {
                best_val = val;
                best = v;
            }
        }
        char next = id_to_char(vocab, cfg.vocab_size, best);
        putchar(next);
        for (int t = 0; t < seq_len - 1; t++) {
            tokens[t] = tokens[t + 1];
        }
        tokens[seq_len - 1] = best;
    }
    putchar('\n');

    free(tokens);
    free(logits);
    free(z_h);
    free(z_l);
    free(input_embed);
    free(l_cache.x);
    free(l_cache.u);
    free(l_cache.swish);
    free(l_cache.v);
    free(l_cache.y);
    free(l_cache.rms_norm);
    free(l_cache.x_norm);
    free(h_cache.x);
    free(h_cache.u);
    free(h_cache.swish);
    free(h_cache.v);
    free(h_cache.y);
    free(h_cache.rms_norm);
    free(h_cache.x_norm);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <text_file> [--save path] [--load path] [--generate N] [--prompt text] [--steps N]\n",
               argv[0]);
        return 1;
    }

    char vocab[256];
    Model model;
    srand(0);
    int generate_steps = 200;
    int train_steps = 500;
    const char *prompt = "Hello";
    const char *save_path = NULL;
    const char *load_path = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            load_path = argv[++i];
        } else if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            train_steps = atoi(argv[++i]);
        }
    }

    size_t len = 0;
    char *text = NULL;
    if (load_path) {
        if (!load_model(load_path, &model, vocab)) {
            printf("Failed to load model\n");
            return 1;
        }
        printf("Loaded model from %s\n", load_path);
    } else {
        text = read_file(argv[1], &len);
        if (!text) {
            printf("Failed to read file\n");
            return 1;
        }
        int vocab_size = build_vocab(text, len, vocab);
        printf("Loaded %zu chars, vocab size %d\n", len, vocab_size);
        Config cfg = {
            .vocab_size = vocab_size,
            .seq_len = 32,
            .hidden_size = 64,
            .l_cycles = 2,
            .rms_eps = 1e-5f,
        };
        init_model(&model, cfg);
    }

    float *logits = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.vocab_size);
    float *z_h = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size);
    float *z_l = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size);
    float *input_embed = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size);
    float *d_logits = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.vocab_size);
    int *tokens = malloc(sizeof(int) * model.cfg.seq_len);
    int *targets = malloc(sizeof(int) * model.cfg.seq_len);

    BlockCache l_cache = {
        .x = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
        .u = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size * 2),
        .swish = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
        .v = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
        .y = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
        .rms_norm = malloc(sizeof(float) * (size_t)model.cfg.seq_len),
        .x_norm = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
    };
    BlockCache h_cache = {
        .x = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
        .u = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size * 2),
        .swish = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
        .v = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
        .y = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
        .rms_norm = malloc(sizeof(float) * (size_t)model.cfg.seq_len),
        .x_norm = malloc(sizeof(float) * (size_t)model.cfg.seq_len * (size_t)model.cfg.hidden_size),
    };

    if (text) {
        float lr = 0.01f;
        for (int step = 0; step < train_steps; step++) {
            int start = rand() % (int)(len - model.cfg.seq_len - 1);
            for (int t = 0; t < model.cfg.seq_len; t++) {
                tokens[t] = char_to_id(vocab, model.cfg.vocab_size, text[start + t]);
                targets[t] = char_to_id(vocab, model.cfg.vocab_size, text[start + t + 1]);
            }
            zero_grad(&model);
            forward(&model, tokens, logits, z_h, z_l, &l_cache, &h_cache, input_embed);
            float loss = softmax_loss(logits, targets, model.cfg.seq_len, model.cfg.vocab_size, d_logits);
            backward(&model, tokens, d_logits, z_h, z_l, &l_cache, &h_cache);
            update_params(&model, lr);
            if (step % 50 == 0) {
                printf("Step %d loss %.4f\n", step, loss);
            }
        }
    }

    if (save_path) {
        if (!save_model(save_path, &model, vocab)) {
            printf("Failed to save model\n");
        } else {
            printf("Saved model to %s\n", save_path);
        }
    }

    generate(&model, vocab, prompt, generate_steps);

    free(text);
    free(logits);
    free(z_h);
    free(z_l);
    free(input_embed);
    free(d_logits);
    free(tokens);
    free(targets);
    free(l_cache.x);
    free(l_cache.u);
    free(l_cache.swish);
    free(l_cache.v);
    free(l_cache.y);
    free(l_cache.rms_norm);
    free(l_cache.x_norm);
    free(h_cache.x);
    free(h_cache.u);
    free(h_cache.swish);
    free(h_cache.v);
    free(h_cache.y);
    free(h_cache.rms_norm);
    free(h_cache.x_norm);
    return 0;
}
