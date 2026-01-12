#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        free(buf);
        return NULL;
    }
    buf[size] = '\0';
    fclose(f);
    if (out_size) {
        *out_size = (size_t)size;
    }
    return buf;
}

static int arg_int(int argc, char **argv, const char *name, int def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return atoi(argv[i + 1]);
        }
    }
    return def;
}

static const char *arg_str(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return def;
}

static void train_model(const char *data_path, const char *save_path,
                        int layers, int recursions, int d_model, int seq_len,
                        int epochs, float lr) {
    size_t data_size = 0;
    char *data = read_file(data_path, &data_size);
    if (!data) {
        fprintf(stderr, "Failed to read data file.\n");
        return;
    }

    TRMConfig config = {
        .vocab_size = 256,
        .max_seq = seq_len,
        .d_model = d_model,
        .n_layers = layers,
        .n_recursions = recursions
    };

    TRMModel *model = trm_model_create(config);
    TRMCache cache;
    trm_cache_alloc(&cache, &config, recursions + 1, seq_len);

    int *tokens = (int *)calloc((size_t)seq_len, sizeof(int));
    int *targets = (int *)calloc((size_t)seq_len, sizeof(int));
    float loss = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i + seq_len + 1 < data_size; i += seq_len) {
            for (int j = 0; j < seq_len; ++j) {
                tokens[j] = (unsigned char)data[i + j];
                targets[j] = (unsigned char)data[i + j + 1];
            }
            trm_backward(model, tokens, seq_len, targets, &loss, lr, &cache);
        }
        printf("Epoch %d loss %.4f\n", epoch + 1, loss);
    }

    trm_save(model, save_path);

    free(tokens);
    free(targets);
    trm_cache_free(&cache, &config, recursions + 1);
    trm_model_free(model);
    free(data);
}

static void generate_text(const char *model_path, const char *prompt, int steps, int seq_len) {
    TRMModel *model = trm_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model.\n");
        return;
    }

    int total_steps = model->config.n_recursions + 1;
    TRMCache cache;
    trm_cache_alloc(&cache, &model->config, total_steps, seq_len);

    int *tokens = (int *)calloc((size_t)seq_len, sizeof(int));
    size_t prompt_len = strlen(prompt);
    for (int i = 0; i < seq_len; ++i) {
        if ((size_t)i < prompt_len) {
            tokens[i] = (unsigned char)prompt[i];
        } else {
            tokens[i] = ' ';
        }
    }

    printf("%s", prompt);
    for (int step = 0; step < steps; ++step) {
        float *logits = (float *)calloc((size_t)seq_len * model->config.vocab_size, sizeof(float));
        trm_forward(model, tokens, seq_len, logits, &cache);
        int last = seq_len - 1;
        float *row = logits + last * model->config.vocab_size;
        int best = 0;
        for (int v = 1; v < model->config.vocab_size; ++v) {
            if (row[v] > row[best]) {
                best = v;
            }
        }
        memmove(tokens, tokens + 1, (size_t)(seq_len - 1) * sizeof(int));
        tokens[seq_len - 1] = best;
        putchar(best);
        free(logits);
    }
    putchar('\n');

    free(tokens);
    trm_cache_free(&cache, &model->config, total_steps);
    trm_model_free(model);
}

int main(int argc, char **argv) {
    const char *mode = arg_str(argc, argv, "--mode", "train");
    if (strcmp(mode, "train") == 0) {
        const char *data_path = arg_str(argc, argv, "--data", NULL);
        const char *save_path = arg_str(argc, argv, "--save", "trm.bin");
        int layers = arg_int(argc, argv, "--layers", 2);
        int recursions = arg_int(argc, argv, "--recursions", 4);
        int d_model = arg_int(argc, argv, "--dmodel", 64);
        int seq_len = arg_int(argc, argv, "--seq", 64);
        int epochs = arg_int(argc, argv, "--epochs", 1);
        float lr = (float)atof(arg_str(argc, argv, "--lr", "0.0005"));
        if (!data_path) {
            fprintf(stderr, "--data is required for training.\n");
            return 1;
        }
        train_model(data_path, save_path, layers, recursions, d_model, seq_len, epochs, lr);
    } else if (strcmp(mode, "generate") == 0) {
        const char *model_path = arg_str(argc, argv, "--model", NULL);
        const char *prompt = arg_str(argc, argv, "--prompt", "");
        int steps = arg_int(argc, argv, "--steps", 64);
        int seq_len = arg_int(argc, argv, "--seq", 64);
        if (!model_path) {
            fprintf(stderr, "--model is required for generation.\n");
            return 1;
        }
        generate_text(model_path, prompt, steps, seq_len);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }
    return 0;
}
