#include "gemm.h"

#include <math.h>
#include <string.h>

void matmul_nn(const float *a, const float *b, float *c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            const float *a_row = a + i * k;
            const float *b_col = b + j;
            for (int t = 0; t < k; ++t) {
                sum += a_row[t] * b_col[t * n];
            }
            c[i * n + j] = sum;
        }
    }
}

void matmul_nt(const float *a, const float *b, float *c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            const float *a_row = a + i * k;
            const float *b_row = b + j * k;
            for (int t = 0; t < k; ++t) {
                sum += a_row[t] * b_row[t];
            }
            c[i * n + j] = sum;
        }
    }
}

void matmul_tn(const float *a, const float *b, float *c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            const float *a_col = a + i;
            const float *b_col = b + j;
            for (int t = 0; t < k; ++t) {
                sum += a_col[t * m] * b_col[t * n];
            }
            c[i * n + j] = sum;
        }
    }
}

void add_inplace(float *dst, const float *src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

void add_scaled_inplace(float *dst, const float *src, float scale, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
}

void scale_inplace(float *dst, float scale, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] *= scale;
    }
}

void softmax_rows(float *dst, const float *src, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        const float *row = src + i * cols;
        float *out = dst + i * cols;
        float max_val = row[0];
        for (int j = 1; j < cols; ++j) {
            if (row[j] > max_val) {
                max_val = row[j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float v = expf(row[j] - max_val);
            out[j] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < cols; ++j) {
            out[j] *= inv;
        }
    }
}

void zero_buf(float *dst, int n) {
    memset(dst, 0, (size_t)n * sizeof(float));
}
