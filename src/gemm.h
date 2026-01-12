#ifndef TRM_GEMM_H
#define TRM_GEMM_H

#include <stddef.h>

void matmul_nn(const float *a, const float *b, float *c, int m, int n, int k);
void matmul_nt(const float *a, const float *b, float *c, int m, int n, int k);
void matmul_tn(const float *a, const float *b, float *c, int m, int n, int k);
void add_inplace(float *dst, const float *src, int n);
void add_scaled_inplace(float *dst, const float *src, float scale, int n);
void scale_inplace(float *dst, float scale, int n);
void softmax_rows(float *dst, const float *src, int rows, int cols);
void zero_buf(float *dst, int n);

#endif
