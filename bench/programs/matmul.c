// Runtime kernel: double-precision multiply-add over heap matrices (SSE2 path).
// (Heap buffers and 1-D indexing: tinyc does not accept global double arrays
// or multi-dimensional global arrays yet.)
#include <stdio.h>
#include <stdlib.h>

#define N 256

int main(void) {
    double *a = (double *)malloc(sizeof(double) * N * N);
    double *b = (double *)malloc(sizeof(double) * N * N);
    double *c = (double *)malloc(sizeof(double) * N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i * N + j] = (double)((i * 7 + j * 3) % 17) * 0.25;
            b[i * N + j] = (double)((i * 5 + j * 11) % 13) * 0.5;
        }
    for (int rep = 0; rep < 4; rep++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double s = 0.0;
                for (int k = 0; k < N; k++) s += a[i * N + k] * b[k * N + j];
                c[i * N + j] = s + (double)rep;
            }
    double sum = 0.0;
    for (int i = 0; i < N * N; i++) sum += c[i];
    printf("checksum = %.1f\n", sum);
    free(a); free(b); free(c);
    return 0;
}
