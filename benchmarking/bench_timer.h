#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Monotonic wall time in milliseconds (libc via upstream_main builds).
double bench_now_ms(void);

typedef struct {
    double start_ms;
} BenchTimer;

static inline void bench_timer_start(BenchTimer *t) {
    t->start_ms = bench_now_ms();
}

static inline double bench_timer_elapsed_ms(const BenchTimer *t) {
    return bench_now_ms() - t->start_ms;
}

#ifdef __cplusplus
}
#endif
