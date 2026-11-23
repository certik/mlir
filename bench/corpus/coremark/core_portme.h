/* CoreMark port for tinyc (corec-stdlib libc, no time.h).
 * No timer: the benchmark harness times the whole process externally, so
 * ITERATIONS must be fixed at build time (-DITERATIONS=N, N > 0). The reported
 * time is a fixed fake 10 s (see core_portme.c); correctness is the CRC block
 * ending in "Correct operation validated". */
#ifndef CORE_PORTME_H
#define CORE_PORTME_H

#include <stddef.h>

#define HAS_FLOAT 0
#define HAS_TIME_H 0
#define USE_CLOCK 0
#define HAS_STDIO 1
#define HAS_PRINTF 1

#ifndef ITERATIONS
#define ITERATIONS 2000
#endif

typedef unsigned int CORE_TICKS;

/* The same string for every compiler, so bench/run.py can compare outputs. */
#ifndef COMPILER_VERSION
#define COMPILER_VERSION "bench"
#endif
#ifndef COMPILER_FLAGS
#define COMPILER_FLAGS "unknown"
#endif
#define MEM_LOCATION "STACK"

typedef signed short ee_s16;
typedef unsigned short ee_u16;
typedef signed int ee_s32;
typedef double ee_f32;
typedef unsigned char ee_u8;
typedef unsigned int ee_u32;
typedef unsigned long ee_ptr_int;
typedef size_t ee_size_t;
#define align_mem(x) (void *)(4 + (((ee_ptr_int)(x) - 1) & ~3))

#define SEED_METHOD SEED_VOLATILE
#define MEM_METHOD MEM_STACK
#define MULTITHREAD 1
#define USE_PTHREAD 0
#define USE_FORK 0
#define USE_SOCKET 0
#define MAIN_HAS_NOARGC 0
#define MAIN_HAS_NORETURN 0

extern ee_u32 default_num_contexts;

typedef struct CORE_PORTABLE_S {
    ee_u8 portable_id;
} core_portable;

void portable_init(core_portable *p, int *argc, char *argv[]);
void portable_fini(core_portable *p);

#if !defined(PROFILE_RUN) && !defined(PERFORMANCE_RUN) && !defined(VALIDATION_RUN)
#if (TOTAL_DATA_SIZE == 1200)
#define PROFILE_RUN 1
#elif (TOTAL_DATA_SIZE == 2000)
#define PERFORMANCE_RUN 1
#else
#define VALIDATION_RUN 1
#endif
#endif

#endif /* CORE_PORTME_H */
