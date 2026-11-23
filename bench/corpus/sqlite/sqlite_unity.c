/* Whole-program unity TU: SQLite amalgamation + minimal OS layer + workload.
 * Build with -DSQLITE_OS_OTHER=1 (bench/corpus/__init__.py has the full flag set). */
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libc_extra.h"

#define main bench_main
#include "sqlite3.c"
#include "sqlite_os_bench.c"
#include "sqlite_bench.c"
#undef main

#include "tinyc_rt.c"
#include "libc_extra.c"
