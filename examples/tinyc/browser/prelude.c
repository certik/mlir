/* tinyC browser runtime prelude.
 *
 * The real corec runtime that an in-browser tinyC snippet links against:
 * printf / malloc / free (corec-stdlib), the WASI platform layer
 * (platform_wasm.c), and the WASI `_start` entry point. It is the same
 * single-TU root the test harness and selfhost build, minus the user
 * program — `app_main` is left undefined here and supplied by the
 * compiled snippet at link time (platform_wasm.c's `_start` ->
 * platform_init_and_run -> app_main).
 *
 * Built once at `pixi run build_tinyc_wasm` time into corec_runtime.wasm.o
 * (see examples/tinyc/build_tinyc_wasm.py) and served to the browser, so
 * the page never ships a hand-written runtime shim.
 */
#define SINGLE_TU_BUILD 1
#define COREC_STDLIB_PROVIDES_MEM 1

#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../corec-stdlib/corec/base/assert.c"
#include "../../../corec-stdlib/corec/base/exit.c"
#include "../../../corec-stdlib/corec/base/mem.c"
#include "../../../corec-stdlib/corec/base/string.c"
#include "../../../corec-stdlib/corec/base/numconv.c"
#include "../../../corec-stdlib/corec/base/format.c"
#include "../../../corec-stdlib/corec/base/strbuf.c"
#include "../../../corec-stdlib/corec/base/io.c"
#include "../../../corec-stdlib/corec/base/buddy.c"
#include "../../../corec-stdlib/corec/base/arena.c"
#include "../../../corec-stdlib/corec/base/scratch.c"
#include "../../../corec-stdlib/stdlib/string_impl.c"
#include "../../../corec-stdlib/stdlib/stdlib.c"
#include "../../../corec-stdlib/stdlib/stdio.c"
#include "../../../corec-stdlib/stdlib/printf.c"

#include "../../../corec-stdlib/corec/platform/platform_wasm.c"
