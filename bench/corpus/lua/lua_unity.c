/* Whole-program unity TU for the Lua 5.4 interpreter (lua.c) built against
 * corec-stdlib + shims/libc. Include path order: bench/corpus/lua,
 * bench/corpus/libc/include, the Lua src dir, bench/corpus/libc,
 * bench/corpus/common, corec-stdlib/stdlib, corec. */
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libc_extra.h"

#define main bench_main
/* core */
#include "lapi.c"
#include "lcode.c"
#include "lctype.c"
#include "ldebug.c"
#include "ldo.c"
#include "ldump.c"
#include "lfunc.c"
#include "lgc.c"
#include "llex.c"
#include "lmem.c"
#include "lobject.c"
#include "lopcodes.c"
#include "lparser.c"
#include "lstate.c"
#include "lstring.c"
#include "ltable.c"
#include "ltm.c"
#include "lundump.c"
#include "lvm.c"
#include "lzio.c"
/* auxiliary + standard libraries (no io/os/package) */
#include "lauxlib.c"
#include "lbaselib.c"
#include "lcorolib.c"
#include "ldblib.c"
#include "lmathlib.c"
#include "lstrlib.c"
#include "ltablib.c"
#include "lutf8lib.c"
#include "linit_bench.c"
/* stand-alone interpreter */
#include "lua.c"
#undef main

#include "tinyc_rt.c"
#include "libc_extra.c"
