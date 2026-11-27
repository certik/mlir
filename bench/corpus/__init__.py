"""Benchmark corpus for bench/run.py, section (d).

Each program is a pinned upstream archive (downloaded into
$TINYC_BENCH_CACHE, sha256-checked, never vendored) plus the small porting
layer in this directory. Every program is built as ONE translation unit
(`<name>/..._unity.c` or a generated one) together with corec + corec-stdlib
(common/tinyc_rt.c) and, for Lua/SQLite, the libc gap fillers in libc/. gcc
and clang compile exactly the same TU (`-O0 -nostdlib -nostdinc -static`), so
the runtime comparison is a pure code-generation comparison.

bench/run.py always tries the tinyc build, so a program starts counting as
soon as its blockers are fixed. Known blockers (2026-11) are listed in
`blockers`; libc gaps are #342, driver gaps #338, and `main`'s argv under
--emit=elf is #334 (common/tinyc_rt.c works around it).

Fields:
  name, version, url, file, sha256, subdir   the download
  unity(src, work, for_tinyc) -> Path        the TU to compile
  inc(src) -> [dirs]                         -I dirs before corec-stdlib/corec
  defines                                    -D flags for every compiler
  tinyc_flags                                extra flags for tinyc only
  argv(src) -> [args]                        command line of the timed run
  check                                      substring the output must contain
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --- CoreMark --------------------------------------------------------------
def _coremark_unity(src: Path, work: Path, for_tinyc: bool) -> Path:
    """coremark.h has no include guard: include it once and drop the per-file
    includes."""
    parts = ["#include <stddef.h>", "#include <stdio.h>", "#include <stdlib.h>",
             "#include <string.h>", '#include "coremark.h"', "#define main bench_main"]
    for f in ("core_list_join.c", "core_main.c", "core_matrix.c", "core_state.c",
              "core_util.c"):
        text = (src / f).read_text()
        parts.append(f"/* ---- {f} ---- */")
        parts.append(re.sub(r'^#include "coremark.h"', "/* coremark.h already included */",
                            text, flags=re.M))
    parts += ["#undef main", '#include "core_portme.c"']
    body = "\n".join(parts) + "\n"
    body += '#include "tinyc_rt.c"\n'
    out = work / f"coremark.{'tinyc' if for_tinyc else 'cc'}.c"
    out.write_text(body)
    return out


# --- Lua / SQLite ----------------------------------------------------------
def _fixed_unity(path: Path):
    return lambda src, work, for_tinyc: path


LIBC_INC = [HERE / "libc" / "include"]
LIBC_TAIL = [HERE / "libc", HERE / "common"]

CORPUS = [
    {
        "name": "coremark",
        "version": "v1.01 (cfa9ab3)",
        # The archive of the v1.01 tag's commit (not of the tag name, which
        # could be moved); the sha256 catches any change of the bytes.
        "url": "https://github.com/eembc/coremark/archive/"
               "cfa9ab377835911f23d9b0831c7be302ed1f58de.tar.gz",
        "file": "coremark-cfa9ab3.tar.gz",
        "sha256": "7b342893481df2514140738bf0f077549dd1dc8a6f7b2c39025e64aea6a411af",
        "subdir": "coremark-cfa9ab377835911f23d9b0831c7be302ed1f58de",
        "unity": _coremark_unity,
        "inc": lambda src: [HERE / "coremark", src, HERE / "common"],
        "defines": ["TOTAL_DATA_SIZE=2000", "ITERATIONS=10000"],
        "tinyc_flags": ["--host-varargs"],
        "argv": lambda src: [],
        "check": "Correct operation validated",
        "blockers": "",
    },
    {
        "name": "lua",
        "version": "5.4.7",
        "url": "https://www.lua.org/ftp/lua-5.4.7.tar.gz",
        "file": "lua-5.4.7.tar.gz",
        "sha256": "9fbf5e28ef86c69858f6d3d34eccc32e911c1a28b4120ff3e84aaa70cfbf1e30",
        "subdir": "lua-5.4.7/src",
        "unity": _fixed_unity(HERE / "lua" / "lua_unity.c"),
        "inc": lambda src: [HERE / "lua", *LIBC_INC, src, *LIBC_TAIL],
        "defines": [],
        "tinyc_flags": ["--host-varargs"],
        "argv": lambda src: [str(HERE / "lua" / "bench.lua"), "10"],
        "check": "fib\t196418",
        "blockers": "comma operator (#287); #333 #335 #339 #340 #341 #314 #317; setjmp/longjmp (#337)",
    },
    {
        "name": "sqlite",
        "version": "3.46.1 amalgamation",
        "url": "https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip",
        "file": "sqlite-amalgamation-3460100.zip",
        "sha256": "77823cb110929c2bcb0f5d48e4833b5c59a8a6e40cdea3936b99e199dbbe5784",
        "subdir": "sqlite-amalgamation-3460100",
        "unity": _fixed_unity(HERE / "sqlite" / "sqlite_unity.c"),
        "inc": lambda src: [HERE / "sqlite", *LIBC_INC, src, *LIBC_TAIL],
        "defines": ["SQLITE_OS_OTHER=1", "SQLITE_THREADSAFE=0", "SQLITE_TEMP_STORE=3",
                    "SQLITE_OMIT_LOAD_EXTENSION", "SQLITE_OMIT_WAL",
                    "SQLITE_OMIT_SHARED_CACHE", "SQLITE_DEFAULT_MEMSTATUS=0",
                    "SQLITE_OMIT_LOCALTIME", "SQLITE_OMIT_DEPRECATED", "SQLITE_DQS=0"],
        "tinyc_flags": ["--host-varargs"],
        "argv": lambda src: ["200000"],
        # The recursive-CTE query (sum of squares 1..1000) prints this for any
        # row count.
        "check": "333833500",
        "blockers": "comma operator (#287), bit-fields (#336), #314 #315 #316 #317 "
                    "#290 #333 #339 #340 #341",
    },
]
