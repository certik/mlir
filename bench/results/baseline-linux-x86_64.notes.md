## Notes

This section is written by hand and is not computed by the script. `bench/run.py --notes` inserts it into the report.

### Where the time goes (from reading the code)

- **x64-regalloc.** It is the largest per-file stage. Its cost is concentrated in the files with the largest functions (see the slowest-files table in (b1)).
- **x64-setup in the lift (b2).** The reachability fixpoint in `mlir_llvm_to_elf` (`mlir_llvm_to_x64_emit.inc`) tests `SymSet` membership with a linear scan, inside loops over every call of every function. `gmap_get` is also a linear scan and calls `strlen` on every entry.
- **mem2reg and arith-gvn in the lift (b2).** These two passes take most of the lift time, which matches the picture in #225.
- **preprocess.** It was the second-largest per-file stage before #376, which replaced the linked-list macro lookup with a hash table and removed the quadratic rescan splice.

### How to read the tables

- **What workload (a) measures for tinyc.** tinyc has no object-file writer, so it runs `--emit=x64`: the whole per-file pipeline down to x86-64 machine code, which it writes as raw code instead of a `.o`. A whole-program `tinyc --emit=elf` of the same sources does not work, because file-static names collide across files.
- **The tinyc-built tinyc (c2).** This is the self-host ELF, built C → wasm → lift → x86-64. Its speed therefore reflects the wasm32 lift path and corec-stdlib, not only tinyc's direct x86 code generation.
- **CoreMark timing.** CoreMark uses a fixed fake timer (`bench/corpus/coremark/core_portme.c`) and is timed from outside the process. That keeps its output identical across compilers, so the harness can compare outputs byte for byte. With a real timer the reported times would differ, and runs under 10 s would also print CoreMark's "must execute for at least 10 secs" error.
- **Lua and SQLite.** Both are blocked by the issues listed in (d). Their gcc -O0 builds against the same libc work, and `run.py` tries tinyc on every run.

### Hosted tinyc builds use corec's byte-loop `memcpy`

Hosted (glibc-linked) tinyc builds get corec's weak `memcpy`/`memset` from `corec/platform/platform_linux.c`: a byte loop with an asm barrier. Inside the executable it takes precedence over glibc's. `clang -O0` lowers struct copies to `memcpy` calls, and gcc -O0 does not.

Before #376 this made the clang -O0 build of tinyc 2.9x slower than the gcc -O0 build. gprof put 80% of its time in that loop, and its preprocess stage was 16x slower. The new preprocessor copies much less, so on `emit.c` the clang -O0 build is now only about 1.3x slower than the gcc -O0 build (preprocess 1.9x). The (c2) table shows the current gap for the whole workload.
