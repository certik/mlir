## WASI _start shim that calls the LLVM-generated `__original_main`
## (signature `() -> i32`) and forwards its return code to
## `proc_exit`.
##
## When LLVM's WebAssembly backend emits an `int main(...)` function it
## also emits a wrapper named plain `main` whose signature is `() ->
## ()` (the wasi-libc-compatible reactor entry shape). The actual
## program body lives in `__original_main` with the original `i32`
## return type. We hand-write `_start` here in assembly so we can call
## `__original_main` directly with the right signature; clang's C
## frontend would not let us reference `__original_main` that way
## (it rewrites our own `extern int main(void)` declarations into
## `__original_main` calls with potentially-different prototypes).

        .functype       __original_main () -> (i32)
        .functype       proc_exit (i32) -> ()
        .import_module  proc_exit, wasi_snapshot_preview1
        .import_name    proc_exit, proc_exit

        .globl          _start
        .export_name    _start, _start
        .functype       _start () -> ()
_start:
        .functype       _start () -> ()
        call            __original_main
        call            proc_exit
        end_function
