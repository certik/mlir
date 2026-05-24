// Minimal va_arg ABI shim for tinyC-compiled wasm32 binaries.
//
// When tinyC compiles `va_arg(ap, T)` it emits a call to one of these
// `tinyc_va_arg_*` helpers rather than inlining the platform-specific
// va_arg sequence directly. The helpers themselves cannot be compiled
// by tinyC (they'd recursively call themselves through va_arg!) so
// they must come from a clang-built wasm object that's linked into
// every tinyC-compiled wasm32-wasi executable.
//
// We intentionally keep this file tiny — no printf / malloc / strlen
// definitions — so it can be linked alongside a tinyC-compiled
// corec-stdlib without causing duplicate-symbol errors. The fuller
// `runtime_wasm.c` is shipped separately and used by tinyC-in-the-
// browser to compile and run hand-typed test snippets that don't pull
// in the corec stdlib.

#include <stdarg.h>
#include <stdint.h>

int            tinyc_va_arg_i32(va_list *ap) { return va_arg(*ap, int); }
long long      tinyc_va_arg_i64(va_list *ap) { return va_arg(*ap, long long); }
double         tinyc_va_arg_f64(va_list *ap) { return va_arg(*ap, double); }
void          *tinyc_va_arg_ptr(va_list *ap) { return va_arg(*ap, void *); }

// Struct varargs: tinyC emits a `call @sum_pts(i32, i64, ...)` where each
// struct vararg gets unpacked into `(sizeof(struct) + 7) / 8` consecutive
// i64 words at the call site (see the variadic-struct loop in emit.c).
// The helper must match: read that many i64 words straight out of the
// va_list and write them into the destination buffer. This is the same
// implementation as `runtime_wasm.c`'s `tinyc_va_arg_struct`, kept in
// sync intentionally — the selfhost (`selfhost_tinyc_wasm.sh`) links
// THIS object instead of `runtime_wasm.c` to avoid pulling in printf /
// malloc symbols.
void tinyc_va_arg_struct(va_list *ap, void *out, long long size) {
    long long *o = (long long *)out;
    long long words = (size + 7) / 8;
    for (long long i = 0; i < words; i++) {
        o[i] = va_arg(*ap, long long);
    }
}
