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

extern void *memcpy(void *dst, const void *src, unsigned long n);

int            tinyc_va_arg_i32(va_list *ap) { return va_arg(*ap, int); }
long long      tinyc_va_arg_i64(va_list *ap) { return va_arg(*ap, long long); }
double         tinyc_va_arg_f64(va_list *ap) { return va_arg(*ap, double); }
void          *tinyc_va_arg_ptr(va_list *ap) { return va_arg(*ap, void *); }

// Struct varargs: clang's wasm32 ABI passes structs by-reference for
// varargs, so the helper just chases the pointer and memcpys.
void tinyc_va_arg_struct(va_list *ap, void *out, long long size) {
    void *p = va_arg(*ap, void *);
    memcpy(out, p, (unsigned long)size);
}
