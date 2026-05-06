// Tiny runtime for tinyC: implements vector.print's lowered hooks
// (printI64, printNewline) so that programs lowered via the standard
// MLIR conversion pipeline can be linked into a real executable.

#include <stdio.h>
#include <stdint.h>

void printI64(int64_t v) { printf("%lld", (long long)v); }
void printNewline(void)  { printf("\n"); }
// vector.print on i32 actually lowers via signext to printI64 above, but
// some MLIR versions emit a direct printI32 call — provide it for safety.
void printI32(int32_t v) { printf("%d\n", v); }
// Float printing: tinyC's `print(<float>)` lowers to a direct call to
// printF32 followed by printNewline. We use %g for compact, exact-enough
// output that matches typical C float formatting.
void printF32(float v) { printf("%g", (double)v); }
void printF64(double v) { printf("%g", v); }

// tinyC's `print(<string>)` lowers to a direct call to @printStr. We
// emit the bytes (which are NUL-terminated by the string-literal global)
// followed by a newline to mirror the behavior of `print(<int>)` and
// `print(<float>)`.
void printStr(const char *s) {
    if (s) fputs(s, stdout);
    fputc('\n', stdout);
}

