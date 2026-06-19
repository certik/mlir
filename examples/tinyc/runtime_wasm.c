// WebAssembly/WASI runtime for tinyC. Provides lowered print hooks
// (printI64, printNewline, printI32, printF32, printF64, printStr,
// plus tinyc_va_arg_*) built on raw WASI
// `fd_write`/`proc_exit` imports so that it can be compiled with
// `clang --target=wasm32-wasi -nostdlib -nostdinc -fno-builtin` and
// linked with `wasm-ld` without a wasi-libc sysroot.
//
// Also provides the `_start` WASI entry point that calls `main()`.

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

// --- WASI imports ---------------------------------------------------------

typedef struct {
    const void *buf;
    uint32_t buf_len;
} ciovec_t;

#define WASI_IMPORT(name) \
    __attribute__((import_module("wasi_snapshot_preview1"), import_name(#name)))

WASI_IMPORT(fd_write)
uint32_t fd_write(uint32_t fd, const ciovec_t *iovs, uint32_t iovs_len,
                  uint32_t *nwritten);

// --- Tiny output helpers --------------------------------------------------

static void wasm_write(const char *buf, uint32_t len) {
    if (len == 0) return;
    ciovec_t iov = { buf, len };
    uint32_t nw = 0;
    fd_write(1, &iov, 1, &nw);
}

static uint32_t wasm_strlen(const char *s) {
    uint32_t n = 0;
    while (s[n]) n++;
    return n;
}

// Format a signed 64-bit integer into `buf` (no NUL). Returns length.
static uint32_t fmt_i64(int64_t v, char *buf) {
    char tmp[32];
    uint64_t u;
    int neg = 0;
    if (v < 0) { neg = 1; u = (uint64_t)(-(v + 1)) + 1u; }
    else       {           u = (uint64_t)v; }
    int n = 0;
    if (u == 0) tmp[n++] = '0';
    while (u) { tmp[n++] = (char)('0' + (u % 10)); u /= 10; }
    uint32_t out = 0;
    if (neg) buf[out++] = '-';
    for (int i = n - 1; i >= 0; i--) buf[out++] = tmp[i];
    return out;
}

// Format a double in a manner roughly compatible with C's "%g". This is
// intentionally minimal: enough for tinyC test output, not a full
// printf. We emit up to 6 significant digits, trimming trailing zeros.
static uint32_t fmt_f64(double v, char *buf) {
    uint32_t out = 0;
    if (v != v) { // NaN
        buf[out++] = 'n'; buf[out++] = 'a'; buf[out++] = 'n';
        return out;
    }
    if (v < 0) { buf[out++] = '-'; v = -v; }
    if (v > 1e308) { // inf
        buf[out++] = 'i'; buf[out++] = 'n'; buf[out++] = 'f';
        return out;
    }
    if (v == 0.0) {
        buf[out++] = '0';
        return out;
    }
    // Compute exponent for scientific normalization.
    int exp10 = 0;
    double m = v;
    while (m >= 10.0) { m /= 10.0; exp10++; }
    while (m < 1.0)   { m *= 10.0; exp10--; }
    // 6 significant digits: extract digits.
    char digits[8] = {0};
    for (int i = 0; i < 6; i++) {
        int d = (int)m;
        if (d < 0) d = 0; if (d > 9) d = 9;
        digits[i] = (char)('0' + d);
        m = (m - d) * 10.0;
    }
    // Trim trailing zeros.
    int ndig = 6;
    while (ndig > 1 && digits[ndig - 1] == '0') ndig--;

    if (exp10 >= -4 && exp10 < 6) {
        // Fixed notation.
        if (exp10 >= 0) {
            int int_digits = exp10 + 1;
            for (int i = 0; i < int_digits; i++) {
                buf[out++] = (i < ndig) ? digits[i] : '0';
            }
            if (int_digits < ndig) {
                buf[out++] = '.';
                for (int i = int_digits; i < ndig; i++) buf[out++] = digits[i];
            }
        } else {
            buf[out++] = '0';
            buf[out++] = '.';
            for (int i = 0; i < -exp10 - 1; i++) buf[out++] = '0';
            for (int i = 0; i < ndig; i++) buf[out++] = digits[i];
        }
    } else {
        // Scientific notation.
        buf[out++] = digits[0];
        if (ndig > 1) {
            buf[out++] = '.';
            for (int i = 1; i < ndig; i++) buf[out++] = digits[i];
        }
        buf[out++] = 'e';
        if (exp10 < 0) { buf[out++] = '-'; exp10 = -exp10; }
        else           { buf[out++] = '+'; }
        char ebuf[8]; int en = 0;
        if (exp10 == 0) ebuf[en++] = '0';
        while (exp10) { ebuf[en++] = (char)('0' + (exp10 % 10)); exp10 /= 10; }
        if (en < 2) buf[out++] = '0';
        for (int i = en - 1; i >= 0; i--) buf[out++] = ebuf[i];
    }
    return out;
}

// --- tinyC print helpers -------------------------------------------------

void printI64(int64_t v) { char b[32]; uint32_t n = fmt_i64(v, b); wasm_write(b, n); }
void printNewline(void)  { wasm_write("\n", 1); }
void printI32(int32_t v) { char b[32]; uint32_t n = fmt_i64((int64_t)v, b); wasm_write(b, n); wasm_write("\n", 1); }
void printF32(float v)   { char b[64]; uint32_t n = fmt_f64((double)v, b); wasm_write(b, n); }
void printF64(double v)  { char b[64]; uint32_t n = fmt_f64(v, b); wasm_write(b, n); }
void printStr(const char *s) {
    if (s) wasm_write(s, wasm_strlen(s));
    wasm_write("\n", 1);
}

// --- va_arg helpers ------------------------------------------------------

int            tinyc_va_arg_i32(va_list *ap) { return va_arg(*ap, int); }
long long      tinyc_va_arg_i64(va_list *ap) { return va_arg(*ap, long long); }
double         tinyc_va_arg_f64(va_list *ap) { return va_arg(*ap, double); }
void          *tinyc_va_arg_ptr(va_list *ap) { return va_arg(*ap, void *); }

void tinyc_va_arg_struct(va_list *ap, void *out, long long size) {
    long long *o = (long long *)out;
    long long words = (size + 7) / 8;
    for (long long i = 0; i < words; i++) {
        o[i] = va_arg(*ap, long long);
    }
}

// --- Minimal printf -------------------------------------------------------
//
// Supports the conversion specifiers actually used by the tinyC test
// suite: %d, %u, %ld, %lu, %lld, %llu, %x, %X, %s, %c, %g, %f
// (plus precision, e.g. %.2f), %%, and %p. Flags / widths beyond
// precision are not implemented.

static uint32_t fmt_u64(uint64_t v, char *buf, int base, int upper) {
    char tmp[32];
    const char *digits_l = "0123456789abcdef";
    const char *digits_u = "0123456789ABCDEF";
    const char *D = upper ? digits_u : digits_l;
    int n = 0;
    if (v == 0) tmp[n++] = '0';
    while (v) { tmp[n++] = D[v % (uint64_t)base]; v /= (uint64_t)base; }
    uint32_t out = 0;
    for (int i = n - 1; i >= 0; i--) buf[out++] = tmp[i];
    return out;
}

// Format a double with fixed precision (digits after decimal point).
static uint32_t fmt_f64_prec(double v, char *buf, int prec) {
    uint32_t out = 0;
    if (v != v) { buf[out++]='n'; buf[out++]='a'; buf[out++]='n'; return out; }
    if (v < 0) { buf[out++] = '-'; v = -v; }
    // Round to `prec` decimals.
    double scale = 1.0;
    for (int i = 0; i < prec; i++) scale *= 10.0;
    double rounded = (double)(long long)(v * scale + 0.5) / scale;
    long long ip = (long long)rounded;
    out += fmt_i64(ip, buf + out);
    if (prec > 0) {
        buf[out++] = '.';
        double frac = rounded - (double)ip;
        if (frac < 0) frac = -frac;
        for (int i = 0; i < prec; i++) {
            frac *= 10.0;
            int d = (int)frac;
            if (d < 0) d = 0; if (d > 9) d = 9;
            buf[out++] = (char)('0' + d);
            frac -= d;
        }
    }
    return out;
}

static int vprintf_impl(const char *fmt, va_list ap) {
    char buf[64];
    int total = 0;
    while (*fmt) {
        if (*fmt != '%') {
            const char *start = fmt;
            while (*fmt && *fmt != '%') fmt++;
            uint32_t n = (uint32_t)(fmt - start);
            wasm_write(start, n);
            total += (int)n;
            continue;
        }
        fmt++; // skip '%'
        // Parse precision: ".N"
        int prec = -1;
        if (*fmt == '.') {
            fmt++;
            prec = 0;
            while (*fmt >= '0' && *fmt <= '9') { prec = prec*10 + (*fmt - '0'); fmt++; }
        }
        // Parse length modifiers: l, ll
        int lcount = 0;
        while (*fmt == 'l') { lcount++; fmt++; }
        char c = *fmt;
        if (c == '\0') break;
        fmt++;
        switch (c) {
        case 'd':
        case 'i': {
            long long v;
            if (lcount >= 2)      v = va_arg(ap, long long);
            else if (lcount == 1) v = (long long)va_arg(ap, long);
            else                  v = (long long)va_arg(ap, int);
            uint32_t n = fmt_i64(v, buf);
            wasm_write(buf, n); total += (int)n;
            break;
        }
        case 'u': {
            uint64_t v;
            if (lcount >= 2)      v = va_arg(ap, unsigned long long);
            else if (lcount == 1) v = (uint64_t)va_arg(ap, unsigned long);
            else                  v = (uint64_t)va_arg(ap, unsigned int);
            uint32_t n = fmt_u64(v, buf, 10, 0);
            wasm_write(buf, n); total += (int)n;
            break;
        }
        case 'x':
        case 'X': {
            uint64_t v;
            if (lcount >= 2)      v = va_arg(ap, unsigned long long);
            else if (lcount == 1) v = (uint64_t)va_arg(ap, unsigned long);
            else                  v = (uint64_t)va_arg(ap, unsigned int);
            uint32_t n = fmt_u64(v, buf, 16, c == 'X');
            wasm_write(buf, n); total += (int)n;
            break;
        }
        case 'p': {
            void *p = va_arg(ap, void *);
            wasm_write("0x", 2); total += 2;
            uint32_t n = fmt_u64((uint64_t)(uintptr_t)p, buf, 16, 0);
            wasm_write(buf, n); total += (int)n;
            break;
        }
        case 'c': {
            int v = va_arg(ap, int);
            char ch = (char)v;
            wasm_write(&ch, 1); total += 1;
            break;
        }
        case 's': {
            const char *s = va_arg(ap, const char *);
            if (!s) s = "(null)";
            uint32_t n = wasm_strlen(s);
            wasm_write(s, n); total += (int)n;
            break;
        }
        case 'f': {
            double v = va_arg(ap, double);
            int p = (prec < 0) ? 6 : prec;
            uint32_t n = fmt_f64_prec(v, buf, p);
            wasm_write(buf, n); total += (int)n;
            break;
        }
        case 'g':
        case 'G': {
            double v = va_arg(ap, double);
            uint32_t n = fmt_f64(v, buf);
            wasm_write(buf, n); total += (int)n;
            break;
        }
        case '%':
            wasm_write("%", 1); total += 1;
            break;
        default:
            // Unknown: emit verbatim so the test fails visibly.
            wasm_write("%", 1);
            wasm_write(&c, 1);
            total += 2;
            break;
        }
    }
    return total;
}

int printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vprintf_impl(fmt, ap);
    va_end(ap);
    return n;
}

// --- Minimal libc subset -------------------------------------------------
//
// A handful of tinyC tests call out to libc functions (malloc/free,
// strlen/strcmp/memcmp/memchr). Provide minimal implementations so the
// suite can be linked under wasm32 without a real wasi-libc sysroot.
//
// On wasm32-wasi `unsigned long` and `size_t` are both 32-bit, and
// since tinyC's parser sizes `long` per-target (32-bit when emitting
// wasm) the signatures we expose here match exactly what tinyC-
// compiled tests reference.
#ifndef TINYC_WASM_RUNTIME_NO_LIBC

unsigned long strlen(const char *s) {
    unsigned long n = 0;
    while (s[n]) n++;
    return n;
}

int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

int memcmp(const void *a, const void *b, unsigned long n) {
    const unsigned char *x = (const unsigned char *)a;
    const unsigned char *y = (const unsigned char *)b;
    for (unsigned long i = 0; i < n; i++) {
        if (x[i] != y[i]) return (int)x[i] - (int)y[i];
    }
    return 0;
}

void *memchr(const void *s, int c, unsigned long n) {
    const unsigned char *p = (const unsigned char *)s;
    unsigned char target = (unsigned char)c;
    for (unsigned long i = 0; i < n; i++) {
        if (p[i] == target) return (void *)(p + i);
    }
    return (void *)0;
}

// Bump allocator backed by a static arena. Free is a no-op. Sufficient
// for the tinyC test suite (peak allocations are tiny).
#define WASM_HEAP_BYTES (4u * 1024u * 1024u)
static unsigned char wasm_heap[WASM_HEAP_BYTES];
static unsigned long wasm_heap_off = 0;

void *malloc(unsigned long size) {
    if (size == 0) return (void *)0;
    unsigned long off = (wasm_heap_off + 15u) & ~(unsigned long)15u;
    if (off + size > WASM_HEAP_BYTES) return (void *)0;
    wasm_heap_off = off + size;
    return &wasm_heap[off];
}

void *calloc(unsigned long n, unsigned long size) {
    unsigned long total = n * size;
    void *p = malloc(total);
    if (!p) return p;
    unsigned char *b = (unsigned char *)p;
    for (unsigned long i = 0; i < total; i++) b[i] = 0;
    return p;
}

void free(void *p) { (void)p; }
#endif

// Note: `_start` (the WASI entry point) is provided by start_wasm.s,
// which calls `main` directly to bypass clang's wasm32-wasi rewrite of
// `extern int main(void)` references into `__original_main`.
