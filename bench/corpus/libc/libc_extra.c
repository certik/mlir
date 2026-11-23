/* Minimal implementations of libc functions missing from corec-stdlib.
 * Include this AFTER corec-stdlib's stdio.c in the unity TU (it uses the
 * private `struct FILE` fields fd/eof/error). Correctness goal: enough for
 * benchmark programs, not a full libc (strtod and the math functions are
 * accurate to a few ulp at best, not correctly rounded). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libc_extra.h"
#include "errno.h"
#include "math.h"
#include "locale.h"
#include "time.h"
#include "signal.h"
#include "setjmp.h"

int errno;

/* ---- stdio ---- */
#define LX_NPUSH 32
static FILE *lx_push_file[LX_NPUSH];
static int lx_push_char[LX_NPUSH];

int fgetc(FILE *f) {
    for (int i = 0; i < LX_NPUSH; i++) {
        if (lx_push_file[i] == f) {
            lx_push_file[i] = NULL;
            return lx_push_char[i];
        }
    }
    unsigned char c;
    if (fread(&c, 1, 1, f) != 1) return EOF;
    return (int)c;
}
int getc(FILE *f) { return fgetc(f); }
int ungetc(int c, FILE *f) {
    if (c == EOF) return EOF;
    for (int i = 0; i < LX_NPUSH; i++) {
        if (lx_push_file[i] == NULL) {
            lx_push_file[i] = f;
            lx_push_char[i] = (unsigned char)c;
            f->eof = 0;
            return c;
        }
    }
    return EOF;
}
char *fgets(char *s, int n, FILE *f) {
    int i = 0;
    while (i < n - 1) {
        int c = fgetc(f);
        if (c == EOF) break;
        s[i] = (char)c;
        i = i + 1;
        if (c == '\n') break;
    }
    if (i == 0) return NULL;
    s[i] = 0;
    return s;
}
int feof(FILE *f) { return f->eof; }
int ferror(FILE *f) { return f->error; }
void clearerr(FILE *f) { f->eof = 0; f->error = 0; }
int fflush(FILE *f) { (void)f; return 0; }
FILE *freopen(const char *path, const char *mode, FILE *f) {
    (void)path; (void)mode; (void)f;
    return NULL;
}
int putchar(int c) { return fputc(c, stdout); }
int puts(const char *s) {
    if (fputs(s, stdout) == EOF) return EOF;
    return fputc('\n', stdout);
}

/* ---- stdlib ---- */
int abs(int x) { return x < 0 ? -x : x; }

static int lx_digit(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return 99;
}

unsigned long strtoul(const char *s, char **end, int base) {
    const char *p = s;
    while (*p == ' ' || (*p >= 9 && *p <= 13)) p++;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = *p == '-'; p++; }
    if ((base == 0 || base == 16) && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        p += 2; base = 16;
    } else if (base == 0 && p[0] == '0') {
        base = 8;
    } else if (base == 0) {
        base = 10;
    }
    unsigned long v = 0;
    const char *start = p;
    while (lx_digit(*p) < base) { v = v * (unsigned long)base + (unsigned long)lx_digit(*p); p++; }
    if (end) *end = (char *)(p == start ? s : p);
    return neg ? (unsigned long)0 - v : v;
}
long strtol(const char *s, char **end, int base) { return (long)strtoul(s, end, base); }

static double lx_pow10(int e) {
    double r = 1.0, b = 10.0;
    int n = e < 0 ? -e : e;
    while (n) { if (n & 1) r = r * b; b = b * b; n = n >> 1; }
    return e < 0 ? 1.0 / r : r;
}

double strtod(const char *s, char **end) {
    const char *p = s;
    while (*p == ' ' || (*p >= 9 && *p <= 13)) p++;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = *p == '-'; p++; }
    if ((p[0] == 'i' || p[0] == 'I') && (p[1] == 'n' || p[1] == 'N') && (p[2] == 'f' || p[2] == 'F')) {
        p += 3;
        if (end) *end = (char *)p;
        return neg ? -HUGE_VAL : HUGE_VAL;
    }
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        /* hex float: mantissa digits, optional '.', optional p exponent */
        p += 2;
        double v = 0.0; int any = 0, exp2 = 0;
        while (lx_digit(*p) < 16) { v = v * 16.0 + lx_digit(*p); p++; any = 1; }
        if (*p == '.') { p++; while (lx_digit(*p) < 16) { v = v * 16.0 + lx_digit(*p); exp2 -= 4; p++; any = 1; } }
        if (!any) { if (end) *end = (char *)s; return 0.0; }
        if (*p == 'p' || *p == 'P') {
            const char *q = p + 1; int en = 0, eneg = 0;
            if (*q == '+' || *q == '-') { eneg = *q == '-'; q++; }
            if (*q >= '0' && *q <= '9') {
                while (*q >= '0' && *q <= '9') { en = en * 10 + (*q - '0'); q++; }
                exp2 += eneg ? -en : en; p = q;
            }
        }
        if (end) *end = (char *)p;
        v = ldexp(v, exp2);
        return neg ? -v : v;
    }
    unsigned long long m = 0; int ndig = 0, dexp = 0, any = 0;
    while (*p >= '0' && *p <= '9') {
        if (ndig < 19) { m = m * 10 + (unsigned long long)(*p - '0'); if (m) ndig++; } else dexp++;
        p++; any = 1;
    }
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') {
            if (ndig < 19) { m = m * 10 + (unsigned long long)(*p - '0'); if (m) ndig++; dexp--; }
            p++; any = 1;
        }
    }
    if (!any) { if (end) *end = (char *)s; return 0.0; }
    if (*p == 'e' || *p == 'E') {
        const char *q = p + 1; int en = 0, eneg = 0;
        if (*q == '+' || *q == '-') { eneg = *q == '-'; q++; }
        if (*q >= '0' && *q <= '9') {
            while (*q >= '0' && *q <= '9') { if (en < 10000) en = en * 10 + (*q - '0'); q++; }
            dexp += eneg ? -en : en; p = q;
        }
    }
    if (end) *end = (char *)p;
    double v = (double)m;
    if (dexp < -300) { v = v * lx_pow10(-300); dexp += 300; }
    v = dexp < 0 ? v / lx_pow10(-dexp) : v * lx_pow10(dexp);
    return neg ? -v : v;
}

/* ---- string ---- */
char *strerror(int e) { (void)e; return (char *)"error"; }
int strcoll(const char *a, const char *b) { return strcmp(a, b); }
size_t strspn(const char *s, const char *accept) {
    size_t n = 0;
    while (s[n] && strchr(accept, s[n])) n++;
    return n;
}
char *strpbrk(const char *s, const char *accept) {
    for (; *s; s++) if (strchr(accept, *s)) return (char *)s;
    return NULL;
}
char *strcat(char *d, const char *s) { strcpy(d + strlen(d), s); return d; }
char *strncat(char *d, const char *s, size_t n) {
    char *p = d + strlen(d);
    while (n && *s) { *p++ = *s++; n--; }
    *p = 0;
    return d;
}

/* ---- locale / time / signal / setjmp ---- */
static char lx_dot[2] = ".";
static struct lconv lx_lconv;
struct lconv *localeconv(void) { lx_lconv.decimal_point = lx_dot; return &lx_lconv; }
char *setlocale(int category, const char *locale) { (void)category; (void)locale; return (char *)"C"; }
time_t time(time_t *t) { if (t) *t = 0; return 0; }
clock_t clock(void) { return 0; }
sighandler_t signal(int sig, sighandler_t handler) { (void)sig; (void)handler; return SIG_DFL; }
void longjmp(jmp_buf env, int val) {
    (void)env; (void)val;
    fputs("fatal: longjmp is not supported (error raised)\n", stderr);
    exit(70);
}

/* ---- math ---- */
static unsigned long long lx_bits(double x) { unsigned long long u; memcpy(&u, &x, 8); return u; }
static double lx_from_bits(unsigned long long u) { double x; memcpy(&x, &u, 8); return x; }

double fabs(double x) { return x < 0 ? -x : x; }
double sqrt(double x) { return __builtin_sqrt(x); }
double floor(double x) {
    if (!(x > -4503599627370496.0 && x < 4503599627370496.0)) return x; /* |x|>=2^52, inf, nan */
    double t = (double)(long long)x;
    return t > x ? t - 1.0 : t;
}
double ceil(double x) { return -floor(-x); }
double fmod(double x, double y) {
    if (y == 0.0 || x != x || y != y) return 0.0 / 0.0;
    double ay = fabs(y), ax = fabs(x);
    if (ax < ay) return x;
    double r = ax;
    while (r >= ay) {         /* exact long division by scaled y */
        double d = ay;
        while (d * 2.0 <= r) d = d * 2.0;
        r = r - d;
    }
    return x < 0 ? -r : r;
}
double frexp(double x, int *e) {
    unsigned long long u = lx_bits(x);
    int ex = (int)((u >> 52) & 0x7ff);
    if (ex == 0) {
        if (x == 0.0) { *e = 0; return x; }
        x = x * 18014398509481984.0; /* 2^54 */
        u = lx_bits(x);
        ex = (int)((u >> 52) & 0x7ff) - 54;
    } else if (ex == 0x7ff) { *e = 0; return x; }
    *e = ex - 1022;
    u = (u & 0x800fffffffffffffULL) | 0x3fe0000000000000ULL;
    return lx_from_bits(u);
}
double ldexp(double x, int e) {
    while (e > 1000) { x = x * lx_from_bits(0x7e70000000000000ULL); e -= 1000; } /* 2^1000 */
    while (e < -1000) { x = x * lx_from_bits(0x0170000000000000ULL); e += 1000; } /* 2^-1000 */
    return x * lx_from_bits((unsigned long long)(e + 1023) << 52);
}
#define LX_LN2 0.693147180559945309417232121458
double exp(double x) {
    if (x != x) return x;
    if (x > 709.8) return HUGE_VAL;
    if (x < -745.2) return 0.0;
    double kf = floor(x / LX_LN2 + 0.5);
    double r = x - kf * LX_LN2;
    double term = 1.0, sum = 1.0;
    for (int i = 1; i < 30; i++) { term = term * r / i; sum = sum + term; }
    return ldexp(sum, (int)kf);
}
double log(double x) {
    if (x < 0 || x != x) return 0.0 / 0.0;
    if (x == 0) return -HUGE_VAL;
    if (x > 1.7976931348623157e308) return x;
    int e;
    double m = frexp(x, &e);         /* x = m * 2^e, m in [0.5, 1) */
    if (m < 0.70710678118654752) { m = m * 2.0; e = e - 1; }
    double s = (m - 1.0) / (m + 1.0), s2 = s * s, term = s, sum = 0.0;
    for (int i = 1; i < 60; i += 2) { sum = sum + term / i; term = term * s2; }
    return 2.0 * sum + e * LX_LN2;
}
double log2(double x) { return log(x) / LX_LN2; }
double log10(double x) { return log(x) / 2.30258509299404568401799145468; }
double pow(double x, double y) {
    if (y == 0.0) return 1.0;
    if (y == floor(y) && fabs(y) < 9007199254740992.0) {
        unsigned long long n = (unsigned long long)fabs(y);
        double r = 1.0, b = x;
        while (n) { if (n & 1) r = r * b; b = b * b; n = n >> 1; }
        return y < 0 ? 1.0 / r : r;
    }
    if (x < 0) return 0.0 / 0.0;
    return exp(y * log(x));
}
#define LX_PI 3.14159265358979323846264338328
double sin(double x) {
    if (x != x || fabs(x) > 1e15) return 0.0 / 0.0;
    double k = floor(x / (2 * LX_PI) + 0.5);
    x = x - k * 2 * LX_PI;            /* now in [-pi, pi] */
    double term = x, sum = x;
    for (int i = 1; i < 30; i++) { term = -term * x * x / ((2 * i) * (2 * i + 1)); sum = sum + term; }
    return sum;
}
double cos(double x) { return sin(x + LX_PI / 2); }
double tan(double x) { return sin(x) / cos(x); }
double atan(double x) {
    if (x != x) return x;
    int neg = x < 0; if (neg) x = -x;
    int inv = x > 1.0; if (inv) x = 1.0 / x;
    /* reduce further: atan(x) = 2*atan(x / (1 + sqrt(1 + x^2))) */
    double y = x / (1.0 + sqrt(1.0 + x * x));
    double y2 = y * y, term = y, sum = 0.0;
    for (int i = 1; i < 80; i += 2) { sum = sum + term / i; term = -term * y2; }
    double r = 2.0 * sum;
    if (inv) r = LX_PI / 2 - r;
    return neg ? -r : r;
}
double atan2(double y, double x) {
    if (x > 0) return atan(y / x);
    if (x < 0) return y >= 0 ? atan(y / x) + LX_PI : atan(y / x) - LX_PI;
    if (y > 0) return LX_PI / 2;
    if (y < 0) return -LX_PI / 2;
    return 0.0;
}
double asin(double x) { return atan2(x, sqrt(1.0 - x * x)); }
double acos(double x) { return atan2(sqrt(1.0 - x * x), x); }
double sinh(double x) { return (exp(x) - exp(-x)) / 2; }
double cosh(double x) { return (exp(x) + exp(-x)) / 2; }
double tanh(double x) { double a = exp(x), b = exp(-x); return (a - b) / (a + b); }
