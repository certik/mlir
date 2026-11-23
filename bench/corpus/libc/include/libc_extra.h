/* Declarations of libc functions that corec-stdlib's headers lack.
 * Include it (once) after the standard headers in a unity TU; the
 * definitions are in libc_extra.c. */
#pragma once
#include <stddef.h>
#include <stdio.h>
#define BUFSIZ 1024

/* <stdint.h> gaps */
typedef long intmax_t;
typedef unsigned long uintmax_t;
#ifndef INT32_MAX
#define INT32_MAX 2147483647
#endif
#ifndef UINT32_MAX
#define UINT32_MAX 4294967295u
#endif
#ifndef INT64_MAX
#define INT64_MAX 9223372036854775807L
#endif
#ifndef UINT64_MAX
#define UINT64_MAX 18446744073709551615UL
#endif
#ifndef SIZE_MAX
#define SIZE_MAX 18446744073709551615UL
#endif

/* <float.h> gaps (IEEE-754 binary32/binary64) */
#define FLT_RADIX 2
#define FLT_MANT_DIG 24
#define FLT_DIG 6
#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38F
#endif
#define FLT_MIN 1.17549435e-38F
#define FLT_EPSILON 1.19209290e-7F
#define DBL_MANT_DIG 53
#define DBL_DIG 15
#define DBL_MAX_EXP 1024
#define DBL_MIN_EXP (-1021)
#define DBL_MAX_10_EXP 308
#define DBL_MAX 1.7976931348623157e+308
#define DBL_MIN 2.2250738585072014e-308
#define DBL_EPSILON 2.2204460492503131e-16
int getc(FILE *f);
int fgetc(FILE *f);
int ungetc(int c, FILE *f);
char *fgets(char *s, int n, FILE *f);
int feof(FILE *f);
int ferror(FILE *f);
void clearerr(FILE *f);
int fflush(FILE *f);
FILE *freopen(const char *path, const char *mode, FILE *f);
int puts(const char *s);
int putchar(int c);
double strtod(const char *s, char **end);
long strtol(const char *s, char **end, int base);
unsigned long strtoul(const char *s, char **end, int base);
int abs(int x);
char *strerror(int e);
int strcoll(const char *a, const char *b);
char *strpbrk(const char *s, const char *accept);
size_t strspn(const char *s, const char *accept);
char *strcat(char *d, const char *s);
char *strncat(char *d, const char *s, size_t n);
