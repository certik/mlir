#pragma once
struct lconv { char *decimal_point; };
struct lconv *localeconv(void);
char *setlocale(int category, const char *locale);
#define LC_ALL 6
#define LC_NUMERIC 1
