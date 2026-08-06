#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <platform/platform.h>
#include <base/arena.h>
#include <base/string.h>

static inline size_t bench_n(void) {
    const char *e = getenv("BENCH_N");
    if (!e || !e[0]) return 10000;
    unsigned long v = strtoul(e, NULL, 10);
    return v ? (size_t)v : 10000;
}

static inline int bench_write_file(const char *path, string text) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    if (text.size && fwrite(text.str, 1, text.size, f) != text.size) {
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

// Copy argv value for `flag`, or first non-option positional arg, into buf.
bool bench_argv_flag(const char *flag, char *buf, size_t bufsz);
bool bench_argv_pos(char *buf, size_t bufsz);
