#include "bench_common.h"

static bool copy_cstr(char *buf, size_t bufsz, const char *s) {
    if (!s) return false;
    size_t n = strlen(s);
    if (n >= bufsz) return false;
    memcpy(buf, s, n + 1);
    return true;
}

static void with_argv(void (*fn)(size_t, char **, void *), void *ctx) {
    size_t pargc = 0, argv_buf_size = 0;
    if (platform_args_sizes_get(&pargc, &argv_buf_size) != 0 || pargc == 0) {
        fn(0, NULL, ctx);
        return;
    }
    Arena *arena = arena_create(4096);
    char **argv = arena_new_array(arena, char *, pargc + 1);
    char *argv_buf = arena_new_array(arena, char, argv_buf_size + 1);
    platform_args_get(argv, argv_buf);
    fn(pargc, argv, ctx);
    arena_destroy(arena);
}

typedef struct { const char *flag; char *buf; size_t bufsz; bool found; } FlagCtx;
static void find_flag(size_t pargc, char **argv, void *ctx) {
    FlagCtx *c = ctx;
    for (size_t i = 1; i + 1 < pargc; i++) {
        if (!strcmp(argv[i], c->flag)) {
            c->found = copy_cstr(c->buf, c->bufsz, argv[i + 1]);
            return;
        }
    }
}

typedef struct { char *buf; size_t bufsz; bool found; } PosCtx;
static void find_pos(size_t pargc, char **argv, void *ctx) {
    PosCtx *c = ctx;
    for (size_t i = 1; i < pargc; i++) {
        if (argv[i][0] != '-') {
            c->found = copy_cstr(c->buf, c->bufsz, argv[i]);
            return;
        }
    }
}

bool bench_argv_flag(const char *flag, char *buf, size_t bufsz) {
    FlagCtx ctx = {flag, buf, bufsz, false};
    with_argv(find_flag, &ctx);
    return ctx.found;
}

bool bench_argv_pos(char *buf, size_t bufsz) {
    PosCtx ctx = {buf, bufsz, false};
    with_argv(find_pos, &ctx);
    return ctx.found;
}
