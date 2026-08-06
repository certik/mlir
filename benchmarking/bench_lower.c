#include "bench_lower.h"

#include <base/strbuf.h>

#include "mlir_lift_cf_to_scf.h"
#include "mlir_llvm_to_wasmssa.h"
#include "mlir_wasmssa_to_wasmstack.h"
#include "mlir_wasmstack_to_bin.h"
#if BENCH_LOWER_NATIVE_X64
#include "mlir_llvm_to_x64.h"
#elif BENCH_LOWER_NATIVE_AARCH64
#include "mlir_llvm_to_aarch64.h"
#include "mlir_aarch64_to_macho.h"
#endif

extern string read_file_ok(Arena *arena, string path);

#define BENCH_TIME(out_ms, expr) do { \
    double _t0 = bench_now_ms(); \
    expr; \
    out_ms = bench_now_ms() - _t0; \
} while (0)

static void print_step(const char *name, double ms) {
    printf("  %-28s %10.3f ms\n", name, ms);
}

static bool write_bytes_file(const char *path, const uint8_t *data, size_t size) {
    if (!path || !data || size == 0) return false;
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(data, 1, size, f) == size;
    fclose(f);
    return ok;
}

static int write_text_file(const char *path, string text) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    if (text.size && fwrite(text.str, 1, text.size, f) != text.size) {
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

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

static bool argv_flag(const char *flag, char *buf, size_t bufsz) {
    FlagCtx ctx = {flag, buf, bufsz, false};
    with_argv(find_flag, &ctx);
    return ctx.found;
}

static bool argv_pos(char *buf, size_t bufsz) {
    PosCtx ctx = {buf, bufsz, false};
    with_argv(find_pos, &ctx);
    return ctx.found;
}

static void append_tc_func(Arena *arena, strbuf *buf, size_t idx) {
    size_t num = idx + 1, hi = idx + 10;
    strbuf_append_cstr(arena, buf, "void g");
    strbuf_append(arena, buf, uint_to_string(arena, num));
    strbuf_append_cstr(arena, buf, "(int *x) {\n"
        "    int i;\n    *x = 0;\n    for (i = ");
    strbuf_append(arena, buf, uint_to_string(arena, num));
    strbuf_append_cstr(arena, buf, "; i <= ");
    strbuf_append(arena, buf, uint_to_string(arena, hi));
    strbuf_append_cstr(arena, buf,
        "; i = i + 1) {\n        *x = *x + i;\n    }\n}\n");
}

static void append_clang_func(Arena *arena, strbuf *buf, size_t idx) {
    size_t num = idx + 1, hi = idx + 10;
    strbuf_append_cstr(arena, buf, "void g");
    strbuf_append(arena, buf, uint_to_string(arena, num));
    strbuf_append_cstr(arena, buf, "(Arena *a, int *x) {\n"
        "    int *i = arena_new(a, int);\n    *x = 0;\n    for (*i = ");
    strbuf_append(arena, buf, uint_to_string(arena, num));
    strbuf_append_cstr(arena, buf, "; *i <= ");
    strbuf_append(arena, buf, uint_to_string(arena, hi));
    strbuf_append_cstr(arena, buf,
        "; *i = *i + 1) {\n        *x = *x + *i;\n    }\n}\n");
}

string bench_lower_construct_tc(Arena *arena, size_t n_funcs) {
    strbuf buf = strbuf_make_cap(arena, 2u * 1024u * 1024u);
    for (size_t i = 0; i < n_funcs; i++) append_tc_func(arena, &buf, i);
    strbuf_append_cstr(arena, &buf, "int main() {\n    int c;\n    c = 0;\n");
    for (size_t i = 0; i < n_funcs; i++) {
        strbuf_append_cstr(arena, &buf, "    g");
        strbuf_append(arena, &buf, uint_to_string(arena, i + 1));
        strbuf_append_cstr(arena, &buf, "(&c);\n");
    }
    strbuf_append_cstr(arena, &buf, "    return c;\n}\n");
    return strbuf_to_string(buf);
}

string bench_lower_construct_clang_c(Arena *arena, size_t n_funcs) {
    strbuf buf = strbuf_make_cap(arena, 2u * 1024u * 1024u);
    strbuf_append_cstr(arena, &buf, "#include <base/arena.h>\n\n"
        "extern void platform_init(int argc, char **argv, char **envp);\n\n");
    for (size_t i = 0; i < n_funcs; i++) append_clang_func(arena, &buf, i);
    strbuf_append_cstr(arena, &buf,
        "int main(int argc, char **argv) {\n"
        "    platform_init(argc, argv, NULL);\n"
        "    Arena *a = arena_create(64u * 1024u * 1024u);\n"
        "    int *c = arena_new(a, int);\n    *c = 0;\n");
    for (size_t i = 0; i < n_funcs; i++) {
        strbuf_append_cstr(arena, &buf, "    g");
        strbuf_append(arena, &buf, uint_to_string(arena, i + 1));
        strbuf_append_cstr(arena, &buf, "(a, c);\n");
    }
    strbuf_append_cstr(arena, &buf,
        "    int ret = *c;\n    arena_destroy(a);\n    return ret;\n}\n");
    return strbuf_to_string(buf);
}

bool bench_lower_load_tc(Arena *arena, MLIR_Context *ctx, const char *path,
                         Program **out_prog, MLIR_OpHandle *out_mod) {
    double ignored = 0;
    return bench_lower_run_frontend(arena, ctx, path, out_prog, out_mod, &ignored);
}

bool bench_lower_run_frontend(Arena *arena, MLIR_Context *ctx, const char *path,
                              Program **out_prog, MLIR_OpHandle *out_mod,
                              double *out_ms) {
    double ms = 0;
    BENCH_TIME(ms, {
        string src = read_file_ok(arena, str_from_cstr_view((char *)path));
        if (src.size && src.str[src.size - 1] == '\0') src.size--;

        Program *prog = arena_new(arena, Program);
        *prog = (Program){0};
        tinyc_parse_into(arena, prog, tinyc_lex(arena, src), false);

        MLIR_SetArenaAllocator(ctx, arena);
        *out_prog = prog;
        *out_mod = tinyc_emit_module(ctx, prog);
    });
    *out_ms = ms;
    return true;
}

bool bench_lower_emit_exec_native(Arena *arena, const char *tc_path,
                                  const char *out_path) {
#if !BENCH_LOWER_HAS_NATIVE
    (void)arena; (void)tc_path; (void)out_path;
    return false;
#else
    MLIR_Context ctx = {0};
    Program *prog = NULL;
    MLIR_OpHandle module = MLIR_INVALID_HANDLE;
    if (!bench_lower_load_tc(arena, &ctx, tc_path, &prog, &module)) return false;

    MLIR_SetArenaAllocator(&ctx, arena);
    module = tinyc_emit_module(&ctx, prog);
    if (!MLIR_LowerToLLVMDialect(&ctx, module)) return false;

    uint8_t *data = NULL;
    size_t size = 0;
    bool ok = false;
#if BENCH_LOWER_NATIVE_X64
    ok = mlir_llvm_to_elf(&ctx, module, &data, &size);
#elif BENCH_LOWER_NATIVE_AARCH64
    ok = mlir_llvm_to_macho(&ctx, module, &data, &size);
#endif
    if (!ok) {
        free(data);
        return false;
    }
    ok = write_bytes_file(out_path, data, size);
    free(data);
    return ok;
#endif
}

bool bench_lower_run_wasm(MLIR_Context *ctx, MLIR_OpHandle module,
                          BenchLowerWasmTimings *out) {
    out->cf_to_scf_ms = 0;
    BENCH_TIME(out->to_llvm_keep_scf_ms, MLIR_LowerToLLVMDialectForWasm(ctx, module));

    MLIR_OpHandle wasmssa;
    BENCH_TIME(out->llvm_to_wasmssa_ms, wasmssa = mlir_llvm_to_wasmssa(ctx, module));

    MLIR_OpHandle wasmstack;
    BENCH_TIME(out->wasmssa_to_wasmstack_ms,
               wasmstack = mlir_wasmssa_to_wasmstack(ctx, wasmssa));

    string wasm_obj;
    BENCH_TIME(out->wasmstack_to_wasm_ms, wasm_obj = mlir_wasmstack_to_bin(ctx, wasmstack));
    out->wasm_bytes = (size_t)wasm_obj.size;
    return true;
}

bool bench_lower_run_native(MLIR_Context *ctx, Arena *arena, Program *prog,
                            BenchLowerNativeTimings *out) {
#if !BENCH_LOWER_HAS_NATIVE
    (void)ctx; (void)arena; (void)prog; (void)out;
    return true;
#else
    MLIR_SetArenaAllocator(ctx, arena);
    MLIR_OpHandle module = tinyc_emit_module(ctx, prog);

    BENCH_TIME(out->scf_cf_to_llvm_ms, MLIR_LowerToLLVMDialect(ctx, module));

#if BENCH_LOWER_NATIVE_X64
    uint8_t *elf_data = NULL;
    size_t elf_size = 0;
    BENCH_TIME(out->llvm_to_elf_ms,
               out->binary_ok = mlir_llvm_to_elf(ctx, module, &elf_data, &elf_size));
    out->binary_bytes = out->binary_ok ? elf_size : 0;
    const char *out_path = getenv("BENCH_NATIVE_OUT");
    if (out_path && out->binary_ok && elf_data && elf_size > 0)
        write_bytes_file(out_path, elf_data, elf_size);
    free(elf_data);
#elif BENCH_LOWER_NATIVE_AARCH64
    MLIR_OpHandle a64 = MLIR_INVALID_HANDLE;
    BENCH_TIME(out->llvm_to_aarch64_ms, a64 = mlir_llvm_to_aarch64(ctx, module));
    uint8_t *macho_data = NULL;
    size_t macho_size = 0;
    BENCH_TIME(out->aarch64_to_macho_ms,
               out->binary_ok = (a64 != MLIR_INVALID_HANDLE) &&
                                mlir_aarch64_to_macho(ctx, a64, &macho_data, &macho_size));
    out->binary_bytes = out->binary_ok ? macho_size : 0;
    const char *out_path = getenv("BENCH_NATIVE_OUT");
    if (out_path && out->binary_ok && macho_data && macho_size > 0)
        write_bytes_file(out_path, macho_data, macho_size);
    free(macho_data);
#endif
    return true;
#endif
}

void bench_lower_print_wasm(const BenchLowerWasmTimings *t) {
    print_step("cf/scf/arith -> llvm (wasm)", t->to_llvm_keep_scf_ms);
    print_step("llvm -> wasmssa", t->llvm_to_wasmssa_ms);
    print_step("wasmssa -> wasmstack", t->wasmssa_to_wasmstack_ms);
    print_step("wasmstack -> wasm", t->wasmstack_to_wasm_ms);
    printf("  wasm object size (bytes): %zu\n", t->wasm_bytes);
}

void bench_lower_print_native(const BenchLowerNativeTimings *t) {
#if BENCH_LOWER_HAS_NATIVE
    print_step("scf/cf -> llvm", t->scf_cf_to_llvm_ms);
#if BENCH_LOWER_NATIVE_X64
    print_step("llvm -> x86 (elf)", t->llvm_to_elf_ms);
#elif BENCH_LOWER_NATIVE_AARCH64
    print_step("llvm -> aarch64", t->llvm_to_aarch64_ms);
    print_step("aarch64 -> macho", t->aarch64_to_macho_ms);
#endif
    if (t->binary_ok) printf("  native binary size (bytes): %zu\n", t->binary_bytes);
#else
    (void)t;
#endif
}

bool bench_lower_bench_file(const char *tc_path) {
    printf("Lowering benchmark (file: %s)\n\n", tc_path);

    Arena *arena = arena_create(64 * 1024 * 1024);
    MLIR_Context ctx = {0};
    Program *prog = NULL;
    MLIR_OpHandle module = MLIR_INVALID_HANDLE;
    double frontend_ms = 0;
    if (!bench_lower_run_frontend(arena, &ctx, tc_path, &prog, &module, &frontend_ms)) {
        arena_destroy(arena);
        return false;
    }

    printf("Frontend:\n");
    print_step("frontend (lex/parse/emit)", frontend_ms);

    printf("\nWasm pipeline:\n");
    BenchLowerWasmTimings wasm = {0};
    if (!bench_lower_run_wasm(&ctx, module, &wasm)) {
        arena_destroy(arena);
        return false;
    }
    bench_lower_print_wasm(&wasm);

#if BENCH_LOWER_HAS_NATIVE
#if BENCH_LOWER_NATIVE_X64
    printf("\nNative pipeline (x86_64 ELF, fresh module):\n");
#elif BENCH_LOWER_NATIVE_AARCH64
    printf("\nNative pipeline (aarch64 Mach-O, fresh module):\n");
#endif
    Arena *arena2 = arena_create(64 * 1024 * 1024);
    MLIR_Context ctx2 = {0};
    BenchLowerNativeTimings native = {0};
    if (!bench_lower_run_native(&ctx2, arena2, prog, &native)) {
        arena_destroy(arena2);
        arena_destroy(arena);
        return false;
    }
    bench_lower_print_native(&native);
    arena_destroy(arena2);
#else
    printf("\nNative pipeline: skipped (linux/x86_64 or darwin/arm64 only)\n");
#endif

    arena_destroy(arena);
    return true;
}

#if !defined(BENCH_LOWER_NO_MAIN)
int app_main(void) {
    char path[4096];
    if (argv_flag("--emit-tc", path, sizeof path)) {
        Arena *arena = arena_create(64 * 1024 * 1024);
        size_t n = bench_n();
        string tc = bench_lower_construct_tc(arena, n);
        if (write_text_file(path, tc) != 0) return 1;
        printf("Wrote %zu bytes to %s (N=%zu)\n", (size_t)tc.size, path, n);
        arena_destroy(arena);
        return 0;
    }

    if (argv_flag("--emit-c", path, sizeof path)) {
        Arena *arena = arena_create(64 * 1024 * 1024);
        size_t n = bench_n();
        string src = bench_lower_construct_clang_c(arena, n);
        if (write_text_file(path, src) != 0) return 1;
        printf("Wrote %zu bytes to %s (N=%zu, corec arena workload)\n",
               (size_t)src.size, path, n);
        arena_destroy(arena);
        return 0;
    }

    if (argv_flag("--emit-exec-native", path, sizeof path) ||
        argv_flag("--emit-exec-elf", path, sizeof path)) {
        const char *tc_path = getenv("BENCH_TC");
        char tc_buf[4096];
        if (!tc_path || !tc_path[0]) {
            if (!argv_pos(tc_buf, sizeof tc_buf)) tc_path = "benchmarking/build/compare/bench_lower.tc";
            else tc_path = tc_buf;
        }
        Arena *arena = arena_create(64 * 1024 * 1024);
        bool ok = bench_lower_emit_exec_native(arena, tc_path, path);
        arena_destroy(arena);
        return ok ? 0 : 1;
    }

    const char *tc_path = getenv("BENCH_TC");
    if (!tc_path || !tc_path[0]) {
        if (!argv_pos(path, sizeof path)) tc_path = "benchmarking/build/compare/bench_lower.tc";
        else tc_path = path;
    }
    return bench_lower_bench_file(tc_path) ? 0 : 1;
}
#endif
