// tinyC driver — read .tc file, lex/parse/emit MLIR, print to stdout.

#include <stdio.h>
#include <string.h>

#include <base/arena.h>
#include <base/io.h>
#include <base/string.h>
#include <platform/platform.h>

#include "mlir_api.h"
#include "tinyc.h"

extern string read_file_ok(Arena *arena, string path);

int app_main(void) {
    size_t pargc = 0, argv_buf_size = 0;
    int rc = platform_args_sizes_get(&pargc, &argv_buf_size);
    if (rc != 0) pargc = 0;
    int argc = (int)pargc;
    Arena *boot_arena = arena_create(64 * 1024);
    char **argv = arena_new_array(boot_arena, char *, argc + 1);
    char *argv_buf = arena_new_array(boot_arena, char, argv_buf_size + 1);
    if (argc > 0) platform_args_get(argv, argv_buf);
    argv[argc] = NULL;

    typedef string (*PrintFn)(MLIR_Context *, MLIR_OpHandle);
    PrintFn print_fn = MLIR_PrintOperationUpstream;
    char *input_file = NULL;
    bool emit_llvm = false;
    bool emit_lowered = false;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--emit=mlir")    == 0) { emit_llvm = false; emit_lowered = false; }
        else if (strcmp(argv[i], "--emit=lowered") == 0) { emit_lowered = true;  emit_llvm = false; }
        else if (strcmp(argv[i], "--emit=llvm")    == 0) { emit_llvm = true;     emit_lowered = false; }
        else if (argv[i][0] != '-') input_file = argv[i];
    }

    if (!input_file) {
        println(str_lit("usage: tinyc [--emit=mlir|lowered|llvm] FILE.tc"));
        arena_destroy(boot_arena);
        return 1;
    }

    Arena *arena = arena_create(64 * 1024 * 1024);
    MLIR_Context ctx = {0};
    MLIR_SetArenaAllocator(&ctx, arena);

    string src = read_file_ok(arena, str_from_cstr_view(input_file));
    if (src.size > 0 && src.str[src.size - 1] == '\0') src.size -= 1;
    VecTcTok toks = tinyc_lex(arena, src);
    Program *prog = tinyc_parse(arena, toks);
    MLIR_OpHandle module = tinyc_emit_module(&ctx, prog);

    if (emit_lowered || emit_llvm) {
        if (!MLIR_LowerToLLVMDialect(&ctx, module)) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
    }
    if (emit_llvm) {
        string ll = MLIR_TranslateModuleToLLVMIR(&ctx, module);
        println(str_lit("{}"), ll);
    } else {
        println(str_lit("{}"), print_fn(&ctx, module));
    }

    arena_destroy(arena);
    arena_destroy(boot_arena);
    return 0;
}
