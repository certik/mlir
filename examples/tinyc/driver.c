// tinyC driver — read one or more .tc files, lex/parse/emit a single
// MLIR module, print to stdout (or to -o <path>).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/io.h>
#include <base/string.h>
#include <platform/platform.h>

#include "mlir_api.h"
#include "mlir_llvm_mem2reg.h"
#include "mlir_llvm_to_wasmssa.h"
#include "mlir_wasmssa_to_wasmstack.h"
#include "mlir_wasm_to_wat.h"
#include "mlir_wasm_to_macho.h"
#include "mlir_wasm_to_wasmstack.h"
#include "mlir_wasmstack_to_wasmssa.h"
#include "mlir_wasm_link.h"
#include "mlir_wasmssa_to_llvm.h"
#include "mlir_llvm_to_aarch64.h"
#include "mlir_aarch64_to_macho.h"
#include "tinyc.h"
#include "tinyc_native_runtime.h"

#if !defined(_WIN32) && !defined(__wasm__) && !defined(__TINYC__)
#include <sys/stat.h>
#endif

extern string read_file_ok(Arena *arena, string path);

static int write_string_to_file(string out, const char *path) {
    size_t plen = 0; while (path[plen]) plen++;
    platform_fd_t fd = platform_path_open(path, plen,
        PLATFORM_RIGHTS_WRITE, PLATFORM_O_CREAT | PLATFORM_O_TRUNC);
    if (fd < 0) return 1;
    ciovec_t iovs[2];
    iovs[0].buf = out.str;
    iovs[0].buf_len = out.size;
    iovs[1].buf = "\n";
    iovs[1].buf_len = 1;
    uint32_t werr = write_all(fd, iovs, 2);
    platform_fd_close(fd);
    return werr ? 1 : 0;
}

// Like write_string_to_file but writes the bytes verbatim with no
// trailing newline. Used for binary outputs (e.g. wasm object files).
static int write_bytes_to_file(string out, const char *path) {
    size_t plen = 0; while (path[plen]) plen++;
    platform_fd_t fd = platform_path_open(path, plen,
        PLATFORM_RIGHTS_WRITE, PLATFORM_O_CREAT | PLATFORM_O_TRUNC);
    if (fd < 0) return 1;
    ciovec_t iovs[1];
    iovs[0].buf = out.str;
    iovs[0].buf_len = out.size;
    uint32_t werr = write_all(fd, iovs, 1);
    platform_fd_close(fd);
    return werr ? 1 : 0;
}

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
    // Use the generic printer: works with both backends (native + upstream)
    // and against any op, so `--emit=mlir` and `--emit=lowered` produce
    // identical output regardless of which tinyc binary was built.
    PrintFn print_fn = MLIR_PrintOperationGeneric;
    bool emit_llvm = false;
    bool emit_lowered = false;
    bool emit_wasm = false;
    bool emit_wasmssa = false;
    bool emit_wasmstack = false;
    bool emit_wat = false;
    bool emit_macho = false;
    // --macho-backend selects which pipeline produces the final Mach-O
    // binary. "wasm" (default) keeps the established wasm-link +
    // wasm->macho path. "llvm" routes the `llvm` dialect directly through
    // the unified llvm -> aarch64 -> macho backend.
    bool macho_backend_llvm = false;
    // `--lowering=upstream` switches the four lowering / translation calls
    // below to the *Upstream variants (which run upstream MLIR's pass
    // pipeline / translator / LLVM target machine). Only available in the
    // upstream-backed tinyc binary, which is the only build that defines
    // TINYC_HAS_UPSTREAM and links the *Upstream implementations.
    typedef bool (*LowerFn)(MLIR_Context *, MLIR_OpHandle);
    typedef string (*TranslateFn)(MLIR_Context *, MLIR_OpHandle);
    LowerFn lower_fn;
    LowerFn lower_for_wasm_fn;
    TranslateFn translate_to_llvm_fn;
    TranslateFn translate_to_wasm_fn;
#ifdef TINYC_HAS_UPSTREAM
    bool use_upstream = true;       // default in the upstream build
    bool lowering_explicit = false; // tracks whether user passed --lowering=
#endif
    char *output_file = NULL;

    // Multiple positional input files (multi-file compilation merges them
    // into a single Program / single MLIR module).
    char **input_files = arena_new_array(boot_arena, char *, argc + 1);
    size_t n_input_files = 0;

    // -I include directories (repeatable). Allocated in boot_arena.
    string *include_dirs = arena_new_array(boot_arena, string, argc + 1);
    size_t n_include_dirs = 0;

    // -D define list (repeatable). Each entry is "NAME" or "NAME=BODY".
    string *defines = arena_new_array(boot_arena, string, argc + 1);
    size_t n_defines = 0;

    // --wasm-runtime-obj=PATH (repeatable). Paths of additional .wasm.o
    // objects to link into the module when producing a Mach-O binary.
    // For the macho_exit test, the test runner passes
    // `runtime_wasm.wasm.o` and `start_wasm.wasm.o`.
    char **wasm_runtime_objs = arena_new_array(boot_arena, char *, argc + 1);
    size_t n_wasm_runtime_objs = 0;

    // --from-wasm=PATH: bypass the C/MLIR front end and read a linked
    // wasm32 module directly. This routes the wasm bytes through
    // wasm -> wasmstack -> wasmssa -> ... -> chosen emit target. Used
    // by the llvm-via-wasm Mach-O backend.
    char *from_wasm_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--link") == 0) {
            // Linker mode: collect remaining positional arguments as
            // input .wasm.o files, expect -o <out.wasm>. All other
            // tinyc flags are ignored.
            //
            // If --emit=macho is also present, instead of writing the
            // linked wasm module we translate it to a signed Mach-O
            // ARM64 binary. This lets a tinyC-compiled Mach-O `tinyc`
            // self-host: stage 2 is the binary produced by upstream
            // tinyc; stage 3 is the binary produced by stage 2 from
            // the same inputs. The two are compared byte-for-byte.
            const char *link_out = NULL;
            const char *link_entry = "_start";
            bool link_emit_macho = false;
            // Upper bound on input count: every remaining argv slot.
            size_t link_inputs_cap = (size_t)(argc - (i + 1));
            const char **link_inputs = arena_new_array(
                boot_arena, const char *, link_inputs_cap ? link_inputs_cap : 1);
            size_t n_link_inputs = 0;
            for (int j = i + 1; j < argc; j++) {
                if (strcmp(argv[j], "-o") == 0 && j + 1 < argc) {
                    link_out = argv[++j];
                } else if (strncmp(argv[j], "-o", 2) == 0 && argv[j][2] != '\0') {
                    link_out = argv[j] + 2;
                } else if (strncmp(argv[j], "--export=", 9) == 0) {
                    link_entry = argv[j] + 9;
                } else if (strcmp(argv[j], "--emit=macho") == 0) {
                    link_emit_macho = true;
                } else if (argv[j][0] != '-') {
                    link_inputs[n_link_inputs++] = argv[j];
                }
            }
            if (!link_out || n_link_inputs == 0) {
                fprintf(stderr, "usage: tinyc --link [--emit=macho] [--export=NAME] -o OUT.wasm IN1.wasm.o [IN2.wasm.o ...]\n");
                arena_destroy(boot_arena);
                return 1;
            }
            // Read each input.
            MLIR_WasmLinkInput *ins = (MLIR_WasmLinkInput *)arena_new_array(
                boot_arena, MLIR_WasmLinkInput, n_link_inputs);
            for (size_t k = 0; k < n_link_inputs; k++) {
                string buf = read_file_ok(boot_arena, str_from_cstr_view((char *)link_inputs[k]));
                ins[k].data = (const uint8_t *)buf.str;
                // read_file_ok stores filesize+1 (it appends a NUL).
                // The linker needs the exact file size.
                ins[k].size = buf.size > 0 ? buf.size - 1 : 0;
                ins[k].name = link_inputs[k];
            }
            uint8_t *out_data = NULL; size_t out_size = 0;
            if (!MLIR_WasmLink(ins, n_link_inputs, link_entry, &out_data, &out_size)) {
                fprintf(stderr, "tinyc --link: link failed\n");
                arena_destroy(boot_arena);
                return 1;
            }
            if (link_emit_macho) {
                uint8_t *macho_data = NULL; size_t macho_size = 0;
                bool macho_ok = MLIR_WasmToMachoArm64(out_data, out_size,
                                                     &macho_data, &macho_size);
                free(out_data);
                if (!macho_ok) {
                    fprintf(stderr, "tinyc --link --emit=macho: wasm->macho translation failed\n");
                    arena_destroy(boot_arena);
                    return 1;
                }
                string macho_buf = { .str = (char *)macho_data, .size = macho_size };
                int wrc = write_bytes_to_file(macho_buf, link_out);
                free(macho_data);
#if !defined(_WIN32) && !defined(__wasm__) && !defined(__TINYC__)
                if (wrc == 0) {
                    // chmod 0755 so the resulting Mach-O is directly
                    // runnable without an explicit `chmod +x` step.
                    chmod(link_out, 0755);
                }
#endif
                arena_destroy(boot_arena);
                return wrc;
            }
            string out_buf = { .str = (char *)out_data, .size = out_size };
            int wrc = write_bytes_to_file(out_buf, link_out);
            free(out_data);
            arena_destroy(boot_arena);
            return wrc;
        }
        if      (strcmp(argv[i], "--emit=mlir")    == 0) { emit_llvm = false; emit_lowered = false; emit_wasm = false; emit_wasmssa = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; }
        else if (strcmp(argv[i], "--emit=lowered") == 0) { emit_lowered = true;  emit_llvm = false; emit_wasm = false; emit_wasmssa = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; }
        else if (strcmp(argv[i], "--emit=llvm")    == 0) { emit_llvm = true;     emit_lowered = false; emit_wasm = false; emit_wasmssa = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; }
        else if (strcmp(argv[i], "--emit=wasm")    == 0) { emit_wasm = true;     emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; }
        else if (strcmp(argv[i], "--emit=wasmssa") == 0) { emit_wasmssa = true; emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; }
        else if (strcmp(argv[i], "--emit=wasmstack") == 0) { emit_wasmstack = true; emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wat = false; emit_macho = false; }
        else if (strcmp(argv[i], "--emit=wat")     == 0) { emit_wat = true;     emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wasmstack = false; emit_macho = false; }
        else if (strcmp(argv[i], "--emit=macho")   == 0) { emit_macho = true;   emit_wat = false; emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wasmstack = false; }
        else if (strcmp(argv[i], "--macho-backend=wasm") == 0) { macho_backend_llvm = false; }
        else if (strcmp(argv[i], "--macho-backend=llvm") == 0) { macho_backend_llvm = true; }
        else if (strncmp(argv[i], "--wasm-runtime-obj=", 19) == 0) {
            wasm_runtime_objs[n_wasm_runtime_objs++] = argv[i] + 19;
        }
        else if (strncmp(argv[i], "--from-wasm=", 12) == 0) {
            from_wasm_path = argv[i] + 12;
        }
        else if (strcmp(argv[i], "--from-wasm") == 0 && i + 1 < argc) {
            from_wasm_path = argv[++i];
        }
        else if (strcmp(argv[i], "--lowering=upstream") == 0) {
#ifdef TINYC_HAS_UPSTREAM
            use_upstream = true; lowering_explicit = true;
#else
            fprintf(stderr,
                    "tinyc: --lowering=upstream is not supported in this "
                    "build (rebuild with the upstream backend to use it)\n");
            arena_destroy(boot_arena);
            return 1;
#endif
        }
        else if (strcmp(argv[i], "--lowering=native") == 0) {
#ifdef TINYC_HAS_UPSTREAM
            use_upstream = false; lowering_explicit = true;
#endif
            // In the native-only build this is already the only option;
            // accept it silently.
        }
        else if (strncmp(argv[i], "-I", 2) == 0 && argv[i][2] != '\0') {
            include_dirs[n_include_dirs++] = str_from_cstr_view(argv[i] + 2);
        } else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            include_dirs[n_include_dirs++] = str_from_cstr_view(argv[++i]);
        } else if (strncmp(argv[i], "-D", 2) == 0 && argv[i][2] != '\0') {
            defines[n_defines++] = str_from_cstr_view(argv[i] + 2);
        } else if (strcmp(argv[i], "-D") == 0 && i + 1 < argc) {
            defines[n_defines++] = str_from_cstr_view(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strncmp(argv[i], "-o", 2) == 0 && argv[i][2] != '\0') {
            output_file = argv[i] + 2;
        } else if (argv[i][0] != '-') {
            input_files[n_input_files++] = argv[i];
        }
    }

    if (n_input_files == 0 && from_wasm_path == NULL) {
        println(str_lit("usage: tinyc [--emit=mlir|lowered|llvm|wasm|wasmssa|wasmstack|wat|macho] [--lowering=upstream|native] [--from-wasm PATH] [--wasm-runtime-obj=PATH ...] [-I dir ...] [-D name[=value] ...] [-o OUT] FILE.tc [FILE2.tc ...]"));
        arena_destroy(boot_arena);
        return 1;
    }

    if ((emit_wasmssa || emit_wasmstack) &&
#ifdef TINYC_HAS_UPSTREAM
            lowering_explicit && use_upstream
#else
            false
#endif
            ) {
        fprintf(stderr,
                "tinyc: --emit=wasmssa/wasmstack require "
                "--lowering=native (upstream lowering does not produce "
                "these IRs)\n");
        arena_destroy(boot_arena);
        return 1;
    }
#ifdef TINYC_HAS_UPSTREAM
    if (emit_wasmssa || emit_wasmstack) {
        use_upstream = false;
    }
    // The native llvm Mach-O backend consumes the in-house `llvm` dialect,
    // which is only produced by the native lowering. Force native lowering on.
    if (emit_macho && macho_backend_llvm) {
        use_upstream = false;
    }
    if (use_upstream) {
        lower_fn = MLIR_LowerToLLVMDialectUpstream;
        // Upstream's wasm path goes through LLVM's WebAssembly target,
        // which expects full LLVM-dialect (post scf->cf) input. The
        // *ForWasmUpstream variant (cf->scf lift + leave scf in place)
        // exists for the cross-case "upstream lift, native wasm
        // pipeline" comparison, not for the regular upstream wasm flow.
        lower_for_wasm_fn = MLIR_LowerToLLVMDialectUpstream;
        translate_to_llvm_fn = MLIR_TranslateModuleToLLVMIRUpstream;
        translate_to_wasm_fn = MLIR_TranslateModuleToWasmUpstream;
    } else
#endif
    {
        lower_fn = MLIR_LowerToLLVMDialect;
        lower_for_wasm_fn = MLIR_LowerToLLVMDialectForWasm;
        translate_to_llvm_fn = MLIR_TranslateModuleToLLVMIR;
        translate_to_wasm_fn = MLIR_TranslateModuleToWasm;
    }

    // Size IR-arena chunks so the *total* buddy request (this 64 MiB data
    // area plus the arena_chunk header, alignment padding, and the buddy
    // block header) stays just under 64 MiB and therefore fits in a single
    // 64 MiB (power-of-two) buddy block. A bare 64 MiB request spills a few
    // dozen bytes past 2^26 and forces buddy to round up to a 128 MiB block,
    // wasting ~50% of every chunk. On the host that waste is only virtual
    // (untouched pages never become resident), but on the wasm32 path the
    // doubled block reservations push the linear-memory high-water past the
    // hard 4 GiB cap, so this halves linmem use on the self-host lift.
    Arena *arena = arena_create(64 * 1024 * 1024 - 64 * 1024);
    MLIR_Context ctx = {0};
    MLIR_SetArenaAllocator(&ctx, arena);

    // --from-wasm short-circuits the C/MLIR front end entirely. We
    // read the linked wasm bytes, lift to wasmstack, then to wasmssa,
    // and from there reuse the wasmssa -> llvm -> aarch64 -> macho
    // pipeline. Only a few emit targets are meaningful for this path:
    // wasmstack / wasmssa (for inspection) and macho.
    if (from_wasm_path) {
        // The wasm -> wasmstack -> wasmssa -> llvm -> aarch64 -> macho
        // pipeline never queries def-use chains, so skip building them.
        // This removes the dominant per-op memory overhead and is required
        // to fit the stage1 self-lift under the 4 GiB wasm32 linmem cap.
        ctx.no_def_use_tracking = true;
        string wasm_buf = read_file_ok(arena, str_from_cstr_view(from_wasm_path));
        // read_file_ok appends a trailing NUL — strip it for the lifter,
        // which expects the exact wasm file size.
        size_t wasm_size = wasm_buf.size > 0 ? wasm_buf.size - 1 : 0;
        MLIR_OpHandle stack_mod = mlir_wasm_to_wasmstack(
            &ctx, (const uint8_t *)wasm_buf.str, wasm_size);
        if (stack_mod == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
        string out_fw;
        if (emit_wasmstack) {
            out_fw = MLIR_PrintOperationGeneric(&ctx, stack_mod);
        } else {
            MLIR_OpHandle ssa = mlir_wasmstack_to_wasmssa(&ctx, stack_mod);
            if (ssa == MLIR_INVALID_HANDLE) {
                arena_destroy(arena);
                arena_destroy(boot_arena);
                return 1;
            }
            if (emit_wasmssa) {
                out_fw = MLIR_PrintOperationGeneric(&ctx, ssa);
            } else if (emit_macho) {
                // Lift wasmssa to the in-house `llvm` dialect, then stream it
                // straight to Mach-O one function at a time (low peak memory).
                // Move the IR into a fresh arena, then free the
                // wasmstack/wasmssa arena. The lifter rebuilds every
                // type/attribute/location/value fresh (see normalise_carrier),
                // so once the intern registry is reset the produced module
                // references nothing in the old arena and it can be released.
                // This keeps the early modules from coexisting with the
                // (larger) llvm module, cutting peak RSS by roughly the size
                // of wasmstack + wasmssa.
                Arena *late_arena = arena_create(64 * 1024 * 1024 - 64 * 1024);
                MLIR_SetArenaAllocator(&ctx, late_arena);
                MLIR_ResetInternRegistry();
                uint8_t *macho_data = NULL; size_t macho_size = 0;
                MLIR_OpHandle llvm_mod = mlir_wasmssa_to_llvm(&ctx, ssa);
                if (llvm_mod == MLIR_INVALID_HANDLE) {
                    arena_destroy(late_arena);
                    arena_destroy(arena);
                    arena_destroy(boot_arena);
                    return 1;
                }
                // Promote non-escaping local allocas to SSA so the backend
                // can keep them in registers instead of a load/store per
                // access. Skippable via TINYC_NO_MEM2REG for A/B debugging.
                if (!getenv("TINYC_NO_MEM2REG"))
                    mlir_llvm_mem2reg(&ctx, llvm_mod);
                arena_destroy(arena);
                arena = late_arena;
                if (!mlir_llvm_to_macho(&ctx, llvm_mod,
                                        &macho_data, &macho_size)) {
                    arena_destroy(arena);
                    arena_destroy(boot_arena);
                    return 1;
                }
                out_fw.str = (char *)macho_data;
                out_fw.size = macho_size;
            } else {
                fprintf(stderr,
                    "tinyc --from-wasm: requires --emit=wasmstack|wasmssa|macho\n");
                arena_destroy(arena);
                arena_destroy(boot_arena);
                return 1;
            }
        }
        int wr;
        if (emit_macho) {
            if (!output_file) {
                fprintf(stderr, "tinyc --from-wasm --emit=macho: -o OUT required\n");
                arena_destroy(arena);
                arena_destroy(boot_arena);
                return 1;
            }
            wr = write_bytes_to_file(out_fw, output_file);
#if !defined(_WIN32) && !defined(__wasm__) && !defined(__TINYC__)
            if (wr == 0) {
                chmod(output_file, 0755);
            }
#endif
        } else if (output_file) {
            wr = write_string_to_file(out_fw, output_file);
        } else {
            println(str_lit("{}"), out_fw);
            wr = 0;
        }
        arena_destroy(arena);
        arena_destroy(boot_arena);
        return wr;
    }

    // Per-file preprocess + lex + parse-into accumulating Program. The
    // preprocessor is per-file (so #define / #pragma once do NOT leak
    // across files); -I dirs are shared. Cross-file func / global /
    // struct dedup happens inside tinyc_parse_into.
    //
    // `target_wasm32` flips `long` (and `size_t`/`intptr_t`/...) to
    // 32-bit so the imported function signatures we generate match
    // wasm32-wasi's ABI. tinyC otherwise hardcodes them at 64-bit
    // (the size on every 64-bit native host we support).
    //
    // The `--macho-backend=llvm` path is a NATIVE arm64/Darwin (LP64)
    // target: it uses real 64-bit pointers and a 64-bit `long`, so it
    // must NOT use wasm32 sizing even though it goes through `--emit=macho`.
    // The wasm Mach-O path keeps wasm32 sizing (its pointers are
    // 32-bit linear-memory offsets).
    bool target_wasm32 = (emit_wasm || emit_wasmssa || emit_wasmstack ||
                          emit_wat || emit_macho) &&
                         !macho_backend_llvm;
    Program *prog = arena_new(arena, Program);
    *prog = (Program){0};
    int total_errs = 0;
    for (size_t k = 0; k < n_input_files; k++) {
        string src = tinyc_preprocess(arena, str_from_cstr_view(input_files[k]),
                                      include_dirs, n_include_dirs,
                                      defines, n_defines);
        if (src.size > 0 && src.str[src.size - 1] == '\0') src.size -= 1;
        VecTcTok toks = tinyc_lex(arena, src);
        total_errs += tinyc_parse_into(arena, prog, toks, target_wasm32);
    }
    if (total_errs > 0) {
        arena_destroy(arena);
        arena_destroy(boot_arena);
        return 1;
    }

    // The native llvm backend has no libc; provide its runtime (printI64,
    // printStr, ...) as tinyC-subset C parsed into this same module, so it
    // lowers through the llvm -> aarch64 path alongside user code. Only
    // functions the user hasn't defined are injected.
    if (macho_backend_llvm)
        tinyc_inject_native_runtime(arena, prog, target_wasm32);
    MLIR_OpHandle module = tinyc_emit_module(&ctx, prog);
    if (tinyc_last_emit_errors() > 0) {
        arena_destroy(arena);
        arena_destroy(boot_arena);
        return 1;
    }

    // Promote non-escaping `llvm.alloca` locals to SSA (mem2reg) on the flat
    // cf CFG, before any lowering. This keeps non-address-taken locals out of
    // the linear-memory shadow stack so the downstream register allocator can
    // hold them in registers instead of spilling every access. Skippable via
    // TINYC_NO_MEM2REG for A/B debugging.
    if (!getenv("TINYC_NO_MEM2REG"))
        mlir_llvm_mem2reg(&ctx, module);

    if (emit_lowered || emit_llvm || emit_wasm || emit_wasmssa || emit_wasmstack || emit_wat || emit_macho) {
        bool needs_wasm_lowering = emit_wasm || emit_wasmssa || emit_wasmstack || emit_wat || emit_macho;
        bool ok = needs_wasm_lowering
                      ? lower_for_wasm_fn(&ctx, module)
                      : lower_fn(&ctx, module);
        if (!ok) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
        if (getenv("TINYC_DUMP_LOWERED")) {
            string s = MLIR_PrintOperationGeneric(&ctx, module);
            println(str_lit("{}"), s);
        }
    }
    string out;
    if (emit_wasmssa) {
        MLIR_OpHandle ssa = mlir_llvm_to_wasmssa(&ctx, module);
        if (ssa == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
        out = MLIR_PrintOperationGeneric(&ctx, ssa);
    } else if (emit_wasmstack) {
        MLIR_OpHandle ssa = mlir_llvm_to_wasmssa(&ctx, module);
        if (ssa == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
        MLIR_OpHandle stk = mlir_wasmssa_to_wasmstack(&ctx, ssa);
        if (stk == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
        out = MLIR_PrintOperationGeneric(&ctx, stk);
    } else if (emit_wat) {
        string bin = translate_to_wasm_fn(&ctx, module);
        if (bin.size == 0) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
        out = mlir_wasm_binary_to_wat(&ctx, bin);
    } else if (emit_wasm) {
        out = translate_to_wasm_fn(&ctx, module);
        if (out.size == 0) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
    } else if (emit_macho) {
        // Two paths:
        // - macho_backend_llvm == false (default): the established
        //   wasm-emit + wasm-link + wasm-to-macho pipeline.
        // - macho_backend_llvm == true: the unified llvm -> aarch64 ->
        //   macho pipeline.
        if (macho_backend_llvm) {
            // Unified llvm -> aarch64 -> macho backend.
            MLIR_OpHandle aarch64 = mlir_llvm_to_aarch64(&ctx, module);
            if (aarch64 == MLIR_INVALID_HANDLE) {
                arena_destroy(arena);
                arena_destroy(boot_arena);
                return 1;
            }
            uint8_t *macho_data = NULL; size_t macho_size = 0;
            if (!mlir_aarch64_to_macho(&ctx, aarch64, &macho_data, &macho_size)) {
                arena_destroy(arena);
                arena_destroy(boot_arena);
                return 1;
            }
            out.str = (char *)macho_data;
            out.size = macho_size;
        } else {
            // Pipeline: lower to wasm32 object, link with the provided
            // --wasm-runtime-obj inputs, then translate the linked wasm
            // module to a signed Mach-O ARM64 binary.
            string obj = translate_to_wasm_fn(&ctx, module);
            if (obj.size == 0) {
                arena_destroy(arena);
                arena_destroy(boot_arena);
                return 1;
            }

            size_t n_link_inputs = n_wasm_runtime_objs + 1;
            MLIR_WasmLinkInput *ins = (MLIR_WasmLinkInput *)arena_new_array(
                arena, MLIR_WasmLinkInput, n_link_inputs);
            ins[0].data = (const uint8_t *)obj.str;
            ins[0].size = obj.size;
            ins[0].name = "<input>.wasm.o";
            for (size_t k = 0; k < n_wasm_runtime_objs; k++) {
                string buf = read_file_ok(arena, str_from_cstr_view(wasm_runtime_objs[k]));
                ins[1 + k].data = (const uint8_t *)buf.str;
                // read_file_ok appends a trailing NUL — strip it for the
                // linker, which expects the exact file size.
                ins[1 + k].size = buf.size > 0 ? buf.size - 1 : 0;
                ins[1 + k].name = wasm_runtime_objs[k];
            }
            uint8_t *linked_data = NULL; size_t linked_size = 0;
            if (!MLIR_WasmLink(ins, n_link_inputs, "_start", &linked_data, &linked_size)) {
                fprintf(stderr, "tinyc --emit=macho: wasm link failed\n");
                arena_destroy(arena);
                arena_destroy(boot_arena);
                return 1;
            }
            // Optional debugging hook: dump the post-link wasm bytes that
            // feed the macho translator. Useful when a test fails to
            // identify which opcode/feature the backend is missing; pair
            // with `wasm2wat $TINYC_DUMP_LINKED_WASM`.
            if (getenv("TINYC_DUMP_LINKED_WASM")) {
                FILE *df = fopen(getenv("TINYC_DUMP_LINKED_WASM"), "wb");
                if (df) { fwrite(linked_data, 1, linked_size, df); fclose(df); }
            }

            uint8_t *macho_data = NULL; size_t macho_size = 0;
            bool macho_ok = MLIR_WasmToMachoArm64(linked_data, linked_size,
                                                  &macho_data, &macho_size);
            free(linked_data);
            if (!macho_ok) {
                fprintf(stderr, "tinyc --emit=macho: wasm->macho translation failed\n");
                arena_destroy(arena);
                arena_destroy(boot_arena);
                return 1;
            }
            out.str = (char *)macho_data;
            out.size = macho_size;
        }
    } else if (emit_llvm) {
        out = translate_to_llvm_fn(&ctx, module);
        if (out.size == 0) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
    } else {
        out = print_fn(&ctx, module);
    }
    int wrc = 0;
    if (emit_macho && !output_file) {
        fprintf(stderr, "tinyc --emit=macho: -o PATH is required\n");
        if (emit_macho) free(out.str);
        arena_destroy(arena);
        arena_destroy(boot_arena);
        return 1;
    }
    if (output_file) {
        if (emit_wasm || emit_macho) {
            wrc = write_bytes_to_file(out, output_file);
#if !defined(_WIN32) && !defined(__wasm__) && !defined(__TINYC__)
            if (wrc == 0 && emit_macho) {
                // chmod 0755 so the resulting Mach-O is directly runnable.
                chmod(output_file, 0755);
            }
#endif
        } else {
            wrc = write_string_to_file(out, output_file);
        }
    } else {
        println(str_lit("{}"), out);
    }

    if (emit_macho) {
        // out.str was malloc'd by MLIR_WasmToMachoArm64. Free it.
        free(out.str);
    }

    arena_destroy(arena);
    arena_destroy(boot_arena);
    return wrc;
}
