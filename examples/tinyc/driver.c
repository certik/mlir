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
#include "mlir_llvm_load_cse.h"
#include "mlir_llvm_arith_gvn.h"
#include "mlir_llvm_dce.h"
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
// The x86_64 -> ELF backend is native-Linux-only (it emits x86_64 machine code
// and a Linux ELF, and is not part of the tinyC self-host source set), so it is
// compiled in only for Linux/x86_64 hosts, where mlir_llvm_to_x64.c /
// mlir_elf.c are linked. Every other build (wasm/self-host, macOS, Windows)
// omits it.
#if defined(__linux__) && defined(__x86_64__) && !defined(__TINYC__)
#include "mlir_llvm_to_x64.h"
#define TINYC_HAS_ELF 1
#endif
#include "tinyc.h"

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

// True if `op` is an `llvm.func` carrying a non-empty body (a definition, not
// a declaration).
static bool driver_llvm_func_has_body(MLIR_OpHandle op) {
    string nm = MLIR_GetOpName(op);
    if (!(nm.size == 9 && memcmp(nm.str, "llvm.func", 9) == 0)) return false;
    if (MLIR_GetOpNumRegions(op) < 1) return false;
    return MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
}

// Compile corec's platform_<os>.c as LP64 and return its file-I/O + exit
// function definitions (as `llvm.func` ops) in `picks`, so a caller can splice
// them into a lifted wasm->llvm module. MUST be called BEFORE the wasm module
// is lifted: the optimization passes (mem2reg / lowering) mutate per-value
// def-use chains, and compiling this second module after the lifted one is
// already built corrupts its operands. Compiling first — when the interning
// epoch is clean — keeps it correct, exactly like a standalone compile.
//
// The public platform_* entry points are renamed to __host_platform_* (the
// lifted module already contains the wasm-side platform_* from platform_wasm.c)
// and libc calls are renamed to the underscored libSystem symbols the Mach-O
// import table binds. pmod's arena is intentionally leaked so its funcs stay
// live after they are later moved into the lifted module.
static bool tinyc_compile_host_platform(MLIR_Context *ctx, const char *path,
                                        string *include_dirs, size_t n_include_dirs,
                                        bool (*lower_fn)(MLIR_Context *, MLIR_OpHandle),
                                        bool is_wasi_adapter,
                                        MLIR_OpHandle *picks, size_t *n_picks,
                                        size_t max_picks) {
    *n_picks = 0;
    Arena *saved_arena = MLIR_GetArenaAllocator(ctx);
    Arena *pmod_arena = arena_create(16 * 1024 * 1024);
    MLIR_SetArenaAllocator(ctx, pmod_arena);
    MLIR_ResetInternRegistry();
    // The --from-wasm pipeline disables def-use tracking (ctx.no_def_use_
    // tracking) to save memory, but mem2reg / lowering rewrite operands through
    // ReplaceAllUsesOfValue, which needs the per-value use lists. Re-enable
    // tracking just for this compile (only the platform funcs pay the cost),
    // then restore the caller's setting.
    bool saved_no_def_use = ctx->no_def_use_tracking;
    ctx->no_def_use_tracking = false;

    string defs[16]; size_t nd = 0;
    if (is_wasi_adapter) {
        // The WASI adapter defines fd_write/path_open/... and calls the
        // already-renamed __host_platform_* plus fchmod (Mach-O: _fchmod).
        defs[nd++] = str_from_cstr_view((char *)"fchmod=_fchmod");
    } else {
    defs[nd++] = str_from_cstr_view((char *)"PLATFORM_SKIP_ENTRY=1");
    defs[nd++] = str_from_cstr_view((char *)"platform_fd_write=__host_platform_fd_write");
    defs[nd++] = str_from_cstr_view((char *)"platform_fd_read=__host_platform_fd_read");
    defs[nd++] = str_from_cstr_view((char *)"platform_fd_close=__host_platform_fd_close");
    defs[nd++] = str_from_cstr_view((char *)"platform_fd_seek=__host_platform_fd_seek");
    defs[nd++] = str_from_cstr_view((char *)"platform_fd_tell=__host_platform_fd_tell");
    defs[nd++] = str_from_cstr_view((char *)"platform_path_open=__host_platform_path_open");
    defs[nd++] = str_from_cstr_view((char *)"platform_exit=__host_platform_exit");
    defs[nd++] = str_from_cstr_view((char *)"writev=_writev");
    defs[nd++] = str_from_cstr_view((char *)"readv=_readv");
    defs[nd++] = str_from_cstr_view((char *)"fcntl=_fcntl");
    defs[nd++] = str_from_cstr_view((char *)"open=_open");
    defs[nd++] = str_from_cstr_view((char *)"close=_close");
    defs[nd++] = str_from_cstr_view((char *)"lseek=_lseek");
    defs[nd++] = str_from_cstr_view((char *)"__error=___error");
    }

    string src = tinyc_preprocess(pmod_arena, str_from_cstr_view((char *)path),
                                  include_dirs, n_include_dirs, defs, nd);
    if (src.size > 0 && src.str[src.size - 1] == '\0') src.size -= 1;
    VecTcTok toks = tinyc_lex(pmod_arena, src);
    Program *prog = arena_new(pmod_arena, Program);
    *prog = (Program){0};
    if (tinyc_parse_into(pmod_arena, prog, toks, false) > 0) {
        fprintf(stderr, "tinyc: parse failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        ctx->no_def_use_tracking = saved_no_def_use;
        return false;
    }
    MLIR_OpHandle pmod = tinyc_emit_module(ctx, prog);
    if (tinyc_last_emit_errors() > 0 || pmod == MLIR_INVALID_HANDLE) {
        fprintf(stderr, "tinyc: emit failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        ctx->no_def_use_tracking = saved_no_def_use;
        return false;
    }
    if (!getenv("TINYC_NO_MEM2REG"))
        mlir_llvm_mem2reg(ctx, pmod);
    if (!lower_fn(ctx, pmod)) {
        fprintf(stderr, "tinyc: lowering failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        ctx->no_def_use_tracking = saved_no_def_use;
        return false;
    }
    MLIR_BlockHandle pbody = MLIR_GetRegionBlock(MLIR_GetOpRegion(pmod, 0), 0);
    size_t pn = MLIR_GetBlockNumOps(pbody);
    size_t np = 0;
    for (size_t i = 0; i < pn && np < max_picks; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(pbody, i);
        if (!driver_llvm_func_has_body(op)) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) continue;
        string s = MLIR_GetAttributeString(sa);
        // The platform file also defines non-renamed platform_* that collide
        // with the wasm-side copies, so pick only the __host_platform_* entry
        // points. The WASI adapter has no such collisions: pick every func.
        if (!is_wasi_adapter &&
            (s.size < 16 || memcmp(s.str, "__host_platform_", 16) != 0)) continue;
        picks[np++] = op;
    }
    MLIR_SetArenaAllocator(ctx, saved_arena);
        ctx->no_def_use_tracking = saved_no_def_use;
    if (np == 0) {
        fprintf(stderr, "tinyc: no spliceable definitions in '%s'\n", path);
        return false;
    }
    *n_picks = np;
    return true;
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
    // `--emit=aarch64` stops the unified llvm -> aarch64 -> macho backend
    // at the physical-register `aarch64` dialect and prints it (the
    // "MachO MLIR"), instead of encoding the final Mach-O bytes. It is a
    // debugging view of the same pipeline `--emit=macho --macho-backend=llvm`
    // drives, so `--emit=mlir`, `--emit=llvm`, `--emit=aarch64` give the
    // three successive IRs for the native llvm backend.
    bool emit_aarch64 = false;
    // --macho-backend selects which pipeline produces the final Mach-O
    // binary. "wasm" (default) keeps the established wasm-link +
    // wasm->macho path. "llvm" routes the `llvm` dialect directly through
    // the unified llvm -> aarch64 -> macho backend.
    bool macho_backend_llvm = false;
    // `--emit=elf` lowers the (native, LP64) `llvm` dialect through the flat
    // (scf->cf) standard lowering and the in-tree x86_64 code generator to a
    // statically-linked Linux ELF executable (direct syscalls, no libc).
    bool emit_elf = false;
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
    // objects to link into the module when producing a Mach-O binary
    // (general-purpose; tinyC's own builds no longer need any).
    char **wasm_runtime_objs = arena_new_array(boot_arena, char *, argc + 1);
    size_t n_wasm_runtime_objs = 0;

    // --from-wasm=PATH: bypass the C/MLIR front end and read a linked
    // wasm32 module directly. This routes the wasm bytes through
    // wasm -> wasmstack -> wasmssa -> ... -> chosen emit target. Used
    // by the llvm-via-wasm Mach-O backend.
    char *from_wasm_path = NULL;

    // --host-platform=PATH: corec platform_<os>.c whose file-I/O + exit
    // primitives are compiled (LP64) and spliced into the wasm->llvm module so
    // the WASI adapters call the real platform implementation. Only meaningful
    // on the `--from-wasm --emit=macho --macho-backend=llvm` path.
    char *host_platform_path = NULL;
    // --wasi-adapter=PATH: corec/wasm/wasi_adapter.c, the C WASI shim
    // (fd_write/path_open/args_*/...) spliced into the lifted module instead of
    // synthesising the shims in mlir_wasmssa_to_llvm.c. Used with --from-wasm
    // --emit=macho.
    char *wasi_adapter_path = NULL;

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
        if      (strcmp(argv[i], "--emit=mlir")    == 0) { emit_llvm = false; emit_lowered = false; emit_wasm = false; emit_wasmssa = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; emit_aarch64 = false; }
        else if (strcmp(argv[i], "--emit=lowered") == 0) { emit_lowered = true;  emit_llvm = false; emit_wasm = false; emit_wasmssa = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; emit_aarch64 = false; }
        else if (strcmp(argv[i], "--emit=llvm")    == 0) { emit_llvm = true;     emit_lowered = false; emit_wasm = false; emit_wasmssa = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; emit_aarch64 = false; }
        else if (strcmp(argv[i], "--emit=wasm")    == 0) { emit_wasm = true;     emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; emit_aarch64 = false; }
        else if (strcmp(argv[i], "--emit=wasmssa") == 0) { emit_wasmssa = true; emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmstack = false; emit_wat = false; emit_macho = false; emit_aarch64 = false; }
        else if (strcmp(argv[i], "--emit=wasmstack") == 0) { emit_wasmstack = true; emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wat = false; emit_macho = false; emit_aarch64 = false; }
        else if (strcmp(argv[i], "--emit=wat")     == 0) { emit_wat = true;     emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wasmstack = false; emit_macho = false; emit_aarch64 = false; }
        else if (strcmp(argv[i], "--emit=macho")   == 0) { emit_macho = true;   emit_wat = false; emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wasmstack = false; emit_aarch64 = false; }
        else if (strcmp(argv[i], "--emit=aarch64") == 0) { emit_aarch64 = true; emit_macho = false; emit_wat = false; emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wasmstack = false; }
#ifdef TINYC_HAS_ELF
        else if (strcmp(argv[i], "--emit=elf")     == 0) { emit_elf = true; emit_aarch64 = false; emit_macho = false; emit_wat = false; emit_wasm = false; emit_llvm = false; emit_lowered = false; emit_wasmssa = false; emit_wasmstack = false; }
#endif
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
        else if (strncmp(argv[i], "--host-platform=", 16) == 0) {
            host_platform_path = argv[i] + 16;
        }
        else if (strcmp(argv[i], "--host-platform") == 0 && i + 1 < argc) {
            host_platform_path = argv[++i];
        }
        else if (strncmp(argv[i], "--wasi-adapter=", 15) == 0) {
            wasi_adapter_path = argv[i] + 15;
        }
        else if (strcmp(argv[i], "--wasi-adapter") == 0 && i + 1 < argc) {
            wasi_adapter_path = argv[++i];
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
        println(str_lit("usage: tinyc [--emit=mlir|lowered|llvm|wasm|wasmssa|wasmstack|wat|aarch64|macho] [--lowering=upstream|native] [--from-wasm PATH] [--wasm-runtime-obj=PATH ...] [-I dir ...] [-D name[=value] ...] [-o OUT] FILE.tc [FILE2.tc ...]"));
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
    // `--emit=aarch64` drives the same backend (stopping at the aarch64
    // dialect), so it needs native lowering too.
    if ((emit_macho && macho_backend_llvm) || emit_aarch64) {
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
                // The via-wasm Mach-O backend's WASI adapters call corec's
                // platform_<os>.c (as __host_platform_*), so the path to it is
                // required.
                if (!host_platform_path) {
                    fprintf(stderr,
                        "tinyc --from-wasm --emit=macho: --host-platform=PATH "
                        "is required (e.g. corec/platform/platform_macos.c)\n");
                    arena_destroy(arena);
                    arena_destroy(boot_arena);
                    return 1;
                }
                // Compile the host platform file FIRST, while the interning
                // epoch is clean: its funcs are spliced into the lifted module
                // after the opt passes (see tinyc_compile_host_platform).
                MLIR_OpHandle hp_picks[8]; size_t hp_n = 0;
                if (!tinyc_compile_host_platform(&ctx, host_platform_path,
                                                 include_dirs, n_include_dirs,
                                                 lower_for_wasm_fn, false,
                                                 hp_picks, &hp_n, 8)) {
                    arena_destroy(arena);
                    arena_destroy(boot_arena);
                    return 1;
                }
                // Compile + splice the C WASI adapter (fd_write/path_open/...)
                // the same way, when provided; otherwise the lifter's
                // synthesised shims are used. Baked while the epoch is clean.
                MLIR_OpHandle ad_picks[16]; size_t ad_n = 0;
                if (wasi_adapter_path &&
                    !tinyc_compile_host_platform(&ctx, wasi_adapter_path,
                                                 include_dirs, n_include_dirs,
                                                 lower_for_wasm_fn, true,
                                                 ad_picks, &ad_n, 16)) {
                    arena_destroy(arena);
                    arena_destroy(boot_arena);
                    return 1;
                }
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
                // Load CSE on the lifted llvm CFG: within single-entry
                // regions, removes redundant llvm.loads of a structurally
                // identical address (no memory clobber in between) and DCEs the
                // now-detached inttoptr(add(base,zext(idx))) address chains.
                // Targets the repeated mem[i] reloads that the short-circuit
                // `||` comparison chains emit, which dominate the runtime of the
                // code we generate. Skippable via TINYC_NO_LOAD_CSE for A/B.
                if (!getenv("TINYC_NO_LOAD_CSE"))
                    mlir_llvm_load_cse(&ctx, llvm_mod);
                // Intra-block value numbering of pure integer arithmetic: dedups
                // the repeated index/address arithmetic (base+index*stride) the
                // wasm frontend emits, cutting op count and register pressure.
                // The memfuse address spine is excluded so [x28,Widx,UXTW]
                // fusion is preserved. Skippable via TINYC_NO_ARITH_GVN.
                mlir_llvm_arith_gvn(&ctx, llvm_mod);
                // Whole-function dead-code elimination: the tinyC front end
                // emits the value of every expression even in statement context
                // (e.g. the (char)v result of `d[i] = v;`), which survives as a
                // zero-use pure value after mem2reg/CSE/GVN. The backend has no
                // DCE and would slot+spill each one (a wasted store per loop
                // iteration in hot mem/str helpers). Remove them at the source.
                // Skippable via TINYC_NO_DCE.
                mlir_llvm_dce(&ctx, llvm_mod);
                // Splice the pre-compiled host platform funcs into the lifted
                // module now (after the wasm-oriented opt passes, which assume
                // linmem shapes and must not see the LP64 platform funcs). The
                // funcs were baked while the epoch was clean; moving them is a
                // pure re-parent that preserves their operands.
                if (hp_n > 0 || ad_n > 0) {
                    MLIR_BlockHandle dbody =
                        MLIR_GetRegionBlock(MLIR_GetOpRegion(llvm_mod, 0), 0);
                    for (size_t hi = 0; hi < hp_n; hi++)
                        MLIR_MoveOpToBlockEnd(&ctx, hp_picks[hi], dbody);
                    for (size_t hi = 0; hi < ad_n; hi++)
                        MLIR_MoveOpToBlockEnd(&ctx, ad_picks[hi], dbody);
                }
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
            } else if (emit_aarch64) {
                // Same wasmssa -> llvm lift + optimization passes as the
                // macho path above, but stop at the physical-register
                // `aarch64` dialect and print it (the "MachO MLIR"). No
                // late-arena swap: this is a debug view, not the memory-
                // constrained self-host emit.
                MLIR_OpHandle hp_picks2[8]; size_t hp_n2 = 0;
                if (host_platform_path &&
                    !tinyc_compile_host_platform(&ctx, host_platform_path,
                                                 include_dirs, n_include_dirs,
                                                 lower_for_wasm_fn, false,
                                                 hp_picks2, &hp_n2, 8)) {
                    arena_destroy(arena);
                    arena_destroy(boot_arena);
                    return 1;
                }
                MLIR_OpHandle llvm_mod = mlir_wasmssa_to_llvm(&ctx, ssa);
                if (llvm_mod == MLIR_INVALID_HANDLE) {
                    arena_destroy(arena);
                    arena_destroy(boot_arena);
                    return 1;
                }
                if (!getenv("TINYC_NO_MEM2REG"))
                    mlir_llvm_mem2reg(&ctx, llvm_mod);
                if (!getenv("TINYC_NO_LOAD_CSE"))
                    mlir_llvm_load_cse(&ctx, llvm_mod);
                mlir_llvm_arith_gvn(&ctx, llvm_mod);
                mlir_llvm_dce(&ctx, llvm_mod);
                if (hp_n2 > 0) {
                    MLIR_BlockHandle dbody =
                        MLIR_GetRegionBlock(MLIR_GetOpRegion(llvm_mod, 0), 0);
                    for (size_t hi = 0; hi < hp_n2; hi++)
                        MLIR_MoveOpToBlockEnd(&ctx, hp_picks2[hi], dbody);
                }
                MLIR_OpHandle aarch64 = mlir_llvm_to_aarch64(&ctx, llvm_mod);
                if (aarch64 == MLIR_INVALID_HANDLE) {
                    arena_destroy(arena);
                    arena_destroy(boot_arena);
                    return 1;
                }
                out_fw = MLIR_PrintOperationGeneric(&ctx, aarch64);
            } else {
                fprintf(stderr,
                    "tinyc --from-wasm: requires --emit=wasmstack|wasmssa|aarch64|macho\n");
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
    prog->print_via_printf = emit_llvm || (emit_macho && macho_backend_llvm);
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

    // --emit=elf --host-platform=PATH: compile corec's platform_<os>.c into the
    // same module so the runtime's OS primitives bind to the real platform layer
    // (e.g. _write -> platform_fd_write -> __builtin_syscall6) instead of
    // synthesized syscall thunks. PLATFORM_SKIP_ENTRY drops its _start (tinyC
    // supplies its own); unreached platform funcs are trimmed by the backend.
    if (emit_elf && host_platform_path) {
        string *pdefs = arena_new_array(arena, string, n_defines + 1);
        for (size_t k = 0; k < n_defines; k++) pdefs[k] = defines[k];
        pdefs[n_defines] = str_from_cstr_view((char *)"PLATFORM_SKIP_ENTRY=1");
        string psrc = tinyc_preprocess(arena, str_from_cstr_view(host_platform_path),
                                       include_dirs, n_include_dirs, pdefs, n_defines + 1);
        if (psrc.size > 0 && psrc.str[psrc.size - 1] == '\0') psrc.size -= 1;
        VecTcTok ptoks = tinyc_lex(arena, psrc);
        if (tinyc_parse_into(arena, prog, ptoks, target_wasm32) > 0) {
            fprintf(stderr, "tinyc: parse failed for host platform '%s'\n",
                    host_platform_path);
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
    }

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

    if (emit_lowered || emit_llvm || emit_wasm || emit_wasmssa || emit_wasmstack || emit_wat || emit_macho || emit_aarch64 || emit_elf) {
        bool needs_wasm_lowering = emit_wasm || emit_wasmssa || emit_wasmstack || emit_wat || emit_macho || emit_aarch64;
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
    } else if (emit_aarch64) {
        // Stop the unified llvm -> aarch64 -> macho backend at the
        // physical-register `aarch64` dialect and print it (the "MachO MLIR").
        MLIR_OpHandle aarch64 = mlir_llvm_to_aarch64(&ctx, module);
        if (aarch64 == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
        out = MLIR_PrintOperationGeneric(&ctx, aarch64);
#ifdef TINYC_HAS_ELF
    } else if (emit_elf) {
        // Native x86_64 -> static ELF executable (Linux, direct syscalls).
        uint8_t *elf_data = NULL; size_t elf_size = 0;
        if (!mlir_llvm_to_elf(&ctx, module, &elf_data, &elf_size)) {
            arena_destroy(arena);
            arena_destroy(boot_arena);
            return 1;
        }
        out.str = (char *)elf_data;
        out.size = elf_size;
#endif
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
    if ((emit_macho || emit_elf) && !output_file) {
        fprintf(stderr, "tinyc --emit=%s: -o PATH is required\n",
                emit_elf ? "elf" : "macho");
        if (emit_macho || emit_elf) free(out.str);
        arena_destroy(arena);
        arena_destroy(boot_arena);
        return 1;
    }
    if (output_file) {
        if (emit_wasm || emit_macho || emit_elf) {
            wrc = write_bytes_to_file(out, output_file);
#if !defined(_WIN32) && !defined(__wasm__) && !defined(__TINYC__)
            if (wrc == 0 && (emit_macho || emit_elf)) {
                // chmod 0755 so the resulting binary is directly runnable.
                chmod(output_file, 0755);
            }
#endif
        } else {
            wrc = write_string_to_file(out, output_file);
        }
    } else {
        println(str_lit("{}"), out);
    }

    if (emit_macho || emit_elf) {
        // out.str was malloc'd by the backend. Free it.
        free(out.str);
    }

    arena_destroy(arena);
    arena_destroy(boot_arena);
    return wrc;
}
