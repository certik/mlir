// tinyC driver — read one or more .tc files, lex/parse/emit a single
// MLIR module, print to stdout (or to -o <path>).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/io.h>
#include <base/string.h>
#include <base/strbuf.h>
#include <platform/platform.h>

#include "mlir_api.h"
#include "mlir_passes.h"
#include "mlir_printer.h"
#ifdef TINYC_HAS_UPSTREAM
#include "mlir_api_upstream.h"
#endif
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
// and a Linux ELF, and is not part of the tinyC self-host source set by
// default). It is compiled in for:
//   - native Linux/x86_64 host builds (tinyc_native; mlir_llvm_to_x64.c /
//     mlir_elf.c are linked), and
//   - self-hosted ELF stages (tinyc compiled to wasm with -DTINYC_ENABLE_ELF=1,
//     then lifted back to a native ELF that must retain --emit=elf / --from-wasm).
// Every other build (wasm/self-host without TINYC_ENABLE_ELF, macOS, Windows)
// omits it.
#if (defined(__linux__) && defined(__x86_64__) && !defined(__TINYC__)) \
    || defined(TINYC_ENABLE_ELF)
#include "mlir_llvm_to_x64.h"
#define TINYC_HAS_ELF 1
#endif
#include "tinyc.h"

#if !defined(_WIN32) && !defined(__wasm__) && !defined(__TINYC__)
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// --time-stages: wall time per compiler stage (platform_clock_ns, a monotonic
// clock on every platform build), printed to stderr when the driver exits.
// Off by default; then every stage boundary costs one flag test and nothing
// else. Stages that run several times (e.g. preprocess per input file) add up.
// The ids are plain macros: tinyC does not accept enum constants in constant
// expressions such as array sizes.
// ---------------------------------------------------------------------------
#define TS_PREPROCESS      0
#define TS_LEX             1
#define TS_PARSE           2
#define TS_EMIT            3
#define TS_WASM_READ       4
#define TS_WASM_TO_STACK   5
#define TS_STACK_TO_SSA    6
#define TS_HOST_HELPERS    7
#define TS_SSA_TO_LLVM     8
#define TS_MEM2REG         9
#define TS_LOAD_CSE        10
#define TS_ARITH_GVN       11
#define TS_DCE             12
#define TS_LOWER           13
#define TS_X64_SETUP       14
#define TS_X64_REGALLOC    15
#define TS_X64_ISEL_ENCODE 16
#define TS_X64_LINK        17
#define TS_CODEGEN         18
#define TS_LINK_WASM       19
#define TS_WRITE           20
#define TS_COUNT           21

static bool g_ts_on = false;
static uint64_t g_ts_ns[TS_COUNT];
static uint64_t g_ts_start;
#ifdef TINYC_HAS_ELF
static MLIR_X64StageTimes g_ts_x64;
#endif

static uint64_t ts_now(void) { return g_ts_on ? platform_clock_ns() : 0; }

static void ts_add(int stage, uint64_t t0) {
    if (g_ts_on) g_ts_ns[stage] += platform_clock_ns() - t0;
}

// Stage names, one word each so scripts can split the report on whitespace.
// (A switch, not a string table: tinyC has no pointer initializers in global
// arrays.)
static const char *ts_name(int stage) {
    switch (stage) {
        case TS_PREPROCESS:      return "preprocess";
        case TS_LEX:             return "lex";
        case TS_PARSE:           return "parse";
        case TS_EMIT:            return "emit";
        case TS_WASM_READ:       return "wasm-read";
        case TS_WASM_TO_STACK:   return "wasm->wasmstack";
        case TS_STACK_TO_SSA:    return "wasmstack->wasmssa";
        case TS_HOST_HELPERS:    return "host-helpers";
        case TS_SSA_TO_LLVM:     return "wasmssa->llvm";
        case TS_MEM2REG:         return "mem2reg";
        case TS_LOAD_CSE:        return "load-cse";
        case TS_ARITH_GVN:       return "arith-gvn";
        case TS_DCE:             return "dce";
        case TS_LOWER:           return "lower-to-llvm";
        case TS_X64_SETUP:       return "x64-setup";
        case TS_X64_REGALLOC:    return "x64-regalloc";
        case TS_X64_ISEL_ENCODE: return "x64-isel+encode";
        case TS_X64_LINK:        return "x64-elf-link";
        case TS_CODEGEN:         return "codegen";
        case TS_LINK_WASM:       return "wasm-link";
        case TS_WRITE:           return "write";
        default:                 return "?";
    }
}

static void ts_print_row(const char *name, uint64_t ns, uint64_t total) {
    unsigned long long us = (unsigned long long)(ns / 1000);
    unsigned long long pm = total ? (unsigned long long)(ns * 1000 / total) : 0;
    fprintf(stderr, "  %-20s %8llu.%03llu %4llu.%llu%%\n", name,
            us / 1000, us % 1000, pm / 10, pm % 10);
}

// The report: every stage that ran, then `other` (driver work between the
// timed stages: argument parsing, arena setup/teardown, ...) and `total`
// (whole driver run). Times in milliseconds.
static void ts_report(void) {
#ifdef TINYC_HAS_ELF
    g_ts_ns[TS_X64_SETUP]       += g_ts_x64.setup_ns;
    g_ts_ns[TS_X64_REGALLOC]    += g_ts_x64.regalloc_ns;
    g_ts_ns[TS_X64_ISEL_ENCODE] += g_ts_x64.isel_encode_ns;
    g_ts_ns[TS_X64_LINK]        += g_ts_x64.link_ns;
#endif
    uint64_t total = platform_clock_ns() - g_ts_start;
    uint64_t sum = 0;
    fprintf(stderr, "tinyc --time-stages (wall ms):\n");
    for (int s = 0; s < TS_COUNT; s++) {
        if (g_ts_ns[s] == 0) continue;
        sum += g_ts_ns[s];
        ts_print_row(ts_name(s), g_ts_ns[s], total);
    }
    ts_print_row("other", total > sum ? total - sum : 0, total);
    ts_print_row("total", total, total);
}

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
    // An empty output (e.g. --emit=x64 of a file with no function bodies) is
    // just the truncated file: there is nothing to write.
    if (out.size == 0) {
        platform_fd_close(fd);
        return 0;
    }
    ciovec_t iovs[1];
    iovs[0].buf = out.str;
    iovs[0].buf_len = out.size;
    uint32_t werr = write_all(fd, iovs, 1);
    platform_fd_close(fd);
    return werr ? 1 : 0;
}

// True if `op` is an `llvm.func` carrying a non-empty body (a definition, not
// a declaration).
static bool driver_llvm_func_has_body(MLIR_Context *ctx, MLIR_OpHandle op) {
    if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_FUNC) return false;
    if (MLIR_GetOpNumRegions(ctx, op) < 1) return false;
    return MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, op, 0)) > 0;
}

// Compile corec's platform_<os>.c as LP64 and return its file-I/O + exit
// function definitions (as `llvm.func` ops) in `picks`, so a caller can splice
// them into a lifted wasm->llvm module. MUST be called BEFORE the wasm module
// is lifted: the optimization passes (mem2reg / lowering) mutate per-value
// def-use chains, and compiling this second module after the lifted one is
// already built corrupts its operands. Compiling first keeps it correct,
// exactly like a standalone compile.
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
    // The --from-wasm pipeline disables def-use tracking
    // (MLIR_SetContextDefUseTracking) to save memory, but mem2reg / lowering rewrite operands through
    // ReplaceAllUsesOfValue, which needs the per-value use lists. Re-enable
    // tracking just for this compile (only the platform funcs pay the cost),
    // then restore the caller's setting.
    bool saved_def_use = MLIR_HasContextDefUseTracking(ctx);
    MLIR_SetContextDefUseTracking(ctx, true);

    string defs[16]; size_t nd = 0;
    if (is_wasi_adapter) {
        // wasi_adapter.c calls platform_*; PLATFORM_HOST_SHIM (platform.h) renames
        // those to __host_platform_* so they reach the spliced host copies.
        defs[nd++] = str_from_cstr_view((char *)"PLATFORM_HOST_SHIM=1");
    } else {
        defs[nd++] = str_from_cstr_view((char *)"PLATFORM_SKIP_ENTRY=1");
        defs[nd++] = str_from_cstr_view((char *)"PLATFORM_HOST_SHIM=1");
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
    if (tinyc_last_preprocess_errors() > 0) {
        fprintf(stderr, "tinyc: preprocessing failed for %s '%s'\n",
                is_wasi_adapter ? "WASI adapter" : "host platform", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
        return false;
    }
    if (src.size > 0 && src.str[src.size - 1] == '\0') src.size -= 1;
    VecTcTok toks = tinyc_lex(pmod_arena, src);
    Program *prog = arena_new(pmod_arena, Program);
    *prog = (Program){0};
    // These sources run natively after the wasm module is lifted, so extern
    // variadic calls such as open(path, flags, mode) must use the host ABI.
    prog->host_varargs = true;
    if (tinyc_parse_into(pmod_arena, prog, toks, false) > 0) {
        fprintf(stderr, "tinyc: parse failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
        return false;
    }
    MLIR_OpHandle pmod = tinyc_emit_module(ctx, prog);
    if (tinyc_last_emit_errors() > 0 || pmod == MLIR_INVALID_HANDLE) {
        fprintf(stderr, "tinyc: emit failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
        return false;
    }
    if (!getenv("TINYC_NO_MEM2REG"))
        mlir_llvm_mem2reg(ctx, pmod);
    if (!lower_fn(ctx, pmod)) {
        fprintf(stderr, "tinyc: lowering failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
        return false;
    }
    MLIR_BlockHandle pbody = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, pmod, 0), 0);
    size_t pn = MLIR_GetBlockNumOps(ctx, pbody);
    size_t np = 0;
    for (size_t i = 0; i < pn; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, pbody, i);
        if (!driver_llvm_func_has_body(ctx, op)) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME);
        if (sa == MLIR_INVALID_HANDLE) continue;
        string s = MLIR_GetAttributeString(ctx, sa);
        // The platform file also defines non-renamed platform_* that collide
        // with the wasm-side copies, so pick only the __host_platform_* entry
        // points. Skip init only: it calls ensure_heap_initialized helpers the
        // pick-based splice does not move (wasm crt0 handles init). Keep exit:
        // mlir_wasmssa_to_llvm rewrites proc_exit -> __host_platform_exit.
        // The WASI adapter has no such collisions: pick every func.
        if (!is_wasi_adapter) {
            if (s.size < 16 || memcmp(s.str, "__host_platform_", 16) != 0) continue;
            if (s.size == 20 && memcmp(s.str, "__host_platform_init", 20) == 0)
                continue;
        }
        if (np == max_picks) {
            // Never drop a definition silently: the lifted module would call
            // it and fail to link (or bind it to the wasm-side copy).
            fprintf(stderr, "tinyc: '%s' has more than %zu spliceable "
                    "definitions; raise the pick array size in driver.c\n",
                    path, max_picks);
            MLIR_SetArenaAllocator(ctx, saved_arena);
            MLIR_SetContextDefUseTracking(ctx, saved_def_use);
            return false;
        }
        picks[np++] = op;
    }
    MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
    if (np == 0) {
        fprintf(stderr, "tinyc: no spliceable definitions in '%s'\n", path);
        return false;
    }
    *n_picks = np;
    return true;
}

#ifdef TINYC_HAS_ELF
// ELF self-host helpers: compile platform/buddy/WASI sources as LP64 modules
// and splice globals + selected llvm.func ops into a lifted wasm->llvm module.
// macOS Mach-O keeps the simpler pick-based tinyc_compile_host_platform above.

typedef enum {
    TINYC_ELF_HELPER_PLATFORM,
    TINYC_ELF_HELPER_BUDDY,
    TINYC_ELF_HELPER_WASI,
} TinycElfHelperKind;

static bool driver_elf_is_host_platform_support_symbol(string s) {
    return (s.size == 6 && memcmp(s.str, "memcpy", 6) == 0) ||
           (s.size == 6 && memcmp(s.str, "memset", 6) == 0) ||
           (s.size == 18 && memcmp(s.str, "platform_heap_base", 18) == 0) ||
           (s.size == 18 && memcmp(s.str, "platform_heap_size", 18) == 0) ||
           (s.size == 18 && memcmp(s.str, "platform_heap_grow", 18) == 0);
}

static bool driver_elf_is_llvm_global_op(MLIR_Context *ctx, MLIR_OpHandle op) {
    return MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_MLIR_GLOBAL;
}

static bool driver_elf_llvm_op_is_internal(MLIR_Context *ctx, MLIR_OpHandle op) {
    MLIR_AttributeHandle linka = MLIR_GetOpAttr(ctx, op, MLIR_KEY_LLVM_LINKAGE);
    if (linka == MLIR_INVALID_HANDLE)
        linka = MLIR_GetOpAttr(ctx, op, MLIR_KEY_LINKAGE);
    if (linka != MLIR_INVALID_HANDLE) {
        string ls = MLIR_GetAttributeString(ctx, linka);
        const char *want = "#llvm.linkage<internal>";
        size_t wantn = strlen(want);
        if (ls.size == wantn && memcmp(ls.str, want, wantn) == 0)
            return true;
    }
    MLIR_AttributeHandle visa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_VISIBILITY);
    if (visa != MLIR_INVALID_HANDLE) {
        string vs = MLIR_GetAttributeString(ctx, visa);
        if (vs.size == 7 && memcmp(vs.str, "private", 7) == 0)
            return true;
    }
    return false;
}

static void driver_elf_erase_module_func_by_name(MLIR_Context *ctx, MLIR_OpHandle mod,
                                                 string name) {
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, mod, 0), 0);
    size_t no = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = no; i > 0; i--) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i - 1);
        if (!driver_llvm_func_has_body(ctx, op)) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME);
        if (sa == MLIR_INVALID_HANDLE) continue;
        string s = MLIR_GetAttributeString(ctx, sa);
        if (s.size == name.size && memcmp(s.str, name.str, name.size) == 0)
            MLIR_EraseOp(ctx, op);
    }
}

static void driver_elf_erase_module_global_by_name(MLIR_Context *ctx, MLIR_OpHandle mod,
                                                   string name) {
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, mod, 0), 0);
    size_t no = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = no; i > 0; i--) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i - 1);
        if (!driver_elf_is_llvm_global_op(ctx, op)) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME);
        if (sa == MLIR_INVALID_HANDLE) continue;
        string s = MLIR_GetAttributeString(ctx, sa);
        if (s.size == name.size && memcmp(s.str, name.str, name.size) == 0)
            MLIR_EraseOp(ctx, op);
    }
}

static void driver_elf_splice_module_globals(MLIR_Context *ctx, MLIR_OpHandle llvm_mod,
                                           MLIR_OpHandle src_mod) {
    if (src_mod == MLIR_INVALID_HANDLE) return;
    MLIR_BlockHandle sbody =
        MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, src_mod, 0), 0);
    MLIR_BlockHandle dbody =
        MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, llvm_mod, 0), 0);
    size_t no = MLIR_GetBlockNumOps(ctx, sbody);
    for (size_t i = no; i > 0; i--) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, sbody, i - 1);
        if (!driver_elf_is_llvm_global_op(ctx, op)) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME);
        if (sa != MLIR_INVALID_HANDLE)
            driver_elf_erase_module_global_by_name(ctx, llvm_mod,
                                                   MLIR_GetAttributeString(ctx, sa));
        MLIR_MoveOpToBlockEnd(ctx, op, dbody);
    }
}

static bool driver_elf_helper_func_should_splice(MLIR_Context *ctx, MLIR_OpHandle op,
                                                 TinycElfHelperKind kind) {
    if (!driver_llvm_func_has_body(ctx, op)) return false;
    MLIR_AttributeHandle sa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME);
    if (sa == MLIR_INVALID_HANDLE) return false;
    string s = MLIR_GetAttributeString(ctx, sa);
    if (kind == TINYC_ELF_HELPER_BUDDY || kind == TINYC_ELF_HELPER_WASI)
        return true;
    bool is_host = s.size >= 16 && memcmp(s.str, "__host_platform_", 16) == 0;
    if (is_host) return true;
    if (driver_elf_llvm_op_is_internal(ctx, op)) return true;
    if (driver_elf_is_host_platform_support_symbol(s)) return true;
    return false;
}

static void driver_elf_splice_module_funcs(MLIR_Context *ctx, MLIR_OpHandle llvm_mod,
                                           MLIR_OpHandle src_mod,
                                           TinycElfHelperKind kind) {
    if (src_mod == MLIR_INVALID_HANDLE) return;
    MLIR_BlockHandle sbody =
        MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, src_mod, 0), 0);
    MLIR_BlockHandle dbody =
        MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, llvm_mod, 0), 0);
    size_t no = MLIR_GetBlockNumOps(ctx, sbody);
    for (size_t i = no; i > 0; i--) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, sbody, i - 1);
        if (driver_elf_is_llvm_global_op(ctx, op)) continue;
        if (!driver_elf_helper_func_should_splice(ctx, op, kind)) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME);
        if (sa != MLIR_INVALID_HANDLE) {
            string sym = MLIR_GetAttributeString(ctx, sa);
            driver_elf_erase_module_func_by_name(ctx, llvm_mod, sym);
            driver_elf_erase_module_global_by_name(ctx, llvm_mod, sym);
        }
        MLIR_MoveOpToBlockEnd(ctx, op, dbody);
    }
}

static void driver_elf_splice_helper_mod(MLIR_Context *ctx, MLIR_OpHandle llvm_mod,
                                         MLIR_OpHandle helper_mod,
                                         TinycElfHelperKind kind) {
    if (helper_mod == MLIR_INVALID_HANDLE) return;
    driver_elf_splice_module_globals(ctx, llvm_mod, helper_mod);
    driver_elf_splice_module_funcs(ctx, llvm_mod, helper_mod, kind);
}

static bool driver_elf_lower_for_native_codegen(
        MLIR_Context *ctx, MLIR_OpHandle module,
        bool (*lower_fn)(MLIR_Context *, MLIR_OpHandle)) {
    bool saved_def_use = MLIR_HasContextDefUseTracking(ctx);
    MLIR_SetContextDefUseTracking(ctx, true);
    bool ok = lower_fn(ctx, module);
    MLIR_SetContextDefUseTracking(ctx, saved_def_use);
    return ok;
}

// Path of corec's base/buddy.c, taken from the same corec tree as the host
// platform file: <dir of platform_<os>.c>/../base/buddy.c. The platform file
// and the allocator it initialises must come from one checkout, and deriving
// the path (rather than assuming "corec/base/buddy.c" under the current
// directory) keeps `--from-wasm --emit=elf` working from any cwd.
static const char *driver_elf_buddy_source_path(Arena *arena,
                                                const char *host_platform_path) {
    const char *rel = "../base/buddy.c";
    size_t dir_len = 0;   // length of the directory prefix, incl. trailing '/'
    for (size_t i = 0; host_platform_path[i]; i++)
        if (host_platform_path[i] == '/') dir_len = i + 1;
    size_t rel_len = strlen(rel);
    char *out = (char *)arena_alloc(arena, dir_len + rel_len + 1);
    memcpy(out, host_platform_path, dir_len);
    memcpy(out + dir_len, rel, rel_len + 1);
    return out;
}

// Compile corec's platform_<os>.c (or buddy.c / WASI adapter) as LP64 into a
// standalone lowered MLIR module. A caller splices its globals and selected
// llvm.func ops into a lifted wasm->llvm module via driver_elf_splice_helper_mod.
// MUST be called BEFORE the wasm module is lifted: the optimization passes
// (mem2reg / lowering) mutate per-value def-use chains, and compiling this
// second module after the lifted one is already built corrupts its operands.
// Compiling first keeps it correct, exactly like a standalone compile.
static bool driver_elf_compile_helper(MLIR_Context *ctx, const char *path,
                                      string *include_dirs, size_t n_include_dirs,
                                      bool (*lower_fn)(MLIR_Context *, MLIR_OpHandle),
                                      TinycElfHelperKind kind,
                                      MLIR_OpHandle *out_mod) {
    *out_mod = MLIR_INVALID_HANDLE;
    Arena *saved_arena = MLIR_GetArenaAllocator(ctx);
    Arena *pmod_arena = arena_create(16 * 1024 * 1024);
    MLIR_SetArenaAllocator(ctx, pmod_arena);
    // The --from-wasm pipeline disables def-use tracking
    // (MLIR_SetContextDefUseTracking) to save memory, but mem2reg / lowering rewrite operands through
    // ReplaceAllUsesOfValue, which needs the per-value use lists. Re-enable
    // tracking just for this compile (only the platform funcs pay the cost),
    // then restore the caller's setting.
    bool saved_def_use = MLIR_HasContextDefUseTracking(ctx);
    MLIR_SetContextDefUseTracking(ctx, true);

    string defs[16]; size_t nd = 0;
    if (kind == TINYC_ELF_HELPER_WASI)
        defs[nd++] = str_from_cstr_view((char *)"PLATFORM_HOST_SHIM=1");
    if (kind == TINYC_ELF_HELPER_PLATFORM) {
        // platform_linux.c talks to the kernel through __builtin_syscall6, so
        // (unlike the Mach-O helper) there are no libc symbols to rename.
        defs[nd++] = str_from_cstr_view((char *)"PLATFORM_SKIP_ENTRY=1");
        defs[nd++] = str_from_cstr_view((char *)"PLATFORM_HOST_SHIM=1");
    }

    string src = tinyc_preprocess(pmod_arena, str_from_cstr_view((char *)path),
                                  include_dirs, n_include_dirs, defs, nd);
    if (tinyc_last_preprocess_errors() > 0) {
        fprintf(stderr, "tinyc: preprocessing failed for %s '%s'\n",
                kind == TINYC_ELF_HELPER_WASI ? "WASI adapter" :
                kind == TINYC_ELF_HELPER_BUDDY ? "allocator source" : "host platform",
                path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
        return false;
    }
    if (src.size > 0 && src.str[src.size - 1] == '\0') src.size -= 1;
    VecTcTok toks = tinyc_lex(pmod_arena, src);
    Program *prog = arena_new(pmod_arena, Program);
    *prog = (Program){0};
    if (tinyc_parse_into(pmod_arena, prog, toks, false) > 0) {
        fprintf(stderr, "tinyc: parse failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
        return false;
    }
    MLIR_OpHandle pmod = tinyc_emit_module(ctx, prog);
    if (tinyc_last_emit_errors() > 0 || pmod == MLIR_INVALID_HANDLE) {
        fprintf(stderr, "tinyc: emit failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
        return false;
    }
    if (!getenv("TINYC_NO_MEM2REG"))
        mlir_llvm_mem2reg(ctx, pmod);
    if (!lower_fn(ctx, pmod)) {
        fprintf(stderr, "tinyc: lowering failed for host platform '%s'\n", path);
        MLIR_SetArenaAllocator(ctx, saved_arena);
        MLIR_SetContextDefUseTracking(ctx, saved_def_use);
        return false;
    }
    *out_mod = pmod;

    MLIR_BlockHandle pbody = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, pmod, 0), 0);
    size_t pn = MLIR_GetBlockNumOps(ctx, pbody);
    size_t n_funcs = 0;
    bool has_globals = false;
    for (size_t i = 0; i < pn; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, pbody, i);
        if (driver_elf_is_llvm_global_op(ctx, op)) {
            has_globals = true;
            continue;
        }
        if (driver_elf_helper_func_should_splice(ctx, op, kind))
            n_funcs++;
    }
    MLIR_SetArenaAllocator(ctx, saved_arena);
    MLIR_SetContextDefUseTracking(ctx, saved_def_use);
    // Every helper must contribute functions; only the platform file may get
    // by on globals alone.
    bool globals_suffice = kind == TINYC_ELF_HELPER_PLATFORM && has_globals;
    if (n_funcs == 0 && !globals_suffice) {
        fprintf(stderr, "tinyc: no spliceable definitions in '%s'\n", path);
        *out_mod = MLIR_INVALID_HANDLE;
        return false;
    }
    return true;
}

// Full `--from-wasm --emit=elf` self-host pipeline: compile host helpers,
// lift wasmssa -> llvm, run the wasm-oriented opt passes, splice helpers,
// native-lower, and emit a static Linux/x86_64 ELF via mlir_llvm_to_elf.
// On success `*io_arena` is swapped to the late-llvm arena (early arena
// destroyed) and `out` receives malloc'd ELF bytes.
static void driver_run_llvm_opt_passes(MLIR_Context *ctx, MLIR_OpHandle llvm_mod) {
    MLIR_Pass p;
    if (!getenv("TINYC_NO_MEM2REG")) {
        uint64_t t_p = ts_now();
        p = MLIR_PassMem2Reg();
        MLIR_RunPasses(ctx, llvm_mod, &p, 1, NULL);
        ts_add(TS_MEM2REG, t_p);
    }
    if (!getenv("TINYC_NO_LOAD_CSE")) {
        uint64_t t_p = ts_now();
        p = MLIR_PassLoadCSE();
        MLIR_RunPasses(ctx, llvm_mod, &p, 1, NULL);
        ts_add(TS_LOAD_CSE, t_p);
    }
    {
        uint64_t t_p = ts_now();
        p = MLIR_PassArithGVN();
        MLIR_RunPasses(ctx, llvm_mod, &p, 1, NULL);
        ts_add(TS_ARITH_GVN, t_p);
    }
    {
        uint64_t t_p = ts_now();
        p = MLIR_PassDCE();
        MLIR_RunPasses(ctx, llvm_mod, &p, 1, NULL);
        ts_add(TS_DCE, t_p);
    }
}

static int driver_emit_from_wasm_elf(
        MLIR_Context *ctx,
        Arena **io_arena,
        MLIR_OpHandle ssa,
        const char *host_platform_path,
        const char *wasi_adapter_path,
        string *include_dirs,
        size_t n_include_dirs,
        bool (*lower_fn)(MLIR_Context *, MLIR_OpHandle),
        string *out) {
    Arena *arena = *io_arena;

    // The via-wasm ELF backend's WASI adapters call corec's platform_<os>.c
    // (as __host_platform_*), so the path to it is required.
    if (!host_platform_path) {
        fprintf(stderr,
            "tinyc --from-wasm --emit=elf: --host-platform=PATH "
            "is required (e.g. corec/platform/platform_linux.c)\n");
        return 1;
    }
    // Compile the host platform file FIRST, before the lift:
    // its funcs are spliced into the lifted module after the opt passes.
    MLIR_OpHandle hp_mod = MLIR_INVALID_HANDLE;
    uint64_t t_hh = ts_now();
    if (!driver_elf_compile_helper(ctx, host_platform_path,
                                   include_dirs, n_include_dirs,
                                   lower_fn, TINYC_ELF_HELPER_PLATFORM,
                                   &hp_mod)) {
        return 1;
    }
    MLIR_OpHandle buddy_mod = MLIR_INVALID_HANDLE;
    if (!driver_elf_compile_helper(ctx,
                                   driver_elf_buddy_source_path(arena, host_platform_path),
                                   include_dirs, n_include_dirs,
                                   lower_fn, TINYC_ELF_HELPER_BUDDY,
                                   &buddy_mod)) {
        return 1;
    }
    // Compile + splice the C WASI adapter (fd_write/path_open/...) the same way,
    // when provided; otherwise the lifter's synthesised shims are used.
    MLIR_OpHandle ad_mod = MLIR_INVALID_HANDLE;
    if (wasi_adapter_path &&
        !driver_elf_compile_helper(ctx, wasi_adapter_path,
                                   include_dirs, n_include_dirs,
                                   lower_fn, TINYC_ELF_HELPER_WASI,
                                   &ad_mod)) {
        return 1;
    }
    // Lift wasmssa to the in-house `llvm` dialect, then stream it straight to
    // ELF one function at a time (low peak memory). Move the IR into a fresh
    // arena, then free the wasmstack/wasmssa arena.
    ts_add(TS_HOST_HELPERS, t_hh);
    Arena *late_arena = arena_create(64 * 1024 * 1024 - 64 * 1024);
    MLIR_SetArenaAllocator(ctx, late_arena);
    uint8_t *elf_data = NULL; size_t elf_size = 0;
    uint64_t t_p = ts_now();
    MLIR_OpHandle llvm_mod = mlir_wasmssa_to_llvm(ctx, ssa);
    ts_add(TS_SSA_TO_LLVM, t_p);
    if (llvm_mod == MLIR_INVALID_HANDLE) {
        arena_destroy(late_arena);
        return 1;
    }
    driver_run_llvm_opt_passes(ctx, llvm_mod);
    // Splice pre-compiled host helpers (globals + funcs).
    t_p = ts_now();
    driver_elf_splice_helper_mod(ctx, llvm_mod, hp_mod, TINYC_ELF_HELPER_PLATFORM);
    driver_elf_splice_helper_mod(ctx, llvm_mod, buddy_mod, TINYC_ELF_HELPER_BUDDY);
    driver_elf_splice_helper_mod(ctx, llvm_mod, ad_mod, TINYC_ELF_HELPER_WASI);
    ts_add(TS_HOST_HELPERS, t_p);
    t_p = ts_now();
    if (!driver_elf_lower_for_native_codegen(ctx, llvm_mod, lower_fn)) {
        arena_destroy(late_arena);
        return 1;
    }
    ts_add(TS_LOWER, t_p);
    arena_destroy(arena);
    *io_arena = late_arena;
    if (!mlir_llvm_to_elf(ctx, llvm_mod, true, &elf_data, &elf_size)) {
        return 1;
    }
    out->str = (char *)elf_data;
    out->size = elf_size;
    return 0;
}

static int driver_emit_native_elf(MLIR_Context *ctx, MLIR_OpHandle module,
                                  string *out) {
    uint8_t *elf_data = NULL; size_t elf_size = 0;
    if (!mlir_llvm_to_elf(ctx, module, false, &elf_data, &elf_size))
        return 1;
    out->str = (char *)elf_data;
    out->size = elf_size;
    return 0;
}

static int driver_emit_native_x64(MLIR_Context *ctx, MLIR_OpHandle module,
                                  string *out) {
    uint8_t *code = NULL; size_t code_size = 0;
    if (!mlir_llvm_to_x64_code(ctx, module, &code, &code_size))
        return 1;
    out->str = (char *)code;
    out->size = code_size;
    return 0;
}
#else
static void driver_run_llvm_opt_passes(MLIR_Context *ctx, MLIR_OpHandle llvm_mod) {
    MLIR_Pass p;
    if (!getenv("TINYC_NO_MEM2REG")) {
        uint64_t t_p = ts_now();
        p = MLIR_PassMem2Reg();
        MLIR_RunPasses(ctx, llvm_mod, &p, 1, NULL);
        ts_add(TS_MEM2REG, t_p);
    }
    if (!getenv("TINYC_NO_LOAD_CSE")) {
        uint64_t t_p = ts_now();
        p = MLIR_PassLoadCSE();
        MLIR_RunPasses(ctx, llvm_mod, &p, 1, NULL);
        ts_add(TS_LOAD_CSE, t_p);
    }
    {
        uint64_t t_p = ts_now();
        p = MLIR_PassArithGVN();
        MLIR_RunPasses(ctx, llvm_mod, &p, 1, NULL);
        ts_add(TS_ARITH_GVN, t_p);
    }
    {
        uint64_t t_p = ts_now();
        p = MLIR_PassDCE();
        MLIR_RunPasses(ctx, llvm_mod, &p, 1, NULL);
        ts_add(TS_DCE, t_p);
    }
}

static int driver_emit_from_wasm_elf(
        MLIR_Context *ctx,
        Arena **io_arena,
        MLIR_OpHandle ssa,
        const char *host_platform_path,
        const char *wasi_adapter_path,
        string *include_dirs,
        size_t n_include_dirs,
        bool (*lower_fn)(MLIR_Context *, MLIR_OpHandle),
        string *out) {
    (void)ctx;
    (void)io_arena;
    (void)ssa;
    (void)host_platform_path;
    (void)wasi_adapter_path;
    (void)include_dirs;
    (void)n_include_dirs;
    (void)lower_fn;
    (void)out;
    fprintf(stderr,
            "tinyc: --emit=elf is not supported in this build "
            "(rebuild with TINYC_ENABLE_ELF=1 or on Linux/x86_64)\n");
    return 1;
}

static int driver_emit_native_elf(MLIR_Context *ctx, MLIR_OpHandle module,
                                  string *out) {
    (void)ctx;
    (void)module;
    (void)out;
    fprintf(stderr,
            "tinyc: --emit=elf is not supported in this build "
            "(rebuild with TINYC_ENABLE_ELF=1 or on Linux/x86_64)\n");
    return 1;
}

static int driver_emit_native_x64(MLIR_Context *ctx, MLIR_OpHandle module,
                                  string *out) {
    (void)ctx;
    (void)module;
    (void)out;
    fprintf(stderr,
            "tinyc: --emit=x64 is not supported in this build "
            "(rebuild with TINYC_ENABLE_ELF=1 or on Linux/x86_64)\n");
    return 1;
}
#endif  // TINYC_HAS_ELF

// The target the native `--emit=llvm` output is for, from `--target TRIPLE`.
// The IR goes to the host's llc, so nothing in the IR itself says which
// machine it is for; the driver must be told (or know) the target to make
// target-dependent choices such as defining @__tinyc_syscall6 below.
//
// TRIPLE is an LLVM triple, ARCH-VENDOR-OS[-ENV], read by position the way
// llc reads it, e.g. x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu,
// aarch64-apple-darwin, arm64-apple-macosx14.0 or x86_64-pc-windows-msvc. The
// vendor and the environment are not interpreted. An explicit triple is also
// written to the IR as `target triple`, so llc compiles for the same target.
//
// Without --target there is no triple line. A gcc/clang-built tinyc on
// Linux/x86_64 then still targets that host (for the definitions below); every
// other build, including a self-hosted tinyc (__TINYC__), cannot know where its
// output will run, so its target is unknown and gets no target-specific code.
// (The enumerators are TINYC_-prefixed: Apple clang predefines TARGET_OS_*.)
typedef enum {
    TINYC_ARCH_UNKNOWN,
    TINYC_ARCH_X86_64,
    TINYC_ARCH_AARCH64
} TargetArch;

typedef enum {
    TINYC_OS_UNKNOWN,
    TINYC_OS_LINUX,
    TINYC_OS_DARWIN,
    TINYC_OS_WINDOWS
} TargetOS;

typedef struct {
    TargetArch arch;
    TargetOS os;
    const char *triple;  // written to the IR; NULL without --target
} Target;

static void target_set_default(Target *t) {
    t->arch = TINYC_ARCH_UNKNOWN;
    t->os = TINYC_OS_UNKNOWN;
    t->triple = NULL;
#if defined(__linux__) && defined(__x86_64__) && !defined(__TINYC__)
    t->arch = TINYC_ARCH_X86_64;
    t->os = TINYC_OS_LINUX;
#endif
}

// Is the triple component s[0..n) `name`, optionally followed by a version
// (digits and dots) when `versioned` (darwin23.1.0, macosx14.0)?
static bool target_component_is(const char *s, size_t n, const char *name,
                                bool versioned) {
    size_t len = strlen(name);
    if (n < len || memcmp(s, name, len) != 0) return false;
    if (n > len && !versioned) return false;
    for (size_t i = len; i < n; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || s[i] == '.')) return false;
    return true;
}

// Parse an LLVM triple into `t`. Returns false (with `t` untouched) when it
// has fewer than three components or its arch or OS is not one tinyC knows.
static bool target_parse(const char *triple, Target *t) {
    TargetArch arch = TINYC_ARCH_UNKNOWN;
    TargetOS os = TINYC_OS_UNKNOWN;
    size_t i = 0;
    int k = 0;
    while (k < 3) {
        size_t start = i;
        while (triple[i] && triple[i] != '-') i++;
        const char *c = triple + start;
        size_t n = i - start;
        if (k == 0) {
            if (target_component_is(c, n, "x86_64", false) ||
                target_component_is(c, n, "amd64", false))
                arch = TINYC_ARCH_X86_64;
            else if (target_component_is(c, n, "aarch64", false) ||
                     target_component_is(c, n, "arm64", false))
                arch = TINYC_ARCH_AARCH64;
        } else if (k == 2) {
            if (target_component_is(c, n, "linux", false))
                os = TINYC_OS_LINUX;
            else if (target_component_is(c, n, "darwin", true) ||
                     target_component_is(c, n, "macos", true) ||
                     target_component_is(c, n, "macosx", true))
                os = TINYC_OS_DARWIN;
            else if (target_component_is(c, n, "windows", false) ||
                     target_component_is(c, n, "win32", false))
                os = TINYC_OS_WINDOWS;
        }
        k++;
        if (!triple[i]) break;
        i++;
    }
    if (k < 3 || arch == TINYC_ARCH_UNKNOWN || os == TINYC_OS_UNKNOWN)
        return false;
    t->arch = arch;
    t->os = os;
    t->triple = triple;
    return true;
}

// Write `target triple = "..."` into the LLVM IR, after the
// `source_filename` line where LLVM's own printer puts it (or at the top if
// there is none). Both LLVM-IR translators start their output the same way, so
// they stay byte-identical.
static string inject_target_triple(Arena *arena, string ir, const char *triple) {
    const char *key = "source_filename";
    uint64_t kl = strlen(key);
    uint64_t pos = 0;
    for (uint64_t i = 0; i + kl <= ir.size; i++) {
        if ((i == 0 || ir.str[i - 1] == '\n') &&
            memcmp(ir.str + i, key, kl) == 0) {
            pos = i;
            while (pos < ir.size && ir.str[pos] != '\n') pos++;
            if (pos < ir.size) pos++;
            break;
        }
    }
    strbuf sb = strbuf_make_cap(arena, ir.size + 64);
    strbuf_append_bytes(arena, &sb, ir.str, pos);
    strbuf_append_cstr(arena, &sb, "target triple = \"");
    strbuf_append_cstr(arena, &sb, triple);
    strbuf_append_cstr(arena, &sb, "\"\n");
    strbuf_append_bytes(arena, &sb, ir.str + pos, ir.size - pos);
    return strbuf_to_string(sb);
}

// On the native `--emit=llvm` -> llc -> host-link path, the raw-syscall
// intrinsic __builtin_syscall6 lowers to a call to @__tinyc_syscall6. The
// in-tree x86_64 ELF backend synthesises that symbol as a `syscall` trap, but
// llc leaves it undefined, so corec/platform/platform_linux.c (which routes
// every raw syscall through it) would fail to link on the host toolchain.
// Define it here as an x86_64 `syscall` via LLVM inline asm — the same
// SysV->kernel register marshalling the clang/gcc polyfill in
// corec/platform/syscall6.h uses, preserving the raw (negative-errno) return
// convention platform_linux.c relies on. Both LLVM-IR translators (in-tree and
// upstream) emit the identical declaration line, so one textual rewrite covers
// both. Only an x86_64-Linux target gets it (macOS/Windows use the libc/WinAPI
// platforms; the ELF backend keeps its own thunk).
static string inject_syscall6_definition(Arena *arena, string ir) {
    const char *decl =
        "declare i64 @__tinyc_syscall6(i64, i64, i64, i64, i64, i64, i64)";
    uint64_t dl = strlen(decl);
    uint64_t pos = ir.size;
    for (uint64_t i = 0; i + dl <= ir.size; i++) {
        if (memcmp(ir.str + i, decl, dl) == 0) { pos = i; break; }
    }
    if (pos == ir.size) return ir;  // intrinsic unused; nothing to inject
    const char *def =
        "define internal i64 @__tinyc_syscall6(i64 %n, i64 %a1, i64 %a2, "
        "i64 %a3, i64 %a4, i64 %a5, i64 %a6) {\n"
        "  %r = call i64 asm sideeffect \"syscall\", "
        "\"={rax},{rax},{rdi},{rsi},{rdx},{r10},{r8},{r9},~{rcx},~{r11},~{memory}\""
        "(i64 %n, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6)\n"
        "  ret i64 %r\n"
        "}";
    strbuf sb = strbuf_make_cap(arena, ir.size + 256);
    strbuf_append_bytes(arena, &sb, ir.str, pos);
    strbuf_append_cstr(arena, &sb, def);
    strbuf_append_bytes(arena, &sb, ir.str + pos + dl, ir.size - pos - dl);
    return strbuf_to_string(sb);
}

// Output mode selected by `--emit=`; the last one on the command line wins.
// emit_mode_name() drives both the option parser and the usage text, so the
// two cannot disagree.
typedef enum {
    EMIT_MLIR,
    EMIT_LOWERED,
    EMIT_LLVM,
    EMIT_WASM,
    EMIT_WASMSSA,
    EMIT_WASMSTACK,
    EMIT_WAT,
    // `--emit=aarch64` stops the unified llvm -> aarch64 -> macho backend
    // at the physical-register `aarch64` dialect and prints it (the
    // "MachO MLIR"), instead of encoding the final Mach-O bytes. It is a
    // debugging view of the same pipeline `--emit=macho --macho-backend=llvm`
    // drives, so `--emit=mlir`, `--emit=llvm`, `--emit=aarch64` give the
    // three successive IRs for the native llvm backend.
    EMIT_AARCH64,
    EMIT_MACHO,
    // `--emit=elf` lowers the (native, LP64) `llvm` dialect through the flat
    // (scf->cf) standard lowering and the in-tree x86_64 code generator to a
    // statically-linked Linux ELF executable (direct syscalls, no libc).
    EMIT_ELF,
    // `--emit=x64` runs the same x86_64 code generator as `--emit=elf` on
    // every defined function of the input (no `main` needed, no linking) and
    // writes the raw concatenated machine code: a per-translation-unit
    // compile for benchmarking until the backend can write object files.
    EMIT_X64,
    EMIT_N_MODES
} EmitMode;

// The `--emit=` spelling of each mode. (A switch rather than a table of
// string literals: tinyC, which compiles this file when self-hosting, does
// not support pointer initializers in global arrays.)
static const char *emit_mode_name(EmitMode m) {
    switch (m) {
        case EMIT_MLIR:      return "mlir";
        case EMIT_LOWERED:   return "lowered";
        case EMIT_LLVM:      return "llvm";
        case EMIT_WASM:      return "wasm";
        case EMIT_WASMSSA:   return "wasmssa";
        case EMIT_WASMSTACK: return "wasmstack";
        case EMIT_WAT:       return "wat";
        case EMIT_AARCH64:   return "aarch64";
        case EMIT_MACHO:     return "macho";
        case EMIT_ELF:       return "elf";
        case EMIT_X64:       return "x64";
        default:             return "";
    }
}

static void print_usage(Arena *arena) {
    strbuf sb = strbuf_make_cap(arena, 256);
    strbuf_append_cstr(arena, &sb, "usage: tinyc [--emit=");
    for (int m = 0; m < EMIT_N_MODES; m++) {
        if (m > 0) strbuf_append_cstr(arena, &sb, "|");
        strbuf_append_cstr(arena, &sb, emit_mode_name((EmitMode)m));
    }
    strbuf_append_cstr(arena, &sb,
        "] [--lowering=upstream|native] [--target=TRIPLE] [--from-wasm PATH] "
        "[--wasm-runtime-obj=PATH ...] [-I dir ...] [-D name[=value] ...] "
        "[--time-stages] [-o OUT] FILE.tc [FILE2.tc ...]");
    println(str_lit("{}"), strbuf_to_string(sb));
}

static int tinyc_main(void) {
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
    EmitMode emit_mode = EMIT_MLIR;
    // --macho-backend selects which pipeline produces the final Mach-O
    // binary. "wasm" (default) keeps the established wasm-link +
    // wasm->macho path. "llvm" routes the `llvm` dialect directly through
    // the unified llvm -> aarch64 -> macho backend.
    bool macho_backend_llvm = false;
    bool host_varargs = false;
    // `--lowering=upstream` switches the four lowering / translation calls
    // below to the *Upstream variants (which run upstream MLIR's pass
    // pipeline / translator / LLVM target machine). Only available in the
    // upstream-backed tinyc binary, which is the only build that defines
    // TINYC_HAS_UPSTREAM and links the *Upstream implementations.
    typedef bool (*LowerFn)(MLIR_Context *, MLIR_OpHandle);
    typedef string (*TranslateFn)(MLIR_Context *, MLIR_OpHandle);
    LowerFn lower_fn;
    TranslateFn translate_to_llvm_fn;
    TranslateFn translate_to_wasm_fn;
#ifdef TINYC_HAS_UPSTREAM
    bool use_upstream = true;       // default in the upstream build
    bool lowering_explicit = false; // tracks whether user passed --lowering=
#endif
    char *output_file = NULL;
    // --target TRIPLE (see Target above); only `--emit=llvm` consults it.
    Target target;
    target_set_default(&target);

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
        if (strcmp(argv[i], "--time-stages") == 0) {
            g_ts_on = true;
#ifdef TINYC_HAS_ELF
            mlir_llvm_to_x64_set_stage_times(&g_ts_x64);
#endif
        }
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--version") == 0) {
            // Names the MLIR API implementation this binary is linked
            // against; the test runner asks for it (run_tinyc_tests.py).
#ifdef TINYC_HAS_UPSTREAM
            println(str_lit("tinyc (MLIR API: upstream)"));
#else
            println(str_lit("tinyc (MLIR API: native)"));
#endif
            arena_destroy(boot_arena);
            return 0;
        }
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
            uint64_t t_rd = ts_now();
            for (size_t k = 0; k < n_link_inputs; k++) {
                string buf = read_file_ok(boot_arena, str_from_cstr_view((char *)link_inputs[k]));
                ins[k].data = (const uint8_t *)buf.str;
                // read_file_ok stores filesize+1 (it appends a NUL).
                // The linker needs the exact file size.
                ins[k].size = buf.size > 0 ? buf.size - 1 : 0;
                ins[k].name = link_inputs[k];
            }
            ts_add(TS_WASM_READ, t_rd);
            uint8_t *out_data = NULL; size_t out_size = 0;
            uint64_t t_ln = ts_now();
            if (!MLIR_WasmLink(ins, n_link_inputs, link_entry, &out_data, &out_size)) {
                fprintf(stderr, "tinyc --link: link failed\n");
                arena_destroy(boot_arena);
                return 1;
            }
            ts_add(TS_LINK_WASM, t_ln);
            if (link_emit_macho) {
                uint8_t *macho_data = NULL; size_t macho_size = 0;
                uint64_t t_cg = ts_now();
                bool macho_ok = MLIR_WasmToMachoArm64(out_data, out_size,
                                                     &macho_data, &macho_size);
                ts_add(TS_CODEGEN, t_cg);
                free(out_data);
                if (!macho_ok) {
                    fprintf(stderr, "tinyc --link --emit=macho: wasm->macho translation failed\n");
                    arena_destroy(boot_arena);
                    return 1;
                }
                string macho_buf = { .str = (char *)macho_data, .size = macho_size };
                uint64_t t_wr = ts_now();
                int wrc = write_bytes_to_file(macho_buf, link_out);
                ts_add(TS_WRITE, t_wr);
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
            uint64_t t_wr = ts_now();
            int wrc = write_bytes_to_file(out_buf, link_out);
            ts_add(TS_WRITE, t_wr);
            free(out_data);
            arena_destroy(boot_arena);
            return wrc;
        }
        if (strncmp(argv[i], "--emit=", 7) == 0) {
            int m = 0;
            while (m < EMIT_N_MODES && strcmp(argv[i] + 7, emit_mode_name((EmitMode)m)) != 0) m++;
            if (m == EMIT_N_MODES) {
                fprintf(stderr, "tinyc: unknown output mode '%s'\n", argv[i]);
                print_usage(boot_arena);
                arena_destroy(boot_arena);
                return 1;
            }
            emit_mode = (EmitMode)m;
        }
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
        else if (strncmp(argv[i], "--target", 8) == 0 &&
                 (argv[i][8] == '=' || argv[i][8] == '\0')) {
            const char *triple = argv[i][8] == '=' ? argv[i] + 9
                               : (i + 1 < argc ? argv[++i] : NULL);
            if (!triple) {
                fprintf(stderr, "tinyc: --target needs a triple, e.g. "
                        "--target=x86_64-unknown-linux-gnu\n");
                arena_destroy(boot_arena);
                return 1;
            }
            if (!target_parse(triple, &target)) {
                fprintf(stderr,
                        "tinyc: unsupported target '%s' (expected "
                        "ARCH-VENDOR-OS[-ENV] with ARCH x86_64|aarch64 and OS "
                        "linux|darwin|macosx|windows, e.g. "
                        "x86_64-unknown-linux-gnu)\n", triple);
                arena_destroy(boot_arena);
                return 1;
            }
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
        else if (strcmp(argv[i], "--host-varargs") == 0) {
            host_varargs = true;
        }
        else if (strcmp(argv[i], "--time-stages") == 0) {
            // Handled by the pre-scan above.
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
        print_usage(boot_arena);
        arena_destroy(boot_arena);
        return 1;
    }

    bool emit_lowered   = emit_mode == EMIT_LOWERED;
    bool emit_llvm      = emit_mode == EMIT_LLVM;
    bool emit_wasm      = emit_mode == EMIT_WASM;
    bool emit_wasmssa   = emit_mode == EMIT_WASMSSA;
    bool emit_wasmstack = emit_mode == EMIT_WASMSTACK;
    bool emit_wat       = emit_mode == EMIT_WAT;
    bool emit_aarch64   = emit_mode == EMIT_AARCH64;
    bool emit_macho     = emit_mode == EMIT_MACHO;
    bool emit_elf       = emit_mode == EMIT_ELF;
    bool emit_x64       = emit_mode == EMIT_X64;

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
        translate_to_llvm_fn = MLIR_TranslateModuleToLLVMIRUpstream;
        translate_to_wasm_fn = MLIR_TranslateModuleToWasmUpstream;
    } else
#endif
    {
        lower_fn = MLIR_LowerToLLVMDialect;
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
    MLIR_ContextOptions ctx_options = {0};
    ctx_options.arena = arena;
    MLIR_Context *ctx = MLIR_CreateContext(&ctx_options);

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
        MLIR_SetContextDefUseTracking(ctx, false);
        uint64_t t_rd = ts_now();
        string wasm_buf = read_file_ok(arena, str_from_cstr_view(from_wasm_path));
        ts_add(TS_WASM_READ, t_rd);
        // read_file_ok appends a trailing NUL — strip it for the lifter,
        // which expects the exact wasm file size.
        size_t wasm_size = wasm_buf.size > 0 ? wasm_buf.size - 1 : 0;
        // The lift's wasmstack, wasmssa and llvm IR come to about 1.7 ops
        // per byte of wasm: reserve the native IR storage for that.
        MLIR_SetContextExpectedOps(ctx, wasm_size + wasm_size * 7 / 10);
        uint64_t t_ws = ts_now();
        MLIR_OpHandle stack_mod = mlir_wasm_to_wasmstack(
            ctx, (const uint8_t *)wasm_buf.str, wasm_size);
        ts_add(TS_WASM_TO_STACK, t_ws);
        if (stack_mod == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        string out_fw;
        if (emit_wasmstack) {
            uint64_t t_pr = ts_now();
            out_fw = MLIR_PrintOperationGeneric(ctx, stack_mod);
            ts_add(TS_CODEGEN, t_pr);
        } else {
            uint64_t t_ss = ts_now();
            MLIR_OpHandle ssa = mlir_wasmstack_to_wasmssa(ctx, stack_mod);
            ts_add(TS_STACK_TO_SSA, t_ss);
            if (ssa == MLIR_INVALID_HANDLE) {
                arena_destroy(arena);
                MLIR_DestroyContext(ctx);
                arena_destroy(boot_arena);
                return 1;
            }
            if (emit_wasmssa) {
                uint64_t t_pr = ts_now();
                out_fw = MLIR_PrintOperationGeneric(ctx, ssa);
                ts_add(TS_CODEGEN, t_pr);
            } else if (emit_macho) {
                // The via-wasm Mach-O backend's WASI adapters call corec's
                // platform_<os>.c (as __host_platform_*), so the path to it is
                // required.
                if (!host_platform_path) {
                    fprintf(stderr,
                        "tinyc --from-wasm --emit=macho: --host-platform=PATH "
                        "is required (e.g. corec/platform/platform_macos.c)\n");
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
                // Compile the host platform file FIRST, before the lift:
                // its funcs are spliced into the lifted module
                // after the opt passes (see tinyc_compile_host_platform).
                MLIR_OpHandle hp_picks[16]; size_t hp_n = 0;
                uint64_t t_hh = ts_now();
                if (!tinyc_compile_host_platform(ctx, host_platform_path,
                                                 include_dirs, n_include_dirs,
                                                 lower_fn, false,
                                                 hp_picks, &hp_n, 16)) {
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
                // Compile + splice the C WASI adapter (fd_write/path_open/...)
                // the same way, when provided; otherwise the lifter's
                // synthesised shims are used. Also compiled before the lift.
                MLIR_OpHandle ad_picks[16]; size_t ad_n = 0;
                if (wasi_adapter_path &&
                    !tinyc_compile_host_platform(ctx, wasi_adapter_path,
                                                 include_dirs, n_include_dirs,
                                                 lower_fn, true,
                                                 ad_picks, &ad_n, 16)) {
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
                ts_add(TS_HOST_HELPERS, t_hh);
                // Lift wasmssa to the in-house `llvm` dialect, then stream it
                // straight to Mach-O one function at a time (low peak memory).
                // Move the IR into a fresh arena, then free the
                // wasmstack/wasmssa arena. The lifter rebuilds every
                // attribute/location/value fresh (see normalise_carrier),
                // and types live in the context, so the produced module
                // references nothing in the old arena and it can be released.
                // This keeps the early modules from coexisting with the
                // (larger) llvm module, cutting peak RSS by roughly the size
                // of wasmstack + wasmssa.
                Arena *late_arena = arena_create(64 * 1024 * 1024 - 64 * 1024);
                MLIR_SetArenaAllocator(ctx, late_arena);
                uint8_t *macho_data = NULL; size_t macho_size = 0;
                uint64_t t_sl = ts_now();
                MLIR_OpHandle llvm_mod = mlir_wasmssa_to_llvm(ctx, ssa);
                ts_add(TS_SSA_TO_LLVM, t_sl);
                if (llvm_mod == MLIR_INVALID_HANDLE) {
                    arena_destroy(late_arena);
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
                driver_run_llvm_opt_passes(ctx, llvm_mod);
                // Splice the pre-compiled host platform funcs into the lifted
                // module now (after the wasm-oriented opt passes, which assume
                // linmem shapes and must not see the LP64 platform funcs). The
                // funcs were compiled before the lift; moving them is a
                // pure re-parent that preserves their operands.
                if (hp_n > 0 || ad_n > 0) {
                    MLIR_BlockHandle dbody =
                        MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, llvm_mod, 0), 0);
                    for (size_t hi = 0; hi < hp_n; hi++)
                        MLIR_MoveOpToBlockEnd(ctx, hp_picks[hi], dbody);
                    for (size_t hi = 0; hi < ad_n; hi++)
                        MLIR_MoveOpToBlockEnd(ctx, ad_picks[hi], dbody);
                }
                arena_destroy(arena);
                arena = late_arena;
                uint64_t t_cg = ts_now();
                if (!mlir_llvm_to_macho(ctx, llvm_mod,
                                        &macho_data, &macho_size)) {
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
                ts_add(TS_CODEGEN, t_cg);
                out_fw.str = (char *)macho_data;
                out_fw.size = macho_size;
            } else if (emit_elf) {
                if (driver_emit_from_wasm_elf(ctx, &arena, ssa,
                                              host_platform_path, wasi_adapter_path,
                                              include_dirs, n_include_dirs,
                                              lower_fn, &out_fw) != 0) {
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
            } else if (emit_aarch64) {
                // Same wasmssa -> llvm lift + optimization passes as the
                // macho path above, but stop at the physical-register
                // `aarch64` dialect and print it (the "MachO MLIR"). No
                // late-arena swap: this is a debug view, not the memory-
                // constrained self-host emit.
                MLIR_OpHandle hp_picks2[16]; size_t hp_n2 = 0;
                if (host_platform_path &&
                    !tinyc_compile_host_platform(ctx, host_platform_path,
                                                 include_dirs, n_include_dirs,
                                                 lower_fn, false,
                                                 hp_picks2, &hp_n2, 16)) {
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
                MLIR_OpHandle llvm_mod = mlir_wasmssa_to_llvm(ctx, ssa);
                if (llvm_mod == MLIR_INVALID_HANDLE) {
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
                driver_run_llvm_opt_passes(ctx, llvm_mod);
                if (hp_n2 > 0) {
                    MLIR_BlockHandle dbody =
                        MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, llvm_mod, 0), 0);
                    for (size_t hi = 0; hi < hp_n2; hi++)
                        MLIR_MoveOpToBlockEnd(ctx, hp_picks2[hi], dbody);
                }
                uint64_t t_cg = ts_now();
                MLIR_OpHandle aarch64 = mlir_llvm_to_aarch64(ctx, llvm_mod);
                if (aarch64 == MLIR_INVALID_HANDLE) {
                    arena_destroy(arena);
                    MLIR_DestroyContext(ctx);
                    arena_destroy(boot_arena);
                    return 1;
                }
                out_fw = MLIR_PrintOperationGeneric(ctx, aarch64);
                ts_add(TS_CODEGEN, t_cg);
            } else {
                fprintf(stderr,
                    "tinyc --from-wasm: requires --emit=wasmstack|wasmssa|aarch64|macho|elf\n");
                arena_destroy(arena);
                MLIR_DestroyContext(ctx);
                arena_destroy(boot_arena);
                return 1;
            }
        }
        int wr;
        if (emit_macho || emit_elf) {
            if (!output_file) {
                fprintf(stderr, "tinyc --from-wasm --emit=%s: -o OUT required\n",
                        emit_elf ? "elf" : "macho");
                if (emit_elf) free(out_fw.str);
                arena_destroy(arena);
                MLIR_DestroyContext(ctx);
                arena_destroy(boot_arena);
                return 1;
            }
            uint64_t t_wr = ts_now();
            wr = write_bytes_to_file(out_fw, output_file);
#if !defined(_WIN32) && !defined(__wasm__) && !defined(__TINYC__)
            if (wr == 0) {
                chmod(output_file, 0755);
            }
#endif
            ts_add(TS_WRITE, t_wr);
            if (emit_elf) free(out_fw.str);
        } else if (output_file) {
            uint64_t t_wr = ts_now();
            wr = write_string_to_file(out_fw, output_file);
            ts_add(TS_WRITE, t_wr);
        } else {
            println(str_lit("{}"), out_fw);
            wr = 0;
        }
        arena_destroy(arena);
        MLIR_DestroyContext(ctx);
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
    prog->host_varargs = host_varargs;
    int total_errs = 0;
    size_t total_toks = 0;
    for (size_t k = 0; k < n_input_files; k++) {
        uint64_t t_pp = ts_now();
        VecTcTok toks = tinyc_preprocess_lex(arena, str_from_cstr_view(input_files[k]),
                                             include_dirs, n_include_dirs,
                                             defines, n_defines);
        ts_add(TS_PREPROCESS, t_pp);
        int pp_errs = tinyc_last_preprocess_errors();
        if (pp_errs > 0) {
            total_errs += pp_errs;
            continue;
        }
        total_toks += toks.size;
        uint64_t t_pa = ts_now();
        total_errs += tinyc_parse_into(arena, prog, toks, target_wasm32);
        ts_add(TS_PARSE, t_pa);
    }
    if (total_errs > 0) {
        arena_destroy(arena);
        MLIR_DestroyContext(ctx);
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
        uint64_t t_pp = ts_now();
        string psrc = tinyc_preprocess(arena, str_from_cstr_view(host_platform_path),
                                       include_dirs, n_include_dirs, pdefs, n_defines + 1);
        ts_add(TS_PREPROCESS, t_pp);
        if (tinyc_last_preprocess_errors() > 0) {
            fprintf(stderr, "tinyc: preprocessing failed for host platform '%s'\n",
                    host_platform_path);
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        if (psrc.size > 0 && psrc.str[psrc.size - 1] == '\0') psrc.size -= 1;
        uint64_t t_lx = ts_now();
        VecTcTok ptoks = tinyc_lex(arena, psrc);
        ts_add(TS_LEX, t_lx);
        total_toks += ptoks.size;
        uint64_t t_pa = ts_now();
        int perrs = tinyc_parse_into(arena, prog, ptoks, target_wasm32);
        ts_add(TS_PARSE, t_pa);
        if (perrs > 0) {
            fprintf(stderr, "tinyc: parse failed for host platform '%s'\n",
                    host_platform_path);
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
    }

    // Size the native backend's IR storage from the input. The module,
    // its lowering and the backends' IR come to at most about 1.5 ops per
    // token (after preprocessing; most header tokens make no IR), and with
    // the wasm pipeline's wasmssa and wasmstack IR to 3-6.4 (tinyc's own
    // sources). Too low costs doublings, too high wastes heap; the backend
    // caps the reservation, so a large input cannot exhaust its heap.
    MLIR_SetContextExpectedOps(ctx, target_wasm32 ? total_toks * 3 : total_toks / 3);
    uint64_t t_em = ts_now();
    MLIR_OpHandle module = tinyc_emit_module(ctx, prog);
    ts_add(TS_EMIT, t_em);
    if (tinyc_last_emit_errors() > 0) {
        arena_destroy(arena);
        MLIR_DestroyContext(ctx);
        arena_destroy(boot_arena);
        return 1;
    }

    // Promote non-escaping `llvm.alloca` locals to SSA (mem2reg) on the flat
    // cf CFG, before the lowering to the LLVM dialect. This keeps non-address-taken locals out of
    // the linear-memory shadow stack so the downstream register allocator can
    // hold them in registers instead of spilling every access. Skippable via
    // TINYC_NO_MEM2REG for A/B debugging.
    uint64_t t_m2r = ts_now();
    if (!getenv("TINYC_NO_MEM2REG"))
        mlir_llvm_mem2reg(ctx, module);
    ts_add(TS_MEM2REG, t_m2r);

    if (emit_lowered || emit_llvm || emit_wasm || emit_wasmssa || emit_wasmstack || emit_wat || emit_macho || emit_aarch64 || emit_elf || emit_x64) {
        // One lowering for every backend: the result is LLVM dialect only,
        // with control flow as a flat llvm.br / llvm.cond_br CFG. The wasm
        // backend rebuilds structured control flow from it internally.
        uint64_t t_lo = ts_now();
        if (!lower_fn(ctx, module)) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        ts_add(TS_LOWER, t_lo);
        if (getenv("TINYC_DUMP_LOWERED")) {
            string s = MLIR_PrintOperationGeneric(ctx, module);
            println(str_lit("{}"), s);
        }
    }
    string out;
    // Every output mode except elf / x64 (whose backend times its own phases)
    // counts as one `codegen` stage.
    uint64_t t_cg = ts_now();
    if (emit_wasmssa) {
        MLIR_OpHandle ssa = mlir_llvm_to_wasmssa(ctx, module);
        if (ssa == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        out = MLIR_PrintOperationGeneric(ctx, ssa);
    } else if (emit_wasmstack) {
        MLIR_OpHandle ssa = mlir_llvm_to_wasmssa(ctx, module);
        if (ssa == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        MLIR_OpHandle stk = mlir_wasmssa_to_wasmstack(ctx, ssa);
        if (stk == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        out = MLIR_PrintOperationGeneric(ctx, stk);
    } else if (emit_wat) {
        string bin = translate_to_wasm_fn(ctx, module);
        if (bin.size == 0) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        out = mlir_wasm_binary_to_wat(ctx, bin);
    } else if (emit_wasm) {
        out = translate_to_wasm_fn(ctx, module);
        if (out.size == 0) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
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
            MLIR_OpHandle aarch64 = mlir_llvm_to_aarch64(ctx, module);
            if (aarch64 == MLIR_INVALID_HANDLE) {
                arena_destroy(arena);
                MLIR_DestroyContext(ctx);
                arena_destroy(boot_arena);
                return 1;
            }
            uint8_t *macho_data = NULL; size_t macho_size = 0;
            if (!mlir_aarch64_to_macho(ctx, aarch64, &macho_data, &macho_size)) {
                arena_destroy(arena);
                MLIR_DestroyContext(ctx);
                arena_destroy(boot_arena);
                return 1;
            }
            out.str = (char *)macho_data;
            out.size = macho_size;
        } else {
            // Pipeline: lower to wasm32 object, link with the provided
            // --wasm-runtime-obj inputs, then translate the linked wasm
            // module to a signed Mach-O ARM64 binary.
            string obj = translate_to_wasm_fn(ctx, module);
            if (obj.size == 0) {
                arena_destroy(arena);
                MLIR_DestroyContext(ctx);
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
                MLIR_DestroyContext(ctx);
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
                MLIR_DestroyContext(ctx);
                arena_destroy(boot_arena);
                return 1;
            }
            out.str = (char *)macho_data;
            out.size = macho_size;
        }
    } else if (emit_aarch64) {
        // Stop the unified llvm -> aarch64 -> macho backend at the
        // physical-register `aarch64` dialect and print it (the "MachO MLIR").
        MLIR_OpHandle aarch64 = mlir_llvm_to_aarch64(ctx, module);
        if (aarch64 == MLIR_INVALID_HANDLE) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        out = MLIR_PrintOperationGeneric(ctx, aarch64);
    } else if (emit_elf) {
        if (driver_emit_native_elf(ctx, module, &out) != 0) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        t_cg = 0;
    } else if (emit_x64) {
        if (driver_emit_native_x64(ctx, module, &out) != 0) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        t_cg = 0;
    } else if (emit_llvm) {
        out = translate_to_llvm_fn(ctx, module);
        if (out.size == 0) {
            arena_destroy(arena);
            MLIR_DestroyContext(ctx);
            arena_destroy(boot_arena);
            return 1;
        }
        if (target.triple)
            out = inject_target_triple(arena, out, target.triple);
        if (target.arch == TINYC_ARCH_X86_64 && target.os == TINYC_OS_LINUX)
            out = inject_syscall6_definition(arena, out);
    } else {
        out = print_fn(ctx, module);
    }
    if (t_cg) ts_add(TS_CODEGEN, t_cg);
    int wrc = 0;
    if ((emit_macho || emit_elf || emit_x64) && !output_file) {
        fprintf(stderr, "tinyc --emit=%s: -o PATH is required\n",
                emit_mode_name(emit_mode));
        free(out.str);
        arena_destroy(arena);
        MLIR_DestroyContext(ctx);
        arena_destroy(boot_arena);
        return 1;
    }
    uint64_t t_wr = ts_now();
    if (output_file) {
        if (emit_wasm || emit_macho || emit_elf || emit_x64) {
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
        ts_add(TS_WRITE, t_wr);
    } else {
        println(str_lit("{}"), out);
        ts_add(TS_WRITE, t_wr);
    }

    if (emit_macho || emit_elf || emit_x64) {
        // out.str was malloc'd by the backend. Free it.
        free(out.str);
    }

    arena_destroy(arena);
    MLIR_DestroyContext(ctx);
    arena_destroy(boot_arena);
    return wrc;
}

int app_main(void) {
    g_ts_start = platform_clock_ns();
    int rc = tinyc_main();
    if (g_ts_on) {
#ifdef TINYC_HAS_ELF
        mlir_llvm_to_x64_set_stage_times(NULL);
#endif
        ts_report();
    }
    return rc;
}
