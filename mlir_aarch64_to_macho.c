// aarch64 (MLIR dialect) -> Mach-O ARM64 binary translator.
//
// First-light scope: lower a module whose top-level ops are
// aarch64.func (with a flat sequence of aarch64.movz / movk / mov_x /
// bl / svc / ret instruction ops inside) into a runnable Mach-O ARM64
// binary that does not require any libSystem stubs or GOT. The
// special function name `_start` is the program entry; it is placed
// first in the `__text` section so LC_MAIN.entryoff lands on it.
//
// The envelope mirrors the macho_exit reference layout used by
// mlir_wasm_to_macho.c (the n_stubs == 0 / no __DATA shape), so we
// know the load-command sequence and ad-hoc signature format are
// accepted by macOS / AMFI. The only difference from the existing
// backend is that the contents of __text are produced by walking the
// aarch64 dialect rather than the WASM bytecode.
//
// Everything outside the first-light op set returns `false` with a
// diagnostic; coverage grows as new aarch64.* ops are added.

#include "mlir_aarch64_to_macho.h"
#include "mlir_aarch64_asm.h"
#include "mlir_machine.h"
#include "mlir_llvm_to_aarch64.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"

// =============================================================================
// Growable byte buffer + endian helpers (shared) + SHA-256 (Mach-O code
// signature only). The Buf helpers live in mlir_codegen_buf.h so the aarch64
// assembler and the Mach-O container share one copy; SHA-256 stays here
// because only the Mach-O ad-hoc code signature needs it.
// =============================================================================
#include "mlir_codegen_buf.h"

#define SHA256_DIGEST_LEN 32
typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint32_t datalen;
    uint8_t  data[64];
} Sha256;
static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
static inline uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
static void sha256_xform(Sha256 *s, const uint8_t *d) {
    uint32_t a,b,c,e,f,g,h,t1,t2,m[64];
    uint32_t dd;
    for (int i = 0, j = 0; i < 16; i++, j += 4) {
        m[i] = ((uint32_t)(d[j]   & 0xFFu) << 24) | ((uint32_t)(d[j+1] & 0xFFu) << 16) |
               ((uint32_t)(d[j+2] & 0xFFu) << 8)  |  (uint32_t)(d[j+3] & 0xFFu);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(m[i-15], 7) ^ rotr32(m[i-15], 18) ^ (m[i-15] >> 3);
        uint32_t s1 = rotr32(m[i-2], 17) ^ rotr32(m[i-2], 19)  ^ (m[i-2] >> 10);
        m[i] = m[i-16] + s0 + m[i-7] + s1;
    }
    a = s->state[0]; b = s->state[1]; c = s->state[2]; dd = s->state[3];
    e = s->state[4]; f = s->state[5]; g = s->state[6]; h = s->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        t1 = h + S1 + ch + SHA256_K[i] + m[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        t2 = S0 + mj;
        h = g; g = f; f = e; e = dd + t1;
        dd = c; c = b; b = a; a = t1 + t2;
    }
    s->state[0] += a; s->state[1] += b; s->state[2] += c; s->state[3] += dd;
    s->state[4] += e; s->state[5] += f; s->state[6] += g; s->state[7] += h;
}
static void sha256(const uint8_t *data, size_t n, uint8_t out[32]) {
    Sha256 s;
    s.state[0] = 0x6a09e667; s.state[1] = 0xbb67ae85;
    s.state[2] = 0x3c6ef372; s.state[3] = 0xa54ff53a;
    s.state[4] = 0x510e527f; s.state[5] = 0x9b05688c;
    s.state[6] = 0x1f83d9ab; s.state[7] = 0x5be0cd19;
    s.bitlen = 0; s.datalen = 0;
    while (n) {
        size_t take = 64 - s.datalen;
        if (take > n) take = n;
        memcpy(s.data + s.datalen, data, take);
        s.datalen += (uint32_t)take; data += take; n -= take;
        if (s.datalen == 64) { sha256_xform(&s, s.data); s.bitlen += 512; s.datalen = 0; }
    }
    uint32_t i = s.datalen;
    if (s.datalen < 56) {
        s.data[i++] = 0x80;
        while (i < 56) s.data[i++] = 0;
    } else {
        s.data[i++] = 0x80;
        while (i < 64) s.data[i++] = 0;
        sha256_xform(&s, s.data);
        memset(s.data, 0, 56);
    }
    s.bitlen += (uint64_t)s.datalen * 8;
    for (int k = 7; k >= 0; k--) s.data[56 + (7 - k)] = (uint8_t)(s.bitlen >> (8 * k));
    sha256_xform(&s, s.data);
    for (int k = 0; k < 8; k++) {
        out[4*k + 0] = (uint8_t)(s.state[k] >> 24);
        out[4*k + 1] = (uint8_t)(s.state[k] >> 16);
        out[4*k + 2] = (uint8_t)(s.state[k] >> 8);
        out[4*k + 3] = (uint8_t)(s.state[k]);
    }
}

// =============================================================================
// libSystem stub table.
//
// macOS only commits to libSystem.B.dylib as a stable ABI; raw BSD
// syscalls (`svc #0x80`) are private/unstable. Every external call the
// backend wants to make therefore goes through a `bl _<name>` to a
// PC-relative stub in __TEXT,__stubs that does `ADRP x16, GOT_page;
// LDR x16, [x16, GOT_lo12]; BR x16`. dyld fills the corresponding
// __DATA_CONST,__got slot at load time via chained fixups.
//
// LibSysSym enumerates every libSystem symbol the backend may
// reference from its synth_* shims. The actual set used by any given
// program is discovered by walking BL relocs after function emission
// (see `n_libsys_stubs` further down). Enum order matters: it controls
// the dense stub index assigned to each symbol, which in turn
// determines the order of entries in __stubs / __got / chained-fixups
// imports / dysymtab indirect-syms.
//
// The lookup helpers are written as if/else cascades rather than as a
// `static const char *names[N] = {...};` table because tinyc (which
// must compile this file for selfhost) does not yet support string-
// literal initialisers in global arrays of `char *`, nor `sizeof(arr)`
// in a constant expression.
// =============================================================================
typedef enum {
    LS_EXIT  = 0,
    LS_WRITE,
    LS_READ,
    LS_OPEN,
    LS_CLOSE,
    LS_LSEEK,
    LS_ERRNO,
    LS_MMAP,
    LS_MEMCPY,
    LS_FCHMOD,
    LS_WRITEV,
    LS_READV,
    LS_FCNTL
} LibSysSym;
#define LS_COUNT 13

static const char *libsys_name(int sym) {
    if (sym == LS_EXIT)   return "_exit";
    if (sym == LS_WRITE)  return "_write";
    if (sym == LS_READ)   return "_read";
    if (sym == LS_OPEN)   return "_open";
    if (sym == LS_CLOSE)  return "_close";
    if (sym == LS_LSEEK)  return "_lseek";
    if (sym == LS_ERRNO)  return "___error";
    if (sym == LS_MMAP)   return "_mmap";
    if (sym == LS_MEMCPY) return "_memcpy";
    if (sym == LS_FCHMOD) return "_fchmod";
    if (sym == LS_WRITEV) return "_writev";
    if (sym == LS_READV)  return "_readv";
    if (sym == LS_FCNTL)  return "_fcntl";
    return "";
}

// Returns the LibSysSym value matching `callee`, or -1 if `callee` is
// not a known libSystem symbol.
static int libsys_lookup(string callee) {
    for (int i = 0; i < LS_COUNT; i++) {
        const char *nm = libsys_name(i);
        size_t kl = strlen(nm);
        if (callee.size == kl && memcmp(callee.str, nm, kl) == 0) {
            return i;
        }
    }
    return -1;
}

// Per-translation registry: which LibSysSym values are referenced, and
// in what dense order. Populated by a discovery pass before layout.
typedef struct {
    bool     used[LS_COUNT];
    uint32_t stub_index[LS_COUNT];   // dense index, valid iff used[i]
    uint32_t n_stubs;                 // total libSystem symbols used
} LibSysRegistry;

// =============================================================================
// Mach-O envelope constants (mirror mlir_wasm_to_macho.c).
// =============================================================================
#define MH_MAGIC_64            0xfeedfacfu
#define CPU_ARCH_ABI64         0x01000000u
#define CPU_TYPE_ARM           12u
#define CPU_TYPE_ARM64         (CPU_TYPE_ARM | CPU_ARCH_ABI64)

#define LC_REQ_DYLD            0x80000000u
#define LC_SEGMENT_64          0x19u
#define LC_SYMTAB              0x02u
#define LC_DYSYMTAB            0x0bu
#define LC_LOAD_DYLIB          0x0cu
#define LC_LOAD_DYLINKER       0x0eu
#define LC_UUID                0x1bu
#define LC_CODE_SIGNATURE      0x1du
#define LC_FUNCTION_STARTS     0x26u
#define LC_DATA_IN_CODE        0x29u
#define LC_SOURCE_VERSION      0x2au
#define LC_BUILD_VERSION       0x32u
#define LC_MAIN                (0x28u | LC_REQ_DYLD)
#define LC_DYLD_EXPORTS_TRIE   (0x33u | LC_REQ_DYLD)
#define LC_DYLD_CHAINED_FIXUPS (0x34u | LC_REQ_DYLD)

#define MH_EXECUTE             2u
#define MH_FLAGS_EXEC          2097285u

#define VM_PROT_READ           1u
#define VM_PROT_WRITE          2u
#define VM_PROT_EXECUTE        4u

#define TEXT_VM_BASE           0x100000000ULL
#define TEXT_FILE_BASE         0u
#define VMSEG_SIZE             0x4000u   // 16 KiB minimum mach-o page

// =============================================================================
// Top-level translator.
// =============================================================================

// Assemble the final Mach-O image from a format-neutral MachineModule. Owns
// nothing on entry; consumes the module (frees each function's heap buffers,
// the `funcs` array, and the `inits` array) before returning. The MachineModule
// struct itself is caller-owned. This is the shared back half of both the
// whole-module path (mlir_aarch64_to_macho) and the streaming per-function path
// (mlir_llvm_to_macho).
static bool finalize_macho(MachineModule *mm,
                           uint8_t **out_data, size_t *out_size);

bool mlir_aarch64_to_macho(MLIR_Context *ctx, MLIR_OpHandle module,
                           uint8_t **out_data, size_t *out_size) {
    (void)ctx;
    *out_data = NULL; *out_size = 0;
    if (!module) return false;
    if (MLIR_GetOpNumRegions(module) < 1) return false;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(MLIR_GetOpRegion(module, 0), 0);
    size_t n_top = MLIR_GetBlockNumOps(mb);

    // -----------------------------------------------------------------
    // Read module-level layout attributes set by mlir_llvm_to_aarch64.c.
    // -----------------------------------------------------------------
    uint32_t n_globals    = (uint32_t)attr_i(module, "n_globals");
    uint64_t global0_init = (uint64_t)attr_i(module, "global0_init");
    uint64_t linmem_size  = (uint64_t)attr_i(module, "linmem_size");

    // -----------------------------------------------------------------
    // Collect functions, find `_start`. `_start` must be placed first
    // in __text so LC_MAIN.entryoff equals text_section_off.
    // -----------------------------------------------------------------
    MachineFunc *efs = (MachineFunc *)calloc(n_top, sizeof(MachineFunc));
    size_t n_funcs = 0;
    size_t start_idx = (size_t)-1;
    // -----------------------------------------------------------------
    // Walk top-level for data_init ops first: collect the contributions
    // to the linmem __DATA section.
    // -----------------------------------------------------------------
    MachineDataInit  *inits = NULL;
    size_t    n_inits = 0;
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(op) != OP_TYPE_AARCH64_DATA_INIT) continue;
        int64_t off = attr_i(op, "offset");
        string  bs  = attr_s(op, "init_data");
        inits = (MachineDataInit *)realloc(inits, (n_inits + 1) * sizeof(MachineDataInit));
        inits[n_inits].offset = (uint32_t)off;
        inits[n_inits].bytes  = bs;
        n_inits++;
    }

    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        MLIR_OpType ot = MLIR_GetOpType(op);
        if (ot == OP_TYPE_AARCH64_DATA_INIT) continue;
        if (ot != OP_TYPE_AARCH64_FUNC) {
            string nm = MLIR_GetOpName(op);
            fprintf(stderr,
                "aarch64->macho: unexpected top-level op '%.*s'\n",
                (int)nm.size, nm.str);
            free(efs); free(inits); return false;
        }
        if (!emit_aarch64_func(op, &efs[n_funcs])) {
            for (size_t k = 0; k <= n_funcs; k++) {
                free(efs[k].code.data); free(efs[k].relocs); free(efs[k].dr);
                free(efs[k].br); free(efs[k].bp);
            }
            free(efs); free(inits); return false;
        }
        if (!patch_branches(&efs[n_funcs])) {
            for (size_t k = 0; k <= n_funcs; k++) {
                free(efs[k].code.data); free(efs[k].relocs); free(efs[k].dr);
                free(efs[k].br); free(efs[k].bp);
            }
            free(efs); free(inits); return false;
        }
        if (efs[n_funcs].name.size == 6
            && memcmp(efs[n_funcs].name.str, "_start", 6) == 0) {
            start_idx = n_funcs;
        }
        n_funcs++;
    }
    if (start_idx == (size_t)-1) {
        fprintf(stderr, "aarch64->macho: no `_start` function in module\n");
        for (size_t k = 0; k < n_funcs; k++) {
            free(efs[k].code.data); free(efs[k].relocs); free(efs[k].dr);
            free(efs[k].br); free(efs[k].bp);
        }
        free(efs); free(inits); return false;
    }

    MachineModule mm = {0};
    mm.funcs = efs;
    mm.n_funcs = n_funcs;
    mm.entry_idx = start_idx;
    mm.inits = inits;
    mm.n_inits = n_inits;
    mm.n_globals = n_globals;
    mm.global0_init = global0_init;
    mm.linmem_size = linmem_size;
    return finalize_macho(&mm, out_data, out_size);
}

static bool finalize_macho(MachineModule *mm,
                           uint8_t **out_data, size_t *out_size) {
    MachineFunc     *efs          = mm->funcs;
    size_t           n_funcs      = mm->n_funcs;
    size_t           start_idx    = mm->entry_idx;
    MachineDataInit *inits        = mm->inits;
    size_t           n_inits      = mm->n_inits;
    uint32_t         n_globals    = mm->n_globals;
    uint64_t         global0_init = mm->global0_init;
    uint64_t         linmem_size  = mm->linmem_size;
    *out_data = NULL; *out_size = 0;

    // Derived layout values (recomputed here so this entry is self-contained
    // for both the whole-module and streaming callers).
    uint32_t globals_bytes  = n_globals * 8u;
    uint32_t globals_padded = (globals_bytes + 15u) & ~15u;
    bool     has_data_seg   = (n_globals > 0) || (linmem_size > 0);
    uint32_t data_priv_size = has_data_seg ? 32u : 0u;
    (void)globals_bytes;

    // linmem_init_size = high-water mark across all data-init records.
    uint32_t linmem_init_size = 0;
    for (size_t i = 0; i < n_inits; i++) {
        uint32_t end = inits[i].offset + (uint32_t)inits[i].bytes.size;
        if (end > linmem_init_size) linmem_init_size = end;
    }
    if (linmem_init_size > 0) {
        // Round up to 16 for ARM64 alignment expectations.
        linmem_init_size = (linmem_init_size + 15u) & ~15u;
        // The native llvm->aarch64 path stores globals in the linmem
        // template section without any linmem/globals; ensure the
        // __DATA segment (and its 32-byte data_priv prefix) is emitted.
        if (!has_data_seg) { has_data_seg = true; data_priv_size = 32u; }
    }

    // -----------------------------------------------------------------
    // Layout: _start first, then everything else in source order.
    // -----------------------------------------------------------------
    uint32_t cursor = 0;
    efs[start_idx].text_off = cursor; cursor += (uint32_t)efs[start_idx].code.len;
    for (size_t i = 0; i < n_funcs; i++) {
        if (i == start_idx) continue;
        efs[i].text_off = cursor;
        cursor += (uint32_t)efs[i].code.len;
    }
    uint32_t text_size = cursor;
    while (text_size % 4) text_size++;

    // -----------------------------------------------------------------
    // Discovery pass: walk every BL reloc and record which libSystem
    // names are referenced. Each unique referenced name gets a dense
    // stub index in first-seen order, which determines its position
    // in __stubs / __got / chained-fixups imports / dysymtab
    // indirect-syms.
    //
    // Done BEFORE layout because n_libsys_stubs feeds into
    // stubs_size, got_size, sizeofcmds (via section count and load-cmd
    // body sizes), and the strtab. Done AFTER text layout because
    // unreachable BL relocs in functions that ended up at non-zero
    // text_off are still real references that need stubs.
    // -----------------------------------------------------------------
    LibSysRegistry libsys = {0};
    for (size_t i = 0; i < n_funcs; i++) {
        for (size_t k = 0; k < efs[i].n_relocs; k++) {
            MachineCallReloc *r = &efs[i].relocs[k];
            int ls = libsys_lookup(r->callee);
            if (ls < 0) continue;
            if (!libsys.used[ls]) {
                libsys.used[ls] = true;
                libsys.stub_index[ls] = libsys.n_stubs++;
            }
        }
    }

    // -----------------------------------------------------------------
    // Patch `bl` PC-relative displacements. Each reloc identifies its
    // callee by symbol name. Resolve against (in order):
    //   1. local functions  — branch to efs[j].text_off
    //   2. libSystem stubs   — branch to text_size + stub_index * 12
    // anything else is a hard error (no inter-module-text linking yet).
    // -----------------------------------------------------------------
    for (size_t i = 0; i < n_funcs; i++) {
        for (size_t k = 0; k < efs[i].n_relocs; k++) {
            MachineCallReloc *r = &efs[i].relocs[k];
            uint32_t dst_pc;
            bool resolved = false;
            for (size_t j = 0; j < n_funcs; j++) {
                if (efs[j].name.size == r->callee.size
                    && memcmp(efs[j].name.str, r->callee.str,
                              r->callee.size) == 0) {
                    dst_pc = efs[j].text_off;
                    resolved = true;
                    break;
                }
            }
            if (!resolved) {
                int ls = libsys_lookup(r->callee);
                if (ls >= 0 && libsys.used[ls]) {
                    dst_pc = text_size + libsys.stub_index[ls] * 12u;
                    resolved = true;
                }
            }
            if (!resolved) {
                fprintf(stderr,
                    "aarch64->macho: bl to unknown symbol '%.*s'\n",
                    (int)r->callee.size, r->callee.str);
                for (size_t k2 = 0; k2 < n_funcs; k2++) {
                    free(efs[k2].code.data); free(efs[k2].relocs); free(efs[k2].dr);
                    free(efs[k2].br); free(efs[k2].bp);
                }
                free(efs); return false;
            }
            uint32_t src_pc = efs[i].text_off + r->fn_off;
            int32_t  rel    = (int32_t)dst_pc - (int32_t)src_pc;
            int32_t  imm26  = rel >> 2;
            uint32_t insn   = arm64_bl(imm26);
            efs[i].code.data[r->fn_off + 0] = (uint8_t)(insn      );
            efs[i].code.data[r->fn_off + 1] = (uint8_t)(insn >>  8);
            efs[i].code.data[r->fn_off + 2] = (uint8_t)(insn >> 16);
            efs[i].code.data[r->fn_off + 3] = (uint8_t)(insn >> 24);
        }
    }

    // -----------------------------------------------------------------
    // Layout constants. n_stubs is the count of libSystem imports the
    // BL discovery pass found. See mlir_wasm_to_macho.c for the full
    // annotated walk-through of the load-command shape.
    // -----------------------------------------------------------------
    const uint32_t n_stubs = libsys.n_stubs;
    const uint32_t stub_size = 12;
    const uint32_t got_size  = n_stubs * 8;

    // sizeofcmds varies with __DATA presence. We use up to 2 sections in
    // __DATA: __data + __linmem_template. Linear memory itself is
    // allocated dynamically by `_start` via mmap (see synth_start in
    // mlir_llvm_to_aarch64.c), so the binary no longer reserves a
    // multi-GiB __linmem_bss zerofill section — that placement is
    // incompatible with the dyld shared cache on macOS arm64.
    uint32_t n_data_sections  = 0;
    if (has_data_seg) n_data_sections = 1;
    if (linmem_init_size > 0) n_data_sections = 2;
    uint32_t data_seg_lc_size = has_data_seg
        ? (72u + n_data_sections * 80u)
        : 0u;
    const uint32_t n_cmds      = has_data_seg ? 18u : 17u;
    const uint32_t sizeofcmds  = 976u + data_seg_lc_size;
    uint32_t text_section_off  = (32u + sizeofcmds + 15u) & ~15u;
    if (text_section_off < 1040u) text_section_off = 1040u;

    const uint32_t stubs_off    = text_section_off + text_size;
    const uint32_t stubs_size   = n_stubs * stub_size;
    const uint32_t cstring_off  = stubs_off + stubs_size;
    const uint32_t cstring_size = 0;

    uint32_t text_seg_end  = cstring_off + cstring_size;
    uint32_t text_seg_size = (text_seg_end + (VMSEG_SIZE - 1u)) & ~(VMSEG_SIZE - 1u);
    if (text_seg_size < VMSEG_SIZE) text_seg_size = VMSEG_SIZE;
    const uint64_t data_const_vm_base   = TEXT_VM_BASE + text_seg_size;
    const uint32_t data_const_file_base = text_seg_size;

    // __DATA segment (between __DATA_CONST and __LINKEDIT). Up to 2
    // sections, all file-backed and laid out sequentially:
    //   __data            S_REGULAR: data_priv (32B) + globals
    //   __linmem_template S_REGULAR: byte-for-byte image of the wasm
    //                                data segments; copied into the
    //                                mmap-allocated linmem by `_start`
    //                                (only present if linmem_init_size > 0)
    uint32_t data_section_payload = data_priv_size + globals_padded;
    uint32_t data_seg_payload    = data_section_payload + linmem_init_size;
    uint64_t data_seg_filesize_v = has_data_seg
        ? (((uint64_t)data_seg_payload + (VMSEG_SIZE - 1u)) & ~(uint64_t)(VMSEG_SIZE - 1u))
        : 0u;
    uint64_t data_seg_vmsize     = data_seg_filesize_v;

    const uint64_t data_vm_base       = data_const_vm_base + VMSEG_SIZE;
    const uint32_t data_file_base     = data_const_file_base + VMSEG_SIZE;
    const uint64_t data_priv_vmaddr   = data_vm_base;
    const uint64_t globals_vmaddr     = data_priv_vmaddr + data_priv_size;
    const uint64_t linmem_tpl_vmaddr  = globals_vmaddr + globals_padded;

    const uint64_t linkedit_vm_base   = has_data_seg
        ? (data_vm_base + data_seg_vmsize)
        : (data_const_vm_base + VMSEG_SIZE);
    const uint32_t linkedit_file_base = has_data_seg
        ? (data_file_base + (uint32_t)data_seg_filesize_v)
        : (data_const_file_base + VMSEG_SIZE);

    // -----------------------------------------------------------------
    // Build the image.
    // -----------------------------------------------------------------
    Buf img = {0};
    img.cap = 1 << 15; img.data = (uint8_t *)malloc(img.cap);

    // mach_header_64
    buf_le32(&img, MH_MAGIC_64);
    buf_le32(&img, CPU_TYPE_ARM64);
    buf_le32(&img, 0);
    buf_le32(&img, MH_EXECUTE);
    buf_le32(&img, n_cmds);
    buf_le32(&img, sizeofcmds);
    buf_le32(&img, MH_FLAGS_EXEC);
    buf_le32(&img, 0);

    // LC_SEGMENT_64 __PAGEZERO
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 72);
    { static const char SEG[16] = "__PAGEZERO"; buf_append(&img, SEG, 16); }
    buf_le64(&img, 0);
    buf_le64(&img, TEXT_VM_BASE);
    buf_le64(&img, 0); buf_le64(&img, 0);
    buf_le32(&img, 0); buf_le32(&img, 0);
    buf_le32(&img, 0); buf_le32(&img, 0);

    // LC_SEGMENT_64 __TEXT (3 sections: __text, __stubs, __cstring)
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 312);
    { static const char SEG[16] = "__TEXT"; buf_append(&img, SEG, 16); }
    buf_le64(&img, TEXT_VM_BASE);
    buf_le64(&img, (uint64_t)text_seg_size);
    buf_le64(&img, TEXT_FILE_BASE);
    buf_le64(&img, (uint64_t)text_seg_size);
    buf_le32(&img, VM_PROT_READ | VM_PROT_EXECUTE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_EXECUTE);
    buf_le32(&img, 3);
    buf_le32(&img, 0);
    {
        static const char SN[16] = "__text";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + text_section_off);
        buf_le64(&img, (uint64_t)text_size);
        buf_le32(&img, text_section_off);
        buf_le32(&img, 4);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0x80000400u);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0);
    }
    {
        static const char SN[16] = "__stubs";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + stubs_off);
        buf_le64(&img, (uint64_t)stubs_size);
        buf_le32(&img, stubs_off);
        buf_le32(&img, 2);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0x80000408u);
        buf_le32(&img, n_stubs);
        buf_le32(&img, stub_size);
        buf_le32(&img, 0);
    }
    {
        static const char SN[16] = "__cstring";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + cstring_off);
        buf_le64(&img, (uint64_t)cstring_size);
        buf_le32(&img, cstring_off);
        buf_le32(&img, 0);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 2);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0);
    }

    // LC_SEGMENT_64 __DATA_CONST (1 section: __got, empty)
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 152);
    { static const char SEG[16] = "__DATA_CONST"; buf_append(&img, SEG, 16); }
    buf_le64(&img, data_const_vm_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le64(&img, (uint64_t)data_const_file_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
    buf_le32(&img, 1);
    buf_le32(&img, 16);                  // SG_READ_ONLY
    {
        static const char SN[16] = "__got";
        static const char SG[16] = "__DATA_CONST";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, data_const_vm_base);
        buf_le64(&img, (uint64_t)got_size);
        buf_le32(&img, data_const_file_base);
        buf_le32(&img, 3);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 6);
        buf_le32(&img, 0);
        buf_le32(&img, 0);
        buf_le32(&img, 0);
    }

    // LC_SEGMENT_64 __DATA (optional). Up to two sections, both
    // file-backed and laid out sequentially:
    //   __data            : data_priv + globals
    //   __linmem_template : init bytes copied by `_start` into mmap'd linmem
    if (has_data_seg) {
        buf_le32(&img, LC_SEGMENT_64);
        buf_le32(&img, 72u + n_data_sections * 80u);
        { static const char SEG[16] = "__DATA"; buf_append(&img, SEG, 16); }
        buf_le64(&img, data_vm_base);
        buf_le64(&img, data_seg_vmsize);
        buf_le64(&img, (uint64_t)data_file_base);
        buf_le64(&img, data_seg_filesize_v);
        buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
        buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
        buf_le32(&img, n_data_sections);
        buf_le32(&img, 0);
        {
            static const char SN[16] = "__data";
            static const char SG[16] = "__DATA";
            buf_append(&img, SN, 16); buf_append(&img, SG, 16);
            buf_le64(&img, data_vm_base);
            buf_le64(&img, (uint64_t)data_section_payload);
            buf_le32(&img, data_file_base);
            buf_le32(&img, 3);                // align = 2^3 = 8
            buf_le32(&img, 0); buf_le32(&img, 0);
            buf_le32(&img, 0);                // S_REGULAR
            buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0);
        }
        if (linmem_init_size > 0) {
            static const char SN[16] = "__linmem_tpl";
            static const char SG[16] = "__DATA";
            buf_append(&img, SN, 16); buf_append(&img, SG, 16);
            buf_le64(&img, linmem_tpl_vmaddr);
            buf_le64(&img, (uint64_t)linmem_init_size);
            buf_le32(&img, data_file_base + data_section_payload);
            buf_le32(&img, 4);                // align = 2^4 = 16
            buf_le32(&img, 0); buf_le32(&img, 0);
            buf_le32(&img, 0);                // S_REGULAR
            buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0);
        }
    }

    // LC_SEGMENT_64 __LINKEDIT (filesize patched at end)
    size_t pos_linkedit_seg = img.len;
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 72);
    { static const char SEG[16] = "__LINKEDIT"; buf_append(&img, SEG, 16); }
    buf_le64(&img, linkedit_vm_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le64(&img, linkedit_file_base);
    buf_le64(&img, 0);                   // PLACEHOLDER filesize
    buf_le32(&img, VM_PROT_READ);
    buf_le32(&img, VM_PROT_READ);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_linkedit_filesize = pos_linkedit_seg + 8 + 16 + 8 + 8 + 8;

    size_t pos_lc_chained_fixups   = img.len;
    buf_le32(&img, LC_DYLD_CHAINED_FIXUPS); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_exports_trie     = img.len;
    buf_le32(&img, LC_DYLD_EXPORTS_TRIE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_symtab           = img.len;
    buf_le32(&img, LC_SYMTAB); buf_le32(&img, 24);
    buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_dysymtab         = img.len;
    buf_le32(&img, LC_DYSYMTAB); buf_le32(&img, 80);
    for (int k = 0; k < 18; k++) buf_le32(&img, 0);

    // LC_LOAD_DYLINKER
    buf_le32(&img, LC_LOAD_DYLINKER);
    buf_le32(&img, 32);
    buf_le32(&img, 12);
    { static const char nm[20] = "/usr/lib/dyld"; buf_append(&img, nm, 20); }

    // LC_UUID
    buf_le32(&img, LC_UUID); buf_le32(&img, 24);
    {
        static const uint8_t U[16] = {
            0x27,0x07,0xdd,0x62,0x09,0x67,0x3c,0xc0,
            0xb2,0xac,0xef,0xc3,0x2b,0x1c,0xf6,0x3a};
        buf_append(&img, U, 16);
    }

    // LC_BUILD_VERSION
    buf_le32(&img, LC_BUILD_VERSION); buf_le32(&img, 32);
    buf_le32(&img, 1);
    buf_le32(&img, 0x000f0700);
    buf_le32(&img, 0);
    buf_le32(&img, 1);
    buf_le32(&img, 3);
    buf_le32(&img, 0x04ce0100);

    // LC_SOURCE_VERSION
    buf_le32(&img, LC_SOURCE_VERSION); buf_le32(&img, 16);
    buf_le64(&img, 0);

    // LC_MAIN
    // Initial stack size: 256 MB. This is large because the
    // backend does not promote wasm locals to physical
    // registers — every local lives in an 8-byte stack cell, and
    // the regalloc spills aggressively (no live-range splitting).
    // Real-world tinyc functions like emit_expr end up with ~225 KB
    // per-call frames, so the parser only gets ~35 levels of
    // recursion on the macOS default 8 MB stack and OOMs on
    // deeply-nested expressions. 256 MB gives ~1100 emit_expr
    // levels which is plenty for tinyc's selfhost workload.
    buf_le32(&img, LC_MAIN); buf_le32(&img, 24);
    buf_le64(&img, (uint64_t)text_section_off);
    buf_le64(&img, 256ULL * 1024 * 1024);

    // LC_LOAD_DYLIB libSystem (kept so the load-command layout matches
    // the macho_exit reference even though we don't import anything).
    buf_le32(&img, LC_LOAD_DYLIB); buf_le32(&img, 56);
    buf_le32(&img, 24);
    buf_le32(&img, 2);
    buf_le32(&img, 0x054c0000);
    buf_le32(&img, 0x00010000);
    { static const char nm[32] = "/usr/lib/libSystem.B.dylib"; buf_append(&img, nm, 32); }

    size_t pos_lc_function_starts  = img.len;
    buf_le32(&img, LC_FUNCTION_STARTS); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_data_in_code     = img.len;
    buf_le32(&img, LC_DATA_IN_CODE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_code_sig         = img.len;
    buf_le32(&img, LC_CODE_SIGNATURE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);

    buf_pad_to(&img, text_section_off);

    // __text — _start first, then everything else.
    buf_append(&img, efs[start_idx].code.data, efs[start_idx].code.len);
    for (size_t i = 0; i < n_funcs; i++) {
        if (i == start_idx) continue;
        buf_append(&img, efs[i].code.data, efs[i].code.len);
    }
    buf_pad_to(&img, stubs_off);

    // __stubs: one ADRP/LDR/BR triple per libSystem import. Stub `i`
    // points at __got slot `i`; dyld populates the slot at load time
    // via the chained-fixups blob we emit below.
    {
        uint64_t data_const_vm_base_64 = TEXT_VM_BASE + (uint64_t)text_seg_size;
        for (uint32_t i = 0; i < n_stubs; i++) {
            uint64_t got_target = data_const_vm_base_64 + 8ULL * i;
            uint64_t stub_addr  = TEXT_VM_BASE + (uint64_t)stubs_off
                                + (uint64_t)i * stub_size;
            uint64_t page_dst   = got_target & ~0xfffULL;
            uint64_t page_src   = stub_addr  & ~0xfffULL;
            int64_t  page_diff  = (int64_t)(page_dst - page_src);
            int64_t  page_imm   = page_diff >> 12;
            uint32_t immlo = (uint32_t)(page_imm & 0x3);
            uint32_t immhi = (uint32_t)((page_imm >> 2) & 0x7ffff);
            uint32_t adrp  = 0x90000010u | (immlo << 29) | (immhi << 5);
            uint32_t lo12  = (uint32_t)(got_target & 0xfffu);
            uint32_t ldr   = 0xf9400210u | (((lo12 >> 3) & 0xfffu) << 10);
            uint32_t br    = 0xd61f0200u;  // br x16
            buf_le32(&img, adrp);
            buf_le32(&img, ldr);
            buf_le32(&img, br);
        }
    }
    buf_pad_to(&img, cstring_off);
    // __cstring empty.

    // Pad to __DATA_CONST.
    buf_pad_to(&img, data_const_file_base);
    // __got: one chained-fixup-format pointer per libSystem import. The
    // format is DYLD_CHAINED_PTR_64_OFFSET (pointer_format=6). Layout
    // (LSB-first):
    //   bits  0..23   ordinal (= imports[] index)
    //   bits 24..31   addend
    //   bits 32..50   reserved
    //   bits 51..62   next (stride 4; step 2 to reach the next 8B slot)
    //   bit      63   bind (1)
    for (uint32_t i = 0; i < n_stubs; i++) {
        uint64_t v = (1ULL << 63) | (uint64_t)i;
        if (i + 1 < n_stubs) v |= (2ULL << 51);
        buf_le64(&img, v);
    }

    // -----------------------------------------------------------------
    // __DATA file content (if any): __data section (data_priv + globals)
    // followed by __linmem_template bytes (file-backed wasm init data
    // that `_start` will memcpy into mmap'd linmem). No zerofill — the
    // remainder of linmem is anonymous mmap'd RAM.
    // -----------------------------------------------------------------
    if (has_data_seg) {
        buf_pad_to(&img, data_file_base);
        // data_priv: 32 zero bytes.
        for (uint32_t k = 0; k < data_priv_size; k++) buf_u8(&img, 0);
        // globals: 8 bytes each. global[0] = global0_init; rest = 0.
        for (uint32_t k = 0; k < n_globals; k++) {
            uint64_t v = (k == 0) ? global0_init : 0;
            buf_le64(&img, v);
        }
        // Pad globals to 16.
        while ((img.len - (data_file_base + data_priv_size)) < globals_padded) {
            buf_u8(&img, 0);
        }
        // __linmem_template section: zero-initialise the whole window
        // then overlay each data_init record's bytes at its
        // (wasm-offset relative) position. _start memcpys this into
        // mmap'd linmem at runtime.
        if (linmem_init_size > 0) {
            uint32_t init_start = (uint32_t)img.len;
            for (uint32_t k = 0; k < linmem_init_size; k++) buf_u8(&img, 0);
            for (size_t k = 0; k < n_inits; k++) {
                uint32_t off = inits[k].offset;
                string   bs  = inits[k].bytes;
                if ((uint32_t)(off + bs.size) > linmem_init_size) {
                    fprintf(stderr,
                        "aarch64->macho: data_init overflows init region\n");
                    free(efs); free(inits); free(img.data); return false;
                }
                memcpy(img.data + init_start + off, bs.str, bs.size);
            }
        }
        // Pad to data_seg_filesize (VMSEG boundary).
        buf_pad_to(&img, data_file_base + (uint32_t)data_seg_filesize_v);
    }

    buf_pad_to(&img, linkedit_file_base);

    // -----------------------------------------------------------------
    // Patch ADRP / ADD imm12 for data-segment references. Done after
    // text content lives in `img` so we know its absolute offset.
    // -----------------------------------------------------------------
    for (size_t i = 0; i < n_funcs; i++) {
        for (size_t k = 0; k < efs[i].n_dr; k++) {
            MachineDataReloc *dr = &efs[i].dr[k];
            uint64_t dst_vm;
            if (dr->kind.size == 9 && memcmp(dr->kind.str, "data_priv", 9) == 0) {
                dst_vm = data_priv_vmaddr;
            } else if (dr->kind.size == 7 && memcmp(dr->kind.str, "globals", 7) == 0) {
                dst_vm = globals_vmaddr;
            } else if (dr->kind.size == 15 && memcmp(dr->kind.str, "linmem_template", 15) == 0) {
                dst_vm = linmem_tpl_vmaddr;
            } else {
                // Otherwise treat the reloc kind as a local function symbol:
                // resolve to that function's absolute VM address. Used to
                // materialise function-pointer values (addressof @func).
                bool found = false;
                for (size_t j = 0; j < n_funcs; j++) {
                    if (efs[j].name.size == dr->kind.size &&
                        memcmp(efs[j].name.str, dr->kind.str, dr->kind.size) == 0) {
                        dst_vm = TEXT_VM_BASE + (uint64_t)text_section_off
                               + (uint64_t)efs[j].text_off;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    fprintf(stderr,
                        "aarch64->macho: unknown data reloc kind '%.*s'\n",
                        (int)dr->kind.size, dr->kind.str);
                    for (size_t k2 = 0; k2 < n_funcs; k2++) {
                        free(efs[k2].code.data); free(efs[k2].relocs); free(efs[k2].dr);
                    }
                    free(efs); free(inits); free(img.data); return false;
                }
            }
            uint64_t src_pc = TEXT_VM_BASE + (uint64_t)text_section_off
                            + (uint64_t)efs[i].text_off + (uint64_t)dr->fn_off;
            dst_vm += dr->addend;
            size_t   img_off = text_section_off + efs[i].text_off + dr->fn_off;
            uint32_t insn;
            if (!dr->is_add_lo) {
                int64_t dst_page = (int64_t)(dst_vm >> 12);
                int64_t src_page = (int64_t)(src_pc >> 12);
                int64_t rel_pages = dst_page - src_page;
                insn = arm64_adrp(dr->rd, rel_pages);
            } else {
                uint16_t imm12 = (uint16_t)(dst_vm & 0xfffu);
                insn = arm64_add_imm(dr->rd, dr->rn, imm12, /*sf=*/true);
            }
            img.data[img_off + 0] = (uint8_t)(insn      );
            img.data[img_off + 1] = (uint8_t)(insn >>  8);
            img.data[img_off + 2] = (uint8_t)(insn >> 16);
            img.data[img_off + 3] = (uint8_t)(insn >> 24);
        }
    }

    // =====================================================================
    // __LINKEDIT
    // =====================================================================
    size_t linkedit_start = img.len;

    // ---- chained fixups blob (no imports) ----
    size_t chained_start = img.len;
    uint32_t fx_imports_off = 0x50u;
    uint32_t fx_symbols_off = fx_imports_off + n_stubs * 4u;
    buf_le32(&img, 0);
    buf_le32(&img, 0x20);
    buf_le32(&img, fx_imports_off);
    buf_le32(&img, fx_symbols_off);
    buf_le32(&img, n_stubs);
    buf_le32(&img, 1);
    buf_le32(&img, 0);
    buf_le32(&img, 0);

    uint32_t fx_seg_count = 4u;
    buf_le32(&img, fx_seg_count);
    buf_le32(&img, 0); buf_le32(&img, 0);
    buf_le32(&img, 0x18);
    buf_le32(&img, 0);
    buf_le32(&img, 0);

    buf_le32(&img, 0x18);
    buf_le16(&img, 0x4000);
    buf_le16(&img, 6);
    buf_le64(&img, (uint64_t)data_const_file_base);
    buf_le32(&img, 0);
    buf_le16(&img, 1);
    // page_start[0]: 0 = chain starts at byte 0 of the page; 0xffff =
    // DYLD_CHAINED_PTR_START_NONE (no fixups on this page). When n_stubs
    // is 0 there's nothing to fix up so we mark the page as having no
    // chain to avoid dyld walking uninitialised bytes.
    buf_le16(&img, n_stubs > 0 ? 0 : 0xffff);

    // imports table: one DYLD_CHAINED_IMPORT (4B) per stub.
    //   bits  0..7   lib_ordinal (= 1 for libSystem.B.dylib, the only
    //                  LC_LOAD_DYLIB load command)
    //   bit       8  weak_import (0)
    //   bits  9..31  name_offset into symbols table
    //
    // For each stub_index i, we find which libsys name has stub_index==i
    // and emit it in that order — the linker builds the chained-fixup
    // chain in this order, so import[i] must match GOT slot i.
    uint32_t libsys_name_offsets[LS_COUNT] = {0};
    {
        // Pre-compute the byte offset of each libsys name in the
        // symbol-names blob. Leading 0 byte is at offset 0; first name
        // at offset 1.
        uint32_t off = 1;  // skip the leading 0
        for (uint32_t i = 0; i < n_stubs; i++) {
            // Find which libsys name has stub_index == i.
            for (size_t k = 0; k < LS_COUNT; k++) {
                if (libsys.used[k] && libsys.stub_index[k] == i) {
                    libsys_name_offsets[k] = off;
                    off += (uint32_t)strlen(libsys_name(k)) + 1u;
                    break;
                }
            }
        }
    }
    for (uint32_t i = 0; i < n_stubs; i++) {
        for (size_t k = 0; k < LS_COUNT; k++) {
            if (libsys.used[k] && libsys.stub_index[k] == i) {
                uint32_t lib_ordinal = 1;
                uint32_t name_off    = libsys_name_offsets[k];
                buf_le32(&img,
                    (lib_ordinal & 0xffu)
                    | ((name_off & 0x7fffffu) << 9));
                break;
            }
        }
    }

    // symbols table: leading 0, then each used libsys name in
    // stub-index order, NUL-terminated. Padded to multiple of 8 below.
    buf_u8(&img, 0);
    for (uint32_t i = 0; i < n_stubs; i++) {
        for (size_t k = 0; k < LS_COUNT; k++) {
            if (libsys.used[k] && libsys.stub_index[k] == i) {
                const char *nm = libsys_name(k);
                size_t nl = strlen(nm);
                for (size_t j = 0; j < nl; j++) buf_u8(&img, (uint8_t)nm[j]);
                buf_u8(&img, 0);
                break;
            }
        }
    }
    while ((img.len - chained_start) % 8) buf_u8(&img, 0);
    uint32_t chained_size = (uint32_t)(img.len - chained_start);

    // ---- exports trie ----
    size_t exports_start = img.len;
    buf_u8(&img, 0x00); buf_u8(&img, 0x01);
    buf_cstr(&img, "_");
    buf_uleb(&img, 0x12);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00); buf_u8(&img, 0x00);

    buf_u8(&img, 0x00); buf_u8(&img, 0x02);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00); buf_u8(&img, 0x00);

    buf_u8(&img, 0x03); buf_u8(&img, 0x00);
    buf_uleb(&img, (uint64_t)text_section_off);
    buf_u8(&img, 0x00);

    buf_u8(&img, 0x00); buf_u8(&img, 0x02);
    buf_cstr(&img, "_mh_execute_header"); buf_uleb(&img, 0x09);
    buf_cstr(&img, "main"); buf_uleb(&img, 0x0d);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00);
    while ((img.len - exports_start) % 8) buf_u8(&img, 0);
    uint32_t exports_size = (uint32_t)(img.len - exports_start);

    // ---- function starts ----
    size_t fs_start = img.len;
    buf_uleb(&img, (uint64_t)text_section_off);
    buf_u8(&img, 0);
    while ((img.len - fs_start) % 8) buf_u8(&img, 0);
    uint32_t fs_size = (uint32_t)(img.len - fs_start);

    // ---- symtab + strtab ----
    // Layout: [defined syms (mh, main)] + [undef syms (libsys imports)],
    // then strtab. Undef syms come AFTER defined ones because LC_DYSYMTAB
    // describes them as contiguous ranges (iundefsym = 2 for our two
    // defined syms). Indirect-syms references __stubs section, one
    // entry per stub, pointing at the symtab index of its undef sym.
    Buf strtab = {0};
    buf_u8(&strtab, 0x20);
    buf_u8(&strtab, 0x00);
    uint32_t str_mh   = (uint32_t)strtab.len; buf_cstr(&strtab, "__mh_execute_header");
    uint32_t str_main = (uint32_t)strtab.len; buf_cstr(&strtab, "_main");
    // libsys symbol names in stub-index order.
    uint32_t libsys_str_off[LS_COUNT] = {0};
    for (uint32_t i = 0; i < n_stubs; i++) {
        for (size_t k = 0; k < LS_COUNT; k++) {
            if (libsys.used[k] && libsys.stub_index[k] == i) {
                libsys_str_off[k] = (uint32_t)strtab.len;
                buf_cstr(&strtab, libsys_name(k));
                break;
            }
        }
    }
    while (strtab.len % 8) buf_u8(&strtab, 0);

    // Optional: per-function local symbols (gated by env var) so `sample`/
    // `nm`/`atos` can name functions in the stripped output. Names are the
    // lifter's `func_<wasmidx>` (or import name); n_value points at the
    // function's text VM address.
    bool emit_local_syms = (getenv("A64_LOCAL_SYMS") != NULL);
    Buf locals_strs = {0};
    uint32_t *local_stroff = NULL;
    if (emit_local_syms) {
        local_stroff = (uint32_t *)malloc(n_funcs * sizeof(uint32_t));
        for (size_t i = 0; i < n_funcs; i++) {
            local_stroff[i] = (uint32_t)strtab.len + (uint32_t)locals_strs.len;
            // leading '_' for tool friendliness
            buf_u8(&locals_strs, (uint8_t)'_');
            for (size_t k = 0; k < efs[i].name.size; k++)
                buf_u8(&locals_strs, (uint8_t)efs[i].name.str[k]);
            buf_u8(&locals_strs, 0);
        }
        buf_append(&strtab, locals_strs.data, locals_strs.len);
        while (strtab.len % 8) buf_u8(&strtab, 0);
    }

    Buf symtab = {0};
    // Local function symbols come FIRST (dysymtab requires locals, then
    // external-defined, then undefined, each contiguous).
    if (emit_local_syms) {
        for (size_t i = 0; i < n_funcs; i++) {
            buf_le32(&symtab, local_stroff[i]);
            buf_u8(&symtab, 0x0e);  // N_SECT (local, not external)
            buf_u8(&symtab, 1);     // n_sect = __text
            buf_le16(&symtab, 0);
            buf_le64(&symtab, TEXT_VM_BASE + text_section_off + efs[i].text_off);
        }
    }
    // __mh_execute_header
    buf_le32(&symtab, str_mh);
    buf_u8(&symtab, 0x0f); buf_u8(&symtab, 1);
    buf_le16(&symtab, 0x0010);
    buf_le64(&symtab, TEXT_VM_BASE);
    // _main
    buf_le32(&symtab, str_main);
    buf_u8(&symtab, 0x0f); buf_u8(&symtab, 1);
    buf_le16(&symtab, 0x0000);
    buf_le64(&symtab, TEXT_VM_BASE + text_section_off);
    // libsys undefs (n_type=N_UNDF|N_EXT=0x01, n_sect=0, n_desc=lib_ordinal<<8,
    // n_value=0). Order MUST match stub_index so indirect-syms can
    // point at symtab index (2 + stub_index).
    for (uint32_t i = 0; i < n_stubs; i++) {
        for (size_t k = 0; k < LS_COUNT; k++) {
            if (libsys.used[k] && libsys.stub_index[k] == i) {
                buf_le32(&symtab, libsys_str_off[k]);
                buf_u8(&symtab, 0x01);          // N_UNDF | N_EXT
                buf_u8(&symtab, 0);             // n_sect = NO_SECT
                // n_desc: lib_ordinal in high byte (libSystem = 1).
                // Bits: REFERENCE_FLAG_UNDEFINED_NON_LAZY (0) + ordinal.
                buf_le16(&symtab, (uint16_t)(1u << 8));
                buf_le64(&symtab, 0);
                break;
            }
        }
    }

    uint32_t n_locals     = emit_local_syms ? (uint32_t)n_funcs : 0u;
    uint32_t n_syms       = n_locals + 2u + n_stubs;
    uint32_t n_undefs     = n_stubs;
    uint32_t iextdefsym   = n_locals;
    uint32_t iundefsym    = n_locals + 2u;

    // Indirect-symbols table: 2*n_stubs entries. The first n_stubs are
    // for the __got section (reserved1=0 in the section header); the
    // next n_stubs are for __stubs (reserved1=n_stubs). Both blocks
    // hold the same values: indirect_sym[i] = symtab index of the
    // undef sym that backs slot i. Undefs start at iundefsym.
    Buf indsyms = {0};
    for (uint32_t i = 0; i < n_stubs; i++) buf_le32(&indsyms, iundefsym + i);  // GOT
    for (uint32_t i = 0; i < n_stubs; i++) buf_le32(&indsyms, iundefsym + i);  // stubs

    size_t symtab_off  = img.len;
    buf_append(&img, symtab.data, symtab.len);
    size_t indsyms_off = img.len;
    buf_append(&img, indsyms.data, indsyms.len);
    size_t strtab_off  = img.len;
    buf_append(&img, strtab.data, strtab.len);

    while ((img.len - linkedit_start) % 16) buf_u8(&img, 0);

    // ---- code signature ----
    uint32_t code_limit = (uint32_t)img.len;
    size_t code_sig_off = img.len;

    const uint32_t page_size  = 4096;
    const uint32_t page_shift = 12;
    const char *ident = "tinyc.out";
    uint32_t ident_len = (uint32_t)strlen(ident);
    const uint32_t n_slots = (code_limit + page_size - 1) / page_size;
    const uint32_t ident_offset = 88;
    const uint32_t n_special_slots = 2;
    const uint32_t hash_offset = ident_offset + ident_len + 1
                               + n_special_slots * SHA256_DIGEST_LEN;
    const uint32_t cd_len  = hash_offset + n_slots * SHA256_DIGEST_LEN;
    const uint32_t req_len = 12;
    const uint32_t cms_len = 8;
    const uint32_t sb_header = 12 + 3 * 8;
    const uint32_t sb_len_unpadded = sb_header + cd_len + req_len + cms_len;
    const uint32_t code_sig_size = (sb_len_unpadded + 15u) & ~15u;

    buf_patch_le32(&img, pos_lc_chained_fixups   + 8,  (uint32_t)chained_start);
    buf_patch_le32(&img, pos_lc_chained_fixups   + 12, chained_size);
    buf_patch_le32(&img, pos_lc_exports_trie     + 8,  (uint32_t)exports_start);
    buf_patch_le32(&img, pos_lc_exports_trie     + 12, exports_size);
    buf_patch_le32(&img, pos_lc_symtab           + 8,  (uint32_t)symtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 12, n_syms);
    buf_patch_le32(&img, pos_lc_symtab           + 16, (uint32_t)strtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 20, (uint32_t)strtab.len);
    buf_patch_le32(&img, pos_lc_dysymtab         + 8,  0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 12, n_locals);
    buf_patch_le32(&img, pos_lc_dysymtab         + 16, iextdefsym);
    buf_patch_le32(&img, pos_lc_dysymtab         + 20, 2);
    buf_patch_le32(&img, pos_lc_dysymtab         + 24, iundefsym);
    buf_patch_le32(&img, pos_lc_dysymtab         + 28, n_undefs);
    buf_patch_le32(&img, pos_lc_dysymtab         + 32, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 36, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 40, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 44, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 48, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 52, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 56, (uint32_t)indsyms_off);
    buf_patch_le32(&img, pos_lc_dysymtab         + 60, 2u * n_stubs);
    buf_patch_le32(&img, pos_lc_dysymtab         + 64, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 68, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 72, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 76, 0);
    buf_patch_le32(&img, pos_lc_function_starts  + 8,  (uint32_t)fs_start);
    buf_patch_le32(&img, pos_lc_function_starts  + 12, fs_size);
    buf_patch_le32(&img, pos_lc_data_in_code     + 8,  (uint32_t)fs_start + fs_size);
    buf_patch_le32(&img, pos_lc_data_in_code     + 12, 0);
    buf_patch_le32(&img, pos_lc_code_sig         + 8,  (uint32_t)code_sig_off);
    buf_patch_le32(&img, pos_lc_code_sig         + 12, code_sig_size);

    uint64_t linkedit_filesize_v =
        (uint64_t)(code_limit + code_sig_size) - linkedit_file_base;
    uint64_t linkedit_vmsize_v =
        (linkedit_filesize_v + (VMSEG_SIZE - 1u)) & ~(uint64_t)(VMSEG_SIZE - 1u);
    if (linkedit_vmsize_v < VMSEG_SIZE) linkedit_vmsize_v = VMSEG_SIZE;
    buf_patch_le64(&img, pos_linkedit_filesize,        linkedit_filesize_v);
    buf_patch_le64(&img, pos_linkedit_filesize - 16,   linkedit_vmsize_v);

    free(symtab.data); free(indsyms.data); free(strtab.data);

    Buf cs = {0};
    {
        Buf req = {0};
        buf_be32(&req, 0xfade0c01);
        buf_be32(&req, req_len);
        buf_be32(&req, 0);

        uint8_t req_hash[SHA256_DIGEST_LEN];
        sha256(req.data, req.len, req_hash);

        Buf cd = {0};
        buf_be32(&cd, 0xfade0c02);
        buf_be32(&cd, cd_len);
        buf_be32(&cd, 0x00020400);
        buf_be32(&cd, 0x00000002);          // CS_ADHOC
        buf_be32(&cd, hash_offset);
        buf_be32(&cd, ident_offset);
        buf_be32(&cd, n_special_slots);
        buf_be32(&cd, n_slots);
        buf_be32(&cd, code_limit);
        buf_u8(&cd, SHA256_DIGEST_LEN);
        buf_u8(&cd, 2);
        buf_u8(&cd, 0);
        buf_u8(&cd, page_shift);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        while (cd.len < 64) buf_u8(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, text_seg_size);
        buf_be32(&cd, 0);
        buf_be32(&cd, 1);                   // CS_EXECSEG_MAIN_BINARY
        while (cd.len < ident_offset) buf_u8(&cd, 0);
        buf_append(&cd, ident, ident_len);
        buf_u8(&cd, 0);
        buf_append(&cd, req_hash, SHA256_DIGEST_LEN);
        for (int z = 0; z < SHA256_DIGEST_LEN; z++) buf_u8(&cd, 0);
        for (uint32_t i = 0; i < n_slots; i++) {
            uint32_t start  = i * page_size;
            uint32_t remain = code_limit - start;
            uint32_t len    = remain < page_size ? remain : page_size;
            uint8_t  d[SHA256_DIGEST_LEN];
            sha256(img.data + start, len, d);
            buf_append(&cd, d, sizeof(d));
        }

        const uint32_t cd_off  = sb_header;
        const uint32_t req_off = cd_off + cd_len;
        const uint32_t cms_off = req_off + req_len;
        buf_be32(&cs, 0xfade0cc0);
        buf_be32(&cs, sb_len_unpadded);
        buf_be32(&cs, 3);
        buf_be32(&cs, 0x00000000);
        buf_be32(&cs, cd_off);
        buf_be32(&cs, 0x00000002);
        buf_be32(&cs, req_off);
        buf_be32(&cs, 0x00010000);
        buf_be32(&cs, cms_off);
        buf_append(&cs, cd.data, cd.len);
        buf_append(&cs, req.data, req.len);
        buf_be32(&cs, 0xfade0b01);
        buf_be32(&cs, cms_len);
        free(cd.data);
        free(req.data);
    }
    while (cs.len < code_sig_size) buf_u8(&cs, 0);
    buf_append(&img, cs.data, cs.len);
    free(cs.data);

    for (size_t i = 0; i < n_funcs; i++) {
        free(efs[i].code.data);
        free(efs[i].relocs);
        free(efs[i].dr);
        free(efs[i].br);
        free(efs[i].bp);
    }
    free(efs);
    free(inits);

    *out_data = img.data;
    *out_size = img.len;
    return true;
}

// ===========================================================================
// Streaming llvm -> Mach-O backend (low peak memory).
//
// The whole-module path (mlir_llvm_to_aarch64 + mlir_aarch64_to_macho) keeps
// every function's aarch64 IR live at once, which — with the trivial
// spill-everything allocator — balloons peak RSS past the 4GB self-host
// budget. Here we instead lower ONE function at a time into a throwaway temp
// arena, encode it to a heap `MachineFunc`, deep-copy the few strings the
// finalizer needs out of the temp arena, then reset the temp arena before the
// next function. Type interning is pinned to the persistent arena for the
// duration so cached type handles never dangle across a reset.
// ===========================================================================

// Track heap copies of MachineFunc strings so they can be freed after the
// finalizer (which only reads, never frees, the string contents).
typedef struct { char **p; size_t n, c; } OwnedStrs;

static char *owned_dup(OwnedStrs *o, string s) {
    char *p = (char *)malloc(s.size + 1);
    memcpy(p, s.str, s.size);
    p[s.size] = 0;
    if (o->n == o->c) {
        o->c = o->c ? o->c * 2 : 256;
        o->p = (char **)realloc(o->p, o->c * sizeof(char *));
    }
    o->p[o->n++] = p;
    return p;
}

// Deep-copy name + reloc callees + data-reloc kinds out of the temp arena.
static void ef_heapdup_strings(MachineFunc *e, OwnedStrs *o) {
    if (e->name.size) e->name = (string){ owned_dup(o, e->name), e->name.size };
    for (size_t i = 0; i < e->n_relocs; i++) {
        string c = e->relocs[i].callee;
        if (c.size) e->relocs[i].callee = (string){ owned_dup(o, c), c.size };
    }
    for (size_t i = 0; i < e->n_dr; i++) {
        string k = e->dr[i].kind;
        if (k.size) e->dr[i].kind = (string){ owned_dup(o, k), k.size };
    }
}

static void ef_free_one(MachineFunc *e) {
    free(e->code.data); free(e->relocs); free(e->dr);
    free(e->br); free(e->bp);
}

bool mlir_llvm_to_macho(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                        uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;

    uint8_t *gblob = NULL; uint32_t gblob_len = 0;
    LlvmSelState *sel = mlir_llvm_sel_begin(ctx, llvm_module, &gblob, &gblob_len);
    if (!sel) return false;
    if (!mlir_llvm_sel_saw_main(sel)) {
        fprintf(stderr, "llvm->aarch64: no defined 'main' function\n");
        mlir_llvm_sel_end(sel); free(gblob); return false;
    }

    size_t nf = mlir_llvm_sel_num_funcs(sel);
    MachineFunc *efs = (MachineFunc *)calloc(nf + 1, sizeof(MachineFunc));
    size_t n_funcs = 0;
    size_t start_idx = (size_t)-1;
    OwnedStrs owned = {0};

    Arena *persist = MLIR_GetArenaAllocator(ctx);
    ctx->type_arena = persist;   // pin type interning to the persistent arena

    Arena *tmp = arena_create(1 * 1024 * 1024);
    arena_pos_t tmp0 = arena_get_pos(tmp);

    bool ok = true;
    // Lower _start first, then each defined function in source order.
    for (size_t i = 0; ok && i <= nf; i++) {
        MLIR_SetArenaAllocator(ctx, tmp);
        MLIR_OpHandle fn = (i == 0)
            ? mlir_llvm_sel_synth_start(ctx, sel)
            : mlir_llvm_sel_func(ctx, sel, i - 1);
        MachineFunc *e = &efs[n_funcs];
        if (fn == MLIR_INVALID_HANDLE) {
            ok = false;
        } else if (!emit_aarch64_func(fn, e) || !patch_branches(e)) {
            ef_free_one(e);
            *e = (MachineFunc){0};
            ok = false;
        }
        MLIR_SetArenaAllocator(ctx, persist);
        if (ok) {
            // br/bp are only needed by patch_branches; drop them now.
            free(e->br); e->br = NULL; e->n_br = e->c_br = 0;
            free(e->bp); e->bp = NULL; e->n_bp = e->c_bp = 0;
            ef_heapdup_strings(e, &owned);
            if (e->name.size == 6 && memcmp(e->name.str, "_start", 6) == 0)
                start_idx = n_funcs;
            n_funcs++;
        }
        arena_reset(tmp, tmp0);
    }

    arena_destroy(tmp);
    ctx->type_arena = NULL;       // unpin
    mlir_llvm_sel_end(sel);

    if (!ok || start_idx == (size_t)-1) {
        if (ok)
            fprintf(stderr, "llvm->aarch64: no `_start` after streaming\n");
        for (size_t k = 0; k < n_funcs; k++) ef_free_one(&efs[k]);
        free(efs);
        for (size_t k = 0; k < owned.n; k++) free(owned.p[k]);
        free(owned.p);
        free(gblob);
        return false;
    }

    // One data-init record carrying the global blob (offset 0).
    MachineDataInit *inits = NULL; size_t n_inits = 0;
    if (gblob_len > 0) {
        inits = (MachineDataInit *)malloc(sizeof(MachineDataInit));
        inits[0].offset = 0;
        inits[0].bytes  = (string){ (char *)gblob, gblob_len };
        n_inits = 1;
    }

    // The llvm path carries no separate linmem/globals layout attrs; the
    // data blob alone drives the __DATA segment.
    MachineModule mm = {0};
    mm.funcs = efs;
    mm.n_funcs = n_funcs;
    mm.entry_idx = start_idx;
    mm.inits = inits;
    mm.n_inits = n_inits;
    bool r = finalize_macho(&mm, out_data, out_size);

    // finalize_macho consumed/freed efs and inits; the blob bytes and the
    // duped strings are no longer referenced.
    for (size_t k = 0; k < owned.n; k++) free(owned.p[k]);
    free(owned.p);
    free(gblob);
    return r;
}
