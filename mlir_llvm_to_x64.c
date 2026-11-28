// llvm dialect (flat CFG, LP64) -> static x86_64 ELF executable.
//
// The Linux-native sibling of mlir_llvm_to_aarch64.c + mlir_aarch64_to_macho.c.
// It consumes the SAME shared front half (C -> llvm dialect, then mem2reg /
// load_cse / arith_gvn / dce), but expects the standard (scf->cf) lowering so
// the body is a flat basic-block CFG with block arguments (MLIR's phi form):
// llvm.br / llvm.cond_br / llvm.icmp / llvm.add / llvm.call / llvm.return / ...
//
// Milestone-1 code generator: a deliberately simple, correct, slot-based
// allocator. Every SSA value lives in a stack slot; operands are loaded into
// scratch registers, the operation is performed, and the result is stored back.
// No register allocation yet (that converges onto the shared, target-
// parameterized allocator in a follow-up). System access is via direct Linux
// syscalls emitted as tiny local thunks (no libc / PLT / GOT / dynamic linker).
//
// Built into native Linux/x86_64 tinyC binaries and into the self-hosted ELF
// stages (driver.c TINYC_HAS_ELF).

#include "mlir_llvm_to_x64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir_api.h"
#include "mlir_obj.h"
#include "mlir_elf.h"
#include "mlir_llvm_type_layout.h"
#include "mlir_regalloc.h"
#include <base/buddy.h>
#include <platform/platform.h>

// ---------------------------------------------------------------------------
// x86-64 register numbers.
// ---------------------------------------------------------------------------
enum { RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7,
       R8=8, R9=9, R10=10, R11=11, R12=12, R13=13, R14=14, R15=15 };
// SysV integer argument registers, in order.
static const uint8_t ARG_REGS[6] = { RDI, RSI, RDX, RCX, R8, R9 };
// Linux x86_64 syscall numbers baked into the synthesised _start crt0.
// Every other syscall goes through the __tinyc_syscall6 thunk (the runtime
// number is its first argument).
#define NR_WRITE 1
#define NR_MMAP  9
#define NR_EXIT  60
// mmap() arguments for the lifted-wasm linear memory reservation in _start.
#define X64_PROT_READ_WRITE   0x3      // PROT_READ | PROT_WRITE
#define X64_MAP_LINMEM_FLAGS  0x4022   // MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
// wasm32 linear memory is addressed with zero-extended i32 offsets from
// __wasm_linmem_base, so the whole 4 GiB range is reserved up front.
#define X64_WASM_LINMEM_RESERVE (1ULL << 32)

// condition codes (jcc/setcc low nibble)
enum { CC_E=0x4, CC_NE=0x5, CC_B=0x2, CC_AE=0x3, CC_BE=0x6, CC_A=0x7,
       CC_L=0xC, CC_GE=0xD, CC_LE=0xE, CC_G=0xF };

// ---------------------------------------------------------------------------
// Value -> stack-slot map (open-addressing on the value handle).
// ---------------------------------------------------------------------------
typedef struct { uintptr_t *k; int32_t *v; size_t cap, n; } VMap;
static uintptr_t *g_x64_vmap_k = NULL;
static int32_t   *g_x64_vmap_v = NULL;
static size_t     g_x64_vmap_cap = 0;
static void vmap_init(VMap *m, size_t want) {
    size_t cap = 16; while (cap < want * 2) cap *= 2;
    if (cap > g_x64_vmap_cap) {
        if (g_x64_vmap_k) buddy_free(g_x64_vmap_k);
        if (g_x64_vmap_v) buddy_free(g_x64_vmap_v);
        size_t actual_k = 0, actual_v = 0;
        g_x64_vmap_k = (uintptr_t *)buddy_alloc(cap * sizeof(uintptr_t), &actual_k);
        g_x64_vmap_v = (int32_t *)buddy_alloc(cap * sizeof(int32_t), &actual_v);
        g_x64_vmap_cap = cap;
    }
    memset(g_x64_vmap_k, 0, cap * sizeof(uintptr_t));
    m->k = g_x64_vmap_k;
    m->v = g_x64_vmap_v;
    m->cap = cap; m->n = 0;
}
static void vmap_grow(VMap *m) {
    size_t ncap = m->cap * 2;
    size_t actual_k = 0, actual_v = 0;
    uintptr_t *nk = (uintptr_t *)buddy_alloc(ncap * sizeof(uintptr_t), &actual_k);
    int32_t   *nv = (int32_t *)buddy_alloc(ncap * sizeof(int32_t), &actual_v);
    memset(nk, 0, ncap * sizeof(uintptr_t));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->k[i]) continue;
        size_t j = (m->k[i] * 1099511628211ull) & mask;
        while (nk[j]) j = (j + 1) & mask;
        nk[j] = m->k[i]; nv[j] = m->v[i];
    }
    if (g_x64_vmap_k) buddy_free(g_x64_vmap_k);
    if (g_x64_vmap_v) buddy_free(g_x64_vmap_v);
    g_x64_vmap_k = nk;
    g_x64_vmap_v = nv;
    g_x64_vmap_cap = ncap;
    m->k = nk; m->v = nv; m->cap = ncap;
}
static void vmap_put(VMap *m, MLIR_ValueHandle key, int32_t val) {
    if ((m->n + 1) * 2 >= m->cap) vmap_grow(m);
    uintptr_t k = (uintptr_t)key;
    size_t i = (k * 1099511628211ull) & (m->cap - 1);
    while (m->k[i]) { if (m->k[i] == k) { m->v[i] = val; return; } i = (i + 1) & (m->cap - 1); }
    m->k[i] = k; m->v[i] = val; m->n++;
}
static bool vmap_get(VMap *m, MLIR_ValueHandle key, int32_t *out) {
    uintptr_t k = (uintptr_t)key;
    size_t i = (k * 1099511628211ull) & (m->cap - 1);
    while (m->k[i]) { if (m->k[i] == k) { *out = m->v[i]; return true; } i = (i + 1) & (m->cap - 1); }
    return false;
}
static void vmap_free(VMap *m) { m->k = NULL; m->v = NULL; m->cap = 0; m->n = 0; }
// The slot map also holds each alloca's region, keyed by its op handle with
// the top bit set: op and value handles can be equal natively (both are
// indices), and neither kind of handle has the top bit.
static inline MLIR_ValueHandle x64_region_key(MLIR_OpHandle op) {
    return (MLIR_ValueHandle)((uintptr_t)op | ((uintptr_t)1 << (sizeof(uintptr_t) * 8 - 1)));
}

static inline uint32_t x64_str_hash(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint32_t)(unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

// Global symbol -> byte offset (and size) in the module data blob (O(1) hash table).
typedef struct {
    char **name;
    size_t *nlen;
    uint32_t *off;
    uint32_t *sz;
    size_t n, cap;
    size_t *ht;    // 0 = empty, else index + 1
    size_t hcap;
} GMap;
static void gmap_rehash(GMap *g, size_t ncap) {
    size_t *ht = (size_t *)calloc(ncap, sizeof(size_t));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < g->n; i++) {
        size_t k = x64_str_hash(g->name[i], g->nlen[i]) & mask;
        while (ht[k]) k = (k + 1) & mask;
        ht[k] = i + 1;
    }
    free(g->ht);
    g->ht = ht;
    g->hcap = ncap;
}
static bool gmap_get(GMap *g, string nm, uint32_t *out) {
    if (!g->hcap) return false;
    size_t mask = g->hcap - 1;
    size_t k = x64_str_hash(nm.str, nm.size) & mask;
    while (g->ht[k]) {
        size_t idx = g->ht[k] - 1;
        if (g->nlen[idx] == nm.size && memcmp(g->name[idx], nm.str, nm.size) == 0) {
            *out = g->off[idx];
            return true;
        }
        k = (k + 1) & mask;
    }
    return false;
}
static bool gmap_get_cstr(GMap *g, const char *nm, uint32_t *off, uint32_t *size) {
    if (!g->hcap) return false;
    size_t len = strlen(nm);
    size_t mask = g->hcap - 1;
    size_t k = x64_str_hash(nm, len) & mask;
    while (g->ht[k]) {
        size_t idx = g->ht[k] - 1;
        if (g->nlen[idx] == len && memcmp(g->name[idx], nm, len) == 0) {
            if (off) *off = g->off[idx];
            if (size) *size = g->sz[idx];
            return true;
        }
        k = (k + 1) & mask;
    }
    return false;
}
static void gmap_put(GMap *g, string nm, uint32_t off, uint32_t sz) {
    if (g->n == g->cap) {
        g->cap = g->cap ? g->cap * 2 : 16;
        g->name = realloc(g->name, g->cap * sizeof(char *));
        g->nlen = realloc(g->nlen, g->cap * sizeof(size_t));
        g->off  = realloc(g->off,  g->cap * sizeof(uint32_t));
        g->sz   = realloc(g->sz,   g->cap * sizeof(uint32_t));
    }
    if ((g->n + 1) * 2 > g->hcap) {
        gmap_rehash(g, g->hcap ? g->hcap * 2 : 32);
    }
    size_t idx = g->n++;
    g->name[idx] = (char *)malloc(nm.size + 1);
    memcpy(g->name[idx], nm.str, nm.size);
    g->name[idx][nm.size] = 0;
    g->nlen[idx] = nm.size;
    g->off[idx] = off;
    g->sz[idx]  = sz;
    size_t mask = g->hcap - 1;
    size_t k = x64_str_hash(nm.str, nm.size) & mask;
    while (g->ht[k]) k = (k + 1) & mask;
    g->ht[k] = idx + 1;
}
static void gmap_free(GMap *g) {
    for (size_t i = 0; i < g->n; i++) free(g->name[i]);
    free(g->name); free(g->nlen); free(g->off); free(g->sz); free(g->ht);
}

// Reserve `sz` zero bytes at the end of a .data blob; return the slot offset.
static uint32_t x64_data_begin_global(Buf *data, size_t sz) {
    uint32_t off = (uint32_t)data->len;
    for (size_t z = 0; z < sz; z++) buf_u8(data, 0);
    return off;
}
// Register a reserved slot in gm and 8-byte-align the blob tail.
static void x64_data_finish_global(GMap *gm, Buf *data, string name, uint32_t off, size_t sz) {
    gmap_put(gm, name, off, (uint32_t)sz);
    while (data->len & 7) buf_u8(data, 0);
}
// Zero-filled global: reserve, register, align; return offset for lea_rip_data.
static uint32_t x64_data_alloc_global(GMap *gm, Buf *data, string name, size_t sz) {
    uint32_t off = x64_data_begin_global(data, sz);
    x64_data_finish_global(gm, data, name, off, sz);
    return off;
}

// Referenced external symbols (callees / addressof targets), for thunk synthesis.
// Names are stored as views into MLIR attribute storage or string literals.
typedef struct { string *n; size_t c, cap; size_t *ht; size_t hcap; } SymSet;
static void sym_add(SymSet *s, string nm) {
    if (s->hcap) {
        size_t mask = s->hcap - 1;
        size_t k = x64_str_hash(nm.str, nm.size) & mask;
        while (s->ht[k]) {
            if (str_eq(s->n[s->ht[k] - 1], nm)) return;
            k = (k + 1) & mask;
        }
    }
    if (s->c == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->n = realloc(s->n, s->cap * sizeof(string));
    }
    if ((s->c + 1) * 2 > s->hcap) {
        size_t ncap = s->hcap ? s->hcap * 2 : 32;
        size_t *ht = (size_t *)calloc(ncap, sizeof(size_t));
        size_t mask = ncap - 1;
        for (size_t i = 0; i < s->c; i++) {
            size_t k = x64_str_hash(s->n[i].str, s->n[i].size) & mask;
            while (ht[k]) k = (k + 1) & mask;
            ht[k] = i + 1;
        }
        free(s->ht);
        s->ht = ht;
        s->hcap = ncap;
    }
    size_t idx = s->c++;
    s->n[idx] = nm;
    size_t mask = s->hcap - 1;
    size_t k = x64_str_hash(nm.str, nm.size) & mask;
    while (s->ht[k]) k = (k + 1) & mask;
    s->ht[k] = idx + 1;
}
static void sym_free(SymSet *s) { free(s->n); free(s->ht); }

// ---------------------------------------------------------------------------
// Per-function lowering state.
// ---------------------------------------------------------------------------
typedef struct {
    int    target_block;  // index of the target block
    uint32_t field_off;   // offset of the rel32 field within the function code
} BrFix;

typedef struct {
    uint32_t lea_disp_off; // offset of rel32 in `lea rcx, [rip + disp32]`
    uint32_t span;         // number of table entries
    int     *targets;      // target block index for each entry [0..span-1]
} X64JTab;

typedef struct {
    MLIR_Context *ctx;
    ObjFunc      *f;
    VMap          slots;     // value -> rbp-relative byte offset (positive = below rbp)
    GMap         *gm;
    // Copies of mlir_llvm_to_elf()'s g_x64_mb / g_x64_data_buf (set in emit_func).
    // x64_ensure_global_in_gmap uses them when llvm.mlir.addressof hits a global
    // missing from gm after x64_init_data_section: scan module_mb for the global op,
    // append its storage to data_buf, then lea_rip_data uses the new offset.
    MLIR_BlockHandle module_mb;
    Buf          *data_buf;
    // True when the module was lifted from wasm (mlir_llvm_to_elf's wasm_lifted
    // argument). Only then do `__wasm_*` names denote lifter-owned data symbols;
    // in a native module they are ordinary user identifiers.
    bool          wasm_lifted;
    uint32_t      frame;     // total frame size (16-aligned)
    uint32_t      edge_tmp;  // rbp offset base of the parallel-move temp area
    // register allocation (shared mlir_regalloc): value -> physical register.
    // Values absent from `rm` live in `spill` (shared global-alloc slots) or `slots`.
    SlotMap       rm;
    SlotMap       xrm;       // FP value -> XMM register (2..15, caller-saved)
    bool          has_xrm;
    SlotMap       spill;
    bool          has_spill;
    bool          fast_ra;   // true when mlir_regalloc_fast ran (never maps remat ops)
    uint32_t      used_mask; // callee-saved pool regs the allocator used
    int32_t       cs_base;   // rbp offset base of the callee-saved spill area
    int           n_cs;      // # of callee-saved regs to save/restore
    uint8_t       cs_reg[8]; // those registers, in save order
    bool          omit_fp;       // true when frame pointer (rbp) is omitted (0-slot frame)
    bool          fp_pad;        // true when omit_fp needs 8-byte rsp alignment pad for calls
    bool          use_rsp_frame; // true when frame slots use [rsp + (frame - off)], freeing RBP as 6th callee-saved reg
    // intra-function branch fixups (used by x64_encode_block)
    BrFix        *fix; size_t n_fix, c_fix;
    X64JTab      *jtabs; size_t n_jtabs, c_jtabs;
    uint32_t      n_fixed;   // # of fixed (named) params, for llvm.intr.vastart
    bool          ok;
    // x64.* MLIR dialect lowering state: non-NULL during llvm.* -> x64.* isel,
    // MLIR_INVALID_HANDLE during x64_encode_block (and synth_start/synth_thunk).
    MLIR_BlockHandle x64_blk;
    MLIR_Builder     x64_bd;
    MLIR_TypeHandle  x64_i64;
    int              x64_next_label;
    // Per-value use counts (clamped at 2) when !fast_ra and def-use tracking is off,
    // so icmp -> cond_br fusion works at -O1/-O2.
    uint8_t         *use_cnt;
    uintptr_t        use_base;
    size_t           use_len;
} FnCtx;

static inline void x64_emit0(FnCtx *F, MLIR_OpType t) {
    MLIR_BuildVoid(&F->x64_bd, t);
}
static inline void x64_emit1(FnCtx *F, MLIR_OpType t, int64_t v) {
    MLIR_BuildVoidInlineI64(&F->x64_bd, t, v);
}
static inline void x64_emit2i(FnCtx *F, MLIR_OpType t, int64_t v, int64_t imm) {
    MLIR_BuildVoid2Attr(&F->x64_bd, t,
                        MLIR_KEY_VALUE, MLIR_CreateAttributeInteger(F->ctx, v, F->x64_i64),
                        MLIR_KEY_IMM,   MLIR_CreateAttributeInteger(F->ctx, imm, F->x64_i64));
}
static inline void x64_emit1s(FnCtx *F, MLIR_OpType t, string sym) {
    MLIR_BuildVoidAttr(&F->x64_bd, t, MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(F->ctx, sym));
}
static inline void x64_emit2s(FnCtx *F, MLIR_OpType t, int64_t v, string sym) {
    MLIR_BuildVoid2Attr(&F->x64_bd, t,
                        MLIR_KEY_VALUE,    MLIR_CreateAttributeInteger(F->ctx, v, F->x64_i64),
                        MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(F->ctx, sym));
}

// Register pool for the x86_64 backend.
// Caller-saved: r10, r8, r9, rsi, rdi (entry param homing and call arg setup
// use parallel-move resolution so arg regs can live in the pool).
// Callee-saved: rbx, r12..r15.
// Kept outside pool as scratch/implicit-clobber regs: rax, rcx, rdx, r11.
static RegTarget x64_regtarget(void) {
    RegTarget T; memset(&T, 0, sizeof(T));
    T.npool = 10; T.ncaller = 5; T.nhome = 3;
    T.pool[0] = R10; T.pool[1] = R8; T.pool[2] = R9; T.pool[3] = RSI; T.pool[4] = RDI;
    T.pool[5] = RBX; T.pool[6] = R12; T.pool[7] = R13; T.pool[8] = R14; T.pool[9] = R15;
    T.remat_alloca = true;
    return T;
}
// True (with *reg set) if value `v` has a physical-register home.
static inline bool val_in_reg(FnCtx *F, MLIR_ValueHandle v, uint8_t *reg) {
    int32_t r;
    if (sm_get_fast(&F->rm, v, &r)) { *reg = (uint8_t)r; return true; }
    return false;
}
// True (with *xreg set) if FP value `v` has a physical XMM register home (xmm2..xmm15).
static inline bool val_in_xmm(FnCtx *F, MLIR_ValueHandle v, uint8_t *xreg) {
    if (!F->has_xrm) return false;
    int32_t r;
    if (sm_get_fast(&F->xrm, v, &r)) { *xreg = (uint8_t)r; return true; }
    return false;
}

static void fail(FnCtx *F, const char *msg, string extra) {
    if (F->ok) {
        if (extra.size) fprintf(stderr, "x64: %s '%.*s'\n", msg, (int)extra.size, extra.str);
        else            fprintf(stderr, "x64: %s\n", msg);
    }
    F->ok = false;
}

// --- byte emission ---------------------------------------------------------
static inline void eb(FnCtx *F, uint8_t b) { buf_u8(&F->f->code, b); }
static inline void e32(FnCtx *F, uint32_t v) { buf_le32(&F->f->code, v); }
static inline uint32_t here(FnCtx *F) { return (uint32_t)F->f->code.len; }

static inline void rex(FnCtx *F, int w, int r, int x, int b) {
    uint8_t v = 0x40 | (w << 3) | ((r >= 8) << 2) | ((x >= 8) << 1) | (b >= 8);
    if (v != 0x40 || w) eb(F, v);
}
static inline void modrm(FnCtx *F, int mod, int reg, int rm) {
    eb(F, (uint8_t)((mod << 6) | ((reg & 7) << 3) | (rm & 7)));
}

// mov reg, imm  (w=1 -> full 64-bit immediate via movabs)
static void mov_ri(FnCtx *F, uint8_t reg, int64_t imm, int w) {
    if (F->x64_blk) {
        if (!w || (imm >= INT32_MIN && imm <= INT32_MAX)) {
            x64_emit1(F, OP_TYPE_X64_MOV_RI32,
                      (int64_t)reg | ((int64_t)w << 8) | ((int64_t)(uint32_t)(int32_t)imm << 32));
        } else {
            x64_emit2i(F, OP_TYPE_X64_MOV_RI, (int64_t)reg | ((int64_t)w << 8), imm);
        }
        return;
    }
    if (imm == 0) {
        // xor reg32, reg32 (2-3 bytes, zero-extends to 64 bits)
        uint8_t rv = (uint8_t)(0x40 | ((reg >= 8) << 2) | (reg >= 8));
        if (rv != 0x40) eb(F, rv);
        eb(F, 0x31);
        modrm(F, 3, reg, reg);
        return;
    }
    if (w) {
        if (imm >= 0 && imm <= UINT32_MAX) {
            if (reg >= 8) eb(F, 0x41);
            eb(F, 0xB8 + (reg & 7));
            e32(F, (uint32_t)imm);
        } else {
            rex(F, 1, 0, 0, reg);
            eb(F, 0xB8 + (reg & 7));
            buf_le64(&F->f->code, (uint64_t)imm);
        }
    } else {
        if (reg >= 8) eb(F, 0x41);
        eb(F, 0xB8 + (reg & 7));
        e32(F, (uint32_t)imm);
    }
}
// generic ALU r/m64-or-32, r : opcode form (dst = dst OP src)
static inline void alu_rr(FnCtx *F, uint8_t op, uint8_t dst, uint8_t src, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_ALU_RR,
                  (int64_t)dst | ((int64_t)src << 4) | ((int64_t)w << 8) | ((int64_t)op << 12));
        return;
    }
    Buf *b = &F->f->code;
    if (b->len + 3 > b->cap) buf_grow(b, 3);
    uint8_t *p = b->data + b->len;
    uint8_t rv = (uint8_t)(0x40 | (w << 3) | ((src >= 8) << 2) | (dst >= 8));
    if (rv != 0x40 || w) *p++ = rv;
    *p++ = op;
    *p++ = (uint8_t)(0xC0 | ((src & 7) << 3) | (dst & 7));
    b->len = (size_t)(p - b->data);
}
// ALU r/m64-or-32, imm8/imm32 (ext: 0=add, 1=or, 4=and, 5=sub, 6=xor, 7=cmp)
static inline void alu_ri(FnCtx *F, uint8_t ext, uint8_t dst, int32_t imm, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_ALU_RI,
                  (int64_t)dst | ((int64_t)ext << 4) | ((int64_t)w << 8) | ((int64_t)(uint32_t)imm << 32));
        return;
    }
    Buf *b = &F->f->code;
    if (b->len + 7 > b->cap) buf_grow(b, 7);
    uint8_t *p = b->data + b->len;
    uint8_t rv = (uint8_t)(0x40 | (w << 3) | (dst >= 8));
    if (rv != 0x40 || w) *p++ = rv;
    if (imm >= -128 && imm <= 127) {
        *p++ = 0x83;
        *p++ = (uint8_t)(0xC0 | ((ext & 7) << 3) | (dst & 7));
        *p++ = (uint8_t)(int8_t)imm;
    } else {
        *p++ = 0x81;
        *p++ = (uint8_t)(0xC0 | ((ext & 7) << 3) | (dst & 7));
        uint32_t u = (uint32_t)imm;
        p[0] = (uint8_t)(u & 0xff); p[1] = (uint8_t)((u >> 8) & 0xff);
        p[2] = (uint8_t)((u >> 16) & 0xff); p[3] = (uint8_t)((u >> 24) & 0xff);
        p += 4;
    }
    b->len = (size_t)(p - b->data);
}
static inline void mov_rr(FnCtx *F, uint8_t dst, uint8_t src, int w) {
    if (dst == src && w) return;
    if (dst == src && !w) return;
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_MOV_RR,
                  (int64_t)dst | ((int64_t)src << 4) | ((int64_t)w << 8));
        return;
    }
    alu_rr(F, 0x89, dst, src, w);
}
static void imul_rr(FnCtx *F, uint8_t dst, uint8_t src, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_IMUL_RR,
                  (int64_t)dst | ((int64_t)src << 4) | ((int64_t)w << 8));
        return;
    }
    rex(F, w, dst, 0, src);
    eb(F, 0x0F); eb(F, 0xAF);
    modrm(F, 3, dst, src);
}
// Emit ModRM (+ SIB/disp) for a frame slot: [rsp + frame - off] if use_rsp_frame, else [rbp - off].
static inline void emit_frame_modrm(FnCtx *F, uint8_t reg, int32_t off) {
    if (F->use_rsp_frame) {
        int32_t d = (int32_t)F->frame - off;
        if (d == 0)                     { modrm(F, 0, reg, RSP); eb(F, 0x24); }
        else if (d >= -128 && d <= 127) { modrm(F, 1, reg, RSP); eb(F, 0x24); eb(F, (uint8_t)(int8_t)d); }
        else                            { modrm(F, 2, reg, RSP); eb(F, 0x24); e32(F, (uint32_t)d); }
    } else {
        int32_t d = -off;
        if (d >= -128 && d <= 127) { modrm(F, 1, reg, RBP); eb(F, (uint8_t)(int8_t)d); }
        else                       { modrm(F, 2, reg, RBP); e32(F, (uint32_t)d); }
    }
}
// load reg, [rbp - off] (or [rsp + frame - off])
static inline void load_rbp(FnCtx *F, uint8_t reg, int32_t off, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_LOAD_RBP,
                  (int64_t)reg | ((int64_t)w << 8) | ((int64_t)(uint32_t)off << 32));
        return;
    }
    rex(F, w, reg, 0, 0); eb(F, 0x8B); emit_frame_modrm(F, reg, off);
}
// store [rbp - off], reg (or [rsp + frame - off])
static inline void store_rbp(FnCtx *F, int32_t off, uint8_t reg, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_STORE_RBP,
                  (int64_t)reg | ((int64_t)w << 8) | ((int64_t)(uint32_t)off << 32));
        return;
    }
    rex(F, w, reg, 0, 0); eb(F, 0x89); emit_frame_modrm(F, reg, off);
}
static inline void lea_rbp(FnCtx *F, uint8_t reg, int32_t off) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_LEA_RBP,
                  (int64_t)reg | ((int64_t)(uint32_t)off << 32));
        return;
    }
    rex(F, 1, reg, 0, 0); eb(F, 0x8D); emit_frame_modrm(F, reg, off);
}
static void load_rbp_sized(FnCtx *F, uint8_t dst, int32_t off, int bytes) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_LOAD_RBP_SIZED,
                  (int64_t)dst | ((int64_t)bytes << 4) | ((int64_t)(uint32_t)off << 32));
        return;
    }
    if (bytes == 8) { load_rbp(F, dst, off, 1); return; }
    if (bytes == 4) { load_rbp(F, dst, off, 0); return; }
    rex(F, 1, dst, 0, 0);
    eb(F, 0x0F); eb(F, bytes == 2 ? 0xB7 : 0xB6);
    emit_frame_modrm(F, dst, off);
}
static void store_rbp_sized(FnCtx *F, int32_t off, uint8_t src, int bytes) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_STORE_RBP_SIZED,
                  (int64_t)src | ((int64_t)bytes << 4) | ((int64_t)(uint32_t)off << 32));
        return;
    }
    if (bytes == 8) { store_rbp(F, off, src, 1); return; }
    if (bytes == 4) { store_rbp(F, off, src, 0); return; }
    if (bytes == 2) { eb(F, 0x66); rex(F, 0, src, 0, 0); eb(F, 0x89); }
    else {
        if (src >= 4) eb(F, (uint8_t)(0x40 | ((src >= 8) << 2)));
        eb(F, 0x88);
    }
    emit_frame_modrm(F, src, off);
}
// Emit `lea reg, [rip+disp32]` with disp32 left at 0; return offset of disp32 field.
static uint32_t lea_rip_emit(FnCtx *F, uint8_t reg) {
    rex(F, 1, reg, 0, 0);
    eb(F, 0x8D);
    modrm(F, 0, reg, RBP); /* rm=101 + mod=00 => RIP-relative */
    uint32_t field = here(F);
    e32(F, 0);
    return field;
}

// Heap-copy a NUL-terminated C string (ObjModule owns the result).
static char *heapdup_cstr(const char *s) {
    size_t n = strlen(s) + 1;
    char *r = (char *)malloc(n);
    if (r) memcpy(r, s, n);
    return r;
}

// When true (mlir_llvm_to_x64_code), reloc symbol pointers are borrowed directly
// from MLIR_Context attribute storage (zero malloc/free per call site).
static bool g_x64_borrow_reloc_syms = false;

// Record a PC-relative text-section reloc at `field` (shared by lea/call sites).
static void add_text_reloc(FnCtx *F, uint32_t field, string sym) {
    char *cs;
    if (g_x64_borrow_reloc_syms) {
        cs = (char *)sym.str;
    } else {
        cs = (char *)malloc(sym.size + 1);
        memcpy(cs, sym.str, sym.size); cs[sym.size] = 0;
    }
    obj_add_reloc(F->f, field, field + 4, false, cs, 0);
    F->f->relocs[F->f->n_relocs - 1].sym_len = (uint32_t)sym.size;
}

// lea reg, [rip + disp32]  (records a data reloc onto the global at `data_off`)
static void lea_rip_data(FnCtx *F, uint8_t reg, int64_t data_off) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_LEA_RIP_DATA,
                  (int64_t)reg | ((int64_t)(uint32_t)data_off << 32));
        return;
    }
    uint32_t field = lea_rip_emit(F, reg);
    obj_add_reloc(F->f, field, field + 4, true, NULL, data_off);
}
// lea reg, [rip + disp32]  (PC-relative to a function symbol in the text section)
static void lea_rip_func(FnCtx *F, uint8_t reg, string sym) {
    if (F->x64_blk) {
        x64_emit2s(F, OP_TYPE_X64_LEA_RIP_FUNC, (int64_t)reg, sym);
        return;
    }
    uint32_t field = lea_rip_emit(F, reg);
    add_text_reloc(F, field, sym);
}
// Emit ModRM (+ SIB/disp8 if needed) for [addr] addressing with any GP register.
static inline void modrm_mem0(FnCtx *F, uint8_t reg, uint8_t addr) {
    if ((addr & 7) == 4)      { modrm(F, 0, reg, addr); eb(F, 0x24); }
    else if ((addr & 7) == 5) { modrm(F, 1, reg, addr); eb(F, 0x00); }
    else                      { modrm(F, 0, reg, addr); }
}
// width-aware memory load/store through an address already in `addr`
static void load_mem(FnCtx *F, uint8_t dst, uint8_t addr, int bytes, bool sgn) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_LOAD_MEM,
                  (int64_t)dst | ((int64_t)addr << 4) | ((int64_t)bytes << 8) | ((int64_t)(sgn ? 1 : 0) << 12));
        return;
    }
    if (bytes == 8) { rex(F,1,dst,0,addr); eb(F,0x8B); modrm_mem0(F,dst,addr); }
    else if (bytes == 4) { rex(F,0,dst,0,addr); eb(F,0x8B); modrm_mem0(F,dst,addr);
                           if (sgn) { /* movsxd done by caller width */ } }
    else if (bytes == 2) { rex(F,1,dst,0,addr); eb(F,0x0F); eb(F, sgn?0xBF:0xB7); modrm_mem0(F,dst,addr); }
    else { rex(F,1,dst,0,addr); eb(F,0x0F); eb(F, sgn?0xBE:0xB6); modrm_mem0(F,dst,addr); }
}
static void store_mem(FnCtx *F, uint8_t addr, uint8_t src, int bytes) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_STORE_MEM,
                  (int64_t)addr | ((int64_t)src << 4) | ((int64_t)bytes << 8));
        return;
    }
    if (bytes == 1) {
        if (src >= 4 && src < 8 && addr < 8) eb(F, 0x40);
        else rex(F,0,src,0,addr);
        eb(F,0x88); modrm_mem0(F,src,addr);
    }
    else if (bytes == 2) { eb(F,0x66); rex(F,0,src,0,addr); eb(F,0x89); modrm_mem0(F,src,addr); }
    else if (bytes == 4) { rex(F,0,src,0,addr); eb(F,0x89); modrm_mem0(F,src,addr); }
    else { rex(F,1,src,0,addr); eb(F,0x89); modrm_mem0(F,src,addr); }
}

// Emit ModRM (+ SIB/disp8/disp32) for [base + disp32] addressing.
static inline void modrm_mem_disp(FnCtx *F, uint8_t reg, uint8_t base, int32_t disp) {
    if (disp == 0 && (base & 7) != 5) {
        modrm(F, 0, reg, base);
        if ((base & 7) == 4) eb(F, 0x24);
    } else if (disp >= -128 && disp <= 127) {
        modrm(F, 1, reg, base);
        if ((base & 7) == 4) eb(F, 0x24);
        eb(F, (uint8_t)(int8_t)disp);
    } else {
        modrm(F, 2, reg, base);
        if ((base & 7) == 4) eb(F, 0x24);
        e32(F, (uint32_t)disp);
    }
}
// Emit ModRM + SIB (+ disp8=0 for rbp/r13) for [base + idx * (1<<scale)].
// Caller must guarantee idx != RSP (4).
static inline void modrm_mem_sib(FnCtx *F, uint8_t reg, uint8_t base, uint8_t idx, int scale) {
    uint8_t sib = (uint8_t)(((scale & 3) << 6) | ((idx & 7) << 3) | (base & 7));
    if ((base & 7) == 5) {
        modrm(F, 1, reg, 4);
        eb(F, sib);
        eb(F, 0);
    } else {
        modrm(F, 0, reg, 4);
        eb(F, sib);
    }
}
static void load_mem_disp(FnCtx *F, uint8_t dst, uint8_t base, int bytes, bool sgn, int32_t disp) {
    if (bytes == 8)      { rex(F,1,dst,0,base); eb(F,0x8B); modrm_mem_disp(F,dst,base,disp); }
    else if (bytes == 4) { rex(F,0,dst,0,base); eb(F,0x8B); modrm_mem_disp(F,dst,base,disp); }
    else if (bytes == 2) { rex(F,1,dst,0,base); eb(F,0x0F); eb(F, sgn?0xBF:0xB7); modrm_mem_disp(F,dst,base,disp); }
    else                 { rex(F,1,dst,0,base); eb(F,0x0F); eb(F, sgn?0xBE:0xB6); modrm_mem_disp(F,dst,base,disp); }
}
static void store_mem_disp(FnCtx *F, uint8_t base, uint8_t src, int bytes, int32_t disp) {
    if (bytes == 1) {
        if (src >= 4 && src < 8 && base < 8) eb(F, 0x40);
        else rex(F,0,src,0,base);
        eb(F,0x88); modrm_mem_disp(F,src,base,disp);
    }
    else if (bytes == 2) { eb(F,0x66); rex(F,0,src,0,base); eb(F,0x89); modrm_mem_disp(F,src,base,disp); }
    else if (bytes == 4) { rex(F,0,src,0,base); eb(F,0x89); modrm_mem_disp(F,src,base,disp); }
    else                 { rex(F,1,src,0,base); eb(F,0x89); modrm_mem_disp(F,src,base,disp); }
}
static void load_mem_sib(FnCtx *F, uint8_t dst, uint8_t base, uint8_t idx, int scale, int bytes, bool sgn) {
    if (bytes == 8)      { rex(F,1,dst,idx,base); eb(F,0x8B); modrm_mem_sib(F,dst,base,idx,scale); }
    else if (bytes == 4) { rex(F,0,dst,idx,base); eb(F,0x8B); modrm_mem_sib(F,dst,base,idx,scale); }
    else if (bytes == 2) { rex(F,1,dst,idx,base); eb(F,0x0F); eb(F, sgn?0xBF:0xB7); modrm_mem_sib(F,dst,base,idx,scale); }
    else                 { rex(F,1,dst,idx,base); eb(F,0x0F); eb(F, sgn?0xBE:0xB6); modrm_mem_sib(F,dst,base,idx,scale); }
}
static void store_mem_sib(FnCtx *F, uint8_t base, uint8_t idx, uint8_t src, int scale, int bytes) {
    if (bytes == 1) {
        if (src >= 4 && src < 8 && idx < 8 && base < 8) eb(F, 0x40);
        else rex(F,0,src,idx,base);
        eb(F,0x88); modrm_mem_sib(F,src,base,idx,scale);
    }
    else if (bytes == 2) { eb(F,0x66); rex(F,0,src,idx,base); eb(F,0x89); modrm_mem_sib(F,src,base,idx,scale); }
    else if (bytes == 4) { rex(F,0,src,idx,base); eb(F,0x89); modrm_mem_sib(F,src,base,idx,scale); }
    else                 { rex(F,1,src,idx,base); eb(F,0x89); modrm_mem_sib(F,src,base,idx,scale); }
}
// mov BYTE PTR [base + idx*(1<<scale)], imm8   ([REX] C6 /0 SIB ib)
static void store_mem_sib_imm8(FnCtx *F, uint8_t base, uint8_t idx, int scale, uint8_t imm8) {
    rex(F, 0, 0, idx, base);
    eb(F, 0xC6);
    modrm_mem_sib(F, 0, base, idx, scale);
    eb(F, imm8);
}
static void lea_mem_disp(FnCtx *F, uint8_t dst, uint8_t base, int32_t disp) {
    rex(F, 1, dst, 0, base);
    eb(F, 0x8D);
    modrm_mem_disp(F, dst, base, disp);
}

// ---------------------------------------------------------------------------
// type / constant helpers
// ---------------------------------------------------------------------------
// Bit width of an integer-typed value; 0 for pointers and other types.
static int int_bits(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_TypeHandle t = MLIR_GetValueType(ctx, v);
    return MLIR_GetTypeKind(ctx, t) == MLIR_TYPE_KIND_INTEGER ? (int)MLIR_GetTypeWidth(ctx, t) : 0;
}
static int val_w(MLIR_Context *ctx, MLIR_ValueHandle v) {
    int b = int_bits(ctx, v);
    return (b == 0 || b > 32) ? 1 : 0;  // ptr and i64 -> 64-bit; else 32-bit
}
static bool is_const(MLIR_Context *ctx, MLIR_OpHandle def, int64_t *val) {
    if (def == MLIR_INVALID_HANDLE) return false;
    switch (MLIR_GetOpType(ctx, def)) {
    case OP_TYPE_LLVM_MLIR_CONSTANT: {
        MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, def, MLIR_KEY_VALUE);
        if (a == MLIR_INVALID_HANDLE) return false;
        *val = MLIR_GetAttributeInteger(ctx, a);
        return true;
    }
    case OP_TYPE_LLVM_MLIR_ZERO:
    case OP_TYPE_LLVM_MLIR_NULL:
        *val = 0; return true;
    case OP_TYPE_LLVM_SEXT: {
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, def, 0);
        int64_t c = 0;
        if (!is_const(ctx, MLIR_GetValueDefiningOp(ctx, a0), &c)) return false;
        int sb = int_bits(ctx, a0);
        if (sb > 0 && sb < 64) {
            uint64_t mask = (1ull << sb) - 1ull;
            uint64_t sign = 1ull << (sb - 1);
            c = (int64_t)(((uint64_t)c & mask) ^ sign) - (int64_t)sign;
        }
        *val = c; return true;
    }
    case OP_TYPE_LLVM_ZEXT: {
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, def, 0);
        int64_t c = 0;
        if (!is_const(ctx, MLIR_GetValueDefiningOp(ctx, a0), &c)) return false;
        int sb = int_bits(ctx, a0);
        if (sb > 0 && sb < 64) c = (int64_t)((uint64_t)c & ((1ull << sb) - 1ull));
        *val = c; return true;
    }
    case OP_TYPE_LLVM_TRUNC: {
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, def, 0);
        int64_t c = 0;
        if (!is_const(ctx, MLIR_GetValueDefiningOp(ctx, a0), &c)) return false;
        int rb = int_bits(ctx, MLIR_GetOpResult(ctx, def, 0));
        if (rb > 0 && rb < 64) c = (int64_t)((uint64_t)c & ((1ull << rb) - 1ull));
        *val = c; return true;
    }
    default:
        return false;
    }
}

// Float width (in bytes) of a value's type: 4 for f32, 8 for f64, else 0.
static int fp_bytes(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_TypeHandle t = MLIR_GetValueType(ctx, v);
    if (MLIR_GetTypeKind(ctx, t) != MLIR_TYPE_KIND_FLOAT) return 0;
    uint32_t w = MLIR_GetTypeWidth(ctx, t);
    return (w == 32 || w == 64) ? (int)(w / 8) : 0;
}

// A floating-point `llvm.mlir.constant`: read its value as a raw bit pattern
// (f32 zero-extended into the low 32 bits). Returns false for non-FP / non-const
// values. The integer accessor used by is_const would misread a float attribute,
// so FP constants are materialized here as their exact IEEE-754 bits.
static bool fp_const_val(MLIR_Context *ctx, MLIR_ValueHandle v, uint64_t *bits) {
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
    if (def == MLIR_INVALID_HANDLE || MLIR_GetOpType(ctx, def) != OP_TYPE_LLVM_MLIR_CONSTANT) return false;
    int fb = fp_bytes(ctx, v);
    if (fb == 0) return false;
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, def, MLIR_KEY_VALUE);
    if (a == MLIR_INVALID_HANDLE) return false;
    double d = MLIR_GetAttributeFloat(ctx, a);
    if (fb == 4) { float f = (float)d; uint32_t b32; memcpy(&b32, &f, 4); *bits = b32; }
    else         { memcpy(bits, &d, 8); }
    return true;
}

// SymbolRef attributes print with a leading '@' (e.g. "@malloc"); function
// sym_name attributes do not. Strip it so the two match.
static string strip_at(string s) {
    if (s.size && s.str[0] == '@') { s.str++; s.size--; }
    return s;
}

// Forward decls: defined in mlir_llvm_to_x64_emit.inc.
static bool x64_ensure_global_in_gmap(FnCtx *F, string gn, uint32_t *out_off);
static void movq_xmm_gpr(FnCtx *F, uint8_t xmm, uint8_t gpr);
static void movq_gpr_xmm(FnCtx *F, uint8_t gpr, uint8_t xmm);

static bool x64_is_wasm_data_sym(string gn) {
    return gn.size >= 8 && memcmp(gn.str, "__wasm_", 7) == 0;
}

static bool op_is_addressof(MLIR_Context *ctx, MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) return false;
    return MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_MLIR_ADDRESSOF;
}

// Load an operand value into `reg`, using caller-supplied width `w` (0=32b, 1=64b).
static void load_val_w(FnCtx *F, MLIR_ValueHandle v, uint8_t reg, int w) {
    // Fast path: mlir_regalloc_fast never puts rematerialized ops (constant,
    // zero, null, undef, addressof) in rm or spill, so check those dense maps
    // before MLIR_GetValueDefiningOp.
    if (F->fast_ra) {
        uint8_t rH;
        if (val_in_reg(F, v, &rH)) { mov_rr(F, reg, rH, w); return; }
        int32_t slot;
        if (F->has_spill && sm_get_fast(&F->spill, v, &slot)) {
            load_rbp(F, reg, (slot + 1) * 8, w);
            return;
        }
    }
    MLIR_Context *ctx = F->ctx;
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
    if (def != MLIR_INVALID_HANDLE) {
        MLIR_OpType dt = MLIR_GetOpType(ctx, def);
        if (dt == OP_TYPE_LLVM_MLIR_CONSTANT) {
            MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, def, MLIR_KEY_VALUE);
            if (a != MLIR_INVALID_HANDLE) {
                int fb = fp_bytes(ctx, v);
                if (fb) {
                    double d = MLIR_GetAttributeFloat(ctx, a);
                    uint64_t fbits;
                    if (fb == 4) { float f = (float)d; uint32_t b32; memcpy(&b32, &f, 4); fbits = b32; }
                    else         { memcpy(&fbits, &d, 8); }
                    mov_ri(F, reg, (int64_t)fbits, 1);
                    return;
                }
                mov_ri(F, reg, MLIR_GetAttributeInteger(ctx, a), w);
                return;
            }
        } else if (dt == OP_TYPE_LLVM_MLIR_ZERO || dt == OP_TYPE_LLVM_MLIR_NULL ||
                   dt == OP_TYPE_LLVM_MLIR_UNDEF) {
            mov_ri(F, reg, 0, w);
            return;
        } else if (dt == OP_TYPE_LLVM_SEXT || dt == OP_TYPE_LLVM_ZEXT || dt == OP_TYPE_LLVM_TRUNC) {
            int64_t cv = 0;
            if (is_const(ctx, def, &cv)) {
                mov_ri(F, reg, cv, w);
                return;
            }
        } else if (dt == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
            MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, def, MLIR_KEY_GLOBAL_NAME);
            if (a == MLIR_INVALID_HANDLE) a = MLIR_GetOpAttr(ctx, def, MLIR_KEY_VALUE);
            string gn = strip_at(a ? MLIR_GetAttributeString(ctx, a) : (string){0});
            if (!gn.size) { fail(F, "addressof missing symbol", (string){0}); return; }
            uint32_t off;
            if (gmap_get(F->gm, gn, &off) || x64_ensure_global_in_gmap(F, gn, &off))
                lea_rip_data(F, reg, off);
            else if (F->wasm_lifted && x64_is_wasm_data_sym(gn))
                fail(F, "wasm global missing from data map", gn);
            else
                lea_rip_func(F, reg, gn);   // function / external symbol
            return;
        } else if (dt == OP_TYPE_LLVM_ALLOCA) {
            int32_t region;
            if (vmap_get(&F->slots, x64_region_key(def), &region)) {
                lea_rbp(F, reg, region);
                return;
            }
        }
    }
    if (!F->fast_ra) {
        uint8_t xH;
        if (val_in_xmm(F, v, &xH)) { movq_gpr_xmm(F, reg, xH); return; }
        uint8_t rH;
        if (val_in_reg(F, v, &rH)) { mov_rr(F, reg, rH, w); return; }
        int32_t slot;
        if (F->has_spill && sm_get_fast(&F->spill, v, &slot)) {
            load_rbp(F, reg, (slot + 1) * 8, w);
            return;
        }
    }
    int32_t slot;
    if (!vmap_get(&F->slots, v, &slot)) { fail(F, "value with no slot", (string){0}); return; }
    load_rbp(F, reg, slot, w);
}
static inline void load_val(FnCtx *F, MLIR_ValueHandle v, uint8_t reg) {
    load_val_w(F, v, reg, val_w(F->ctx, v));
}
static void store_val_w(FnCtx *F, MLIR_ValueHandle v, uint8_t reg, int w) {
    uint8_t xH;
    if (val_in_xmm(F, v, &xH)) { movq_xmm_gpr(F, xH, reg); return; }
    uint8_t rH;
    if (val_in_reg(F, v, &rH)) { mov_rr(F, rH, reg, w); return; }
    int32_t slot;
    if (F->has_spill && sm_get_fast(&F->spill, v, &slot)) {
        store_rbp(F, (slot + 1) * 8, reg, w);
        return;
    }
    if (!vmap_get(&F->slots, v, &slot)) { fail(F, "result with no slot", (string){0}); return; }
    store_rbp(F, slot, reg, w);
}
static inline void store_val(FnCtx *F, MLIR_ValueHandle v, uint8_t reg) {
    store_val_w(F, v, reg, val_w(F->ctx, v));
}

static inline bool x64_is_remat_type(MLIR_OpType t) {
    return t == OP_TYPE_LLVM_MLIR_CONSTANT || t == OP_TYPE_LLVM_MLIR_ADDRESSOF ||
           t == OP_TYPE_LLVM_MLIR_UNDEF    || t == OP_TYPE_LLVM_MLIR_ZERO ||
           t == OP_TYPE_LLVM_MLIR_NULL;
}
static inline uint8_t val_reg_or_load(FnCtx *F, MLIR_ValueHandle v, uint8_t scratch, int w) {
    uint8_t rH;
    if (F->fast_ra) {
        if (val_in_reg(F, v, &rH)) return rH;
    } else {
        uint8_t xH;
        if (!val_in_xmm(F, v, &xH)) {
            MLIR_OpHandle def = MLIR_GetValueDefiningOp(F->ctx, v);
            int64_t dummy;
            if ((def == MLIR_INVALID_HANDLE ||
                 (!x64_is_remat_type(MLIR_GetOpType(F->ctx, def)) && !is_const(F->ctx, def, &dummy))) &&
                val_in_reg(F, v, &rH))
                return rH;
        }
    }
    load_val_w(F, v, scratch, w);
    return scratch;
}
static inline bool alloca_slot(FnCtx *F, MLIR_ValueHandle v, int32_t *out_off) {
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(F->ctx, v);
    if (def == MLIR_INVALID_HANDLE || MLIR_GetOpType(F->ctx, def) != OP_TYPE_LLVM_ALLOCA)
        return false;
    return vmap_get(&F->slots, x64_region_key(def), out_off);
}

#include "mlir_llvm_to_x64_emit.inc"
