// Op selection + function emission + module orchestration for the x86_64 ELF
// backend. Included at the end of mlir_llvm_to_x64.c (shares its statics).

// --- small attribute helpers ----------------------------------------------
static string attr_str(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, key);
    return a ? MLIR_GetAttributeString(ctx, a) : (string){0};
}
static bool str_eqz(string s, const char *c) {
    size_t l = strlen(c);
    return s.size == l && memcmp(s.str, c, l) == 0;
}

// width in bytes of an integer/pointer/float-typed value
static int type_bytes(MLIR_Context *ctx, MLIR_ValueHandle v) {
    int fb = fp_bytes(ctx, v);
    if (fb) return fb;                // f32 -> 4, f64 -> 8
    int b = int_bits(ctx, v);
    if (b == 0) return 8;           // pointer
    return (b + 7) / 8;
}

// --- LLVM-type layout (mlir_llvm_type_layout.h; LP64: pointers are 8 bytes) --
// Direct-mapped cache for composite (struct/array) size & alignment so GEP
// lowering doesn't re-walk nested struct fields on every instruction.
#define X64_TY_CACHE_SZ 512u
typedef struct {
    MLIR_Context *ctx;
    MLIR_TypeHandle ty;
    unsigned size;
    unsigned align;
} X64TyCacheEnt;
static X64TyCacheEnt g_x64_ty_cache[X64_TY_CACHE_SZ];

static unsigned x64_type_align(MLIR_Context *ctx, MLIR_TypeHandle ty);
static unsigned x64_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    MLIR_TypeKind k = MLIR_GetTypeKind(ctx, ty);
    if (k == MLIR_TYPE_KIND_LLVM_POINTER) return 8;
    if (k == MLIR_TYPE_KIND_INTEGER) {
        uint32_t w = MLIR_GetTypeWidth(ctx, ty);
        if (w == 1 || w == 8) return 1;
        if (w == 16 || w == 32 || w == 64) return w / 8;
        return 0;
    }
    if (k == MLIR_TYPE_KIND_FLOAT) {
        uint32_t w = MLIR_GetTypeWidth(ctx, ty);
        return (w == 32 || w == 64) ? w / 8 : 0;
    }
    size_t slot = (((uintptr_t)ty * 2654435761u) >> 4) & (X64_TY_CACHE_SZ - 1);
    X64TyCacheEnt *e = &g_x64_ty_cache[slot];
    if (e->ctx == ctx && e->ty == ty) return e->size;
    unsigned sz = 0, al = 1;
    if (k == MLIR_TYPE_KIND_LLVM_ARRAY) {
        MLIR_TypeHandle et = MLIR_GetTypeElement(ctx, ty);
        unsigned esz = x64_type_size(ctx, et);
        al = x64_type_align(ctx, et);
        sz = esz ? esz * (unsigned)MLIR_GetTypeNumElements(ctx, ty) : 0;
    } else if (k == MLIR_TYPE_KIND_LLVM_STRUCT) {
        size_t nf = MLIR_GetTypeLLVMStructNumFields(ctx, ty);
        unsigned off = 0, ma = 1;
        bool ok = true;
        for (size_t i = 0; i < nf; i++) {
            MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ctx, ty, i);
            unsigned fsz = x64_type_size(ctx, ft);
            unsigned fal = x64_type_align(ctx, ft);
            if ((fsz == 0 && !llvm_type_is_empty(ctx, ft, 8)) || fal == 0) {
                ok = false; break;
            }
            off = llvm_align_up(off, fal) + fsz;
            if (fal > ma) ma = fal;
        }
        al = ma;
        sz = ok ? llvm_align_up(off, ma) : 0;
    } else {
        sz = llvm_type_size(ctx, ty, 8);
        al = sz ? sz : 1;
    }
    e->ctx = ctx; e->ty = ty; e->size = sz; e->align = al;
    return sz;
}
static unsigned x64_type_align(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    MLIR_TypeKind k = MLIR_GetTypeKind(ctx, ty);
    if (k == MLIR_TYPE_KIND_LLVM_POINTER) return 8;
    if (k == MLIR_TYPE_KIND_INTEGER) {
        uint32_t w = MLIR_GetTypeWidth(ctx, ty);
        if (w == 1 || w == 8) return 1;
        if (w == 16 || w == 32 || w == 64) return w / 8;
        return 1;
    }
    if (k == MLIR_TYPE_KIND_FLOAT) {
        uint32_t w = MLIR_GetTypeWidth(ctx, ty);
        return (w == 32 || w == 64) ? w / 8 : 1;
    }
    size_t slot = (((uintptr_t)ty * 2654435761u) >> 4) & (X64_TY_CACHE_SZ - 1);
    X64TyCacheEnt *e = &g_x64_ty_cache[slot];
    if (e->ctx == ctx && e->ty == ty) return e->align;
    (void)x64_type_size(ctx, ty);
    return e->align;
}
#define X64_FLD_CACHE_SZ 256
typedef struct {
    MLIR_Context *ctx;
    MLIR_TypeHandle sty;
    uint32_t fld;
    uint32_t off;
    MLIR_TypeHandle fty;
} X64FldCacheEnt;
static X64FldCacheEnt g_x64_fld_cache[X64_FLD_CACHE_SZ];

static unsigned x64_struct_field_info(MLIR_Context *ctx, MLIR_TypeHandle sty, size_t fld,
                                      MLIR_TypeHandle *out_fty) {
    size_t slot = ((((uintptr_t)sty * 2654435761u) ^ ((uintptr_t)fld * 40503u))) & (X64_FLD_CACHE_SZ - 1);
    X64FldCacheEnt *e = &g_x64_fld_cache[slot];
    if (e->ctx == ctx && e->sty == sty && e->fld == (uint32_t)fld) {
        if (out_fty) *out_fty = e->fty;
        return e->off;
    }
    unsigned off = 0;
    MLIR_TypeHandle ft = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i <= fld; i++) {
        ft = MLIR_GetTypeLLVMStructField(ctx, sty, i);
        off = llvm_align_up(off, x64_type_align(ctx, ft));
        if (i == fld) break;
        off += x64_type_size(ctx, ft);
    }
    e->ctx = ctx; e->sty = sty; e->fld = (uint32_t)fld; e->off = off; e->fty = ft;
    if (out_fty) *out_fty = ft;
    return off;
}

// imul dst, src, imm32   (REX.W 69 /r id)
static void imul_ri(FnCtx *F, uint8_t dst, uint8_t src, int32_t imm, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_IMUL_RI,
                  (int64_t)dst | ((int64_t)src << 4) | ((int64_t)w << 8) | ((int64_t)(uint32_t)imm << 32));
        return;
    }
    rex(F, w, dst, 0, src); eb(F, 0x69); modrm(F, 3, dst, src); e32(F, (uint32_t)imm);
}
// reg += off: folds into `add reg, imm32` when it fits, else materialises the
// 64-bit offset in rcx (a scratch reg) and adds that.
static void add_const_off(FnCtx *F, uint8_t reg, int64_t off) {
    if (off == 0) return;
    if (off >= INT32_MIN && off <= INT32_MAX) {
        x64_emit1(F, OP_TYPE_X64_ADD_RI,
                  (int64_t)reg | ((int64_t)(uint32_t)(int32_t)off << 32));
    } else {
        mov_ri(F, RCX, off, 1);
        alu_rr(F, 0x01, reg, RCX, 1);
    }
}

// llvm.icmp predicate -> setcc/jcc condition code
static int icmp_cc(int64_t pred) {
    switch (pred) {
        case 0: return CC_E;  case 1: return CC_NE;
        case 2: return CC_L;  case 3: return CC_LE; case 4: return CC_G; case 5: return CC_GE;
        case 6: return CC_B;  case 7: return CC_BE; case 8: return CC_A; case 9: return CC_AE;
    }
    return CC_E;
}

static BrFix *g_x64_fix = NULL;
static size_t g_x64_fix_cap = 0;

static void add_fix(FnCtx *F, int target_block, uint32_t field_off) {
    if (F->n_fix == F->c_fix) {
        size_t ncap = F->c_fix ? F->c_fix * 2 : 64;
        size_t actual = 0;
        BrFix *nb = (BrFix *)buddy_alloc(ncap * sizeof(BrFix), &actual);
        if (F->n_fix) memcpy(nb, F->fix, F->n_fix * sizeof(BrFix));
        if (g_x64_fix) buddy_free(g_x64_fix);
        g_x64_fix = nb;
        g_x64_fix_cap = actual / sizeof(BrFix);
        F->fix = g_x64_fix;
        F->c_fix = g_x64_fix_cap;
    }
    F->fix[F->n_fix].target_block = target_block;
    F->fix[F->n_fix].field_off = field_off;
    F->n_fix++;
}

// Patch a rel32 field emitted at `field_off` to branch to `target_off`.
static void patch_rel32_at(FnCtx *F, uint32_t field_off, uint32_t target_off) {
    int32_t rel = (int32_t)target_off - (int32_t)(field_off + 4);
    buf_patch_le32(&F->f->code, field_off, (uint32_t)rel);
}

// block index of a successor block within this function's single region (O(1))
static int block_index(MLIR_Context *ctx, MLIR_RegionHandle reg, MLIR_BlockHandle blk) {
    (void)reg;
    size_t idx = MLIR_GetBlockIndex(ctx, blk);
    return idx == (size_t)-1 ? -1 : (int)idx;
}

// emit setcc al ; movzx eax, al
static void setcc_al(FnCtx *F, int cc) {
    if (F->x64_blk) { x64_emit1(F, OP_TYPE_X64_SETCC_AL, cc); return; }
    eb(F, 0x0F); eb(F, (uint8_t)(0x90 + cc)); modrm(F, 3, 0, RAX);   // setcc al
    eb(F, 0x0F); eb(F, 0xB6); modrm(F, 3, RAX, RAX);                 // movzx eax, al
}

// cmovnz dst, src
static void cmovnz(FnCtx *F, uint8_t dst, uint8_t src, int w) {
    x64_emit1(F, OP_TYPE_X64_CMOVCC,
              (int64_t)dst | ((int64_t)src << 4) | ((int64_t)w << 8) | ((int64_t)CC_NE << 12));
}
static void cmovcc(FnCtx *F, uint8_t dst, uint8_t src, int cc, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_CMOVCC,
                  (int64_t)dst | ((int64_t)src << 4) | ((int64_t)w << 8) | ((int64_t)cc << 12));
        return;
    }
    rex(F, w, dst, 0, src); eb(F, 0x0F); eb(F, (uint8_t)(0x40 + cc)); modrm(F, 3, dst, src);
}

// ---------------------------------------------------------------------------
// SSE2 scalar floating-point (real FP support, issue #212 Option A).
//
// Double/float SSA values flow through GP regs / 8-byte stack slots as opaque
// bits (the same path integers use). Each FP op transiently moves operand bits
// GP -> xmm with `movq`, computes with a real SSE2 scalar instruction, and moves
// the result xmm -> GP with `movq`. xmm0/xmm1 are pure scratch and never live
// across ops, so no xmm allocation / spill model is required.
// ---------------------------------------------------------------------------

// movq xmm, r/m64   (66 REX.W 0F 6E /r): GP -> xmm (low 64 bits)
static void movq_xmm_gpr(FnCtx *F, uint8_t xmm, uint8_t gpr) {
    if (F->x64_blk) { x64_emit1(F, OP_TYPE_X64_MOVQ_XMM_GPR, (int64_t)xmm | ((int64_t)gpr << 4)); return; }
    eb(F, 0x66); rex(F, 1, xmm, 0, gpr); eb(F, 0x0F); eb(F, 0x6E); modrm(F, 3, xmm, gpr);
}
// movq r/m64, xmm   (66 REX.W 0F 7E /r): xmm (low 64 bits) -> GP
static void movq_gpr_xmm(FnCtx *F, uint8_t gpr, uint8_t xmm) {
    if (F->x64_blk) { x64_emit1(F, OP_TYPE_X64_MOVQ_GPR_XMM, (int64_t)gpr | ((int64_t)xmm << 4)); return; }
    eb(F, 0x66); rex(F, 1, xmm, 0, gpr); eb(F, 0x0F); eb(F, 0x7E); modrm(F, 3, xmm, gpr);
}
// scalar double op: F2 0F <opc> /r   (xd = xd OP xs)
static void sse_sd(FnCtx *F, uint8_t opc, uint8_t xd, uint8_t xs) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_SSE_OP, (int64_t)xd | ((int64_t)xs << 4) | ((int64_t)opc << 12));
        return;
    }
    eb(F, 0xF2); rex(F, 0, xd, 0, xs); eb(F, 0x0F); eb(F, opc); modrm(F, 3, xd, xs);
}
// scalar single op: F3 0F <opc> /r
static void sse_ss(FnCtx *F, uint8_t opc, uint8_t xd, uint8_t xs) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_SSE_OP, (int64_t)xd | ((int64_t)xs << 4) | (1ll << 8) | ((int64_t)opc << 12));
        return;
    }
    eb(F, 0xF3); rex(F, 0, xd, 0, xs); eb(F, 0x0F); eb(F, opc); modrm(F, 3, xd, xs);
}
static void sse_mov_rr(FnCtx *F, uint8_t xd, uint8_t xs, int is_ss) {
    if (xd == xs) return;
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_SSE_MOV_RR, (int64_t)xd | ((int64_t)xs << 4) | ((int64_t)(is_ss ? 1 : 0) << 8));
        return;
    }
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xd, 0, xs); eb(F, 0x0F); eb(F, 0x10); modrm(F, 3, xd, xs);
}
static void sse_xorpd(FnCtx *F, uint8_t xd, uint8_t xs) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_SSE_XORPD, (int64_t)xd | ((int64_t)xs << 4));
        return;
    }
    eb(F, 0x66); rex(F, 0, xd, 0, xs); eb(F, 0x0F); eb(F, 0x57); modrm(F, 3, xd, xs);
}
static void load_rbp_xmm(FnCtx *F, uint8_t xd, int32_t off, int is_ss) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_LOAD_RBP_XMM,
                  (int64_t)xd | ((int64_t)(is_ss ? 1 : 0) << 4) | ((int64_t)(uint32_t)off << 32));
        return;
    }
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xd, 0, RBP); eb(F, 0x0F); eb(F, 0x10); modrm_mem_disp(F, xd, RBP, -off);
}
static void store_rbp_xmm(FnCtx *F, int32_t off, uint8_t xs, int is_ss) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_STORE_RBP_XMM,
                  (int64_t)xs | ((int64_t)(is_ss ? 1 : 0) << 4) | ((int64_t)(uint32_t)off << 32));
        return;
    }
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xs, 0, RBP); eb(F, 0x0F); eb(F, 0x11); modrm_mem_disp(F, xs, RBP, -off);
}
static void load_mem_xmm(FnCtx *F, uint8_t xd, uint8_t addr, int is_ss) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_LOAD_MEM_XMM,
                  (int64_t)xd | ((int64_t)addr << 4) | ((int64_t)(is_ss ? 1 : 0) << 8));
        return;
    }
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xd, 0, addr); eb(F, 0x0F); eb(F, 0x10); modrm_mem0(F, xd, addr);
}
static void store_mem_xmm(FnCtx *F, uint8_t addr, uint8_t xs, int is_ss) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_STORE_MEM_XMM,
                  (int64_t)addr | ((int64_t)xs << 4) | ((int64_t)(is_ss ? 1 : 0) << 8));
        return;
    }
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xs, 0, addr); eb(F, 0x0F); eb(F, 0x11); modrm_mem0(F, xs, addr);
}
static void load_mem_xmm_disp(FnCtx *F, uint8_t xd, uint8_t base, int is_ss, int32_t disp) {
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xd, 0, base); eb(F, 0x0F); eb(F, 0x10); modrm_mem_disp(F, xd, base, disp);
}
static void store_mem_xmm_disp(FnCtx *F, uint8_t base, uint8_t xs, int is_ss, int32_t disp) {
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xs, 0, base); eb(F, 0x0F); eb(F, 0x11); modrm_mem_disp(F, xs, base, disp);
}
static void load_mem_xmm_sib(FnCtx *F, uint8_t xd, uint8_t base, uint8_t idx, int scale, int is_ss) {
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xd, idx, base); eb(F, 0x0F); eb(F, 0x10); modrm_mem_sib(F, xd, base, idx, scale);
}
static void store_mem_xmm_sib(FnCtx *F, uint8_t base, uint8_t idx, uint8_t xs, int scale, int is_ss) {
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xs, idx, base); eb(F, 0x0F); eb(F, 0x11); modrm_mem_sib(F, xs, base, idx, scale);
}
static void sse_op_mem_sib(FnCtx *F, uint8_t opc, uint8_t xd, uint8_t base, uint8_t idx, int scale, int is_ss) {
    eb(F, is_ss ? 0xF3 : 0xF2); rex(F, 0, xd, idx, base); eb(F, 0x0F); eb(F, opc); modrm_mem_sib(F, xd, base, idx, scale);
}
// Load an FP value directly into XMM register `xdst` (uses RAX as scratch only for non-zero constants).
static void load_xmm(FnCtx *F, MLIR_ValueHandle v, uint8_t xdst, int is_ss) {
    uint8_t xH;
    if (val_in_xmm(F, v, &xH)) { sse_mov_rr(F, xdst, xH, is_ss); return; }
    uint64_t fbits = 0;
    if (fp_const_val(F->ctx, v, &fbits)) {
        if (fbits == 0) { sse_xorpd(F, xdst, xdst); return; }
        mov_ri(F, RAX, (int64_t)fbits, 1);
        movq_xmm_gpr(F, xdst, RAX);
        return;
    }
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(F->ctx, v);
    if (def != MLIR_INVALID_HANDLE) {
        MLIR_OpType dt = MLIR_GetOpType(F->ctx, def);
        if (dt == OP_TYPE_LLVM_MLIR_ZERO || dt == OP_TYPE_LLVM_MLIR_UNDEF) {
            sse_xorpd(F, xdst, xdst);
            return;
        }
    }
    uint8_t rH;
    if (val_in_reg(F, v, &rH)) { movq_xmm_gpr(F, xdst, rH); return; }
    int32_t slot;
    if (F->has_spill && sm_get_fast(&F->spill, v, &slot)) {
        load_rbp_xmm(F, xdst, (slot + 1) * 8, is_ss);
        return;
    }
    if (vmap_get(&F->slots, v, &slot)) {
        load_rbp_xmm(F, xdst, slot, is_ss);
        return;
    }
    load_val(F, v, RAX);
    movq_xmm_gpr(F, xdst, RAX);
}
static void store_xmm(FnCtx *F, MLIR_ValueHandle v, uint8_t xsrc, int is_ss) {
    uint8_t xH;
    if (val_in_xmm(F, v, &xH)) { sse_mov_rr(F, xH, xsrc, is_ss); return; }
    uint8_t rH;
    if (val_in_reg(F, v, &rH)) { movq_gpr_xmm(F, rH, xsrc); return; }
    int32_t slot;
    if (F->has_spill && sm_get_fast(&F->spill, v, &slot)) {
        store_rbp_xmm(F, (slot + 1) * 8, xsrc, is_ss);
        return;
    }
    if (vmap_get(&F->slots, v, &slot)) {
        store_rbp_xmm(F, slot, xsrc, is_ss);
        return;
    }
    movq_gpr_xmm(F, RAX, xsrc);
    store_val(F, v, RAX);
}
static inline uint8_t val_xmm_or_load(FnCtx *F, MLIR_ValueHandle v, uint8_t xscratch, int is_ss) {
    uint8_t xH;
    if (val_in_xmm(F, v, &xH)) return xH;
    load_xmm(F, v, xscratch, is_ss);
    return xscratch;
}
// cvtsi2sd/ss xmm, r/m (pfx [REX.W] 0F 2A /r): signed int (GP) -> float (xmm)
static void cvtsi2f(FnCtx *F, uint8_t pfx, uint8_t xmm, uint8_t gpr, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_CVTSI2F,
                  (int64_t)xmm | ((int64_t)gpr << 4) | ((int64_t)w << 8) | ((int64_t)pfx << 12));
        return;
    }
    eb(F, pfx); rex(F, w, xmm, 0, gpr); eb(F, 0x0F); eb(F, 0x2A); modrm(F, 3, xmm, gpr);
}
// cvttsd2si/ss r, xmm (pfx [REX.W] 0F 2C /r): float (xmm) -> signed int (GP), trunc
static void cvtf2si(FnCtx *F, uint8_t pfx, uint8_t gpr, uint8_t xmm, int w) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_CVTF2SI,
                  (int64_t)gpr | ((int64_t)xmm << 4) | ((int64_t)w << 8) | ((int64_t)pfx << 12));
        return;
    }
    eb(F, pfx); rex(F, w, gpr, 0, xmm); eb(F, 0x0F); eb(F, 0x2C); modrm(F, 3, gpr, xmm);
}
// cvtss2sd / cvtsd2ss xmm, xmm (pfx 0F 5A /r): float-width convert
static void cvt_f2f(FnCtx *F, uint8_t pfx, uint8_t xd, uint8_t xs) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_CVT_F2F,
                  (int64_t)xd | ((int64_t)xs << 4) | ((int64_t)pfx << 12));
        return;
    }
    eb(F, pfx); rex(F, 0, xd, 0, xs); eb(F, 0x0F); eb(F, 0x5A); modrm(F, 3, xd, xs);
}
// ucomisd (66 0F 2E) / ucomiss (0F 2E): ordered compare, sets ZF/PF/CF
static void ucomis(FnCtx *F, int is_f32, uint8_t x1, uint8_t x2) {
    if (F->x64_blk) {
        x64_emit1(F, OP_TYPE_X64_UCOMIS,
                  (int64_t)x1 | ((int64_t)x2 << 4) | ((int64_t)(is_f32 ? 1 : 0) << 8));
        return;
    }
    if (!is_f32) eb(F, 0x66);
    rex(F, 0, x1, 0, x2); eb(F, 0x0F); eb(F, 0x2E); modrm(F, 3, x1, x2);
}
// setcc into the low byte of AL/CL (rm in {RAX,RCX})
static void setcc8(FnCtx *F, int cc, uint8_t rm) {
    if (F->x64_blk) { x64_emit1(F, OP_TYPE_X64_SETCC8, (int64_t)cc | ((int64_t)rm << 4)); return; }
    eb(F, 0x0F); eb(F, (uint8_t)(0x90 + cc)); modrm(F, 3, 0, rm);
}

// setcc opcode low-nibble values not in the shared CC enum
enum { CC_P = 0xA, CC_NP = 0xB };

// llvm.fcmp: operands' bit patterns are already in xmm0 (a0) and xmm1 (a1);
// leave a 0/1 i32 result in EAX. Handles all 16 LLVM FCmp predicates, mapping
// each to ucomisd/ucomiss (with operand swap where needed) plus a one- or
// two-setcc sequence so ordered/unordered (NaN) semantics are correct.
static void emit_fcmp_regs(FnCtx *F, int64_t pred, int is_f32, uint8_t x0, uint8_t x1) {
    if (pred == 0)  { mov_ri(F, RAX, 0, 0); return; }   // _false
    if (pred == 15) { mov_ri(F, RAX, 1, 0); return; }   // _true
    bool swap = false; int cc1, combine = 0, cc2 = 0;   // combine: 0 none, 1 and, 2 or
    switch (pred) {
        case 1:  cc1 = CC_E;  combine = 1; cc2 = CC_NP; break;  // oeq: sete & setnp
        case 2:  cc1 = CC_A;  break;                            // ogt
        case 3:  cc1 = CC_AE; break;                            // oge
        case 4:  swap = true; cc1 = CC_A;  break;               // olt
        case 5:  swap = true; cc1 = CC_AE; break;               // ole
        case 6:  cc1 = CC_NE; combine = 1; cc2 = CC_NP; break;  // one: setne & setnp
        case 7:  cc1 = CC_NP; break;                            // ord
        case 8:  cc1 = CC_E;  combine = 2; cc2 = CC_P;  break;  // ueq: sete | setp
        case 9:  swap = true; cc1 = CC_B;  break;               // ugt
        case 10: swap = true; cc1 = CC_BE; break;               // uge
        case 11: cc1 = CC_B;  break;                            // ult
        case 12: cc1 = CC_BE; break;                            // ule
        case 13: cc1 = CC_NE; combine = 2; cc2 = CC_P;  break;  // une: setne | setp
        case 14: cc1 = CC_P;  break;                            // uno
        default: cc1 = CC_E;  break;
    }
    ucomis(F, is_f32, swap ? x1 : x0, swap ? x0 : x1);
    setcc8(F, cc1, RAX);
    if (combine) {
        setcc8(F, cc2, RCX);
        x64_emit1(F, OP_TYPE_X64_ALU8_AL_CL, combine == 1 ? 0x20 : 0x08);
    }
    x64_emit0(F, OP_TYPE_X64_MOVZX_EAX_AL);
}

// Lower an FP op (arith / convert / compare). Returns false if `ty` is not an
// FP op handled here. Values allocated in F->xrm stay in xmm2..xmm15; xmm0/xmm1 scratch.
static bool lower_fp_op(FnCtx *F, MLIR_OpHandle op, MLIR_OpType ty) {
    MLIR_Context *ctx = F->ctx;
    switch (ty) {
    // arithmetic: dst = op0 OP op1
    case OP_TYPE_LLVM_FADD: case OP_TYPE_LLVM_FSUB:
    case OP_TYPE_LLVM_FMUL: case OP_TYPE_LLVM_FDIV: {
        uint8_t opc = ty == OP_TYPE_LLVM_FADD ? 0x58 : ty == OP_TYPE_LLVM_FSUB ? 0x5C :
                      ty == OP_TYPE_LLVM_FMUL ? 0x59 : 0x5E;
        bool comm = (ty == OP_TYPE_LLVM_FADD || ty == OP_TYPE_LLVM_FMUL);
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, op, 1);
        int is_ss = (fp_bytes(ctx, r) == 4);
        uint8_t xD = 0, x0 = 0, x1 = 0;
        bool has_xD = val_in_xmm(F, r, &xD);
        bool has_x0 = val_in_xmm(F, a0, &x0);
        bool has_x1 = val_in_xmm(F, a1, &x1);
        if (has_xD) {
            if (has_x0 && x0 == xD) {
                uint8_t rhs = val_xmm_or_load(F, a1, 1, is_ss);
                if (is_ss) sse_ss(F, opc, xD, rhs); else sse_sd(F, opc, xD, rhs);
            } else if (comm && has_x1 && x1 == xD) {
                uint8_t rhs = val_xmm_or_load(F, a0, 1, is_ss);
                if (is_ss) sse_ss(F, opc, xD, rhs); else sse_sd(F, opc, xD, rhs);
            } else if (!has_x1 || x1 != xD) {
                load_xmm(F, a0, xD, is_ss);
                uint8_t rhs = val_xmm_or_load(F, a1, 1, is_ss);
                if (is_ss) sse_ss(F, opc, xD, rhs); else sse_sd(F, opc, xD, rhs);
            } else {
                // Non-commutative and a1 lives in xD: compute in xmm0 then move to xD.
                load_xmm(F, a0, 0, is_ss);
                if (is_ss) sse_ss(F, opc, 0, xD); else sse_sd(F, opc, 0, xD);
                sse_mov_rr(F, xD, 0, is_ss);
            }
        } else {
            load_xmm(F, a0, 0, is_ss);
            uint8_t rhs = val_xmm_or_load(F, a1, 1, is_ss);
            if (is_ss) sse_ss(F, opc, 0, rhs); else sse_sd(F, opc, 0, rhs);
            store_xmm(F, r, 0, is_ss);
        }
        return true;
    }
    // fneg: flip the sign bit in the integer domain (exact, no xmm needed)
    case OP_TYPE_LLVM_FNEG: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        int fb = fp_bytes(ctx, r);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);
        mov_ri(F, RCX, (fb == 4) ? (int64_t)0x80000000ll : (int64_t)0x8000000000000000ll, 1);
        alu_rr(F, 0x31, RAX, RCX, 1);          // xor rax, rcx
        store_val(F, r, RAX);
        return true;
    }
    // float -> signed/unsigned int (truncating)
    case OP_TYPE_LLVM_FPTOSI: case OP_TYPE_LLVM_FPTOUI: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(ctx, op, 0);
        int sfb = fp_bytes(ctx, s);
        uint8_t pfx = sfb == 4 ? 0xF3 : 0xF2;
        int conv_w = ty == OP_TYPE_LLVM_FPTOUI ? 1 : val_w(ctx, r);
        uint8_t xs = val_xmm_or_load(F, s, 0, sfb == 4);
        cvtf2si(F, pfx, RAX, xs, conv_w);
        if (ty == OP_TYPE_LLVM_FPTOUI && val_w(ctx, r)) {
            if (xs != 0) sse_mov_rr(F, 0, xs, sfb == 4);
            mov_ri(F, RCX, sfb == 4 ? 0x5F000000ll : 0x43E0000000000000ll, 1);
            movq_xmm_gpr(F, 1, RCX);
            if (sfb == 4) sse_ss(F, 0x5C, 0, 1); else sse_sd(F, 0x5C, 0, 1);  // x - 2^63
            cvtf2si(F, pfx, RCX, 0, 1);
            mov_rr(F, RDX, RAX, 1);
            x64_emit1(F, OP_TYPE_X64_SHIFT_IMM, (int64_t)RDX | (7ll << 4) | (1ll << 8) | (63ll << 12)); // sar rdx, 63
            alu_rr(F, 0x21, RCX, RDX, 1);                                       // and rcx, rdx
            alu_rr(F, 0x09, RAX, RCX, 1);                                       // or rax, rcx
        }
        store_val(F, r, RAX);
        return true;
    }
    // signed/unsigned int -> float
    case OP_TYPE_LLVM_SITOFP: case OP_TYPE_LLVM_UITOFP: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(ctx, op, 0);
        int rfb = fp_bytes(ctx, r);
        int sb  = int_bits(ctx, s);
        int conv_w = (ty == OP_TYPE_LLVM_UITOFP || sb > 32) ? 1 : 0;
        uint8_t pfx = rfb == 4 ? 0xF3 : 0xF2;
        if (ty == OP_TYPE_LLVM_UITOFP && sb > 32) {
            load_val(F, s, RAX);
            mov_rr(F, RCX, RAX, 1);
            x64_emit1(F, OP_TYPE_X64_SHIFT_IMM, (int64_t)RCX | (5ll << 4) | (1ll << 8) | (1ll << 12)); // shr rcx, 1
            mov_ri(F, RDX, 1, 1);
            alu_rr(F, 0x21, RDX, RAX, 1);                          // and rdx, rax
            alu_rr(F, 0x09, RCX, RDX, 1);                          // or rcx, rdx
            alu_rr(F, 0x85, RAX, RAX, 1);                          // test rax, rax
            cmovcc(F, RCX, RAX, 0x9 /*ns*/, 1);
            cvtsi2f(F, pfx, 0, RCX, 1);
            movq_gpr_xmm(F, RDX, 0);
            if (rfb == 4) sse_ss(F, 0x58, 0, 0); else sse_sd(F, 0x58, 0, 0);  // f + f
            movq_gpr_xmm(F, RCX, 0);
            alu_rr(F, 0x85, RAX, RAX, 1);                          // test rax, rax
            cmovcc(F, RDX, RCX, 0x8 /*s*/, 1);
            store_val(F, r, RDX);
            return true;
        }
        uint8_t gs;
        if (ty == OP_TYPE_LLVM_UITOFP && sb <= 32) {
            load_val_w(F, s, RAX, 0);   // zero-extend u32 into 64-bit RAX
            gs = RAX;
        } else {
            gs = val_reg_or_load(F, s, RAX, conv_w);
        }
        uint8_t xD = 0;
        if (val_in_xmm(F, r, &xD)) {
            cvtsi2f(F, pfx, xD, gs, conv_w);
        } else {
            cvtsi2f(F, pfx, 0, gs, conv_w);
            store_xmm(F, r, 0, rfb == 4);
        }
        return true;
    }
    // f32 -> f64 (extend) / f64 -> f32 (truncate)
    case OP_TYPE_LLVM_FPEXT: case OP_TYPE_LLVM_FPTRUNC: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
        int src_ss = (ty == OP_TYPE_LLVM_FPEXT);
        uint8_t xs = val_xmm_or_load(F, a0, 0, src_ss);
        uint8_t xD = 0;
        if (val_in_xmm(F, r, &xD)) {
            cvt_f2f(F, src_ss ? 0xF3 : 0xF2, xD, xs);
        } else {
            cvt_f2f(F, src_ss ? 0xF3 : 0xF2, 0, xs);
            store_xmm(F, r, 0, !src_ss);
        }
        return true;
    }
    // ordered/unordered float comparison -> 0/1
    case OP_TYPE_LLVM_FCMP: {
        MLIR_ValueHandle r  = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
        int is_f32 = (fp_bytes(ctx, a0) == 4);
        int64_t pred = 0;
        MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_PREDICATE);
        if (pa) pred = MLIR_GetAttributeInteger(ctx, pa);
        uint8_t x0 = val_xmm_or_load(F, a0, 0, is_f32);
        uint8_t x1 = val_xmm_or_load(F, MLIR_GetOpOperand(ctx, op, 1), 1, is_f32);
        emit_fcmp_regs(F, pred, is_f32, x0, x1);
        store_val(F, r, RAX);
        return true;
    }
    // llvm.intr.sqrt: hardware sqrt (mirrors llvm->aarch64 fsqrt lowering).
    case OP_TYPE_LLVM_INTR_SQRT: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        int fb = fp_bytes(ctx, r);
        if (fb != 4 && fb != 8) {
            fail(F, "sqrt on non-float result", MLIR_GetOpTypeName(ctx, ty));
            return true;
        }
        uint8_t xs = val_xmm_or_load(F, MLIR_GetOpOperand(ctx, op, 0), 0, fb == 4);
        uint8_t xD = 0;
        if (val_in_xmm(F, r, &xD)) {
            if (fb == 4) sse_ss(F, 0x51, xD, xs); else sse_sd(F, 0x51, xD, xs);
        } else {
            if (fb == 4) sse_ss(F, 0x51, 0, xs); else sse_sd(F, 0x51, 0, xs);
            store_xmm(F, r, 0, fb == 4);
        }
        return true;
    }
    default: break;
    }
    return false;
}

// Recover a value's home (physical reg or rbp-relative stack slot) for parallel-move
// resolution in emit_edge(). Rematerialized constants/addressof return reg=-1, slot=0.
static void x64_val_home(FnCtx *F, MLIR_ValueHandle v, bool is_src,
                         int *out_reg, int32_t *out_slot) {
    *out_reg = -1;
    *out_slot = 0;
    if (is_src && !F->fast_ra) {
        MLIR_OpHandle def = MLIR_GetValueDefiningOp(F->ctx, v);
        int64_t dummy;
        if (def != MLIR_INVALID_HANDLE &&
            (x64_is_remat_type(MLIR_GetOpType(F->ctx, def)) || is_const(F->ctx, def, &dummy)))
            return;
    }
    uint8_t xH;
    if (val_in_xmm(F, v, &xH)) { *out_reg = 16 + (int)xH; return; }
    uint8_t rH;
    if (val_in_reg(F, v, &rH)) { *out_reg = (int)rH; return; }
    int32_t slot;
    if (F->has_spill && sm_get_fast(&F->spill, v, &slot)) {
        *out_slot = (slot + 1) * 8;
        return;
    }
    if (vmap_get(&F->slots, v, &slot)) { *out_slot = slot; return; }
}

// Emit parallel-move-safe edge copies for the `si`-th successor of `term`.
// Resolves register-to-register and register/slot copies directly without
// spilling to [rbp - edge_tmp], dropping self-copies and breaking cycles via R11.
static void emit_edge(FnCtx *F, MLIR_OpHandle term, size_t si, MLIR_BlockHandle target) {
    size_t n = MLIR_GetOpNumSuccessorOperands(F->ctx, term, si);
    if (n == 0) return;
    if (n > 64) {
        // Fallback for unusually wide edges (> 64 block args).
        for (size_t i = 0; i < n; i++) {
            MLIR_ValueHandle src = MLIR_GetOpSuccessorOperand(F->ctx, term, si, i);
            load_val(F, src, RAX);
            store_rbp(F, (int32_t)(F->edge_tmp - i * 8), RAX, 1);
        }
        for (size_t i = 0; i < n; i++) {
            MLIR_ValueHandle darg = MLIR_GetBlockArg(F->ctx, target, i);
            load_rbp(F, RAX, (int32_t)(F->edge_tmp - i * 8), 1);
            store_val(F, darg, RAX);
        }
        return;
    }

    MLIR_ValueHandle src_v[64];
    int     src_r[64], dst_r[64];
    int32_t src_s[64], dst_s[64];
    int     mv_w[64];
    uint8_t done[64];
    size_t  rem = 0;

    for (size_t i = 0; i < n; i++) {
        MLIR_ValueHandle sv = MLIR_GetOpSuccessorOperand(F->ctx, term, si, i);
        MLIR_ValueHandle dv = MLIR_GetBlockArg(F->ctx, target, i);
        src_v[i] = sv;
        x64_val_home(F, sv, true,  &src_r[i], &src_s[i]);
        x64_val_home(F, dv, false, &dst_r[i], &dst_s[i]);
        mv_w[i] = val_w(F->ctx, dv);
        if ((src_r[i] >= 0 && src_r[i] == dst_r[i]) ||
            (src_s[i] > 0  && src_s[i] == dst_s[i])) {
            done[i] = 1;
        } else {
            done[i] = 0;
            rem++;
        }
    }

    while (rem > 0) {
        bool progress = false;
        for (size_t i = 0; i < n; i++) {
            if (done[i]) continue;
            // Ready iff no other pending move reads our destination.
            bool blocked = false;
            for (size_t j = 0; j < n; j++) {
                if (done[j] || j == i) continue;
                if ((dst_r[i] >= 0 && src_r[j] == dst_r[i]) ||
                    (dst_s[i] > 0  && src_s[j] == dst_s[i])) {
                    blocked = true;
                    break;
                }
            }
            if (blocked) continue;

            int w = mv_w[i];
            int is_ss = (fp_bytes(F->ctx, MLIR_GetBlockArg(F->ctx, target, i)) == 4);
            if (src_r[i] >= 16) {
                uint8_t xs = (uint8_t)(src_r[i] - 16);
                if (dst_r[i] >= 16)      sse_mov_rr(F, (uint8_t)(dst_r[i] - 16), xs, is_ss);
                else if (dst_r[i] >= 0)  movq_gpr_xmm(F, (uint8_t)dst_r[i], xs);
                else                     store_rbp_xmm(F, dst_s[i], xs, is_ss);
            } else if (dst_r[i] >= 16) {
                uint8_t xd = (uint8_t)(dst_r[i] - 16);
                if (src_r[i] >= 0) movq_xmm_gpr(F, xd, (uint8_t)src_r[i]);
                else               load_xmm(F, src_v[i], xd, is_ss);
            } else if (src_r[i] >= 0) {
                if (dst_r[i] >= 0) mov_rr(F, (uint8_t)dst_r[i], (uint8_t)src_r[i], w);
                else               store_rbp(F, dst_s[i], (uint8_t)src_r[i], w);
            } else {
                if (dst_r[i] >= 0) {
                    load_val_w(F, src_v[i], (uint8_t)dst_r[i], w);
                } else {
                    load_val_w(F, src_v[i], RAX, w);
                    store_rbp(F, dst_s[i], RAX, w);
                }
            }
            done[i] = 1;
            rem--;
            progress = true;
        }
        if (!progress) {
            // Pure cycle: save one pending destination into R11 (GP) or xmm1 (XMM)
            // and redirect all pending reads of it.
            size_t k = 0;
            while (k < n && done[k]) k++;
            if (dst_r[k] >= 16) {
                int saved_r = dst_r[k];
                sse_mov_rr(F, 1, (uint8_t)(saved_r - 16), 0);
                for (size_t j = 0; j < n; j++) {
                    if (!done[j] && src_r[j] == saved_r) src_r[j] = 16 + 1;
                }
            } else if (dst_r[k] >= 0) {
                int saved_r = dst_r[k];
                mov_rr(F, R11, (uint8_t)saved_r, 1);
                for (size_t j = 0; j < n; j++) {
                    if (!done[j] && src_r[j] == saved_r) src_r[j] = R11;
                }
            } else {
                int32_t saved_s = dst_s[k];
                load_rbp(F, R11, saved_s, 1);
                for (size_t j = 0; j < n; j++) {
                    if (!done[j] && src_s[j] == saved_s) {
                        src_r[j] = R11;
                        src_s[j] = 0;
                    }
                }
            }
        }
    }
}

// Restore callee-saved registers, then `leave ; ret`.
static void emit_epilogue(FnCtx *F) {
    for (int j = F->n_cs - 1; j >= 0; j--)
        load_rbp(F, F->cs_reg[j], F->cs_base + 8 * (j + 1), 1);
    x64_emit0(F, OP_TYPE_X64_LEAVE);
    x64_emit0(F, OP_TYPE_X64_RET);
}

// Module-wide lowering state for one mlir_llvm_to_elf() call (set in that entry
// point, cleared at x64_done). emit_func snapshots mb/data_buf into FnCtx.
static GMap *g_x64_gmap = NULL;              // global name -> .data offset (+ size)
static bool  g_x64_wasm_lifted = false;      // module came from --from-wasm (caller-supplied)
static MLIR_BlockHandle g_x64_mb = MLIR_INVALID_HANDLE;  // module IR block (funcs/globals)
static Buf *g_x64_data_buf = NULL;           // -> stack Buf in mlir_llvm_to_elf (.data blob)
static bool  g_x64_ok = true;
// --time-stages sink (mlir_llvm_to_x64_set_stage_times); NULL = no timing.
static MLIR_X64StageTimes *g_x64_times = NULL;
static int g_x64_opt_level = 0;

void mlir_llvm_to_x64_set_stage_times(MLIR_X64StageTimes *t) { g_x64_times = t; }
void mlir_llvm_to_x64_set_opt_level(int level) { g_x64_opt_level = level; }

static uint64_t x64_clock(void) { return g_x64_times ? platform_clock_ns() : 0; }

// Parallel-move call operands op[arg_base .. arg_base + n_reg - 1] into
// ARG_REGS[0 .. n_reg - 1]. Uses RAX (not R11) as cycle-break temp so indirect
// call targets held in R11 remain intact.
static void emit_parallel_arg_moves(FnCtx *F, MLIR_OpHandle op, size_t arg_base, size_t n_reg) {
    if (n_reg == 0) return;
    MLIR_ValueHandle src_v[6];
    int     src_r[6], dst_r[6], mv_w[6];
    int32_t src_s[6];
    uint8_t done[6];
    size_t  rem = 0;
    for (size_t i = 0; i < n_reg; i++) {
        src_v[i] = MLIR_GetOpOperand(F->ctx, op, arg_base + i);
        x64_val_home(F, src_v[i], true, &src_r[i], &src_s[i]);
        dst_r[i] = (int)ARG_REGS[i];
        mv_w[i]  = val_w(F->ctx, src_v[i]);
        if (src_r[i] == dst_r[i]) {
            done[i] = 1;
        } else {
            done[i] = 0;
            rem++;
        }
    }
    while (rem > 0) {
        bool progress = false;
        for (size_t i = 0; i < n_reg; i++) {
            if (done[i]) continue;
            bool blocked = false;
            for (size_t j = 0; j < n_reg; j++) {
                if (!done[j] && j != i && src_r[j] == dst_r[i]) { blocked = true; break; }
            }
            if (blocked) continue;
            if (src_r[i] >= 16)     movq_gpr_xmm(F, (uint8_t)dst_r[i], (uint8_t)(src_r[i] - 16));
            else if (src_r[i] >= 0) mov_rr(F, (uint8_t)dst_r[i], (uint8_t)src_r[i], mv_w[i]);
            else                    load_val_w(F, src_v[i], (uint8_t)dst_r[i], mv_w[i]);
            done[i] = 1;
            rem--;
            progress = true;
        }
        if (!progress) {
            size_t k = 0;
            while (k < n_reg && done[k]) k++;
            int saved = dst_r[k];
            mov_rr(F, RAX, (uint8_t)saved, 1);
            for (size_t j = 0; j < n_reg; j++) {
                if (!done[j] && src_r[j] == saved) src_r[j] = RAX;
            }
        }
    }
}

// SysV AMD64 argument marshaling for llvm.call. `arg_base` skips the fn-pointer
// operand on indirect calls. Returns stack-arg count and alignment pad for cleanup.
static void emit_sysv_call_args(FnCtx *F, MLIR_OpHandle op, size_t arg_base,
                                size_t *nstack_out, size_t *pad_out) {
    size_t no = MLIR_GetOpNumOperands(F->ctx, op) - arg_base;
    size_t reg_cutoff = 6;
    bool is_vararg_call = false;
    MLIR_AttributeHandle vct = MLIR_GetOpAttr(F->ctx, op, MLIR_KEY_VAR_CALLEE_TYPE);
    if (vct != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle ft = MLIR_GetAttributeTypeValue(F->ctx, vct);
        if (ft != MLIR_INVALID_HANDLE && MLIR_GetTypeFunctionIsVarArg(F->ctx, ft)) {
            size_t nf = MLIR_GetTypeFunctionNumInputs(F->ctx, ft);
            reg_cutoff = nf < 6 ? nf : 6;
            is_vararg_call = true;
        }
    }
    if (reg_cutoff > no) reg_cutoff = no;
    size_t nstack = no - reg_cutoff;
    size_t pad = nstack & 1;
    if (pad) x64_emit1(F, OP_TYPE_X64_SUB_RSP, 8);
    for (size_t i = no; i > reg_cutoff; i--) {
        load_val(F, MLIR_GetOpOperand(F->ctx, op, arg_base + i - 1), RAX);
        x64_emit1(F, OP_TYPE_X64_PUSH_R, RAX);
    }
    emit_parallel_arg_moves(F, op, arg_base, reg_cutoff);
    if (is_vararg_call) alu_rr(F, 0x31, RAX, RAX, 0);     // xor eax, eax (AL = # xmm regs)
    *nstack_out = nstack;
    *pad_out = pad;
}

// emit `call rel32` to a named symbol (direct-call fast path).
static void emit_direct_call(FnCtx *F, string callee, SymSet *refs) {
    x64_emit1s(F, OP_TYPE_X64_CALL_SYM, callee);
    if (refs) sym_add(refs, callee);
}

// ---------------------------------------------------------------------------
// Per-op selection (non-terminators).
// ---------------------------------------------------------------------------
static void lower_op(FnCtx *F, MLIR_OpHandle op, SymSet *refs) {
    MLIR_Context *ctx = F->ctx;
    // The op type is derived from the op name by MLIR_CreateOp, so one switch
    // replaces a string comparison per candidate op. Every arm returns; the
    // floating-point ops are selected by lower_fp_op below it.
    MLIR_OpType ty = MLIR_GetOpType(ctx, op);
    switch (ty) {
    // value-less / rematerialized-at-use ops
    case OP_TYPE_LLVM_MLIR_CONSTANT: case OP_TYPE_LLVM_MLIR_ADDRESSOF:
    case OP_TYPE_LLVM_MLIR_UNDEF: case OP_TYPE_LLVM_MLIR_ZERO:
    case OP_TYPE_LLVM_MLIR_NULL:
        return;

    // integer binops: dst = op0 OP op1
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB: case OP_TYPE_LLVM_AND:
    case OP_TYPE_LLVM_OR: case OP_TYPE_LLVM_XOR: {
        uint8_t opc = ty == OP_TYPE_LLVM_ADD ? 0x01 : ty == OP_TYPE_LLVM_SUB ? 0x29 :
                      ty == OP_TYPE_LLVM_AND ? 0x21 : ty == OP_TYPE_LLVM_OR ? 0x09 : 0x31;
        uint8_t ext = ty == OP_TYPE_LLVM_ADD ? 0 : ty == OP_TYPE_LLVM_SUB ? 5 :
                      ty == OP_TYPE_LLVM_AND ? 4 : ty == OP_TYPE_LLVM_OR ? 1 : 6;
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, op, 1);
        int w = val_w(ctx, r);
        uint8_t rD;
        bool has_rD = val_in_reg(F, r, &rD);
        int64_t imm;
        if (is_const(ctx, MLIR_GetValueDefiningOp(ctx, a1), &imm) &&
            imm >= INT32_MIN && imm <= INT32_MAX) {
            uint8_t dst = has_rD ? rD : RAX;
            if (ty == OP_TYPE_LLVM_AND && imm == 0) {
                mov_ri(F, dst, 0, w);
            } else {
                load_val_w(F, a0, dst, w);
                if (imm != 0 || ty == OP_TYPE_LLVM_AND)
                    alu_ri(F, ext, dst, (int32_t)imm, w);
            }
            if (!has_rD) store_val_w(F, r, dst, w);
            return;
        }
        uint8_t r1 = val_reg_or_load(F, a1, RCX, w);
        if (has_rD && rD == r1 && ty != OP_TYPE_LLVM_SUB) {
            uint8_t r0 = val_reg_or_load(F, a0, RAX, w);
            alu_rr(F, opc, rD, r0, w);
            return;
        }
        uint8_t dst = (has_rD && rD != r1) ? rD : RAX;
        load_val_w(F, a0, dst, w);
        alu_rr(F, opc, dst, r1, w);
        if (!has_rD || dst != rD) store_val_w(F, r, dst, w);
        return;
    }
    case OP_TYPE_LLVM_MUL: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, op, 1);
        int w = val_w(ctx, r);
        uint8_t rD;
        bool has_rD = val_in_reg(F, r, &rD);
        int64_t imm;
        if (is_const(ctx, MLIR_GetValueDefiningOp(ctx, a1), &imm) &&
            imm >= INT32_MIN && imm <= INT32_MAX) {
            uint8_t dst = has_rD ? rD : RAX;
            if (imm == 0) {
                mov_ri(F, dst, 0, w);
            } else if (imm == 1) {
                load_val_w(F, a0, dst, w);
            } else if (imm > 1 && ((uint64_t)imm & ((uint64_t)imm - 1)) == 0) {
                int sh = 0;
                uint64_t u = (uint64_t)imm;
                while ((u >>= 1) != 0) sh++;
                load_val_w(F, a0, dst, w);
                x64_emit1(F, OP_TYPE_X64_SHIFT_IMM,
                          (int64_t)dst | (4ll << 4) | ((int64_t)w << 8) | ((int64_t)sh << 12));
            } else {
                uint8_t s0 = val_reg_or_load(F, a0, RAX, w);
                imul_ri(F, dst, s0, (int32_t)imm, w);
            }
            if (!has_rD) store_val_w(F, r, dst, w);
            return;
        }
        uint8_t r1 = val_reg_or_load(F, a1, RCX, w);
        if (has_rD && rD == r1) {
            uint8_t r0 = val_reg_or_load(F, a0, RAX, w);
            imul_rr(F, rD, r0, w);
            return;
        }
        uint8_t dst = (has_rD && rD != r1) ? rD : RAX;
        load_val_w(F, a0, dst, w);
        imul_rr(F, dst, r1, w);
        if (!has_rD || dst != rD) store_val_w(F, r, dst, w);
        return;
    }
    // division / remainder
    case OP_TYPE_LLVM_SDIV: case OP_TYPE_LLVM_UDIV:
    case OP_TYPE_LLVM_SREM: case OP_TYPE_LLVM_UREM: {
        bool sdiv = ty == OP_TYPE_LLVM_SDIV, udiv = ty == OP_TYPE_LLVM_UDIV;
        bool srem = ty == OP_TYPE_LLVM_SREM;
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, op, 1);
        int w = val_w(ctx, r);
        int64_t imm = 0;
        if (is_const(ctx, MLIR_GetValueDefiningOp(ctx, a1), &imm) && imm > 0 &&
            ((uint64_t)imm & ((uint64_t)imm - 1)) == 0) {
            int sh = 0;
            uint64_t u = (uint64_t)imm;
            while ((u >>= 1) != 0) sh++;
            uint8_t rD;
            bool has_rD = val_in_reg(F, r, &rD);
            uint8_t dst = has_rD ? rD : RAX;
            if (udiv) {
                load_val_w(F, a0, dst, w);
                if (sh > 0)
                    x64_emit1(F, OP_TYPE_X64_SHIFT_IMM,
                              (int64_t)dst | (5ll << 4) | ((int64_t)w << 8) | ((int64_t)sh << 12));
                if (!has_rD) store_val_w(F, r, dst, w);
                return;
            }
            if (!sdiv && !srem && imm <= (int64_t)INT32_MAX + 1) { // urem by power of 2
                load_val_w(F, a0, dst, w);
                alu_ri(F, 4 /*and*/, dst, (int32_t)(imm - 1), w);
                if (!has_rD) store_val_w(F, r, dst, w);
                return;
            }
            if (sdiv && imm == 2) {
                load_val_w(F, a0, RAX, w);
                mov_rr(F, RCX, RAX, w);
                x64_emit1(F, OP_TYPE_X64_SHIFT_IMM,
                          (int64_t)RCX | (5ll << 4) | ((int64_t)w << 8) | ((int64_t)(w ? 63 : 31) << 12));
                alu_rr(F, 0x01, RAX, RCX, w);
                x64_emit1(F, OP_TYPE_X64_SHIFT_IMM,
                          (int64_t)RAX | (7ll << 4) | ((int64_t)w << 8) | (1ll << 12));
                store_val_w(F, r, RAX, w);
                return;
            }
        }
        load_val(F, a0, RAX);
        load_val(F, a1, RCX);
        if (sdiv || srem) {
            x64_emit1(F, OP_TYPE_X64_IDIV, (int64_t)RCX | ((int64_t)w << 8));
        } else {
            x64_emit1(F, OP_TYPE_X64_DIV, (int64_t)RCX | ((int64_t)w << 8));
        }
        store_val(F, r, (sdiv || udiv) ? RAX : RDX);
        return;
    }
    // shifts: dst = op0 SH op1(cl / imm)
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_LSHR: case OP_TYPE_LLVM_ASHR: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, op, 1);
        int w = val_w(ctx, r);
        int ext = ty == OP_TYPE_LLVM_SHL ? 4 : (ty == OP_TYPE_LLVM_LSHR ? 5 : 7);
        int64_t imm;
        if (is_const(ctx, MLIR_GetValueDefiningOp(ctx, a1), &imm) && imm >= 0 && imm < 64) {
            uint8_t rD;
            bool has_rD = val_in_reg(F, r, &rD);
            uint8_t dst = has_rD ? rD : RAX;
            load_val_w(F, a0, dst, w);
            if (imm != 0)
                x64_emit1(F, OP_TYPE_X64_SHIFT_IMM,
                          (int64_t)dst | ((int64_t)ext << 4) | ((int64_t)w << 8) | ((int64_t)imm << 12));
            if (!has_rD) store_val_w(F, r, dst, w);
            return;
        }
        load_val(F, a0, RAX);
        load_val(F, a1, RCX);
        x64_emit1(F, OP_TYPE_X64_SHIFT_CL, (int64_t)RAX | ((int64_t)ext << 4) | ((int64_t)w << 8));
        store_val(F, r, RAX);
        return;
    }
    // icmp -> 0/1
    case OP_TYPE_LLVM_ICMP: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, op, 1);
        int w = val_w(ctx, a0);
        int64_t pred = 0;
        MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_PREDICATE);
        if (pa) pred = MLIR_GetAttributeInteger(ctx, pa);
        uint8_t r0 = val_reg_or_load(F, a0, RAX, w);
        int64_t imm;
        if (is_const(ctx, MLIR_GetValueDefiningOp(ctx, a1), &imm) &&
            imm >= INT32_MIN && imm <= INT32_MAX) {
            if (imm == 0) alu_rr(F, 0x85, r0, r0, w);
            else          alu_ri(F, 7, r0, (int32_t)imm, w);
        } else {
            uint8_t r1 = val_reg_or_load(F, a1, RCX, w);
            alu_rr(F, 0x39, r0, r1, w);
        }
        setcc_al(F, icmp_cc(pred));
        store_val_w(F, r, RAX, 0);
        return;
    }
    // select cond, t, f
    case OP_TYPE_LLVM_SELECT: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        int w = val_w(ctx, r);
        load_val(F, MLIR_GetOpOperand(ctx, op, 2), RAX);   // default = false value
        load_val(F, MLIR_GetOpOperand(ctx, op, 1), RCX);   // true value
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RDX);   // cond
        alu_rr(F, 0x85, RDX, RDX, 0);                 // test edx, edx
        cmovnz(F, RAX, RCX, w);                        // cond!=0 -> true value
        store_val(F, r, RAX);
        return;
    }
    // zext / trunc / sext / pointer casts
    case OP_TYPE_LLVM_ZEXT: {
        int64_t cv;
        if (!F->fast_ra && is_const(ctx, op, &cv)) return;
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);   // 32-bit load zero-extends
        store_val(F, r, RAX);
        return;
    }
    case OP_TYPE_LLVM_TRUNC: {
        int64_t cv;
        if (!F->fast_ra && is_const(ctx, op, &cv)) return;
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        int rb = int_bits(ctx, r);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);
        if (rb && rb < 32) {                           // mask low rb bits
            uint32_t mask = (rb >= 32) ? 0xffffffffu : ((1u << rb) - 1u);
            x64_emit1(F, OP_TYPE_X64_AND_EAX_IMM, (int64_t)mask);
        }
        store_val(F, r, RAX);
        return;
    }
    case OP_TYPE_LLVM_SEXT: {
        int64_t cv;
        if (!F->fast_ra && is_const(ctx, op, &cv)) return;
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(ctx, op, 0);
        int sb = int_bits(ctx, s);
        load_val(F, s, RAX);
        int bits = (sb == 32) ? 32 : (sb == 16) ? 16 : 8;
        x64_emit1(F, OP_TYPE_X64_MOVSX, (int64_t)RAX | ((int64_t)RAX << 4) | ((int64_t)bits << 8));
        store_val(F, r, RAX);
        return;
    }
    case OP_TYPE_LLVM_INTTOPTR: case OP_TYPE_LLVM_PTRTOINT: case OP_TYPE_LLVM_BITCAST: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);
        store_val(F, r, RAX);
        return;
    }
    // memory
    case OP_TYPE_LLVM_LOAD: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle addr = MLIR_GetOpOperand(ctx, op, 0);
        int nb = type_bytes(ctx, r);
        uint8_t xD;
        if (fp_bytes(ctx, r) && val_in_xmm(F, r, &xD)) {
            int is_ss = (nb == 4);
            int32_t aoff;
            if (alloca_slot(F, addr, &aoff)) {
                load_rbp_xmm(F, xD, aoff, is_ss);
            } else {
                uint8_t ra = val_reg_or_load(F, addr, RCX, 1);
                load_mem_xmm(F, xD, ra, is_ss);
            }
            return;
        }
        uint8_t rD;
        bool has_rD = val_in_reg(F, r, &rD);
        uint8_t dst = has_rD ? rD : RAX;
        int32_t aoff;
        if (alloca_slot(F, addr, &aoff)) {
            load_rbp_sized(F, dst, aoff, nb);
        } else {
            uint8_t ra = val_reg_or_load(F, addr, RCX, 1);
            load_mem(F, dst, ra, nb, false);
        }
        if (!has_rD) store_val_w(F, r, dst, nb > 4 ? 1 : 0);
        return;
    }
    case OP_TYPE_LLVM_STORE: {
        MLIR_ValueHandle val = MLIR_GetOpOperand(ctx, op, 0);
        MLIR_ValueHandle addr = MLIR_GetOpOperand(ctx, op, 1);
        int nb = type_bytes(ctx, val);
        uint8_t xS;
        if (fp_bytes(ctx, val) && val_in_xmm(F, val, &xS)) {
            int is_ss = (nb == 4);
            int32_t aoff;
            if (alloca_slot(F, addr, &aoff)) {
                store_rbp_xmm(F, aoff, xS, is_ss);
            } else {
                uint8_t ra = val_reg_or_load(F, addr, RCX, 1);
                store_mem_xmm(F, ra, xS, is_ss);
            }
            return;
        }
        uint8_t rv = val_reg_or_load(F, val, RDX, nb > 4 ? 1 : 0);
        int32_t aoff;
        if (alloca_slot(F, addr, &aoff)) {
            store_rbp_sized(F, aoff, rv, nb);
        } else {
            uint8_t ra = val_reg_or_load(F, addr, RCX, 1);
            store_mem(F, ra, rv, nb);
        }
        return;
    }
    case OP_TYPE_LLVM_GEP: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_AttributeHandle eta = MLIR_GetOpAttr(ctx, op, MLIR_KEY_ELEM_TYPE);
        MLIR_AttributeHandle ria = MLIR_GetOpAttr(ctx, op, MLIR_KEY_RAW_CONSTANT_INDICES);
        if (!eta || !ria) { fail(F, "getelementptr missing elem_type/indices", (string){0}); return; }
        MLIR_TypeHandle elem_ty = MLIR_GetAttributeTypeValue(ctx, eta);
        int32_t cidx[64];
        size_t n_idx = MLIR_GetAttributeDenseI32(ctx, ria, cidx, 64);
        if (n_idx > 64) n_idx = 64;
        if (n_idx == 0) { fail(F, "getelementptr with no indices", (string){0}); return; }
        MLIR_ValueHandle base = MLIR_GetOpOperand(ctx, op, 0);
        int32_t aoff;
        if (alloca_slot(F, base, &aoff)) lea_rbp(F, RAX, aoff);
        else                             load_val_w(F, base, RAX, 1);  // running address (ptr = 64-bit)
        size_t op_idx = 1;                                 // next dynamic operand
        MLIR_TypeHandle cur_ty = elem_ty;
        for (size_t i = 0; i < n_idx; i++) {
            int32_t ci = cidx[i];
            bool is_dyn = (ci == (int32_t)0x80000000);
            unsigned stride;
            if (i == 0) {                                  // index over `elem_type`
                if (ci == 0) continue;                     // 0 * stride == 0 (skip x64_type_size!)
                stride = x64_type_size(ctx, elem_ty);
            } else if (MLIR_GetTypeKind(ctx, cur_ty) == MLIR_TYPE_KIND_LLVM_STRUCT) {    // struct field: constant offset
                if (is_dyn) { fail(F, "dynamic struct index in gep", (string){0}); return; }
                MLIR_TypeHandle next_ty = MLIR_INVALID_HANDLE;
                unsigned foff = x64_struct_field_info(ctx, cur_ty, (size_t)ci, &next_ty);
                cur_ty = next_ty;
                add_const_off(F, RAX, (int64_t)foff);
                continue;
            } else if (MLIR_GetTypeKind(ctx, cur_ty) == MLIR_TYPE_KIND_LLVM_ARRAY) {     // array element stride
                MLIR_TypeHandle et = MLIR_GetTypeElement(ctx, cur_ty);
                cur_ty = et;
                if (ci == 0) continue;
                stride = x64_type_size(ctx, et);
            } else {
                fail(F, "gep into non-aggregate type", (string){0}); return;
            }
            if (stride == 0) continue;
            if (is_dyn) {
                if (op_idx >= MLIR_GetOpNumOperands(ctx, op)) { fail(F, "gep dynamic index operand missing", (string){0}); return; }
                MLIR_ValueHandle idx_v = MLIR_GetOpOperand(ctx, op, op_idx++);
                int ib = int_bits(ctx, idx_v);
                load_val_w(F, idx_v, RCX, (ib == 0 || ib > 32) ? 1 : 0);
                // GEP indices are signed and sign-extend to the pointer width.
                // load_val zero-extends a 32-bit index, so a negative index
                // (e.g. `p - 1`) would scale as 0xFFFFFFFF*stride; sign-extend
                // narrow indices first.
                if (ib == 32 || ib == 16 || (ib && ib < 16)) {
                    int bits = (ib == 32) ? 32 : (ib == 16) ? 16 : 8;
                    x64_emit1(F, OP_TYPE_X64_MOVSX, (int64_t)RCX | ((int64_t)RCX << 4) | ((int64_t)bits << 8));
                }
                if (stride > 1 && (stride & (stride - 1)) == 0) {
                    int sh = 0;
                    unsigned u = stride;
                    while ((u >>= 1) != 0) sh++;
                    x64_emit1(F, OP_TYPE_X64_SHIFT_IMM,
                              (int64_t)RCX | (4ll << 4) | (1ll << 8) | ((int64_t)sh << 12));
                } else if (stride != 1) {
                    imul_ri(F, RCX, RCX, (int32_t)stride, 1);  // rcx *= stride
                }
                alu_rr(F, 0x01, RAX, RCX, 1);              // add rax, rcx
            } else if (ci != 0) {
                add_const_off(F, RAX, (int64_t)ci * (int64_t)stride);
            }
        }
        store_val_w(F, r, RAX, 1);
        return;
    }
    case OP_TYPE_LLVM_ALLOCA: {
        // region already reserved in the pre-pass; recover its rbp offset from
        // the slot map (negative sentinel keyed by the op result is the region).
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        int32_t region;
        if (!vmap_get(&F->slots, x64_region_key(op), &region)) {
            fail(F, "alloca region missing", (string){0}); return;
        }
        lea_rbp(F, RAX, region);
        store_val_w(F, r, RAX, 1);
        return;
    }
    // calls
    case OP_TYPE_LLVM_CALL: case OP_TYPE_FUNC_CALL: {
        string callee = strip_at(attr_str(ctx, op, MLIR_KEY_CALLEE));
        if (!callee.size) callee = strip_at(attr_str(ctx, op, MLIR_KEY_CALLEE_NAME));
        bool is_indirect = (callee.size == 0);
        if (!is_indirect) {
            size_t no = MLIR_GetOpNumOperands(ctx, op);
            // __builtin_syscall6(num, a1..a6) -> @__tinyc_syscall6 (7 args). The
            // SysV C ABI passes 6 args in registers and the 7th (a6) on the stack;
            // the synthesised thunk reads it from [rsp+8] and traps via `syscall`.
            bool is_sc6 = str_eqz(callee, "__tinyc_syscall6");
            if (is_sc6) {
                if (no != 7) { fail(F, "__tinyc_syscall6 expects 7 args", callee); return; }
                load_val(F, MLIR_GetOpOperand(ctx, op, 6), RAX);
                x64_emit1(F, OP_TYPE_X64_PUSH_R, RAX);       // push rax (a6)
                emit_parallel_arg_moves(F, op, 0, 6);
                emit_direct_call(F, callee, refs);
                x64_emit1(F, OP_TYPE_X64_ADD_RSP, 8);        // add rsp, 8
                if (MLIR_GetOpNumResults(ctx, op) == 1)
                    store_val(F, MLIR_GetOpResult(ctx, op, 0), RAX);
                return;
            }
        }
        size_t arg_base = is_indirect ? 1 : 0;
        if (is_indirect)
            load_val(F, MLIR_GetOpOperand(ctx, op, 0), R11);
        size_t nstack = 0, pad = 0;
        emit_sysv_call_args(F, op, arg_base, &nstack, &pad);
        if (is_indirect) {
            x64_emit1(F, OP_TYPE_X64_CALL_REG, R11);         // call *r11
        } else {
            emit_direct_call(F, callee, refs);
        }
        if (nstack) {
            uint32_t cleanup = (uint32_t)((nstack + pad) * 8);   // pop stack args + pad
            x64_emit1(F, OP_TYPE_X64_ADD_RSP, (int64_t)cleanup);
        }
        if (MLIR_GetOpNumResults(ctx, op) == 1)
            store_val(F, MLIR_GetOpResult(ctx, op, 0), RAX);
        return;
    }
    // variadics: va_list is a single `char*` cursor over the on-stack variadic
    // args (matching the tinyc_va_arg_* runtime helpers). va_start writes the
    // address of the first variadic arg; va_arg is lowered to helper calls by
    // the front end, so nothing is needed here for it.
    case OP_TYPE_LLVM_INTR_VASTART: {
        // First variadic arg sits just past the saved rbp + return address and
        // any fixed params that spilled to the stack: [rbp + 16 + max(0,
        // n_fixed-6)*8]. Store that address into the va_list slot (operand 0).
        int32_t disp = 16 + (F->n_fixed > 6 ? (int32_t)(F->n_fixed - 6) * 8 : 0);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RCX);              // va_list address
        lea_rbp(F, RAX, -disp);                                       // lea rax, [rbp + disp]
        store_mem(F, RCX, RAX, 8);                                    // *va_list = cursor
        return;
    }
    case OP_TYPE_LLVM_INTR_VAEND: return;                  // stack cursor: no teardown
    case OP_TYPE_LLVM_INTR_VACOPY: {
        load_val(F, MLIR_GetOpOperand(ctx, op, 1), RCX);              // src va_list address
        load_mem(F, RAX, RCX, 8, false);                        // rax = *src
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RCX);              // dst va_list address
        store_mem(F, RCX, RAX, 8);                               // *dst = *src
        return;
    }

    case OP_TYPE_LLVM_INTR_WASM_MEMORY_SIZE: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        uint32_t off;
        if (!gmap_get(F->gm, str_lit("__wasm_mem_pages"), &off)) {
            mov_ri(F, RAX, 0, 0);
            store_val(F, r, RAX);
            return;
        }
        lea_rip_data(F, RAX, off);
        load_mem(F, RAX, RAX, 4, false);
        store_val(F, r, RAX);
        return;
    }
    case OP_TYPE_LLVM_INTR_WASM_MEMORY_GROW: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        uint32_t off;
        if (!gmap_get(F->gm, str_lit("__wasm_mem_pages"), &off)) {
            mov_ri(F, RAX, 0, 0);
            store_val(F, r, RAX);
            return;
        }
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RCX);               // delta
        lea_rip_data(F, R11, off);
        load_mem(F, R11, R11, 4, false);                        // r11 = old
        mov_rr(F, RDX, R11, 0);                                  // rdx = old
        alu_rr(F, 0x01, RDX, RCX, 0);                            // rdx = old+delta
        mov_ri(F, RCX, 65536, 0);
        alu_rr(F, 0x39, RDX, RCX, 0);                            // cmp new, cap
        cmovcc(F, RDX, R11, CC_A, 0);                              // rdx = ok? new : old
        lea_rip_data(F, RAX, off);
        store_mem(F, RAX, RDX, 4);
        mov_ri(F, RAX, -1, 0);
        cmovcc(F, RAX, R11, CC_BE, 0);                           // rax = ok? old : -1
        store_val(F, r, RAX);
        return;
    }
    default: break;
    }

    // floating-point arithmetic / conversion / comparison (issue #212 Option A)
    if (lower_fp_op(F, op, ty)) return;

    fail(F, "unsupported op", MLIR_GetOpTypeName(F->ctx, ty));
}

// emit a terminator (also collects callee refs via lower_op for calls? no -
// terminators here are br/cond_br/return/unreachable only)
static void lower_term(FnCtx *F, MLIR_RegionHandle reg, MLIR_OpHandle op,
                       MLIR_BlockHandle next_blk) {
    switch (MLIR_GetOpType(F->ctx, op)) {
    case OP_TYPE_LLVM_RETURN: case OP_TYPE_FUNC_RETURN: {
        if (MLIR_GetOpNumOperands(F->ctx, op) >= 1)
            load_val(F, MLIR_GetOpOperand(F->ctx, op, 0), RAX);
        emit_epilogue(F);
        return;
    }
    case OP_TYPE_LLVM_BR: {
        MLIR_BlockHandle t = MLIR_GetOpSuccessor(F->ctx, op, 0);
        emit_edge(F, op, 0, t);
        if (t != next_blk) {
            x64_emit1(F, OP_TYPE_X64_JMP, block_index(F->ctx, reg, t));
        }
        return;
    }
    case OP_TYPE_LLVM_COND_BR: {
        // Short-circuit edge copies for operand-carrying successors.
        // MLIR cond_br can pass values into a successor block as block arguments
        // (successor operands). emit_edge() performs those copies. They must run
        // only on the branch actually taken — otherwise && / || break.
        MLIR_BlockHandle tt = MLIR_GetOpSuccessor(F->ctx, op, 0);
        MLIR_BlockHandle ft = MLIR_GetOpSuccessor(F->ctx, op, 1);
        size_t nso_t = MLIR_GetOpNumSuccessorOperands(F->ctx, op, 0);
        size_t nso_f = MLIR_GetOpNumSuccessorOperands(F->ctx, op, 1);
        bool need_t = nso_t > 0;
        bool need_f = nso_f > 0;
        load_val(F, MLIR_GetOpOperand(F->ctx, op, 0), RAX);      // cond
        alu_rr(F, 0x85, RAX, RAX, 0);                            // test eax, eax
        int tti = block_index(F->ctx, reg, tt);
        int fti = block_index(F->ctx, reg, ft);
        if (!need_t && !need_f) {
            if (ft == next_blk) {
                x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)CC_NE | ((int64_t)(uint32_t)tti << 32));
                return;
            }
            if (tt == next_blk) {
                x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)CC_E | ((int64_t)(uint32_t)fti << 32));
                return;
            }
        }
        int pad_t = need_t ? F->x64_next_label++ : tti;
        int pad_f = need_f ? F->x64_next_label++ : fti;
        x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)CC_NE | ((int64_t)(uint32_t)pad_t << 32));
        x64_emit1(F, OP_TYPE_X64_JMP, pad_f);
        if (need_t) {
            x64_emit1(F, OP_TYPE_X64_LABEL, pad_t);
            emit_edge(F, op, 0, tt);
            if (need_f || tt != next_blk) x64_emit1(F, OP_TYPE_X64_JMP, tti);
        }
        if (need_f) {
            x64_emit1(F, OP_TYPE_X64_LABEL, pad_f);
            emit_edge(F, op, 1, ft);
            if (ft != next_blk) x64_emit1(F, OP_TYPE_X64_JMP, fti);
        }
        return;
    }
    case OP_TYPE_LLVM_UNREACHABLE: x64_emit0(F, OP_TYPE_X64_UD2); return;  // ud2
    default: break;
    }
    fail(F, "unsupported terminator", MLIR_GetOpTypeName(F->ctx, MLIR_GetOpType(F->ctx, op)));
}

// ---------------------------------------------------------------------------
// Function emission: frame pre-pass, prologue, blocks, branch patch.
// ---------------------------------------------------------------------------
// Ops whose result load_val() rematerializes at every use (lower_op emits no
// code for them), so the result is never stored and needs no frame slot.
static bool x64_op_rematerialized(MLIR_Context *ctx, MLIR_OpHandle op) {
    switch (MLIR_GetOpType(ctx, op)) {
    case OP_TYPE_LLVM_MLIR_CONSTANT: case OP_TYPE_LLVM_MLIR_ADDRESSOF:
    case OP_TYPE_LLVM_MLIR_UNDEF: case OP_TYPE_LLVM_MLIR_ZERO:
    case OP_TYPE_LLVM_MLIR_NULL:
        return true;
    default:
        return false;
    }
}

// Frame slot of `v` for the pre-pass below: none when it lives in a register,
// its shared spill slot when the allocator spilled it, else a fresh slot.
static void x64_assign_slot(FnCtx *F, SlotMap *spill, MLIR_ValueHandle v, int32_t *cur) {
    int32_t idx;
    if (sm_get(&F->rm, v, &idx)) return;
    if (sm_get(spill, v, &idx)) { vmap_put(&F->slots, v, (idx + 1) * 8); return; }
    *cur += 8;
    vmap_put(&F->slots, v, *cur);
}

static unsigned alloca_bytes(MLIR_Context *ctx, MLIR_OpHandle op) {
    // element type size * count; default 8. Uses the shared LP64 layout
    // (x64_type_size), so nested arrays / structs / unions are sized correctly.
    MLIR_AttributeHandle et = MLIR_GetOpAttr(ctx, op, MLIR_KEY_ELEM_TYPE);
    unsigned esz = 8;
    if (et) {
        MLIR_TypeHandle t = MLIR_GetAttributeTypeValue(ctx, et);
        if (t != MLIR_INVALID_HANDLE) {
            unsigned s = x64_type_size(ctx, t);
            if (s) esz = s;
        }
    }
    if (esz == 0) esz = 8;
    return (esz + 7) & ~7u;
}

// Overwrite (or insert) a physical-register assignment in F->rm (sm_put is insert-only).
static void x64_rm_set(SlotMap *m, MLIR_ValueHandle k, int32_t reg) {
    if (m->dense) {
        uintptr_t idx = (uintptr_t)k - m->dense_base;
        if (idx < m->dense_len) { m->dense[idx] = reg; return; }
    }
    if (m->cap > 0) {
        size_t mask = m->cap - 1;
        size_t i = (((uintptr_t)k) * 11400714819323198485ull >> 32) & mask;
        // Linear scan fallback if hash differs: check direct hash or scan.
        // Wait: match mlir_regalloc.c's exact sm_hash or update in-place via sm_get?
        // Actually sm_hash is (k * 0x9e3779b97f4a7c15ULL) >> 32; check all slots if needed.
        for (size_t probe = 0; probe < m->cap; probe++) {
            size_t j = (i + probe) & mask;
            if (m->t[j].key == (uintptr_t)k) { m->t[j].slot = reg; return; }
            if (m->t[j].key == 0) break;
        }
        // Fallback linear scan in case sm_hash differs
        for (size_t j = 0; j < m->cap; j++) {
            if (m->t[j].key == (uintptr_t)k) { m->t[j].slot = reg; return; }
        }
    }
    sm_put(m, k, reg);
}

static bool x64_val_dead_after_term(FnCtx *F, MLIR_ValueHandle v) {
    if (F->fast_ra) {
        uint8_t dummy;
        return val_in_reg(F, v, &dummy);
    }
    if (F->use_cnt) {
        uintptr_t idx = (uintptr_t)v - F->use_base;
        if (idx < F->use_len) return F->use_cnt[idx] <= 1;
        return false;
    }
    if (MLIR_HasContextDefUseTracking(F->ctx))
        return MLIR_ValueHasOneUse(F->ctx, v);
    return false;
}

static void emit_fused_cond_br(FnCtx *F, MLIR_RegionHandle reg, MLIR_OpHandle br_op,
                               MLIR_BlockHandle next_blk, MLIR_OpHandle icmp_op, bool invert) {
    MLIR_Context *ctx = F->ctx;
    MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, icmp_op, 0);
    MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, icmp_op, 1);
    int w = val_w(ctx, a0);
    int64_t pred = 0;
    MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, icmp_op, MLIR_KEY_PREDICATE);
    if (pa) pred = MLIR_GetAttributeInteger(ctx, pa);
    int cc = icmp_cc(pred);
    if (invert) cc ^= 1;

    uint8_t r0 = val_reg_or_load(F, a0, RAX, w);
    int64_t imm;
    if (is_const(ctx, MLIR_GetValueDefiningOp(ctx, a1), &imm) &&
        imm >= INT32_MIN && imm <= INT32_MAX) {
        if (imm == 0) alu_rr(F, 0x85, r0, r0, w);
        else          alu_ri(F, 7, r0, (int32_t)imm, w);
    } else {
        uint8_t r1 = val_reg_or_load(F, a1, RCX, w);
        alu_rr(F, 0x39, r0, r1, w);
    }

    MLIR_BlockHandle tt = MLIR_GetOpSuccessor(ctx, br_op, 0);
    MLIR_BlockHandle ft = MLIR_GetOpSuccessor(ctx, br_op, 1);
    size_t nso_t = MLIR_GetOpNumSuccessorOperands(ctx, br_op, 0);
    size_t nso_f = MLIR_GetOpNumSuccessorOperands(ctx, br_op, 1);
    bool need_t = nso_t > 0;
    bool need_f = nso_f > 0;
    int tti = block_index(ctx, reg, tt);
    int fti = block_index(ctx, reg, ft);
    if (!need_t && !need_f) {
        if (ft == next_blk) {
            x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)cc | ((int64_t)(uint32_t)tti << 32));
            return;
        }
        if (tt == next_blk) {
            x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)(cc ^ 1) | ((int64_t)(uint32_t)fti << 32));
            return;
        }
        x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)cc | ((int64_t)(uint32_t)tti << 32));
        x64_emit1(F, OP_TYPE_X64_JMP, fti);
        return;
    }
    if (need_t && !need_f) {
        x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)(cc ^ 1) | ((int64_t)(uint32_t)fti << 32));
        emit_edge(F, br_op, 0, tt);
        if (tt != next_blk) x64_emit1(F, OP_TYPE_X64_JMP, tti);
        return;
    }
    if (!need_t && need_f) {
        x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)cc | ((int64_t)(uint32_t)tti << 32));
        emit_edge(F, br_op, 1, ft);
        if (ft != next_blk) x64_emit1(F, OP_TYPE_X64_JMP, fti);
        return;
    }
    int pad_t = F->x64_next_label++;
    int pad_f = F->x64_next_label++;
    x64_emit1(F, OP_TYPE_X64_JCC, (int64_t)cc | ((int64_t)(uint32_t)pad_t << 32));
    x64_emit1(F, OP_TYPE_X64_JMP, pad_f);
    x64_emit1(F, OP_TYPE_X64_LABEL, pad_t);
    emit_edge(F, br_op, 0, tt);
    x64_emit1(F, OP_TYPE_X64_JMP, tti);
    x64_emit1(F, OP_TYPE_X64_LABEL, pad_f);
    emit_edge(F, br_op, 1, ft);
    if (ft != next_blk) x64_emit1(F, OP_TYPE_X64_JMP, fti);
}

// ---------------------------------------------------------------------------
// x64.* MLIR dialect optimization pass (runs at -O1/-O2 between isel and
// x64_encode_block, and is visible in `--emit=x64-mlir -O2`).
//
// Operates in-place on the L1-hot x64.* block via MLIR_RewriteVoidInlineI64
// and OP_TYPE_X64_NOP (zero allocations, O(N) linear scan).
// ---------------------------------------------------------------------------
static bool x64_op_regs(MLIR_Context *ctx, MLIR_OpHandle op,
                        uint16_t *reads, uint16_t *writes) {
    *reads = 0; *writes = 0;
    MLIR_OpType ty = MLIR_GetOpType(ctx, op);
    switch (ty) {
    case OP_TYPE_X64_NOP: return true;
    case OP_TYPE_X64_LABEL: case OP_TYPE_X64_JMP: case OP_TYPE_X64_JCC:
    case OP_TYPE_X64_CALL_SYM: case OP_TYPE_X64_CALL_REG:
    case OP_TYPE_X64_LEAVE: case OP_TYPE_X64_RET: case OP_TYPE_X64_UD2:
    case OP_TYPE_X64_PUSH_R: case OP_TYPE_X64_SUB_RSP: case OP_TYPE_X64_ADD_RSP:
        return false;
    case OP_TYPE_X64_MOV_RR: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << (v & 15));
        *reads  = (uint16_t)(1u << ((v >> 4) & 15));
        return true;
    }
    case OP_TYPE_X64_MOV_RI32: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << (v & 15));
        return true;
    }
    case OP_TYPE_X64_MOV_RI: {
        uint64_t v = (uint64_t)MLIR_GetAttributeInteger(ctx, MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE));
        *writes = (uint16_t)(1u << (v & 15));
        return true;
    }
    case OP_TYPE_X64_ALU_RR: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        uint8_t d = (uint8_t)(v & 15), s = (uint8_t)((v >> 4) & 15);
        uint8_t opc = (uint8_t)((v >> 12) & 0xff);
        *reads = (uint16_t)((1u << d) | (1u << s));
        if (opc != 0x39 && opc != 0x85) *writes = (uint16_t)(1u << d);
        return true;
    }
    case OP_TYPE_X64_ALU_RI: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        uint8_t d = (uint8_t)(v & 15), ext = (uint8_t)((v >> 4) & 7);
        *reads = (uint16_t)(1u << d);
        if (ext != 7) *writes = (uint16_t)(1u << d);
        return true;
    }
    case OP_TYPE_X64_ADD_RI: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        uint8_t r = (uint8_t)(v & 15);
        *reads = *writes = (uint16_t)(1u << r);
        return true;
    }
    case OP_TYPE_X64_AND_EAX_IMM:
        *reads = *writes = (uint16_t)(1u << RAX);
        return true;
    case OP_TYPE_X64_IMUL_RR: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        uint8_t d = (uint8_t)(v & 15), s = (uint8_t)((v >> 4) & 15);
        *reads = (uint16_t)((1u << d) | (1u << s));
        *writes = (uint16_t)(1u << d);
        return true;
    }
    case OP_TYPE_X64_IMUL_RI: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << (v & 15));
        *reads  = (uint16_t)(1u << ((v >> 4) & 15));
        return true;
    }
    case OP_TYPE_X64_IDIV: case OP_TYPE_X64_DIV: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads  = (uint16_t)((1u << RAX) | (1u << RDX) | (1u << (v & 15)));
        *writes = (uint16_t)((1u << RAX) | (1u << RDX));
        return true;
    }
    case OP_TYPE_X64_SHIFT_CL: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        uint8_t r = (uint8_t)(v & 15);
        *reads  = (uint16_t)((1u << r) | (1u << RCX));
        *writes = (uint16_t)(1u << r);
        return true;
    }
    case OP_TYPE_X64_SHIFT_IMM: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        uint8_t r = (uint8_t)(v & 15);
        *reads = *writes = (uint16_t)(1u << r);
        return true;
    }
    case OP_TYPE_X64_LOAD_RBP: case OP_TYPE_X64_LOAD_RBP_SIZED: case OP_TYPE_X64_LEA_RBP: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads  = (uint16_t)(1u << RBP);
        *writes = (uint16_t)(1u << (v & 15));
        return true;
    }
    case OP_TYPE_X64_STORE_RBP: case OP_TYPE_X64_STORE_RBP_SIZED: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)((1u << RBP) | (1u << (v & 15)));
        return true;
    }
    case OP_TYPE_X64_LOAD_MEM: case OP_TYPE_X64_LOAD_MEM_DISP: case OP_TYPE_X64_LEA_MEM_DISP: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << (v & 15));
        *reads  = (uint16_t)(1u << ((v >> 4) & 15));
        return true;
    }
    case OP_TYPE_X64_STORE_MEM: case OP_TYPE_X64_STORE_MEM_DISP: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)((1u << (v & 15)) | (1u << ((v >> 4) & 15)));
        return true;
    }
    case OP_TYPE_X64_LOAD_MEM_SIB: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << (v & 15));
        *reads  = (uint16_t)((1u << ((v >> 4) & 15)) | (1u << ((v >> 8) & 15)));
        return true;
    }
    case OP_TYPE_X64_STORE_MEM_SIB: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)((1u << (v & 15)) | (1u << ((v >> 4) & 15)) | (1u << ((v >> 8) & 15)));
        return true;
    }
    case OP_TYPE_X64_STORE_MEM_SIB_IMM8: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)((1u << (v & 15)) | (1u << ((v >> 4) & 15)));
        return true;
    }
    case OP_TYPE_X64_LEA_RIP_DATA: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << (v & 15));
        return true;
    }
    case OP_TYPE_X64_LEA_RIP_FUNC: {
        uint8_t r = (uint8_t)MLIR_GetAttributeInteger(ctx, MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE));
        *writes = (uint16_t)(1u << r);
        return true;
    }
    case OP_TYPE_X64_MOVSX: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << (v & 15));
        *reads  = (uint16_t)(1u << ((v >> 4) & 15));
        return true;
    }
    case OP_TYPE_X64_SETCC_AL: case OP_TYPE_X64_MOVZX_EAX_AL:
        *writes = (uint16_t)(1u << RAX);
        return true;
    case OP_TYPE_X64_SETCC8: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << ((v >> 4) & 15));
        return true;
    }
    case OP_TYPE_X64_ALU8_AL_CL:
        *reads  = (uint16_t)((1u << RAX) | (1u << RCX));
        *writes = (uint16_t)(1u << RAX);
        return true;
    case OP_TYPE_X64_CMOVCC: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads  = (uint16_t)((1u << (v & 15)) | (1u << ((v >> 4) & 15)));
        *writes = (uint16_t)(1u << (v & 15));
        return true;
    }
    case OP_TYPE_X64_MOVQ_XMM_GPR: case OP_TYPE_X64_CVTSI2F: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)(1u << ((v >> 4) & 15));
        return true;
    }
    case OP_TYPE_X64_MOVQ_GPR_XMM: case OP_TYPE_X64_CVTF2SI: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *writes = (uint16_t)(1u << (v & 15));
        return true;
    }
    case OP_TYPE_X64_SSE_OP: case OP_TYPE_X64_SSE_MOV_RR: case OP_TYPE_X64_SSE_XORPD:
    case OP_TYPE_X64_CVT_F2F: case OP_TYPE_X64_UCOMIS:
        return true;
    case OP_TYPE_X64_LOAD_RBP_XMM: case OP_TYPE_X64_STORE_RBP_XMM:
        *reads = (uint16_t)(1u << RBP);
        return true;
    case OP_TYPE_X64_LOAD_MEM_XMM: case OP_TYPE_X64_LOAD_MEM_XMM_DISP: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)(1u << ((v >> 4) & 15));
        return true;
    }
    case OP_TYPE_X64_STORE_MEM_XMM: case OP_TYPE_X64_STORE_MEM_XMM_DISP: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)(1u << (v & 15));
        return true;
    }
    case OP_TYPE_X64_LOAD_MEM_XMM_SIB: case OP_TYPE_X64_SSE_OP_MEM_SIB: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)((1u << ((v >> 4) & 15)) | (1u << ((v >> 8) & 15)));
        return true;
    }
    case OP_TYPE_X64_STORE_MEM_XMM_SIB: {
        uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
        *reads = (uint16_t)((1u << (v & 15)) | (1u << ((v >> 4) & 15)));
        return true;
    }
    default:
        return false;
    }
}

static bool x64_reg_dead_after(MLIR_Context *ctx, MLIR_BlockHandle blk,
                               size_t start, size_t n_ops, uint8_t reg) {
    uint16_t bit = (uint16_t)(1u << reg);
    bool is_scratch = (reg == RAX || reg == RCX || reg == RDX || reg == R11);
    int seen = 0;
    for (size_t k = start; k < n_ops && seen < 24; k++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blk, k);
        MLIR_OpType ty = MLIR_GetOpType(ctx, op);
        if (ty == OP_TYPE_X64_NOP) continue;
        seen++;
        // Scratch regs (RAX/RCX/RDX/R11) are never live across block boundaries
        // (JMP/JCC/LABEL), though RET reads RAX and CALL reads RCX/RDX/R11.
        if (is_scratch && (ty == OP_TYPE_X64_JMP || ty == OP_TYPE_X64_JCC || ty == OP_TYPE_X64_LABEL))
            return true;
        uint16_t r = 0, w = 0;
        if (!x64_op_regs(ctx, op, &r, &w)) return false;
        if (r & bit) return false;
        if (w & bit) return true;
    }
    return false;
}

static size_t x64_next_live(MLIR_Context *ctx, MLIR_BlockHandle blk, size_t start, size_t n_ops) {
    while (start < n_ops && MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, blk, start)) == OP_TYPE_X64_NOP)
        start++;
    return start;
}

static void x64_optimize_block(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    size_t n_ops = MLIR_GetBlockNumOps(ctx, blk);
    if (n_ops < 2) return;

    for (int pass = 0; pass < 3; pass++) {
        bool changed = false;
        for (size_t i = 0; i < n_ops; i++) {
            MLIR_OpHandle op0 = MLIR_GetBlockOp(ctx, blk, i);
            MLIR_OpType t0 = MLIR_GetOpType(ctx, op0);
            if (t0 == OP_TYPE_X64_NOP) continue;

            size_t j = x64_next_live(ctx, blk, i + 1, n_ops);
            if (j >= n_ops) break;
            MLIR_OpHandle op1 = MLIR_GetBlockOp(ctx, blk, j);
            MLIR_OpType t1 = MLIR_GetOpType(ctx, op1);

            // 0. Dead stack-slot store elimination (within same basic block):
            //    store_rbp [rbp-off], r1 ; ... (no read of [rbp-off] or barrier) ; store_rbp [rbp-off], r2
            if (t0 == OP_TYPE_X64_STORE_RBP) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                int32_t off0 = (int32_t)(uint32_t)(v0 >> 32);
                int w0 = (int)((v0 >> 8) & 1);
                bool dead_store = false;
                int seen = 0;
                for (size_t k = j; k < n_ops && seen < 16; k++) {
                    MLIR_OpHandle opk = MLIR_GetBlockOp(ctx, blk, k);
                    MLIR_OpType tk = MLIR_GetOpType(ctx, opk);
                    if (tk == OP_TYPE_X64_NOP) continue;
                    seen++;
                    if (tk == OP_TYPE_X64_LABEL || tk == OP_TYPE_X64_JMP || tk == OP_TYPE_X64_JCC ||
                        tk == OP_TYPE_X64_CALL_SYM || tk == OP_TYPE_X64_CALL_REG ||
                        tk == OP_TYPE_X64_LEAVE || tk == OP_TYPE_X64_RET ||
                        tk == OP_TYPE_X64_LEA_RBP || tk == OP_TYPE_X64_LOAD_RBP_SIZED)
                        break;
                    if (tk == OP_TYPE_X64_LOAD_RBP) {
                        int32_t offk = (int32_t)(uint32_t)((uint64_t)MLIR_GetOpInlineI64(ctx, opk) >> 32);
                        if (offk == off0) break;
                    }
                    if (tk == OP_TYPE_X64_STORE_RBP) {
                        uint64_t vk = (uint64_t)MLIR_GetOpInlineI64(ctx, opk);
                        int32_t offk = (int32_t)(uint32_t)(vk >> 32);
                        int wk = (int)((vk >> 8) & 1);
                        if (offk == off0 && wk >= w0) { dead_store = true; break; }
                    }
                }
                if (dead_store) {
                    MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                    changed = true;
                    continue;
                }
            }

            // 0b. FP GP<->XMM round-trip elimination:
            //     movq_gpr_xmm t, xmm ; movq_xmm_gpr xmm, t (t dead) -> nop, nop
            if (t0 == OP_TYPE_X64_MOVQ_GPR_XMM && t1 == OP_TYPE_X64_MOVQ_XMM_GPR) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                uint8_t gpr0 = (uint8_t)(v0 & 15), xmm0 = (uint8_t)((v0 >> 4) & 15);
                uint8_t xmm1 = (uint8_t)(v1 & 15), gpr1 = (uint8_t)((v1 >> 4) & 15);
                if (gpr0 == gpr1 && xmm0 == xmm1 && x64_reg_dead_after(ctx, blk, j + 1, n_ops, gpr0)) {
                    MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                    MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                    changed = true;
                    continue;
                }
            }

            // 0c. Narrow zero-extending load + movsx -> sign-extending load:
            //     load_mem*(dst, ..., nb=1/2, sgn=0) ; movsx(d2, dst, bits=nb*8) -> load_mem*(d2, ..., sgn=1)
            if ((t0 == OP_TYPE_X64_LOAD_MEM || t0 == OP_TYPE_X64_LOAD_MEM_DISP || t0 == OP_TYPE_X64_LOAD_MEM_SIB) &&
                t1 == OP_TYPE_X64_MOVSX) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                uint8_t d0 = (uint8_t)(v0 & 15);
                int nb  = (t0 == OP_TYPE_X64_LOAD_MEM_SIB) ? (int)((v0 >> 16) & 15) : (int)((v0 >> 8) & 15);
                int sgn = (t0 == OP_TYPE_X64_LOAD_MEM_SIB) ? (int)((v0 >> 20) & 1)  : (int)((v0 >> 12) & 1);
                uint8_t d1 = (uint8_t)(v1 & 15), s1 = (uint8_t)((v1 >> 4) & 15);
                int bits = (int)((v1 >> 8) & 0xff);
                if (!sgn && s1 == d0 && ((nb == 1 && bits == 8) || (nb == 2 && bits == 16)) &&
                    (d1 == d0 || x64_reg_dead_after(ctx, blk, j + 1, n_ops, d0))) {
                    uint64_t sgn_bit = (t0 == OP_TYPE_X64_LOAD_MEM_SIB) ? (1ull << 20) : (1ull << 12);
                    uint64_t nv0 = (v0 & ~(uint64_t)15u) | (uint64_t)d1 | sgn_bit;
                    MLIR_RewriteVoidInlineI64(ctx, op0, t0, (int64_t)nv0);
                    MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                    changed = true;
                    continue;
                }
            }

            // 1. Store-to-load forwarding on stack slots:
            //    store_rbp [rbp-off], r1 (w1) ; load_rbp r2, [rbp-off] (w2) -> mov r2, r1 (or nop)
            if (t0 == OP_TYPE_X64_STORE_RBP && t1 == OP_TYPE_X64_LOAD_RBP) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                int32_t off0 = (int32_t)(uint32_t)(v0 >> 32);
                int32_t off1 = (int32_t)(uint32_t)(v1 >> 32);
                int w0 = (int)((v0 >> 8) & 1), w1 = (int)((v1 >> 8) & 1);
                if (off0 == off1 && (w0 == w1 || (w0 == 1 && w1 == 0))) {
                    uint8_t r1 = (uint8_t)(v0 & 15), r2 = (uint8_t)(v1 & 15);
                    if (r1 == r2 && w0 == w1)
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                    else
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_MOV_RR,
                            (int64_t)r2 | ((int64_t)r1 << 4) | ((int64_t)w1 << 8));
                    changed = true;
                    continue;
                }
            }
            if (t0 == OP_TYPE_X64_STORE_RBP_SIZED && t1 == OP_TYPE_X64_LOAD_RBP_SIZED) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                int32_t off0 = (int32_t)(uint32_t)(v0 >> 32);
                int32_t off1 = (int32_t)(uint32_t)(v1 >> 32);
                int b0 = (int)((v0 >> 4) & 15), b1 = (int)((v1 >> 4) & 15);
                if (off0 == off1 && b0 == b1 && (b0 == 4 || b0 == 8)) {
                    uint8_t r1 = (uint8_t)(v0 & 15), r2 = (uint8_t)(v1 & 15);
                    int w = (b0 == 8) ? 1 : 0;
                    if (r1 == r2)
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                    else
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_MOV_RR,
                            (int64_t)r2 | ((int64_t)r1 << 4) | ((int64_t)w << 8));
                    changed = true;
                    continue;
                }
            }

            // 2. MOV_RR peepholes
            if (t0 == OP_TYPE_X64_MOV_RR) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                uint8_t d0 = (uint8_t)(v0 & 15), s0 = (uint8_t)((v0 >> 4) & 15);
                int w0 = (int)((v0 >> 8) & 1);
                if (d0 == s0 && w0 == 1) {
                    MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                    changed = true;
                    continue;
                }
                // mov a, b; mov b, a -> mov a, b
                if (t1 == OP_TYPE_X64_MOV_RR) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t d1 = (uint8_t)(v1 & 15), s1 = (uint8_t)((v1 >> 4) & 15);
                    int w1 = (int)((v1 >> 8) & 1);
                    if (d1 == s0 && s1 == d0 && w0 == w1) {
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                        changed = true;
                        continue;
                    }
                    // mov t, s; mov d, t (t dead) -> mov d, s
                    if (s1 == d0 && (w0 == w1 || (w0 == 1 && w1 == 0)) && x64_reg_dead_after(ctx, blk, j + 1, n_ops, d0)) {
                        if (d1 == s0 && w1 == 1)
                            MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        else
                            MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_MOV_RR,
                                (int64_t)d1 | ((int64_t)s0 << 4) | ((int64_t)w1 << 8));
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                        changed = true;
                        continue;
                    }
                }
                // mov d, s; movsx d, d, bits -> movsx d, s, bits
                if (t1 == OP_TYPE_X64_MOVSX) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t d1 = (uint8_t)(v1 & 15), s1 = (uint8_t)((v1 >> 4) & 15);
                    int bits = (int)((v1 >> 8) & 0xff);
                    if (d1 == d0 && s1 == d0) {
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_MOVSX,
                            (int64_t)d1 | ((int64_t)s0 << 4) | ((int64_t)bits << 8));
                        changed = true;
                        continue;
                    }
                }
                // mov t, src (64b); load_mem dst, [t] (t==dst or t dead) -> load_mem dst, [src]
                if (w0 == 1 && t1 == OP_TYPE_X64_LOAD_MEM) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t dst = (uint8_t)(v1 & 15), addr = (uint8_t)((v1 >> 4) & 15);
                    if (addr == d0 && (dst == d0 || x64_reg_dead_after(ctx, blk, j + 1, n_ops, d0))) {
                        uint64_t nv = (v1 & ~(uint64_t)(15u << 4)) | ((uint64_t)s0 << 4);
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_LOAD_MEM, (int64_t)nv);
                        changed = true;
                        continue;
                    }
                }
                // mov t, src (64b); load_mem_disp dst, [t+disp] -> load_mem_disp dst, [src+disp]
                if (w0 == 1 && t1 == OP_TYPE_X64_LOAD_MEM_DISP) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t dst = (uint8_t)(v1 & 15), base = (uint8_t)((v1 >> 4) & 15);
                    if (base == d0 && (dst == d0 || x64_reg_dead_after(ctx, blk, j + 1, n_ops, d0))) {
                        uint64_t nv = (v1 & ~(uint64_t)(15u << 4)) | ((uint64_t)s0 << 4);
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_LOAD_MEM_DISP, (int64_t)nv);
                        changed = true;
                        continue;
                    }
                }
                // mov t, src (64b); store_mem [t], val (val != t, t dead) -> store_mem [src], val
                if (w0 == 1 && t1 == OP_TYPE_X64_STORE_MEM) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t addr = (uint8_t)(v1 & 15), val = (uint8_t)((v1 >> 4) & 15);
                    if (addr == d0 && val != d0 && x64_reg_dead_after(ctx, blk, j + 1, n_ops, d0)) {
                        uint64_t nv = (v1 & ~(uint64_t)15u) | (uint64_t)s0;
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_STORE_MEM, (int64_t)nv);
                        changed = true;
                        continue;
                    }
                }
                // mov t, RAX (64b); mov_ri32 RAX, imm; store_mem [t], RAX (t dead) ->
                // mov_ri32 RDX, imm; store_mem [RAX], RDX (unlocks store_mem_sib!)
                if (w0 == 1 && s0 == RAX && d0 != RAX && d0 != RDX && t1 == OP_TYPE_X64_MOV_RI32) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t r1 = (uint8_t)(v1 & 15);
                    size_t k = x64_next_live(ctx, blk, j + 1, n_ops);
                    if (r1 == RAX && k < n_ops) {
                        MLIR_OpHandle op2 = MLIR_GetBlockOp(ctx, blk, k);
                        if (MLIR_GetOpType(ctx, op2) == OP_TYPE_X64_STORE_MEM) {
                            uint64_t v2 = (uint64_t)MLIR_GetOpInlineI64(ctx, op2);
                            uint8_t addr = (uint8_t)(v2 & 15), val = (uint8_t)((v2 >> 4) & 15);
                            if (addr == d0 && val == RAX &&
                                x64_reg_dead_after(ctx, blk, k + 1, n_ops, d0) &&
                                x64_reg_dead_after(ctx, blk, k + 1, n_ops, RAX) &&
                                x64_reg_dead_after(ctx, blk, k + 1, n_ops, RDX)) {
                                MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                                MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_MOV_RI32,
                                    (int64_t)((v1 & ~(uint64_t)15u) | (uint64_t)RDX));
                                MLIR_RewriteVoidInlineI64(ctx, op2, OP_TYPE_X64_STORE_MEM,
                                    (int64_t)((v2 & ~(uint64_t)0xffu) | (uint64_t)RAX | ((uint64_t)RDX << 4)));
                                changed = true;
                                continue;
                            }
                        }
                    }
                }
                // mov dst, base (64b); add_ri dst, disp -> lea_mem_disp dst, [base + disp]
                if (w0 == 1 && t1 == OP_TYPE_X64_ADD_RI) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t r1 = (uint8_t)(v1 & 15);
                    int32_t disp = (int32_t)(uint32_t)(v1 >> 32);
                    if (r1 == d0) {
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_LEA_MEM_DISP,
                            (int64_t)d0 | ((int64_t)s0 << 4) | ((int64_t)(uint32_t)disp << 32));
                        changed = true;
                        continue;
                    }
                }
                // mov t, a; alu_rr opc, t, b; mov dst, t -> mov dst, a; alu_rr opc, dst, b
                if (t1 == OP_TYPE_X64_ALU_RR) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t ad = (uint8_t)(v1 & 15), as = (uint8_t)((v1 >> 4) & 15);
                    int aw = (int)((v1 >> 8) & 1);
                    uint8_t opc = (uint8_t)((v1 >> 12) & 0xff);
                    size_t k = x64_next_live(ctx, blk, j + 1, n_ops);
                    if (ad == d0 && aw == w0 && opc != 0x39 && opc != 0x85 && k < n_ops) {
                        MLIR_OpHandle op2 = MLIR_GetBlockOp(ctx, blk, k);
                        if (MLIR_GetOpType(ctx, op2) == OP_TYPE_X64_MOV_RR) {
                            uint64_t v2 = (uint64_t)MLIR_GetOpInlineI64(ctx, op2);
                            uint8_t d2 = (uint8_t)(v2 & 15), s2 = (uint8_t)((v2 >> 4) & 15);
                            int w2 = (int)((v2 >> 8) & 1);
                            if (s2 == d0 && w2 == w0 && d2 != as &&
                                x64_reg_dead_after(ctx, blk, k + 1, n_ops, d0)) {
                                if (d2 == s0 && w0 == 1)
                                    MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                                else
                                    MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_MOV_RR,
                                        (int64_t)d2 | ((int64_t)s0 << 4) | ((int64_t)w0 << 8));
                                MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_ALU_RR,
                                    (int64_t)d2 | ((int64_t)as << 4) | ((int64_t)w0 << 8) | ((int64_t)opc << 12));
                                MLIR_RewriteVoidInlineI64(ctx, op2, OP_TYPE_X64_NOP, 0);
                                changed = true;
                                continue;
                            }
                        }
                    }
                }
                // mov t, a; alu_ri ext, t, imm; mov dst, t -> mov dst, a; alu_ri ext, dst, imm
                if (t1 == OP_TYPE_X64_ALU_RI) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t ad = (uint8_t)(v1 & 15), ext = (uint8_t)((v1 >> 4) & 7);
                    int aw = (int)((v1 >> 8) & 1);
                    size_t k = x64_next_live(ctx, blk, j + 1, n_ops);
                    if (ad == d0 && aw == w0 && ext != 7 && k < n_ops) {
                        MLIR_OpHandle op2 = MLIR_GetBlockOp(ctx, blk, k);
                        if (MLIR_GetOpType(ctx, op2) == OP_TYPE_X64_MOV_RR) {
                            uint64_t v2 = (uint64_t)MLIR_GetOpInlineI64(ctx, op2);
                            uint8_t d2 = (uint8_t)(v2 & 15), s2 = (uint8_t)((v2 >> 4) & 15);
                            int w2 = (int)((v2 >> 8) & 1);
                            if (s2 == d0 && w2 == w0 &&
                                x64_reg_dead_after(ctx, blk, k + 1, n_ops, d0)) {
                                if (d2 == s0 && w0 == 1)
                                    MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                                else
                                    MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_MOV_RR,
                                        (int64_t)d2 | ((int64_t)s0 << 4) | ((int64_t)w0 << 8));
                                uint64_t nv1 = (v1 & ~(uint64_t)15u) | (uint64_t)d2;
                                MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_ALU_RI, (int64_t)nv1);
                                MLIR_RewriteVoidInlineI64(ctx, op2, OP_TYPE_X64_NOP, 0);
                                changed = true;
                                continue;
                            }
                        }
                    }
                }
            }

            // 3. Displaced addressing fusion:
            //    add_ri t, disp ; load_mem dst, [t] -> load_mem_disp dst, [t + disp]
            //    lea_mem_disp t, [base+disp] ; load_mem dst, [t] -> load_mem_disp dst, [base + disp]
            if (t0 == OP_TYPE_X64_ADD_RI || t0 == OP_TYPE_X64_LEA_MEM_DISP) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                uint8_t t = (uint8_t)(v0 & 15);
                uint8_t base = (t0 == OP_TYPE_X64_LEA_MEM_DISP) ? (uint8_t)((v0 >> 4) & 15) : t;
                int32_t disp = (int32_t)(uint32_t)(v0 >> 32);
                if (t1 == OP_TYPE_X64_LOAD_MEM) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t dst = (uint8_t)(v1 & 15), addr = (uint8_t)((v1 >> 4) & 15);
                    int nb = (int)((v1 >> 8) & 15), sgn = (int)((v1 >> 12) & 1);
                    if (addr == t && (dst == t || x64_reg_dead_after(ctx, blk, j + 1, n_ops, t))) {
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_LOAD_MEM_DISP,
                            (int64_t)dst | ((int64_t)base << 4) | ((int64_t)nb << 8) |
                            ((int64_t)sgn << 12) | ((int64_t)(uint32_t)disp << 32));
                        changed = true;
                        continue;
                    }
                } else if (t1 == OP_TYPE_X64_STORE_MEM) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t addr = (uint8_t)(v1 & 15), val = (uint8_t)((v1 >> 4) & 15);
                    int nb = (int)((v1 >> 8) & 15);
                    if (addr == t && val != t && x64_reg_dead_after(ctx, blk, j + 1, n_ops, t)) {
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_STORE_MEM_DISP,
                            (int64_t)base | ((int64_t)val << 4) | ((int64_t)nb << 8) |
                            ((int64_t)(uint32_t)disp << 32));
                        changed = true;
                        continue;
                    }
                } else if (t1 == OP_TYPE_X64_LOAD_MEM_XMM) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t xd = (uint8_t)(v1 & 15), addr = (uint8_t)((v1 >> 4) & 15);
                    int is_ss = (int)((v1 >> 8) & 1);
                    if (addr == t && x64_reg_dead_after(ctx, blk, j + 1, n_ops, t)) {
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_LOAD_MEM_XMM_DISP,
                            (int64_t)xd | ((int64_t)base << 4) | ((int64_t)is_ss << 8) |
                            ((int64_t)(uint32_t)disp << 32));
                        changed = true;
                        continue;
                    }
                } else if (t1 == OP_TYPE_X64_STORE_MEM_XMM) {
                    uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                    uint8_t addr = (uint8_t)(v1 & 15), xs = (uint8_t)((v1 >> 4) & 15);
                    int is_ss = (int)((v1 >> 8) & 1);
                    if (addr == t && x64_reg_dead_after(ctx, blk, j + 1, n_ops, t)) {
                        MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_STORE_MEM_XMM_DISP,
                            (int64_t)base | ((int64_t)xs << 4) | ((int64_t)is_ss << 8) |
                            ((int64_t)(uint32_t)disp << 32));
                        changed = true;
                        continue;
                    }
                }
            }

            // 4. SIB array addressing fusion:
            //    [shift_imm idx, shl, 1..3] ; alu_rr add, base, idx ; load/store [base]
            if (t0 == OP_TYPE_X64_ALU_RR) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                uint8_t base = (uint8_t)(v0 & 15), idx = (uint8_t)((v0 >> 4) & 15);
                int w0 = (int)((v0 >> 8) & 1);
                uint8_t opc = (uint8_t)((v0 >> 12) & 0xff);
                if (opc == 0x01 && w0 == 1 && idx != RSP && base != idx) {
                    if (t1 == OP_TYPE_X64_LOAD_MEM) {
                        uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                        uint8_t dst = (uint8_t)(v1 & 15), addr = (uint8_t)((v1 >> 4) & 15);
                        int nb = (int)((v1 >> 8) & 15), sgn = (int)((v1 >> 12) & 1);
                        if (addr == base && (dst == base || x64_reg_dead_after(ctx, blk, j + 1, n_ops, base))) {
                            int scale = 0;
                            // Check if preceding non-NOP op is shift_imm(idx, shl, 1..3).
                            size_t p = i;
                            while (p > 0) {
                                p--;
                                if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, blk, p)) != OP_TYPE_X64_NOP) break;
                            }
                            if (p < i) {
                                MLIR_OpHandle opp = MLIR_GetBlockOp(ctx, blk, p);
                                if (MLIR_GetOpType(ctx, opp) == OP_TYPE_X64_SHIFT_IMM) {
                                    uint64_t vp = (uint64_t)MLIR_GetOpInlineI64(ctx, opp);
                                    uint8_t sr = (uint8_t)(vp & 15);
                                    int ext = (int)((vp >> 4) & 7), sw = (int)((vp >> 8) & 1);
                                    uint8_t sh = (uint8_t)((vp >> 12) & 0xff);
                                    if (sr == idx && ext == 4 && sw == 1 && sh >= 1 && sh <= 3 &&
                                        (dst == idx || x64_reg_dead_after(ctx, blk, j + 1, n_ops, idx))) {
                                        scale = (int)sh;
                                        MLIR_RewriteVoidInlineI64(ctx, opp, OP_TYPE_X64_NOP, 0);
                                    }
                                }
                            }
                            MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                            MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_LOAD_MEM_SIB,
                                (int64_t)dst | ((int64_t)base << 4) | ((int64_t)idx << 8) |
                                ((int64_t)scale << 12) | ((int64_t)nb << 16) | ((int64_t)sgn << 20));
                            changed = true;
                            continue;
                        }
                    } else if (t1 == OP_TYPE_X64_STORE_MEM) {
                        uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                        uint8_t addr = (uint8_t)(v1 & 15), val = (uint8_t)((v1 >> 4) & 15);
                        int nb = (int)((v1 >> 8) & 15);
                        if (addr == base && val != base && x64_reg_dead_after(ctx, blk, j + 1, n_ops, base)) {
                            int scale = 0;
                            size_t p = i;
                            while (p > 0) {
                                p--;
                                if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, blk, p)) != OP_TYPE_X64_NOP) break;
                            }
                            if (p < i) {
                                MLIR_OpHandle opp = MLIR_GetBlockOp(ctx, blk, p);
                                if (MLIR_GetOpType(ctx, opp) == OP_TYPE_X64_SHIFT_IMM) {
                                    uint64_t vp = (uint64_t)MLIR_GetOpInlineI64(ctx, opp);
                                    uint8_t sr = (uint8_t)(vp & 15);
                                    int ext = (int)((vp >> 4) & 7), sw = (int)((vp >> 8) & 1);
                                    uint8_t sh = (uint8_t)((vp >> 12) & 0xff);
                                    if (sr == idx && ext == 4 && sw == 1 && sh >= 1 && sh <= 3 &&
                                        val != idx && x64_reg_dead_after(ctx, blk, j + 1, n_ops, idx)) {
                                        scale = (int)sh;
                                        MLIR_RewriteVoidInlineI64(ctx, opp, OP_TYPE_X64_NOP, 0);
                                    }
                                }
                            }
                            MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                            MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_STORE_MEM_SIB,
                                (int64_t)base | ((int64_t)idx << 4) | ((int64_t)val << 8) |
                                ((int64_t)scale << 12) | ((int64_t)nb << 16));
                            changed = true;
                            continue;
                        }
                    } else if (t1 == OP_TYPE_X64_LOAD_MEM_XMM) {
                        uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                        uint8_t xd = (uint8_t)(v1 & 15), addr = (uint8_t)((v1 >> 4) & 15);
                        int is_ss = (int)((v1 >> 8) & 1);
                        if (addr == base && x64_reg_dead_after(ctx, blk, j + 1, n_ops, base)) {
                            int scale = 0;
                            size_t p = i;
                            while (p > 0) {
                                p--;
                                if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, blk, p)) != OP_TYPE_X64_NOP) break;
                            }
                            if (p < i) {
                                MLIR_OpHandle opp = MLIR_GetBlockOp(ctx, blk, p);
                                if (MLIR_GetOpType(ctx, opp) == OP_TYPE_X64_SHIFT_IMM) {
                                    uint64_t vp = (uint64_t)MLIR_GetOpInlineI64(ctx, opp);
                                    uint8_t sr = (uint8_t)(vp & 15);
                                    int ext = (int)((vp >> 4) & 7), sw = (int)((vp >> 8) & 1);
                                    uint8_t sh = (uint8_t)((vp >> 12) & 0xff);
                                    if (sr == idx && ext == 4 && sw == 1 && sh >= 1 && sh <= 3 &&
                                        x64_reg_dead_after(ctx, blk, j + 1, n_ops, idx)) {
                                        scale = (int)sh;
                                        MLIR_RewriteVoidInlineI64(ctx, opp, OP_TYPE_X64_NOP, 0);
                                    }
                                }
                            }
                            MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                            MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_LOAD_MEM_XMM_SIB,
                                (int64_t)xd | ((int64_t)base << 4) | ((int64_t)idx << 8) |
                                ((int64_t)scale << 12) | ((int64_t)is_ss << 16));
                            changed = true;
                            continue;
                        }
                    } else if (t1 == OP_TYPE_X64_STORE_MEM_XMM) {
                        uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                        uint8_t addr = (uint8_t)(v1 & 15), xs = (uint8_t)((v1 >> 4) & 15);
                        int is_ss = (int)((v1 >> 8) & 1);
                        if (addr == base && x64_reg_dead_after(ctx, blk, j + 1, n_ops, base)) {
                            int scale = 0;
                            size_t p = i;
                            while (p > 0) {
                                p--;
                                if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, blk, p)) != OP_TYPE_X64_NOP) break;
                            }
                            if (p < i) {
                                MLIR_OpHandle opp = MLIR_GetBlockOp(ctx, blk, p);
                                if (MLIR_GetOpType(ctx, opp) == OP_TYPE_X64_SHIFT_IMM) {
                                    uint64_t vp = (uint64_t)MLIR_GetOpInlineI64(ctx, opp);
                                    uint8_t sr = (uint8_t)(vp & 15);
                                    int ext = (int)((vp >> 4) & 7), sw = (int)((vp >> 8) & 1);
                                    uint8_t sh = (uint8_t)((vp >> 12) & 0xff);
                                    if (sr == idx && ext == 4 && sw == 1 && sh >= 1 && sh <= 3 &&
                                        x64_reg_dead_after(ctx, blk, j + 1, n_ops, idx)) {
                                        scale = (int)sh;
                                        MLIR_RewriteVoidInlineI64(ctx, opp, OP_TYPE_X64_NOP, 0);
                                    }
                                }
                            }
                            MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                            MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_STORE_MEM_XMM_SIB,
                                (int64_t)base | ((int64_t)idx << 4) | ((int64_t)xs << 8) |
                                ((int64_t)scale << 12) | ((int64_t)is_ss << 16));
                            changed = true;
                            continue;
                        }
                    } else if (t1 == OP_TYPE_X64_MOV_RI32) {
                        // alu_rr add, base, idx ; mov_ri32 val, imm ; store_mem [base], val
                        // -> store_mem_sib_imm8 [base + idx*scale], imm8 (for nb==1, val dead)
                        //    or mov_ri32 val, imm ; store_mem_sib [base + idx*scale], val
                        uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                        uint8_t vr = (uint8_t)(v1 & 15);
                        uint32_t imm32 = (uint32_t)(v1 >> 32);
                        size_t k = x64_next_live(ctx, blk, j + 1, n_ops);
                        if (vr != base && vr != idx && k < n_ops) {
                            MLIR_OpHandle op2 = MLIR_GetBlockOp(ctx, blk, k);
                            if (MLIR_GetOpType(ctx, op2) == OP_TYPE_X64_STORE_MEM) {
                                uint64_t v2 = (uint64_t)MLIR_GetOpInlineI64(ctx, op2);
                                uint8_t addr = (uint8_t)(v2 & 15), val = (uint8_t)((v2 >> 4) & 15);
                                int nb = (int)((v2 >> 8) & 15);
                                if (addr == base && val == vr &&
                                    x64_reg_dead_after(ctx, blk, k + 1, n_ops, base)) {
                                    int scale = 0;
                                    size_t p = i;
                                    while (p > 0) {
                                        p--;
                                        if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, blk, p)) != OP_TYPE_X64_NOP) break;
                                    }
                                    if (p < i) {
                                        MLIR_OpHandle opp = MLIR_GetBlockOp(ctx, blk, p);
                                        if (MLIR_GetOpType(ctx, opp) == OP_TYPE_X64_SHIFT_IMM) {
                                            uint64_t vp = (uint64_t)MLIR_GetOpInlineI64(ctx, opp);
                                            uint8_t sr = (uint8_t)(vp & 15);
                                            int ext = (int)((vp >> 4) & 7), sw = (int)((vp >> 8) & 1);
                                            uint8_t sh = (uint8_t)((vp >> 12) & 0xff);
                                            if (sr == idx && ext == 4 && sw == 1 && sh >= 1 && sh <= 3 &&
                                                x64_reg_dead_after(ctx, blk, k + 1, n_ops, idx)) {
                                                scale = (int)sh;
                                                MLIR_RewriteVoidInlineI64(ctx, opp, OP_TYPE_X64_NOP, 0);
                                            }
                                        }
                                    }
                                    MLIR_RewriteVoidInlineI64(ctx, op0, OP_TYPE_X64_NOP, 0);
                                    if (nb == 1 && x64_reg_dead_after(ctx, blk, k + 1, n_ops, vr)) {
                                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                                        MLIR_RewriteVoidInlineI64(ctx, op2, OP_TYPE_X64_STORE_MEM_SIB_IMM8,
                                            (int64_t)base | ((int64_t)idx << 4) |
                                            ((int64_t)scale << 12) | ((int64_t)(imm32 & 0xffu) << 16));
                                    } else {
                                        MLIR_RewriteVoidInlineI64(ctx, op2, OP_TYPE_X64_STORE_MEM_SIB,
                                            (int64_t)base | ((int64_t)idx << 4) | ((int64_t)val << 8) |
                                            ((int64_t)scale << 12) | ((int64_t)nb << 16));
                                    }
                                    changed = true;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // 5. Forward base copy into SIB load/store across up to 4 intervening ops
            //    (e.g. mov rax, r10 ; movsxd rcx, r12d ; load_mem_sib r14, [rax + rcx*4])
            if (t0 == OP_TYPE_X64_LOAD_MEM_SIB || t0 == OP_TYPE_X64_STORE_MEM_SIB ||
                t0 == OP_TYPE_X64_LOAD_MEM_XMM_SIB || t0 == OP_TYPE_X64_STORE_MEM_XMM_SIB) {
                uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                bool is_load = (t0 == OP_TYPE_X64_LOAD_MEM_SIB || t0 == OP_TYPE_X64_LOAD_MEM_XMM_SIB);
                bool is_xmm  = (t0 == OP_TYPE_X64_LOAD_MEM_XMM_SIB || t0 == OP_TYPE_X64_STORE_MEM_XMM_SIB);
                uint8_t base = is_load ? (uint8_t)((v0 >> 4) & 15) : (uint8_t)(v0 & 15);
                uint8_t dst_or_val = is_load ? (uint8_t)(v0 & 15) : (uint8_t)((v0 >> 8) & 15);
                bool base_dead = (!is_xmm && is_load && dst_or_val == base) ||
                                 x64_reg_dead_after(ctx, blk, i + 1, n_ops, base);
                if (base_dead && !(!is_xmm && !is_load && dst_or_val == base)) {
                    uint16_t clob = 0;
                    int steps = 0;
                    for (size_t p = i; p > 0 && steps < 4; ) {
                        p--;
                        MLIR_OpHandle opp = MLIR_GetBlockOp(ctx, blk, p);
                        MLIR_OpType tp = MLIR_GetOpType(ctx, opp);
                        if (tp == OP_TYPE_X64_NOP) continue;
                        steps++;
                        if (tp == OP_TYPE_X64_MOV_RR) {
                            uint64_t vp = (uint64_t)MLIR_GetOpInlineI64(ctx, opp);
                            uint8_t dp = (uint8_t)(vp & 15), sp = (uint8_t)((vp >> 4) & 15);
                            int wp = (int)((vp >> 8) & 1);
                            if (dp == base && wp == 1 && !(clob & (1u << sp))) {
                                MLIR_RewriteVoidInlineI64(ctx, opp, OP_TYPE_X64_NOP, 0);
                                uint64_t nv = is_load
                                    ? ((v0 & ~(uint64_t)(15u << 4)) | ((uint64_t)sp << 4))
                                    : ((v0 & ~(uint64_t)15u) | (uint64_t)sp);
                                MLIR_RewriteVoidInlineI64(ctx, op0, t0, (int64_t)nv);
                                changed = true;
                                break;
                            }
                        }
                        uint16_t r = 0, w = 0;
                        if (!x64_op_regs(ctx, opp, &r, &w)) break;
                        if ((r | w) & (1u << base)) break;
                        clob |= w;
                    }
                }
            }

            // 6. Scratch result forwarding into MOV_RR:
            //    load_rbp/mov_ri32/load_mem* t, ... ; mov dst, t (t dead) -> op dst, ...
            if (t1 == OP_TYPE_X64_MOV_RR) {
                uint64_t v1 = (uint64_t)MLIR_GetOpInlineI64(ctx, op1);
                uint8_t d1 = (uint8_t)(v1 & 15), s1 = (uint8_t)((v1 >> 4) & 15);
                int w1 = (int)((v1 >> 8) & 1);
                if (t0 == OP_TYPE_X64_LOAD_RBP || t0 == OP_TYPE_X64_MOV_RI32) {
                    uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                    uint8_t d0 = (uint8_t)(v0 & 15);
                    int w0 = (int)((v0 >> 8) & 1);
                    if (d0 == s1 && w0 == w1 && x64_reg_dead_after(ctx, blk, j + 1, n_ops, d0)) {
                        uint64_t nv0 = (v0 & ~(uint64_t)15u) | (uint64_t)d1;
                        MLIR_RewriteVoidInlineI64(ctx, op0, t0, (int64_t)nv0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                        changed = true;
                        continue;
                    }
                } else if (t0 == OP_TYPE_X64_LOAD_MEM || t0 == OP_TYPE_X64_LOAD_MEM_DISP ||
                           t0 == OP_TYPE_X64_LOAD_MEM_SIB) {
                    uint64_t v0 = (uint64_t)MLIR_GetOpInlineI64(ctx, op0);
                    uint8_t d0 = (uint8_t)(v0 & 15);
                    int nb = (t0 == OP_TYPE_X64_LOAD_MEM_SIB) ? (int)((v0 >> 16) & 15) : (int)((v0 >> 8) & 15);
                    if (d0 == s1 && ((nb == 4 && w1 == 0) || (nb == 8 && w1 == 1)) &&
                        x64_reg_dead_after(ctx, blk, j + 1, n_ops, d0)) {
                        uint64_t nv0 = (v0 & ~(uint64_t)15u) | (uint64_t)d1;
                        MLIR_RewriteVoidInlineI64(ctx, op0, t0, (int64_t)nv0);
                        MLIR_RewriteVoidInlineI64(ctx, op1, OP_TYPE_X64_NOP, 0);
                        changed = true;
                        continue;
                    }
                }
            }
        }
        if (!changed) break;
    }
}

// ---------------------------------------------------------------------------
// Pass 2 of the x64.* MLIR dialect backend: encode a block of `x64.*` ops
// into x86_64 machine bytes + relocations + intra-function branch fixups.
// ---------------------------------------------------------------------------
static void x64_encode_block(FnCtx *F, MLIR_BlockHandle blk, size_t n_labels) {
    MLIR_Context *ctx = F->ctx;
    static uint32_t *g_lbl_off = NULL;
    static uint8_t  *g_lbl_seen = NULL;
    static uint8_t  *g_lbl_loop = NULL;
    static size_t g_lbl_cap = 0;
    size_t need = n_labels ? n_labels : 1;
    if (need > g_lbl_cap) {
        if (g_lbl_off) buddy_free(g_lbl_off);
        if (g_lbl_seen) buddy_free(g_lbl_seen);
        if (g_lbl_loop) buddy_free(g_lbl_loop);
        size_t want = need < 64 ? 64 : need * 2;
        size_t actual_off = 0, actual_seen = 0, actual_loop = 0;
        g_lbl_off = (uint32_t *)buddy_alloc(want * sizeof(uint32_t), &actual_off);
        g_lbl_seen = (uint8_t *)buddy_alloc(want * sizeof(uint8_t), &actual_seen);
        g_lbl_loop = (uint8_t *)buddy_alloc(want * sizeof(uint8_t), &actual_loop);
        g_lbl_cap = actual_off / sizeof(uint32_t);
        if (actual_seen < g_lbl_cap) g_lbl_cap = actual_seen;
        if (actual_loop < g_lbl_cap) g_lbl_cap = actual_loop;
    }
    memset(g_lbl_seen, 0, need);
    memset(g_lbl_loop, 0, need);
    size_t n_ops = MLIR_GetBlockNumOps(ctx, blk);
    // At -O1/-O2, identify backward-branch target labels (loop headers) for 16-byte alignment.
    if (g_x64_opt_level >= 1) {
        for (size_t i = 0; i < n_ops; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blk, i);
            MLIR_OpType ty = MLIR_GetOpType(ctx, op);
            if (ty == OP_TYPE_X64_LABEL) {
                int64_t id = MLIR_GetOpInlineI64(ctx, op);
                if ((size_t)id < need) g_lbl_seen[(size_t)id] = 1;
            } else if (ty == OP_TYPE_X64_JMP) {
                int lbl = (int)MLIR_GetOpInlineI64(ctx, op);
                if ((size_t)lbl < need && g_lbl_seen[(size_t)lbl]) g_lbl_loop[(size_t)lbl] = 1;
            } else if (ty == OP_TYPE_X64_JCC) {
                uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
                int lbl = (int)(uint32_t)(v >> 32);
                if ((size_t)lbl < need && g_lbl_seen[(size_t)lbl]) g_lbl_loop[(size_t)lbl] = 1;
            }
        }
        memset(g_lbl_seen, 0, need);
    }
    F->x64_blk = MLIR_INVALID_HANDLE; // switch low-level helpers to direct byte emission
    F->n_fix = 0;

    for (size_t i = 0; i < n_ops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blk, i);
        MLIR_OpType ty = MLIR_GetOpType(ctx, op);
        switch (ty) {
        case OP_TYPE_X64_NOP:
            break;
        case OP_TYPE_X64_LABEL: {
            int64_t id = MLIR_GetOpInlineI64(ctx, op);
            if (g_lbl_loop[(size_t)id]) {
                uint32_t rem = here(F) & 15u;
                if (rem != 0) {
                    uint32_t pad = 16u - rem;
                    while (pad >= 4) { eb(F, 0x0F); eb(F, 0x1F); eb(F, 0x40); eb(F, 0x00); pad -= 4; }
                    if (pad == 3)      { eb(F, 0x0F); eb(F, 0x1F); eb(F, 0x00); }
                    else if (pad == 2) { eb(F, 0x66); eb(F, 0x90); }
                    else if (pad == 1) { eb(F, 0x90); }
                }
            }
            g_lbl_off[(size_t)id] = here(F);
            g_lbl_seen[(size_t)id] = 1;
            break;
        }
        case OP_TYPE_X64_PUSH_R: {
            uint8_t r = (uint8_t)MLIR_GetOpInlineI64(ctx, op);
            if (r >= 8) eb(F, 0x41);
            eb(F, (uint8_t)(0x50 + (r & 7)));
            break;
        }
        case OP_TYPE_X64_SUB_RSP: {
            uint32_t imm = (uint32_t)MLIR_GetOpInlineI64(ctx, op);
            if (imm <= 127) { eb(F, 0x48); eb(F, 0x83); modrm(F, 3, 5, RSP); eb(F, (uint8_t)imm); }
            else            { rex(F, 1, 0, 0, RSP); eb(F, 0x81); modrm(F, 3, 5, RSP); e32(F, imm); }
            break;
        }
        case OP_TYPE_X64_ADD_RSP: {
            uint32_t imm = (uint32_t)MLIR_GetOpInlineI64(ctx, op);
            if (imm <= 127) { eb(F, 0x48); eb(F, 0x83); modrm(F, 3, 0, RSP); eb(F, (uint8_t)imm); }
            else            { rex(F, 1, 0, 0, RSP); eb(F, 0x81); modrm(F, 3, 0, RSP); e32(F, imm); }
            break;
        }
        case OP_TYPE_X64_LEAVE: eb(F, 0xC9); break;
        case OP_TYPE_X64_RET:   eb(F, 0xC3); break;
        case OP_TYPE_X64_UD2:   eb(F, 0x0F); eb(F, 0x0B); break;
        case OP_TYPE_X64_MOV_RR: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            mov_rr(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_MOV_RI: {
            uint64_t v = (uint64_t)MLIR_GetAttributeInteger(ctx, MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE));
            int64_t imm = MLIR_GetAttributeInteger(ctx, MLIR_GetOpAttr(ctx, op, MLIR_KEY_IMM));
            mov_ri(F, (uint8_t)(v & 15), imm, (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_MOV_RI32: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            int w = (int)((v >> 8) & 1);
            int64_t imm = w ? (int64_t)(int32_t)(uint32_t)(v >> 32) : (int64_t)(uint32_t)(v >> 32);
            mov_ri(F, (uint8_t)(v & 15), imm, w);
            break;
        }
        case OP_TYPE_X64_ALU_RR: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            alu_rr(F, (uint8_t)((v >> 12) & 0xff), (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_ALU_RI: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            alu_ri(F, (uint8_t)((v >> 4) & 7), (uint8_t)(v & 15), (int32_t)(uint32_t)(v >> 32), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_ADD_RI: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            uint8_t reg = (uint8_t)(v & 15);
            uint32_t imm = (uint32_t)(v >> 32);
            rex(F, 1, 0, 0, reg); eb(F, 0x81); modrm(F, 3, 0, reg); e32(F, imm);
            break;
        }
        case OP_TYPE_X64_AND_EAX_IMM: {
            uint32_t mask = (uint32_t)MLIR_GetOpInlineI64(ctx, op);
            eb(F, 0x25); e32(F, mask);
            break;
        }
        case OP_TYPE_X64_IMUL_RR: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            imul_rr(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_IMUL_RI: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            imul_ri(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int32_t)(uint32_t)(v >> 32), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_IDIV: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            uint8_t reg = (uint8_t)(v & 15); int w = (int)((v >> 8) & 1);
            if (w) eb(F, 0x48);
            eb(F, 0x99);
            rex(F, w, 0, 0, reg); eb(F, 0xF7); modrm(F, 3, 7, reg);
            break;
        }
        case OP_TYPE_X64_DIV: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            uint8_t reg = (uint8_t)(v & 15); int w = (int)((v >> 8) & 1);
            alu_rr(F, 0x31, RDX, RDX, 0);
            rex(F, w, 0, 0, reg); eb(F, 0xF7); modrm(F, 3, 6, reg);
            break;
        }
        case OP_TYPE_X64_SHIFT_CL: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            uint8_t reg = (uint8_t)(v & 15); int ext = (int)((v >> 4) & 7); int w = (int)((v >> 8) & 1);
            rex(F, w, 0, 0, reg); eb(F, 0xD3); modrm(F, 3, ext, reg);
            break;
        }
        case OP_TYPE_X64_SHIFT_IMM: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            uint8_t reg = (uint8_t)(v & 15); int ext = (int)((v >> 4) & 7);
            int w = (int)((v >> 8) & 1); uint8_t imm = (uint8_t)((v >> 12) & 0xff);
            if (imm == 1) { rex(F, w, 0, 0, reg); eb(F, 0xD1); modrm(F, 3, ext, reg); }
            else          { rex(F, w, 0, 0, reg); eb(F, 0xC1); modrm(F, 3, ext, reg); eb(F, imm); }
            break;
        }
        case OP_TYPE_X64_LOAD_RBP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_rbp(F, (uint8_t)(v & 15), (int32_t)(uint32_t)(v >> 32), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_STORE_RBP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_rbp(F, (int32_t)(uint32_t)(v >> 32), (uint8_t)(v & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_LOAD_RBP_SIZED: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_rbp_sized(F, (uint8_t)(v & 15), (int32_t)(uint32_t)(v >> 32), (int)((v >> 4) & 15));
            break;
        }
        case OP_TYPE_X64_STORE_RBP_SIZED: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_rbp_sized(F, (int32_t)(uint32_t)(v >> 32), (uint8_t)(v & 15), (int)((v >> 4) & 15));
            break;
        }
        case OP_TYPE_X64_LEA_RBP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            lea_rbp(F, (uint8_t)(v & 15), (int32_t)(uint32_t)(v >> 32));
            break;
        }
        case OP_TYPE_X64_LOAD_MEM: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_mem(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 15), ((v >> 12) & 1) != 0);
            break;
        }
        case OP_TYPE_X64_STORE_MEM: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_mem(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 15));
            break;
        }
        case OP_TYPE_X64_LOAD_MEM_DISP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_mem_disp(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                          (int)((v >> 8) & 15), ((v >> 12) & 1) != 0,
                          (int32_t)(uint32_t)(v >> 32));
            break;
        }
        case OP_TYPE_X64_STORE_MEM_DISP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_mem_disp(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                           (int)((v >> 8) & 15), (int32_t)(uint32_t)(v >> 32));
            break;
        }
        case OP_TYPE_X64_LOAD_MEM_SIB: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_mem_sib(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                         (uint8_t)((v >> 8) & 15), (int)((v >> 12) & 3),
                         (int)((v >> 16) & 15), ((v >> 20) & 1) != 0);
            break;
        }
        case OP_TYPE_X64_STORE_MEM_SIB: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_mem_sib(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                          (uint8_t)((v >> 8) & 15), (int)((v >> 12) & 3),
                          (int)((v >> 16) & 15));
            break;
        }
        case OP_TYPE_X64_STORE_MEM_SIB_IMM8: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_mem_sib_imm8(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                               (int)((v >> 12) & 3), (uint8_t)((v >> 16) & 0xff));
            break;
        }
        case OP_TYPE_X64_LEA_MEM_DISP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            lea_mem_disp(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                         (int32_t)(uint32_t)(v >> 32));
            break;
        }
        case OP_TYPE_X64_LEA_RIP_DATA: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            lea_rip_data(F, (uint8_t)(v & 15), (int64_t)(uint32_t)(v >> 32));
            break;
        }
        case OP_TYPE_X64_LEA_RIP_FUNC: {
            uint8_t reg = (uint8_t)MLIR_GetAttributeInteger(ctx, MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE));
            string sym = MLIR_GetAttributeString(ctx, MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME));
            lea_rip_func(F, reg, sym);
            break;
        }
        case OP_TYPE_X64_MOVSX: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            uint8_t dst = (uint8_t)(v & 15), src = (uint8_t)((v >> 4) & 15);
            int bits = (int)((v >> 8) & 0xff);
            if (bits == 32)      { rex(F, 1, dst, 0, src); eb(F, 0x63); modrm(F, 3, dst, src); }
            else if (bits == 16) { rex(F, 1, dst, 0, src); eb(F, 0x0F); eb(F, 0xBF); modrm(F, 3, dst, src); }
            else                 { rex(F, 1, dst, 0, src); eb(F, 0x0F); eb(F, 0xBE); modrm(F, 3, dst, src); }
            break;
        }
        case OP_TYPE_X64_SETCC_AL: {
            setcc_al(F, (int)MLIR_GetOpInlineI64(ctx, op));
            break;
        }
        case OP_TYPE_X64_SETCC8: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            setcc8(F, (int)(v & 15), (uint8_t)((v >> 4) & 15));
            break;
        }
        case OP_TYPE_X64_MOVZX_EAX_AL: {
            eb(F, 0x0F); eb(F, 0xB6); modrm(F, 3, RAX, RAX);
            break;
        }
        case OP_TYPE_X64_ALU8_AL_CL: {
            eb(F, (uint8_t)MLIR_GetOpInlineI64(ctx, op)); modrm(F, 3, RCX, RAX);
            break;
        }
        case OP_TYPE_X64_CMOVCC: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            cmovcc(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 12) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_JMP: {
            int lbl = (int)MLIR_GetOpInlineI64(ctx, op);
            if (g_lbl_seen[lbl]) {
                int32_t rel = (int32_t)g_lbl_off[lbl] - (int32_t)(here(F) + 2);
                if (rel >= -128 && rel <= 127) {
                    eb(F, 0xEB); eb(F, (uint8_t)(int8_t)rel);
                    break;
                }
            }
            eb(F, 0xE9); uint32_t fld = here(F); e32(F, 0);
            add_fix(F, lbl, fld);
            break;
        }
        case OP_TYPE_X64_JCC: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            int cc = (int)(v & 15);
            int lbl = (int)(uint32_t)(v >> 32);
            if (g_lbl_seen[lbl]) {
                int32_t rel = (int32_t)g_lbl_off[lbl] - (int32_t)(here(F) + 2);
                if (rel >= -128 && rel <= 127) {
                    eb(F, (uint8_t)(0x70 + cc)); eb(F, (uint8_t)(int8_t)rel);
                    break;
                }
            }
            eb(F, 0x0F); eb(F, (uint8_t)(0x80 + cc));
            uint32_t fld = here(F); e32(F, 0);
            add_fix(F, lbl, fld);
            break;
        }
        case OP_TYPE_X64_JTAB_DISPATCH: {
            size_t jid = (size_t)MLIR_GetOpInlineI64(ctx, op);
            // lea rcx, [rip + disp32]
            eb(F, 0x48); eb(F, 0x8D); eb(F, 0x0D);
            uint32_t fld = here(F); e32(F, 0);
            if (jid < F->n_jtabs) F->jtabs[jid].lea_disp_off = fld;
            // movsxd rdx, DWORD PTR [rcx + rax*4]
            eb(F, 0x48); eb(F, 0x63); eb(F, 0x14); eb(F, 0x81);
            // add rcx, rdx
            eb(F, 0x48); eb(F, 0x01); eb(F, 0xD1);
            // jmp rcx
            eb(F, 0xFF); eb(F, 0xE1);
            break;
        }
        case OP_TYPE_X64_CALL_SYM: {
            string sym = MLIR_GetAttributeString(ctx, MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME));
            eb(F, 0xE8); uint32_t fld = here(F); e32(F, 0);
            add_text_reloc(F, fld, sym);
            break;
        }
        case OP_TYPE_X64_CALL_REG: {
            uint8_t r = (uint8_t)MLIR_GetOpInlineI64(ctx, op);
            rex(F, 0, 0, 0, r); eb(F, 0xFF); modrm(F, 3, 2, r);
            break;
        }
        case OP_TYPE_X64_MOVQ_XMM_GPR: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            movq_xmm_gpr(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15));
            break;
        }
        case OP_TYPE_X64_MOVQ_GPR_XMM: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            movq_gpr_xmm(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15));
            break;
        }
        case OP_TYPE_X64_SSE_OP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            uint8_t xd = (uint8_t)(v & 15), xs = (uint8_t)((v >> 4) & 15);
            int is_ss = (int)((v >> 8) & 1); uint8_t opc = (uint8_t)((v >> 12) & 0xff);
            if (is_ss) sse_ss(F, opc, xd, xs); else sse_sd(F, opc, xd, xs);
            break;
        }
        case OP_TYPE_X64_SSE_MOV_RR: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            sse_mov_rr(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_SSE_XORPD: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            sse_xorpd(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15));
            break;
        }
        case OP_TYPE_X64_LOAD_RBP_XMM: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_rbp_xmm(F, (uint8_t)(v & 15), (int32_t)(uint32_t)(v >> 32), (int)((v >> 4) & 1));
            break;
        }
        case OP_TYPE_X64_STORE_RBP_XMM: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_rbp_xmm(F, (int32_t)(uint32_t)(v >> 32), (uint8_t)(v & 15), (int)((v >> 4) & 1));
            break;
        }
        case OP_TYPE_X64_LOAD_MEM_XMM: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_mem_xmm(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_STORE_MEM_XMM: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_mem_xmm(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_LOAD_MEM_XMM_DISP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_mem_xmm_disp(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                              (int)((v >> 8) & 1), (int32_t)(uint32_t)(v >> 32));
            break;
        }
        case OP_TYPE_X64_STORE_MEM_XMM_DISP: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_mem_xmm_disp(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                               (int)((v >> 8) & 1), (int32_t)(uint32_t)(v >> 32));
            break;
        }
        case OP_TYPE_X64_LOAD_MEM_XMM_SIB: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            load_mem_xmm_sib(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                             (uint8_t)((v >> 8) & 15), (int)((v >> 12) & 3), (int)((v >> 16) & 1));
            break;
        }
        case OP_TYPE_X64_STORE_MEM_XMM_SIB: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            store_mem_xmm_sib(F, (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15),
                              (uint8_t)((v >> 8) & 15), (int)((v >> 12) & 3), (int)((v >> 16) & 1));
            break;
        }
        case OP_TYPE_X64_SSE_OP_MEM_SIB: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            sse_op_mem_sib(F, (uint8_t)((v >> 20) & 0xff), (uint8_t)(v & 15),
                           (uint8_t)((v >> 4) & 15), (uint8_t)((v >> 8) & 15),
                           (int)((v >> 12) & 3), (int)((v >> 16) & 1));
            break;
        }
        case OP_TYPE_X64_CVTSI2F: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            cvtsi2f(F, (uint8_t)((v >> 12) & 0xff), (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_CVTF2SI: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            cvtf2si(F, (uint8_t)((v >> 12) & 0xff), (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15), (int)((v >> 8) & 1));
            break;
        }
        case OP_TYPE_X64_CVT_F2F: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            cvt_f2f(F, (uint8_t)((v >> 12) & 0xff), (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15));
            break;
        }
        case OP_TYPE_X64_UCOMIS: {
            uint64_t v = (uint64_t)MLIR_GetOpInlineI64(ctx, op);
            ucomis(F, (int)((v >> 8) & 1), (uint8_t)(v & 15), (uint8_t)((v >> 4) & 15));
            break;
        }
        default:
            fail(F, "unsupported x64 dialect op", MLIR_GetOpTypeName(ctx, ty));
            return;
        }
    }

    for (size_t i = 0; i < F->n_fix; i++) {
        int tb = F->fix[i].target_block;
        uint32_t fld = F->fix[i].field_off;
        int32_t rel = (int32_t)g_lbl_off[tb] - (int32_t)(fld + 4);
        buf_patch_le32(&F->f->code, fld, (uint32_t)rel);
    }

    // Emit PC-relative 32-bit jump tables in .text after function code.
    for (size_t j = 0; j < F->n_jtabs; j++) {
        X64JTab *jt = &F->jtabs[j];
        while (F->f->code.len & 3) eb(F, 0xCC);
        uint32_t jtab_off = here(F);
        int32_t disp = (int32_t)jtab_off - (int32_t)(jt->lea_disp_off + 4);
        buf_patch_le32(&F->f->code, jt->lea_disp_off, (uint32_t)disp);
        for (uint32_t k = 0; k < jt->span; k++) {
            int tb = jt->targets[k];
            int32_t rel = (int32_t)g_lbl_off[tb] - (int32_t)jtab_off;
            e32(F, (uint32_t)rel);
        }
    }
}

// Reusable per-function x64.* MLIR scratch block (L1-hot across functions;
// reset via MLIR_ResetBlockToCheckpoint after each function is encoded).
static MLIR_Context    *g_x64_cached_ctx = NULL;
static MLIR_TypeHandle  g_x64_cached_i64 = MLIR_INVALID_HANDLE;
static MLIR_BlockHandle g_x64_reusable_blk = MLIR_INVALID_HANDLE;

// Linear-scan FP register allocator for caller-saved XMM registers (xmm2..xmm15).
// Runs at -O1/-O2 after GP regalloc. Only allocates non-call-crossing FP values
// (xmm0/xmm1 remain pure scratch); values live across calls stay in GP/stack.
static void x64_alloc_xmm(FnCtx *F, MLIR_RegionHandle reg, size_t nb) {
    MLIR_Context *ctx = F->ctx;
    if (nb == 0 || nb > 16384) return;
    // Quick pass: check if any FP values exist at all.
    size_t n_fp = 0, n_calls = 0;
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        if (b > 0) {
            size_t na = MLIR_GetBlockNumArgs(ctx, bk);
            for (size_t a = 0; a < na; a++) {
                if (fp_bytes(ctx, MLIR_GetBlockArg(ctx, bk, a)) > 0) n_fp++;
            }
        }
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            MLIR_OpType ot = MLIR_GetOpType(ctx, op);
            if (ot == OP_TYPE_LLVM_CALL || ot == OP_TYPE_FUNC_CALL) n_calls++;
            if (MLIR_GetOpNumResults(ctx, op) == 1 && !x64_is_remat_type(ot)) {
                if (fp_bytes(ctx, MLIR_GetOpResult(ctx, op, 0)) > 0) n_fp++;
            }
        }
    }
    if (n_fp == 0 || n_fp > 4096) return;

    int32_t *blk_start = (int32_t *)malloc(nb * sizeof(int32_t));
    int32_t *blk_end   = (int32_t *)malloc(nb * sizeof(int32_t));
    int32_t *call_pos  = n_calls ? (int32_t *)malloc(n_calls * sizeof(int32_t)) : NULL;
    MLIR_ValueHandle *fp_val  = (MLIR_ValueHandle *)malloc(n_fp * sizeof(MLIR_ValueHandle));
    MLIR_ValueHandle *fp_hint = (MLIR_ValueHandle *)malloc(n_fp * sizeof(MLIR_ValueHandle));
    int32_t *fp_def = (int32_t *)malloc(n_fp * sizeof(int32_t));
    int32_t *fp_end = (int32_t *)malloc(n_fp * sizeof(int32_t));
    uint8_t *fp_bad = (uint8_t *)calloc(n_fp, 1);
    int8_t  *fp_reg = (int8_t  *)malloc(n_fp * sizeof(int8_t));
    if (!blk_start || !blk_end || !fp_val || !fp_hint || !fp_def || !fp_end || !fp_bad || !fp_reg) {
        free(blk_start); free(blk_end); free(call_pos);
        free(fp_val); free(fp_hint); free(fp_def); free(fp_end); free(fp_bad); free(fp_reg);
        return;
    }

    // Map value -> index in fp_val via a small open-addressed hash table.
    size_t hcap = 16;
    while (hcap < n_fp * 2) hcap *= 2;
    uintptr_t *hk = (uintptr_t *)calloc(hcap, sizeof(uintptr_t));
    int32_t   *hv = (int32_t   *)malloc(hcap * sizeof(int32_t));
    size_t hmask = hcap - 1;

    size_t fi = 0, ci = 0;
    int32_t pos = 1;
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        blk_start[b] = pos;
        if (b > 0) {
            size_t na = MLIR_GetBlockNumArgs(ctx, bk);
            for (size_t a = 0; a < na; a++) {
                MLIR_ValueHandle v = MLIR_GetBlockArg(ctx, bk, a);
                if (fp_bytes(ctx, v) > 0) {
                    fp_val[fi] = v; fp_hint[fi] = MLIR_INVALID_HANDLE;
                    fp_def[fi] = pos; fp_end[fi] = pos; fp_reg[fi] = -1;
                    uintptr_t k = (uintptr_t)v;
                    size_t s = (k * 1099511628211ull) & hmask;
                    while (hk[s]) s = (s + 1) & hmask;
                    hk[s] = k; hv[s] = (int32_t)fi;
                    fi++;
                }
            }
        }
        pos++;
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            MLIR_OpType ot = MLIR_GetOpType(ctx, op);
            if (ot == OP_TYPE_LLVM_CALL || ot == OP_TYPE_FUNC_CALL) {
                if (call_pos && ci < n_calls) call_pos[ci++] = pos;
            }
            if (MLIR_GetOpNumResults(ctx, op) == 1 && !x64_is_remat_type(ot)) {
                MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
                if (fp_bytes(ctx, r) > 0) {
                    fp_val[fi] = r; fp_hint[fi] = MLIR_INVALID_HANDLE;
                    if (ot == OP_TYPE_LLVM_FADD || ot == OP_TYPE_LLVM_FSUB ||
                        ot == OP_TYPE_LLVM_FMUL || ot == OP_TYPE_LLVM_FDIV) {
                        fp_hint[fi] = MLIR_GetOpOperand(ctx, op, 0);
                    }
                    fp_def[fi] = pos; fp_end[fi] = pos; fp_reg[fi] = -1;
                    uintptr_t k = (uintptr_t)r;
                    size_t s = (k * 1099511628211ull) & hmask;
                    while (hk[s]) s = (s + 1) & hmask;
                    hk[s] = k; hv[s] = (int32_t)fi;
                    fi++;
                }
            }
            pos++;
        }
        blk_end[b] = pos;
        pos++;
    }

    // Record uses (op operands and terminator successor operands).
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        int32_t opos = blk_start[b] + 1;
        for (size_t i = 0; i < no; i++, opos++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            size_t nop = MLIR_GetOpNumOperands(ctx, op);
            for (size_t k = 0; k < nop; k++) {
                uintptr_t uv = (uintptr_t)MLIR_GetOpOperand(ctx, op, k);
                size_t s = (uv * 1099511628211ull) & hmask;
                while (hk[s]) {
                    if (hk[s] == uv) {
                        int32_t idx = hv[s];
                        if (opos < fp_def[idx]) fp_bad[idx] = 1;
                        else if (opos > fp_end[idx]) fp_end[idx] = opos;
                        break;
                    }
                    s = (s + 1) & hmask;
                }
            }
            if (i + 1 == no) {
                size_t ns = MLIR_GetOpNumSuccessors(ctx, op);
                int32_t epos = blk_end[b];
                for (size_t si = 0; si < ns; si++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, si);
                    for (size_t k = 0; k < nso; k++) {
                        uintptr_t uv = (uintptr_t)MLIR_GetOpSuccessorOperand(ctx, op, si, k);
                        size_t s = (uv * 1099511628211ull) & hmask;
                        while (hk[s]) {
                            if (hk[s] == uv) {
                                int32_t idx = hv[s];
                                if (epos < fp_def[idx]) fp_bad[idx] = 1;
                                else if (epos > fp_end[idx]) fp_end[idx] = epos;
                                break;
                            }
                            s = (s + 1) & hmask;
                        }
                    }
                }
            }
        }
    }

    // Build block handle -> index map for back-edge detection.
    size_t bhcap = 16;
    while (bhcap < nb * 2) bhcap *= 2;
    uintptr_t *bhk = (uintptr_t *)calloc(bhcap, sizeof(uintptr_t));
    int32_t   *bhv = (int32_t   *)malloc(bhcap * sizeof(int32_t));
    size_t bhmask = bhcap - 1;
    for (size_t b = 0; b < nb; b++) {
        uintptr_t k = (uintptr_t)MLIR_GetRegionBlock(ctx, reg, b);
        size_t s = (k * 1099511628211ull) & bhmask;
        while (bhk[s]) s = (s + 1) & bhmask;
        bhk[s] = k; bhv[s] = (int32_t)b;
    }

    // Extend intervals across back-edges to fixed point (max 8 iterations).
    for (int iter = 0; iter < 8; iter++) {
        bool ext = false;
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            if (no == 0) continue;
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
            for (size_t si = 0; si < ns; si++) {
                uintptr_t sk = (uintptr_t)MLIR_GetOpSuccessor(ctx, term, si);
                size_t s = (sk * 1099511628211ull) & bhmask;
                int32_t tb = -1;
                while (bhk[s]) { if (bhk[s] == sk) { tb = bhv[s]; break; } s = (s + 1) & bhmask; }
                if (tb >= 0 && (size_t)tb <= b) {
                    int32_t loop_s = blk_start[tb], loop_e = blk_end[b];
                    for (size_t k = 0; k < fi; k++) {
                        if (!fp_bad[k] && fp_def[k] < loop_s && fp_end[k] >= loop_s && fp_end[k] < loop_e) {
                            fp_end[k] = loop_e;
                            ext = true;
                        }
                    }
                }
            }
        }
        if (!ext) break;
    }

    // Exclude any FP value whose live interval spans a call.
    for (size_t k = 0; k < fi; k++) {
        if (fp_bad[k]) continue;
        for (size_t c = 0; c < ci; c++) {
            if (fp_def[k] < call_pos[c] && call_pos[c] < fp_end[k]) {
                fp_bad[k] = 1;
                break;
            }
        }
    }

    // Linear scan over xmm2..xmm15 (14 caller-saved registers).
    int32_t active_end[16];
    memset(active_end, 0, sizeof(active_end));
    F->xrm.arena = MLIR_GetArenaAllocator(ctx);
    for (size_t k = 0; k < fi; k++) {
        if (fp_bad[k]) continue;
        int32_t start = fp_def[k];
        for (int r = 2; r < 16; r++) {
            if (active_end[r] && active_end[r] <= start) active_end[r] = 0;
        }
        int chosen = -1;
        // Prefer coalescing with operand 0 if it just expired (in-place accumulator!).
        if (fp_hint[k] != MLIR_INVALID_HANDLE) {
            uintptr_t uv = (uintptr_t)fp_hint[k];
            size_t s = (uv * 1099511628211ull) & hmask;
            while (hk[s]) {
                if (hk[s] == uv) {
                    int8_t hr = fp_reg[hv[s]];
                    if (hr >= 2 && active_end[hr] == 0) chosen = hr;
                    break;
                }
                s = (s + 1) & hmask;
            }
        }
        if (chosen < 0) {
            for (int r = 2; r < 16; r++) {
                if (active_end[r] == 0) { chosen = r; break; }
            }
        }
        if (chosen >= 2) {
            active_end[chosen] = fp_end[k];
            fp_reg[k] = (int8_t)chosen;
            sm_put(&F->xrm, fp_val[k], chosen);
            F->has_xrm = true;
        }
    }

    free(bhk); free(bhv);
    free(hk); free(hv);
    free(blk_start); free(blk_end); free(call_pos);
    free(fp_val); free(fp_hint); free(fp_def); free(fp_end); free(fp_bad); free(fp_reg);
}

static void emit_func_to_x64(MLIR_Context *ctx, MLIR_OpHandle fn, ObjFunc *f,
                             SymSet *refs, MLIR_BlockHandle retain_blk) {
    FnCtx F; memset(&F, 0, sizeof(F));
    F.ctx = ctx; F.f = f; F.ok = true;
    F.fix = g_x64_fix;
    F.c_fix = g_x64_fix_cap;
    F.gm = g_x64_gmap;
    F.module_mb = g_x64_mb;
    F.data_buf = g_x64_data_buf;
    F.wasm_lifted = g_x64_wasm_lifted;

    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);

    if (ctx != g_x64_cached_ctx) {
        g_x64_cached_ctx = ctx;
        g_x64_cached_i64 = MLIR_CreateTypeInteger(ctx, 64, false);
        g_x64_reusable_blk = MLIR_CreateBlock(ctx);
        MLIR_ReserveBlockOps(ctx, g_x64_reusable_blk, 8192);
    }
    MLIR_BlockHandle x64_blk = (retain_blk != MLIR_INVALID_HANDLE) ? retain_blk : g_x64_reusable_blk;
    uint32_t x64_op_cp = (retain_blk == MLIR_INVALID_HANDLE) ? MLIR_GetOpCheckpoint(ctx) : 0;
    F.x64_blk = x64_blk;
    F.x64_bd = MLIR_BuilderAtEnd(ctx, x64_blk);
    F.x64_i64 = g_x64_cached_i64;
    F.x64_next_label = (int)nb;

    // --- register allocation: shared linear-scan allocator -> F.rm
    //     (value -> physical register). Values it leaves out stay in frame
    //     slots. TINYC_X64_NOREGALLOC forces the pure slot model (A/B aid). ---
    F.rm.arena = MLIR_GetArenaAllocator(ctx);
    // The global allocator also packs the values it spills into `nslots`
    // shared frame slots (`sm_spill`: value -> slot index; values whose live
    // intervals don't overlap share a slot, by the same interval reasoning
    // that lets them share a register).
    SlotMap sm_spill; memset(&sm_spill, 0, sizeof(sm_spill));
    sm_spill.arena = MLIR_GetArenaAllocator(ctx);
    int32_t nslots = 0;
    bool used_global = false;
    static int x64_ra_mode = 0; // 1 = fast (-O0 default), 2 = global, 3 = noregalloc
    if (x64_ra_mode == 0) {
        if (getenv("TINYC_X64_NOREGALLOC"))      x64_ra_mode = 3;
        else if (getenv("TINYC_X64_GLOBAL_RA"))  x64_ra_mode = 2;
        else                                     x64_ra_mode = 1;
    }
    uint64_t t_ra = x64_clock();
    if (x64_ra_mode != 3) {
        RegTarget T = x64_regtarget();
        bool use_fast = (x64_ra_mode == 1 && g_x64_opt_level == 0 && !F.wasm_lifted);
        uint32_t um = use_fast
            ? mlir_regalloc_fast(ctx, reg, nb, &F.rm, &sm_spill, &nslots, &T)
            : mlir_regalloc_global(ctx, reg, nb, &F.rm, &sm_spill, &nslots, &T);
        if (um != RA_BAIL) F.fast_ra = use_fast;
        if (um == RA_BAIL) {
            memset(&F.rm, 0, sizeof(F.rm));
            F.rm.arena = MLIR_GetArenaAllocator(ctx);
            memset(&sm_spill, 0, sizeof(sm_spill));
            sm_spill.arena = MLIR_GetArenaAllocator(ctx);
            nslots = 0;
            um = mlir_regalloc_cfg(ctx, reg, nb, &F.rm, &T);
        } else {
            used_global = true;
        }
        F.used_mask = um;
        for (int k = 0; k < T.npool - T.ncaller; k++)
            if (um & (1u << k)) F.cs_reg[F.n_cs++] = T.pool[T.ncaller + k];
        if (g_x64_opt_level >= 1 && !F.fast_ra)
            x64_alloc_xmm(&F, reg, nb);
    }
    uint64_t t_isel = x64_clock();
    if (g_x64_times) g_x64_times->regalloc_ns += t_isel - t_ra;

    // Build O(N) exact clamped use counts when !fast_ra so icmp -> cond_br fusion,
    // in-place loop counters, and GEP-in-RAX work reliably at -O1/-O2 (even after
    // LLVM opt passes that mutate operands without def-use tracking).
    static uint8_t *g_x64_use_cnt = NULL;
    static size_t g_x64_use_cap = 0;
    if (!F.fast_ra) {
        uintptr_t min_v = ~(uintptr_t)0, max_v = 0;
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                if (MLIR_GetOpNumResults(ctx, op) == 1) {
                    uintptr_t vh = (uintptr_t)MLIR_GetOpResult(ctx, op, 0);
                    if (vh < min_v) min_v = vh;
                    if (vh > max_v) max_v = vh;
                }
            }
        }
        if (max_v >= min_v && (max_v - min_v) < 2000000u) {
            size_t need = (size_t)(max_v - min_v) + 1;
            if (need > g_x64_use_cap) {
                if (g_x64_use_cnt) buddy_free(g_x64_use_cnt);
                size_t want = need < 256 ? 256 : need * 2;
                size_t actual = 0;
                g_x64_use_cnt = (uint8_t *)buddy_alloc(want, &actual);
                g_x64_use_cap = actual;
            }
            memset(g_x64_use_cnt, 0, need);
            for (size_t b = 0; b < nb; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
                size_t no = MLIR_GetBlockNumOps(ctx, bk);
                for (size_t i = 0; i < no; i++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                    size_t nop = MLIR_GetOpNumOperands(ctx, op);
                    for (size_t k = 0; k < nop; k++) {
                        uintptr_t idx = (uintptr_t)MLIR_GetOpOperand(ctx, op, k) - min_v;
                        if (idx < need && g_x64_use_cnt[idx] < 2) g_x64_use_cnt[idx]++;
                    }
                    if (i + 1 == no) {
                        size_t ns = MLIR_GetOpNumSuccessors(ctx, op);
                        for (size_t s = 0; s < ns; s++) {
                            size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, s);
                            for (size_t k = 0; k < nso; k++) {
                                uintptr_t idx = (uintptr_t)MLIR_GetOpSuccessorOperand(ctx, op, s, k) - min_v;
                                if (idx < need && g_x64_use_cnt[idx] < 2) g_x64_use_cnt[idx]++;
                            }
                        }
                    }
                }
            }
            F.use_cnt = g_x64_use_cnt;
            F.use_base = min_v;
            F.use_len = need;
        }
    }

    // --- frame pre-pass: give a frame slot to every block arg + op result that
    //     has no register home; reserve alloca regions; size the parallel-move
    //     temp area. Spilled values use the allocator's shared slots
    //     (rbp-8 .. rbp-8*nslots); values the allocator did not place (all of
    //     them under TINYC_X64_NOREGALLOC or the block-local fallback) get a
    //     slot of their own after those; rematerialized constants get none.
    //     A slot per SSA value made the frame of a large function tens of KiB,
    //     which overflowed the 8 MiB stack in recursive code (the wasm lifter's
    //     lower_op <-> lower_region) of the self-hosted ELF compiler. ---
    int32_t cur = nslots * 8;
    uint32_t max_edge = 0;
    if (used_global) {
        F.spill = sm_spill;
        F.has_spill = (sm_spill.n > 0);
        // Only alloca regions (and unhomed entry args if TINYC_NO_PARAM_HOME) go in F.slots.
        vmap_init(&F.slots, 16);
        if (nb > 0) {
            MLIR_BlockHandle e0 = MLIR_GetRegionBlock(ctx, reg, 0);
            size_t na = MLIR_GetBlockNumArgs(ctx, e0);
            for (size_t a = 0; a < na; a++) {
                MLIR_ValueHandle av = MLIR_GetBlockArg(ctx, e0, a);
                int32_t dummy;
                if (!sm_get_fast(&F.rm, av, &dummy) &&
                    (!F.has_spill || !sm_get_fast(&F.spill, av, &dummy))) {
                    cur += 8;
                    vmap_put(&F.slots, av, cur);
                }
            }
        }
        if (F.fast_ra) {
            size_t n_allocas = 0;
            const MLIR_OpHandle *allocas = mlir_regalloc_fast_allocas(&n_allocas);
            for (size_t ai = 0; ai < n_allocas; ai++) {
                MLIR_OpHandle op = allocas[ai];
                unsigned rb = alloca_bytes(ctx, op);
                cur += (int32_t)rb;
                vmap_put(&F.slots, x64_region_key(op), cur);
            }
            max_edge = mlir_regalloc_fast_max_edge();
        } else {
            for (size_t b = 0; b < nb; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
                size_t no = MLIR_GetBlockNumOps(ctx, bk);
                for (size_t i = 0; i < no; i++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                    if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_ALLOCA) {
                        unsigned rb = alloca_bytes(ctx, op);
                        cur += (int32_t)rb;
                        vmap_put(&F.slots, x64_region_key(op), cur);
                    }
                }
                if (nb > 1 && no > 0) {
                    MLIR_OpHandle t = MLIR_GetBlockOp(ctx, bk, no - 1);
                    size_t ns = MLIR_GetOpNumSuccessors(ctx, t);
                    for (size_t s = 0; s < ns; s++) {
                        uint32_t c = (uint32_t)MLIR_GetOpNumSuccessorOperands(ctx, t, s);
                        if (c > max_edge) max_edge = c;
                    }
                }
            }
        }
    } else {
        size_t want = 0;
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            want += MLIR_GetBlockNumArgs(ctx, bk) + MLIR_GetBlockNumOps(ctx, bk) + 4;
        }
        vmap_init(&F.slots, want ? want : 8);
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t na = MLIR_GetBlockNumArgs(ctx, bk);
            for (size_t a = 0; a < na; a++) x64_assign_slot(&F, &sm_spill, MLIR_GetBlockArg(ctx, bk, a), &cur);
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                if (MLIR_GetOpNumRegions(ctx, op) > 0) { fail(&F, "nested region (expected flat cf)", MLIR_GetOpTypeName(ctx, MLIR_GetOpType(ctx, op))); }
                if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_ALLOCA) {
                    unsigned rb = alloca_bytes(ctx, op);
                    cur += (int32_t)rb;
                    vmap_put(&F.slots, x64_region_key(op), cur);  // region
                }
                if (MLIR_GetOpNumResults(ctx, op) == 1 && !x64_op_rematerialized(ctx, op))
                    x64_assign_slot(&F, &sm_spill, MLIR_GetOpResult(ctx, op, 0), &cur);
                if (i + 1 == no) {
                    size_t ns = MLIR_GetOpNumSuccessors(ctx, op);
                    for (size_t s = 0; s < ns; s++) {
                        uint32_t c = (uint32_t)MLIR_GetOpNumSuccessorOperands(ctx, op, s);
                        if (c > max_edge) max_edge = c;
                    }
                }
            }
        }
    }
    cur += (int32_t)(max_edge * 8);
    F.edge_tmp = (uint32_t)cur;          // temps occupy [rbp-cur .. ]
    F.cs_base = cur;                     // callee-saved spill area follows
    cur += (int32_t)(F.n_cs * 8);
    F.frame = (uint32_t)((cur + 15) & ~15);

    // --- prologue: push rbp ; mov rbp,rsp ; sub rsp, frame ---
    x64_emit1(&F, OP_TYPE_X64_PUSH_R, RBP);
    mov_rr(&F, RBP, RSP, 1);
    if (F.frame) x64_emit1(&F, OP_TYPE_X64_SUB_RSP, (int64_t)F.frame);

    // save the callee-saved registers the allocator used
    for (int j = 0; j < F.n_cs; j++)
        store_rbp(&F, F.cs_base + 8 * (j + 1), F.cs_reg[j], 1);

    // --- entry params: incoming arg regs -> entry block-arg homes (reg/slot).
    //     Uses parallel-move resolution (R11 cycle-break temp) because RDI/RSI/
    //     R8/R9 are in the allocatable caller-saved pool.
    //     Params 6.. arrive on the caller's stack at [rbp+16], [rbp+24], ... ---
    if (nb > 0) {
        MLIR_BlockHandle e0 = MLIR_GetRegionBlock(ctx, reg, 0);
        size_t na = MLIR_GetBlockNumArgs(ctx, e0);
        F.n_fixed = (uint32_t)na;
        size_t n_reg = na < 6 ? na : 6;
        if (n_reg > 0) {
            int     src_r[6], dst_r[6], mv_w[6];
            int32_t dst_s[6];
            uint8_t done[6];
            size_t  rem = 0;
            for (size_t a = 0; a < n_reg; a++) {
                MLIR_ValueHandle av = MLIR_GetBlockArg(ctx, e0, a);
                src_r[a] = (int)ARG_REGS[a];
                x64_val_home(&F, av, false, &dst_r[a], &dst_s[a]);
                mv_w[a] = val_w(ctx, av);
                if (dst_r[a] == src_r[a]) { done[a] = 1; }
                else { done[a] = 0; rem++; }
            }
            while (rem > 0) {
                bool progress = false;
                for (size_t a = 0; a < n_reg; a++) {
                    if (done[a]) continue;
                    bool blocked = false;
                    if (dst_r[a] >= 0) {
                        for (size_t j = 0; j < n_reg; j++) {
                            if (!done[j] && j != a && src_r[j] == dst_r[a]) { blocked = true; break; }
                        }
                    }
                    if (blocked) continue;
                    if (dst_r[a] >= 0) mov_rr(&F, (uint8_t)dst_r[a], (uint8_t)src_r[a], mv_w[a]);
                    else               store_rbp(&F, dst_s[a], (uint8_t)src_r[a], mv_w[a]);
                    done[a] = 1;
                    rem--;
                    progress = true;
                }
                if (!progress) {
                    size_t k = 0;
                    while (k < n_reg && done[k]) k++;
                    int saved = dst_r[k];
                    mov_rr(&F, R11, (uint8_t)saved, 1);
                    for (size_t j = 0; j < n_reg; j++) {
                        if (!done[j] && src_r[j] == saved) src_r[j] = R11;
                    }
                }
            }
        }
        for (size_t a = 6; a < na; a++) {
            int32_t disp = (int32_t)(16 + (a - 6) * 8);   // [rbp + disp]
            load_rbp(&F, RAX, -disp, 1);
            store_val(&F, MLIR_GetBlockArg(ctx, e0, a), RAX);
        }
    }

    // Precompute block predecessor counts & block-handle->index map for dense
    // switch jump-table detection at -O1/-O2.
    uint32_t *pred_cnt = NULL;
    uint8_t  *skip_blk = NULL;
    uintptr_t *jbh_k = NULL;
    int32_t   *jbh_v = NULL;
    size_t jbh_mask = 0;
    if (g_x64_opt_level >= 1 && !F.fast_ra && nb >= 7) {
        size_t jbh_cap = 16;
        while (jbh_cap < nb * 2) jbh_cap *= 2;
        jbh_k = (uintptr_t *)calloc(jbh_cap, sizeof(uintptr_t));
        jbh_v = (int32_t   *)malloc(jbh_cap * sizeof(int32_t));
        pred_cnt = (uint32_t *)calloc(nb, sizeof(uint32_t));
        skip_blk = (uint8_t  *)calloc(nb, 1);
        if (jbh_k && jbh_v && pred_cnt && skip_blk) {
            jbh_mask = jbh_cap - 1;
            for (size_t b = 0; b < nb; b++) {
                uintptr_t k = (uintptr_t)MLIR_GetRegionBlock(ctx, reg, b);
                size_t s = (k * 1099511628211ull) & jbh_mask;
                while (jbh_k[s]) s = (s + 1) & jbh_mask;
                jbh_k[s] = k; jbh_v[s] = (int32_t)b;
            }
            for (size_t b = 0; b < nb; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
                size_t no = MLIR_GetBlockNumOps(ctx, bk);
                if (no == 0) continue;
                MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
                size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
                for (size_t si = 0; si < ns; si++) {
                    uintptr_t sk = (uintptr_t)MLIR_GetOpSuccessor(ctx, term, si);
                    size_t s = (sk * 1099511628211ull) & jbh_mask;
                    while (jbh_k[s]) {
                        if (jbh_k[s] == sk) { pred_cnt[jbh_v[s]]++; break; }
                        s = (s + 1) & jbh_mask;
                    }
                }
            }
        }
    }

    // --- emit blocks ---
    for (size_t b = 0; b < nb && F.ok; b++) {
        if (skip_blk && skip_blk[b]) continue;
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        MLIR_BlockHandle next_bk = (b + 1 < nb) ? MLIR_GetRegionBlock(ctx, reg, b + 1) : MLIR_INVALID_HANDLE;
        x64_emit1(&F, OP_TYPE_X64_LABEL, (int64_t)b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        // Check for fusable `icmp [-> select(1,0) -> icmp ne/eq 0] -> cond_br` at block tail,
        // skipping any rematerialized constant ops in between (which emit no code).
        size_t emit_end = no > 0 ? no - 1 : 0;
        MLIR_OpHandle fused_icmp = MLIR_INVALID_HANDLE;
        bool fused_invert = false;
        if (no >= 2) {
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            if (MLIR_GetOpType(ctx, term) == OP_TYPE_LLVM_COND_BR) {
                MLIR_ValueHandle cond = MLIR_GetOpOperand(ctx, term, 0);
                size_t p1 = no - 1;
                while (p1 > 0 && x64_op_rematerialized(ctx, MLIR_GetBlockOp(ctx, bk, p1 - 1))) p1--;
                if (p1 > 0) {
                    MLIR_OpHandle prev = MLIR_GetBlockOp(ctx, bk, p1 - 1);
                    if (MLIR_GetOpType(ctx, prev) == OP_TYPE_LLVM_ICMP &&
                        MLIR_GetOpResult(ctx, prev, 0) == cond &&
                        x64_val_dead_after_term(&F, cond)) {
                        MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, prev, MLIR_KEY_PREDICATE);
                        int64_t pred = pa ? MLIR_GetAttributeInteger(ctx, pa) : 0;
                        int64_t c0 = 0;
                        if ((pred == 0 || pred == 1) &&
                            is_const(ctx, MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, prev, 1)), &c0) &&
                            c0 == 0) {
                            MLIR_ValueHandle inner = MLIR_GetOpOperand(ctx, prev, 0);
                            size_t p2 = p1 - 1;
                            while (p2 > 0 && x64_op_rematerialized(ctx, MLIR_GetBlockOp(ctx, bk, p2 - 1))) p2--;
                            if (p2 > 0) {
                                MLIR_OpHandle sel = MLIR_GetBlockOp(ctx, bk, p2 - 1);
                                int64_t vt = 0, vf = 0;
                                if (MLIR_GetOpType(ctx, sel) == OP_TYPE_LLVM_SELECT &&
                                    MLIR_GetOpResult(ctx, sel, 0) == inner &&
                                    x64_val_dead_after_term(&F, inner) &&
                                    is_const(ctx, MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, sel, 1)), &vt) &&
                                    is_const(ctx, MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, sel, 2)), &vf) &&
                                    ((vt == 1 && vf == 0) || (vt == 0 && vf == 1))) {
                                    size_t p3 = p2 - 1;
                                    while (p3 > 0 && x64_op_rematerialized(ctx, MLIR_GetBlockOp(ctx, bk, p3 - 1))) p3--;
                                    if (p3 > 0) {
                                        MLIR_OpHandle ic0 = MLIR_GetBlockOp(ctx, bk, p3 - 1);
                                        if (MLIR_GetOpType(ctx, ic0) == OP_TYPE_LLVM_ICMP &&
                                            MLIR_GetOpResult(ctx, ic0, 0) == MLIR_GetOpOperand(ctx, sel, 0) &&
                                            x64_val_dead_after_term(&F, MLIR_GetOpResult(ctx, ic0, 0))) {
                                            fused_icmp = ic0;
                                            fused_invert = (pred == 0) ^ (vt == 0);
                                            emit_end = p3 - 1;
                                        }
                                    }
                                }
                            }
                            if (fused_icmp == MLIR_INVALID_HANDLE) {
                                fused_icmp = prev;
                                fused_invert = false;
                                emit_end = p1 - 1;
                            }
                        } else {
                            fused_icmp = prev;
                            fused_invert = false;
                            emit_end = p1 - 1;
                        }
                    }
                }
            }
        }
        // In-place loop-counter/accumulator coalescing at block tail:
        // If the last non-remat op before `llvm.br ^target(...)` is a binop
        // `r = a0 OP a1` whose ONLY use is as successor operand `si` of `br`,
        // and `darg[si]` has the SAME register home `rH` as `a0` (and no other
        // succ operand reads `rH`), map `r -> rH` in F.rm so the binop executes
        // in-place (`add rH, 1`) and emit_edge() drops the copy completely!
        if (!F.fast_ra && no >= 2) {
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            if (MLIR_GetOpType(ctx, term) == OP_TYPE_LLVM_BR) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, term, 0);
                size_t p = no - 1;
                while (p > 0 && x64_op_rematerialized(ctx, MLIR_GetBlockOp(ctx, bk, p - 1))) p--;
                if (nso > 0 && p > 0) {
                    MLIR_OpHandle prev = MLIR_GetBlockOp(ctx, bk, p - 1);
                    MLIR_OpType pt = MLIR_GetOpType(ctx, prev);
                    if (pt == OP_TYPE_LLVM_ADD || pt == OP_TYPE_LLVM_SUB ||
                        pt == OP_TYPE_LLVM_AND || pt == OP_TYPE_LLVM_OR ||
                        pt == OP_TYPE_LLVM_XOR) {
                        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, prev, 0);
                        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, prev, 0);
                        MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, prev, 1);
                        uint8_t r0, r1;
                        bool a1_in_r0 = val_in_reg(&F, a1, &r1) && val_in_reg(&F, a0, &r0) && r1 == r0;
                        if (x64_val_dead_after_term(&F, r) && val_in_reg(&F, a0, &r0) && !a1_in_r0) {
                            MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(ctx, term, 0);
                            bool match = false, conflict = false;
                            for (size_t s = 0; s < nso; s++) {
                                MLIR_ValueHandle sv = MLIR_GetOpSuccessorOperand(ctx, term, 0, s);
                                if (sv == r) {
                                    uint8_t rd;
                                    if (val_in_reg(&F, MLIR_GetBlockArg(ctx, tgt, s), &rd) && rd == r0)
                                        match = true;
                                    else
                                        conflict = true;
                                } else {
                                    uint8_t rs;
                                    if (val_in_reg(&F, sv, &rs) && rs == r0)
                                        conflict = true;
                                }
                            }
                            if (match && !conflict) {
                                x64_rm_set(&F.rm, r, (int32_t)r0);
                            }
                        }
                    }
                }
            }
        }
        // Keep single-use GEP addresses directly in RAX when immediately consumed
        // by a load or store (avoids copying RAX -> pool reg and unlocks SIB fusion).
        if (!F.fast_ra && emit_end >= 2) {
            for (size_t i = 0; i + 1 < emit_end; i++) {
                MLIR_OpHandle gop = MLIR_GetBlockOp(ctx, bk, i);
                if (MLIR_GetOpType(ctx, gop) != OP_TYPE_LLVM_GEP) continue;
                MLIR_ValueHandle gr = MLIR_GetOpResult(ctx, gop, 0);
                if (!x64_val_dead_after_term(&F, gr)) continue;
                size_t k = i + 1;
                while (k < emit_end && x64_op_rematerialized(ctx, MLIR_GetBlockOp(ctx, bk, k))) k++;
                if (k >= emit_end) continue;
                MLIR_OpHandle nop = MLIR_GetBlockOp(ctx, bk, k);
                MLIR_OpType nt = MLIR_GetOpType(ctx, nop);
                bool ok_fuse = (nt == OP_TYPE_LLVM_LOAD && MLIR_GetOpOperand(ctx, nop, 0) == gr) ||
                               (nt == OP_TYPE_LLVM_STORE && MLIR_GetOpOperand(ctx, nop, 1) == gr &&
                                MLIR_GetOpOperand(ctx, nop, 0) != gr);
                if (ok_fuse) {
                    x64_rm_set(&F.rm, gr, (int32_t)RAX);
                }
            }
        }
        for (size_t i = 0; i < emit_end && F.ok; i++)
            lower_op(&F, MLIR_GetBlockOp(ctx, bk, i), refs);
        if (no > 0 && F.ok) {
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            bool emitted_jtab = false;
            // Dense switch jump-table detection:
            // Recognize a chain of single-predecessor `icmp eq(sw_val, c_i) -> cond_br`
            // blocks with 0 successor operands and >= 5 dense cases (span <= 512, density >= 50%).
            if (jbh_k && fused_icmp != MLIR_INVALID_HANDLE && !fused_invert &&
                MLIR_GetOpType(ctx, term) == OP_TYPE_LLVM_COND_BR &&
                MLIR_GetOpNumSuccessorOperands(ctx, term, 0) == 0 &&
                MLIR_GetOpNumSuccessorOperands(ctx, term, 1) == 0) {
                MLIR_AttributeHandle pa0 = MLIR_GetOpAttr(ctx, fused_icmp, MLIR_KEY_PREDICATE);
                int64_t pred0 = pa0 ? MLIR_GetAttributeInteger(ctx, pa0) : -1;
                if (pred0 == 0) { // eq
                    MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, fused_icmp, 0);
                    MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, fused_icmp, 1);
                    MLIR_ValueHandle sw_val = MLIR_INVALID_HANDLE;
                    int64_t c0 = 0;
                    if (is_const(ctx, MLIR_GetValueDefiningOp(ctx, a1), &c0)) sw_val = a0;
                    else if (is_const(ctx, MLIR_GetValueDefiningOp(ctx, a0), &c0)) sw_val = a1;
                    int sw_ib = (sw_val != MLIR_INVALID_HANDLE) ? int_bits(ctx, sw_val) : 0;
                    if (sw_ib == 32 || sw_ib == 64) {
                        int64_t case_vals[512];
                        int     case_tgts[512];
                        int     chain_blks[512];
                        size_t  n_cases = 0, n_chain = 0;
                        // Lookup true/false succ indices of head block
                        int t0 = -1, f0 = -1;
                        uintptr_t tk0 = (uintptr_t)MLIR_GetOpSuccessor(ctx, term, 0);
                        size_t s = (tk0 * 1099511628211ull) & jbh_mask;
                        while (jbh_k[s]) { if (jbh_k[s] == tk0) { t0 = jbh_v[s]; break; } s = (s + 1) & jbh_mask; }
                        uintptr_t fk0 = (uintptr_t)MLIR_GetOpSuccessor(ctx, term, 1);
                        s = (fk0 * 1099511628211ull) & jbh_mask;
                        while (jbh_k[s]) { if (jbh_k[s] == fk0) { f0 = jbh_v[s]; break; } s = (s + 1) & jbh_mask; }
                        if (t0 >= 0 && f0 >= 0) {
                            case_vals[0] = c0; case_tgts[0] = t0; n_cases = 1;
                            int cur_bi = f0;
                            while (cur_bi >= 0 && (size_t)cur_bi != b && n_cases < 512 &&
                                   pred_cnt[cur_bi] == 1 && !skip_blk[cur_bi]) {
                                // Prevent cycles in chain
                                bool seen_c = false;
                                for (size_t q = 0; q < n_chain; q++) if (chain_blks[q] == cur_bi) { seen_c = true; break; }
                                if (seen_c) break;
                                MLIR_BlockHandle cbk = MLIR_GetRegionBlock(ctx, reg, (size_t)cur_bi);
                                if (MLIR_GetBlockNumArgs(ctx, cbk) != 0) break;
                                size_t cno = MLIR_GetBlockNumOps(ctx, cbk);
                                if (cno < 2) break;
                                MLIR_OpHandle cterm = MLIR_GetBlockOp(ctx, cbk, cno - 1);
                                if (MLIR_GetOpType(ctx, cterm) != OP_TYPE_LLVM_COND_BR ||
                                    MLIR_GetOpNumSuccessorOperands(ctx, cterm, 0) != 0 ||
                                    MLIR_GetOpNumSuccessorOperands(ctx, cterm, 1) != 0) break;
                                MLIR_ValueHandle ccond = MLIR_GetOpOperand(ctx, cterm, 0);
                                MLIR_OpHandle cicmp = MLIR_INVALID_HANDLE;
                                bool ok_blk = true;
                                for (size_t oi = 0; oi + 1 < cno; oi++) {
                                    MLIR_OpHandle cop = MLIR_GetBlockOp(ctx, cbk, oi);
                                    if (x64_op_rematerialized(ctx, cop)) continue;
                                    if (cicmp == MLIR_INVALID_HANDLE &&
                                        MLIR_GetOpType(ctx, cop) == OP_TYPE_LLVM_ICMP &&
                                        MLIR_GetOpResult(ctx, cop, 0) == ccond &&
                                        x64_val_dead_after_term(&F, ccond)) {
                                        cicmp = cop;
                                    } else {
                                        ok_blk = false;
                                        break;
                                    }
                                }
                                if (!ok_blk || cicmp == MLIR_INVALID_HANDLE) break;
                                MLIR_AttributeHandle cpa = MLIR_GetOpAttr(ctx, cicmp, MLIR_KEY_PREDICATE);
                                if (!cpa || MLIR_GetAttributeInteger(ctx, cpa) != 0) break;
                                MLIR_ValueHandle ca0 = MLIR_GetOpOperand(ctx, cicmp, 0);
                                MLIR_ValueHandle ca1 = MLIR_GetOpOperand(ctx, cicmp, 1);
                                int64_t cv = 0;
                                if (ca0 == sw_val && is_const(ctx, MLIR_GetValueDefiningOp(ctx, ca1), &cv)) {}
                                else if (ca1 == sw_val && is_const(ctx, MLIR_GetValueDefiningOp(ctx, ca0), &cv)) {}
                                else break;
                                int ct = -1, cf = -1;
                                uintptr_t ctk = (uintptr_t)MLIR_GetOpSuccessor(ctx, cterm, 0);
                                s = (ctk * 1099511628211ull) & jbh_mask;
                                while (jbh_k[s]) { if (jbh_k[s] == ctk) { ct = jbh_v[s]; break; } s = (s + 1) & jbh_mask; }
                                uintptr_t cfk = (uintptr_t)MLIR_GetOpSuccessor(ctx, cterm, 1);
                                s = (cfk * 1099511628211ull) & jbh_mask;
                                while (jbh_k[s]) { if (jbh_k[s] == cfk) { cf = jbh_v[s]; break; } s = (s + 1) & jbh_mask; }
                                if (ct < 0 || cf < 0) break;
                                chain_blks[n_chain++] = cur_bi;
                                case_vals[n_cases] = cv;
                                case_tgts[n_cases] = ct;
                                n_cases++;
                                cur_bi = cf;
                            }
                            if (n_cases >= 5 && cur_bi >= 0) {
                                int64_t min_c = case_vals[0], max_c = case_vals[0];
                                for (size_t ci = 1; ci < n_cases; ci++) {
                                    if (case_vals[ci] < min_c) min_c = case_vals[ci];
                                    if (case_vals[ci] > max_c) max_c = case_vals[ci];
                                }
                                uint64_t span64 = (uint64_t)max_c - (uint64_t)min_c + 1u;
                                if (span64 >= 5 && span64 <= 512 && (uint64_t)n_cases * 2u >= span64) {
                                    // Ensure all chain blocks are > b (not already emitted!).
                                    bool all_after = true;
                                    for (size_t q = 0; q < n_chain; q++)
                                        if ((size_t)chain_blks[q] <= b) { all_after = false; break; }
                                    if (all_after) {
                                        for (size_t q = 0; q < n_chain; q++)
                                            skip_blk[chain_blks[q]] = 1;
                                        if (F.n_jtabs == F.c_jtabs) {
                                            size_t nc = F.c_jtabs ? F.c_jtabs * 2 : 4;
                                            F.jtabs = (X64JTab *)realloc(F.jtabs, nc * sizeof(X64JTab));
                                            F.c_jtabs = nc;
                                        }
                                        size_t jid = F.n_jtabs++;
                                        X64JTab *jt = &F.jtabs[jid];
                                        jt->lea_disp_off = 0;
                                        jt->span = (uint32_t)span64;
                                        jt->targets = (int *)malloc(span64 * sizeof(int));
                                        uint8_t *filled = (uint8_t *)calloc(span64, 1);
                                        for (uint32_t k = 0; k < jt->span; k++) jt->targets[k] = cur_bi;
                                        for (size_t ci = 0; ci < n_cases; ci++) {
                                            uint32_t idx = (uint32_t)((uint64_t)case_vals[ci] - (uint64_t)min_c);
                                            if (!filled[idx]) { jt->targets[idx] = case_tgts[ci]; filled[idx] = 1; }
                                        }
                                        free(filled);
                                        int w = (sw_ib > 32) ? 1 : 0;
                                        load_val_w(&F, sw_val, RAX, w);
                                        if (min_c != 0) {
                                            if (w == 0) {
                                                alu_ri(&F, 5 /*sub*/, RAX, (int32_t)min_c, 0);
                                            } else if (min_c >= INT32_MIN && min_c <= INT32_MAX) {
                                                alu_ri(&F, 5 /*sub*/, RAX, (int32_t)min_c, 1);
                                            } else {
                                                mov_ri(&F, RCX, min_c, 1);
                                                alu_rr(&F, 0x29 /*sub*/, RAX, RCX, 1);
                                            }
                                        }
                                        alu_ri(&F, 7 /*cmp*/, RAX, (int32_t)(jt->span - 1), w);
                                        x64_emit1(&F, OP_TYPE_X64_JCC, (int64_t)CC_A | ((int64_t)cur_bi << 32));
                                        x64_emit1(&F, OP_TYPE_X64_JTAB_DISPATCH, (int64_t)jid);
                                        emitted_jtab = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (!emitted_jtab) {
                if (fused_icmp != MLIR_INVALID_HANDLE)
                    emit_fused_cond_br(&F, reg, term, next_bk, fused_icmp, fused_invert);
                else
                    lower_term(&F, reg, term, next_bk);
            }
        }
    }
    free(jbh_k); free(jbh_v); free(pred_cnt); free(skip_blk);

    // --- x64.* MLIR dialect peephole optimization pass (-O1/-O2) ---
    if (F.ok && g_x64_opt_level >= 1)
        x64_optimize_block(ctx, x64_blk);

    // --- encode x64.* MLIR block to x86_64 machine bytes (unless retaining IR) ---
    if (retain_blk == MLIR_INVALID_HANDLE) {
        if (F.ok) x64_encode_block(&F, x64_blk, (size_t)F.x64_next_label);
        MLIR_ResetBlockToCheckpoint(ctx, x64_blk, x64_op_cp);
    }

    if (!F.ok) g_x64_ok = false;
    g_x64_ok = g_x64_ok && F.ok;

    for (size_t j = 0; j < F.n_jtabs; j++) free(F.jtabs[j].targets);
    free(F.jtabs);
    vmap_free(&F.slots);
    if (g_x64_times) g_x64_times->isel_encode_ns += x64_clock() - t_isel;
}

static void emit_func_into(MLIR_Context *ctx, MLIR_OpHandle fn, ObjFunc *f, SymSet *refs) {
    emit_func_to_x64(ctx, fn, f, refs, MLIR_INVALID_HANDLE);
}

static void emit_func(MLIR_Context *ctx, MLIR_OpHandle fn, ObjModule *m, SymSet *refs) {
    string name = attr_str(ctx, fn, MLIR_KEY_SYM_NAME);
    char *nm = (char *)malloc(name.size + 1);
    memcpy(nm, name.str, name.size); nm[name.size] = 0;
    bool exported = str_eqz(name, "main");
    ObjFunc *f = obj_add_func(m, nm, exported);
    emit_func_into(ctx, fn, f, refs);
}

// ---------------------------------------------------------------------------
// Globals, synth start + syscall thunks, reachability, entry point.
// ---------------------------------------------------------------------------

// Wasm global names — must match mlir_wasmssa_to_llvm.c / aarch64 synth_start.
#define X64_WASM_LINMEM_BASE  "__wasm_linmem_base"
#define X64_WASM_LINMEM       "__wasm_linmem"
#define X64_WASM_ARGC         "__wasm_argc"
#define X64_WASM_ARGV         "__wasm_argv"

typedef struct { uint32_t slot_off; uint32_t target_off; string fn_target; } X64PtrReloc;
typedef struct { uint32_t slot_off; string target; int64_t addend; } PendingPtrReloc;

static bool serialize_global_value_attr(MLIR_Context *ctx, MLIR_AttributeHandle va, string name,
                                        Buf *data, uint32_t off, size_t sz) {
    MLIR_AttrKind vk = MLIR_GetAttributeKind(ctx, va);
    if (vk == MLIR_ATTR_KIND_STRING) {
        string s = MLIR_GetAttributeString(ctx, va);
        if (s.size > sz) s.size = sz;
        memcpy(data->data + off, s.str, s.size);
    } else if (vk == MLIR_ATTR_KIND_INTEGER || vk == MLIR_ATTR_KIND_BOOL) {
        uint64_t v = (uint64_t)MLIR_GetAttributeInteger(ctx, va);
        for (size_t b = 0; b < sz; b++)
            data->data[off + b] = (uint8_t)(b < 8 ? (v >> (8 * b)) : 0);
    } else if (vk == MLIR_ATTR_KIND_FLOAT) {
        double d = MLIR_GetAttributeFloat(ctx, va);
        uint64_t bits = 0;
        if (sz == 4) { float f = (float)d; uint32_t b32; memcpy(&b32, &f, 4); bits = b32; }
        else         { memcpy(&bits, &d, 8); }
        for (size_t b = 0; b < sz; b++)
            data->data[off + b] = (uint8_t)(bits >> (8 * b));
    } else {
        fprintf(stderr, "x64: unsupported global initializer kind for '%.*s'\n",
                (int)name.size, name.str);
        g_x64_ok = false;
        return false;
    }
    return true;
}

static bool parse_trivial_ptr_global_init(MLIR_Context *ctx, MLIR_OpHandle op, string name, uint32_t off,
                                          size_t sz, PendingPtrReloc **pend,
                                          size_t *pend_cap, size_t *n_pend) {
    if (MLIR_GetOpNumRegions(ctx, op) == 0 ||
        MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, op, 0)) == 0)
        return true;

    MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, op, 0), 0);
    size_t nb = MLIR_GetBlockNumOps(ctx, blk);
    string target = (string){0};
    int64_t addend = 0;
    bool only_trivial = true;

    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_OpHandle bop = MLIR_GetBlockOp(ctx, blk, bi);
        MLIR_OpType bty = MLIR_GetOpType(ctx, bop);
        if (bty == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
            MLIR_AttributeHandle ga =
                MLIR_GetOpAttr(ctx, bop, MLIR_KEY_GLOBAL_NAME);
            if (ga == MLIR_INVALID_HANDLE)
                ga = MLIR_GetOpAttr(ctx, bop, MLIR_KEY_VALUE);
            if (ga == MLIR_INVALID_HANDLE) {
                fprintf(stderr, "x64: addressof global missing name in '%.*s'\n",
                        (int)name.size, name.str);
                g_x64_ok = false;
                return false;
            }
            target = strip_at(MLIR_GetAttributeString(ctx, ga));
        } else if (bty == OP_TYPE_LLVM_GEP) {
            MLIR_AttributeHandle ia = MLIR_GetOpAttr(ctx, bop, MLIR_KEY_RAW_CONSTANT_INDICES);
            if (ia != MLIR_INVALID_HANDLE) {
                int64_t idx[4];
                size_t ni = llvm_position_indices(ctx, ia, idx, 4);
                if (ni == 1) addend += idx[0];
                else only_trivial = false;
            } else {
                only_trivial = false;
            }
        } else if (bty != OP_TYPE_LLVM_MLIR_ZERO &&
                   bty != OP_TYPE_LLVM_MLIR_UNDEF &&
                   bty != OP_TYPE_LLVM_RETURN) {
            only_trivial = false;
        }
    }

    if (!target.size) {
        if (!only_trivial) {
            fprintf(stderr, "x64: unsupported global initializer for '%.*s'\n",
                    (int)name.size, name.str);
            g_x64_ok = false;
            return false;
        }
        return true;
    }
    if (sz < 8) {
        fprintf(stderr, "x64: pointer global '%.*s' is too small\n",
                (int)name.size, name.str);
        g_x64_ok = false;
        return false;
    }
    if (*n_pend == *pend_cap) {
        *pend_cap = *pend_cap ? *pend_cap * 2 : 4;
        *pend = (PendingPtrReloc *)realloc(*pend, *pend_cap * sizeof(**pend));
    }
    (*pend)[*n_pend].slot_off = off;
    (*pend)[*n_pend].target = target;
    (*pend)[*n_pend].addend = addend;
    (*n_pend)++;
    return true;
}

static bool x64_module_has_func_str(MLIR_Context *ctx, MLIR_BlockHandle mb, string name);

static void resolve_pending_ptr_globals(MLIR_Context *ctx, MLIR_BlockHandle mb,
                                        GMap *gm, PendingPtrReloc *pend, size_t n_pend,
                                        X64PtrReloc **relocs_out, size_t *n_relocs_out) {
    *relocs_out = NULL;
    *n_relocs_out = 0;
    if (!n_pend || !g_x64_ok) return;

    X64PtrReloc *relocs = (X64PtrReloc *)calloc(n_pend, sizeof(X64PtrReloc));
    for (size_t k = 0; k < n_pend; k++) {
        uint32_t base_off = 0;
        if (gmap_get(gm, pend[k].target, &base_off)) {
            relocs[k].target_off = (uint32_t)((int64_t)base_off + pend[k].addend);
            relocs[k].slot_off = pend[k].slot_off;
        } else if (pend[k].addend == 0 && x64_module_has_func_str(ctx, mb, pend[k].target)) {
            relocs[k].fn_target = pend[k].target;
            relocs[k].slot_off = pend[k].slot_off;
        } else {
            fprintf(stderr, "x64: pointer global target '%.*s' not found\n",
                    (int)pend[k].target.size, pend[k].target.str);
            g_x64_ok = false;
            break;
        }
    }
    if (g_x64_ok) {
        *relocs_out = relocs;
        *n_relocs_out = n_pend;
    } else {
        free(relocs);
    }
}

// True if `op` is llvm.mlir.global (in-tree or upstream GlobalOp form).
static bool x64_is_global_op(MLIR_Context *ctx, MLIR_OpHandle op) {
    if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_MLIR_GLOBAL) return true;
    return MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_MLIR_GLOBAL;
}

// Bare global name from sym_name (upstream stores `@foo`; strip the `@`).
static string x64_global_sym_name(MLIR_Context *ctx, MLIR_OpHandle op) {
    return strip_at(attr_str(ctx, op, MLIR_KEY_SYM_NAME));
}

// True if `op` is llvm.func (in-tree or upstream FuncOp form).
static bool x64_is_func_op(MLIR_Context *ctx, MLIR_OpHandle op) {
    if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_FUNC) return true;
    return MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_FUNC;
}

// True if the module has a function named `name` (defined when require_body).
static bool x64_module_has_func_str(MLIR_Context *ctx, MLIR_BlockHandle mb, string name) {
    size_t no = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (!x64_is_func_op(ctx, op)) continue;
        if (str_eq(strip_at(attr_str(ctx, op, MLIR_KEY_SYM_NAME)), name)) return true;
    }
    return false;
}

static bool x64_module_has_func(MLIR_Context *ctx, MLIR_BlockHandle mb, const char *name, bool require_body) {
    size_t no = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (!x64_is_func_op(ctx, op)) continue;
        if (!str_eqz(attr_str(ctx, op, MLIR_KEY_SYM_NAME), name)) continue;
        if (!require_body) return true;
        if (MLIR_GetOpNumRegions(ctx, op) == 0) return false;
        return MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, op, 0)) > 0;
    }
    return false;
}

// True if the module declares a global named `gn`.
static bool x64_module_has_global_sym(MLIR_Context *ctx, MLIR_BlockHandle mb, string gn) {
    if (!gn.size) return false;
    size_t no = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (!x64_is_global_op(ctx, op)) continue;
        if (str_eq(x64_global_sym_name(ctx, op), gn)) return true;
    }
    return false;
}

typedef struct {
    uint32_t base_off;
    PendingPtrReloc **pend;
    size_t *pend_cap;
    size_t *n_pend;
} X64RelocCtx;

static void x64_on_global_reloc(void *ud, unsigned rel_off, string target) {
    X64RelocCtx *rc = (X64RelocCtx *)ud;
    if (*rc->n_pend == *rc->pend_cap) {
        *rc->pend_cap = *rc->pend_cap ? *rc->pend_cap * 2 : 4;
        *rc->pend = (PendingPtrReloc *)realloc(*rc->pend, *rc->pend_cap * sizeof(**rc->pend));
    }
    (*rc->pend)[*rc->n_pend].slot_off = rc->base_off + rel_off;
    (*rc->pend)[*rc->n_pend].target = target;
    (*rc->pend)[*rc->n_pend].addend = 0;
    (*rc->n_pend)++;
}

// Allocate one global in .data, serialize its initializer, and register it in gm.
static void x64_register_global_op(MLIR_Context *ctx, MLIR_OpHandle op, GMap *gm, Buf *data,
                                   PendingPtrReloc **pend, size_t *pend_cap, size_t *n_pend) {
    string name = x64_global_sym_name(ctx, op);
    if (!name.size) {
        fprintf(stderr, "x64: llvm.mlir.global missing sym_name\n");
        g_x64_ok = false;
        return;
    }
    uint32_t dummy;
    if (gmap_get(gm, name, &dummy)) return;

    MLIR_AttributeHandle gt = MLIR_GetOpAttr(ctx, op, MLIR_KEY_GLOBAL_TYPE);
    MLIR_TypeHandle t = gt ? MLIR_GetAttributeTypeValue(ctx, gt) : MLIR_INVALID_HANDLE;
    size_t sz = t != MLIR_INVALID_HANDLE ? x64_type_size(ctx, t) : 0;
    // An empty struct (GNU C) still gets one byte, as an object of its own.
    if (sz == 0 && t != MLIR_INVALID_HANDLE && llvm_type_is_empty(ctx, t, 8)) sz = 1;
    uint32_t off = x64_data_begin_global(data, sz);

    MLIR_AttributeHandle va = MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE);
    if (va) serialize_global_value_attr(ctx, va, name, data, off, sz);
    else {
        size_t save_pend = *n_pend;
        X64RelocCtx rc = { off, pend, pend_cap, n_pend };
        if (t == MLIR_INVALID_HANDLE ||
            !llvm_global_init_bytes_relocs(ctx, op, t, 8, data->data + off, (unsigned)sz,
                                           x64_on_global_reloc, &rc)) {
            *n_pend = save_pend;
            parse_trivial_ptr_global_init(ctx, op, name, off, sz, pend, pend_cap, n_pend);
        }
    }

    x64_data_finish_global(gm, data, name, off, sz);
}

// Reserve a zero-filled wasm scalar slot in .data if not already in gm.
static void x64_ensure_wasm_scalar(GMap *gm, Buf *data, const char *name, size_t sz) {
    uint32_t dummy;
    if (gmap_get_cstr(gm, name, &dummy, NULL)) return;
    x64_data_alloc_global(gm, data, str_from_cstr_view((char *)name), sz);
}

// Ensure __wasm_linmem_base / __wasm_argc / __wasm_argv slots exist in .data.
static void x64_ensure_wasm_scalars(GMap *gm, Buf *data) {
    x64_ensure_wasm_scalar(gm, data, X64_WASM_LINMEM_BASE, 8);
    x64_ensure_wasm_scalar(gm, data, X64_WASM_ARGC, 4);
    x64_ensure_wasm_scalar(gm, data, X64_WASM_ARGV, 8);
}

// True if an `llvm.mlir.global` is purely zero-initialized (no `value` attr,
// and either no initializer region or a trivial `{ llvm.mlir.zero; llvm.return }`
// region), so it can live in `.bss` (p_memsz > p_filesz).
static bool x64_global_is_zero_bss(MLIR_Context *ctx, MLIR_OpHandle op) {
    if (MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE) != MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumRegions(ctx, op) == 0) return true;
    MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, op, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, r);
    if (nb == 0) return true;
    if (nb != 1) return false;
    MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, r, 0);
    size_t no = MLIR_GetBlockNumOps(ctx, bk);
    if (no == 0) return true;
    if (no == 2 &&
        MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, bk, 0)) == OP_TYPE_LLVM_MLIR_ZERO &&
        MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, bk, 1)) == OP_TYPE_LLVM_RETURN)
        return true;
    return false;
}

// Lay out .data (+ optional .bss tail when `bss_len_out != NULL`):
// initialized globals first in `data`, then zero-initialized globals in `.bss`
// without allocating/zeroing/writing their bytes.
static void x64_init_data_section(MLIR_Context *ctx, MLIR_BlockHandle mb, GMap *gm,
                                  Buf *data, size_t *bss_len_out, bool wasm_lifted,
                                  X64PtrReloc **relocs_out, size_t *n_relocs_out) {
    *relocs_out = NULL;
    *n_relocs_out = 0;
    if (bss_len_out) *bss_len_out = 0;
    size_t no = MLIR_GetBlockNumOps(ctx, mb);
    size_t pend_cap = 0, n_pend = 0;
    PendingPtrReloc *pend = NULL;

    MLIR_OpHandle bss_stk[128];
    MLIR_OpHandle *bss_ops = bss_stk;
    size_t n_bss = 0, c_bss = 128;

    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (!x64_is_global_op(ctx, op)) continue;
        if (bss_len_out && !wasm_lifted && x64_global_is_zero_bss(ctx, op)) {
            if (n_bss == c_bss) {
                size_t nc = c_bss * 2;
                MLIR_OpHandle *nb = (MLIR_OpHandle *)malloc(nc * sizeof(MLIR_OpHandle));
                memcpy(nb, bss_ops, n_bss * sizeof(MLIR_OpHandle));
                if (bss_ops != bss_stk) free(bss_ops);
                bss_ops = nb;
                c_bss = nc;
            }
            bss_ops[n_bss++] = op;
            continue;
        }
        x64_register_global_op(ctx, op, gm, data, &pend, &pend_cap, &n_pend);
    }
    if (wasm_lifted)
        x64_ensure_wasm_scalars(gm, data);

    if (bss_len_out && n_bss > 0) {
        size_t base = data->len;
        size_t cur = base;
        for (size_t i = 0; i < n_bss; i++) {
            MLIR_OpHandle op = bss_ops[i];
            string name = x64_global_sym_name(ctx, op);
            if (!name.size) continue;
            uint32_t dummy;
            if (gmap_get(gm, name, &dummy)) continue;
            MLIR_AttributeHandle gt = MLIR_GetOpAttr(ctx, op, MLIR_KEY_GLOBAL_TYPE);
            MLIR_TypeHandle t = gt ? MLIR_GetAttributeTypeValue(ctx, gt) : MLIR_INVALID_HANDLE;
            size_t sz = t != MLIR_INVALID_HANDLE ? x64_type_size(ctx, t) : 0;
            if (sz == 0 && t != MLIR_INVALID_HANDLE && llvm_type_is_empty(ctx, t, 8)) sz = 1;
            cur = (cur + 7u) & ~(size_t)7u;
            gmap_put(gm, name, (uint32_t)cur, (uint32_t)sz);
            cur += sz;
        }
        *bss_len_out = cur - base;
    }
    if (bss_ops != bss_stk) free(bss_ops);

    resolve_pending_ptr_globals(ctx, mb, gm, pend, n_pend, relocs_out, n_relocs_out);
    free(pend);
}

// Zero-fill a wasm data global when DCE dropped the IR op but addressof remains.
static bool x64_synth_wasm_global(FnCtx *F, string gn, uint32_t *out_off) {
    if (!F->wasm_lifted || !x64_is_wasm_data_sym(gn) || !F->data_buf) return false;
    size_t sz = 4;
    if (gn.size == 18 && memcmp(gn.str, "__wasm_linmem_base", 18) == 0) sz = 8;
    else if (gn.size == 11 && memcmp(gn.str, "__wasm_argv", 11) == 0) sz = 8;
    *out_off = x64_data_alloc_global(F->gm, F->data_buf, gn, sz);
    return true;
}

// Late-register a module global into the data blob (used when collect missed it).
static bool x64_ensure_global_in_gmap(FnCtx *F, string gn, uint32_t *out_off) {
    if (gmap_get(F->gm, gn, out_off)) return true;
    if (F->module_mb == MLIR_INVALID_HANDLE || !F->data_buf) return false;
    size_t no = MLIR_GetBlockNumOps(F->ctx, F->module_mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(F->ctx, F->module_mb, i);
        if (!x64_is_global_op(F->ctx, op)) continue;
        if (!str_eq(x64_global_sym_name(F->ctx, op), gn)) continue;
        size_t pend_cap = 0, n_pend = 0;
        PendingPtrReloc *pend = NULL;
        x64_register_global_op(F->ctx, op, F->gm, F->data_buf,
                               &pend, &pend_cap, &n_pend);
        if (n_pend > 0) {
            fprintf(stderr,
                    "x64: late global '%.*s' needs runtime init (re-run x64_init_data_section)\n",
                    (int)gn.size, gn.str);
            g_x64_ok = false;
        }
        free(pend);
        return gmap_get(F->gm, gn, out_off);
    }
    return x64_synth_wasm_global(F, gn, out_off);
}

// Pick the ELF user entry symbol. A native module always enters at `main`
// (the C entry point; any other function, including one that happens to be
// called app_main, is ordinary user code). Only a module lifted from wasm uses
// the corec entry points: app_main (skipping the wasm crt0, whose job the
// synthesised _start does natively), else the lifted wasm entry.
static bool x64_resolve_entry(MLIR_Context *ctx, MLIR_BlockHandle mb, bool wasm_lifted, char **out_cstr) {
    static const char *lifted_order[] = { "app_main", "main", "wasi_start" };
    static const char *native_order[] = { "main" };
    const char **order = wasm_lifted ? lifted_order : native_order;
    size_t n = wasm_lifted ? sizeof(lifted_order)/sizeof(lifted_order[0])
                           : sizeof(native_order)/sizeof(native_order[0]);
    for (size_t i = 0; i < n; i++) {
        if (x64_module_has_func(ctx, mb, order[i], false)) {
            *out_cstr = heapdup_cstr(order[i]);
            return true;
        }
    }
    return false;
}

// Synthesize _start: wasm linmem/argc setup, ptr inits, init+main, exit syscall.
static void synth_start(ObjModule *m, GMap *gm, Buf *data,
                        X64PtrReloc *relocs, size_t n_relocs,
                        bool wasm_lifted, const char *entry_sym) {
    ObjFunc *f = obj_add_func(m, heapdup_cstr("_start"), true);
    FnCtx F; memset(&F,0,sizeof(F)); F.f = f; F.ok = true;

    uint32_t argc_off = 0, argv_off = 0, linmem_base_off = 0;
    uint32_t template_off = 0, template_size = 0;
    // The __wasm_* slots are lifter-owned only in a lifted module; a native
    // program may use those names for its own globals.
    bool has_argc = wasm_lifted &&
                    gmap_get_cstr(gm, X64_WASM_ARGC, &argc_off, NULL) &&
                    gmap_get_cstr(gm, X64_WASM_ARGV, &argv_off, NULL);
    bool has_linmem_base = wasm_lifted &&
        gmap_get_cstr(gm, X64_WASM_LINMEM_BASE, &linmem_base_off, NULL);
    bool has_template = wasm_lifted &&
        gmap_get_cstr(gm, X64_WASM_LINMEM, &template_off, &template_size);

    if (has_argc) {
        rex(&F, 0, RAX, 0, RSP); eb(&F, 0x8B); modrm(&F, 0, RAX, RSP); eb(&F, 0x24);
        lea_rip_data(&F, RCX, argc_off);
        store_mem(&F, RCX, RAX, 4);
        rex(&F, 1, RSI, 0, RSP); eb(&F, 0x8D); modrm(&F, 1, RSI, RSP);
        eb(&F, 0x24); eb(&F, 0x08);
        lea_rip_data(&F, RCX, argv_off);
        store_mem(&F, RCX, RSI, 8);
    }

    if (has_linmem_base) {
        // Diagnostic for a failed reservation, kept in .data. The name cannot
        // collide with a C identifier.
        const char *oom_msg =
            "tinyc runtime: cannot reserve 4 GiB of address space for the wasm "
            "linear memory (mmap failed; check `ulimit -v` / overcommit limits)\n";
        size_t oom_len = strlen(oom_msg);
        uint32_t oom_off = x64_data_begin_global(data, oom_len);
        memcpy(data->data + oom_off, oom_msg, oom_len);
        x64_data_finish_global(gm, data, str_from_cstr_view((char *)".x64.linmem_oom_msg"),
                               oom_off, oom_len);

        // rax = mmap(NULL, 4 GiB, PROT_READ|PROT_WRITE,
        //            MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
        // MAP_NORESERVE: this is an address-space reservation; pages are only
        // committed when the program touches them.
        mov_ri(&F, RDI, 0, 1);
        mov_ri(&F, RSI, (int64_t)X64_WASM_LINMEM_RESERVE, 1);
        mov_ri(&F, RDX, X64_PROT_READ_WRITE, 1);
        mov_ri(&F, R10, X64_MAP_LINMEM_FLAGS, 1);
        mov_ri(&F, R8, -1, 1);
        mov_ri(&F, R9, 0, 1);
        mov_ri(&F, RAX, NR_MMAP, 0);
        eb(&F, 0x0F); eb(&F, 0x05);                                 // syscall
        // A raw syscall reports failure as -errno, i.e. an unsigned value in
        // (-4096, -1]. Without this check the template copy below would
        // `rep movsb` into that bogus address and die with a bare SIGSEGV.
        eb(&F, 0x48); eb(&F, 0x3D); e32(&F, 0xFFFFF000u);           // cmp rax, -4096
        eb(&F, 0x76); uint32_t ok_rel8 = here(&F); eb(&F, 0);       // jbe .mapped
        mov_ri(&F, RDI, 2, 0);                                      //   write(2, msg, len)
        lea_rip_data(&F, RSI, oom_off);
        mov_ri(&F, RDX, (int64_t)oom_len, 0);
        mov_ri(&F, RAX, NR_WRITE, 0);
        eb(&F, 0x0F); eb(&F, 0x05);
        mov_ri(&F, RDI, 1, 0);                                      //   exit(1)
        mov_ri(&F, RAX, NR_EXIT, 0);
        eb(&F, 0x0F); eb(&F, 0x05);
        f->code.data[ok_rel8] = (uint8_t)(here(&F) - (ok_rel8 + 1)); // .mapped:
        mov_rr(&F, RBX, RAX, 1);
        lea_rip_data(&F, RCX, linmem_base_off);
        store_mem(&F, RCX, RBX, 8);
        if (has_template && template_size > 0) {
            mov_rr(&F, RDI, RBX, 1);
            lea_rip_data(&F, RSI, template_off);
            mov_ri(&F, RCX, (int64_t)template_size, 1);
            eb(&F, 0xFC);
            eb(&F, 0xF3); eb(&F, 0xA4);
        }
    }

    // One-time startup: fill pointer globals (addressof @sym; return initializers).
    for (size_t k = 0; k < n_relocs; k++) {
        if (relocs[k].fn_target.size > 0)
            lea_rip_func(&F, R10, relocs[k].fn_target);
        else
            lea_rip_data(&F, R10, relocs[k].target_off);
        lea_rip_data(&F, R11, relocs[k].slot_off);
        store_mem(&F, R11, R10, 8);
    }

    if (wasm_lifted) {
        mov_ri(&F, RDI, 0, 1);
        mov_ri(&F, RSI, 0, 1);
        mov_ri(&F, RDX, 0, 1);
        eb(&F, 0xE8); uint32_t fld_init = here(&F); e32(&F, 0);
        obj_add_reloc(f, fld_init, fld_init+4, false, heapdup_cstr("__host_platform_init"), 0);
        eb(&F, 0xE8); uint32_t fld_main = here(&F); e32(&F, 0);
        obj_add_reloc(f, fld_main, fld_main+4, false, heapdup_cstr(entry_sym), 0);
    } else {
        rex(&F, 1, 0, 0, RBP); eb(&F, 0x31); modrm(&F, 3, RBP, RBP);
        rex(&F, 1, RDI, 0, RSP); eb(&F, 0x8B); modrm(&F, 0, RDI, RSP); eb(&F, 0x24);
        rex(&F, 1, RSI, 0, RSP); eb(&F, 0x8D); modrm(&F, 1, RSI, RSP);
        eb(&F, 0x24); eb(&F, 0x08);
        rex(&F, 1, RDX, RDI, RSI); eb(&F, 0x8D); modrm(&F, 1, RDX, 4);
        eb(&F, (uint8_t)((3 << 6) | ((RDI & 7) << 3) | (RSI & 7))); eb(&F, 0x08);
        rex(&F, 1, 0, 0, RSP); eb(&F, 0x83); modrm(&F, 3, 4, RSP); eb(&F, 0xF0);
        // Preserve argc/argv/envp in callee-saved registers across platform_init.
        mov_rr(&F, RBX, RDI, 1);
        mov_rr(&F, R12, RSI, 1);
        mov_rr(&F, R13, RDX, 1);
        eb(&F, 0xE8); uint32_t fld_init = here(&F); e32(&F, 0);
        obj_add_reloc(f, fld_init, fld_init+4, false, heapdup_cstr("platform_init"), 0);
        mov_rr(&F, RDI, RBX, 1);
        mov_rr(&F, RSI, R12, 1);
        mov_rr(&F, RDX, R13, 1);
        eb(&F, 0xE8); uint32_t fld_main = here(&F); e32(&F, 0);
        obj_add_reloc(f, fld_main, fld_main+4, false, heapdup_cstr(entry_sym), 0);
    }
    mov_rr(&F, RDI, RAX, 0);
    mov_ri(&F, RAX, NR_EXIT, 0);
    eb(&F, 0x0F); eb(&F, 0x05);
}

// Synthesize compiler-builtin thunks (__tinyc_setjmp, __tinyc_longjmp,
// __tinyc_syscall6 / syscall). All corec/WASI forwarding stubs (platform_init,
// __assert_fail) live in corec/wasm/wasi_adapter.c so the x86_64 backend stays
// agnostic to corec symbol names.
static bool synth_thunk(ObjModule *m, string name) {
    if (str_eqz(name, "__tinyc_setjmp")) {
        char *fn = heapdup_cstr("__tinyc_setjmp");
        ObjFunc *f = obj_add_func(m, fn, false);
        FnCtx F; memset(&F, 0, sizeof(F)); F.f = f;
        // SysV x86_64: rdi = jmp_buf (8 x i64): rbx, rbp, r12..r15, rsp, rip.
        static const uint8_t code[] = {
            0x48, 0x89, 0x1F,             // mov [rdi+0],  rbx
            0x48, 0x89, 0x6F, 0x08,       // mov [rdi+8],  rbp
            0x4C, 0x89, 0x67, 0x10,       // mov [rdi+16], r12
            0x4C, 0x89, 0x6F, 0x18,       // mov [rdi+24], r13
            0x4C, 0x89, 0x77, 0x20,       // mov [rdi+32], r14
            0x4C, 0x89, 0x7F, 0x28,       // mov [rdi+40], r15
            0x48, 0x8D, 0x44, 0x24, 0x08, // lea rax, [rsp+8]
            0x48, 0x89, 0x47, 0x30,       // mov [rdi+48], rax
            0x48, 0x8B, 0x04, 0x24,       // mov rax, [rsp]
            0x48, 0x89, 0x47, 0x38,       // mov [rdi+56], rax
            0x31, 0xC0,                   // xor eax, eax
            0xC3,                         // ret
        };
        for (size_t i = 0; i < sizeof(code); i++) eb(&F, code[i]);
        return true;
    }
    if (str_eqz(name, "__tinyc_longjmp")) {
        char *fn = heapdup_cstr("__tinyc_longjmp");
        ObjFunc *f = obj_add_func(m, fn, false);
        FnCtx F; memset(&F, 0, sizeof(F)); F.f = f;
        // SysV x86_64: rdi = jmp_buf, esi = val (0 -> 1 per C11 7.13.2.1).
        static const uint8_t code[] = {
            0x89, 0xF0,                   // mov eax, esi
            0x85, 0xC0,                   // test eax, eax
            0x75, 0x02,                   // jnz +2
            0xFF, 0xC0,                   // inc eax
            0x48, 0x8B, 0x1F,             // mov rbx, [rdi+0]
            0x48, 0x8B, 0x6F, 0x08,       // mov rbp, [rdi+8]
            0x4C, 0x8B, 0x67, 0x10,       // mov r12, [rdi+16]
            0x4C, 0x8B, 0x6F, 0x18,       // mov r13, [rdi+24]
            0x4C, 0x8B, 0x77, 0x20,       // mov r14, [rdi+32]
            0x4C, 0x8B, 0x7F, 0x28,       // mov r15, [rdi+40]
            0x48, 0x8B, 0x67, 0x30,       // mov rsp, [rdi+48]
            0xFF, 0x67, 0x38,             // jmp qword ptr [rdi+56]
        };
        for (size_t i = 0; i < sizeof(code); i++) eb(&F, code[i]);
        return true;
    }
    // __tinyc_syscall6(num, a1..a6): the runtime syscall number is the first C
    // argument, so we cannot bake an NR_* constant — instead reshuffle the SysV
    // argument registers into the Linux x86_64 kernel convention (number in
    // rax; args in rdi/rsi/rdx/r10/r8/r9) and trap. a6 arrives on the stack.
    // Lifted wasm may also reference the helper as `syscall` (platform_linux.c).
    if (!str_eqz(name, "__tinyc_syscall6") && !str_eqz(name, "syscall")) return false;
    {
        char *fn = (char *)malloc(name.size + 1);
        memcpy(fn, name.str, name.size);
        fn[name.size] = '\0';
        ObjFunc *f = obj_add_func(m, fn, false);
        FnCtx F; memset(&F,0,sizeof(F)); F.f = f;
        mov_rr(&F, RAX, RDI, 1);   // rax = num
        mov_rr(&F, RDI, RSI, 1);   // arg1
        mov_rr(&F, RSI, RDX, 1);   // arg2
        mov_rr(&F, RDX, RCX, 1);   // arg3
        mov_rr(&F, R10, R8, 1);    // arg4 (kernel uses r10, not rcx)
        mov_rr(&F, R8, R9, 1);     // arg5
        // arg6: mov r9, [rsp+8]  (just above the call return address)
        rex(&F, 1, R9, 0, RSP); eb(&F, 0x8B); modrm(&F, 1, R9, RSP);
        eb(&F, 0x24); eb(&F, 0x08);
        eb(&F, 0x0F); eb(&F, 0x05);                                        // syscall
        eb(&F, 0xC3);                                                      // ret
        return true;
    }
}

// True if the emitted ObjModule already defines a text symbol `name`.
static bool x64_obj_has_func(ObjModule *m, const char *name) {
    for (size_t j = 0; j < m->n_funcs; j++)
        if (strcmp(m->funcs[j].name, name) == 0) return true;
    return false;
}

// True if some emitted reloc references `name` as a text-section callee.
static bool x64_obj_needs_func(ObjModule *m, const char *name) {
    for (size_t i = 0; i < m->n_funcs; i++) {
        ObjFunc *f = &m->funcs[i];
        for (size_t r = 0; r < f->n_relocs; r++) {
            ObjReloc *rl = &f->relocs[r];
            if (!rl->to_data && rl->sym && strcmp(rl->sym, name) == 0)
                return true;
        }
    }
    return false;
}

bool mlir_llvm_to_elf(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                      bool wasm_lifted,
                      uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;
    g_x64_ok = true;
    g_x64_wasm_lifted = wasm_lifted;

    MLIR_RegionHandle mreg = MLIR_GetOpRegion(ctx, llvm_module, 0);
    MLIR_BlockHandle  mb   = MLIR_GetRegionBlock(ctx, mreg, 0);
    g_x64_mb = mb;

    // globals — keep g_x64_gmap live through emission, thunks, and ELF link.
    GMap gm; memset(&gm, 0, sizeof(gm));
    Buf data = {0};
    g_x64_data_buf = &data;
    X64PtrReloc *ptr_relocs = NULL;
    size_t n_ptr_relocs = 0;
    char *entry_cstr = NULL;
    ObjModule m; memset(&m, 0, sizeof(m));
    SymSet refs; memset(&refs, 0, sizeof(refs));
    bool ok = false;

    uint64_t t_setup = x64_clock();
    size_t bss_len = 0;
    x64_init_data_section(ctx, mb, &gm, &data, &bss_len, wasm_lifted,
                          &ptr_relocs, &n_ptr_relocs);
    g_x64_gmap = &gm;

    if (!x64_resolve_entry(ctx, mb, wasm_lifted, &entry_cstr)) {
        fprintf(stderr, wasm_lifted
                ? "x64: no defined entry (app_main/main/wasi_start)\n"
                : "x64: no defined main\n");
        g_x64_ok = false;
        goto x64_done;
    }

    // index defined functions by name for O(N + E) BFS reachability + O(1) thunk check.
    size_t no = MLIR_GetBlockNumOps(ctx, mb);

    synth_start(&m, &gm, &data, ptr_relocs, n_ptr_relocs, wasm_lifted, entry_cstr);
    m.start_idx = 0;

    typedef struct { MLIR_OpHandle fn; string name; bool visited; } DefFn;
    DefFn *defs = (DefFn *)malloc((no ? no : 1) * sizeof(DefFn));
    size_t n_defs = 0;
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) continue;
        if (MLIR_GetOpNumRegions(ctx, fn) == 0 ||
            MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, fn, 0)) == 0) continue;
        defs[n_defs].fn = fn;
        defs[n_defs].name = attr_str(ctx, fn, MLIR_KEY_SYM_NAME);
        defs[n_defs].visited = false;
        n_defs++;
    }
    size_t def_hcap = 32;
    while (def_hcap < n_defs * 2u + 1u) def_hcap <<= 1;
    size_t *def_ht = (size_t *)calloc(def_hcap, sizeof(size_t));
    size_t def_hmask = def_hcap - 1;
    for (size_t i = 0; i < n_defs; i++) {
        size_t k = x64_str_hash(defs[i].name.str, defs[i].name.size) & def_hmask;
        while (def_ht[k]) k = (k + 1) & def_hmask;
        def_ht[k] = i + 1;
    }

    size_t *workq = (size_t *)malloc((n_defs ? n_defs : 1) * sizeof(size_t));
    size_t qh = 0, qt = 0;
    {
        string roots[2];
        roots[0] = str_from_cstr_view(wasm_lifted ? (char *)"__host_platform_init"
                                                  : (char *)"platform_init");
        roots[1] = str_from_cstr_view(entry_cstr);
        for (int ri = 0; ri < 2; ri++) {
            size_t k = x64_str_hash(roots[ri].str, roots[ri].size) & def_hmask;
            while (def_ht[k]) {
                size_t idx = def_ht[k] - 1;
                if (str_eq(defs[idx].name, roots[ri])) {
                    if (!defs[idx].visited) {
                        defs[idx].visited = true;
                        workq[qt++] = idx;
                    }
                    break;
                }
                k = (k + 1) & def_hmask;
            }
        }
        for (size_t ri = 0; ri < n_ptr_relocs; ri++) {
            if (!ptr_relocs[ri].fn_target.size) continue;
            string fn_t = ptr_relocs[ri].fn_target;
            size_t k = x64_str_hash(fn_t.str, fn_t.size) & def_hmask;
            while (def_ht[k]) {
                size_t idx = def_ht[k] - 1;
                if (str_eq(defs[idx].name, fn_t)) {
                    if (!defs[idx].visited) {
                        defs[idx].visited = true;
                        workq[qt++] = idx;
                    }
                    break;
                }
                k = (k + 1) & def_hmask;
            }
        }
    }
    while (qh < qt) {
        MLIR_OpHandle fn = defs[workq[qh++]].fn;
        MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, fn, 0);
        size_t nbk = MLIR_GetRegionNumBlocks(ctx, r);
        for (size_t b = 0; b < nbk; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, r, b);
            size_t nop = MLIR_GetBlockNumOps(ctx, bk);
            for (size_t o = 0; o < nop; o++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, o);
                MLIR_OpType ot = MLIR_GetOpType(ctx, op);
                string target = {0};
                if (ot == OP_TYPE_LLVM_CALL || ot == OP_TYPE_FUNC_CALL) {
                    target = strip_at(attr_str(ctx, op, MLIR_KEY_CALLEE));
                    if (!target.size) target = strip_at(attr_str(ctx, op, MLIR_KEY_CALLEE_NAME));
                    if (!target.size) continue;
                    if (wasm_lifted && x64_is_wasm_data_sym(target)) continue;
                } else if (ot == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
                    MLIR_AttributeHandle ga = MLIR_GetOpAttr(ctx, op, MLIR_KEY_GLOBAL_NAME);
                    if (ga == MLIR_INVALID_HANDLE) ga = MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE);
                    target = strip_at(ga ? MLIR_GetAttributeString(ctx, ga) : (string){0});
                    if (!target.size) continue;
                    uint32_t dummy;
                    if (wasm_lifted && x64_is_wasm_data_sym(target)) continue;
                    if (gmap_get(&gm, target, &dummy)) continue;
                } else {
                    continue;
                }
                size_t k = x64_str_hash(target.str, target.size) & def_hmask;
                while (def_ht[k]) {
                    size_t idx = def_ht[k] - 1;
                    if (str_eq(defs[idx].name, target)) {
                        if (!defs[idx].visited) {
                            defs[idx].visited = true;
                            workq[qt++] = idx;
                        }
                        break;
                    }
                    k = (k + 1) & def_hmask;
                }
            }
        }
    }

    if (g_x64_times) g_x64_times->setup_ns += x64_clock() - t_setup;

    // emit reachable defined functions in module order
    bool saw_entry = false;
    for (size_t i = 0; i < n_defs; i++) {
        if (!defs[i].visited) continue;
        if (str_eqz(defs[i].name, entry_cstr)) saw_entry = true;
        emit_func(ctx, defs[i].fn, &m, &refs);
    }

    if (!saw_entry) {
        fprintf(stderr, "x64: no defined entry '%s'\n", entry_cstr);
        g_x64_ok = false;
    }
    uint64_t t_link = x64_clock();

    // synth thunks for referenced-but-undefined symbols
    for (size_t i = 0; i < refs.c && g_x64_ok; i++) {
        bool defined = false;
        size_t k = x64_str_hash(refs.n[i].str, refs.n[i].size) & def_hmask;
        while (def_ht[k]) {
            size_t idx = def_ht[k] - 1;
            if (defs[idx].visited && str_eq(defs[idx].name, refs.n[i])) {
                defined = true; break;
            }
            k = (k + 1) & def_hmask;
        }
        if (defined) continue;
        if (!synth_thunk(&m, refs.n[i])) {
            fprintf(stderr, "x64: unresolved symbol '%.*s'\n",
                    (int)refs.n[i].size, refs.n[i].str);
            g_x64_ok = false;
        }
    }
    free(workq);
    free(def_ht);
    free(defs);

    ok = g_x64_ok;
    if (ok) {
        m.data = data.data; m.data_len = data.len; m.bss_len = bss_len;
        ok = mlir_obj_to_elf(&m, out_data, out_size);
        m.data = NULL;  // owned by `data`
    }
    if (g_x64_times) g_x64_times->link_ns += x64_clock() - t_link;

    // Free module state only after ELF link (relocs may still reference gm/data).
x64_done:
    obj_module_free(&m);
    free(data.data);
    free(entry_cstr);
    gmap_free(&gm);
    sym_free(&refs);
    free(ptr_relocs);
    g_x64_gmap = NULL;
    g_x64_mb = MLIR_INVALID_HANDLE;
    g_x64_data_buf = NULL;
    g_x64_wasm_lifted = false;
    return ok;
}

bool mlir_llvm_to_x64_code(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                           uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;
    g_x64_ok = true;
    g_x64_wasm_lifted = false;

    MLIR_RegionHandle mreg = MLIR_GetOpRegion(ctx, llvm_module, 0);
    MLIR_BlockHandle  mb   = MLIR_GetRegionBlock(ctx, mreg, 0);
    g_x64_mb = mb;

    GMap gm; memset(&gm, 0, sizeof(gm));
    Buf data = {0};
    g_x64_data_buf = &data;
    X64PtrReloc *ptr_relocs = NULL;
    size_t n_ptr_relocs = 0;
    ObjFunc flat_f; memset(&flat_f, 0, sizeof(flat_f));

    uint64_t t_setup = x64_clock();
    x64_init_data_section(ctx, mb, &gm, &data, NULL, false, &ptr_relocs, &n_ptr_relocs);
    g_x64_gmap = &gm;
    if (g_x64_times) g_x64_times->setup_ns += x64_clock() - t_setup;

    // Every defined function: a translation unit has no entry to trim from.
    // Emit all functions directly into a single flat code buffer and wrap
    // into an ELF64 relocatable object (.o, ET_REL) with .text/.data/.rela.text/.symtab.
    size_t no = MLIR_GetBlockNumOps(ctx, mb);
    ObjFuncSym *fsyms = (ObjFuncSym *)malloc((no ? no : 1) * sizeof(ObjFuncSym));
    size_t n_fsyms = 0;
    buf_grow(&flat_f.code, 16384);
    g_x64_borrow_reloc_syms = true;
    for (size_t i = 0; i < no && g_x64_ok; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) continue;
        if (MLIR_GetOpNumRegions(ctx, fn) == 0 ||
            MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, fn, 0)) == 0) continue;
        string fname = attr_str(ctx, fn, MLIR_KEY_SYM_NAME);
        uint32_t foff = (uint32_t)flat_f.code.len;
        emit_func_into(ctx, fn, &flat_f, NULL);
        fsyms[n_fsyms].name = fname.str;
        fsyms[n_fsyms].name_len = fname.size;
        fsyms[n_fsyms].off = foff;
        fsyms[n_fsyms].size = (uint32_t)flat_f.code.len - foff;
        n_fsyms++;
    }
    g_x64_borrow_reloc_syms = false;

    bool ok = g_x64_ok;
    if (ok) {
        uint64_t t_elf = x64_clock();
        ok = mlir_obj_to_elf_reloc(flat_f.code.data, flat_f.code.len,
                                   data.data, data.len,
                                   fsyms, n_fsyms,
                                   flat_f.relocs, flat_f.n_relocs,
                                   out_data, out_size);
        if (g_x64_times) g_x64_times->link_ns += x64_clock() - t_elf;
    }
    free(fsyms);
    free(flat_f.code.data);
    free(flat_f.relocs);
    free(data.data);
    gmap_free(&gm);
    free(ptr_relocs);
    g_x64_gmap = NULL;
    g_x64_mb = MLIR_INVALID_HANDLE;
    g_x64_data_buf = NULL;
    return ok;
}

MLIR_OpHandle mlir_llvm_to_x64_mlir(MLIR_Context *ctx, MLIR_OpHandle llvm_module) {
    g_x64_ok = true;
    g_x64_wasm_lifted = false;

    MLIR_RegionHandle mreg = MLIR_GetOpRegion(ctx, llvm_module, 0);
    MLIR_BlockHandle  mb   = MLIR_GetRegionBlock(ctx, mreg, 0);
    g_x64_mb = mb;

    GMap gm; memset(&gm, 0, sizeof(gm));
    Buf data = {0};
    g_x64_data_buf = &data;
    X64PtrReloc *ptr_relocs = NULL;
    size_t n_ptr_relocs = 0;
    x64_init_data_section(ctx, mb, &gm, &data, NULL, false, &ptr_relocs, &n_ptr_relocs);
    g_x64_gmap = &gm;

    MLIR_RegionHandle x64_mreg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle x64_mb = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, x64_mreg, x64_mb);
    MLIR_OpSpec mspec = {0};
    mspec.type = OP_TYPE_MODULE;
    mspec.regions = &x64_mreg;
    mspec.n_regions = 1;
    MLIR_OpHandle x64_mod = MLIR_CreateOpFromSpec(ctx, &mspec);
    MLIR_ReserveOps(ctx, (size_t)MLIR_GetOpCheckpoint(ctx) * 3 + 256);

    size_t no = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < no && g_x64_ok; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) continue;
        if (MLIR_GetOpNumRegions(ctx, fn) == 0 ||
            MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, fn, 0)) == 0) continue;
        string fname = attr_str(ctx, fn, MLIR_KEY_SYM_NAME);
        MLIR_RegionHandle fn_reg = MLIR_CreateRegion(ctx);
        MLIR_BlockHandle fn_blk = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, fn_reg, fn_blk);
        MLIR_NamedAttr attrs[1] = {
            MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, fname))
        };
        MLIR_OpSpec fspec = {0};
        fspec.type = OP_TYPE_X64_FUNC;
        fspec.attrs = attrs;
        fspec.n_attrs = 1;
        fspec.regions = &fn_reg;
        fspec.n_regions = 1;
        MLIR_OpHandle xfn = MLIR_CreateOpFromSpec(ctx, &fspec);
        MLIR_AppendBlockOp(ctx, x64_mb, xfn);
        emit_func_to_x64(ctx, fn, NULL, NULL, fn_blk);
    }

    bool ok = g_x64_ok;
    free(data.data);
    gmap_free(&gm);
    free(ptr_relocs);
    g_x64_gmap = NULL;
    g_x64_mb = MLIR_INVALID_HANDLE;
    g_x64_data_buf = NULL;
    return ok ? x64_mod : MLIR_INVALID_HANDLE;
}
