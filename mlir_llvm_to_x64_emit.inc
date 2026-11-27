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
    rex(F, w, dst, 0, src); eb(F, 0x69); modrm(F, 3, dst, src); e32(F, (uint32_t)imm);
}
// reg += off: folds into `add reg, imm32` when it fits, else materialises the
// 64-bit offset in rcx (a scratch reg) and adds that.
static void add_const_off(FnCtx *F, uint8_t reg, int64_t off) {
    if (off == 0) return;
    if (off >= INT32_MIN && off <= INT32_MAX) {
        rex(F, 1, 0, 0, reg); eb(F, 0x81); modrm(F, 3, 0, reg); e32(F, (uint32_t)(int32_t)off);
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
    eb(F, 0x0F); eb(F, (uint8_t)(0x90 + cc)); modrm(F, 3, 0, RAX);   // setcc al
    eb(F, 0x0F); eb(F, 0xB6); modrm(F, 3, RAX, RAX);                 // movzx eax, al
}

// cmovnz dst, src
static void cmovnz(FnCtx *F, uint8_t dst, uint8_t src, int w) {
    rex(F, w, dst, 0, src); eb(F, 0x0F); eb(F, 0x45); modrm(F, 3, dst, src);
}
static void cmovcc(FnCtx *F, uint8_t dst, uint8_t src, int cc, int w) {
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
    eb(F, 0x66); rex(F, 1, xmm, 0, gpr); eb(F, 0x0F); eb(F, 0x6E); modrm(F, 3, xmm, gpr);
}
// movq r/m64, xmm   (66 REX.W 0F 7E /r): xmm (low 64 bits) -> GP
static void movq_gpr_xmm(FnCtx *F, uint8_t gpr, uint8_t xmm) {
    eb(F, 0x66); rex(F, 1, xmm, 0, gpr); eb(F, 0x0F); eb(F, 0x7E); modrm(F, 3, xmm, gpr);
}
// scalar double op: F2 0F <opc> /r   (xd = xd OP xs)
static void sse_sd(FnCtx *F, uint8_t opc, uint8_t xd, uint8_t xs) {
    eb(F, 0xF2); rex(F, 0, xd, 0, xs); eb(F, 0x0F); eb(F, opc); modrm(F, 3, xd, xs);
}
// scalar single op: F3 0F <opc> /r
static void sse_ss(FnCtx *F, uint8_t opc, uint8_t xd, uint8_t xs) {
    eb(F, 0xF3); rex(F, 0, xd, 0, xs); eb(F, 0x0F); eb(F, opc); modrm(F, 3, xd, xs);
}
// cvtsi2sd/ss xmm, r/m (pfx [REX.W] 0F 2A /r): signed int (GP) -> float (xmm)
static void cvtsi2f(FnCtx *F, uint8_t pfx, uint8_t xmm, uint8_t gpr, int w) {
    eb(F, pfx); rex(F, w, xmm, 0, gpr); eb(F, 0x0F); eb(F, 0x2A); modrm(F, 3, xmm, gpr);
}
// cvttsd2si/ss r, xmm (pfx [REX.W] 0F 2C /r): float (xmm) -> signed int (GP), trunc
static void cvtf2si(FnCtx *F, uint8_t pfx, uint8_t gpr, uint8_t xmm, int w) {
    eb(F, pfx); rex(F, w, gpr, 0, xmm); eb(F, 0x0F); eb(F, 0x2C); modrm(F, 3, gpr, xmm);
}
// cvtss2sd / cvtsd2ss xmm, xmm (pfx 0F 5A /r): float-width convert
static void cvt_f2f(FnCtx *F, uint8_t pfx, uint8_t xd, uint8_t xs) {
    eb(F, pfx); rex(F, 0, xd, 0, xs); eb(F, 0x0F); eb(F, 0x5A); modrm(F, 3, xd, xs);
}
// ucomisd (66 0F 2E) / ucomiss (0F 2E): ordered compare, sets ZF/PF/CF
static void ucomis(FnCtx *F, int is_f32, uint8_t x1, uint8_t x2) {
    if (!is_f32) eb(F, 0x66);
    rex(F, 0, x1, 0, x2); eb(F, 0x0F); eb(F, 0x2E); modrm(F, 3, x1, x2);
}
// setcc into the low byte of AL/CL (rm in {RAX,RCX})
static void setcc8(FnCtx *F, int cc, uint8_t rm) {
    eb(F, 0x0F); eb(F, (uint8_t)(0x90 + cc)); modrm(F, 3, 0, rm);
}

// setcc opcode low-nibble values not in the shared CC enum
enum { CC_P = 0xA, CC_NP = 0xB };

// llvm.fcmp: operands' bit patterns are already in xmm0 (a0) and xmm1 (a1);
// leave a 0/1 i32 result in EAX. Handles all 16 LLVM FCmp predicates, mapping
// each to ucomisd/ucomiss (with operand swap where needed) plus a one- or
// two-setcc sequence so ordered/unordered (NaN) semantics are correct.
static void emit_fcmp(FnCtx *F, int64_t pred, int is_f32) {
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
    ucomis(F, is_f32, swap ? 1 : 0, swap ? 0 : 1);
    setcc8(F, cc1, RAX);
    if (combine) {
        setcc8(F, cc2, RCX);
        eb(F, combine == 1 ? 0x20 : 0x08); modrm(F, 3, RCX, RAX);  // and/or al, cl
    }
    eb(F, 0x0F); eb(F, 0xB6); modrm(F, 3, RAX, RAX);               // movzx eax, al
}

// Lower an FP op (arith / convert / compare). Returns false if `ty` is not an
// FP op handled here. Floating bits travel in GP regs/slots; xmm0/xmm1 scratch.
static bool lower_fp_op(FnCtx *F, MLIR_OpHandle op, MLIR_OpType ty) {
    MLIR_Context *ctx = F->ctx;
    switch (ty) {
    // arithmetic: dst = op0 OP op1
    case OP_TYPE_LLVM_FADD: case OP_TYPE_LLVM_FSUB:
    case OP_TYPE_LLVM_FMUL: case OP_TYPE_LLVM_FDIV: {
        uint8_t opc = ty == OP_TYPE_LLVM_FADD ? 0x58 : ty == OP_TYPE_LLVM_FSUB ? 0x5C :
                      ty == OP_TYPE_LLVM_FMUL ? 0x59 : 0x5E;
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        int fb = fp_bytes(ctx, r);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);
        load_val(F, MLIR_GetOpOperand(ctx, op, 1), RCX);
        movq_xmm_gpr(F, 0, RAX);
        movq_xmm_gpr(F, 1, RCX);
        if (fb == 4) sse_ss(F, opc, 0, 1);
        else         sse_sd(F, opc, 0, 1);
        movq_gpr_xmm(F, RAX, 0);
        store_val(F, r, RAX);
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
        // Unsigned: convert into a 64-bit register, then store_val truncates to
        // the result width (correct for the full u32 range).
        int conv_w = ty == OP_TYPE_LLVM_FPTOUI ? 1 : val_w(ctx, r);
        load_val(F, s, RAX);
        movq_xmm_gpr(F, 0, RAX);
        cvtf2si(F, pfx, RAX, 0, conv_w);
        if (ty == OP_TYPE_LLVM_FPTOUI && val_w(ctx, r)) {
            // u64 >= 2^63: the signed convert gave 0x8000000000000000. Convert
            // x - 2^63 too and OR it in when the first result was "negative":
            // rax |= cvt(x - 2^63) & (rax >>s 63).
            mov_ri(F, RCX, sfb == 4 ? 0x5F000000ll : 0x43E0000000000000ll, 1);
            movq_xmm_gpr(F, 1, RCX);
            if (sfb == 4) sse_ss(F, 0x5C, 0, 1); else sse_sd(F, 0x5C, 0, 1);  // x - 2^63
            cvtf2si(F, pfx, RCX, 0, 1);
            mov_rr(F, RDX, RAX, 1);
            rex(F, 1, 0, 0, RDX); eb(F, 0xC1); modrm(F, 3, 7, RDX); eb(F, 63);  // sar rdx, 63
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
        // Signed i32: read EAX as signed 32-bit (cvtsi2sd reg32). Signed i64 and
        // unsigned i32 (zero-extended into RAX, hence positive): read RAX 64-bit.
        int conv_w = (ty == OP_TYPE_LLVM_UITOFP || sb > 32) ? 1 : 0;
        uint8_t pfx = rfb == 4 ? 0xF3 : 0xF2;
        load_val(F, s, RAX);
        if (ty == OP_TYPE_LLVM_UITOFP && sb > 32) {
            // u64 with the top bit set does not fit the signed convert: convert
            // (x >> 1) | (x & 1) (the sticky bit keeps the rounding right) and
            // double it, then pick that result when x is "negative".
            mov_rr(F, RCX, RAX, 1);
            rex(F, 1, 0, 0, RCX); eb(F, 0xD1); modrm(F, 3, 5, RCX);  // shr rcx, 1
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
        cvtsi2f(F, pfx, 0, RAX, conv_w);
        movq_gpr_xmm(F, RAX, 0);
        store_val(F, r, RAX);
        return true;
    }
    // f32 -> f64 (extend) / f64 -> f32 (truncate)
    case OP_TYPE_LLVM_FPEXT: case OP_TYPE_LLVM_FPTRUNC: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);
        movq_xmm_gpr(F, 0, RAX);
        cvt_f2f(F, ty == OP_TYPE_LLVM_FPEXT ? 0xF3 : 0xF2, 0, 0);
        movq_gpr_xmm(F, RAX, 0);
        store_val(F, r, RAX);
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
        load_val(F, a0, RAX);
        load_val(F, MLIR_GetOpOperand(ctx, op, 1), RCX);
        movq_xmm_gpr(F, 0, RAX);
        movq_xmm_gpr(F, 1, RCX);
        emit_fcmp(F, pred, is_f32);
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
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);
        movq_xmm_gpr(F, 0, RAX);
        if (fb == 4) sse_ss(F, 0x51, 0, 0);   // sqrtss
        else         sse_sd(F, 0x51, 0, 0);   // sqrtsd
        movq_gpr_xmm(F, RAX, 0);
        store_val(F, r, RAX);
        return true;
    }
    default: break;
    }
    return false;
}

// emit parallel-move-safe edge copies for the `si`-th successor of `term`
static void emit_edge(FnCtx *F, MLIR_OpHandle term, size_t si, MLIR_BlockHandle target) {
    size_t n = MLIR_GetOpNumSuccessorOperands(F->ctx, term, si);
    // pass 1: sources -> temp staging area
    for (size_t i = 0; i < n; i++) {
        MLIR_ValueHandle src = MLIR_GetOpSuccessorOperand(F->ctx, term, si, i);
        load_val(F, src, RAX);
        store_rbp(F, (int32_t)(F->edge_tmp - i * 8), RAX, 1);
    }
    // pass 2: temps -> destination block-arg homes (register or slot)
    for (size_t i = 0; i < n; i++) {
        MLIR_ValueHandle darg = MLIR_GetBlockArg(F->ctx, target, i);
        load_rbp(F, RAX, (int32_t)(F->edge_tmp - i * 8), 1);
        store_val(F, darg, RAX);
    }
}

// Restore callee-saved registers, then `leave ; ret`.
static void emit_epilogue(FnCtx *F) {
    for (int j = F->n_cs - 1; j >= 0; j--)
        load_rbp(F, F->cs_reg[j], F->cs_base + 8 * (j + 1), 1);
    eb(F, 0xC9); eb(F, 0xC3);   // leave ; ret
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

void mlir_llvm_to_x64_set_stage_times(MLIR_X64StageTimes *t) { g_x64_times = t; }

static uint64_t x64_clock(void) { return g_x64_times ? platform_clock_ns() : 0; }

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
    if (pad) { eb(F, 0x48); eb(F, 0x83); modrm(F, 3, 5, RSP); eb(F, 0x08); }  // sub rsp, 8
    for (size_t i = no; i > reg_cutoff; i--) {
        load_val(F, MLIR_GetOpOperand(F->ctx, op, arg_base + i - 1), RAX);
        eb(F, 0x50);                                  // push rax
    }
    for (size_t i = 0; i < reg_cutoff; i++)
        load_val(F, MLIR_GetOpOperand(F->ctx, op, arg_base + i), ARG_REGS[i]);
    if (is_vararg_call) alu_rr(F, 0x31, RAX, RAX, 0);     // xor eax, eax (AL = # xmm regs)
    *nstack_out = nstack;
    *pad_out = pad;
}

// emit `call rel32` to a named symbol (direct-call fast path).
static void emit_direct_call(FnCtx *F, string callee, SymSet *refs) {
    eb(F, 0xE8);
    uint32_t field = here(F); e32(F, 0);
    add_text_reloc(F, field, callee);
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
            load_val_w(F, a0, dst, w);
            alu_ri(F, ext, dst, (int32_t)imm, w);
            if (!has_rD) store_val_w(F, r, dst, w);
            return;
        }
        uint8_t r1 = val_reg_or_load(F, a1, RCX, w);
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
            uint8_t s0 = val_reg_or_load(F, a0, RAX, w);
            uint8_t dst = has_rD ? rD : RAX;
            imul_ri(F, dst, s0, (int32_t)imm, w);
            if (!has_rD) store_val_w(F, r, dst, w);
            return;
        }
        uint8_t r1 = val_reg_or_load(F, a1, RCX, w);
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
        int w = val_w(ctx, r);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);
        load_val(F, MLIR_GetOpOperand(ctx, op, 1), RCX);
        if (sdiv || srem) {
            if (w) eb(F, 0x48);
            eb(F, 0x99);                          // cqo / cdq
            rex(F, w, 0, 0, RCX); eb(F, 0xF7); modrm(F, 3, 7, RCX);   // idiv rcx
        } else {
            alu_rr(F, 0x31, RDX, RDX, 0);         // xor edx,edx
            rex(F, w, 0, 0, RCX); eb(F, 0xF7); modrm(F, 3, 6, RCX);   // div rcx
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
            if (imm == 1)      { rex(F, w, 0, 0, dst); eb(F, 0xD1); modrm(F, 3, ext, dst); }
            else if (imm != 0) { rex(F, w, 0, 0, dst); eb(F, 0xC1); modrm(F, 3, ext, dst); eb(F, (uint8_t)imm); }
            if (!has_rD) store_val_w(F, r, dst, w);
            return;
        }
        load_val(F, a0, RAX);
        load_val(F, a1, RCX);
        rex(F, w, 0, 0, RAX); eb(F, 0xD3); modrm(F, 3, ext, RAX);
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
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);   // 32-bit load zero-extends
        store_val(F, r, RAX);
        return;
    }
    case OP_TYPE_LLVM_TRUNC: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        int rb = int_bits(ctx, r);
        load_val(F, MLIR_GetOpOperand(ctx, op, 0), RAX);
        if (rb && rb < 32) {                           // mask low rb bits
            eb(F, 0x25); e32(F, (rb >= 32) ? 0xffffffffu : ((1u << rb) - 1u));  // and eax, imm32
        }
        store_val(F, r, RAX);
        return;
    }
    case OP_TYPE_LLVM_SEXT: {
        MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(ctx, op, 0);
        int sb = int_bits(ctx, s);
        load_val(F, s, RAX);
        if (sb == 32)      { rex(F,1,RAX,0,RAX); eb(F,0x63); modrm(F,3,RAX,RAX); }       // movsxd rax,eax
        else if (sb == 16) { rex(F,1,RAX,0,RAX); eb(F,0x0F); eb(F,0xBF); modrm(F,3,RAX,RAX); }
        else               { rex(F,1,RAX,0,RAX); eb(F,0x0F); eb(F,0xBE); modrm(F,3,RAX,RAX); } // i1/i8
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
        uint8_t rv = val_reg_or_load(F, val, RAX, nb > 4 ? 1 : 0);
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
                if (ib == 32)      { rex(F,1,RCX,0,RCX); eb(F,0x63); modrm(F,3,RCX,RCX); }              // movsxd rcx,ecx
                else if (ib == 16) { rex(F,1,RCX,0,RCX); eb(F,0x0F); eb(F,0xBF); modrm(F,3,RCX,RCX); }  // movsx rcx,cx
                else if (ib && ib < 16) { rex(F,1,RCX,0,RCX); eb(F,0x0F); eb(F,0xBE); modrm(F,3,RCX,RCX); } // movsx rcx,cl
                if (stride != 1) imul_ri(F, RCX, RCX, (int32_t)stride, 1);  // rcx *= stride
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
                eb(F, 0x50);                                 // push rax (a6)
                for (size_t i = 0; i < 6; i++)
                    load_val(F, MLIR_GetOpOperand(ctx, op, i), ARG_REGS[i]);
                emit_direct_call(F, callee, refs);
                eb(F, 0x48); eb(F, 0x83); modrm(F, 3, 0, RSP); eb(F, 0x08);  // add rsp, 8
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
            rex(F, 0, 0, 0, R11);
            eb(F, 0xFF);
            modrm(F, 3, 2, R11);   // call *r11
        } else {
            emit_direct_call(F, callee, refs);
        }
        if (nstack) {
            uint32_t cleanup = (uint32_t)((nstack + pad) * 8);   // pop stack args + pad
            rex(F, 1, 0, 0, RSP); eb(F, 0x81); modrm(F, 3, 0, RSP); e32(F, cleanup);  // add rsp, imm32
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
        rex(F, 1, RAX, 0, 0); eb(F, 0x8D); modrm(F, 2, RAX, RBP); e32(F, (uint32_t)disp); // lea rax,[rbp+disp]
        store_mem(F, RCX, RAX, 8);                               // *va_list = cursor
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
        lea_rip_data(F, RSI, off);
        load_mem(F, RSI, RSI, 4, false);                        // rsi = old
        mov_rr(F, RDX, RSI, 0);                                  // rdx = old
        alu_rr(F, 0x01, RDX, RCX, 0);                            // rdx = old+delta
        mov_ri(F, RCX, 65536, 0);
        alu_rr(F, 0x39, RDX, RCX, 0);                            // cmp new, cap
        cmovcc(F, RDX, RSI, CC_A, 0);                              // rdx = ok? new : old
        lea_rip_data(F, RAX, off);
        store_mem(F, RAX, RDX, 4);
        mov_ri(F, RAX, -1, 0);
        cmovcc(F, RAX, RSI, CC_BE, 0);                           // rax = ok? old : -1
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
            eb(F, 0xE9); uint32_t field = here(F); e32(F, 0);
            add_fix(F, block_index(F->ctx, reg, t), field);
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
        if (!need_t && !need_f) {
            if (ft == next_blk) {
                eb(F, 0x0F); eb(F, (uint8_t)(0x80 + CC_NE));     // jne true
                uint32_t f_jne = here(F); e32(F, 0);
                add_fix(F, block_index(F->ctx, reg, tt), f_jne);
                return;
            }
            if (tt == next_blk) {
                eb(F, 0x0F); eb(F, (uint8_t)(0x80 + CC_E));      // je false
                uint32_t f_je = here(F); e32(F, 0);
                add_fix(F, block_index(F->ctx, reg, ft), f_je);
                return;
            }
        }
        eb(F, 0x0F); eb(F, (uint8_t)(0x80 + CC_NE));     // jne true
        uint32_t f_jne = here(F); e32(F, 0);
        eb(F, 0xE9); uint32_t f_jmp = here(F); e32(F, 0); // jmp false
        if (need_t) {
            patch_rel32_at(F, f_jne, here(F));
            emit_edge(F, op, 0, tt);
            if (need_f || tt != next_blk) {
                eb(F, 0xE9); uint32_t f_br = here(F); e32(F, 0);
                add_fix(F, block_index(F->ctx, reg, tt), f_br);
            }
        } else {
            add_fix(F, block_index(F->ctx, reg, tt), f_jne);
        }
        if (need_f) {
            patch_rel32_at(F, f_jmp, here(F));
            emit_edge(F, op, 1, ft);
            if (ft != next_blk) {
                eb(F, 0xE9); uint32_t f_br = here(F); e32(F, 0);
                add_fix(F, block_index(F->ctx, reg, ft), f_br);
            }
        } else {
            add_fix(F, block_index(F->ctx, reg, ft), f_jmp);
        }
        return;
    }
    case OP_TYPE_LLVM_UNREACHABLE: eb(F, 0x0F); eb(F, 0x0B); return;  // ud2
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

static bool x64_val_dead_after_term(FnCtx *F, MLIR_ValueHandle v) {
    if (F->fast_ra) {
        uint8_t dummy;
        return val_in_reg(F, v, &dummy);
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
    if (ft == next_blk) {
        eb(F, 0x0F); eb(F, (uint8_t)(0x80 + cc));
        uint32_t fld = here(F); e32(F, 0);
        add_fix(F, block_index(ctx, reg, tt), fld);
        return;
    }
    if (tt == next_blk) {
        eb(F, 0x0F); eb(F, (uint8_t)(0x80 + (cc ^ 1)));
        uint32_t fld = here(F); e32(F, 0);
        add_fix(F, block_index(ctx, reg, ft), fld);
        return;
    }
    eb(F, 0x0F); eb(F, (uint8_t)(0x80 + cc));
    uint32_t f_t = here(F); e32(F, 0);
    add_fix(F, block_index(ctx, reg, tt), f_t);
    eb(F, 0xE9); uint32_t f_f = here(F); e32(F, 0);
    add_fix(F, block_index(ctx, reg, ft), f_f);
}

static void emit_func_into(MLIR_Context *ctx, MLIR_OpHandle fn, ObjFunc *f, SymSet *refs) {
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
    F.n_blocks = nb;
    static uint32_t *g_x64_block_off = NULL;
    static size_t g_x64_block_off_cap = 0;
    size_t nb_need = nb ? nb : 1;
    if (nb_need > g_x64_block_off_cap) {
        if (g_x64_block_off) buddy_free(g_x64_block_off);
        size_t want_cap = nb_need < 64 ? 64 : nb_need * 2;
        size_t actual_bo = 0;
        g_x64_block_off = (uint32_t *)buddy_alloc(want_cap * sizeof(uint32_t), &actual_bo);
        g_x64_block_off_cap = actual_bo / sizeof(uint32_t);
    }
    memset(g_x64_block_off, 0, nb_need * sizeof(uint32_t));
    F.block_off = g_x64_block_off;

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
        bool use_fast = (x64_ra_mode == 1 && !F.wasm_lifted);
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
    }
    uint64_t t_isel = x64_clock();
    if (g_x64_times) g_x64_times->regalloc_ns += t_isel - t_ra;

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
    eb(&F, 0x55);
    rex(&F, 1, 0, 0, RBP); eb(&F, 0x89); modrm(&F, 3, RSP, RBP);    // mov rbp, rsp
    if (F.frame) { rex(&F, 1, 0, 0, RSP); eb(&F, 0x81); modrm(&F, 3, 5, RSP); e32(&F, F.frame); } // sub rsp, imm32

    // save the callee-saved registers the allocator used
    for (int j = 0; j < F.n_cs; j++)
        store_rbp(&F, F.cs_base + 8 * (j + 1), F.cs_reg[j], 1);

    // --- entry params: incoming arg regs -> entry block-arg homes (reg/slot).
    //     Homes never alias an incoming arg register (arg regs are out of the
    //     pool), so the sequential moves need no parallel-move resolution.
    //     Params 6.. arrive on the caller's stack at [rbp+16], [rbp+24], ... ---
    if (nb > 0) {
        MLIR_BlockHandle e0 = MLIR_GetRegionBlock(ctx, reg, 0);
        size_t na = MLIR_GetBlockNumArgs(ctx, e0);
        F.n_fixed = (uint32_t)na;
        for (size_t a = 0; a < na && a < 6; a++)
            store_val(&F, MLIR_GetBlockArg(ctx, e0, a), ARG_REGS[a]);
        for (size_t a = 6; a < na; a++) {
            int32_t disp = (int32_t)(16 + (a - 6) * 8);   // [rbp + disp]
            load_rbp(&F, RAX, -disp, 1);
            store_val(&F, MLIR_GetBlockArg(ctx, e0, a), RAX);
        }
    }

    // --- emit blocks ---
    for (size_t b = 0; b < nb && F.ok; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        MLIR_BlockHandle next_bk = (b + 1 < nb) ? MLIR_GetRegionBlock(ctx, reg, b + 1) : MLIR_INVALID_HANDLE;
        F.block_off[b] = here(&F);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        // Check for fusable `icmp [-> select(1,0) -> icmp ne/eq 0] -> cond_br` at block tail,
        // skipping any rematerialized constant ops in between (which emit no code).
        size_t emit_end = no > 0 ? no - 1 : 0;
        MLIR_OpHandle fused_icmp = MLIR_INVALID_HANDLE;
        bool fused_invert = false;
        if (no >= 2) {
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            if (MLIR_GetOpType(ctx, term) == OP_TYPE_LLVM_COND_BR &&
                MLIR_GetOpNumSuccessorOperands(ctx, term, 0) == 0 &&
                MLIR_GetOpNumSuccessorOperands(ctx, term, 1) == 0) {
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
        for (size_t i = 0; i < emit_end && F.ok; i++)
            lower_op(&F, MLIR_GetBlockOp(ctx, bk, i), refs);
        if (no > 0 && F.ok) {
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            if (fused_icmp != MLIR_INVALID_HANDLE)
                emit_fused_cond_br(&F, reg, term, next_bk, fused_icmp, fused_invert);
            else
                lower_term(&F, reg, term, next_bk);
        }
    }

    // --- patch intra-function branch rel32s ---
    for (size_t i = 0; i < F.n_fix; i++) {
        int tb = F.fix[i].target_block;
        uint32_t fld = F.fix[i].field_off;
        int32_t rel = (int32_t)F.block_off[tb] - (int32_t)(fld + 4);
        buf_patch_le32(&f->code, fld, (uint32_t)rel);
    }

    if (!F.ok) g_x64_ok = false;
    g_x64_ok = g_x64_ok && F.ok;

    vmap_free(&F.slots);
    if (g_x64_times) g_x64_times->isel_encode_ns += x64_clock() - t_isel;
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

typedef struct { uint32_t slot_off; uint32_t target_off; } X64PtrReloc;
typedef struct { uint32_t slot_off; string target; } PendingPtrReloc;

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
    (*n_pend)++;
    return true;
}

static void resolve_pending_ptr_globals(GMap *gm, PendingPtrReloc *pend, size_t n_pend,
                                        X64PtrReloc **relocs_out, size_t *n_relocs_out) {
    *relocs_out = NULL;
    *n_relocs_out = 0;
    if (!n_pend || !g_x64_ok) return;

    X64PtrReloc *relocs = (X64PtrReloc *)malloc(n_pend * sizeof(X64PtrReloc));
    for (size_t k = 0; k < n_pend; k++) {
        if (!gmap_get(gm, pend[k].target, &relocs[k].target_off)) {
            fprintf(stderr, "x64: pointer global target '%.*s' not found\n",
                    (int)pend[k].target.size, pend[k].target.str);
            g_x64_ok = false;
            break;
        }
        relocs[k].slot_off = pend[k].slot_off;
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

    resolve_pending_ptr_globals(gm, pend, n_pend, relocs_out, n_relocs_out);
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

// Synthesize the raw-syscall thunk for __tinyc_syscall6 / syscall.
// All corec/WASI forwarding stubs (platform_init, __assert_fail) live in
// corec/wasm/wasi_adapter.c so the x86_64 backend stays agnostic to corec
// symbol names.
static bool synth_thunk(ObjModule *m, string name) {
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
