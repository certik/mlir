// Op selection + function emission + module orchestration for the x86_64 ELF
// backend. Included at the end of mlir_llvm_to_x64.c (shares its statics).

// --- small attribute helpers ----------------------------------------------
static string attr_str(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
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

// --- LLVM-type layout (LP64: pointers are 8 bytes) -------------------------
// Shared by getelementptr stride math and typed global serialization.
static unsigned x64_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty);
static unsigned x64_align_up(unsigned x, unsigned a) { return (x + a - 1) & ~(a - 1); }
static unsigned x64_type_align(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_IsTypeLLVMArray(ty))
        return x64_type_align(ctx, MLIR_GetTypeLLVMArrayElement(ty));
    if (MLIR_IsTypeLLVMStruct(ty)) {
        size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
        unsigned ma = 1;
        for (size_t i = 0; i < nf; i++) {
            unsigned fa = x64_type_align(ctx, MLIR_GetTypeLLVMStructField(ty, i));
            if (fa > ma) ma = fa;
        }
        return ma;
    }
    unsigned sz = x64_type_size(ctx, ty);
    return sz ? sz : 1;
}
static unsigned x64_struct_size(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
    unsigned off = 0, max_align = 1;
    for (size_t i = 0; i < nf; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ty, i);
        unsigned fsz = x64_type_size(ctx, ft);
        unsigned fal = x64_type_align(ctx, ft);
        if (fsz == 0 || fal == 0) return 0;
        off = x64_align_up(off, fal);
        off += fsz;
        if (fal > max_align) max_align = fal;
    }
    return x64_align_up(off, max_align);
}
// Byte offset of field `fld` within an LLVM struct (native LP64 layout).
static unsigned x64_struct_field_offset(MLIR_Context *ctx, MLIR_TypeHandle sty, size_t fld) {
    unsigned off = 0;
    for (size_t i = 0; i <= fld; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(sty, i);
        off = x64_align_up(off, x64_type_align(ctx, ft));
        if (i == fld) return off;
        off += x64_type_size(ctx, ft);
    }
    return off;
}
static unsigned x64_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return 8;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return 8;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return 8;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8) return 1;
        if (w == 16) return 2;
        if (w == 32) return 4;
        if (w == 64) return 8;
    }
    if (MLIR_IsTypeLLVMArray(ty)) {
        unsigned esz = x64_type_size(ctx, MLIR_GetTypeLLVMArrayElement(ty));
        if (esz == 0) return 0;
        return esz * (unsigned)MLIR_GetTypeLLVMArrayNumElements(ty);
    }
    if (MLIR_IsTypeLLVMStruct(ty)) return x64_struct_size(ctx, ty);
    return 0;
}

// Parse a decimal i32 list "...: a, b, c>" (GEP rawConstantIndices). The
// sentinel 0x80000000 marks a dynamic index supplied as an SSA operand.
static size_t parse_i32_array(string cs, int32_t *out, size_t cap) {
    size_t p = 0, got = 0;
    while (p < cs.size && cs.str[p] != ':') p++;
    if (p < cs.size) p++;
    while (p < cs.size && got < cap) {
        while (p < cs.size && (cs.str[p] == ' ' || cs.str[p] == ',')) p++;
        if (p >= cs.size || cs.str[p] == '>') break;
        int64_t sign = 1;
        if (cs.str[p] == '-') { sign = -1; p++; }
        int64_t v = 0;
        while (p < cs.size && cs.str[p] >= '0' && cs.str[p] <= '9')
            v = v * 10 + (cs.str[p++] - '0');
        out[got++] = (int32_t)(sign * v);
    }
    return got;
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

static void add_fix(FnCtx *F, int target_block, uint32_t field_off) {
    if (F->n_fix == F->c_fix) {
        F->c_fix = F->c_fix ? F->c_fix * 2 : 8;
        F->fix = (BrFix *)realloc(F->fix, F->c_fix * sizeof(BrFix));
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

// block index of a successor block within this function's single region
static int block_index(MLIR_RegionHandle reg, MLIR_BlockHandle blk) {
    size_t n = MLIR_GetRegionNumBlocks(reg);
    for (size_t i = 0; i < n; i++)
        if (MLIR_GetRegionBlock(reg, i) == blk) return (int)i;
    return -1;
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

// Lower an FP op (arith / convert / compare). Returns false if `nm` is not an
// FP op handled here. Floating bits travel in GP regs/slots; xmm0/xmm1 scratch.
static bool lower_fp_op(FnCtx *F, MLIR_OpHandle op, string nm) {
    MLIR_Context *ctx = F->ctx;
    // arithmetic: dst = op0 OP op1
    struct { const char *n; uint8_t opc; } fpalu[] = {
        {"llvm.fadd",0x58},{"llvm.fsub",0x5C},{"llvm.fmul",0x59},{"llvm.fdiv",0x5E}
    };
    for (size_t i = 0; i < sizeof(fpalu)/sizeof(fpalu[0]); i++) {
        if (str_eqz(nm, fpalu[i].n)) {
            MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
            int fb = fp_bytes(ctx, r);
            load_val(F, MLIR_GetOpOperand(op, 0), RAX);
            load_val(F, MLIR_GetOpOperand(op, 1), RCX);
            movq_xmm_gpr(F, 0, RAX);
            movq_xmm_gpr(F, 1, RCX);
            if (fb == 4) sse_ss(F, fpalu[i].opc, 0, 1);
            else         sse_sd(F, fpalu[i].opc, 0, 1);
            movq_gpr_xmm(F, RAX, 0);
            store_val(F, r, RAX);
            return true;
        }
    }
    // fneg: flip the sign bit in the integer domain (exact, no xmm needed)
    if (str_eqz(nm, "llvm.fneg")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int fb = fp_bytes(ctx, r);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        mov_ri(F, RCX, (fb == 4) ? (int64_t)0x80000000ll : (int64_t)0x8000000000000000ll, 1);
        alu_rr(F, 0x31, RAX, RCX, 1);          // xor rax, rcx
        store_val(F, r, RAX);
        return true;
    }
    // llvm.intr.sqrt: hardware sqrt (mirrors llvm->aarch64 fsqrt lowering).
    if (str_eqz(nm, "llvm.intr.sqrt")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int fb = fp_bytes(ctx, r);
        if (fb != 4 && fb != 8) {
            fail(F, "sqrt on non-float result", nm);
            return true;
        }
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        movq_xmm_gpr(F, 0, RAX);
        if (fb == 4) sse_ss(F, 0x51, 0, 0);   // sqrtss
        else         sse_sd(F, 0x51, 0, 0);   // sqrtsd
        movq_gpr_xmm(F, RAX, 0);
        store_val(F, r, RAX);
        return true;
    }
    // float -> signed/unsigned int (truncating)
    if (str_eqz(nm, "llvm.fptosi") || str_eqz(nm, "llvm.fptoui")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        int sfb = fp_bytes(ctx, s);
        // Unsigned: convert into a 64-bit register, then store_val truncates to
        // the result width (correct for the full u32 range; u64 high bit is an
        // unreachable edge case here).
        int conv_w = str_eqz(nm, "llvm.fptoui") ? 1 : val_w(ctx, r);
        load_val(F, s, RAX);
        movq_xmm_gpr(F, 0, RAX);
        cvtf2si(F, sfb == 4 ? 0xF3 : 0xF2, RAX, 0, conv_w);
        store_val(F, r, RAX);
        return true;
    }
    // signed/unsigned int -> float
    if (str_eqz(nm, "llvm.sitofp") || str_eqz(nm, "llvm.uitofp")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        int rfb = fp_bytes(ctx, r);
        int sb  = int_bits(ctx, s);
        // Signed i32: read EAX as signed 32-bit (cvtsi2sd reg32). Signed i64 and
        // unsigned i32 (zero-extended into RAX, hence positive): read RAX 64-bit.
        int conv_w = (str_eqz(nm, "llvm.uitofp") || sb > 32) ? 1 : 0;
        load_val(F, s, RAX);
        cvtsi2f(F, rfb == 4 ? 0xF3 : 0xF2, 0, RAX, conv_w);
        movq_gpr_xmm(F, RAX, 0);
        store_val(F, r, RAX);
        return true;
    }
    // f32 -> f64 (extend) / f64 -> f32 (truncate)
    if (str_eqz(nm, "llvm.fpext") || str_eqz(nm, "llvm.fptrunc")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        movq_xmm_gpr(F, 0, RAX);
        cvt_f2f(F, str_eqz(nm, "llvm.fpext") ? 0xF3 : 0xF2, 0, 0);
        movq_gpr_xmm(F, RAX, 0);
        store_val(F, r, RAX);
        return true;
    }
    // ordered/unordered float comparison -> 0/1
    if (str_eqz(nm, "llvm.fcmp")) {
        MLIR_ValueHandle r  = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(op, 0);
        int is_f32 = (fp_bytes(ctx, a0) == 4);
        int64_t pred = 0;
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa) pred = MLIR_GetAttributeInteger(pa);
        load_val(F, a0, RAX);
        load_val(F, MLIR_GetOpOperand(op, 1), RCX);
        movq_xmm_gpr(F, 0, RAX);
        movq_xmm_gpr(F, 1, RCX);
        emit_fcmp(F, pred, is_f32);
        store_val(F, r, RAX);
        return true;
    }
    return false;
}

// emit parallel-move-safe edge copies for the `si`-th successor of `term`
static void emit_edge(FnCtx *F, MLIR_OpHandle term, size_t si, MLIR_BlockHandle target) {
    size_t n = MLIR_GetOpNumSuccessorOperands(term, si);
    // pass 1: sources -> temp staging area
    for (size_t i = 0; i < n; i++) {
        MLIR_ValueHandle src = MLIR_GetOpSuccessorOperand(term, si, i);
        load_val(F, src, RAX);
        store_rbp(F, (int32_t)(F->edge_tmp - i * 8), RAX, 1);
    }
    // pass 2: temps -> destination block-arg homes (register or slot)
    for (size_t i = 0; i < n; i++) {
        MLIR_ValueHandle darg = MLIR_GetBlockArg(target, i);
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
static MLIR_BlockHandle g_x64_mb = MLIR_INVALID_HANDLE;  // module IR block (funcs/globals)
static Buf *g_x64_data_buf = NULL;           // -> stack Buf in mlir_llvm_to_elf (.data blob)
static bool  g_x64_ok = true;

// SysV AMD64 argument marshaling for llvm.call. `arg_base` skips the fn-pointer
// operand on indirect calls. Returns stack-arg count and alignment pad for cleanup.
static void emit_sysv_call_args(FnCtx *F, MLIR_OpHandle op, size_t arg_base,
                                size_t *nstack_out, size_t *pad_out) {
    size_t no = MLIR_GetOpNumOperands(op) - arg_base;
    size_t reg_cutoff = 6;
    bool is_vararg_call = false;
    MLIR_AttributeHandle vct = MLIR_GetOpAttributeByName(op, "var_callee_type");
    if (vct != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle ft = MLIR_GetAttributeTypeValue(vct);
        if (ft != MLIR_INVALID_HANDLE && MLIR_GetTypeFunctionIsVarArg(ft)) {
            size_t nf = MLIR_GetTypeFunctionNumInputs(ft);
            reg_cutoff = nf < 6 ? nf : 6;
            is_vararg_call = true;
        }
    }
    if (reg_cutoff > no) reg_cutoff = no;
    size_t nstack = no - reg_cutoff;
    size_t pad = nstack & 1;
    if (pad) { eb(F, 0x48); eb(F, 0x83); modrm(F, 3, 5, RSP); eb(F, 0x08); }  // sub rsp, 8
    for (size_t i = no; i > reg_cutoff; i--) {
        load_val(F, MLIR_GetOpOperand(op, arg_base + i - 1), RAX);
        eb(F, 0x50);                                  // push rax
    }
    for (size_t i = 0; i < reg_cutoff; i++)
        load_val(F, MLIR_GetOpOperand(op, arg_base + i), ARG_REGS[i]);
    if (is_vararg_call) alu_rr(F, 0x31, RAX, RAX, 0);     // xor eax, eax (AL = # xmm regs)
    *nstack_out = nstack;
    *pad_out = pad;
}

// emit `call rel32` to a named symbol (direct-call fast path).
static void emit_direct_call(FnCtx *F, string callee, SymSet *refs) {
    eb(F, 0xE8);
    uint32_t field = here(F); e32(F, 0);
    add_text_reloc(F, field, callee);
    sym_add(refs, callee);
}

// ---------------------------------------------------------------------------
// Per-op selection (non-terminators).
// ---------------------------------------------------------------------------
static void lower_op(FnCtx *F, MLIR_OpHandle op, SymSet *refs) {
    string nm = MLIR_GetOpName(op);
    MLIR_Context *ctx = F->ctx;

    // value-less / rematerialized-at-use ops
    if (str_eqz(nm, "llvm.mlir.constant") || str_eqz(nm, "llvm.mlir.addressof") ||
        str_eqz(nm, "llvm.mlir.undef") || str_eqz(nm, "llvm.mlir.zero") ||
        str_eqz(nm, "llvm.mlir.null"))
        return;

    // integer binops: dst = op0 OP op1
    struct { const char *n; uint8_t opc; } alus[] = {
        {"llvm.add",0x01},{"llvm.sub",0x29},{"llvm.and",0x21},{"llvm.or",0x09},{"llvm.xor",0x31}
    };
    for (size_t i = 0; i < sizeof(alus)/sizeof(alus[0]); i++) {
        if (str_eqz(nm, alus[i].n)) {
            MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
            load_val(F, MLIR_GetOpOperand(op, 0), RAX);
            load_val(F, MLIR_GetOpOperand(op, 1), RCX);
            alu_rr(F, alus[i].opc, RAX, RCX, val_w(ctx, r));
            store_val(F, r, RAX);
            return;
        }
    }
    if (str_eqz(nm, "llvm.mul")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        load_val(F, MLIR_GetOpOperand(op, 1), RCX);
        imul_rr(F, RAX, RCX, val_w(ctx, r));
        store_val(F, r, RAX);
        return;
    }
    // division / remainder
    bool sdiv = str_eqz(nm,"llvm.sdiv"), udiv = str_eqz(nm,"llvm.udiv");
    bool srem = str_eqz(nm,"llvm.srem"), urem = str_eqz(nm,"llvm.urem");
    if (sdiv || udiv || srem || urem) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int w = val_w(ctx, r);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        load_val(F, MLIR_GetOpOperand(op, 1), RCX);
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
    // shifts: dst = op0 SH op1(cl)
    if (str_eqz(nm,"llvm.shl") || str_eqz(nm,"llvm.lshr") || str_eqz(nm,"llvm.ashr")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int w = val_w(ctx, r);
        int ext = str_eqz(nm,"llvm.shl") ? 4 : (str_eqz(nm,"llvm.lshr") ? 5 : 7);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        load_val(F, MLIR_GetOpOperand(op, 1), RCX);
        rex(F, w, 0, 0, RAX); eb(F, 0xD3); modrm(F, 3, ext, RAX);
        store_val(F, r, RAX);
        return;
    }
    // icmp -> 0/1
    if (str_eqz(nm, "llvm.icmp")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(op, 0);
        int w = val_w(ctx, a0);
        int64_t pred = 0;
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa) pred = MLIR_GetAttributeInteger(pa);
        load_val(F, a0, RAX);
        load_val(F, MLIR_GetOpOperand(op, 1), RCX);
        alu_rr(F, 0x39, RAX, RCX, w);             // cmp rax, rcx
        setcc_al(F, icmp_cc(pred));
        store_val(F, r, RAX);
        return;
    }
    // select cond, t, f
    if (str_eqz(nm, "llvm.select")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int w = val_w(ctx, r);
        load_val(F, MLIR_GetOpOperand(op, 2), RAX);   // default = false value
        load_val(F, MLIR_GetOpOperand(op, 1), RCX);   // true value
        load_val(F, MLIR_GetOpOperand(op, 0), RDX);   // cond
        alu_rr(F, 0x85, RDX, RDX, 0);                 // test edx, edx
        cmovnz(F, RAX, RCX, w);                        // cond!=0 -> true value
        store_val(F, r, RAX);
        return;
    }
    // zext / trunc / sext / pointer casts
    if (str_eqz(nm,"llvm.zext")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);   // 32-bit load zero-extends
        store_val(F, r, RAX);
        return;
    }
    if (str_eqz(nm,"llvm.trunc")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int rb = int_bits(ctx, r);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        if (rb && rb < 32) {                           // mask low rb bits
            eb(F, 0x25); e32(F, (rb >= 32) ? 0xffffffffu : ((1u << rb) - 1u));  // and eax, imm32
        }
        store_val(F, r, RAX);
        return;
    }
    if (str_eqz(nm,"llvm.sext")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        int sb = int_bits(ctx, s);
        load_val(F, s, RAX);
        if (sb == 32)      { rex(F,1,RAX,0,RAX); eb(F,0x63); modrm(F,3,RAX,RAX); }       // movsxd rax,eax
        else if (sb == 16) { rex(F,1,RAX,0,RAX); eb(F,0x0F); eb(F,0xBF); modrm(F,3,RAX,RAX); }
        else               { rex(F,1,RAX,0,RAX); eb(F,0x0F); eb(F,0xBE); modrm(F,3,RAX,RAX); } // i1/i8
        store_val(F, r, RAX);
        return;
    }
    if (str_eqz(nm,"llvm.inttoptr") || str_eqz(nm,"llvm.ptrtoint") || str_eqz(nm,"llvm.bitcast")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        store_val(F, r, RAX);
        return;
    }
    // memory
    if (str_eqz(nm, "llvm.load")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        load_val(F, MLIR_GetOpOperand(op, 0), RCX);   // address
        load_mem(F, RAX, RCX, type_bytes(ctx, r), false);
        store_val(F, r, RAX);
        return;
    }
    if (str_eqz(nm, "llvm.store")) {
        MLIR_ValueHandle val = MLIR_GetOpOperand(op, 0);
        load_val(F, val, RAX);
        load_val(F, MLIR_GetOpOperand(op, 1), RCX);   // address
        store_mem(F, RCX, RAX, type_bytes(ctx, val));
        return;
    }
    if (str_eqz(nm, "llvm.getelementptr")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_AttributeHandle eta = MLIR_GetOpAttributeByName(op, "elem_type");
        MLIR_AttributeHandle ria = MLIR_GetOpAttributeByName(op, "rawConstantIndices");
        if (!eta || !ria) { fail(F, "getelementptr missing elem_type/indices", (string){0}); return; }
        MLIR_TypeHandle elem_ty = MLIR_GetAttributeTypeValue(eta);
        int32_t cidx[64];
        size_t n_idx = parse_i32_array(MLIR_GetAttributeAsString(ctx, ria), cidx, 64);
        if (n_idx == 0) { fail(F, "getelementptr with no indices", (string){0}); return; }
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);       // running address
        size_t op_idx = 1;                                 // next dynamic operand
        MLIR_TypeHandle cur_ty = elem_ty;
        for (size_t i = 0; i < n_idx; i++) {
            bool is_dyn = (cidx[i] == (int32_t)0x80000000);
            unsigned stride;
            if (i == 0) {                                  // index over `elem_type`
                stride = x64_type_size(ctx, elem_ty);
            } else if (MLIR_IsTypeLLVMStruct(cur_ty)) {    // struct field: constant offset
                if (is_dyn) { fail(F, "dynamic struct index in gep", (string){0}); return; }
                unsigned foff = x64_struct_field_offset(ctx, cur_ty, (size_t)cidx[i]);
                cur_ty = MLIR_GetTypeLLVMStructField(cur_ty, (size_t)cidx[i]);
                add_const_off(F, RAX, (int64_t)foff);
                continue;
            } else if (MLIR_IsTypeLLVMArray(cur_ty)) {     // array element stride
                MLIR_TypeHandle et = MLIR_GetTypeLLVMArrayElement(cur_ty);
                stride = x64_type_size(ctx, et);
                cur_ty = et;
            } else {
                fail(F, "gep into non-aggregate type", (string){0}); return;
            }
            if (stride == 0) continue;
            if (is_dyn) {
                if (op_idx >= MLIR_GetOpNumOperands(op)) { fail(F, "gep dynamic index operand missing", (string){0}); return; }
                MLIR_ValueHandle idx_v = MLIR_GetOpOperand(op, op_idx++);
                load_val(F, idx_v, RCX);
                // GEP indices are signed and sign-extend to the pointer width.
                // load_val zero-extends a 32-bit index, so a negative index
                // (e.g. `p - 1`) would scale as 0xFFFFFFFF*stride; sign-extend
                // narrow indices first.
                int ib = int_bits(ctx, idx_v);
                if (ib == 32)      { rex(F,1,RCX,0,RCX); eb(F,0x63); modrm(F,3,RCX,RCX); }              // movsxd rcx,ecx
                else if (ib == 16) { rex(F,1,RCX,0,RCX); eb(F,0x0F); eb(F,0xBF); modrm(F,3,RCX,RCX); }  // movsx rcx,cx
                else if (ib && ib < 16) { rex(F,1,RCX,0,RCX); eb(F,0x0F); eb(F,0xBE); modrm(F,3,RCX,RCX); } // movsx rcx,cl
                if (stride != 1) imul_ri(F, RCX, RCX, (int32_t)stride, 1);  // rcx *= stride
                alu_rr(F, 0x01, RAX, RCX, 1);              // add rax, rcx
            } else if (cidx[i] != 0) {
                add_const_off(F, RAX, (int64_t)cidx[i] * (int64_t)stride);
            }
        }
        store_val(F, r, RAX);
        return;
    }
    if (str_eqz(nm, "llvm.alloca")) {
        // region already reserved in the pre-pass; recover its rbp offset from
        // the slot map (negative sentinel keyed by the op result is the region).
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int32_t region;
        if (!vmap_get(&F->slots, (MLIR_ValueHandle)((uintptr_t)op | 1), &region)) {
            fail(F, "alloca region missing", (string){0}); return;
        }
        lea_rbp(F, RAX, region);
        store_val(F, r, RAX);
        return;
    }
    // calls
    if (str_eqz(nm, "llvm.call") || str_eqz(nm, "func.call")) {
        string callee = strip_at(attr_str(op, "callee"));
        if (!callee.size) callee = strip_at(attr_str(op, "callee_name"));
        bool is_indirect = (callee.size == 0);
        if (!is_indirect) {
            size_t no = MLIR_GetOpNumOperands(op);
            // __builtin_syscall6(num, a1..a6) -> @__tinyc_syscall6 (7 args). The
            // SysV C ABI passes 6 args in registers and the 7th (a6) on the stack;
            // the synthesised thunk reads it from [rsp+8] and traps via `syscall`.
            bool is_sc6 = str_eqz(callee, "__tinyc_syscall6");
            if (is_sc6) {
                if (no != 7) { fail(F, "__tinyc_syscall6 expects 7 args", callee); return; }
                load_val(F, MLIR_GetOpOperand(op, 6), RAX);
                eb(F, 0x50);                                 // push rax (a6)
                for (size_t i = 0; i < 6; i++)
                    load_val(F, MLIR_GetOpOperand(op, i), ARG_REGS[i]);
                emit_direct_call(F, callee, refs);
                eb(F, 0x48); eb(F, 0x83); modrm(F, 3, 0, RSP); eb(F, 0x08);  // add rsp, 8
                if (MLIR_GetOpNumResults(op) == 1)
                    store_val(F, MLIR_GetOpResult(op, 0), RAX);
                return;
            }
        }
        size_t arg_base = is_indirect ? 1 : 0;
        if (is_indirect)
            load_val(F, MLIR_GetOpOperand(op, 0), R11);
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
        if (MLIR_GetOpNumResults(op) == 1)
            store_val(F, MLIR_GetOpResult(op, 0), RAX);
        return;
    }

    // variadics: va_list is a single `char*` cursor over the on-stack variadic
    // args (matching the tinyc_va_arg_* runtime helpers). va_start writes the
    // address of the first variadic arg; va_arg is lowered to helper calls by
    // the front end, so nothing is needed here for it.
    if (str_eqz(nm, "llvm.intr.vastart")) {
        // First variadic arg sits just past the saved rbp + return address and
        // any fixed params that spilled to the stack: [rbp + 16 + max(0,
        // n_fixed-6)*8]. Store that address into the va_list slot (operand 0).
        int32_t disp = 16 + (F->n_fixed > 6 ? (int32_t)(F->n_fixed - 6) * 8 : 0);
        load_val(F, MLIR_GetOpOperand(op, 0), RCX);              // va_list address
        rex(F, 1, RAX, 0, 0); eb(F, 0x8D); modrm(F, 2, RAX, RBP); e32(F, (uint32_t)disp); // lea rax,[rbp+disp]
        store_mem(F, RCX, RAX, 8);                               // *va_list = cursor
        return;
    }
    if (str_eqz(nm, "llvm.intr.vaend")) return;                  // stack cursor: no teardown
    if (str_eqz(nm, "llvm.intr.vacopy")) {
        load_val(F, MLIR_GetOpOperand(op, 1), RCX);              // src va_list address
        load_mem(F, RAX, RCX, 8, false);                        // rax = *src
        load_val(F, MLIR_GetOpOperand(op, 0), RCX);              // dst va_list address
        store_mem(F, RCX, RAX, 8);                               // *dst = *src
        return;
    }

    // floating-point arithmetic / conversion / comparison (issue #212 Option A)
    if (lower_fp_op(F, op, nm)) return;

    fail(F, "unsupported op", nm);
}

// emit a terminator (also collects callee refs via lower_op for calls? no -
// terminators here are br/cond_br/return/unreachable only)
static void lower_term(FnCtx *F, MLIR_OpHandle op) {
    string nm = MLIR_GetOpName(op);
    MLIR_RegionHandle reg = MLIR_GetBlockParentRegion(MLIR_GetOpParentBlock(op));
    if (str_eqz(nm, "llvm.return") || str_eqz(nm, "func.return")) {
        if (MLIR_GetOpNumOperands(op) >= 1)
            load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        emit_epilogue(F);
        return;
    }
    if (str_eqz(nm, "llvm.unreachable")) { eb(F, 0x0F); eb(F, 0x0B); return; }  // ud2
    if (str_eqz(nm, "llvm.br")) {
        MLIR_BlockHandle t = MLIR_GetOpSuccessor(op, 0);
        emit_edge(F, op, 0, t);
        eb(F, 0xE9); uint32_t field = here(F); e32(F, 0);
        add_fix(F, block_index(reg, t), field);
        return;
    }
    if (str_eqz(nm, "llvm.cond_br")) {
        // Short-circuit edge copies for operand-carrying successors.
        // MLIR cond_br can pass values into a successor block as block arguments
        // (successor operands). emit_edge() performs those copies. They must run
        // only on the branch actually taken — otherwise && / || break.
        // Example:  if ((n != 0) && (100 / n > 0))
        //   - lowered to: test n != 0; if false -> skip inner check; if true ->
        //     copy edge operands and evaluate 100/n > 0
        //   - wrong: copy BOTH edges before testing n -> inner divide runs even
        //     when n == 0 (divide-by-zero trap)
        //   - right: test n first; only the true-edge landing pad runs emit_edge
        MLIR_BlockHandle tt = MLIR_GetOpSuccessor(op, 0);
        MLIR_BlockHandle ft = MLIR_GetOpSuccessor(op, 1);
        size_t nso_t = MLIR_GetOpNumSuccessorOperands(op, 0);
        size_t nso_f = MLIR_GetOpNumSuccessorOperands(op, 1);
        bool need_t = nso_t > 0;
        bool need_f = nso_f > 0;
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);      // cond
        alu_rr(F, 0x85, RAX, RAX, 0);                    // test eax, eax
        eb(F, 0x0F); eb(F, (uint8_t)(0x80 + CC_NE));     // jne true
        uint32_t f_jne = here(F); e32(F, 0);
        eb(F, 0xE9); uint32_t f_jmp = here(F); e32(F, 0); // jmp false
        if (need_t) {
            patch_rel32_at(F, f_jne, here(F));
            emit_edge(F, op, 0, tt);
            eb(F, 0xE9); uint32_t f_br = here(F); e32(F, 0);
            add_fix(F, block_index(reg, tt), f_br);
        } else {
            add_fix(F, block_index(reg, tt), f_jne);
        }
        if (need_f) {
            patch_rel32_at(F, f_jmp, here(F));
            emit_edge(F, op, 1, ft);
            eb(F, 0xE9); uint32_t f_br = here(F); e32(F, 0);
            add_fix(F, block_index(reg, ft), f_br);
        } else {
            add_fix(F, block_index(reg, ft), f_jmp);
        }
        return;
    }
    fail(F, "unsupported terminator", nm);
}

// ---------------------------------------------------------------------------
// Function emission: frame pre-pass, prologue, blocks, branch patch.
// ---------------------------------------------------------------------------
static unsigned alloca_bytes(MLIR_Context *ctx, MLIR_OpHandle op) {
    // element type size * count; default 8. Uses the shared LP64 layout
    // (x64_type_size), so nested arrays / structs / unions are sized correctly.
    MLIR_AttributeHandle et = MLIR_GetOpAttributeByName(op, "elem_type");
    unsigned esz = 8;
    if (et) {
        MLIR_TypeHandle t = MLIR_GetAttributeTypeValue(et);
        if (t != MLIR_INVALID_HANDLE) {
            unsigned s = x64_type_size(ctx, t);
            if (s) esz = s;
        }
    }
    if (esz == 0) esz = 8;
    return (esz + 7) & ~7u;
}

static void emit_func(MLIR_Context *ctx, MLIR_OpHandle fn, ObjModule *m, SymSet *refs) {
    string name = attr_str(fn, "sym_name");
    char *nm = (char *)malloc(name.size + 1);
    memcpy(nm, name.str, name.size); nm[name.size] = 0;
    bool exported = str_eqz(name, "main");
    ObjFunc *f = obj_add_func(m, nm, exported);

    FnCtx F; memset(&F, 0, sizeof(F));
    F.ctx = ctx; F.f = f; F.ok = true;
    F.gm = g_x64_gmap;
    F.module_mb = g_x64_mb;
    F.data_buf = g_x64_data_buf;

    MLIR_RegionHandle reg = MLIR_GetOpRegion(fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(reg);
    F.n_blocks = nb;
    F.block_off = (uint32_t *)calloc(nb ? nb : 1, sizeof(uint32_t));

    // --- register allocation: shared linear-scan allocator -> F.rm
    //     (value -> physical register). Values it leaves out stay in frame
    //     slots. TINYC_X64_NOREGALLOC forces the pure slot model (A/B aid). ---
    F.rm.arena = MLIR_GetArenaAllocator(ctx);
    if (!getenv("TINYC_X64_NOREGALLOC")) {
        SlotMap sm_spill; memset(&sm_spill, 0, sizeof(sm_spill));
        sm_spill.arena = MLIR_GetArenaAllocator(ctx);
        int32_t nslots = 0;
        RegTarget T = x64_regtarget();
        uint32_t um = mlir_regalloc_global(ctx, reg, nb, &F.rm, &sm_spill, &nslots, &T);
        if (um == RA_BAIL) {
            memset(&F.rm, 0, sizeof(F.rm));
            F.rm.arena = MLIR_GetArenaAllocator(ctx);
            um = mlir_regalloc_cfg(ctx, reg, nb, &F.rm, &T);
        }
        F.used_mask = um;
        for (int k = 0; k < T.npool - T.ncaller; k++)
            if (um & (1u << k)) F.cs_reg[F.n_cs++] = T.pool[T.ncaller + k];
    }

    // --- frame pre-pass: assign slots to every block arg + op result; reserve
    //     alloca regions; size the parallel-move temp area. ---
    size_t want = 0;
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        want += MLIR_GetBlockNumArgs(bk) + MLIR_GetBlockNumOps(bk) + 4;
    }
    vmap_init(&F.slots, want ? want : 8);
    int32_t cur = 0;
    uint32_t max_edge = 0;
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t na = MLIR_GetBlockNumArgs(bk);
        for (size_t a = 0; a < na; a++) { cur += 8; vmap_put(&F.slots, MLIR_GetBlockArg(bk, a), cur); }
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (MLIR_GetOpNumRegions(op) > 0) { fail(&F, "nested region (expected flat cf)", MLIR_GetOpName(op)); }
            if (str_eqz(MLIR_GetOpName(op), "llvm.alloca")) {
                unsigned rb = alloca_bytes(ctx, op);
                cur += (int32_t)rb;
                vmap_put(&F.slots, (MLIR_ValueHandle)((uintptr_t)op | 1), cur);  // region
            }
            if (MLIR_GetOpNumResults(op) == 1) { cur += 8; vmap_put(&F.slots, MLIR_GetOpResult(op, 0), cur); }
            MLIR_OpHandle t = op;
            size_t ns = MLIR_GetOpNumSuccessors(t);
            for (size_t s = 0; s < ns; s++) {
                uint32_t c = (uint32_t)MLIR_GetOpNumSuccessorOperands(t, s);
                if (c > max_edge) max_edge = c;
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
        MLIR_BlockHandle e0 = MLIR_GetRegionBlock(reg, 0);
        size_t na = MLIR_GetBlockNumArgs(e0);
        F.n_fixed = (uint32_t)na;
        for (size_t a = 0; a < na && a < 6; a++)
            store_val(&F, MLIR_GetBlockArg(e0, a), ARG_REGS[a]);
        for (size_t a = 6; a < na; a++) {
            int32_t disp = (int32_t)(16 + (a - 6) * 8);   // [rbp + disp]
            load_rbp(&F, RAX, -disp, 1);
            store_val(&F, MLIR_GetBlockArg(e0, a), RAX);
        }
    }

    // --- emit blocks ---
    for (size_t b = 0; b < nb && F.ok; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        F.block_off[b] = here(&F);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no && F.ok; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (i + 1 == no) lower_term(&F, op);
            else             lower_op(&F, op, refs);
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
    free(F.block_off);
    free(F.fix);
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

static bool serialize_global_value_attr(MLIR_AttributeHandle va, string name,
                                        Buf *data, uint32_t off, size_t sz) {
    MLIR_AttrKind vk = MLIR_GetAttributeKind(va);
    if (vk == MLIR_ATTR_KIND_STRING) {
        string s = MLIR_GetAttributeString(va);
        if (s.size > sz) s.size = sz;
        memcpy(data->data + off, s.str, s.size);
    } else if (vk == MLIR_ATTR_KIND_INTEGER || vk == MLIR_ATTR_KIND_BOOL) {
        uint64_t v = (uint64_t)MLIR_GetAttributeInteger(va);
        for (size_t b = 0; b < sz; b++)
            data->data[off + b] = (uint8_t)(b < 8 ? (v >> (8 * b)) : 0);
    } else if (vk == MLIR_ATTR_KIND_FLOAT) {
        double d = MLIR_GetAttributeFloat(va);
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

static bool parse_trivial_ptr_global_init(MLIR_OpHandle op, string name, uint32_t off,
                                          size_t sz, PendingPtrReloc **pend,
                                          size_t *pend_cap, size_t *n_pend) {
    if (MLIR_GetOpNumRegions(op) == 0 ||
        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) == 0)
        return true;

    MLIR_BlockHandle blk = MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0);
    size_t nb = MLIR_GetBlockNumOps(blk);
    string target = (string){0};
    bool only_trivial = true;

    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_OpHandle bop = MLIR_GetBlockOp(blk, bi);
        string opn = MLIR_GetOpName(bop);
        if (str_eqz(opn, "llvm.mlir.addressof")) {
            MLIR_AttributeHandle ga =
                MLIR_GetOpAttributeByName(bop, "global_name");
            if (ga == MLIR_INVALID_HANDLE)
                ga = MLIR_GetOpAttributeByName(bop, "value");
            if (ga == MLIR_INVALID_HANDLE) {
                fprintf(stderr, "x64: addressof global missing name in '%.*s'\n",
                        (int)name.size, name.str);
                g_x64_ok = false;
                return false;
            }
            target = strip_at(MLIR_GetAttributeString(ga));
        } else if (!str_eqz(opn, "llvm.mlir.zero") &&
                   !str_eqz(opn, "llvm.mlir.undef") &&
                   !str_eqz(opn, "llvm.return")) {
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
static bool x64_is_global_op(MLIR_OpHandle op) {
    if (MLIR_GetOpType(op) == OP_TYPE_LLVM_MLIR_GLOBAL) return true;
    string nm = MLIR_GetOpName(op);
    if (nm.size == 17 && memcmp(nm.str, "llvm.mlir.global", 17) == 0) return true;
    if (MLIR_GetOpAttributeByName(op, "global_type") == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpAttributeByName(op, "sym_name") == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpAttributeByName(op, "function_type") != MLIR_INVALID_HANDLE) return false;
    return true;
}

// Bare global name from sym_name (upstream stores `@foo`; strip the `@`).
static string x64_global_sym_name(MLIR_OpHandle op) {
    return strip_at(attr_str(op, "sym_name"));
}

// True if `op` is llvm.func (in-tree or upstream FuncOp form).
static bool x64_is_func_op(MLIR_OpHandle op) {
    if (MLIR_GetOpType(op) == OP_TYPE_LLVM_FUNC) return true;
    string nm = MLIR_GetOpName(op);
    return nm.size == 9 && memcmp(nm.str, "llvm.func", 9) == 0;
}

// True if the module has a function named `name` (defined when require_body).
static bool x64_module_has_func(MLIR_BlockHandle mb, const char *name, bool require_body) {
    size_t no = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!x64_is_func_op(op)) continue;
        if (!str_eqz(attr_str(op, "sym_name"), name)) continue;
        if (!require_body) return true;
        if (MLIR_GetOpNumRegions(op) == 0) return false;
        return MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
    }
    return false;
}

// True if the module declares a global named `gn`.
static bool x64_module_has_global_sym(MLIR_BlockHandle mb, string gn) {
    if (!gn.size) return false;
    size_t no = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!x64_is_global_op(op)) continue;
        if (str_eq(x64_global_sym_name(op), gn)) return true;
    }
    return false;
}

// True for --from-wasm modules (wasm globals, wasi_start, or spliced host init).
static bool x64_detect_wasm_lifted(MLIR_BlockHandle mb, GMap *gm) {
    uint32_t dummy;
    if (gmap_get_cstr(gm, X64_WASM_LINMEM_BASE, &dummy, NULL) ||
        gmap_get_cstr(gm, X64_WASM_LINMEM, &dummy, NULL))
        return true;
    if (x64_module_has_global_sym(mb,
            str_from_cstr_view((char *)X64_WASM_LINMEM_BASE)) ||
        x64_module_has_global_sym(mb,
            str_from_cstr_view((char *)X64_WASM_LINMEM)) ||
        x64_module_has_global_sym(mb,
            str_from_cstr_view((char *)"__wasm_argc")) ||
        x64_module_has_global_sym(mb,
            str_from_cstr_view((char *)"__wasm_g0")))
        return true;
    // Driver splices platform_linux.c as __host_platform_init; wasm-side
    // platform_init is often DCE'd before native _start references it.
    if (x64_module_has_func(mb, "__host_platform_init", true))
        return true;
    return x64_module_has_func(mb, "wasi_start", false);
}

// Allocate one global in .data, serialize its initializer, and register it in gm.
static void x64_register_global_op(MLIR_Context *ctx, MLIR_OpHandle op, GMap *gm, Buf *data,
                                   PendingPtrReloc **pend, size_t *pend_cap, size_t *n_pend) {
    string name = x64_global_sym_name(op);
    if (!name.size) {
        fprintf(stderr, "x64: llvm.mlir.global missing sym_name\n");
        g_x64_ok = false;
        return;
    }
    uint32_t dummy;
    if (gmap_get(gm, name, &dummy)) return;

    MLIR_AttributeHandle gt = MLIR_GetOpAttributeByName(op, "global_type");
    size_t sz = 0;
    if (gt) {
        MLIR_TypeHandle t = MLIR_GetAttributeTypeValue(gt);
        if (t != MLIR_INVALID_HANDLE) sz = x64_type_size(ctx, t);
    }
    if (sz == 0) sz = 8;
    uint32_t off = x64_data_begin_global(data, sz);

    MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
    if (va) serialize_global_value_attr(va, name, data, off, sz);
    else parse_trivial_ptr_global_init(op, name, off, sz, pend, pend_cap, n_pend);

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

// Lay out .data: one IR-global pass (no second x64_register_all_module_globals
// scan — x64_register_global_op dedupes via gmap_get), wasm scalars, ptr relocs.
static void x64_init_data_section(MLIR_Context *ctx, MLIR_BlockHandle mb, GMap *gm,
                                  Buf *data, bool *wasm_lifted_out,
                                  X64PtrReloc **relocs_out, size_t *n_relocs_out) {
    *relocs_out = NULL;
    *n_relocs_out = 0;
    size_t no = MLIR_GetBlockNumOps(mb);
    size_t pend_cap = 0, n_pend = 0;
    PendingPtrReloc *pend = NULL;

    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!x64_is_global_op(op)) continue;
        x64_register_global_op(ctx, op, gm, data, &pend, &pend_cap, &n_pend);
    }
    *wasm_lifted_out = x64_detect_wasm_lifted(mb, gm);
    if (*wasm_lifted_out)
        x64_ensure_wasm_scalars(gm, data);
    resolve_pending_ptr_globals(gm, pend, n_pend, relocs_out, n_relocs_out);
    free(pend);
}

// Zero-fill a wasm data global when DCE dropped the IR op but addressof remains.
static bool x64_synth_wasm_global(FnCtx *F, string gn, uint32_t *out_off) {
    if (!x64_is_wasm_data_sym(gn) || !F->data_buf) return false;
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
    size_t no = MLIR_GetBlockNumOps(F->module_mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(F->module_mb, i);
        if (!x64_is_global_op(op)) continue;
        if (!str_eq(x64_global_sym_name(op), gn)) continue;
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

// Pick app_main, main, wasi_start, or _start as the ELF user entry symbol.
static bool x64_resolve_entry(MLIR_BlockHandle mb, bool wasm_lifted, char **out_cstr) {
    static const char *order[] = { "app_main", "main", "wasi_start", "_start" };
    (void)wasm_lifted;
    for (size_t i = 0; i < sizeof(order)/sizeof(order[0]); i++) {
        if (x64_module_has_func(mb, order[i], false)) {
            *out_cstr = heapdup_cstr(order[i]);
            return true;
        }
    }
    return false;
}

// Synthesize _start: wasm linmem/argc setup, ptr inits, init+main, exit syscall.
static void synth_start(ObjModule *m, GMap *gm, X64PtrReloc *relocs, size_t n_relocs,
                        bool wasm_lifted, const char *entry_sym) {
    ObjFunc *f = obj_add_func(m, heapdup_cstr("_start"), true);
    FnCtx F; memset(&F,0,sizeof(F)); F.f = f; F.ok = true;

    uint32_t argc_off = 0, argv_off = 0, linmem_base_off = 0;
    uint32_t template_off = 0, template_size = 0;
    bool has_argc = gmap_get_cstr(gm, X64_WASM_ARGC, &argc_off, NULL) &&
                    gmap_get_cstr(gm, X64_WASM_ARGV, &argv_off, NULL);
    bool has_linmem_base = gmap_get_cstr(gm, X64_WASM_LINMEM_BASE, &linmem_base_off, NULL);
    bool has_template = gmap_get_cstr(gm, X64_WASM_LINMEM, &template_off, &template_size);

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
        mov_ri(&F, RDI, 0, 1);
        mov_ri(&F, RSI, (int64_t)(1ULL << 32), 1);
        mov_ri(&F, RDX, 3, 1);
        mov_ri(&F, R10, 0x22, 1);
        mov_ri(&F, R8, -1, 1);
        mov_ri(&F, R9, 0, 1);
        mov_ri(&F, RAX, NR_MMAP, 0);
        eb(&F, 0x0F); eb(&F, 0x05);
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
        eb(&F, 0xE8); uint32_t fld_init = here(&F); e32(&F, 0);
        obj_add_reloc(f, fld_init, fld_init+4, false, heapdup_cstr("platform_init"), 0);
        eb(&F, 0xE8); uint32_t fld_main = here(&F); e32(&F, 0);
        obj_add_reloc(f, fld_main, fld_main+4, false, heapdup_cstr(entry_sym), 0);
    }
    mov_rr(&F, RDI, RAX, 0);
    mov_ri(&F, RAX, NR_EXIT, 0);
    eb(&F, 0x0F); eb(&F, 0x05);
}

// Synthesize a stub for platform_init, __assert_fail, or __tinyc_syscall6/syscall.
static bool synth_thunk(ObjModule *m, string name) {
    // Lifted wasm still calls platform_init; host splice provides __host_platform_init.
    if (str_eqz(name, "platform_init")) {
        char *fn = (char *)malloc(name.size + 1);
        memcpy(fn, name.str, name.size);
        fn[name.size] = '\0';
        ObjFunc *f = obj_add_func(m, fn, false);
        FnCtx F; memset(&F,0,sizeof(F)); F.f = f;
        eb(&F, 0xE8); uint32_t fld = here(&F); e32(&F, 0);
        obj_add_reloc(f, fld, fld + 4, false, heapdup_cstr("__host_platform_init"), 0);
        eb(&F, 0xC3);
        return true;
    }
    // assert() from spliced host platform code; assert.c is GC'd out of the
    // linked wasm module so lift never defines this symbol.
    if (str_eqz(name, "__assert_fail")) {
        char *fn = (char *)malloc(name.size + 1);
        memcpy(fn, name.str, name.size);
        fn[name.size] = '\0';
        ObjFunc *f = obj_add_func(m, fn, false);
        FnCtx F; memset(&F,0,sizeof(F)); F.f = f;
        mov_ri(&F, RDI, 1, 0);
        eb(&F, 0xE8); uint32_t fld = here(&F); e32(&F, 0);
        obj_add_reloc(f, fld, fld + 4, false, heapdup_cstr("__host_platform_exit"), 0);
        eb(&F, 0xC3);
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
                      uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;
    g_x64_ok = true;

    MLIR_RegionHandle mreg = MLIR_GetOpRegion(llvm_module, 0);
    MLIR_BlockHandle  mb   = MLIR_GetRegionBlock(mreg, 0);
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
    SymSet reach; memset(&reach, 0, sizeof(reach));
    bool ok = false;

    bool wasm_lifted = false;
    x64_init_data_section(ctx, mb, &gm, &data, &wasm_lifted,
                          &ptr_relocs, &n_ptr_relocs);
    g_x64_gmap = &gm;

    if (!x64_resolve_entry(mb, wasm_lifted, &entry_cstr)) {
        fprintf(stderr,
                "x64: no defined entry (app_main/main/wasi_start/_start)\n");
        g_x64_ok = false;
        goto x64_done;
    }

    // index defined functions by name
    size_t no = MLIR_GetBlockNumOps(mb);

    synth_start(&m, &gm, ptr_relocs, n_ptr_relocs, wasm_lifted, entry_cstr);
    m.start_idx = 0;

    // Reachability from crt0 roots trims unused runtime.
    if (wasm_lifted) {
        sym_add(&reach, str_from_cstr_view((char *)"__host_platform_init"));
        sym_add(&reach, str_from_cstr_view(entry_cstr));
    } else {
        sym_add(&reach, str_from_cstr_view((char *)"platform_init"));
        sym_add(&reach, str_from_cstr_view(entry_cstr));
    }
    bool grew = true;
    while (grew) {
        grew = false;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle fn = MLIR_GetBlockOp(mb, i);
            if (!str_eqz(MLIR_GetOpName(fn), "llvm.func")) continue;
            if (MLIR_GetOpNumRegions(fn) == 0 || MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn,0)) == 0) continue;
            string fname = attr_str(fn, "sym_name");
            bool wanted = false;
            for (size_t k = 0; k < reach.c; k++) if (str_eq(fname, reach.n[k])) { wanted = true; break; }
            if (!wanted) continue;
            // scan calls
            MLIR_RegionHandle r = MLIR_GetOpRegion(fn, 0);
            size_t nbk = MLIR_GetRegionNumBlocks(r);
            for (size_t b = 0; b < nbk; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(r, b);
                size_t nop = MLIR_GetBlockNumOps(bk);
                for (size_t o = 0; o < nop; o++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(bk, o);
                    if (str_eqz(MLIR_GetOpName(op), "llvm.call") || str_eqz(MLIR_GetOpName(op), "func.call")) {
                        string cal = strip_at(attr_str(op, "callee"));
                        if (!cal.size) cal = strip_at(attr_str(op, "callee_name"));
                        if (cal.size) {
                            if (x64_is_wasm_data_sym(cal)) continue;
                            bool known = false;
                            for (size_t k = 0; k < reach.c; k++) if (str_eq(cal, reach.n[k])) { known = true; break; }
                            if (!known) { sym_add(&reach, cal); grew = true; }
                        }
                    } else if (str_eqz(MLIR_GetOpName(op), "llvm.mlir.addressof")) {
                        MLIR_AttributeHandle ga = MLIR_GetOpAttributeByName(op, "global_name");
                        if (ga == MLIR_INVALID_HANDLE) ga = MLIR_GetOpAttributeByName(op, "value");
                        string gn = strip_at(ga ? MLIR_GetAttributeString(ga) : (string){0});
                        if (!gn.size) continue;
                        uint32_t dummy;
                        if (x64_is_wasm_data_sym(gn)) continue;
                        if (gmap_get(&gm, gn, &dummy)) continue;   // data global
                        if (x64_module_has_global_sym(mb, gn)) continue;
                        bool known = false;
                        for (size_t k = 0; k < reach.c; k++) if (str_eq(gn, reach.n[k])) { known = true; break; }
                        if (!known) { sym_add(&reach, gn); grew = true; }
                    }
                }
            }
        }
    }

    // emit reachable defined functions
    bool saw_entry = false;
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(mb, i);
        if (!str_eqz(MLIR_GetOpName(fn), "llvm.func")) continue;
        if (MLIR_GetOpNumRegions(fn) == 0 || MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn,0)) == 0) continue;
        string fname = attr_str(fn, "sym_name");
        bool wanted = false;
        for (size_t k = 0; k < reach.c; k++) if (str_eq(fname, reach.n[k])) { wanted = true; break; }
        if (!wanted) continue;
        if (str_eqz(fname, entry_cstr)) saw_entry = true;
        emit_func(ctx, fn, &m, &refs);
    }

    if (!saw_entry) {
        fprintf(stderr, "x64: no defined entry '%s'\n", entry_cstr);
        g_x64_ok = false;
    }

    // synth thunks for referenced-but-undefined symbols
    for (size_t i = 0; i < refs.c && g_x64_ok; i++) {
        bool defined = false;
        for (size_t j = 0; j < m.n_funcs; j++)
            if (str_eq(str_from_cstr_view(m.funcs[j].name), refs.n[i])) { defined = true; break; }
        if (defined) continue;
        if (!synth_thunk(&m, refs.n[i])) {
            fprintf(stderr, "x64: unresolved symbol '%.*s'\n",
                    (int)refs.n[i].size, refs.n[i].str);
            g_x64_ok = false;
        }
    }

    // synth_start records platform_init directly on _start (not via refs). If
    // wasm-side platform_init was DCE'd, forward to __host_platform_init.
    if (g_x64_ok && x64_obj_needs_func(&m, "platform_init") &&
            !x64_obj_has_func(&m, "platform_init"))
        synth_thunk(&m, str_from_cstr_view("platform_init"));

    ok = g_x64_ok;
    if (ok) {
        m.data = data.data; m.data_len = data.len;
        ok = mlir_obj_to_elf(&m, out_data, out_size);
        m.data = NULL;  // owned by `data`
    }

    // Free module state only after ELF link (relocs may still reference gm/data).
x64_done:
    obj_module_free(&m);
    free(data.data);
    free(entry_cstr);
    for (size_t i = 0; i < gm.n; i++) free(gm.name[i]);
    free(gm.name); free(gm.off); free(gm.sz);
    free(refs.n);
    free(reach.n);
    free(ptr_relocs);
    g_x64_gmap = NULL;
    g_x64_mb = MLIR_INVALID_HANDLE;
    g_x64_data_buf = NULL;
    return ok;
}
