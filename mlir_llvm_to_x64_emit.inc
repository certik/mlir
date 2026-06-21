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

// width in bytes of a scalar value (int/ptr/float)
static int type_bytes(MLIR_Context *ctx, MLIR_ValueHandle v) {
    int fw = fp_width(ctx, v);
    if (fw == 32) return 4;
    if (fw == 64) return 8;
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

// module-wide lowering state shared across the helpers below
static GMap *g_x64_gmap = NULL;
static MLIR_BlockHandle g_x64_mod_mb = MLIR_INVALID_HANDLE;
static bool  g_x64_ok = true;

// Look up an `llvm.func` declaration/definition by symbol name in the module body.
static MLIR_OpHandle find_llvm_func(string name) {
    if (g_x64_mod_mb == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    size_t no = MLIR_GetBlockNumOps(g_x64_mod_mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(g_x64_mod_mb, i);
        if (!str_eqz(MLIR_GetOpName(op), "llvm.func")) continue;
        string sn = attr_str(op, "sym_name");
        if (sn.size == name.size && memcmp(sn.str, name.str, name.size) == 0) return op;
    }
    return MLIR_INVALID_HANDLE;
}

static bool llvm_func_has_body(MLIR_OpHandle fn) {
    return MLIR_GetOpNumRegions(fn) > 0 &&
           MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn, 0)) > 0;
}

static MLIR_TypeHandle call_param_type(MLIR_Context *ctx, MLIR_OpHandle decl,
                                       size_t idx, MLIR_ValueHandle fallback) {
    if (decl != MLIR_INVALID_HANDLE) {
        MLIR_AttributeHandle fta = MLIR_GetOpAttributeByName(decl, "function_type");
        if (fta != MLIR_INVALID_HANDLE) {
            MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(fta);
            if (idx < MLIR_GetTypeFunctionNumInputs(fty))
                return MLIR_GetTypeFunctionInput(fty, idx);
        }
    }
    return MLIR_GetValueType(fallback);
}

// Inline tinyc_va_arg_* (char **ap): p=*ap; v=*(T*)p; *ap=p+8; return v.
// Inlined so cursor updates are guaranteed 64-bit stores on LP64.
static void emit_va_arg_i32(FnCtx *F, MLIR_OpHandle op) {
    MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
    load_val(F, MLIR_GetOpOperand(op, 0), RDI);
    load_mem(F, R11, RDI, 8, false);
    load_mem(F, RAX, R11, 4, false);
    add_imm_reg(F, R11, 8, 1);
    store_mem(F, RDI, R11, 8);
    store_val(F, r, RAX);
}
static void emit_va_arg_i64(FnCtx *F, MLIR_OpHandle op) {
    MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
    load_val(F, MLIR_GetOpOperand(op, 0), RDI);
    load_mem(F, R11, RDI, 8, false);
    load_mem(F, RAX, R11, 8, false);
    add_imm_reg(F, R11, 8, 1);
    store_mem(F, RDI, R11, 8);
    store_val(F, r, RAX);
}
static void emit_va_arg_ptr(FnCtx *F, MLIR_OpHandle op) {
    emit_va_arg_i64(F, op);
}
static void emit_va_arg_f64(FnCtx *F, MLIR_OpHandle op) {
    MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
    load_val(F, MLIR_GetOpOperand(op, 0), RDI);
    load_mem(F, R11, RDI, 8, false);
    load_mem(F, RAX, R11, 8, false);
    movq_xmm_gp(F, XMM0, RAX);
    add_imm_reg(F, R11, 8, 1);
    store_mem(F, RDI, R11, 8);
    fin_fp(F, r, XMM0);
}

// SysV AMD64 call with separate integer (RDI..R9) and SSE (XMM0..XMM7) arg slots.
static void emit_sysv_call(FnCtx *F, MLIR_OpHandle op, string callee, SymSet *refs,
                           bool is_indirect) {
    MLIR_Context *ctx = F->ctx;
    size_t arg_base = is_indirect ? 1 : 0;
    size_t no = MLIR_GetOpNumOperands(op) - arg_base;
    if (no > 16) { fail(F, "too many call args", callee); return; }
    if (is_indirect)
        load_val(F, MLIR_GetOpOperand(op, 0), R11);
    MLIR_OpHandle decl = is_indirect ? MLIR_INVALID_HANDLE : find_llvm_func(callee);
    MLIR_TypeHandle pty[16];
    SysVArg loc[16];
    for (size_t i = 0; i < no; i++)
        pty[i] = call_param_type(ctx, decl, i, MLIR_GetOpOperand(op, arg_base + i));
    bool is_variadic = false;
    size_t n_fixed = 0;
    MLIR_AttributeHandle vct = MLIR_GetOpAttributeByName(op, "var_callee_type");
    if (vct != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle ft = MLIR_GetAttributeTypeValue(vct);
        if (MLIR_GetTypeFunctionIsVarArg(ft)) {
            is_variadic = true;
            n_fixed = MLIR_GetTypeFunctionNumInputs(ft);
            if (n_fixed > no) { fail(F, "variadic call missing args", callee); return; }
        }
    }
    size_t nstack = is_variadic
        ? sysv_vararg_layout(ctx, pty, no, n_fixed, loc)
        : sysv_layout(ctx, pty, no, loc);
    size_t pad = nstack & 1;
    if (pad) { eb(F, 0x48); eb(F, 0x83); modrm(F, 3, 5, RSP); eb(F, 0x08); }
    for (size_t i = no; i > 0; i--) {
        size_t k = i - 1;
        if (!loc[k].stack) continue;
        if (loc[k].sse || (is_variadic && fp_width_ty(ctx, pty[k]) != 0)) {
            use_fp(F, MLIR_GetOpOperand(op, arg_base + k), XMM0);
            if (fp_width_ty(ctx, pty[k]) == 32) movd_gp_xmm(F, RAX, XMM0);
            else                               movq_gp_xmm(F, RAX, XMM0);
        } else {
            load_val(F, MLIR_GetOpOperand(op, arg_base + k), RAX);
        }
        eb(F, 0x50);
    }
    for (size_t i = 0; i < no; i++) {
        if (loc[i].stack) continue;
        if (loc[i].sse) use_fp(F, MLIR_GetOpOperand(op, arg_base + i), (uint8_t)loc[i].idx);
        else            load_val(F, MLIR_GetOpOperand(op, arg_base + i), ARG_REGS[loc[i].idx]);
    }
    if (is_indirect) {
        rex(F, 0, 0, 0, R11);
        eb(F, 0xFF);
        modrm(F, 3, 2, R11);   // call *r11
    } else {
        eb(F, 0xE8);
        uint32_t field = here(F); e32(F, 0);
        char *cs = (char *)malloc(callee.size + 1);
        memcpy(cs, callee.str, callee.size); cs[callee.size] = 0;
        obj_add_reloc(F->f, field, field + 4, false, cs, 0);
        sym_add(refs, callee);
    }
    if (nstack) {
        uint32_t cleanup = (uint32_t)((nstack + pad) * 8);
        rex(F, 1, 0, 0, RSP); eb(F, 0x81); modrm(F, 3, 0, RSP); e32(F, cleanup);
    }
    if (MLIR_GetOpNumResults(op) == 1) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        if (fp_width(ctx, r) != 0) fin_fp(F, r, XMM0);
        else                       store_val(F, r, RAX);
    }
}

// Home incoming SysV args into the entry block's block-arg slots/registers.
static void home_entry_params(FnCtx *F, MLIR_OpHandle fn, MLIR_RegionHandle reg) {
    size_t nb = MLIR_GetRegionNumBlocks(reg);
    if (nb == 0) return;
    MLIR_BlockHandle e0 = MLIR_GetRegionBlock(reg, 0);
    size_t na = MLIR_GetBlockNumArgs(e0);
    if (na == 0) return;

    MLIR_Context *ctx = F->ctx;
    MLIR_AttributeHandle fta = MLIR_GetOpAttributeByName(fn, "function_type");
    MLIR_TypeHandle fty = fta ? MLIR_GetAttributeTypeValue(fta) : MLIR_INVALID_HANDLE;
    MLIR_TypeHandle pty[16];
    SysVArg loc[16];
    if (na > 16) { fail(F, "too many params", (string){0}); return; }
    for (size_t a = 0; a < na; a++) {
        MLIR_ValueHandle barg = MLIR_GetBlockArg(e0, a);
        pty[a] = (fty != MLIR_INVALID_HANDLE && a < MLIR_GetTypeFunctionNumInputs(fty))
                 ? MLIR_GetTypeFunctionInput(fty, a)
                 : MLIR_GetValueType(barg);
    }
    sysv_layout(ctx, pty, na, loc);
    for (size_t a = 0; a < na; a++) {
        MLIR_ValueHandle barg = MLIR_GetBlockArg(e0, a);
        if (loc[a].sse) {
            if (!loc[a].stack) fin_fp(F, barg, (uint8_t)loc[a].idx);
            else {
                int32_t off = sysv_stack_off(loc[a].idx);
                if (fp_width_ty(ctx, pty[a]) == 32) fp_movss_rbp(F, XMM0, off, false);
                else                                fp_movsd_rbp(F, XMM0, off, false);
                fin_fp(F, barg, XMM0);
            }
        } else if (!loc[a].stack) {
            store_val(F, barg, ARG_REGS[loc[a].idx]);
        } else {
            load_rbp(F, RAX, sysv_stack_off(loc[a].idx), 1);
            store_val(F, barg, RAX);
        }
    }
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
    if (str_eqz(nm, "llvm.intr.vastart")) {
        if (!F->is_vararg) { fail(F, "vastart in non-variadic function", (string){0}); return; }
        // Stack-based va_list: cursor points at the first variadic stack slot
        // (after any fixed args that arrived on the stack). tinyc_va_arg_* walk
        // 8-byte slots from here — same model as the aarch64 Mach-O backend.
        load_val(F, MLIR_GetOpOperand(op, 0), RCX);
        lea_rbp_disp(F, RAX, 16 + (int32_t)F->va_stack_skip);
        store_mem(F, RCX, RAX, 8);
        return;
    }
    if (str_eqz(nm, "llvm.intr.vaend")) {
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
                load_val(F, MLIR_GetOpOperand(op, op_idx++), RCX);
                // LLVM GEP indices are i32; on LP64 pointer math is 64-bit. Without
                // sign-extension, e.g. index -1 (p--) leaves garbage in rcx[63:32]
                // and the scaled offset points past the object (segfault on *p).
                movsxd_rr(F, RCX, RCX);
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
    if (str_eqz(nm, "llvm.fadd") || str_eqz(nm, "llvm.fsub") ||
        str_eqz(nm, "llvm.fmul") || str_eqz(nm, "llvm.fdiv")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int fw = fp_width(ctx, r);
        const char *kind = str_eqz(nm, "llvm.fadd") ? "fadd"
                         : str_eqz(nm, "llvm.fsub") ? "fsub"
                         : str_eqz(nm, "llvm.fmul") ? "fmul" : "fdiv";
        if (fw == 0) { fail(F, "fp binop non-float result", nm); return; }
        use_fp(F, MLIR_GetOpOperand(op, 0), XMM0);
        use_fp(F, MLIR_GetOpOperand(op, 1), XMM1);
        if (!F->ok) return;
        emit_fp_binop(F, kind, fw, XMM0, XMM0, XMM1);
        fin_fp(F, r, XMM0);
        return;
    }
    if (str_eqz(nm, "llvm.fneg")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int fw = fp_width(ctx, r);
        if (fw == 0) { fail(F, "fneg non-float", nm); return; }
        use_fp(F, MLIR_GetOpOperand(op, 0), XMM0);
        if (!F->ok) return;
        emit_fp_unop(F, "fneg", fw, XMM0, XMM0);
        fin_fp(F, r, XMM0);
        return;
    }
    if (str_eqz(nm, "llvm.fcmp")) {
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE) { fail(F, "fcmp missing predicate", nm); return; }
        int cond = fcmp_pred_to_cond(MLIR_GetAttributeInteger(pa));
        if (cond < 0) { fail(F, "unsupported fcmp predicate", nm); return; }
        int fw = fp_width(ctx, MLIR_GetOpOperand(op, 0));
        if (fw == 0) { fail(F, "fcmp non-float operand", nm); return; }
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        use_fp(F, MLIR_GetOpOperand(op, 0), XMM0);
        use_fp(F, MLIR_GetOpOperand(op, 1), XMM1);
        if (!F->ok) return;
        emit_fcmp(F, fw, XMM0, XMM1);
        setcc_al(F, cond);
        store_val(F, r, RAX);
        return;
    }
    if (str_eqz(nm, "llvm.fptosi") || str_eqz(nm, "llvm.fptoui")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        bool sign = str_eqz(nm, "llvm.fptosi");
        int src_w = fp_width(ctx, MLIR_GetOpOperand(op, 0));
        if (src_w == 0) { fail(F, "fptosi/fptoui non-float source", nm); return; }
        int rw = int_bits(ctx, r);
        int dst_w = rw > 32 ? 64 : 32;
        use_fp(F, MLIR_GetOpOperand(op, 0), XMM0);
        if (!F->ok) return;
        emit_fp_cvt(F, "f2i", src_w, dst_w, sign, RAX, XMM0);
        if (rw > 0 && rw < 32)
            eb(F, 0x25), e32(F, (uint32_t)((1u << rw) - 1u));
        store_val(F, r, RAX);
        return;
    }
    if (str_eqz(nm, "llvm.sitofp") || str_eqz(nm, "llvm.uitofp")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        bool sign = str_eqz(nm, "llvm.sitofp");
        int dst_w = fp_width(ctx, r);
        if (dst_w == 0) { fail(F, "sitofp/uitofp non-float result", nm); return; }
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);
        emit_fp_cvt(F, "i2f", 64, dst_w, sign, XMM0, RAX);
        fin_fp(F, r, XMM0);
        return;
    }
    if (str_eqz(nm, "llvm.fpext") || str_eqz(nm, "llvm.fptrunc")) {
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        int src_w = fp_width(ctx, MLIR_GetOpOperand(op, 0));
        int dst_w = fp_width(ctx, r);
        if (src_w == 0 || dst_w == 0) { fail(F, "fpext/fptrunc non-float", nm); return; }
        use_fp(F, MLIR_GetOpOperand(op, 0), XMM0);
        if (!F->ok) return;
        if ((src_w == 32 && dst_w == 64) || (src_w == 64 && dst_w == 32))
            emit_fp_cvt(F, "f2f", src_w, dst_w, false, XMM0, XMM0);
        else { fail(F, "unsupported f2f conversion", nm); return; }
        fin_fp(F, r, XMM0);
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
                eb(F, 0xE8);
                uint32_t field = here(F); e32(F, 0);
                char *cs = (char *)malloc(callee.size + 1);
                memcpy(cs, callee.str, callee.size); cs[callee.size] = 0;
                obj_add_reloc(F->f, field, field + 4, false, cs, 0);
                sym_add(refs, callee);
                eb(F, 0x48); eb(F, 0x83); modrm(F, 3, 0, RSP); eb(F, 0x08);  // add rsp, 8
                if (MLIR_GetOpNumResults(op) == 1)
                    store_val(F, MLIR_GetOpResult(op, 0), RAX);
                return;
            }
            if (str_eqz(callee, "tinyc_va_arg_i32") && no == 1) { emit_va_arg_i32(F, op); return; }
            if (str_eqz(callee, "tinyc_va_arg_i64") && no == 1) { emit_va_arg_i64(F, op); return; }
            if (str_eqz(callee, "tinyc_va_arg_ptr") && no == 1) { emit_va_arg_ptr(F, op); return; }
            if (str_eqz(callee, "tinyc_va_arg_f64") && no == 1) { emit_va_arg_f64(F, op); return; }
        }
        emit_sysv_call(F, op, callee, refs, is_indirect);
        return;
    }

    fail(F, "unsupported op", nm);
}

// emit a terminator (also collects callee refs via lower_op for calls? no -
// terminators here are br/cond_br/return/unreachable only)
static void lower_term(FnCtx *F, MLIR_OpHandle op) {
    string nm = MLIR_GetOpName(op);
    MLIR_RegionHandle reg = MLIR_GetBlockParentRegion(MLIR_GetOpParentBlock(op));
    if (str_eqz(nm, "llvm.return") || str_eqz(nm, "func.return")) {
        if (MLIR_GetOpNumOperands(op) >= 1) {
            MLIR_ValueHandle rv = MLIR_GetOpOperand(op, 0);
            if (fp_width(F->ctx, rv) != 0) use_fp(F, rv, XMM0);
            else                          load_val(F, rv, RAX);
        }
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
        // Operand-carrying edges get a landing pad (edge copies, then jmp to the
        // real successor), matching llvm->aarch64: copies run only on the taken
        // path. Operand-free edges branch straight to the successor block.
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
    F.refs = refs;

    MLIR_AttributeHandle fta_var = MLIR_GetOpAttributeByName(fn, "function_type");
    if (fta_var) {
        MLIR_TypeHandle fty_v = MLIR_GetAttributeTypeValue(fta_var);
        if (MLIR_GetTypeFunctionIsVarArg(fty_v)) {
            F.is_vararg = true;
            size_t nf = MLIR_GetTypeFunctionNumInputs(fty_v);
            MLIR_TypeHandle pty_v[16];
            size_t nn = nf < 16 ? nf : 16;
            for (size_t i = 0; i < nn; i++)
                pty_v[i] = MLIR_GetTypeFunctionInput(fty_v, i);
            F.va_named_gp = sysv_named_gp(ctx, pty_v, nn);
            SysVArg loc_v[16];
            F.va_stack_skip = (uint32_t)(sysv_layout(ctx, pty_v, nn, loc_v) * 8u);
        }
    }

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
    if (F.is_vararg) {
        F.va_save_base = cur + 8;
        cur += VA_REG_SAVE_BYTES;
    }
    F.frame = (uint32_t)((cur + 15) & ~15);

    // --- prologue: push rbp ; mov rbp,rsp ; sub rsp, frame ---
    eb(&F, 0x55);
    rex(&F, 1, 0, 0, RBP); eb(&F, 0x89); modrm(&F, 3, RSP, RBP);    // mov rbp, rsp
    if (F.frame) { rex(&F, 1, 0, 0, RSP); eb(&F, 0x81); modrm(&F, 3, 5, RSP); e32(&F, F.frame); } // sub rsp, imm32

    // save the callee-saved registers the allocator used
    for (int j = 0; j < F.n_cs; j++)
        store_rbp(&F, F.cs_base + 8 * (j + 1), F.cs_reg[j], 1);

    if (F.is_vararg)
        spill_va_save(&F);
    
    // --- entry params: incoming SysV arg regs -> entry block-arg homes -------
    home_entry_params(&F, fn, reg);

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

static void gmap_put(GMap *g, string nm, uint32_t off) {
    if (g->n == g->cap) { g->cap = g->cap ? g->cap*2 : 8; g->name = realloc(g->name, g->cap*sizeof(char*)); g->off = realloc(g->off, g->cap*sizeof(uint32_t)); }
    g->name[g->n] = (char*)malloc(nm.size+1); memcpy(g->name[g->n], nm.str, nm.size); g->name[g->n][nm.size]=0;
    g->off[g->n] = off; g->n++;
}

typedef struct { uint32_t slot_off; uint32_t target_off; } X64PtrReloc;

// Collect llvm.mlir.global initializers into the data blob. Scalar/aggregate
// globals store their `value` attribute; pointer globals whose initializer
// region is just `addressof @other; return` become runtime stores in _start.
static void collect_globals(MLIR_Context *ctx, MLIR_BlockHandle mb, GMap *gm, Buf *data,
                            X64PtrReloc **relocs_out, size_t *n_relocs_out) {
    *relocs_out = NULL;
    *n_relocs_out = 0;
    size_t no = MLIR_GetBlockNumOps(mb);
    size_t pend_cap = 0, n_pend = 0;
    typedef struct { uint32_t slot_off; string target; } PendingPtrReloc;
    PendingPtrReloc *pend = NULL;

    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!str_eqz(MLIR_GetOpName(op), "llvm.mlir.global")) continue;
        string name = attr_str(op, "sym_name");
        uint32_t off = (uint32_t)data->len;
        MLIR_AttributeHandle gt = MLIR_GetOpAttributeByName(op, "global_type");
        size_t sz = 0;
        if (gt) {
            MLIR_TypeHandle t = MLIR_GetAttributeTypeValue(gt);
            if (t != MLIR_INVALID_HANDLE) sz = x64_type_size(ctx, t);
        }
        if (sz == 0) sz = 8;
        for (size_t z = 0; z < sz; z++) buf_u8(data, 0);

        MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
        if (va) {
            // Serialize the initializer per its actual attribute kind. Using the
            // string accessor for an integer/float/bool attribute would read an
            // unrelated field and emit garbage (or read an invalid pointer).
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
            }
        } else if (MLIR_GetOpNumRegions(op) > 0 &&
                   MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0) {
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
                    if (ga == MLIR_INVALID_HANDLE) {
                        fprintf(stderr, "x64: addressof global missing name in '%.*s'\n",
                                (int)name.size, name.str);
                        g_x64_ok = false;
                        break;
                    }
                    target = strip_at(MLIR_GetAttributeString(ga));
                } else if (!str_eqz(opn, "llvm.mlir.zero") &&
                           !str_eqz(opn, "llvm.mlir.undef") &&
                           !str_eqz(opn, "llvm.return")) {
                    only_trivial = false;
                }
            }
            if (target.size) {
                if (sz < 8) {
                    fprintf(stderr, "x64: pointer global '%.*s' is too small\n",
                            (int)name.size, name.str);
                    g_x64_ok = false;
                } else {
                    if (n_pend == pend_cap) {
                        pend_cap = pend_cap ? pend_cap * 2 : 4;
                        pend = (PendingPtrReloc *)realloc(pend,
                                                          pend_cap * sizeof(*pend));
                    }
                    pend[n_pend].slot_off = off;
                    pend[n_pend].target = target;
                    n_pend++;
                }
            } else if (!only_trivial) {
                fprintf(stderr, "x64: unsupported global initializer for '%.*s'\n",
                        (int)name.size, name.str);
                g_x64_ok = false;
            }
        }
        gmap_put(gm, name, off);
        while (data->len & 7) buf_u8(data, 0);   // 8-align next global
    }

    if (n_pend && g_x64_ok) {
        X64PtrReloc *relocs = (X64PtrReloc *)calloc(n_pend, sizeof(X64PtrReloc));
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
    free(pend);
}

static void synth_start(ObjModule *m, GMap *gm, X64PtrReloc *relocs, size_t n_relocs) {
    ObjFunc *f = obj_add_func(m, strdup("_start"), true);
    FnCtx F; memset(&F,0,sizeof(F)); F.f = f; F.gm = gm; F.ok = true;
    for (size_t k = 0; k < n_relocs; k++) {
        lea_rip_data(&F, R10, relocs[k].target_off);
        lea_rip_data(&F, R11, relocs[k].slot_off);
        store_mem(&F, R11, R10, 8);
    }
    eb(&F, 0xE8); uint32_t fld = here(&F); e32(&F, 0);          // call main
    obj_add_reloc(f, fld, fld+4, false, strdup("main"), 0);
    mov_rr(&F, RDI, RAX, 0);                                    // mov edi, eax
    mov_ri(&F, RAX, NR_EXIT, 0);                                // mov eax, 60
    eb(&F, 0x0F); eb(&F, 0x05);                                 // syscall
}

// synth a Linux syscall thunk for `name`; returns false if `name` is not a
// known syscall primitive.
static bool synth_thunk(ObjModule *m, const char *name) {
    // __tinyc_syscall6(num, a1..a6): the runtime syscall number is the first C
    // argument, so we cannot bake an NR_* constant — instead reshuffle the SysV
    // argument registers into the Linux x86_64 kernel convention (number in
    // rax; args in rdi/rsi/rdx/r10/r8/r9) and trap. a6 arrives on the stack.
    if (!strcmp(name, "__tinyc_syscall6")) {
        ObjFunc *f = obj_add_func(m, strdup(name), false);
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
    int nr; bool needs_r10 = false;
    if (!strcmp(name, "_write")) nr = NR_WRITE;
    else if (!strcmp(name, "_read")) nr = NR_READ;
    else if (!strcmp(name, "_exit")) nr = NR_EXIT;
    else if (!strcmp(name, "_mmap")) { nr = NR_MMAP; needs_r10 = true; }
    else return false;
    ObjFunc *f = obj_add_func(m, strdup(name), false);
    FnCtx F; memset(&F,0,sizeof(F)); F.f = f;
    if (needs_r10) { eb(&F,0x49); eb(&F,0x89); modrm(&F,3,RCX,R10); }  // mov r10, rcx
    mov_ri(&F, RAX, nr, 0);                                            // mov eax, nr
    eb(&F, 0x0F); eb(&F, 0x05);                                        // syscall
    eb(&F, 0xC3);                                                      // ret
    return true;
}

bool mlir_llvm_to_elf(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                      uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;
    g_x64_ok = true;

    MLIR_RegionHandle mreg = MLIR_GetOpRegion(llvm_module, 0);
    MLIR_BlockHandle  mb   = MLIR_GetRegionBlock(mreg, 0);

    // globals
    GMap gm; memset(&gm, 0, sizeof(gm));
    Buf data = {0};
    X64PtrReloc *ptr_relocs = NULL;
    size_t n_ptr_relocs = 0;
    collect_globals(ctx, mb, &gm, &data, &ptr_relocs, &n_ptr_relocs);
    g_x64_gmap = &gm;
    g_x64_mod_mb = mb;

    // index defined functions by name
    size_t no = MLIR_GetBlockNumOps(mb);
    SymSet refs; memset(&refs, 0, sizeof(refs));

    ObjModule m; memset(&m, 0, sizeof(m));
    synth_start(&m, &gm, ptr_relocs, n_ptr_relocs);   // index 0
    m.start_idx = 0;

    // reachability from main over direct-call edges (trims unused runtime)
    SymSet reach; memset(&reach, 0, sizeof(reach));
    sym_add(&reach, str_from_cstr_view("main"));
    bool grew = true;
    while (grew) {
        grew = false;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle fn = MLIR_GetBlockOp(mb, i);
            if (!str_eqz(MLIR_GetOpName(fn), "llvm.func")) continue;
            if (MLIR_GetOpNumRegions(fn) == 0 || MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn,0)) == 0) continue;
            string fname = attr_str(fn, "sym_name");
            bool wanted = false;
            for (size_t k = 0; k < reach.c; k++) if (str_eqz(fname, reach.n[k])) { wanted = true; break; }
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
                        if (cal.size) {
                            bool known = false;
                            for (size_t k = 0; k < reach.c; k++) if (str_eqz(cal, reach.n[k])) { known = true; break; }
                            if (!known) { sym_add(&reach, cal); grew = true; }
                        }
                    } else if (str_eqz(MLIR_GetOpName(op), "llvm.mlir.addressof")) {
                        // addressof @func (func.constant lowering): pull defined
                        // functions into the reach set even when only called
                        // indirectly (e.g. ptr_to_fnptr.tc).
                        MLIR_AttributeHandle ga = MLIR_GetOpAttributeByName(op, "global_name");
                        if (ga == MLIR_INVALID_HANDLE) ga = MLIR_GetOpAttributeByName(op, "value");
                        string gn = strip_at(ga ? MLIR_GetAttributeString(ga) : (string){0});
                        if (!gn.size) continue;
                        uint32_t dummy;
                        if (gmap_get(&gm, gn, &dummy)) continue;   // data global
                        MLIR_OpHandle tgt = find_llvm_func(gn);
                        if (tgt == MLIR_INVALID_HANDLE || !llvm_func_has_body(tgt)) continue;
                        bool known = false;
                        for (size_t k = 0; k < reach.c; k++) if (str_eqz(gn, reach.n[k])) { known = true; break; }
                        if (!known) { sym_add(&reach, gn); grew = true; }
                    }
                }
            }
        }
    }

    // emit reachable defined functions
    bool saw_main = false;
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(mb, i);
        if (!str_eqz(MLIR_GetOpName(fn), "llvm.func")) continue;
        if (MLIR_GetOpNumRegions(fn) == 0 || MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn,0)) == 0) continue;
        string fname = attr_str(fn, "sym_name");
        bool wanted = false;
        for (size_t k = 0; k < reach.c; k++) if (str_eqz(fname, reach.n[k])) { wanted = true; break; }
        if (!wanted) continue;
        if (str_eqz(fname, "main")) saw_main = true;
        emit_func(ctx, fn, &m, &refs);
    }

    if (!saw_main) { fprintf(stderr, "x64: no defined main\n"); g_x64_ok = false; }

    // synth thunks for referenced-but-undefined symbols
    for (size_t i = 0; i < refs.c && g_x64_ok; i++) {
        bool defined = false;
        for (size_t j = 0; j < m.n_funcs; j++) if (!strcmp(m.funcs[j].name, refs.n[i])) { defined = true; break; }
        if (defined) continue;
        if (!synth_thunk(&m, refs.n[i])) { fprintf(stderr, "x64: unresolved symbol '%s'\n", refs.n[i]); g_x64_ok = false; }
    }

    bool ok = g_x64_ok;
    if (ok) {
        m.data = data.data; m.data_len = data.len;
        ok = mlir_obj_to_elf(&m, out_data, out_size);
        m.data = NULL;  // owned by `data`
    }

    obj_module_free(&m);
    free(data.data);
    for (size_t i = 0; i < gm.n; i++) free(gm.name[i]);
    free(gm.name); free(gm.off);
    for (size_t i = 0; i < refs.c; i++) free(refs.n[i]);
    free(refs.n);
    for (size_t i = 0; i < reach.c; i++) free(reach.n[i]);
    free(reach.n);
    free(ptr_relocs);
    g_x64_gmap = NULL;
    g_x64_mod_mb = MLIR_INVALID_HANDLE;
    return ok;
}
