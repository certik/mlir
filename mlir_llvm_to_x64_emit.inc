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

// width in bytes of an integer/pointer-typed value
static int type_bytes(MLIR_Context *ctx, MLIR_ValueHandle v) {
    int b = int_bits(ctx, v);
    if (b == 0) return 8;           // pointer
    return (b + 7) / 8;
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
static bool  g_x64_ok = true;

// referenced external symbols (callees), for thunk synthesis + reachability
typedef struct { char **n; size_t c, cap; } SymSet;
static void sym_add(SymSet *s, string nm) {
    for (size_t i = 0; i < s->c; i++)
        if (strlen(s->n[i]) == nm.size && memcmp(s->n[i], nm.str, nm.size) == 0) return;
    if (s->c == s->cap) { s->cap = s->cap ? s->cap * 2 : 8; s->n = realloc(s->n, s->cap * sizeof(char*)); }
    s->n[s->c] = (char *)malloc(nm.size + 1);
    memcpy(s->n[s->c], nm.str, nm.size); s->n[s->c][nm.size] = 0; s->c++;
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
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);   // base
        size_t no = MLIR_GetOpNumOperands(op);
        for (size_t i = 1; i < no; i++) {             // byte-stride indices (i8 element)
            load_val(F, MLIR_GetOpOperand(op, i), RCX);
            alu_rr(F, 0x01, RAX, RCX, 1);             // add rax, rcx
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
        if (!callee.size) { fail(F, "indirect call unsupported", (string){0}); return; }
        size_t no = MLIR_GetOpNumOperands(op);
        if (no > 6) { fail(F, "call with >6 args", callee); return; }
        for (size_t i = 0; i < no; i++)
            load_val(F, MLIR_GetOpOperand(op, i), ARG_REGS[i]);
        eb(F, 0xE8);
        uint32_t field = here(F); e32(F, 0);
        char *cs = (char *)malloc(callee.size + 1);
        memcpy(cs, callee.str, callee.size); cs[callee.size] = 0;
        obj_add_reloc(F->f, field, field + 4, false, cs, 0);
        sym_add(refs, callee);
        if (MLIR_GetOpNumResults(op) == 1)
            store_val(F, MLIR_GetOpResult(op, 0), RAX);
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
        MLIR_BlockHandle tt = MLIR_GetOpSuccessor(op, 0);
        MLIR_BlockHandle ft = MLIR_GetOpSuccessor(op, 1);
        emit_edge(F, op, 0, tt);
        emit_edge(F, op, 1, ft);
        load_val(F, MLIR_GetOpOperand(op, 0), RAX);      // cond
        alu_rr(F, 0x85, RAX, RAX, 0);                    // test eax, eax
        eb(F, 0x0F); eb(F, (uint8_t)(0x80 + CC_NE));     // jne true
        uint32_t f1 = here(F); e32(F, 0);
        add_fix(F, block_index(reg, tt), f1);
        eb(F, 0xE9); uint32_t f2 = here(F); e32(F, 0);   // jmp false
        add_fix(F, block_index(reg, ft), f2);
        return;
    }
    fail(F, "unsupported terminator", nm);
}

// ---------------------------------------------------------------------------
// Function emission: frame pre-pass, prologue, blocks, branch patch.
// ---------------------------------------------------------------------------
static unsigned alloca_bytes(MLIR_Context *ctx, MLIR_OpHandle op) {
    // element type size * count; default 8. Handles `[N x i8]` and scalar.
    MLIR_AttributeHandle et = MLIR_GetOpAttributeByName(op, "elem_type");
    unsigned esz = 8;
    if (et) {
        MLIR_TypeHandle t = MLIR_GetAttributeType(et);
        if (t != MLIR_INVALID_HANDLE) {
            if (MLIR_IsTypeLLVMArray(t)) {
                MLIR_TypeHandle el = MLIR_GetTypeLLVMArrayElement(t);
                uint64_t nel = MLIR_GetTypeLLVMArrayNumElements(t);
                int eb2 = MLIR_IsTypeInteger(el) ? 0 : 0; (void)eb2;
                string es = MLIR_GetTypeString(ctx, el);
                unsigned per = 8;
                if (es.size >= 2 && es.str[0] == 'i') { int b=0; for(size_t i=1;i<es.size;i++) b=b*10+(es.str[i]-'0'); per=(b+7)/8; }
                esz = (unsigned)(per * nel);
            } else if (MLIR_IsTypeInteger(t)) {
                string es = MLIR_GetTypeString(ctx, t);
                int b=0; for(size_t i=1;i<es.size;i++) b=b*10+(es.str[i]-'0'); esz=(unsigned)((b+7)/8);
            }
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
    //     pool), so the sequential moves need no parallel-move resolution. ---
    if (nb > 0) {
        MLIR_BlockHandle e0 = MLIR_GetRegionBlock(reg, 0);
        size_t na = MLIR_GetBlockNumArgs(e0);
        for (size_t a = 0; a < na && a < 6; a++)
            store_val(&F, MLIR_GetBlockArg(e0, a), ARG_REGS[a]);
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

static void gmap_put(GMap *g, string nm, uint32_t off) {
    if (g->n == g->cap) { g->cap = g->cap ? g->cap*2 : 8; g->name = realloc(g->name, g->cap*sizeof(char*)); g->off = realloc(g->off, g->cap*sizeof(uint32_t)); }
    g->name[g->n] = (char*)malloc(nm.size+1); memcpy(g->name[g->n], nm.str, nm.size); g->name[g->n][nm.size]=0;
    g->off[g->n] = off; g->n++;
}

// Collect llvm.mlir.global string/zero initializers into the data blob.
static void collect_globals(MLIR_Context *ctx, MLIR_BlockHandle mb, GMap *gm, Buf *data) {
    (void)ctx;
    size_t no = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!str_eqz(MLIR_GetOpName(op), "llvm.mlir.global")) continue;
        string name = attr_str(op, "sym_name");
        uint32_t off = (uint32_t)data->len;
        MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
        if (va) {
            string s = MLIR_GetAttributeString(va);
            buf_append(data, s.str, s.size);
        } else {
            // zero-initialized: size from global_type array length
            MLIR_AttributeHandle gt = MLIR_GetOpAttributeByName(op, "global_type");
            size_t sz = 8;
            if (gt) { MLIR_TypeHandle t = MLIR_GetAttributeType(gt);
                if (t != MLIR_INVALID_HANDLE && MLIR_IsTypeLLVMArray(t)) sz = (size_t)MLIR_GetTypeLLVMArrayNumElements(t); }
            for (size_t z = 0; z < sz; z++) buf_u8(data, 0);
        }
        gmap_put(gm, name, off);
        while (data->len & 7) buf_u8(data, 0);   // 8-align next global
    }
}

static void synth_start(ObjModule *m) {
    ObjFunc *f = obj_add_func(m, strdup("_start"), true);
    FnCtx F; memset(&F,0,sizeof(F)); F.f = f;
    eb(&F, 0xE8); uint32_t fld = here(&F); e32(&F, 0);          // call main
    obj_add_reloc(f, fld, fld+4, false, strdup("main"), 0);
    mov_rr(&F, RDI, RAX, 0);                                    // mov edi, eax
    mov_ri(&F, RAX, NR_EXIT, 0);                                // mov eax, 60
    eb(&F, 0x0F); eb(&F, 0x05);                                 // syscall
}

// synth a Linux syscall thunk for `name`; returns false if `name` is not a
// known syscall primitive.
static bool synth_thunk(ObjModule *m, const char *name) {
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
    collect_globals(ctx, mb, &gm, &data);
    g_x64_gmap = &gm;

    // index defined functions by name
    size_t no = MLIR_GetBlockNumOps(mb);
    SymSet refs; memset(&refs, 0, sizeof(refs));

    ObjModule m; memset(&m, 0, sizeof(m));
    synth_start(&m);             // index 0
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
    g_x64_gmap = NULL;
    return ok;
}
