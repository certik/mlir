// llvm dialect -> aarch64 lowering. See mlir_llvm_to_aarch64.h for the
// public API, rationale, and the staged build-up plan.
//
// Status: Step 2 (in progress) — straight-line integer codegen. Lowers a
// single-block `llvm.func` of integer operations directly to the
// physical-register `aarch64` dialect, then relies on the existing
// `mlir_aarch64_to_macho` encoder. A synthesised `_start` calls `main`
// and exits with its return value via libSystem `_exit`.
//
// Supported so far: parameters (<=8 integer args), llvm.mlir.constant,
// integer binops (add/sub/mul/sdiv/udiv/srem/urem/and/or/xor/shl/lshr/
// ashr), llvm.icmp, llvm.select, direct llvm.call (<=8 args), llvm.return.
// NOT yet: structured control flow (scf.if/while/index_switch), memory
// (alloca/load/store/getelementptr), globals/strings, indirect & variadic
// calls, floating point. Each lands a clear stderr diagnostic.
//
// Register model: a trivial "spill everything" allocator — every SSA value
// gets its own 8-byte frame slot; operands are reloaded into scratch regs
// (x9/x10/x11) per instruction and results stored back immediately. Always
// correct and call-safe (no value kept live across an op), if slow. The
// real linear-scan allocator + a materialised virtual-register `a64ssa`
// tier arrive together in Step 3, where keeping values in registers pays
// off. Emitting `aarch64.*` directly here is the GlobalISel "fast isel +
// trivial regalloc" shape.
//
// Nothing here is on the path of the existing wasm/wmir backends; it is
// reached only via the opt-in `--macho-backend=llvm` driver flag.

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "mlir_llvm_to_aarch64.h"
#include "mlir_op_names.h"

// ---------------------------------------------------------------------------
// Small attribute / op builders (mirrors of the wmir backend's helpers).
// ---------------------------------------------------------------------------
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name,
                                     int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}

static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static void emit_movz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "imm16", imm16);
    a[2] = attr_i32(ctx, "hw", hw);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOVZ, a, 4));
}
static void emit_movk(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "imm16", imm16);
    a[2] = attr_i32(ctx, "hw", hw);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOVK, a, 4));
}
static void emit_bl(MLIR_Context *ctx, MLIR_BlockHandle blk, string callee) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_s(ctx, "callee", callee.str, callee.size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_BL, a, 1));
}
static void emit_ret(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_RET, NULL, 0));
}

// Materialise a 64-bit immediate into `rd` with a movz + movk chain,
// emitting only the non-zero 16-bit lanes (movz seeds the lowest lane so
// a zero value still produces `movz rd, #0`).
static void emit_load_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint8_t rd, uint64_t v, bool sf) {
    int lanes = sf ? 4 : 2;
    emit_movz(ctx, blk, rd, (uint16_t)(v & 0xffffu), 0, sf);
    for (int hw = 1; hw < lanes; hw++) {
        uint16_t chunk = (uint16_t)((v >> (16 * hw)) & 0xffffu);
        if (chunk != 0) emit_movk(ctx, blk, rd, chunk, (uint8_t)hw, sf);
    }
}

static void emit_mov_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOV_X, a, 2));
}
static void emit_3reg(MLIR_Context *ctx, MLIR_BlockHandle blk, MLIR_OpType t,
                      uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, t, a, 4));
}
static void emit_msub(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra, bool sf) {
    MLIR_AttributeHandle a[5];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_i32(ctx, "ra", ra);
    a[4] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MSUB, a, 5));
}
static void emit_ldst_x(MLIR_Context *ctx, MLIR_BlockHandle blk, MLIR_OpType t,
                        uint8_t rt, uint8_t rn, uint32_t off_bytes) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", (int64_t)off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, t, a, 3));
}
static void emit_ldr_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint32_t off) {
    emit_ldst_x(ctx, blk, OP_TYPE_AARCH64_LDR_X, rt, rn, off);
}
static void emit_str_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint32_t off) {
    emit_ldst_x(ctx, blk, OP_TYPE_AARCH64_STR_X, rt, rn, off);
}
static void emit_cmp_reg(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rn, uint8_t rm, bool sf) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rn", rn);
    a[1] = attr_i32(ctx, "rm", rm);
    a[2] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CMP_REG, a, 3));
}
static void emit_cmp_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rn, uint16_t imm12, bool sf) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rn", rn);
    a[1] = attr_i32(ctx, "imm12", imm12);
    a[2] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CMP_IMM, a, 3));
}
static void emit_cset(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t cond, bool sf) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "cond", cond);
    a[2] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CSET, a, 3));
}
static void emit_csel(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond, bool sf) {
    MLIR_AttributeHandle a[5];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_i32(ctx, "cond", cond);
    a[4] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CSEL, a, 5));
}
static void emit_prologue(MLIR_Context *ctx, MLIR_BlockHandle blk, uint32_t fs) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", (int32_t)fs);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_PROLOGUE, a, 1));
}
static void emit_epilogue(MLIR_Context *ctx, MLIR_BlockHandle blk, uint32_t fs) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", (int32_t)fs);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_EPILOGUE, a, 1));
}
static MLIR_OpHandle build_branch_op(MLIR_Context *ctx, MLIR_OpType t,
                                     MLIR_AttributeHandle *attrs, size_t na,
                                     MLIR_BlockHandle target) {
    MLIR_BlockHandle succs[1] = { target };
    MLIR_ValueHandle *succ_ops[1] = { NULL };
    size_t           n_succ_ops[1] = { 0 };
    return MLIR_CreateOpWithSuccessors(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        succs, 1, succ_ops, n_succ_ops,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}
static void emit_b(MLIR_Context *ctx, MLIR_BlockHandle blk,
                   MLIR_BlockHandle target) {
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_B, NULL, 0, target));
}
static void emit_bcond(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t cond, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "cond", cond);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_B_COND, a, 1, target));
}
static void emit_cbnz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rt, bool sf, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_CBNZ, a, 2, target));
}

// ---------------------------------------------------------------------------
// Helpers over the input `llvm` dialect module.
// ---------------------------------------------------------------------------
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static bool type_is_i64(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_TypeHandle ty = MLIR_GetValueType(v);
    string s = MLIR_GetTypeString(ctx, ty);
    return s.size == 3 && memcmp(s.str, "i64", 3) == 0;
}

// Bit width of an integer-typed value ("i1"/"i8"/"i16"/"i32"/"i64"). 0 if not
// a recognised integer type.
static int int_type_bits(MLIR_Context *ctx, MLIR_ValueHandle v) {
    string s = MLIR_GetTypeString(ctx, MLIR_GetValueType(v));
    if (s.size < 2 || s.str[0] != 'i') return 0;
    int w = 0;
    for (size_t i = 1; i < s.size; i++) {
        if (s.str[i] < '0' || s.str[i] > '9') return 0;
        w = w * 10 + (s.str[i] - '0');
    }
    return w;
}

static bool func_has_body(MLIR_OpHandle fn) {
    return MLIR_GetOpNumRegions(fn) > 0 &&
           MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn, 0)) > 0;
}

// Map an LLVM-dialect icmp predicate (eq=0, ne=1, slt=2, sle=3, sgt=4,
// sge=5, ult=6, ule=7, ugt=8, uge=9) to an AArch64 condition code. -1 if
// out of range.
static int icmp_pred_to_cond(int64_t p) {
    switch (p) {
        case 0: return 0;   // eq  -> EQ
        case 1: return 1;   // ne  -> NE
        case 2: return 11;  // slt -> LT
        case 3: return 13;  // sle -> LE
        case 4: return 12;  // sgt -> GT
        case 5: return 10;  // sge -> GE
        case 6: return 3;   // ult -> CC/LO
        case 7: return 9;   // ule -> LS
        case 8: return 8;   // ugt -> HI
        case 9: return 2;   // uge -> CS/HS
        default: return -1;
    }
}

// ---------------------------------------------------------------------------
// Value -> stack-slot map (open addressing keyed by the SSA value handle).
// Step 2 uses a trivial "spill everything" allocator: each SSA value lives
// in its own 8-byte frame slot; operands are reloaded into scratch
// registers (x9/x10/x11) for each instruction and the result is stored
// back immediately. This is always correct regardless of value count and
// survives calls (no value is kept live in a register across an op). The
// real linear-scan allocator + materialised a64ssa vreg tier arrive in
// Step 3, where keeping values in registers actually pays off.
// ---------------------------------------------------------------------------
typedef struct { uintptr_t key; int32_t slot; } SlotEnt;
typedef struct { SlotEnt *t; size_t cap; size_t n; Arena *arena; } SlotMap;

static size_t sm_hash(uintptr_t k) {
    k *= 0x9E3779B97F4A7C15ull;
    return (size_t)(k >> 32);
}
static void sm_grow(SlotMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    SlotEnt *nt = (SlotEnt *)arena_alloc(m->arena, ncap * sizeof(SlotEnt));
    memset(nt, 0, ncap * sizeof(SlotEnt));
    for (size_t i = 0; i < m->cap; i++) {
        if (m->t[i].key == 0) continue;
        size_t j = sm_hash(m->t[i].key) & (ncap - 1);
        while (nt[j].key != 0) j = (j + 1) & (ncap - 1);
        nt[j] = m->t[i];
    }
    m->t = nt;
    m->cap = ncap;
}
static void sm_put(SlotMap *m, MLIR_ValueHandle k, int32_t slot) {
    if ((m->n + 1) * 4 >= m->cap * 3) sm_grow(m);
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) return;
        i = (i + 1) & mask;
    }
    m->t[i].key = (uintptr_t)k;
    m->t[i].slot = slot;
    m->n++;
}
static bool sm_get(SlotMap *m, MLIR_ValueHandle k, int32_t *out) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) { *out = m->t[i].slot; return true; }
        i = (i + 1) & mask;
    }
    return false;
}

// Reload an operand value from its frame slot into register `rd`.
static bool load_value(MLIR_Context *ctx, MLIR_BlockHandle blk, SlotMap *sm,
                       MLIR_ValueHandle v, uint8_t rd) {
    int32_t slot;
    if (!sm_get(sm, v, &slot)) return false;
    emit_ldr_x(ctx, blk, rd, 31, (uint32_t)slot * 8u);
    return true;
}
// Store register `rd` into the frame slot for result value `v`.
static bool store_value(MLIR_Context *ctx, MLIR_BlockHandle blk, SlotMap *sm,
                        MLIR_ValueHandle v, uint8_t rd) {
    int32_t slot;
    if (!sm_get(sm, v, &slot)) return false;
    emit_str_x(ctx, blk, rd, 31, (uint32_t)slot * 8u);
    return true;
}

// Integer binops whose AArch64 form is a plain 3-register op.
static bool simple_binop_optype(string on, MLIR_OpType *out) {
    if (name_eq(on, "llvm.add"))  { *out = OP_TYPE_AARCH64_ADD_REG; return true; }
    if (name_eq(on, "llvm.sub"))  { *out = OP_TYPE_AARCH64_SUB_REG; return true; }
    if (name_eq(on, "llvm.mul"))  { *out = OP_TYPE_AARCH64_MUL;     return true; }
    if (name_eq(on, "llvm.sdiv")) { *out = OP_TYPE_AARCH64_SDIV;    return true; }
    if (name_eq(on, "llvm.udiv")) { *out = OP_TYPE_AARCH64_UDIV;    return true; }
    if (name_eq(on, "llvm.and"))  { *out = OP_TYPE_AARCH64_AND_REG; return true; }
    if (name_eq(on, "llvm.or"))   { *out = OP_TYPE_AARCH64_ORR_REG; return true; }
    if (name_eq(on, "llvm.xor"))  { *out = OP_TYPE_AARCH64_EOR_REG; return true; }
    if (name_eq(on, "llvm.shl"))  { *out = OP_TYPE_AARCH64_LSL_REG; return true; }
    if (name_eq(on, "llvm.lshr")) { *out = OP_TYPE_AARCH64_LSR_REG; return true; }
    if (name_eq(on, "llvm.ashr")) { *out = OP_TYPE_AARCH64_ASR_REG; return true; }
    return false;
}

#define A64_FAIL(...) do { fprintf(stderr, __VA_ARGS__); return MLIR_INVALID_HANDLE; } while (0)

// ---------------------------------------------------------------------------
// Lowering context. The selector walks the structured `llvm`+`scf` body and
// emits a flat CFG of `aarch64` blocks into `out_region`. `cur` is the block
// currently being appended to; control-flow ops create new blocks and move
// `cur`. `ok` is cleared (with a stderr diagnostic) on the first failure.
// ---------------------------------------------------------------------------
typedef struct {
    MLIR_Context     *ctx;
    SlotMap          *sm;
    MLIR_RegionHandle out_region;
    MLIR_BlockHandle  cur;
    string            sym;
    uint32_t          frame_size;
    bool              ok;
} LowerCtx;

#define LFAIL(...) do { fprintf(stderr, __VA_ARGS__); L->ok = false; return; } while (0)

static MLIR_BlockHandle new_block(LowerCtx *L) {
    MLIR_BlockHandle b = MLIR_CreateBlock(L->ctx);
    MLIR_AppendRegionBlock(L->ctx, L->out_region, b);
    return b;
}

// Slot-to-slot copy (block-arg / phi resolution and yield forwarding).
static void copy_slot(LowerCtx *L, MLIR_ValueHandle src, MLIR_ValueHandle dst) {
    if (!load_value(L->ctx, L->cur, L->sm, src, 9) ||
        !store_value(L->ctx, L->cur, L->sm, dst, 9))
        LFAIL("llvm->aarch64: undefined value in copy (%.*s)\n",
              (int)L->sym.size, L->sym.str);
}

// Store an scf.yield's operands into the enclosing op's result slots.
static void store_yield(LowerCtx *L, MLIR_OpHandle yield, MLIR_OpHandle owner) {
    size_t n = MLIR_GetOpNumOperands(yield);
    for (size_t i = 0; i < n; i++) {
        copy_slot(L, MLIR_GetOpOperand(yield, i), MLIR_GetOpResult(owner, i));
        if (!L->ok) return;
    }
}

static void lower_op(LowerCtx *L, MLIR_OpHandle op);

// Lower every non-terminator op of `block` into the current CFG and return
// the block's terminator op (the caller interprets it).
static MLIR_OpHandle lower_block_ops(LowerCtx *L, MLIR_BlockHandle block) {
    size_t n = MLIR_GetBlockNumOps(block);
    if (n == 0) { L->ok = false; return MLIR_INVALID_HANDLE; }
    for (size_t i = 0; i + 1 < n; i++) {
        lower_op(L, MLIR_GetBlockOp(block, i));
        if (!L->ok) return MLIR_INVALID_HANDLE;
    }
    return MLIR_GetBlockOp(block, n - 1);
}

// Parse a `cases` attribute that prints as "array<i64: 1, 2, 3>".
static bool parse_cases(string cs, int64_t *out, size_t n) {
    size_t p = 0;
    while (p < cs.size && cs.str[p] != ':') p++;
    if (p < cs.size) p++;
    size_t got = 0;
    while (p < cs.size && got < n) {
        while (p < cs.size && (cs.str[p] == ' ' || cs.str[p] == ',')) p++;
        if (p >= cs.size || cs.str[p] == '>') break;
        int64_t sign = 1;
        if (cs.str[p] == '-') { sign = -1; p++; }
        int64_t v = 0;
        while (p < cs.size && cs.str[p] >= '0' && cs.str[p] <= '9')
            v = v * 10 + (cs.str[p++] - '0');
        out[got++] = sign * v;
    }
    return got == n;
}

static void lower_scf_if(LowerCtx *L, MLIR_OpHandle op) {
    bool he = MLIR_GetOpNumRegions(op) >= 2 &&
              MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 1)) > 0;
    if (!load_value(L->ctx, L->cur, L->sm, MLIR_GetOpOperand(op, 0), 9))
        LFAIL("llvm->aarch64: undefined scf.if condition\n");

    MLIR_BlockHandle then_blk = new_block(L);
    MLIR_BlockHandle else_blk = he ? new_block(L) : MLIR_INVALID_HANDLE;
    MLIR_BlockHandle end_blk  = new_block(L);

    emit_cbnz(L->ctx, L->cur, 9, false, then_blk);
    emit_b(L->ctx, L->cur, he ? else_blk : end_blk);

    L->cur = then_blk;
    MLIR_OpHandle yt = lower_block_ops(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0));
    if (!L->ok) return;
    store_yield(L, yt, op);
    if (!L->ok) return;
    emit_b(L->ctx, L->cur, end_blk);

    if (he) {
        L->cur = else_blk;
        MLIR_OpHandle ye = lower_block_ops(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 1), 0));
        if (!L->ok) return;
        store_yield(L, ye, op);
        if (!L->ok) return;
        emit_b(L->ctx, L->cur, end_blk);
    }
    L->cur = end_blk;
}

static void lower_scf_while(LowerCtx *L, MLIR_OpHandle op) {
    MLIR_BlockHandle before_src = MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0);
    MLIR_BlockHandle after_src  = MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 1), 0);
    size_t nb = MLIR_GetBlockNumArgs(before_src);

    // init operands -> before-region block-arg slots.
    for (size_t i = 0; i < nb; i++) {
        copy_slot(L, MLIR_GetOpOperand(op, i), MLIR_GetBlockArg(before_src, i));
        if (!L->ok) return;
    }
    MLIR_BlockHandle before_blk = new_block(L);
    emit_b(L->ctx, L->cur, before_blk);

    L->cur = before_blk;
    MLIR_OpHandle cterm = lower_block_ops(L, before_src);   // scf.condition
    if (!L->ok) return;
    size_t nf = MLIR_GetOpNumOperands(cterm) - 1;           // forwarded values
    if (!load_value(L->ctx, L->cur, L->sm, MLIR_GetOpOperand(cterm, 0), 9))
        LFAIL("llvm->aarch64: undefined scf.condition value\n");

    MLIR_BlockHandle exit_store = new_block(L);
    MLIR_BlockHandle cont_blk   = new_block(L);
    MLIR_BlockHandle exit_blk   = new_block(L);
    emit_cbnz(L->ctx, L->cur, 9, false, cont_blk);          // cond true -> after
    emit_b(L->ctx, L->cur, exit_store);                     // cond false -> exit

    // exit: forwarded values become the scf.while results.
    L->cur = exit_store;
    for (size_t i = 0; i < nf; i++) {
        copy_slot(L, MLIR_GetOpOperand(cterm, i + 1), MLIR_GetOpResult(op, i));
        if (!L->ok) return;
    }
    emit_b(L->ctx, L->cur, exit_blk);

    // continue: forwarded values become the after-region block args.
    L->cur = cont_blk;
    for (size_t i = 0; i < nf; i++) {
        copy_slot(L, MLIR_GetOpOperand(cterm, i + 1), MLIR_GetBlockArg(after_src, i));
        if (!L->ok) return;
    }
    MLIR_OpHandle yterm = lower_block_ops(L, after_src);    // scf.yield
    if (!L->ok) return;
    for (size_t i = 0; i < nb; i++) {                       // yield -> before args
        copy_slot(L, MLIR_GetOpOperand(yterm, i), MLIR_GetBlockArg(before_src, i));
        if (!L->ok) return;
    }
    emit_b(L->ctx, L->cur, before_blk);

    L->cur = exit_blk;
}

static void lower_scf_index_switch(LowerCtx *L, MLIR_OpHandle op) {
    size_t n_regions = MLIR_GetOpNumRegions(op);
    if (n_regions < 1) LFAIL("llvm->aarch64: malformed scf.index_switch\n");
    size_t n_cases = n_regions - 1;
    int64_t case_vals[256];
    if (n_cases > 256) LFAIL("llvm->aarch64: scf.index_switch too large\n");
    if (n_cases > 0) {
        MLIR_AttributeHandle ca = MLIR_GetOpAttributeByName(op, "cases");
        if (ca == MLIR_INVALID_HANDLE ||
            !parse_cases(MLIR_GetAttributeAsString(L->ctx, ca), case_vals, n_cases))
            LFAIL("llvm->aarch64: cannot parse scf.index_switch cases\n");
    }
    if (!load_value(L->ctx, L->cur, L->sm, MLIR_GetOpOperand(op, 0), 9))
        LFAIL("llvm->aarch64: undefined scf.index_switch selector\n");

    MLIR_BlockHandle end_blk = new_block(L);
    MLIR_BlockHandle def_blk = new_block(L);
    MLIR_BlockHandle case_blk[256];
    for (size_t i = 0; i < n_cases; i++) case_blk[i] = new_block(L);

    // Compare chain: cmp selector, #case; b.eq case_blk[i]. Selector stays
    // in x9 throughout this (single) block.
    for (size_t i = 0; i < n_cases; i++) {
        emit_load_imm(L->ctx, L->cur, 10, (uint64_t)case_vals[i], false);
        emit_cmp_reg(L->ctx, L->cur, 9, 10, false);
        emit_bcond(L->ctx, L->cur, /*EQ*/0, case_blk[i]);
    }
    emit_b(L->ctx, L->cur, def_blk);

    L->cur = def_blk;
    MLIR_OpHandle dt = lower_block_ops(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0));
    if (!L->ok) return;
    store_yield(L, dt, op);
    if (!L->ok) return;
    emit_b(L->ctx, L->cur, end_blk);

    for (size_t i = 0; i < n_cases; i++) {
        L->cur = case_blk[i];
        MLIR_OpHandle ct = lower_block_ops(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, i + 1), 0));
        if (!L->ok) return;
        store_yield(L, ct, op);
        if (!L->ok) return;
        emit_b(L->ctx, L->cur, end_blk);
    }
    L->cur = end_blk;
}

// Lower a single non-terminator op into the current block / CFG.
static void lower_op(LowerCtx *L, MLIR_OpHandle op) {
    MLIR_Context     *ctx = L->ctx;
    MLIR_BlockHandle  blk = L->cur;
    SlotMap          *sm  = L->sm;
    string            on  = MLIR_GetOpName(op);
    MLIR_OpType       bt;

    if (name_eq(on, "llvm.mlir.constant")) {
        MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
        if (va == MLIR_INVALID_HANDLE || MLIR_GetOpNumResults(op) != 1)
            LFAIL("llvm->aarch64: malformed llvm.mlir.constant\n");
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        emit_load_imm(ctx, blk, 9, (uint64_t)MLIR_GetAttributeInteger(va),
                      type_is_i64(ctx, res));
        store_value(ctx, blk, sm, res, 9);

    } else if (simple_binop_optype(on, &bt)) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_i64(ctx, res);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_3reg(ctx, blk, bt, 9, 9, 10, sf);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.srem") || name_eq(on, "llvm.urem")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_i64(ctx, res);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        MLIR_OpType dt = name_eq(on, "llvm.srem") ? OP_TYPE_AARCH64_SDIV
                                                  : OP_TYPE_AARCH64_UDIV;
        emit_3reg(ctx, blk, dt, 11, 9, 10, sf);
        emit_msub(ctx, blk, 9, 11, 10, 9, sf);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.icmp")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: llvm.icmp without predicate\n");
        int cond = icmp_pred_to_cond(MLIR_GetAttributeInteger(pa));
        if (cond < 0)
            LFAIL("llvm->aarch64: unsupported icmp predicate %lld\n",
                  (long long)MLIR_GetAttributeInteger(pa));
        bool sf = type_is_i64(ctx, MLIR_GetOpOperand(op, 0));
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_cmp_reg(ctx, blk, 9, 10, sf);
        emit_cset(ctx, blk, 9, (uint8_t)cond, false);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.select")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_i64(ctx, res);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9)  ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 2), 11))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_cmp_imm(ctx, blk, 9, 0, false);
        emit_csel(ctx, blk, 9, 10, 11, /*NE*/1, sf);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.call")) {
        MLIR_AttributeHandle callee = MLIR_GetOpAttributeByName(op, "callee");
        if (callee == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: indirect calls not yet supported\n");
        MLIR_AttributeHandle vct = MLIR_GetOpAttributeByName(op, "var_callee_type");
        if (vct != MLIR_INVALID_HANDLE &&
            MLIR_GetTypeFunctionIsVarArg(MLIR_GetAttributeTypeValue(vct)))
            LFAIL("llvm->aarch64: variadic calls not yet supported\n");
        size_t no = MLIR_GetOpNumOperands(op);
        if (no > 8)
            LFAIL("llvm->aarch64: call with %zu args (>8 not supported)\n", no);
        for (size_t k = 0; k < no; k++)
            if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, k), (uint8_t)k))
                LFAIL("llvm->aarch64: undefined call arg in '%.*s'\n",
                      (int)L->sym.size, L->sym.str);
        string nm = MLIR_GetAttributeAsString(ctx, callee);
        if (nm.size > 0 && nm.str[0] == '@') { nm.str++; nm.size--; }
        emit_bl(ctx, blk, nm);
        if (MLIR_GetOpNumResults(op) == 1)
            store_value(ctx, blk, sm, MLIR_GetOpResult(op, 0), 0);

    } else if (name_eq(on, "llvm.mlir.undef")) {
        if (MLIR_GetOpNumResults(op) == 1) {
            emit_load_imm(ctx, blk, 9, 0, false);
            store_value(ctx, blk, sm, MLIR_GetOpResult(op, 0), 9);
        }

    } else if (name_eq(on, "arith.index_cast") ||
               name_eq(on, "arith.index_castui")) {
        // index <-> integer: a plain copy in our 64-bit slot model.
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        store_value(ctx, blk, sm, MLIR_GetOpResult(op, 0), 9);

    } else if (name_eq(on, "llvm.trunc") || name_eq(on, "llvm.zext")) {
        // Both keep the low N bits (our values are stored zero-extended); a
        // mask to the destination (trunc) / source (zext) width suffices.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int w = name_eq(on, "llvm.trunc") ? int_type_bits(ctx, res)
                                          : int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        if (w > 0 && w < 64) {
            emit_load_imm(ctx, blk, 10, (w >= 64) ? ~0ull : ((1ull << w) - 1), true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_AND_REG, 9, 9, 10, true);
        }
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.sext")) {
        // Sign-extend from the source width via a left-then-arithmetic-right
        // shift pair in 64-bit registers.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int w = int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        if (w > 0 && w < 64) {
            emit_load_imm(ctx, blk, 10, (uint64_t)(64 - w), true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_LSL_REG, 9, 9, 10, true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ASR_REG, 9, 9, 10, true);
        }
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "scf.if")) {
        lower_scf_if(L, op);
    } else if (name_eq(on, "scf.while")) {
        lower_scf_while(L, op);
    } else if (name_eq(on, "scf.index_switch")) {
        lower_scf_index_switch(L, op);

    } else {
        LFAIL("llvm->aarch64: unsupported op '%.*s' in '%.*s'\n",
              (int)on.size, on.str, (int)L->sym.size, L->sym.str);
    }
}

// Recursively assign an 8-byte frame slot to every block arg and op result
// reachable in `block` (including nested scf regions).
static void assign_slots_block(SlotMap *sm, MLIR_BlockHandle block,
                               int32_t *nslots) {
    size_t na = MLIR_GetBlockNumArgs(block);
    for (size_t i = 0; i < na; i++)
        sm_put(sm, MLIR_GetBlockArg(block, i), (*nslots)++);
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        size_t nr = MLIR_GetOpNumResults(op);
        for (size_t r = 0; r < nr; r++)
            sm_put(sm, MLIR_GetOpResult(op, r), (*nslots)++);
        size_t ng = MLIR_GetOpNumRegions(op);
        for (size_t g = 0; g < ng; g++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, g);
            size_t nbk = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < nbk; b++)
                assign_slots_block(sm, MLIR_GetRegionBlock(rg, b), nslots);
        }
    }
}

// ---------------------------------------------------------------------------
// Select one defined `llvm.func` into an `aarch64.func`. Supports straight-
// line integer code plus structured control flow (scf.if / scf.while /
// scf.index_switch). Uses the trivial spill-everything allocator: every SSA
// value (incl. block args / scf results) lives in a frame slot, so phi /
// block-arg resolution is a slot-to-slot copy at each edge.
// ---------------------------------------------------------------------------
static MLIR_OpHandle select_func(MLIR_Context *ctx, MLIR_OpHandle fn,
                                 string sym) {
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(fn, 0);
    if (MLIR_GetRegionNumBlocks(src_region) != 1) {
        A64_FAIL("llvm->aarch64: function '%.*s' has a non-structured CFG "
                 "(expected a single entry block with scf control flow)\n",
                 (int)sym.size, sym.str);
    }
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(src_region, 0);
    size_t nargs = MLIR_GetBlockNumArgs(src_blk);
    if (nargs > 8) {
        A64_FAIL("llvm->aarch64: function '%.*s' has %zu parameters "
                 "(>8 integer args not yet supported)\n",
                 (int)sym.size, sym.str, nargs);
    }

    MLIR_BlockHandle  out_blk = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_reg = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_reg, out_blk);

    SlotMap sm = {0};
    sm.arena = MLIR_GetArenaAllocator(ctx);
    int32_t nslots = 0;
    assign_slots_block(&sm, src_blk, &nslots);
    if (nslots > 4095) {
        A64_FAIL("llvm->aarch64: function '%.*s' needs %d slots "
                 "(frame too large for the trivial allocator)\n",
                 (int)sym.size, sym.str, nslots);
    }
    uint32_t frame_size = ((uint32_t)nslots * 8u + 15u) & ~15u;

    emit_prologue(ctx, out_blk, frame_size);
    for (size_t i = 0; i < nargs; i++)
        store_value(ctx, out_blk, &sm, MLIR_GetBlockArg(src_blk, i), (uint8_t)i);

    LowerCtx L = { ctx, &sm, out_reg, out_blk, sym, frame_size, true };
    MLIR_OpHandle term = lower_block_ops(&L, src_blk);
    if (!L.ok) return MLIR_INVALID_HANDLE;

    // Function terminator: llvm.return.
    if (!name_eq(MLIR_GetOpName(term), "llvm.return"))
        A64_FAIL("llvm->aarch64: function '%.*s' does not end in llvm.return\n",
                 (int)sym.size, sym.str);
    size_t no = MLIR_GetOpNumOperands(term);
    if (no == 1) {
        if (!load_value(ctx, L.cur, &sm, MLIR_GetOpOperand(term, 0), 0))
            A64_FAIL("llvm->aarch64: undefined return value in '%.*s'\n",
                     (int)sym.size, sym.str);
    } else if (no > 1) {
        A64_FAIL("llvm->aarch64: multi-value return\n");
    }
    emit_epilogue(ctx, L.cur, frame_size);
    emit_ret(ctx, L.cur);

    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", sym.str, sym.size);
    MLIR_RegionHandle regs[1] = { out_reg };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Synthesise `_start`: `bl main; bl _exit`. Exported and emitted first so
// the Mach-O encoder lands LC_MAIN.entryoff on it.
static MLIR_OpHandle synth_start(MLIR_Context *ctx, string main_name) {
    MLIR_BlockHandle  blk = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, reg, blk);

    emit_bl(ctx, blk, main_name);
    emit_bl(ctx, blk, str_lit("_exit"));

    MLIR_AttributeHandle attrs[2];
    attrs[0] = attr_s(ctx, "sym_name", "_start", 6);
    attrs[1] = attr_b(ctx, "exported", true);
    MLIR_RegionHandle regs[1] = { reg };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 2, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

MLIR_OpHandle mlir_llvm_to_aarch64(MLIR_Context *ctx,
                                   MLIR_OpHandle llvm_module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(llvm_module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    MLIR_BlockHandle  out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);

    // `_start` first (entry point).
    MLIR_AppendBlockOp(ctx, out_body, synth_start(ctx, str_lit("main")));

    bool saw_main = false;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        if (!func_has_body(op)) continue;  // skip declarations (malloc/free).
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) {
            fprintf(stderr, "llvm->aarch64: llvm.func without sym_name\n");
            return MLIR_INVALID_HANDLE;
        }
        string sym = MLIR_GetAttributeString(sa);
        if (sym.size == 4 && memcmp(sym.str, "main", 4) == 0) saw_main = true;
        MLIR_OpHandle fn = select_func(ctx, op, sym);
        if (fn == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, fn);
    }

    if (!saw_main) {
        fprintf(stderr, "llvm->aarch64: no defined 'main' function\n");
        return MLIR_INVALID_HANDLE;
    }

    MLIR_RegionHandle regs[1] = { out_region };
    return MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}
