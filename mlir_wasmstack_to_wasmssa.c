// wasmstack -> wasmssa lifter.
// See mlir_wasmstack_to_wasmssa.h for the public API and rationale.
//
// Lifts a flat wasmstack module (the 1:1 textual mirror of wasm bytecode)
// to wasmssa, which is what the wmir backend consumes. Two responsibilities:
//
//   1. Stack -> SSA: walk the flat opcode sequence, simulate the operand
//      stack, and emit wasmssa ops whose operands are real SSA values.
//
//   2. Structured CF reconstruction: re-nest block/loop/if/end into
//      wasmssa.block / .loop / .if region ops. The wmir backend then
//      flattens them into a real CFG with explicit br ops.
//
// Synth-helper short-circuit: any wasmstack.func whose `sym_name`
// matches a known runtime helper that the wmir backend synthesises
// from scratch (printI64, printNewline, printStr, printf, malloc,
// free, strlen, strcmp, memcmp, memchr, fd_write, proc_exit,
// printF32, printF64, tinyc_va_arg_*) is replaced by a body-less
// `wasmssa.import_func` declaration. This avoids having to lift the
// full WASI runtime — its definition will be ignored anyway when the
// wmir backend emits its own version of the helper.

#include "mlir_wasmstack_to_wasmssa.h"

#include <stdarg.h>
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

#define WT_I32 0x7f
#define WT_I64 0x7e
#define WT_F32 0x7d
#define WT_F64 0x7c

// Wasm-opcode constants used by the lifter when synthesising select /
// br_table chains. Must live at module scope because tinyc — used in
// self-hosting — only accepts enums at file scope.
#define WOP_I32_EQ   0x46
#define WOP_I32_SUB  0x6b

// =============================================================================
// Synth-helper detection. Returns true when `name` is a wmir runtime
// helper that gets re-synthesised; in that case the wasmstack.func's
// body is discarded and the function is lifted as an import_func
// declaration.
// =============================================================================
static bool is_synth_helper(string nm) {
    static const char *kHelpers[] = {
        "printI64", "printNewline", "printStr", "printf",
        "printF32", "printF64",
        "malloc", "free", "strlen", "strcmp", "memcmp", "memchr",
        "fd_write", "proc_exit",
        "tinyc_va_arg_i32", "tinyc_va_arg_i64", "tinyc_va_arg_ptr",
        "tinyc_va_arg_struct", "tinyc_va_arg_f64",
        // WASI host imports synthesised in the wmir aarch64 backend.
        "path_open", "fd_close", "fd_read", "fd_seek", "fd_tell",
        "args_get", "args_sizes_get",
        "environ_get", "environ_sizes_get",
    };
    size_t n = sizeof(kHelpers) / sizeof(kHelpers[0]);
    for (size_t i = 0; i < n; i++) {
        size_t kl = strlen(kHelpers[i]);
        if (nm.size == kl && memcmp(nm.str, kHelpers[i], kl) == 0) return true;
    }
    return false;
}

// =============================================================================
// Attr helpers.
// =============================================================================
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 64, true));
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static int64_t at_i(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : 0;
}
static bool at_b(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeBool(a) : false;
}
static string at_s(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
}

// =============================================================================
// Hex helpers (param_types/result_types attributes are hex-encoded
// byte strings).
// =============================================================================
static uint8_t hex_nibble(char c) {
    if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
    if (c >= 'a' && c <= 'f') return (uint8_t)(10 + c - 'a');
    if (c >= 'A' && c <= 'F') return (uint8_t)(10 + c - 'A');
    return 0;
}
static uint8_t *hex_decode_arena(Arena *a, string s, size_t *out_n) {
    size_t n = s.size / 2;
    uint8_t *p = (uint8_t *)arena_alloc(a, n ? n : 1);
    for (size_t i = 0; i < n; i++) {
        p[i] = (uint8_t)((hex_nibble(s.str[i*2]) << 4) | hex_nibble(s.str[i*2+1]));
    }
    *out_n = n;
    return p;
}

// =============================================================================
// Type mapping.
// =============================================================================
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    switch (vt) {
    case WT_I32: return MLIR_CreateTypeInteger(ctx, 32, true);
    case WT_I64: return MLIR_CreateTypeInteger(ctx, 64, true);
    case WT_F32: return MLIR_CreateTypeFloat(ctx, 32, false);
    case WT_F64: return MLIR_CreateTypeFloat(ctx, 64, false);
    }
    return MLIR_CreateTypeInteger(ctx, 32, true);
}

// =============================================================================
// Operand stack (SSA values).
// =============================================================================
typedef struct {
    MLIR_ValueHandle *data;
    size_t            n, cap;
} Stack;

static void st_push(Stack *s, MLIR_ValueHandle v) {
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->data = (MLIR_ValueHandle *)realloc(s->data, s->cap * sizeof(*s->data));
    }
    s->data[s->n++] = v;
}
// Per-fn / per-op diagnostics so st_pop() underflow messages can name
// the function and op that triggered them. These are written by
// lift_func / lift_body and read in st_pop. Defined at file scope so
// tinyC (which rejects function-local `extern` declarations) can
// compile this file.
const char *g_current_fn_name = NULL;
const char *g_current_op_name = NULL;

static MLIR_ValueHandle st_pop(Stack *s) {
    if (s->n == 0) {
        fprintf(stderr, "ws->ssa: stack underflow in st_pop (fn='%s' op='%s')\n",
            g_current_fn_name ? g_current_fn_name : "?",
            g_current_op_name ? g_current_op_name : "?");
        return MLIR_INVALID_HANDLE;
    }
    return s->data[--s->n];
}

static MLIR_ValueHandle st_peek(Stack *s) {
    if (s->n == 0) return MLIR_INVALID_HANDLE;
    return s->data[s->n - 1];
}
static void st_truncate(Stack *s, size_t n) {
    if (n < s->n) s->n = n;
}
static void st_free(Stack *s) { free(s->data); s->data = NULL; s->n = s->cap = 0; }

// =============================================================================
// Op builders.
// =============================================================================
static MLIR_OpHandle build_op_no_res(MLIR_Context *ctx, MLIR_OpType t,
                                     MLIR_AttributeHandle *attrs, size_t na,
                                     MLIR_ValueHandle *operands, size_t no) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, operands, no, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static MLIR_OpHandle build_op_1res(MLIR_Context *ctx, MLIR_OpType t,
                                   MLIR_AttributeHandle *attrs, size_t na,
                                   MLIR_ValueHandle *operands, size_t no,
                                   MLIR_TypeHandle result_ty,
                                   MLIR_ValueHandle *out_result) {
    MLIR_TypeHandle rts[1] = { result_ty };
    MLIR_ValueHandle rvs[1];
    rvs[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
        result_ty, (string){0},
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_OpHandle op = MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, rts, 1, rvs, 1, operands, no, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    *out_result = rvs[0];
    return op;
}

static MLIR_OpHandle build_op_region(MLIR_Context *ctx, MLIR_OpType t,
                                     MLIR_AttributeHandle *attrs, size_t na,
                                     MLIR_ValueHandle *operands, size_t no,
                                     MLIR_RegionHandle *regions, size_t nr,
                                     MLIR_TypeHandle *res_tys,
                                     MLIR_ValueHandle *res_vals, size_t nres) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, res_tys, nres, res_vals, nres,
        operands, no, regions, nr,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Wasm valtype byte -> bytes per cell as actually stored. We use 8 byte
// cells for every local so 32- and 64-bit values share the same slot
// stride and the address math is `base + 8*idx`.
#define LOCAL_CELL_BYTES 8
static uint32_t vt_byte_size(uint8_t vt) {
    switch (vt) {
    case WT_I32: case WT_F32: return 4;
    case WT_I64: case WT_F64: return 8;
    default: return 4;  // fall-back, treat ref types as i32-sized
    }
}

// =============================================================================
// Sibling-scope lookup (callee signatures). We do a linear scan because
// modules are small and this only runs at lift time.
// =============================================================================
static bool lookup_callee_sig(MLIR_BlockHandle siblings, string name,
                              string *out_pt, string *out_rt) {
    size_t n = MLIR_GetBlockNumOps(siblings);
    for (size_t k = 0; k < n; k++) {
        MLIR_OpHandle cand = MLIR_GetBlockOp(siblings, k);
        MLIR_OpType t = MLIR_GetOpType(cand);
        if (t != OP_TYPE_WASMSTACK_FUNC && t != OP_TYPE_WASMSTACK_IMPORT_FUNC) {
            continue;
        }
        string nm_s = at_s(cand, "sym_name");
        if (nm_s.size == name.size && memcmp(nm_s.str, name.str, name.size) == 0) {
            *out_pt = at_s(cand, "param_types");
            *out_rt = at_s(cand, "result_types");
            return true;
        }
    }
    return false;
}

static bool lookup_type_sig(MLIR_BlockHandle siblings, uint32_t type_idx,
                            string *out_pt, string *out_rt) {
    // Signatures from the wasm TYPE section are not currently preserved
    // at the wasmstack level. For call_indirect we instead let the caller
    // supply them as inline `sig_param_types` / `sig_result_types`
    // attributes (the decoder emits these directly).
    (void)siblings; (void)type_idx; (void)out_pt; (void)out_rt;
    return false;
}

// =============================================================================
// Frame stack for structured control flow.
// =============================================================================
#define F_BLOCK 0
#define F_LOOP  1
#define F_IF    2

typedef struct {
    int               kind;
    uint8_t           rtype_byte;       // 0 if void (blocktype 0x40), else valtype.
    bool              has_rtype;
    MLIR_BlockHandle  body_block;       // current MLIR block being filled.
    MLIR_BlockHandle  then_block;       // IF: saved then-region body (for region build at END).
    MLIR_BlockHandle  else_block;       // IF: lazily created on ELSE.
    bool              has_else;
    MLIR_ValueHandle  if_cond;          // IF: condition operand.
    MLIR_BlockHandle  parent_block;     // mlir block to switch back to on END.
    size_t            entry_stack_depth;
    bool              terminated;       // current region body already has a terminator.
} Frame;

typedef struct {
    Frame  *data;
    size_t  n, cap;
} FrameStack;

static Frame *fs_top(FrameStack *fs) {
    return fs->n ? &fs->data[fs->n - 1] : NULL;
}
static void fs_push(FrameStack *fs, Frame f) {
    if (fs->n == fs->cap) {
        fs->cap = fs->cap ? fs->cap * 2 : 8;
        fs->data = (Frame *)realloc(fs->data, fs->cap * sizeof(Frame));
    }
    fs->data[fs->n++] = f;
}
static Frame fs_pop(FrameStack *fs) {
    Frame f = {0};
    if (fs->n) f = fs->data[--fs->n];
    return f;
}
static void fs_free(FrameStack *fs) {
    free(fs->data);
    fs->data = NULL;
    fs->n = fs->cap = 0;
}

// =============================================================================
// Function-lift context.
// =============================================================================
typedef struct {
    MLIR_Context     *ctx;
    Arena            *arena;
    MLIR_BlockHandle  siblings;

    // Locals are spilled to a stack-allocated region in linmem to keep
    // SSA correctness across blocks (wasm-ld emits local.set inside
    // conditional regions, which would otherwise require phi nodes).
    // Each local cell is 8 bytes (max of i32/i64/f32/f64).
    // F->local_offs[i] = byte offset of cell i within the locals area.
    // local.get/set/tee become wasmssa.load/store(%locals_bp,
    // memory_offset = local_offs[i]).
    uint32_t         *local_offs;
    size_t            nl_total;
    uint8_t          *local_types;
    MLIR_ValueHandle  locals_bp;        // %bp: wasm i32 ptr to locals area
    uint32_t          locals_size;      // total size of locals area (bytes)

    size_t            nr;
    uint8_t          *result_types;

    Stack             stack;
    FrameStack        frames;

    MLIR_BlockHandle  cur;
    bool              cur_terminated;
} FnCtx;

// -----------------------------------------------------------------------------
// Emit helpers that funnel through (FnCtx *).
// -----------------------------------------------------------------------------
static void emit(FnCtx *F, MLIR_OpHandle op) {
    MLIR_AppendBlockOp(F->ctx, F->cur, op);
}

// -----------------------------------------------------------------------------
// Pop n operands off the stack, oldest at args[0]. (Wasm semantics: top
// of stack = last param.)
// -----------------------------------------------------------------------------
static bool pop_n(FnCtx *F, size_t n, MLIR_ValueHandle *args) {
    for (size_t k = n; k > 0; k--) {
        if (F->stack.n == 0) {
            args[k - 1] = MLIR_INVALID_HANDLE;
        } else {
            args[k - 1] = st_pop(&F->stack);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Locals-as-memory helpers. Each wasm local is backed by an 8-byte cell
// in a stack-allocated region of linmem pointed to by F->locals_bp.
// Using memory (instead of carrying SSA values directly) sidesteps the
// need for phi/block-arg construction across blocks — wasm-ld output
// freely mutates locals inside conditional regions.
// -----------------------------------------------------------------------------
static MLIR_ValueHandle emit_local_load(FnCtx *F, uint32_t li) {
    uint8_t vt = F->local_types[li];
    MLIR_AttributeHandle as[4];
    as[0] = attr_i32(F->ctx, "valtype", vt);
    as[1] = attr_i32(F->ctx, "memory_offset", (int64_t)F->local_offs[li]);
    as[2] = attr_i32(F->ctx, "memory_align_log2", vt_byte_size(vt) == 8 ? 3 : 2);
    as[3] = attr_i32(F->ctx, "mem_size_bytes", (int64_t)vt_byte_size(vt));
    MLIR_ValueHandle ops[1] = { F->locals_bp };
    MLIR_TypeHandle ty = vt_to_type(F->ctx, vt);
    MLIR_ValueHandle rv;
    emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_LOAD,
        as, 4, ops, 1, ty, &rv));
    return rv;
}
static void emit_local_store(FnCtx *F, uint32_t li, MLIR_ValueHandle val) {
    uint8_t vt = F->local_types[li];
    MLIR_AttributeHandle as[4];
    as[0] = attr_i32(F->ctx, "valtype", vt);
    as[1] = attr_i32(F->ctx, "memory_offset", (int64_t)F->local_offs[li]);
    as[2] = attr_i32(F->ctx, "memory_align_log2", vt_byte_size(vt) == 8 ? 3 : 2);
    as[3] = attr_i32(F->ctx, "mem_size_bytes", (int64_t)vt_byte_size(vt));
    MLIR_ValueHandle ops[2] = { F->locals_bp, val };
    emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_STORE,
        as, 4, ops, 2));
}

// Restore wasm SP (global 0) by adding F->locals_size back to F->locals_bp.
// Emitted before every wasmssa.return op and before the implicit fall-off-
// the-end return.
static void emit_locals_pop(FnCtx *F) {
    if (F->locals_size == 0) return;
    // %sz = const i32 locals_size
    MLIR_AttributeHandle cas[2];
    cas[0] = attr_i32(F->ctx, "valtype", WT_I32);
    cas[1] = attr_i64(F->ctx, "value", (int64_t)F->locals_size);
    MLIR_ValueHandle sz;
    emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_CONST,
        cas, 2, NULL, 0, vt_to_type(F->ctx, WT_I32), &sz));
    // %restored = add locals_bp, sz
    MLIR_AttributeHandle aas[1];
    aas[0] = attr_i32(F->ctx, "valtype", WT_I32);
    MLIR_ValueHandle aops[2] = { F->locals_bp, sz };
    MLIR_ValueHandle restored;
    emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_ADD,
        aas, 1, aops, 2, vt_to_type(F->ctx, WT_I32), &restored));
    // global_set 0 = restored
    MLIR_AttributeHandle gas[2];
    gas[0] = attr_i32(F->ctx, "valtype", WT_I32);
    gas[1] = attr_i32(F->ctx, "global_idx", 0);
    MLIR_ValueHandle gops[1] = { restored };
    emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_GLOBAL_SET,
        gas, 2, gops, 1));
}

// =============================================================================
// Build a wasmssa.block / .loop / .if op from a populated body block.
// Returns the wrapping op (to append to the parent), and populates
// `out_results[]` with its SSA results (matching frame->rtype_byte).
// =============================================================================
static MLIR_OpHandle finalize_frame(FnCtx *F, Frame *f,
                                    MLIR_ValueHandle *out_results,
                                    size_t *out_n_results) {
    size_t n_res = f->has_rtype ? 1 : 0;
    MLIR_TypeHandle res_ty[1];
    MLIR_ValueHandle res_vals[1];
    if (n_res) {
        res_ty[0] = vt_to_type(F->ctx, f->rtype_byte);
        res_vals[0] = MLIR_CreateValueOpResult(F->ctx, MLIR_INVALID_HANDLE, 0,
            res_ty[0], (string){0},
            MLIR_CreateLocationUnknown(F->ctx, (string){0}));
    }

    // Build result_types attr (hex-encoded) if non-empty.
    MLIR_AttributeHandle attrs[1];
    size_t na = 0;
    if (n_res) {
        char hex[3];
        static const char *digs = "0123456789abcdef";
        hex[0] = digs[f->rtype_byte >> 4];
        hex[1] = digs[f->rtype_byte & 0xf];
        hex[2] = 0;
        attrs[na++] = attr_s(F->ctx, "result_types", hex, 2);
    }

    if (f->kind == F_IF) {
        // Wrap then-block in a region.
        MLIR_RegionHandle then_r = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, then_r, f->then_block);
        MLIR_RegionHandle else_r = MLIR_INVALID_HANDLE;
        if (f->has_else) {
            else_r = MLIR_CreateRegion(F->ctx);
            MLIR_AppendRegionBlock(F->ctx, else_r, f->else_block);
        } else if (n_res) {
            // wasmssa.if with results needs an else region. Create one
            // that just block_returns the (no-op) default values. For
            // simplicity, emit a const-0 of rtype and return it.
            MLIR_BlockHandle eb = MLIR_CreateBlock(F->ctx);
            MLIR_AttributeHandle cas[2];
            cas[0] = attr_i32(F->ctx, "valtype", f->rtype_byte);
            cas[1] = attr_i64(F->ctx, "value", 0);
            MLIR_ValueHandle rv;
            MLIR_OpHandle cop = build_op_1res(F->ctx, OP_TYPE_WASMSSA_CONST,
                cas, 2, NULL, 0, vt_to_type(F->ctx, f->rtype_byte), &rv);
            MLIR_AppendBlockOp(F->ctx, eb, cop);
            MLIR_ValueHandle brets[1] = { rv };
            MLIR_AppendBlockOp(F->ctx, eb,
                build_op_no_res(F->ctx, OP_TYPE_WASMSSA_BLOCK_RETURN,
                                NULL, 0, brets, 1));
            else_r = MLIR_CreateRegion(F->ctx);
            MLIR_AppendRegionBlock(F->ctx, else_r, eb);
        }

        MLIR_RegionHandle regs[2];
        size_t nregs = 1;
        regs[0] = then_r;
        if (else_r) { regs[1] = else_r; nregs = 2; }

        MLIR_ValueHandle cond_ops[1] = { f->if_cond };
        MLIR_OpHandle op = build_op_region(F->ctx, OP_TYPE_WASMSSA_IF,
            attrs, na, cond_ops, 1, regs, nregs,
            n_res ? res_ty : NULL, n_res ? res_vals : NULL, n_res);
        for (size_t i = 0; i < n_res; i++) out_results[i] = res_vals[i];
        *out_n_results = n_res;
        return op;
    }

    if (f->kind == F_BLOCK) {
        MLIR_RegionHandle r = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, r, f->body_block);
        MLIR_RegionHandle regs[1] = { r };
        MLIR_OpHandle op = build_op_region(F->ctx, OP_TYPE_WASMSSA_BLOCK,
            attrs, na, NULL, 0, regs, 1,
            n_res ? res_ty : NULL, n_res ? res_vals : NULL, n_res);
        for (size_t i = 0; i < n_res; i++) out_results[i] = res_vals[i];
        *out_n_results = n_res;
        return op;
    }

    // LOOP: emit a plain wasmssa.loop. The wmir lowering handles
    // fall-through (no `br` at end-of-body) by branching to the
    // post-loop block, and pushes a single F_LOOP frame whose target
    // is the loop header — `br depth=0` then continues correctly.
    // NOTE: we previously wrapped the loop in an outer wasmssa.block
    // + unreachable to match the llvm->wasmssa convention. That extra
    // frame off-by-one'd every `br depth>=1` inside the loop body
    // because wmir saw N+1 enclosing frames where wasm saw N.
    MLIR_RegionHandle loop_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, loop_r, f->body_block);
    MLIR_RegionHandle loop_regs[1] = { loop_r };
    MLIR_OpHandle op = build_op_region(F->ctx, OP_TYPE_WASMSSA_LOOP,
        attrs, na, NULL, 0, loop_regs, 1,
        n_res ? res_ty : NULL, n_res ? res_vals : NULL, n_res);
    for (size_t i = 0; i < n_res; i++) out_results[i] = res_vals[i];
    *out_n_results = n_res;
    return op;
}

// =============================================================================
// Emit `wasmssa.br {depth=N} (vals...)`.
// =============================================================================
static void emit_br(FnCtx *F, uint32_t depth, MLIR_ValueHandle *vals, size_t n) {
    MLIR_AttributeHandle as[2];
    as[0] = attr_i32(F->ctx, "valtype", 0);
    as[1] = attr_i32(F->ctx, "depth", (int64_t)depth);
    emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_BR, as, 2, vals, n));
}

// =============================================================================
// Emit a wasmssa.if(%cond) { br depth=D (vals) } else { fall through }
// shape, used by br_table and br_if lowering when br needs to carry
// values via the wasmssa block-arg discipline.
// =============================================================================
static void emit_br_if_to_depth(FnCtx *F, MLIR_ValueHandle cond, uint32_t depth) {
    MLIR_AttributeHandle as[2];
    as[0] = attr_i32(F->ctx, "valtype", 0);
    as[1] = attr_i32(F->ctx, "depth", (int64_t)depth);
    MLIR_ValueHandle ops[1] = { cond };
    emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_BR_IF, as, 2, ops, 1));
}

// =============================================================================
// Resolve the operand count that a wasm `br` to frame `f` consumes from
// the operand stack. For BLOCK / IF: equals the block's result arity.
// For LOOP: zero (loops in our void-blocktype world don't carry args;
// when they do, it would equal the loop's parameter arity).
// =============================================================================
static size_t frame_br_arity(Frame *f) {
    if (f->kind == F_LOOP) return 0;  // loop continue: no operands carried.
    return f->has_rtype ? 1 : 0;
}

// =============================================================================
// Lift one wasmstack.func body. Walks the flat opcode list, simulating
// the operand stack and rebuilding structured CF.
// =============================================================================
static bool lift_body(FnCtx *F, MLIR_BlockHandle src_blk) {
    size_t n_ops = MLIR_GetBlockNumOps(src_blk);

    for (size_t i = 0; i < n_ops; i++) {
        MLIR_OpHandle bo = MLIR_GetBlockOp(src_blk, i);
        MLIR_OpType t = MLIR_GetOpType(bo);
        uint8_t valtype = (uint8_t)at_i(bo, "valtype");
        {
            string opnm = MLIR_GetOpName(bo);
            static char buf[64];
            size_t sz = opnm.size < sizeof(buf) - 1 ? opnm.size : sizeof(buf) - 1;
            memcpy(buf, opnm.str, sz); buf[sz] = 0;
            g_current_op_name = buf;
        }

        // Honour `cur_terminated`: skip ops in dead code until we hit
        // an END or ELSE that resets the state.
        if (F->cur_terminated) {
            if (t == OP_TYPE_WASMSTACK_BLOCK ||
                t == OP_TYPE_WASMSTACK_LOOP ||
                t == OP_TYPE_WASMSTACK_IF) {
                // Push a fresh dead frame so we still track nesting.
                Frame nf = {0};
                nf.kind = (t == OP_TYPE_WASMSTACK_BLOCK) ? F_BLOCK :
                          (t == OP_TYPE_WASMSTACK_LOOP)  ? F_LOOP : F_IF;
                nf.body_block = MLIR_CreateBlock(F->ctx);
                nf.then_block = nf.body_block;
                nf.parent_block = F->cur;
                nf.terminated = true;
                nf.entry_stack_depth = F->stack.n;
                fs_push(&F->frames, nf);
                continue;
            }
            if (t == OP_TYPE_WASMSTACK_ELSE) {
                Frame *tf = fs_top(&F->frames);
                if (!tf || tf->kind != F_IF) continue;
                // Switch into else branch — fresh region body.
                tf->has_else = true;
                tf->else_block = MLIR_CreateBlock(F->ctx);
                tf->body_block = tf->else_block;
                F->cur = tf->else_block;
                F->cur_terminated = false;
                tf->terminated = false;
                st_truncate(&F->stack, tf->entry_stack_depth);
                continue;
            }
            if (t == OP_TYPE_WASMSTACK_END) {
                // If we're at the function-level end while in dead
                // code, the explicit `wasmstack.return` already emitted
                // everything. Nothing left to do.
                if (F->frames.n == 0) {
                    return true;
                }
                // Pop the current frame and switch back to parent.
                Frame top = fs_pop(&F->frames);
                MLIR_ValueHandle results[1];
                size_t n_results = 0;

                // If the body was actually dead the entire time, we
                // still need to terminate it with something the wmir
                // lowering can flatten. Add an unreachable if not
                // already terminated.
                if (top.kind == F_IF) {
                    if (!top.has_else) {
                        // unwrapped: synth a no-op then if needed.
                    }
                    // For each filled region body, append unreachable
                    // if absent.
                    MLIR_BlockHandle bodies[2] = { top.then_block,
                        top.has_else ? top.else_block : MLIR_INVALID_HANDLE };
                    for (int b = 0; b < 2; b++) {
                        if (!bodies[b]) continue;
                        size_t nn = MLIR_GetBlockNumOps(bodies[b]);
                        bool has_term = false;
                        if (nn > 0) {
                            MLIR_OpType lt = MLIR_GetOpType(
                                MLIR_GetBlockOp(bodies[b], nn - 1));
                            has_term = (lt == OP_TYPE_WASMSSA_RETURN ||
                                        lt == OP_TYPE_WASMSSA_UNREACHABLE ||
                                        lt == OP_TYPE_WASMSSA_BR ||
                                        lt == OP_TYPE_WASMSSA_BLOCK_RETURN);
                        }
                        if (!has_term) {
                            MLIR_AppendBlockOp(F->ctx, bodies[b],
                                build_op_no_res(F->ctx, OP_TYPE_WASMSSA_UNREACHABLE,
                                    NULL, 0, NULL, 0));
                        }
                    }
                } else {
                    size_t nn = MLIR_GetBlockNumOps(top.body_block);
                    bool has_term = false;
                    if (nn > 0) {
                        MLIR_OpType lt = MLIR_GetOpType(
                            MLIR_GetBlockOp(top.body_block, nn - 1));
                        has_term = (lt == OP_TYPE_WASMSSA_RETURN ||
                                    lt == OP_TYPE_WASMSSA_UNREACHABLE ||
                                    lt == OP_TYPE_WASMSSA_BR ||
                                    lt == OP_TYPE_WASMSSA_BLOCK_RETURN);
                    }
                    if (!has_term) {
                        MLIR_AppendBlockOp(F->ctx, top.body_block,
                            build_op_no_res(F->ctx, OP_TYPE_WASMSSA_UNREACHABLE,
                                NULL, 0, NULL, 0));
                    }
                }

                F->cur = top.parent_block;
                MLIR_OpHandle wrap = finalize_frame(F, &top, results, &n_results);
                MLIR_AppendBlockOp(F->ctx, F->cur, wrap);
                for (size_t k = 0; k < n_results; k++) st_push(&F->stack, results[k]);

                // After returning to the parent, are WE terminated?
                Frame *parent_top = fs_top(&F->frames);
                if (parent_top) {
                    F->cur_terminated = parent_top->terminated;
                } else {
                    F->cur_terminated = false;
                }
                continue;
            }
            // All other ops in dead code: discard.
            continue;
        }

        switch (t) {
        case OP_TYPE_WASMSTACK_LOCAL_GET: {
            uint32_t li = (uint32_t)at_i(bo, "local_idx");
            if (li >= F->nl_total) {
                fprintf(stderr, "ws->ssa: local.get %u out of range (%zu)\n",
                    li, F->nl_total);
                return false;
            }
            st_push(&F->stack, emit_local_load(F, li));
            break;
        }
        case OP_TYPE_WASMSTACK_LOCAL_SET: {
            uint32_t li = (uint32_t)at_i(bo, "local_idx");
            MLIR_ValueHandle v = st_pop(&F->stack);
            if (li >= F->nl_total) {
                fprintf(stderr, "ws->ssa: local.set %u out of range\n", li);
                return false;
            }
            emit_local_store(F, li, v);
            break;
        }
        case OP_TYPE_WASMSTACK_LOCAL_TEE: {
            uint32_t li = (uint32_t)at_i(bo, "local_idx");
            MLIR_ValueHandle v = st_peek(&F->stack);
            if (li >= F->nl_total) {
                fprintf(stderr, "ws->ssa: local.tee %u out of range\n", li);
                return false;
            }
            emit_local_store(F, li, v);
            break;
        }
        case OP_TYPE_WASMSTACK_GLOBAL_GET: {
            int64_t gi = at_i(bo, "global_idx");
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            as[1] = attr_i32(F->ctx, "global_idx", gi);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, valtype);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_GLOBAL_GET,
                as, 2, NULL, 0, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_GLOBAL_SET: {
            int64_t gi = at_i(bo, "global_idx");
            MLIR_ValueHandle v = st_pop(&F->stack);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            as[1] = attr_i32(F->ctx, "global_idx", gi);
            MLIR_ValueHandle ops[1] = { v };
            emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_GLOBAL_SET,
                as, 2, ops, 1));
            break;
        }
        case OP_TYPE_WASMSTACK_CONST: {
            int64_t v = at_i(bo, "value");
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            as[1] = attr_i64(F->ctx, "value", v);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, valtype);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_CONST,
                as, 2, NULL, 0, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_ADD:
        case OP_TYPE_WASMSTACK_SUB: {
            MLIR_OpType outt = (t == OP_TYPE_WASMSTACK_ADD)
                ? OP_TYPE_WASMSSA_ADD : OP_TYPE_WASMSSA_SUB;
            MLIR_ValueHandle a[2];
            pop_n(F, 2, a);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, valtype);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, outt, as, 1, a, 2, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_BINOP: {
            int64_t opc = at_i(bo, "wasm_opcode");
            MLIR_ValueHandle a[2];
            pop_n(F, 2, a);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            as[1] = attr_i32(F->ctx, "wasm_opcode", opc);
            // Compare ops always return i32.
            bool is_cmp = (opc >= 0x46 && opc <= 0x4f) ||  // i32 cmps
                          (opc >= 0x51 && opc <= 0x5a) ||  // i64 cmps
                          (opc >= 0x5b && opc <= 0x66);    // f32/f64 cmps
            uint8_t result_vt = is_cmp ? WT_I32 : valtype;
            MLIR_TypeHandle ty = vt_to_type(F->ctx, result_vt);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_BINOP,
                as, 2, a, 2, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_UNOP: {
            int64_t opc = at_i(bo, "wasm_opcode");
            MLIR_ValueHandle a[1];
            pop_n(F, 1, a);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            as[1] = attr_i32(F->ctx, "wasm_opcode", opc);
            // Determine result type from the conversion opcode.
            uint8_t rvt = valtype;
            switch (opc) {
            case 0xa7: rvt = WT_I32; break;  // i32.wrap_i64
            case 0xa8: case 0xa9: rvt = WT_I32; break;  // i32.trunc_f32_*
            case 0xaa: case 0xab: rvt = WT_I32; break;  // i32.trunc_f64_*
            case 0xac: case 0xad: rvt = WT_I64; break;  // i64.extend_i32_*
            case 0xae: case 0xaf: rvt = WT_I64; break;
            case 0xb0: case 0xb1: rvt = WT_I64; break;
            case 0xb2: case 0xb3: rvt = WT_F32; break;  // f32 conv ops
            case 0xb4: case 0xb5: rvt = WT_F32; break;
            case 0xb6: rvt = WT_F32; break;             // f32.demote_f64
            case 0xb7: case 0xb8: rvt = WT_F64; break;  // f64 conv ops
            case 0xb9: case 0xba: rvt = WT_F64; break;
            case 0xbb: rvt = WT_F64; break;             // f64.promote_f32
            case 0xbc: rvt = WT_I32; break;             // i32.reinterpret_f32
            case 0xbd: rvt = WT_I64; break;             // i64.reinterpret_f64
            case 0xbe: rvt = WT_F32; break;             // f32.reinterpret_i32
            case 0xbf: rvt = WT_F64; break;             // f64.reinterpret_i64
            case 0xc0: case 0xc1: rvt = WT_I32; break;  // i32.extend{8,16}_s
            case 0xc2: case 0xc3: case 0xc4: rvt = WT_I64; break;
            default: rvt = valtype; break;              // arithmetic unops keep type
            }
            MLIR_TypeHandle ty = vt_to_type(F->ctx, rvt);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_UNOP,
                as, 2, a, 1, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_EXTEND_I32_S: {
            // The decoder emits this as a follow-up to a sub-word signed
            // load (e.g., i32.load8_s) — extend a sub-word value sitting
            // on the stack to the full valtype width. Lift 1:1 to
            // wasmssa.extend_i32_s.
            MLIR_ValueHandle a[1];
            pop_n(F, 1, a);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, valtype);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_EXTEND_I32_S,
                as, 1, a, 1, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_EQZ: {
            MLIR_ValueHandle a[1];
            pop_n(F, 1, a);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, WT_I32);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_EQZ,
                as, 1, a, 1, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_SELECT: {
            // wasm select: pop cond, b, a, push (cond ? a : b).
            MLIR_ValueHandle cond = st_pop(&F->stack);
            MLIR_ValueHandle b = st_pop(&F->stack);
            MLIR_ValueHandle a = st_pop(&F->stack);
            MLIR_ValueHandle ops[3] = { a, b, cond };
            // Result type matches a / b. We use the i32 default if we
            // can't infer.
            MLIR_TypeHandle ty = MLIR_GetValueType(a);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_SELECT,
                as, 1, ops, 3, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_DROP: {
            (void)st_pop(&F->stack);
            break;
        }
        case OP_TYPE_WASMSTACK_LOAD: {
            int64_t moff = at_i(bo, "memory_offset");
            int64_t malign = at_i(bo, "memory_align_log2");
            int64_t msize = at_i(bo, "mem_size_bytes");
            MLIR_ValueHandle a[1];
            pop_n(F, 1, a);
            MLIR_AttributeHandle as[4];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            as[1] = attr_i32(F->ctx, "memory_offset", moff);
            as[2] = attr_i32(F->ctx, "memory_align_log2", malign);
            as[3] = attr_i32(F->ctx, "mem_size_bytes", msize);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, valtype);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_LOAD,
                as, 4, a, 1, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_STORE: {
            int64_t moff = at_i(bo, "memory_offset");
            int64_t malign = at_i(bo, "memory_align_log2");
            int64_t msize = at_i(bo, "mem_size_bytes");
            MLIR_ValueHandle val = st_pop(&F->stack);
            MLIR_ValueHandle addr = st_pop(&F->stack);
            MLIR_ValueHandle ops[2] = { addr, val };
            MLIR_AttributeHandle as[4];
            as[0] = attr_i32(F->ctx, "valtype", valtype);
            as[1] = attr_i32(F->ctx, "memory_offset", moff);
            as[2] = attr_i32(F->ctx, "memory_align_log2", malign);
            as[3] = attr_i32(F->ctx, "mem_size_bytes", msize);
            emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_STORE,
                as, 4, ops, 2));
            break;
        }
        case OP_TYPE_WASMSTACK_MEMORY_SIZE: {
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, WT_I32);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_MEMORY_SIZE,
                as, 1, NULL, 0, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_MEMORY_GROW: {
            MLIR_ValueHandle a[1];
            pop_n(F, 1, a);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, WT_I32);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_MEMORY_GROW,
                as, 1, a, 1, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_CALL: {
            string tgt = at_s(bo, "target");
            string cpt = (string){0}, crt = (string){0};
            if (!lookup_callee_sig(F->siblings, tgt, &cpt, &crt)) {
                fprintf(stderr, "ws->ssa: call target '%.*s' not found\n",
                    (int)tgt.size, tgt.str);
                return false;
            }
            size_t cnp = 0, cnr = 0;
            uint8_t *cp = hex_decode_arena(F->arena, cpt, &cnp);
            uint8_t *cr = hex_decode_arena(F->arena, crt, &cnr);
            (void)cp;
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena,
                (cnp ? cnp : 1) * sizeof(MLIR_ValueHandle));
            for (size_t k = cnp; k > 0; k--) opnds[k - 1] = st_pop(&F->stack);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            as[1] = attr_s(F->ctx, "target", tgt.str, tgt.size);
            if (cnr == 0) {
                emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_CALL,
                    as, 2, opnds, cnp));
            } else if (cnr == 1) {
                MLIR_TypeHandle rty = vt_to_type(F->ctx, cr[0]);
                MLIR_ValueHandle rv;
                emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_CALL,
                    as, 2, opnds, cnp, rty, &rv));
                st_push(&F->stack, rv);
            } else {
                fprintf(stderr, "ws->ssa: multi-result call not supported\n");
                return false;
            }
            break;
        }
        case OP_TYPE_WASMSTACK_CALL_INDIRECT: {
            int64_t tidx = at_i(bo, "type_idx");
            string ipt = at_s(bo, "sig_params");
            string irt = at_s(bo, "sig_results");
            size_t cnp = 0, cnr = 0;
            uint8_t *cp = hex_decode_arena(F->arena, ipt, &cnp);
            uint8_t *cr = hex_decode_arena(F->arena, irt, &cnr);
            (void)cp;
            MLIR_ValueHandle funcref = st_pop(&F->stack);
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena,
                (cnp + 1) * sizeof(MLIR_ValueHandle));
            for (size_t k = cnp; k > 0; k--) opnds[k - 1] = st_pop(&F->stack);
            opnds[cnp] = funcref;
            MLIR_AttributeHandle as[4];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            as[1] = attr_i32(F->ctx, "type_idx", tidx);
            as[2] = attr_s(F->ctx, "sig_params", ipt.str, ipt.size);
            as[3] = attr_s(F->ctx, "sig_results", irt.str, irt.size);
            if (cnr == 0) {
                emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_CALL_INDIRECT,
                    as, 4, opnds, cnp + 1));
            } else if (cnr == 1) {
                MLIR_TypeHandle rty = vt_to_type(F->ctx, cr[0]);
                MLIR_ValueHandle rv;
                emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_CALL_INDIRECT,
                    as, 4, opnds, cnp + 1, rty, &rv));
                st_push(&F->stack, rv);
            } else {
                fprintf(stderr, "ws->ssa: multi-result call_indirect not supported\n");
                return false;
            }
            break;
        }
        case OP_TYPE_WASMSTACK_RETURN: {
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena,
                (F->nr ? F->nr : 1) * sizeof(MLIR_ValueHandle));
            for (size_t k = F->nr; k > 0; k--) opnds[k - 1] = st_pop(&F->stack);
            emit_locals_pop(F);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_RETURN,
                as, 1, opnds, F->nr));
            F->cur_terminated = true;
            Frame *tf = fs_top(&F->frames);
            if (tf) tf->terminated = true;
            break;
        }
        case OP_TYPE_WASMSTACK_UNREACHABLE: {
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_UNREACHABLE,
                as, 1, NULL, 0));
            F->cur_terminated = true;
            Frame *tf = fs_top(&F->frames);
            if (tf) tf->terminated = true;
            break;
        }
        case OP_TYPE_WASMSTACK_BLOCK:
        case OP_TYPE_WASMSTACK_LOOP: {
            uint8_t bt = valtype;
            Frame nf = {0};
            nf.kind = (t == OP_TYPE_WASMSTACK_BLOCK) ? F_BLOCK : F_LOOP;
            nf.has_rtype = (bt != 0);
            nf.rtype_byte = bt;
            nf.parent_block = F->cur;
            nf.body_block = MLIR_CreateBlock(F->ctx);
            nf.then_block = nf.body_block;
            nf.entry_stack_depth = F->stack.n;
            nf.terminated = false;
            fs_push(&F->frames, nf);
            F->cur = nf.body_block;
            F->cur_terminated = false;
            break;
        }
        case OP_TYPE_WASMSTACK_IF: {
            uint8_t bt = valtype;
            MLIR_ValueHandle cond = st_pop(&F->stack);
            Frame nf = {0};
            nf.kind = F_IF;
            nf.has_rtype = (bt != 0);
            nf.rtype_byte = bt;
            nf.parent_block = F->cur;
            nf.then_block = MLIR_CreateBlock(F->ctx);
            nf.body_block = nf.then_block;
            nf.if_cond = cond;
            nf.has_else = false;
            nf.entry_stack_depth = F->stack.n;
            nf.terminated = false;
            fs_push(&F->frames, nf);
            F->cur = nf.then_block;
            F->cur_terminated = false;
            break;
        }
        case OP_TYPE_WASMSTACK_ELSE: {
            Frame *tf = fs_top(&F->frames);
            if (!tf || tf->kind != F_IF) {
                fprintf(stderr, "ws->ssa: stray else\n");
                return false;
            }
            // Terminate the then-region body if needed.
            MLIR_ValueHandle rets[1];
            size_t nrets = tf->has_rtype ? 1 : 0;
            for (size_t k = nrets; k > 0; k--) rets[k - 1] = st_pop(&F->stack);
            MLIR_AppendBlockOp(F->ctx, tf->then_block,
                build_op_no_res(F->ctx, OP_TYPE_WASMSSA_BLOCK_RETURN,
                    NULL, 0, rets, nrets));
            // Reset for else region.
            st_truncate(&F->stack, tf->entry_stack_depth);
            tf->has_else = true;
            tf->else_block = MLIR_CreateBlock(F->ctx);
            tf->body_block = tf->else_block;
            tf->terminated = false;
            F->cur = tf->else_block;
            F->cur_terminated = false;
            break;
        }
        case OP_TYPE_WASMSTACK_END: {
            // End of function or end of innermost frame.
            if (F->frames.n == 0) {
                // Function-level end. If the body has already issued
                // a wasmstack.return (so F->cur_terminated is set),
                // the value stack is empty by definition and we must
                // NOT pop F->nr times — that's an underflow. The
                // earlier return already emitted everything.
                if (F->cur_terminated) {
                    return true;
                }
                // Implicit return.
                MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena,
                    (F->nr ? F->nr : 1) * sizeof(MLIR_ValueHandle));
                for (size_t k = F->nr; k > 0; k--) opnds[k - 1] = st_pop(&F->stack);
                emit_locals_pop(F);
                MLIR_AttributeHandle as[1];
                as[0] = attr_i32(F->ctx, "valtype", 0);
                emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_RETURN,
                    as, 1, opnds, F->nr));
                F->cur_terminated = true;
                return true;
            }
            Frame top = fs_pop(&F->frames);
            // Terminate the current region body.
            MLIR_ValueHandle rets[1];
            size_t nrets = top.has_rtype ? 1 : 0;
            for (size_t k = nrets; k > 0; k--) rets[k - 1] = st_pop(&F->stack);
            MLIR_AppendBlockOp(F->ctx, top.body_block,
                build_op_no_res(F->ctx, OP_TYPE_WASMSSA_BLOCK_RETURN,
                    NULL, 0, rets, nrets));
            // If IF without else, synth empty else region (handled in finalize).
            // Pop back to parent, wrap, push results.
            F->cur = top.parent_block;
            MLIR_ValueHandle results[1];
            size_t n_results = 0;
            MLIR_OpHandle wrap = finalize_frame(F, &top, results, &n_results);
            MLIR_AppendBlockOp(F->ctx, F->cur, wrap);
            for (size_t k = 0; k < n_results; k++) st_push(&F->stack, results[k]);
            Frame *parent_top = fs_top(&F->frames);
            F->cur_terminated = parent_top ? parent_top->terminated : false;
            break;
        }
        case OP_TYPE_WASMSTACK_BR: {
            int64_t depth = at_i(bo, "depth");
            if ((size_t)depth >= F->frames.n) {
                fprintf(stderr, "ws->ssa: br depth %lld out of range\n",
                    (long long)depth);
                return false;
            }
            Frame *tgt = &F->frames.data[F->frames.n - 1 - (size_t)depth];
            size_t n_args = frame_br_arity(tgt);
            MLIR_ValueHandle args[1];
            for (size_t k = n_args; k > 0; k--) args[k - 1] = st_pop(&F->stack);
            emit_br(F, (uint32_t)depth, args, n_args);
            F->cur_terminated = true;
            Frame *tf = fs_top(&F->frames);
            if (tf) tf->terminated = true;
            break;
        }
        case OP_TYPE_WASMSTACK_BR_IF: {
            int64_t depth = at_i(bo, "depth");
            if ((size_t)depth >= F->frames.n) {
                fprintf(stderr, "ws->ssa: br_if depth %lld out of range\n",
                    (long long)depth);
                return false;
            }
            Frame *tgt = &F->frames.data[F->frames.n - 1 - (size_t)depth];
            MLIR_ValueHandle cond = st_pop(&F->stack);
            size_t n_args = frame_br_arity(tgt);
            if (n_args == 0) {
                emit_br_if_to_depth(F, cond, (uint32_t)depth);
            } else {
                // wasmssa.br_if carries only the cond (block-arg form).
                // For void-blocktype frames this is what we have. If a
                // br_if needs to carry a value, the stack-discipline of
                // wasm makes it fall through to the next op on the
                // FALSE path AND jump-with-value on the TRUE path. We
                // model this as: wasmssa.if(%cond) { br depth (vals) }.
                MLIR_ValueHandle args[1];
                for (size_t k = n_args; k > 0; k--) args[k - 1] = st_pop(&F->stack);
                // Build an inline wasmssa.if with one branch that brs.
                MLIR_BlockHandle then_b = MLIR_CreateBlock(F->ctx);
                MLIR_AttributeHandle bras[2];
                bras[0] = attr_i32(F->ctx, "valtype", 0);
                bras[1] = attr_i32(F->ctx, "depth", depth);
                MLIR_AppendBlockOp(F->ctx, then_b,
                    build_op_no_res(F->ctx, OP_TYPE_WASMSSA_BR,
                                    bras, 2, args, n_args));
                // Else region: just block_return.
                MLIR_BlockHandle else_b = MLIR_CreateBlock(F->ctx);
                MLIR_AppendBlockOp(F->ctx, else_b,
                    build_op_no_res(F->ctx, OP_TYPE_WASMSSA_BLOCK_RETURN,
                                    NULL, 0, NULL, 0));
                MLIR_RegionHandle tr = MLIR_CreateRegion(F->ctx);
                MLIR_AppendRegionBlock(F->ctx, tr, then_b);
                MLIR_RegionHandle er = MLIR_CreateRegion(F->ctx);
                MLIR_AppendRegionBlock(F->ctx, er, else_b);
                MLIR_RegionHandle regs[2] = { tr, er };
                MLIR_ValueHandle cops[1] = { cond };
                emit(F, build_op_region(F->ctx, OP_TYPE_WASMSSA_IF,
                    NULL, 0, cops, 1, regs, 2, NULL, NULL, 0));
                // Re-push the args so a subsequent br/return can consume them.
                for (size_t k = 0; k < n_args; k++) st_push(&F->stack, args[k]);
            }
            break;
        }
        case OP_TYPE_WASMSTACK_BR_TABLE: {
            string targets = at_s(bo, "targets");
            int64_t dflt = at_i(bo, "default");
            MLIR_ValueHandle idx = st_pop(&F->stack);
            // Parse target list.
            uint32_t tlist[256];
            uint32_t n_t = 0;
            const char *p = targets.str;
            const char *e = targets.str + targets.size;
            while (p < e && n_t < 256) {
                uint32_t v = 0;
                while (p < e && *p >= '0' && *p <= '9') {
                    v = v * 10 + (*p - '0'); p++;
                }
                tlist[n_t++] = v;
                if (p < e && *p == ',') p++;
            }
            // Emit nested wasmssa.if's: idx==0 ? br t0 : idx==1 ? br t1 : ... : br dflt.
            // We build them inside-out for simplicity: each level wraps
            // the previous else in another wasmssa.if.
            // Strategy: walk the idx tested against each i, emit a
            // wasmssa.if(eq) { br ti } else { ... }. The outermost gets
            // emitted to the current block, inner ones nest.
            //
            // Implementation: build the deepest else first as { br
            // default }, then wrap in successive ifs.
            //
            // For simplicity (and to avoid nesting limits in our type
            // system), use a flat chain: subtract idx by 1 each time
            // and use br_if (idx == 0).
            //
            // Approach: For k in 0..n_t-1:
            //   cmp = (idx == k)
            //   br_if depth=tk if cmp
            // Final: br depth=default
            //
            // This works because each br_if only fires when idx exactly
            // equals k, so at most one fires.
            //
            // BUT: this doesn't terminate the block — fall-through is
            // the default br at the end.
            for (uint32_t k = 0; k < n_t; k++) {
                // const k
                MLIR_AttributeHandle cas[2];
                cas[0] = attr_i32(F->ctx, "valtype", WT_I32);
                cas[1] = attr_i64(F->ctx, "value", (int64_t)k);
                MLIR_ValueHandle kv;
                emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_CONST,
                    cas, 2, NULL, 0, vt_to_type(F->ctx, WT_I32), &kv));
                // cmp = idx == k
                MLIR_AttributeHandle bas[2];
                bas[0] = attr_i32(F->ctx, "valtype", WT_I32);
                bas[1] = attr_i32(F->ctx, "wasm_opcode", WOP_I32_EQ);
                MLIR_ValueHandle binops[2] = { idx, kv };
                MLIR_ValueHandle cmp;
                emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_BINOP,
                    bas, 2, binops, 2, vt_to_type(F->ctx, WT_I32), &cmp));
                // br_if depth=tlist[k]
                emit_br_if_to_depth(F, cmp, tlist[k]);
            }
            // Final: br to default depth.
            emit_br(F, (uint32_t)dflt, NULL, 0);
            F->cur_terminated = true;
            Frame *tf = fs_top(&F->frames);
            if (tf) tf->terminated = true;
            break;
        }
        case OP_TYPE_WASMSTACK_ADDRESSOF: {
            // Should not appear in wasm-lifted modules (it's a wasmssa
            // synth op that the wasmstack stage normally inlines). If
            // we see one, pass it through.
            string target = at_s(bo, "target");
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            as[1] = attr_s(F->ctx, "target", target.str, target.size);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, WT_I32);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_ADDRESSOF,
                as, 2, NULL, 0, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_FUNC_ADDR: {
            string target = at_s(bo, "target");
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(F->ctx, "valtype", 0);
            as[1] = attr_s(F->ctx, "target", target.str, target.size);
            MLIR_TypeHandle ty = vt_to_type(F->ctx, WT_I32);
            MLIR_ValueHandle rv;
            emit(F, build_op_1res(F->ctx, OP_TYPE_WASMSSA_FUNC_ADDR,
                as, 2, NULL, 0, ty, &rv));
            st_push(&F->stack, rv);
            break;
        }
        default: {
            string nm = MLIR_GetOpName(bo);
            fprintf(stderr,
                "ws->ssa: unsupported wasmstack op '%.*s' (kind=%d)\n",
                (int)nm.size, nm.str, (int)t);
            return false;
        }
        }
    }

    // Function-level walk-off: implicit return if not already terminated.
    if (!F->cur_terminated) {
        MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena,
            (F->nr ? F->nr : 1) * sizeof(MLIR_ValueHandle));
        for (size_t k = F->nr; k > 0; k--) opnds[k - 1] = st_pop(&F->stack);
        emit_locals_pop(F);
        MLIR_AttributeHandle as[1];
        as[0] = attr_i32(F->ctx, "valtype", 0);
        emit(F, build_op_no_res(F->ctx, OP_TYPE_WASMSSA_RETURN,
            as, 1, opnds, F->nr));
    }
    return true;
}

// =============================================================================
// Lift one wasmstack.func to a wasmssa.func.
// =============================================================================
static MLIR_OpHandle lift_func(MLIR_Context *ctx, Arena *arena,
                               MLIR_OpHandle src, MLIR_BlockHandle siblings) {
    // Transient working arena for this function. Allocations from
    // `tmp` (operand-array scratch, hex-decoded type lists, the
    // sym-name buf used for g_current_fn_name diagnostics) all
    // become dead the moment lift_func returns: MLIR_CreateOp /
    // _CreateValue / _AppendBlock* copy their inputs into the
    // context arena, so nothing we write here escapes into the IR.
    //
    // Without a transient arena, every function's working bytes
    // accumulate forever in ctx->arena (which also holds the
    // permanent IR), and lifting a real-sized linked wasm
    // (thousands of functions) blows past the 4 GiB heap. With a
    // transient arena, peak memory is bounded by the largest
    // function instead.
    Arena *tmp = arena_create(64 * 1024);
    (void)arena;  // arena parameter retained for ABI; tmp is used.

    string sym_name      = at_s(src, "sym_name");
    string pt_s          = at_s(src, "param_types");
    string rt_s          = at_s(src, "result_types");
    bool   exported      = at_b(src, "exported");
    string local_types_s = at_s(src, "local_types");

    size_t np = 0, nr = 0, nl_total = 0;
    uint8_t *params  = hex_decode_arena(tmp, pt_s, &np);
    uint8_t *results = hex_decode_arena(tmp, rt_s, &nr);
    uint8_t *locals  = hex_decode_arena(tmp, local_types_s, &nl_total);
    (void)results;

    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);

    // Add block args for params.
    MLIR_ValueHandle *param_args = NULL;
    if (np) {
        param_args = (MLIR_ValueHandle *)arena_alloc(tmp,
            np * sizeof(MLIR_ValueHandle));
    }
    for (size_t i = 0; i < np; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, params[i]);
        MLIR_ValueHandle a = MLIR_CreateValueBlockArg(ctx, (string){0},
            (uint32_t)i, ty, MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, entry, a);
        param_args[i] = a;
    }

    // Compute per-local byte offsets and total locals size (8-byte stride
    // so any vt fits). Locals layout: index 0 at offset 0, index 1 at
    // offset 8, etc. Wasm linmem grows DOWN from SP so the cells appear
    // contiguously above the current SP.
    uint32_t *local_offs = NULL;
    uint32_t locals_size = 0;
    if (nl_total) {
        local_offs = (uint32_t *)arena_alloc(tmp,
            nl_total * sizeof(uint32_t));
        for (size_t i = 0; i < nl_total; i++) {
            local_offs[i] = (uint32_t)(i * LOCAL_CELL_BYTES);
        }
        locals_size = (uint32_t)(nl_total * LOCAL_CELL_BYTES);
        // Round up to 16 for stack alignment paranoia.
        locals_size = (locals_size + 15u) & ~15u;
    }

    FnCtx F = {0};
    F.ctx = ctx;
    F.arena = tmp;
    F.siblings = siblings;
    {
        // Need a stable C string; sym_name is not NUL-terminated. The
        // pointer goes into the diagnostic global g_current_fn_name,
        // which is read by st_pop on underflow — only while we're
        // inside this lift_func call. Clearing the global before
        // arena_destroy keeps it from dangling.
        char *buf = (char *)arena_alloc(tmp, sym_name.size + 1);
        memcpy(buf, sym_name.str, sym_name.size);
        buf[sym_name.size] = '\0';
        g_current_fn_name = buf;
    }
    F.nl_total = nl_total;
    F.local_types = locals;
    F.local_offs = local_offs;
    F.locals_size = locals_size;
    F.nr = nr;
    F.cur = entry;
    F.cur_terminated = false;

    // Synth function prologue: allocate locals region from wasm SP
    // (global 0). %bp = global_get(0) - locals_size; global_set(0, %bp).
    if (locals_size > 0) {
        MLIR_AttributeHandle ggas[2];
        ggas[0] = attr_i32(ctx, "valtype", WT_I32);
        ggas[1] = attr_i32(ctx, "global_idx", 0);
        MLIR_ValueHandle sp;
        emit(&F, build_op_1res(ctx, OP_TYPE_WASMSSA_GLOBAL_GET,
            ggas, 2, NULL, 0, vt_to_type(ctx, WT_I32), &sp));

        MLIR_AttributeHandle cas[2];
        cas[0] = attr_i32(ctx, "valtype", WT_I32);
        cas[1] = attr_i64(ctx, "value", (int64_t)locals_size);
        MLIR_ValueHandle sz;
        emit(&F, build_op_1res(ctx, OP_TYPE_WASMSSA_CONST,
            cas, 2, NULL, 0, vt_to_type(ctx, WT_I32), &sz));

        MLIR_AttributeHandle sas[1];
        sas[0] = attr_i32(ctx, "valtype", WT_I32);
        MLIR_ValueHandle sops[2] = { sp, sz };
        MLIR_ValueHandle bp;
        emit(&F, build_op_1res(ctx, OP_TYPE_WASMSSA_SUB,
            sas, 1, sops, 2, vt_to_type(ctx, WT_I32), &bp));
        F.locals_bp = bp;

        MLIR_AttributeHandle gsas[2];
        gsas[0] = attr_i32(ctx, "valtype", WT_I32);
        gsas[1] = attr_i32(ctx, "global_idx", 0);
        MLIR_ValueHandle gsops[1] = { bp };
        emit(&F, build_op_no_res(ctx, OP_TYPE_WASMSSA_GLOBAL_SET,
            gsas, 2, gsops, 1));

        // Store each param into its local cell. Locals beyond the params
        // are left uninitialised — wasm semantics requires them to start
        // at zero but wasm-ld emits explicit `local.set 0` for each one
        // before reading. (If that turns out to be wrong we can zero-
        // initialise here.)
        for (size_t i = 0; i < np; i++) {
            emit_local_store(&F, (uint32_t)i, param_args[i]);
        }

        // Defensive: zero-initialise non-param locals so reads-before-
        // writes return 0 (wasm semantics). wasm-ld usually emits an
        // explicit `local.set` before each read anyway, but being
        // paranoid avoids subtle bugs.
        for (size_t i = np; i < nl_total; i++) {
            MLIR_AttributeHandle zas[2];
            zas[0] = attr_i32(ctx, "valtype", locals[i]);
            zas[1] = attr_i64(ctx, "value", 0);
            MLIR_TypeHandle zty = vt_to_type(ctx, locals[i]);
            MLIR_ValueHandle z;
            emit(&F, build_op_1res(ctx, OP_TYPE_WASMSSA_CONST,
                zas, 2, NULL, 0, zty, &z));
            emit_local_store(&F, (uint32_t)i, z);
        }
    }

    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(MLIR_GetOpRegion(src, 0), 0);
    bool ok = lift_body(&F, src_blk);

    st_free(&F.stack);
    fs_free(&F.frames);

    // Diagnostic globals point into `tmp`; null them out before destroy.
    g_current_fn_name = NULL;
    g_current_op_name = NULL;
    arena_destroy(tmp);

    if (!ok) return MLIR_INVALID_HANDLE;

    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", sym_name.str, sym_name.size);
    attrs[na++] = attr_s(ctx, "param_types", pt_s.str, pt_s.size);
    attrs[na++] = attr_s(ctx, "result_types", rt_s.str, rt_s.size);
    attrs[na++] = attr_b(ctx, "exported", exported);
    attrs[na++] = attr_b(ctx, "internal", false);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, entry);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_FUNC,
        op_type_to_string(OP_TYPE_WASMSSA_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Lift one wasmstack.import_func to a wasmssa.import_func.
// =============================================================================
static MLIR_OpHandle lift_import_func(MLIR_Context *ctx, MLIR_OpHandle src) {
    string sym_name = at_s(src, "sym_name");
    string pt_s = at_s(src, "param_types");
    string rt_s = at_s(src, "result_types");
    bool exported = at_b(src, "exported");
    MLIR_AttributeHandle attrs[4];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", sym_name.str, sym_name.size);
    attrs[na++] = attr_s(ctx, "param_types", pt_s.str, pt_s.size);
    attrs[na++] = attr_s(ctx, "result_types", rt_s.str, rt_s.size);
    attrs[na++] = attr_b(ctx, "exported", exported);
    return MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_IMPORT_FUNC,
        op_type_to_string(OP_TYPE_WASMSSA_IMPORT_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Lift wasmstack.func that maps to a synth helper -> import_func decl.
// =============================================================================
static MLIR_OpHandle synth_to_import(MLIR_Context *ctx, MLIR_OpHandle src) {
    string sym_name = at_s(src, "sym_name");
    string pt_s = at_s(src, "param_types");
    string rt_s = at_s(src, "result_types");
    MLIR_AttributeHandle attrs[4];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", sym_name.str, sym_name.size);
    attrs[na++] = attr_s(ctx, "param_types", pt_s.str, pt_s.size);
    attrs[na++] = attr_s(ctx, "result_types", rt_s.str, rt_s.size);
    attrs[na++] = attr_b(ctx, "exported", false);
    return MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_IMPORT_FUNC,
        op_type_to_string(OP_TYPE_WASMSSA_IMPORT_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Lift wasmstack.data_segment to wasmssa.import_global with the
// explicit `fixed_offset` attribute the wmir backend now honours.
// =============================================================================
static MLIR_OpHandle lift_data_segment(MLIR_Context *ctx, MLIR_OpHandle src,
                                       uint32_t seg_idx) {
    int64_t off = at_i(src, "offset");
    int64_t sz  = at_i(src, "size");
    string  id  = at_s(src, "init_data");

    // The sym_name pointer is stored in the IR attribute table without
    // being copied, so it must outlive this function. Allocate from
    // the context's arena instead of using a stack buffer.
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char *name_buf = (char *)arena_alloc(arena, 32);
    int nl = snprintf(name_buf, 32, "__wasm_data_%u", seg_idx);

    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", name_buf, (size_t)nl);
    attrs[na++] = attr_i32(ctx, "size", sz);
    attrs[na++] = attr_i32(ctx, "align_pow", 0);
    attrs[na++] = attr_b(ctx, "is_const", false);
    attrs[na++] = attr_s(ctx, "init_data", id.str, id.size);
    attrs[na++] = attr_s(ctx, "relocs", "", 0);
    attrs[na++] = attr_i32(ctx, "fixed_offset", off);
    return MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_IMPORT_GLOBAL,
        op_type_to_string(OP_TYPE_WASMSSA_IMPORT_GLOBAL),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Top-level entry point.
// =============================================================================
MLIR_OpHandle mlir_wasmstack_to_wasmssa(MLIR_Context *ctx,
                                        MLIR_OpHandle stack_module) {
    if (!stack_module) return MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(stack_module) < 1) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(stack_module, 0);
    if (MLIR_GetRegionNumBlocks(mr) < 1) return MLIR_INVALID_HANDLE;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);

    Arena *arena = MLIR_GetArenaAllocator(ctx);

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);
    MLIR_RegionHandle out_regs[1] = { out_region };

    // Propagate `memory_min_pages` from the wasmstack module so the
    // wasmssa -> wmir -> aarch64 chain can size the linear memory image
    // correctly (see mlir_wasm_to_wasmstack.c for the rationale).
    MLIR_AttributeHandle mod_attrs[1];
    size_t n_mod_attrs = 0;
    MLIR_AttributeHandle a_min_pages = MLIR_GetOpAttributeByName(
        stack_module, "memory_min_pages");
    if (a_min_pages) {
        mod_attrs[n_mod_attrs++] = attr_i32(ctx, "memory_min_pages",
            MLIR_GetAttributeInteger(a_min_pages));
    }

    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        mod_attrs, n_mod_attrs, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    // First pass: rewrite the sibling block so that synth-helper bodies
    // get replaced with import_funcs IN-PLACE before we lift. That way
    // when other functions look up signatures via `lookup_callee_sig`,
    // they find the import_func form (which still carries the same
    // sym_name and types).
    //
    // We can't actually rewrite the input module, so instead we build a
    // staging list of "lifted" siblings whose sym_name + param/result
    // types match the input — but every synth helper is an import_func.
    // We pass this staging list as the `siblings` block to lift_func.
    //
    // For simplicity, build a small dispatch table here using the
    // input mb as the siblings source — lookup_callee_sig only inspects
    // sym_name + param_types + result_types which are present on BOTH
    // wasmstack.func and wasmstack.import_func.
    size_t n_top = MLIR_GetBlockNumOps(mb);
    uint32_t data_idx = 0;

    // Collect ELEM entries (slot, target) so we can emit a synthetic
    // helper after all user functions. The helper is a no-op at runtime
    // (it's not called) but its wasmssa.func_addr ops with explicit
    // {slot=…} attributes feed the wmir prepass with the actual wasm
    // table slot indices.
    typedef struct { int64_t slot; string target; } ElemEntry;
    ElemEntry *elems = NULL;
    size_t n_elems = 0, elems_cap = 0;

    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType tt = MLIR_GetOpType(top);
        MLIR_OpHandle out_op = MLIR_INVALID_HANDLE;
        if (tt == OP_TYPE_WASMSTACK_IMPORT_FUNC) {
            out_op = lift_import_func(ctx, top);
        } else if (tt == OP_TYPE_WASMSTACK_FUNC) {
            string nm = at_s(top, "sym_name");
            if (is_synth_helper(nm)) {
                out_op = synth_to_import(ctx, top);
            } else {
                out_op = lift_func(ctx, arena, top, mb);
            }
        } else if (tt == OP_TYPE_WASMSTACK_DATA_SEGMENT) {
            out_op = lift_data_segment(ctx, top, data_idx++);
        } else if (tt == OP_TYPE_WASMSTACK_GLOBAL_DECL) {
            // The wmir backend manages global slots automatically; we
            // don't need to forward the declaration. (Initial values
            // are handled by synth_start for global 0; others default
            // to zero.)
            continue;
        } else if (tt == OP_TYPE_WASMSTACK_FUNC_ADDR_DECL) {
            if (n_elems == elems_cap) {
                elems_cap = elems_cap ? elems_cap * 2 : 8;
                elems = (ElemEntry *)realloc(elems,
                    elems_cap * sizeof(ElemEntry));
            }
            elems[n_elems].slot = at_i(top, "slot");
            elems[n_elems].target = at_s(top, "target");
            n_elems++;
            continue;
        } else {
            string nm = MLIR_GetOpName(top);
            fprintf(stderr, "ws->ssa: unexpected top-level op '%.*s'\n",
                (int)nm.size, nm.str);
            free(elems);
            return MLIR_INVALID_HANDLE;
        }
        if (!out_op) { free(elems); return MLIR_INVALID_HANDLE; }
        MLIR_AppendBlockOp(ctx, out_body, out_op);
    }

    // Emit the synthetic fnptr-init helper if we saw any ELEM entries.
    // The function name uses the leading underscore prefix so it
    // doesn't collide with user code, and is marked `internal` so the
    // backend can omit a stub if it wishes.
    if (n_elems > 0) {
        MLIR_BlockHandle body = MLIR_CreateBlock(ctx);
        for (size_t e = 0; e < n_elems; e++) {
            MLIR_AttributeHandle as[3];
            as[0] = attr_i32(ctx, "valtype", 0);
            as[1] = attr_s(ctx, "target",
                elems[e].target.str, elems[e].target.size);
            as[2] = attr_i32(ctx, "slot", elems[e].slot);
            MLIR_TypeHandle ty = vt_to_type(ctx, WT_I32);
            MLIR_ValueHandle rv;
            MLIR_AppendBlockOp(ctx, body, build_op_1res(ctx,
                OP_TYPE_WASMSSA_FUNC_ADDR, as, 3, NULL, 0, ty, &rv));
        }
        MLIR_AttributeHandle ras[1];
        ras[0] = attr_i32(ctx, "valtype", 0);
        MLIR_AppendBlockOp(ctx, body, build_op_no_res(ctx,
            OP_TYPE_WASMSSA_RETURN, ras, 1, NULL, 0));

        MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
        MLIR_AppendRegionBlock(ctx, reg, body);
        MLIR_RegionHandle regs[1] = { reg };

        MLIR_AttributeHandle fas[5];
        const char *nm = "_tinyc_fnptr_init";
        fas[0] = attr_s(ctx, "sym_name", nm, strlen(nm));
        fas[1] = attr_s(ctx, "param_types", "", 0);
        fas[2] = attr_s(ctx, "result_types", "", 0);
        fas[3] = attr_b(ctx, "exported", false);
        fas[4] = attr_b(ctx, "internal", true);

        MLIR_OpHandle fop = MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_FUNC,
            op_type_to_string(OP_TYPE_WASMSSA_FUNC),
            fas, 5, NULL, 0, NULL, 0, NULL, 0, regs, 1,
            MLIR_CreateLocationUnknown(ctx, (string){0}),
            MLIR_INVALID_HANDLE, (string){0}, -1);
        MLIR_AppendBlockOp(ctx, out_body, fop);
    }
    free(elems);
    return out_module;
}
