// tests/cross/driver.c
//
// Cross-implementation smoke test for the public MLIR C API in
// `mlir_api.h`. Builds a small but non-trivial module containing a
// func.func with two block arguments, an arith.addi, and a
// func.return, then prints it via the generic printer. Compiled twice:
//
//   * `cross_native` — backed by mlir_api_impl.c (corec arena, no libc).
//   * `cross_upstream` — backed by mlir_api_impl_upstream.cpp (upstream
//     LLVM/MLIR C++, system libc).
//
// Both binaries must produce byte-identical output.

#include <base/arena.h>
#include <base/string.h>
#include <base/io.h>
#include <base/format.h>
#include <platform/platform.h>
#include "mlir_api.h"
#include "mlir_passes.h"
#include "mlir_generic_printer.h"

// Parser-only type constructors (mlir_parser.h), for the uniquing checks.
MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *ctx, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space);
MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *ctx, string name);

static void print_string(string s) {
    ciovec_t iov;
    iov.buf = s.str;
    iov.buf_len = s.size;
    write_all(PLATFORM_STDOUT_FD, &iov, 1);
}

// Keep the helpers below out of line so their stack frames really are gone
// (and reused) by the time the attributes are read back.
#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

// Build an array attribute and a dictionary attribute from element arrays
// that live in this function's stack frame. The attributes must stay valid
// after the frame is gone: the API copies the elements, as upstream's
// ArrayAttr / DictionaryAttr do. The native backend used to keep the
// caller's pointer, which made `tinyc_native --emit=lowered` crash on the
// `passthrough` attribute built by lower_func_func.
NOINLINE
static void make_stack_array_attrs(MLIR_Context *ctx, MLIR_TypeHandle i32_ty,
                                   MLIR_AttributeHandle *out_array,
                                   MLIR_AttributeHandle *out_dict) {
    MLIR_AttributeHandle elems[3] = {
        MLIR_CreateAttributeInteger(ctx, 11, i32_ty),
        MLIR_CreateAttributeInteger(ctx, 22, i32_ty),
        MLIR_CreateAttributeInteger(ctx, 33, i32_ty),
    };
    *out_array = MLIR_CreateAttributeArray(ctx, elems, 3);
    // Keys in sorted order: upstream's DictionaryAttr sorts its entries.
    // The key text also lives in this frame: attribute names are copied too.
    char keys[2] = { 'a', 'b' };
    MLIR_NamedAttr entries[2] = {
        MLIR_Named(MLIR_InternAttrKey(ctx, (string){ &keys[0], 1 }), MLIR_CreateAttributeInteger(ctx, 44, i32_ty)),
        MLIR_Named(MLIR_InternAttrKey(ctx, (string){ &keys[1], 1 }), MLIR_CreateAttributeInteger(ctx, 55, i32_ty)),
    };
    *out_dict = MLIR_CreateAttributeDict(ctx, entries, 2);
}

// Overwrite the stack area the previous call used.
NOINLINE
static void clobber_stack(void) {
    volatile uintptr_t junk[256];
    for (size_t i = 0; i < 256; i++) junk[i] = (uintptr_t)0xdeadbeefu;
    (void)junk[0];
}

// Diagnostic handler that records the last diagnostic (see app_main).
typedef struct {
    int count;
    bool saw_error;
    MLIR_DiagSeverity severity;
    MLIR_LocationHandle loc;
    char text[128];
    size_t len;
} DiagCapture;

typedef struct {
    int count;
    MLIR_OpHandle erase_target;
    MLIR_OpHandle skip_target;
    MLIR_OpHandle stop_target;
} WalkState;

static MLIR_WalkResult test_walk_cb(MLIR_Context *ctx, MLIR_OpHandle op, void *user) {
    WalkState *st = (WalkState *)user;
    st->count++;
    if (st->skip_target != MLIR_INVALID_HANDLE && op == st->skip_target)
        return MLIR_WALK_SKIP;
    if (st->stop_target != MLIR_INVALID_HANDLE && op == st->stop_target)
        return MLIR_WALK_INTERRUPT;
    if (st->erase_target != MLIR_INVALID_HANDLE && op == st->erase_target)
        MLIR_EraseOp(ctx, op);
    return MLIR_WALK_ADVANCE;
}

static void capture_diag(void *user, MLIR_DiagSeverity severity,
                         MLIR_LocationHandle loc, string message) {
    DiagCapture *c = (DiagCapture *)user;
    c->count++;
    if (severity == MLIR_DIAG_ERROR) c->saw_error = true;
    c->severity = severity;
    c->loc = loc;
    c->len = message.size < sizeof(c->text) ? message.size : sizeof(c->text);
    for (size_t i = 0; i < c->len; i++) c->text[i] = message.str[i];
}

typedef struct {
    int module_runs;
    int fn_runs;
    int saw_scratch;
} PassTestState;

static bool test_mod_pass(MLIR_PassContext *pc, MLIR_OpHandle mod, void *state) {
    PassTestState *st = (PassTestState *)state;
    st->module_runs++;
    if (pc->scratch && arena_alloc(pc->scratch, 64) != NULL) st->saw_scratch++;
    (void)mod;
    return true;
}

static bool test_fn_pass(MLIR_PassContext *pc, MLIR_OpHandle fn, void *state) {
    PassTestState *st = (PassTestState *)state;
    st->fn_runs++;
    if (pc->scratch && arena_alloc(pc->scratch, 64) != NULL) st->saw_scratch++;
    (void)fn;
    return true;
}

static bool test_fail_pass(MLIR_PassContext *pc, MLIR_OpHandle mod, void *state) {
    (void)pc; (void)mod; (void)state;
    return false;
}

static bool test_break_ir_pass(MLIR_PassContext *pc, MLIR_OpHandle fn, void *state) {
    (void)state;
    // Append an invalid 0-result llvm.add to trigger verify_each failure.
    MLIR_BlockHandle blk = MLIR_GetRegionFirstBlock(pc->ctx, MLIR_GetOpRegion(pc->ctx, fn, 0));
    MLIR_OpSpec bad = { OP_TYPE_UNREGISTERED };
    bad.type = OP_TYPE_LLVM_ADD;
    MLIR_InsertOpBefore(pc->ctx, blk, MLIR_CreateOpFromSpec(pc->ctx, &bad), MLIR_GetBlockFirstOp(pc->ctx, blk));
    return true;
}

int app_main(void) {
    Arena *arena = arena_create(1 * 1024 * 1024);
    MLIR_ContextOptions options = {0};
    options.arena = arena;
    MLIR_Context *ctx = MLIR_CreateContext(&options);

    MLIR_TypeHandle i32_ty = MLIR_CreateTypeInteger(ctx, 32, true);

    {
        MLIR_AttributeHandle arr = MLIR_INVALID_HANDLE;
        MLIR_AttributeHandle dict = MLIR_INVALID_HANDLE;
        make_stack_array_attrs(ctx, i32_ty, &arr, &dict);
        clobber_stack();
        print_string(str_lit("array attr from stack buffer:"));
        for (size_t i = 0; i < MLIR_GetAttributeArraySize(ctx, arr); i++) {
            int64_t v = MLIR_GetAttributeInteger(ctx, MLIR_GetAttributeArrayElement(ctx, arr, i));
            print_string(format(arena, str_lit(" {}"), v));
        }
        print_string(str_lit("\ndict attr from stack buffer:"));
        for (size_t i = 0; i < MLIR_GetAttributeDictSize(ctx, dict); i++) {
            MLIR_AttributeHandle e = MLIR_GetAttributeDictElement(ctx, dict, i);
            print_string(format(arena, str_lit(" {}={}"), MLIR_GetAttrKeyName(ctx, MLIR_GetAttributeDictKey(ctx, dict, i)),
                                MLIR_GetAttributeInteger(ctx, e)));
        }
        print_string(str_lit("\n"));
    }

    // Verify that semantically-equivalent simple types share a single
    // MLIR_TypeHandle. The upstream backend has always done this (MLIR
    // interns IntegerType / IndexType / ptr / etc. via `::get`); the
    // native backend used to allocate a fresh IR_Type per call, which
    // broke type-handle equality checks in the cf->scf lifter
    // (get_undef_value / get_switch_value caches) and produced extra
    // llvm.mlir.undef ops when the same i32 was looked up by handle.
    // See mlir_api_impl.c's intern_type for the native fix.
    {
        MLIR_TypeHandle i32_a = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_TypeHandle i32_b = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_TypeHandle i64_a = MLIR_CreateTypeInteger(ctx, 64, false);
        MLIR_TypeHandle i64_b = MLIR_CreateTypeInteger(ctx, 64, false);
        MLIR_TypeHandle idx_a = MLIR_CreateTypeIndex(ctx);
        MLIR_TypeHandle idx_b = MLIR_CreateTypeIndex(ctx);
        MLIR_TypeHandle ptr_a = MLIR_CreateTypeLLVMPointer(ctx);
        MLIR_TypeHandle ptr_b = MLIR_CreateTypeLLVMPointer(ctx);
        print_string(i32_a == i32_b ? str_lit("intern i32: yes\n") : str_lit("intern i32: no\n"));
        print_string(i64_a == i64_b ? str_lit("intern i64: yes\n") : str_lit("intern i64: no\n"));
        print_string(idx_a == idx_b ? str_lit("intern index: yes\n") : str_lit("intern index: no\n"));
        print_string(ptr_a == ptr_b ? str_lit("intern !llvm.ptr: yes\n") : str_lit("intern !llvm.ptr: no\n"));
        print_string(i32_a == i64_a ? str_lit("intern i32==i64: yes\n") : str_lit("intern i32==i64: no\n"));
    }

    // Type introspection (#349): kinds, widths, elements and element counts
    // are the same on both backends, and equal types are equal handles
    // (identified structs: equal names).
    {
        MLIR_TypeHandle i8 = MLIR_CreateTypeInteger(ctx, 8, true);
        MLIR_TypeHandle f32 = MLIR_CreateTypeFloat(ctx, 32, false);
        MLIR_TypeHandle bf16 = MLIR_CreateTypeFloat(ctx, 16, true);
        MLIR_TypeHandle ptr = MLIR_CreateTypeLLVMPointer(ctx);
        MLIR_TypeHandle arr = MLIR_CreateTypeLLVMArray(ctx, i8, 12);
        int64_t vshape[2] = { 2, 4 };
        MLIR_TypeHandle vec = MLIR_CreateTypeVector(ctx, vshape, 2, f32);
        int64_t tshape[2] = { 3, -1 };
        MLIR_TypeHandle ten = MLIR_CreateTypeTensor(ctx, tshape, 2, i8);
        print_string(format(arena, str_lit("type kinds: i8={}/{} f32={}/{} bf16={}/{}"),
            (int64_t)MLIR_GetTypeKind(ctx, i8), (int64_t)MLIR_GetTypeWidth(ctx, i8),
            (int64_t)MLIR_GetTypeKind(ctx, f32), (int64_t)MLIR_GetTypeWidth(ctx, f32),
            (int64_t)MLIR_GetTypeKind(ctx, bf16), (int64_t)MLIR_GetTypeWidth(ctx, bf16)));
        print_string(format(arena, str_lit(" ptr={}/{} index={} invalid={}\n"),
            (int64_t)MLIR_GetTypeKind(ctx, ptr), (int64_t)MLIR_GetTypeWidth(ctx, ptr),
            (int64_t)MLIR_GetTypeKind(ctx, MLIR_CreateTypeIndex(ctx)),
            (int64_t)MLIR_GetTypeKind(ctx, MLIR_INVALID_HANDLE)));
        print_string(format(arena, str_lit("array: kind={} elem={} n={} rank={}"),
            (int64_t)MLIR_GetTypeKind(ctx, arr),
            MLIR_GetTypeElement(ctx, arr) == i8 ? str_lit("i8") : str_lit("?"),
            (int64_t)MLIR_GetTypeNumElements(ctx, arr), (int64_t)MLIR_GetTypeShapeRank(ctx, arr)));
        print_string(format(arena, str_lit("; vector {}: kind={} elem={} n={} dims={}x{}"),
            MLIR_GetTypeString(ctx, vec), (int64_t)MLIR_GetTypeKind(ctx, vec),
            MLIR_GetTypeElement(ctx, vec) == f32 ? str_lit("f32") : str_lit("?"),
            (int64_t)MLIR_GetTypeNumElements(ctx, vec),
            MLIR_GetTypeShapeDim(ctx, vec, 0), MLIR_GetTypeShapeDim(ctx, vec, 1)));
        print_string(format(arena, str_lit("; {}: rank={} dims={},{} n={}\n"),
            MLIR_GetTypeString(ctx, ten), (int64_t)MLIR_GetTypeShapeRank(ctx, ten),
            MLIR_GetTypeShapeDim(ctx, ten, 0), MLIR_GetTypeShapeDim(ctx, ten, 1),
            (int64_t)MLIR_GetTypeNumElements(ctx, ten)));
        int64_t vshape2[2] = { 2, 4 };
        print_string(MLIR_CreateTypeVector(ctx, vshape2, 2, f32) == vec &&
                     MLIR_CreateTypeLLVMArray(ctx, i8, 12) == arr &&
                     MLIR_CreateTypeLLVMArray(ctx, i8, 13) != arr &&
                     MLIR_CreateTypeFloat(ctx, 16, false) != bf16
                         ? str_lit("structural types interned: yes\n")
                         : str_lit("structural types interned: no\n"));
        MLIR_TypeHandle st = MLIR_CreateTypeLLVMStructIdentified(ctx, str_lit("S"));
        bool was_opaque = MLIR_IsTypeLLVMStructOpaque(ctx, st);
        MLIR_TypeHandle fields[2] = { ptr, arr };
        MLIR_SetTypeLLVMStructBody(ctx, st, fields, 2);
        MLIR_TypeHandle st2 = MLIR_CreateTypeLLVMStructIdentified(ctx, str_lit("T"));
        MLIR_SetTypeLLVMStructBody(ctx, st2, fields, 2);
        print_string(format(arena, str_lit("struct {}: kind={} opaque={}/{} fields={}"),
            MLIR_GetTypeLLVMStructName(ctx, st), (int64_t)MLIR_GetTypeKind(ctx, st),
            was_opaque ? str_lit("yes") : str_lit("no"),
            MLIR_IsTypeLLVMStructOpaque(ctx, st) ? str_lit("yes") : str_lit("no"),
            (int64_t)MLIR_GetTypeLLVMStructNumFields(ctx, st)));
        print_string(format(arena, str_lit(" field1={} same name={} other name={}\n"),
            MLIR_GetTypeLLVMStructField(ctx, st, 1) == arr ? str_lit("arr") : str_lit("?"),
            MLIR_CreateTypeLLVMStructIdentified(ctx, str_lit("S")) == st ? str_lit("yes") : str_lit("no"),
            st2 == st ? str_lit("same") : str_lit("different")));
        // Equal types are equal handles on both backends: any negative
        // tensor size is the dynamic size, a !tt.ptr without an address
        // space ignores the one passed, opaque types are keyed by name,
        // unsupported float widths are f32, and a struct without a body
        // prints as opaque.
        int64_t dyn1[1] = { -1 }, dyn2[1] = { -2 };
        MLIR_TypeHandle f80 = MLIR_CreateTypeFloat(ctx, 80, false);
        MLIR_TypeHandle st_o = MLIR_CreateTypeLLVMStructIdentified(ctx, str_lit("Opq"));
        print_string(format(arena, str_lit("equal types: dyn={} ttptr={} opaque same={} other={} f80={}/{}"),
            MLIR_CreateTypeTensor(ctx, dyn1, 1, f32) == MLIR_CreateTypeTensor(ctx, dyn2, 1, f32)
                ? str_lit("same") : str_lit("different"),
            MLIR_CreateTypePointer(ctx, f32, false, 1) == MLIR_CreateTypePointer(ctx, f32, false, 7)
                ? str_lit("same") : str_lit("different"),
            MLIR_CreateTypeOpaque(ctx, str_lit("a")) == MLIR_CreateTypeOpaque(ctx, str_lit("a"))
                ? str_lit("yes") : str_lit("no"),
            MLIR_CreateTypeOpaque(ctx, str_lit("a")) == MLIR_CreateTypeOpaque(ctx, str_lit("b"))
                ? str_lit("same") : str_lit("different"),
            f80 == f32 ? str_lit("f32") : str_lit("other"), (int64_t)MLIR_GetTypeWidth(ctx, f80)));
        print_string(format(arena, str_lit(" opaque struct={}\n"), MLIR_GetTypeString(ctx, st_o)));
        MLIR_TypeHandle in[2] = { i8, ptr };
        MLIR_TypeHandle vfn = MLIR_CreateTypeLLVMFunction(ctx, MLIR_CreateTypeLLVMVoid(ctx), in, 2, true);
        MLIR_TypeHandle in2[2] = { i8, ptr };
        print_string(format(arena, str_lit("llvm function: kind={} inputs={} input1={} results={} vararg={}"),
            (int64_t)MLIR_GetTypeKind(ctx, vfn),
            (int64_t)MLIR_GetTypeFunctionNumInputs(ctx, vfn),
            MLIR_GetTypeFunctionInput(ctx, vfn, 1) == ptr ? str_lit("ptr") : str_lit("?"),
            (int64_t)MLIR_GetTypeFunctionNumResults(ctx, vfn),
            MLIR_GetTypeFunctionIsVarArg(ctx, vfn) ? str_lit("yes") : str_lit("no")));
        print_string(format(arena, str_lit(" interned={}\n"),
            MLIR_CreateTypeLLVMFunction(ctx, MLIR_CreateTypeLLVMVoid(ctx), in2, 2, true) == vfn &&
            MLIR_CreateTypeLLVMFunction(ctx, MLIR_CreateTypeLLVMVoid(ctx), in2, 2, false) != vfn
                ? str_lit("yes") : str_lit("no")));
    }

    // Verify that the flat-operand contract of MLIR_GetValueUseOwner /
    // MLIR_SetOpOperand also covers successor-operand uses. The cf->scf
    // lifter's `replace_use` helper round-trips a use through
    // SetOpOperand using the synthetic index returned by GetValueUseOwner,
    // which on the upstream backend addresses both regular and
    // successor-operand slots of a BranchOpInterface op uniformly. The
    // native backend used to silently no-op when idx >= n_operands,
    // which left cf.switch / cf.cond_br successor operands pointing to
    // stale values across the cf->scf RAUW chain and produced
    // semantically-equivalent but byte-different wasm. See
    // mlir_api_impl.c's MLIR_SetOpOperand for the fix.
    {
        MLIR_RegionHandle r = MLIR_CreateRegion(ctx);
        MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
        MLIR_BlockHandle tgt = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, r, entry);
        MLIR_AppendRegionBlock(ctx, r, tgt);

        MLIR_ValueHandle defv = MLIR_AddBlockArgument(ctx, entry, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_ValueHandle replv = MLIR_AddBlockArgument(ctx, entry, i32_ty, MLIR_INVALID_HANDLE);

        // Build cf.br tgt(defv : i32) — zero regular operands, one successor
        // with one successor-operand.
        MLIR_Builder eb = MLIR_BuilderAtEnd(ctx, entry);
        MLIR_OpHandle br = MLIR_BuildBr(&eb, OP_TYPE_CF_BR, tgt, &defv, 1);

        size_t n_uses = MLIR_GetValueNumUses(ctx, defv);
        print_string(n_uses == 1 ? str_lit("cf.br succ_op use registered: yes\n")
                                 : str_lit("cf.br succ_op use registered: no\n"));

        size_t op_idx = (size_t)-1;
        MLIR_OpHandle owner = MLIR_GetValueUseOwner(ctx, defv, 0, &op_idx);
        print_string(owner == br ? str_lit("use owner is cf.br: yes\n")
                                 : str_lit("use owner is cf.br: no\n"));

        // Round-trip the synthetic flat index through SetOpOperand. After
        // the call, cf.br's successor operand at slot 0 must point to
        // `replv` instead of `defv`.
        MLIR_SetOpOperand(ctx, br, op_idx, replv);
        MLIR_ValueHandle after = MLIR_GetOpSuccessorOperand(ctx, br, 0, 0);
        print_string(after == replv ? str_lit("SetOpOperand updates succ_op: yes\n")
                                    : str_lit("SetOpOperand updates succ_op: no\n"));
        print_string(MLIR_GetValueNumUses(ctx, defv) == 0
                         ? str_lit("old value uses cleared: yes\n")
                         : str_lit("old value uses cleared: no\n"));
        print_string(MLIR_GetValueNumUses(ctx, replv) == 1
                         ? str_lit("new value uses registered: yes\n")
                         : str_lit("new value uses registered: no\n"));
    }

    // ---- module region/block ------------------------------------------------
    MLIR_RegionHandle module_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  module_block  = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, module_region, module_block);

    // ---- func.func body -----------------------------------------------------
    MLIR_RegionHandle func_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  func_block  = MLIR_CreateBlock(ctx);

    MLIR_ValueHandle arg0 = MLIR_AddBlockArgument(ctx, func_block, i32_ty, MLIR_INVALID_HANDLE);
    MLIR_ValueHandle arg1 = MLIR_AddBlockArgument(ctx, func_block, i32_ty, MLIR_INVALID_HANDLE);

    // %0 = arith.addi(%arg0, %arg1) -> i32
    // func.return(%0)
    MLIR_Builder fb = MLIR_BuilderAtEnd(ctx, func_block);
    MLIR_ValueHandle add_result = MLIR_BuildBinary(&fb, OP_TYPE_ARITH_ADDI, i32_ty, arg0, arg1);
    MLIR_OpHandle add_op = MLIR_GetValueDefiningOp(ctx, add_result);

    MLIR_OpSpec ret_spec = {0};
    ret_spec.type = OP_TYPE_FUNC_RETURN;
    ret_spec.operands = &add_result;
    ret_spec.n_operands = 1;
    MLIR_OpHandle ret_op = MLIR_BuildOp(&fb, &ret_spec);
    (void)ret_op;

    MLIR_AppendRegionBlock(ctx, func_region, func_block);

    // ---- func.func op --------------------------------------------------------
    MLIR_NamedAttr sym_name_attr =
        MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("add")));
    MLIR_OpSpec func_spec = {0};
    func_spec.type = OP_TYPE_FUNC_FUNC;
    func_spec.attrs = &sym_name_attr;
    func_spec.n_attrs = 1;
    func_spec.regions = &func_region;
    func_spec.n_regions = 1;
    MLIR_OpHandle func_op = MLIR_CreateOpFromSpec(ctx, &func_spec);
    MLIR_AppendBlockOp(ctx, module_block, func_op);

    // ---- module op ----------------------------------------------------------
    MLIR_OpSpec mod_spec = {0};
    mod_spec.type = OP_TYPE_MODULE;
    mod_spec.regions = &module_region;
    mod_spec.n_regions = 1;
    MLIR_OpHandle module_op = MLIR_CreateOpFromSpec(ctx, &mod_spec);

    print_string(print_operation_generic(ctx, 0, module_op));

    // ---- context ownership, erase and diagnostics (#346) ---------------------
    {
        // A block's parent op comes from its region (there is no global op
        // registry any more, #140).
        print_string(MLIR_GetBlockParentOp(ctx, func_block) == func_op
                         ? str_lit("block parent op is func.func: yes\n")
                         : str_lit("block parent op is func.func: no\n"));
        print_string(MLIR_GetBlockParentOp(ctx, module_block) == module_op
                         ? str_lit("block parent op is module: yes\n")
                         : str_lit("block parent op is module: no\n"));

        // MLIR_DetachOp keeps the op, which can then be inserted again.
        MLIR_DetachOp(ctx, ret_op);
        print_string(MLIR_GetOpParentBlock(ctx, ret_op) == MLIR_INVALID_HANDLE &&
                     MLIR_GetBlockNumOps(ctx, func_block) == 1 &&
                     MLIR_GetValueNumUses(ctx, add_result) == 1
                         ? str_lit("detached op keeps its operands: yes\n")
                         : str_lit("detached op keeps its operands: no\n"));
        MLIR_AppendBlockOp(ctx, func_block, ret_op);
        print_string(MLIR_GetBlockNumOps(ctx, func_block) == 2 &&
                     MLIR_GetBlockOp(ctx, func_block, 1) == ret_op
                         ? str_lit("detached op reinserted: yes\n")
                         : str_lit("detached op reinserted: no\n"));

        // MLIR_EraseOp drops the erased op's uses; its handle is dead.
        MLIR_EraseOp(ctx, ret_op);
        print_string(MLIR_GetBlockNumOps(ctx, func_block) == 1 &&
                     MLIR_GetValueNumUses(ctx, add_result) == 0
                         ? str_lit("erase drops the op and its uses: yes\n")
                         : str_lit("erase drops the op and its uses: no\n"));

        // Types live in the context: a type created while a temporary IR
        // arena is current (from a stack array) outlives that arena.
        Arena *tmp = arena_create(64 * 1024);
        MLIR_SetArenaAllocator(ctx, tmp);
        MLIR_TypeHandle sig[2] = { i32_ty, i32_ty };
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, sig, 2, sig, 1);
        MLIR_SetArenaAllocator(ctx, arena);
        arena_destroy(tmp);
        clobber_stack();
        MLIR_TypeHandle sig2[2] = { i32_ty, i32_ty };
        print_string(MLIR_GetTypeFunctionNumInputs(ctx, fty) == 2 &&
                     MLIR_GetTypeFunctionInput(ctx, fty, 1) == i32_ty &&
                     MLIR_GetTypeFunctionResult(ctx, fty, 0) == i32_ty &&
                     MLIR_CreateTypeFunction(ctx, sig2, 2, sig2, 1) == fty
                         ? str_lit("function type outlives its IR arena: yes\n")
                         : str_lit("function type outlives its IR arena: no\n"));

        // Diagnostics reach the context's handler.
        DiagCapture cap = {0};
        MLIR_SetDiagnosticHandler(ctx, capture_diag, &cap);
        MLIR_LocationHandle loc = MLIR_CreateLocationFile(ctx, str_lit("t.c"), 3, 7);
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_WARNING, loc, "value %d is %s", 42, "odd");
        MLIR_SetDiagnosticHandler(ctx, NULL, NULL);
        string text = { cap.text, cap.len };
        print_string(format(arena, str_lit("diagnostic: count={} warning={} loc={}:{}:{} text={}\n"),
                            (int64_t)cap.count,
                            (int64_t)(cap.severity == MLIR_DIAG_WARNING),
                            MLIR_GetLocationFileFilename(ctx, cap.loc),
                            (int64_t)MLIR_GetLocationFileLine(ctx, cap.loc),
                            (int64_t)MLIR_GetLocationFileColumn(ctx, cap.loc),
                            text));

        // Contexts are independent: each uniques its own types.
        MLIR_Context *ctx2 = MLIR_CreateContext(NULL);
        Arena *arena2 = arena_create(64 * 1024);
        MLIR_SetArenaAllocator(ctx2, arena2);
        MLIR_TypeHandle i32_2 = MLIR_CreateTypeInteger(ctx2, 32, true);
        print_string(i32_2 != i32_ty &&
                     MLIR_CreateTypeInteger(ctx2, 32, true) == i32_2
                         ? str_lit("second context has its own types: yes\n")
                         : str_lit("second context has its own types: no\n"));
        MLIR_DestroyContext(ctx2);
        arena_destroy(arena2);
        print_string(MLIR_CreateTypeInteger(ctx, 32, true) == i32_ty
                         ? str_lit("first context unaffected: yes\n")
                         : str_lit("first context unaffected: no\n"));

        // Op types: static names, dynamic types, dialect ranges.
        MLIR_OpType add_t = MLIR_LookupOpType(ctx, str_lit("llvm.add"));
        MLIR_OpType miss = MLIR_LookupOpType(ctx, str_lit("test.op"));
        MLIR_OpType dyn = MLIR_RegisterOpType(ctx, str_lit("test.op"));
        print_string(format(arena,
            str_lit("op types: static={} miss={} dynamic={} again={} lookup={} name={} module={}\n"),
            add_t == OP_TYPE_LLVM_ADD ? str_lit("yes") : str_lit("no"),
            miss == OP_TYPE_UNREGISTERED ? str_lit("yes") : str_lit("no"),
            (unsigned)dyn >= (unsigned)OP_TYPE_COUNT ? str_lit("yes") : str_lit("no"),
            MLIR_RegisterOpType(ctx, str_lit("test.op")) == dyn ? str_lit("yes") : str_lit("no"),
            MLIR_LookupOpType(ctx, str_lit("test.op")) == dyn ? str_lit("yes") : str_lit("no"),
            MLIR_GetOpTypeName(ctx, dyn),
            MLIR_GetOpTypeName(ctx, OP_TYPE_MODULE)));
        MLIR_OpSpec dyn_spec = {0};
        dyn_spec.type = dyn;
        MLIR_OpHandle dyn_op = MLIR_CreateOpFromSpec(ctx, &dyn_spec);
        print_string(MLIR_GetOpType(ctx, dyn_op) == dyn &&
                     MLIR_OP_TYPE_IN_DIALECT(OP_TYPE_LLVM_ADD, LLVM) &&
                     MLIR_OP_TYPE_IN_DIALECT(MLIR_DIALECT_LLVM_FIRST, LLVM) &&
                     MLIR_OP_TYPE_IN_DIALECT(MLIR_DIALECT_LLVM_LAST, LLVM) &&
                     !MLIR_OP_TYPE_IN_DIALECT(OP_TYPE_ARITH_ADDI, LLVM) &&
                     !MLIR_OP_TYPE_IN_DIALECT(dyn, LLVM) &&
                     !MLIR_OP_TYPE_IN_DIALECT(OP_TYPE_UNREGISTERED, LLVM)
                         ? str_lit("dynamic op and dialect ranges: yes\n")
                         : str_lit("dynamic op and dialect ranges: no\n"));
        // `builtin.module` (the upstream name) is an alias of OP_TYPE_MODULE:
        // registering it must not give modules another type.
        MLIR_OpType bm_lookup = MLIR_LookupOpType(ctx, str_lit("builtin.module"));
        MLIR_OpType bm_reg = MLIR_RegisterOpType(ctx, str_lit("builtin.module"));
        MLIR_OpSpec mod2_spec = {0};
        mod2_spec.type = OP_TYPE_MODULE;
        MLIR_OpHandle mod2 = MLIR_CreateOpFromSpec(ctx, &mod2_spec);
        print_string(format(arena,
            str_lit("builtin.module alias: lookup={} register={} module op={} name={}\n"),
            bm_lookup == OP_TYPE_MODULE ? str_lit("yes") : str_lit("no"),
            bm_reg == OP_TYPE_MODULE ? str_lit("yes") : str_lit("no"),
            MLIR_GetOpType(ctx, mod2) == OP_TYPE_MODULE && MLIR_GetOpType(ctx, module_op) == OP_TYPE_MODULE
                ? str_lit("yes") : str_lit("no"),
            MLIR_GetOpTypeName(ctx, bm_reg)));

        // Attribute keys (#350): static and dynamic keys, and the op's
        // attributes by key.
        MLIR_AttrKey mykey = MLIR_InternAttrKey(ctx, str_lit("test.my_attr"));
        print_string(format(arena, str_lit("attr keys: static={} dynamic={} again={} name={} miss={}\n"),
            MLIR_LookupAttrKey(ctx, str_lit("sym_name")) == MLIR_KEY_SYM_NAME &&
            str_eq(MLIR_GetAttrKeyName(ctx, MLIR_KEY_VALUE), str_lit("value"))
                ? str_lit("yes") : str_lit("no"),
            mykey >= MLIR_KEY_COUNT ? str_lit("yes") : str_lit("no"),
            MLIR_InternAttrKey(ctx, str_lit("test.my_attr")) == mykey ? str_lit("yes") : str_lit("no"),
            MLIR_GetAttrKeyName(ctx, mykey),
            MLIR_LookupAttrKey(ctx, str_lit("test.no_such_attr")) == MLIR_KEY_INVALID
                ? str_lit("yes") : str_lit("no")));
        MLIR_NamedAttr kattrs[2] = {
            MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("f"))),
            MLIR_Named(mykey, MLIR_CreateAttributeInteger(ctx, 7, i32_ty)),
        };
        MLIR_OpSpec kop_spec = {0};
        kop_spec.type = dyn;
        kop_spec.attrs = kattrs;
        kop_spec.n_attrs = 2;
        MLIR_OpHandle kop = MLIR_CreateOpFromSpec(ctx, &kop_spec);
        int64_t before = MLIR_GetAttributeInteger(ctx, MLIR_GetOpAttr(ctx, kop, mykey));
        MLIR_SetOpAttr(ctx, kop, mykey, MLIR_CreateAttributeInteger(ctx, 8, i32_ty));
        size_t n_replaced = MLIR_GetOpNumAttributes(ctx, kop);
        MLIR_SetOpAttr(ctx, kop, MLIR_KEY_VALUE, MLIR_CreateAttributeBool(ctx, true));
        print_string(format(arena, str_lit("op attrs by key: sym_name={} mine={}->{} after set={} after add={} last={}\n"),
            MLIR_GetAttributeString(ctx, MLIR_GetOpAttr(ctx, kop, MLIR_KEY_SYM_NAME)),
            before, MLIR_GetAttributeInteger(ctx, MLIR_GetOpAttr(ctx, kop, mykey)),
            (int64_t)n_replaced, (int64_t)MLIR_GetOpNumAttributes(ctx, kop),
            MLIR_GetAttrKeyName(ctx, MLIR_GetOpAttributeKey(ctx, kop, 2))));

        // Attributes are uniqued (#350): equal values are equal handles;
        // floats by bit pattern, integers with their type; distinct
        // attributes never are.
        MLIR_TypeHandle i64t = MLIR_CreateTypeInteger(ctx, 64, true);
        MLIR_TypeHandle f64t = MLIR_CreateTypeFloat(ctx, 64, false);
        // Quiet NaN by bits. MSVC rejects the constant expression 0.0 / 0.0
        // (error C2124) while compiling this driver.
        union { uint64_t u; double d; } nan_bits = { 0x7ff8000000000000ULL };
        double nan = nan_bits.d;
        MLIR_AttributeHandle e1[2] = { MLIR_CreateAttributeInteger(ctx, 1, i32_ty),
                                       MLIR_CreateAttributeString(ctx, str_lit("s")) };
        MLIR_AttributeHandle e2[2] = { MLIR_CreateAttributeInteger(ctx, 1, i32_ty),
                                       MLIR_CreateAttributeString(ctx, str_lit("s")) };
        MLIR_AttributeHandle dref = MLIR_CreateAttributeBool(ctx, true);
        print_string(format(arena, str_lit("attr uniquing: int={} int type={} float zero sign={} nan={}"),
            MLIR_CreateAttributeInteger(ctx, 5, i32_ty) == MLIR_CreateAttributeInteger(ctx, 5, i32_ty)
                ? str_lit("same") : str_lit("different"),
            MLIR_CreateAttributeInteger(ctx, 5, i32_ty) == MLIR_CreateAttributeInteger(ctx, 5, i64t)
                ? str_lit("same") : str_lit("different"),
            MLIR_CreateAttributeFloat(ctx, 0.0, f64t) == MLIR_CreateAttributeFloat(ctx, -0.0, f64t)
                ? str_lit("same") : str_lit("different"),
            MLIR_CreateAttributeFloat(ctx, nan, f64t) == MLIR_CreateAttributeFloat(ctx, nan, f64t)
                ? str_lit("same") : str_lit("different")));
        print_string(format(arena, str_lit(" array={} distinct={}\n"),
            MLIR_CreateAttributeArray(ctx, e1, 2) == MLIR_CreateAttributeArray(ctx, e2, 2)
                ? str_lit("same") : str_lit("different"),
            MLIR_CreateAttributeDistinct(ctx, dref) == MLIR_CreateAttributeDistinct(ctx, dref)
                ? str_lit("same") : str_lit("different")));
        int32_t dv[3] = { 1, -2, 3 };
        int64_t dv64[2] = { 5000000000, -7 };
        MLIR_AttributeHandle d32 = MLIR_CreateAttributeDenseI32Array(ctx, dv, 3);
        MLIR_AttributeHandle d64 = MLIR_CreateAttributeDenseI64Array(ctx, dv64, 2);
        print_string(format(arena, str_lit("dense arrays: i32 n={} [1]={} text={}; i64 n={} [0]={} [1]={}\n"),
            (int64_t)MLIR_GetAttributeDenseSize(ctx, d32), MLIR_GetAttributeDenseElement(ctx, d32, 1),
            MLIR_GetAttributeAsString(ctx, d32),
            (int64_t)MLIR_GetAttributeDenseSize(ctx, d64), MLIR_GetAttributeDenseElement(ctx, d64, 0),
            MLIR_GetAttributeDenseElement(ctx, d64, 1)));

        // Locations (#350).
        MLIR_LocationHandle fl = MLIR_CreateLocationFile(ctx, str_lit("a.c"), 12, 3);
        MLIR_LocationHandle nl = MLIR_CreateLocationName(ctx, str_lit("here"));
        print_string(format(arena, str_lit("locations: file={}:{}:{} kind={} name={} kind={} unknown shared={}\n"),
            MLIR_GetLocationFileFilename(ctx, fl), (int64_t)MLIR_GetLocationFileLine(ctx, fl),
            (int64_t)MLIR_GetLocationFileColumn(ctx, fl), (int64_t)MLIR_GetLocationKind(ctx, fl),
            MLIR_GetLocationName(ctx, nl), (int64_t)MLIR_GetLocationKind(ctx, nl),
            MLIR_CreateLocationUnknown(ctx, (string){0}) == MLIR_CreateLocationUnknown(ctx, (string){0})
                ? str_lit("yes") : str_lit("no")));
        MLIR_Context *ctx3 = MLIR_CreateContext(NULL);
        print_string(MLIR_LookupOpType(ctx3, str_lit("test.op")) == OP_TYPE_UNREGISTERED &&
                     MLIR_LookupOpType(ctx3, str_lit("llvm.add")) == OP_TYPE_LLVM_ADD
                         ? str_lit("dynamic types are per context: yes\n")
                         : str_lit("dynamic types are per context: no\n"));
        MLIR_DestroyContext(ctx3);
    }

    // ---- index storage (#354): growth and operand relocation ----------------
    {
        // Enough values and ops that every native IR vector grows several
        // times: a chain of adds that all use the block argument `arg` too.
        MLIR_BlockHandle src = MLIR_CreateBlock(ctx);
        MLIR_BlockHandle dst = MLIR_CreateBlock(ctx);
        MLIR_ValueHandle arg = MLIR_AddBlockArgument(ctx, src, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_ValueHandle x = MLIR_AddBlockArgument(ctx, src, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_AddBlockArgument(ctx, dst, i32_ty, MLIR_INVALID_HANDLE);
        enum { N = 3000 };
        MLIR_ValueHandle last = arg;
        bool chain_ok = true;
        MLIR_Builder sb = MLIR_BuilderAtEnd(ctx, src);
        for (size_t i = 0; i < N; i++) {
            MLIR_ValueHandle res = MLIR_BuildBinary(&sb, OP_TYPE_LLVM_ADD, i32_ty, last, arg);
            MLIR_OpHandle add = MLIR_GetValueDefiningOp(ctx, res);
            if (MLIR_GetOpResult(ctx, add, 0) != res || MLIR_GetValueDefiningOp(ctx, res) != add)
                chain_ok = false;
            last = res;
        }
        MLIR_OpHandle br = MLIR_BuildBr(&sb, OP_TYPE_LLVM_BR, dst, &last, 1);
        print_string(chain_ok && MLIR_GetBlockNumOps(ctx, src) == N + 1 &&
                     MLIR_GetValueNumUses(ctx, arg) == N + 1 &&
                     MLIR_GetValueNumUses(ctx, last) == 1
                         ? str_lit("3000 ops, results and uses: yes\n")
                         : str_lit("3000 ops, results and uses: no\n"));

        // Appending successor operands moves the op's operands natively; the
        // use lists must follow, and the flat operand index stays upstream's.
        MLIR_AppendOpSuccessorOperand(ctx, br, 0, arg);
        MLIR_AppendOpSuccessorOperand(ctx, br, 0, last);
        size_t flat = SIZE_MAX;
        MLIR_OpHandle owner = MLIR_GetValueUseOwner(ctx, arg, 0, &flat);
        print_string(MLIR_GetOpNumSuccessorOperands(ctx, br, 0) == 3 &&
                     MLIR_GetOpSuccessorOperand(ctx, br, 0, 1) == arg &&
                     MLIR_GetValueNumUses(ctx, arg) == N + 2 &&
                     MLIR_GetValueNumUses(ctx, last) == 2 &&
                     owner == br && flat == 1
                         ? str_lit("successor operands appended: yes\n")
                         : str_lit("successor operands appended: no\n"));
        MLIR_ValueHandle repl[2] = { x, x };
        MLIR_SetOpSuccessorOperands(ctx, br, 0, repl, 2);
        MLIR_ReplaceAllUsesOfValue(ctx, arg, x);
        print_string(MLIR_GetOpNumSuccessorOperands(ctx, br, 0) == 2 &&
                     MLIR_GetValueNumUses(ctx, last) == 0 &&
                     MLIR_GetValueNumUses(ctx, arg) == 0 &&
                     MLIR_GetValueNumUses(ctx, x) == N + 3 &&
                     MLIR_GetOpOperand(ctx, MLIR_GetBlockOp(ctx, src, N - 1), 1) == x
                         ? str_lit("successor operands replaced, uses moved: yes\n")
                         : str_lit("successor operands replaced, uses moved: no\n"));
    }

    // ---- op traits and MLIR_VerifyOp (#348) ---------------------------------
    {
        MLIR_OpType dyn = MLIR_RegisterOpType(ctx, str_lit("test.trait_dyn"));
        bool traits_ok =
            (MLIR_GetOpTypeTraits(OP_TYPE_LLVM_ADD) &
             (MLIR_T_PURE | MLIR_T_COMMUTATIVE | MLIR_T_ONE_RESULT)) ==
                (MLIR_T_PURE | MLIR_T_COMMUTATIVE | MLIR_T_ONE_RESULT) &&
            (MLIR_GetOpTypeTraits(OP_TYPE_LLVM_SUB) & MLIR_T_COMMUTATIVE) == 0 &&
            (MLIR_GetOpTypeTraits(OP_TYPE_LLVM_LOAD) & (MLIR_T_MEM_READ | MLIR_T_ONE_RESULT)) ==
                (MLIR_T_MEM_READ | MLIR_T_ONE_RESULT) &&
            (MLIR_GetOpTypeTraits(OP_TYPE_LLVM_LOAD) & MLIR_T_PURE) == 0 &&
            (MLIR_GetOpTypeTraits(OP_TYPE_LLVM_STORE) & (MLIR_T_MEM_WRITE | MLIR_T_ZERO_RESULT)) ==
                (MLIR_T_MEM_WRITE | MLIR_T_ZERO_RESULT) &&
            (MLIR_GetOpTypeTraits(OP_TYPE_LLVM_BR) &
             (MLIR_T_TERMINATOR | MLIR_T_BRANCH | MLIR_T_ZERO_RESULT)) ==
                (MLIR_T_TERMINATOR | MLIR_T_BRANCH | MLIR_T_ZERO_RESULT) &&
            (MLIR_GetOpTypeTraits(OP_TYPE_FUNC_FUNC) &
             (MLIR_T_FUNCTION | MLIR_T_SYMBOL | MLIR_T_ISOLATED)) ==
                (MLIR_T_FUNCTION | MLIR_T_SYMBOL | MLIR_T_ISOLATED) &&
            MLIR_GetOpTypeTraits(OP_TYPE_UNREGISTERED) == MLIR_T_NONE &&
            MLIR_GetOpTypeTraits(dyn) == MLIR_T_NONE;

        // Verify a valid module and a malformed op (arith.addi with 0 results).
        DiagCapture vcap;
        vcap.count = 0;
        vcap.saw_error = false;
        MLIR_SetDiagnosticHandler(ctx, capture_diag, &vcap);

        MLIR_OpSpec vmod_spec = {0};
        vmod_spec.type = OP_TYPE_MODULE;
        vmod_spec.n_regions = 1;
        MLIR_OpHandle vmod = MLIR_CreateOpFromSpec(ctx, &vmod_spec);
        MLIR_BlockHandle mblk = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, MLIR_GetOpRegion(ctx, vmod, 0), mblk);
        bool valid_ok = MLIR_VerifyOp(ctx, vmod) && vcap.count == 0;

        MLIR_OpSpec bad_spec = {0};
        bad_spec.type = OP_TYPE_ARITH_ADDI;
        MLIR_OpHandle bad = MLIR_CreateOpFromSpec(ctx, &bad_spec);
        MLIR_AppendBlockOp(ctx, mblk, bad);
        bool invalid_caught = !MLIR_VerifyOp(ctx, vmod) && vcap.saw_error;
        MLIR_SetDiagnosticHandler(ctx, NULL, NULL);

        print_string(traits_ok && valid_ok && invalid_caught
                         ? str_lit("op traits and verify: yes\n")
                         : str_lit("op traits and verify: no\n"));
    }

    // ---- op builder API (#351) ----------------------------------------------
    {
        MLIR_OpSpec fresh_mod_spec = {0};
        fresh_mod_spec.type = OP_TYPE_MODULE;
        fresh_mod_spec.n_regions = 1; // fresh empty region
        MLIR_OpHandle fresh_mod = MLIR_CreateOpFromSpec(ctx, &fresh_mod_spec);
        MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, fresh_mod, 0);
        MLIR_BlockHandle b0 = MLIR_CreateBlock(ctx);
        MLIR_BlockHandle b1 = MLIR_CreateBlock(ctx);
        MLIR_BlockHandle b2 = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, r, b0);
        MLIR_AppendRegionBlock(ctx, r, b1);
        MLIR_AppendRegionBlock(ctx, r, b2);
        MLIR_AddBlockArgument(ctx, b1, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_AddBlockArgument(ctx, b2, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_AddBlockArgument(ctx, b2, i32_ty, MLIR_INVALID_HANDLE);

        MLIR_TypeHandle i1_ty = MLIR_CreateTypeInteger(ctx, 1, false);
        MLIR_TypeHandle i64_ty = MLIR_CreateTypeInteger(ctx, 64, true);
        MLIR_TypeHandle f64_ty = MLIR_CreateTypeFloat(ctx, 64, false);
        MLIR_LocationHandle bloc = MLIR_CreateLocationFile(ctx, str_lit("b.c"), 7, 1);

        MLIR_Builder b = MLIR_BuilderAtEnd(ctx, b0);
        b.loc = bloc;
        MLIR_ValueHandle c1 = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i32_ty, 10);
        MLIR_ValueHandle c2 = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i32_ty, 20);
        MLIR_ValueHandle cf = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, f64_ty, 1.5);
        MLIR_ValueHandle cond = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i1_ty, 1);

        MLIR_OpHandle c2_op = MLIR_GetValueDefiningOp(ctx, c2);
        MLIR_Builder before_c2 = MLIR_BuilderBefore(ctx, c2_op);
        MLIR_ValueHandle ext = MLIR_BuildUnary(&before_c2, OP_TYPE_LLVM_SEXT, i64_ty, c1);
        MLIR_Builder after_c2 = MLIR_BuilderAfter(ctx, c2_op);
        MLIR_ValueHandle sum = MLIR_BuildBinary(&after_c2, OP_TYPE_LLVM_ADD, i32_ty, c1, c2);

        MLIR_ValueHandle t_args[1] = { sum };
        MLIR_ValueHandle f_args[2] = { c1, c2 };
        MLIR_OpHandle cbr = MLIR_BuildCondBr(&b, OP_TYPE_LLVM_COND_BR, cond,
                                             b1, t_args, 1, b2, f_args, 2);

        size_t f0 = 99, n0 = 99, f1 = 99, n1 = 99;
        MLIR_GetOpSuccessorOperandRange(ctx, cbr, 0, &f0, &n0);
        MLIR_GetOpSuccessorOperandRange(ctx, cbr, 1, &f1, &n1);

        bool ok =
            MLIR_GetOpNumRegions(ctx, fresh_mod) == 1 &&
            MLIR_GetBlockNumOps(ctx, b0) == 7 &&
            MLIR_GetBlockOp(ctx, b0, 1) == MLIR_GetValueDefiningOp(ctx, ext) &&
            MLIR_GetBlockOp(ctx, b0, 2) == c2_op &&
            MLIR_GetBlockOp(ctx, b0, 3) == MLIR_GetValueDefiningOp(ctx, sum) &&
            MLIR_GetOpLocation(ctx, c2_op) == bloc &&
            MLIR_GetAttributeFloat(ctx, MLIR_GetOpAttr(ctx, MLIR_GetValueDefiningOp(ctx, cf), MLIR_KEY_VALUE)) == 1.5 &&
            f0 == 1 && n0 == 1 && f1 == 2 && n1 == 2 &&
            MLIR_GetOpSuccessorOperand(ctx, cbr, 0, 0) == sum &&
            MLIR_GetOpSuccessorOperand(ctx, cbr, 1, 0) == c1 &&
            MLIR_GetOpSuccessorOperand(ctx, cbr, 1, 1) == c2;

        // Move c2_op to the front of b0 using MLIR_InsertOpBefore.
        MLIR_InsertOpBefore(ctx, b0, c2_op, MLIR_GetBlockOp(ctx, b0, 0));
        ok = ok && MLIR_GetBlockOp(ctx, b0, 0) == c2_op;

        print_string(ok ? str_lit("op builder API: yes\n")
                        : str_lit("op builder API: no\n"));
    }

    // ---- traversal, iterators, walks and CFG info (#352) --------------------
    {
        // Build a CFG with a loop: entry -> header -> body -> header, header -> exit.
        MLIR_OpSpec mspec = {0};
        mspec.type = OP_TYPE_MODULE;
        mspec.n_regions = 1;
        MLIR_OpHandle tmod = MLIR_CreateOpFromSpec(ctx, &mspec);
        MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, tmod, 0);

        MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
        MLIR_BlockHandle hdr   = MLIR_CreateBlock(ctx);
        MLIR_BlockHandle body  = MLIR_CreateBlock(ctx);
        MLIR_BlockHandle exitb = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, reg, entry);
        MLIR_AppendRegionBlock(ctx, reg, hdr);
        MLIR_AppendRegionBlock(ctx, reg, body);
        MLIR_AppendRegionBlock(ctx, reg, exitb);

        MLIR_TypeHandle i1_ty = MLIR_CreateTypeInteger(ctx, 1, false);
        MLIR_ValueHandle harg = MLIR_AddBlockArgument(ctx, hdr, i32_ty, MLIR_INVALID_HANDLE);

        MLIR_Builder be = MLIR_BuilderAtEnd(ctx, entry);
        MLIR_ValueHandle c0 = MLIR_BuildConstantInt(&be, OP_TYPE_LLVM_MLIR_CONSTANT, i32_ty, 0);
        MLIR_ValueHandle c1 = MLIR_BuildConstantInt(&be, OP_TYPE_LLVM_MLIR_CONSTANT, i32_ty, 1);
        MLIR_ValueHandle dead_v = MLIR_BuildConstantInt(&be, OP_TYPE_LLVM_MLIR_CONSTANT, i32_ty, 99);
        MLIR_BuildBr(&be, OP_TYPE_LLVM_BR, hdr, &c0, 1);

        MLIR_Builder bh = MLIR_BuilderAtEnd(ctx, hdr);
        MLIR_ValueHandle cond = MLIR_BuildConstantInt(&bh, OP_TYPE_LLVM_MLIR_CONSTANT, i1_ty, 1);
        MLIR_BuildCondBr(&bh, OP_TYPE_LLVM_COND_BR, cond, body, NULL, 0, exitb, NULL, 0);

        MLIR_Builder bb = MLIR_BuilderAtEnd(ctx, body);
        MLIR_ValueHandle inc = MLIR_BuildBinary(&bb, OP_TYPE_LLVM_ADD, i32_ty, harg, c1);
        MLIR_BuildBr(&bb, OP_TYPE_LLVM_BR, hdr, &inc, 1);

        MLIR_Builder bx = MLIR_BuilderAtEnd(ctx, exitb);
        MLIR_OpSpec ret_spec = {0};
        ret_spec.type = OP_TYPE_LLVM_RETURN;
        ret_spec.operands = &harg;
        ret_spec.n_operands = 1;
        MLIR_BuildOp(&bx, &ret_spec);

        // Region block iterators (forward & backward).
        int blk_fwd = 0, blk_bwd = 0;
        for (MLIR_BlockHandle b = MLIR_GetRegionFirstBlock(ctx, reg);
             b != MLIR_INVALID_HANDLE; b = MLIR_GetBlockNext(ctx, b)) blk_fwd++;
        for (MLIR_BlockHandle b = MLIR_GetRegionLastBlock(ctx, reg);
             b != MLIR_INVALID_HANDLE; b = MLIR_GetBlockPrev(ctx, b)) blk_bwd++;

        // Block op iterators (forward & backward), cursor, & GetParentOp.
        int op_fwd = 0, op_bwd = 0, op_cur = 0;
        for (MLIR_OpHandle o = MLIR_GetBlockFirstOp(ctx, entry);
             o != MLIR_INVALID_HANDLE; o = MLIR_GetOpNext(ctx, o)) op_fwd++;
        for (MLIR_OpHandle o = MLIR_GetBlockLastOp(ctx, entry);
             o != MLIR_INVALID_HANDLE; o = MLIR_GetOpPrev(ctx, o)) op_bwd++;
        bool iter_match = true;
        {
            MLIR_BlockOpIter bit = MLIR_BlockOpsBegin(ctx, entry);
            MLIR_OpHandle expected_op = MLIR_GetBlockFirstOp(ctx, entry);
            for (MLIR_OpHandle o; (o = MLIR_BlockOpIterNext(ctx, &bit)) != MLIR_INVALID_HANDLE; ) {
                if (o != expected_op) iter_match = false;
                expected_op = MLIR_GetOpNext(ctx, expected_op);
                op_cur++;
            }
            if (expected_op != MLIR_INVALID_HANDLE || op_cur != op_fwd) iter_match = false;
        }
        bool parent_ok = iter_match && MLIR_GetParentOp(ctx, MLIR_GetBlockFirstOp(ctx, entry)) == tmod;

        // Value use iterators & HasOneUse / IsUnused.
        int harg_uses = 0;
        for (MLIR_UseHandle u = MLIR_GetValueFirstUse(ctx, harg);
             u != MLIR_INVALID_HANDLE; u = MLIR_GetUseNext(ctx, u)) harg_uses++;
        bool uses_ok = harg_uses == 2 &&
                       MLIR_ValueHasOneUse(ctx, c0) &&
                       MLIR_IsValueUnused(ctx, dead_v) &&
                       !MLIR_ValueHasOneUse(ctx, harg);

        // Pred iterators & GetIncomingValue / SetIncomingValue.
        int hdr_preds = 0;
        bool incoming_ok = true;
        for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(ctx, hdr);
             p != MLIR_INVALID_HANDLE; p = MLIR_GetPredNext(ctx, p)) {
            hdr_preds++;
            MLIR_BlockHandle pb = MLIR_GetPredBlock(ctx, p);
            MLIR_ValueHandle iv = MLIR_GetIncomingValue(ctx, p, 0);
            if (pb == entry) {
                if (iv != c0) incoming_ok = false;
                MLIR_SetIncomingValue(ctx, p, 0, c1);
                if (MLIR_GetIncomingValue(ctx, p, 0) != c1) incoming_ok = false;
                MLIR_SetIncomingValue(ctx, p, 0, c0);
            } else if (pb == body) {
                if (iv != inc) incoming_ok = false;
            } else {
                incoming_ok = false;
            }
        }

        // MLIR_Walk: pre-order, skip, interrupt, post-order with erase.
        WalkState ws = {0};
        MLIR_Walk(ctx, tmod, MLIR_WALK_PRE, test_walk_cb, &ws);
        int total_ops = ws.count; // 1 module + 4 + 2 + 2 + 1 = 10

        ws = (WalkState){0};
        ws.skip_target = tmod;
        MLIR_Walk(ctx, tmod, MLIR_WALK_PRE, test_walk_cb, &ws);
        int skip_count = ws.count; // 1 (skipped children)

        ws = (WalkState){0};
        ws.erase_target = MLIR_GetValueDefiningOp(ctx, dead_v);
        MLIR_Walk(ctx, tmod, MLIR_WALK_POST, test_walk_cb, &ws);
        bool erase_walk_ok = ws.count == total_ops && MLIR_GetBlockNumOps(ctx, entry) == 3;

        // MLIR_GetBlockOrder (RPO/PO) and MLIR_ComputeCFGInfo (doms & loops).
        MLIR_BlockHandle *rpo = NULL;
        size_t n_rpo = MLIR_GetBlockOrder(ctx, reg, MLIR_ORDER_RPO, arena, &rpo);
        MLIR_CFGInfo *cfg = MLIR_ComputeCFGInfo(ctx, reg, arena);
        int32_t i_entry = MLIR_CFGGetRPOIndex(cfg, entry);
        int32_t i_hdr   = MLIR_CFGGetRPOIndex(cfg, hdr);
        int32_t i_body  = MLIR_CFGGetRPOIndex(cfg, body);
        int32_t i_exit  = MLIR_CFGGetRPOIndex(cfg, exitb);

        bool cfg_ok =
            n_rpo == 4 && rpo[0] == entry &&
            cfg->n_blocks == 4 &&
            i_entry == 0 && i_hdr == 1 &&
            cfg->idom[i_entry] == -1 &&
            cfg->idom[i_hdr] == i_entry &&
            cfg->idom[i_body] == i_hdr &&
            cfg->idom[i_exit] == i_hdr &&
            MLIR_CFGDominates(cfg, i_entry, i_exit) &&
            MLIR_CFGDominates(cfg, i_hdr, i_body) &&
            !MLIR_CFGDominates(cfg, i_body, i_exit) &&
            cfg->is_loop_header[i_hdr] &&
            !cfg->is_loop_header[i_entry] &&
            cfg->loop_depth[i_entry] == 0 &&
            cfg->loop_depth[i_hdr] == 1 &&
            cfg->loop_depth[i_body] == 1 &&
            cfg->loop_depth[i_exit] == 0;

        bool trav_ok = blk_fwd == 4 && blk_bwd == 4 &&
                       op_fwd == 4 && op_bwd == 4 &&
                       parent_ok && uses_ok &&
                       hdr_preds == 2 && incoming_ok &&
                       total_ops == 10 && skip_count == 1 && erase_walk_ok &&
                       cfg_ok;
        print_string(trav_ok ? str_lit("traversal and CFG info: yes\n")
                             : str_lit("traversal and CFG info: no\n"));
    }

    // ---- LLVM-level completeness (#356) ------------------------------------
    {
        MLIR_OpSpec mod_spec = {0};
        mod_spec.type = OP_TYPE_MODULE;
        mod_spec.n_regions = 1;
        MLIR_OpHandle mod = MLIR_CreateOpFromSpec(ctx, &mod_spec);
        MLIR_BlockHandle mb = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, MLIR_GetOpRegion(ctx, mod, 0), mb);

        MLIR_TypeHandle f32_ty = MLIR_CreateTypeFloat(ctx, 32, false);
        MLIR_TypeHandle void_ty = MLIR_CreateTypeLLVMVoid(ctx);
        int64_t v4[1] = { 4 };
        MLIR_TypeHandle v4f32 = MLIR_CreateTypeVector(ctx, v4, 1, f32_ty);
        MLIR_TypeHandle v4i32 = MLIR_CreateTypeVector(ctx, v4, 1, i32_ty);
        MLIR_TypeHandle st_ty = MLIR_CreateTypeLLVMStructIdentified(ctx, str_lit("Pair356"));
        MLIR_TypeHandle st_fields[2] = { f32_ty, i32_ty };
        MLIR_SetTypeLLVMStructBody(ctx, st_ty, st_fields, 2);

        // Declaration (no blocks in region 0) and definition (1 block).
        MLIR_TypeHandle decl_fty = MLIR_CreateTypeLLVMFunction(ctx, f32_ty, &f32_ty, 1, false);
        MLIR_NamedAttr decl_attrs[2] = {
            MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("sqrtf"))),
            MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, decl_fty)),
        };
        MLIR_OpSpec decl_spec = {0};
        decl_spec.type = OP_TYPE_LLVM_FUNC;
        decl_spec.attrs = decl_attrs;
        decl_spec.n_attrs = 2;
        decl_spec.n_regions = 1;
        MLIR_OpHandle decl_fn = MLIR_CreateOpFromSpec(ctx, &decl_spec);
        MLIR_AppendBlockOp(ctx, mb, decl_fn);

        MLIR_TypeHandle def_in[2] = { v4f32, st_ty };
        MLIR_TypeHandle def_fty = MLIR_CreateTypeLLVMFunction(ctx, f32_ty, def_in, 2, false);
        MLIR_NamedAttr def_attrs[2] = {
            MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("kernel356"))),
            MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, def_fty)),
        };
        MLIR_OpSpec def_spec = {0};
        def_spec.type = OP_TYPE_LLVM_FUNC;
        def_spec.attrs = def_attrs;
        def_spec.n_attrs = 2;
        def_spec.n_regions = 1;
        MLIR_OpHandle def_fn = MLIR_CreateOpFromSpec(ctx, &def_spec);
        MLIR_AppendBlockOp(ctx, mb, def_fn);
        MLIR_BlockHandle fb = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, MLIR_GetOpRegion(ctx, def_fn, 0), fb);
        MLIR_ValueHandle arg_v = MLIR_AddBlockArgument(ctx, fb, v4f32, MLIR_INVALID_HANDLE);
        MLIR_ValueHandle arg_s = MLIR_AddBlockArgument(ctx, fb, st_ty, MLIR_INVALID_HANDLE);

        MLIR_Builder b = MLIR_BuilderAtEnd(ctx, fb);

        // Scalar & splat constants + matchers.
        MLIR_ValueHandle cf = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, f32_ty, 2.5);
        MLIR_ValueHandle vf = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, v4f32, -3.0);
        MLIR_ValueHandle ci = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i32_ty, 7);
        MLIR_ValueHandle vi = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, v4i32, 0);
        MLIR_ValueHandle zf = MLIR_BuildZero(&b, f32_ty);
        MLIR_ValueHandle uf = MLIR_BuildUndef(&b, v4f32);
        MLIR_ValueHandle pf = MLIR_BuildPoison(&b, f32_ty);

        double df = 0.0, dv = 0.0, dz = 1.0;
        int64_t ii = 0, iv = 1;
        bool const_ok =
            MLIR_MatchConstantFloat(ctx, cf, -1, &df) && df == 2.5 &&
            MLIR_MatchConstantFloat(ctx, cf, 0, &df) && df == 2.5 &&
            !MLIR_MatchConstantFloat(ctx, cf, 1, &df) &&
            MLIR_MatchConstantFloat(ctx, vf, -1, &dv) && dv == -3.0 &&
            MLIR_MatchConstantFloat(ctx, vf, 3, &dv) && dv == -3.0 &&
            !MLIR_MatchConstantFloat(ctx, vf, 4, &dv) &&
            MLIR_MatchConstantFloat(ctx, zf, -1, &dz) && dz == 0.0 &&
            MLIR_MatchConstantInt(ctx, ci, -1, &ii) && ii == 7 &&
            MLIR_MatchConstantInt(ctx, vi, 2, &iv) && iv == 0 &&
            MLIR_IsConstantZero(ctx, zf, -1) &&
            MLIR_IsConstantZero(ctx, vi, -1) &&
            !MLIR_IsConstantZero(ctx, cf, -1) &&
            !MLIR_MatchConstantFloat(ctx, arg_v, -1, &df) &&
            MLIR_GetOpType(ctx, MLIR_GetValueDefiningOp(ctx, uf)) == OP_TYPE_LLVM_MLIR_UNDEF &&
            MLIR_GetOpType(ctx, MLIR_GetValueDefiningOp(ctx, pf)) == OP_TYPE_LLVM_MLIR_POISON;

        // Fast-math flags + MLIR_BuildFBinary / MLIR_BuildFUnary / SetOpFastMath.
        MLIR_ValueHandle fadd = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD, v4f32, arg_v, vf, MLIR_FMF_FAST);
        MLIR_ValueHandle fneg = MLIR_BuildFUnary(&b, OP_TYPE_LLVM_FNEG, v4f32, fadd,
                                                 MLIR_FMF_NNAN | MLIR_FMF_NINF);
        MLIR_OpHandle fadd_op = MLIR_GetValueDefiningOp(ctx, fadd);
        MLIR_OpHandle fneg_op = MLIR_GetValueDefiningOp(ctx, fneg);
        bool fmf_ok =
            MLIR_GetOpFastMath(ctx, fadd_op) == MLIR_FMF_FAST &&
            MLIR_GetOpFastMath(ctx, fneg_op) == (MLIR_FMF_NNAN | MLIR_FMF_NINF);
        MLIR_SetOpFastMath(ctx, fneg_op, MLIR_FMF_REASSOC | MLIR_FMF_ARCP);
        fmf_ok = fmf_ok && MLIR_GetOpFastMath(ctx, fneg_op) == (MLIR_FMF_REASSOC | MLIR_FMF_ARCP);

        // Vector & aggregate ops: shufflevector, insertelement, extractelement, extractvalue.
        int32_t mask_vals[4] = { 3, 2, 1, 0 };
        MLIR_NamedAttr shuf_attr = MLIR_Named(MLIR_KEY_MASK,
            MLIR_CreateAttributeDenseI32Array(ctx, mask_vals, 4));
        MLIR_ValueHandle shuf_ops[2] = { fneg, uf };
        MLIR_OpSpec shuf_spec = {0};
        shuf_spec.type = OP_TYPE_LLVM_SHUFFLEVECTOR;
        shuf_spec.operands = shuf_ops;
        shuf_spec.n_operands = 2;
        shuf_spec.result_types = &v4f32;
        shuf_spec.n_results = 1;
        shuf_spec.attrs = &shuf_attr;
        shuf_spec.n_attrs = 1;
        MLIR_ValueHandle shuf = MLIR_GetOpResult(ctx, MLIR_BuildOp(&b, &shuf_spec), 0);

        MLIR_ValueHandle idx0 = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i32_ty, 0);
        MLIR_ValueHandle ins_ops[3] = { shuf, cf, idx0 };
        MLIR_OpSpec ins_spec = {0};
        ins_spec.type = OP_TYPE_LLVM_INSERTELEMENT;
        ins_spec.operands = ins_ops;
        ins_spec.n_operands = 3;
        ins_spec.result_types = &v4f32;
        ins_spec.n_results = 1;
        MLIR_ValueHandle ins = MLIR_GetOpResult(ctx, MLIR_BuildOp(&b, &ins_spec), 0);

        MLIR_ValueHandle ext_ops[2] = { ins, idx0 };
        MLIR_OpSpec ext_spec = {0};
        ext_spec.type = OP_TYPE_LLVM_EXTRACTELEMENT;
        ext_spec.operands = ext_ops;
        ext_spec.n_operands = 2;
        ext_spec.result_types = &f32_ty;
        ext_spec.n_results = 1;
        MLIR_ValueHandle ext = MLIR_GetOpResult(ctx, MLIR_BuildOp(&b, &ext_spec), 0);

        int64_t pos0[1] = { 0 };
        MLIR_NamedAttr ev_attr = MLIR_Named(MLIR_KEY_POSITION,
            MLIR_CreateAttributeDenseI64Array(ctx, pos0, 1));
        MLIR_OpSpec ev_spec = {0};
        ev_spec.type = OP_TYPE_LLVM_EXTRACTVALUE;
        ev_spec.operands = &arg_s;
        ev_spec.n_operands = 1;
        ev_spec.result_types = &f32_ty;
        ev_spec.n_results = 1;
        ev_spec.attrs = &ev_attr;
        ev_spec.n_attrs = 1;
        MLIR_ValueHandle ev = MLIR_GetOpResult(ctx, MLIR_BuildOp(&b, &ev_spec), 0);

        // Common math intrinsics & llvm.call_intrinsic & llvm.call + MLIR_GetCallee.
        MLIR_ValueHandle pw = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_INTR_POW, f32_ty, ext, ev, MLIR_FMF_FAST);
        MLIR_ValueHandle fa = MLIR_BuildFUnary(&b, OP_TYPE_LLVM_INTR_FABS, f32_ty, pw, MLIR_FMF_NONE);
        MLIR_ValueHandle intrin_args[1] = { fa };
        MLIR_ValueHandle ci_res = MLIR_BuildCallIntrinsic(&b, str_lit("llvm.x86.sse.rsqrt.ss"),
            f32_ty, intrin_args, 1, MLIR_FMF_FAST);
        MLIR_ValueHandle void_res = MLIR_BuildCallIntrinsic(&b, str_lit("llvm.donothing"),
            void_ty, NULL, 0, MLIR_FMF_NONE);

        MLIR_NamedAttr call_attr = MLIR_Named(MLIR_KEY_CALLEE,
            MLIR_CreateAttributeSymbolRef(ctx, str_lit("sqrtf")));
        MLIR_OpSpec call_spec = {0};
        call_spec.type = OP_TYPE_LLVM_CALL;
        call_spec.operands = &ci_res;
        call_spec.n_operands = 1;
        call_spec.result_types = &f32_ty;
        call_spec.n_results = 1;
        call_spec.attrs = &call_attr;
        call_spec.n_attrs = 1;
        MLIR_OpHandle call_op = MLIR_BuildOp(&b, &call_spec);
        MLIR_ValueHandle call_res = MLIR_GetOpResult(ctx, call_op, 0);

        MLIR_OpSpec ret_spec = {0};
        ret_spec.type = OP_TYPE_LLVM_RETURN;
        ret_spec.operands = &call_res;
        ret_spec.n_operands = 1;
        MLIR_BuildOp(&b, &ret_spec);

        bool callee_ok =
            str_eq(MLIR_GetCallee(ctx, call_op), str_lit("sqrtf")) &&
            str_eq(MLIR_GetCallee(ctx, MLIR_GetValueDefiningOp(ctx, ci_res)),
                   str_lit("llvm.x86.sse.rsqrt.ss")) &&
            MLIR_GetOpFastMath(ctx, MLIR_GetValueDefiningOp(ctx, ci_res)) == MLIR_FMF_FAST &&
            void_res == MLIR_INVALID_HANDLE &&
            MLIR_IsFunctionDeclaration(ctx, decl_fn) &&
            !MLIR_IsFunctionDeclaration(ctx, def_fn);

        // Passthrough attributes + TargetInfo.
        MLIR_SetFunctionPassthrough(ctx, def_fn, str_lit("noinline"), str_lit(""));
        MLIR_SetFunctionPassthrough(ctx, def_fn, str_lit("reciprocal-estimates"), str_lit("sqrtf:0"));
        MLIR_SetFunctionPassthrough(ctx, def_fn, str_lit("reciprocal-estimates"), str_lit("sqrtf:0,vec-sqrtf:0"));
        MLIR_SetFunctionPassthrough(ctx, def_fn, str_lit("target-features"), str_lit("+avx512f,-fma"));
        bool pt_ok =
            MLIR_HasFunctionPassthrough(ctx, def_fn, str_lit("noinline")) &&
            str_eq(MLIR_GetFunctionPassthrough(ctx, def_fn, str_lit("noinline")), str_lit("")) &&
            MLIR_HasFunctionPassthrough(ctx, def_fn, str_lit("reciprocal-estimates")) &&
            str_eq(MLIR_GetFunctionPassthrough(ctx, def_fn, str_lit("reciprocal-estimates")),
                   str_lit("sqrtf:0,vec-sqrtf:0")) &&
            !MLIR_HasFunctionPassthrough(ctx, def_fn, str_lit("missing"));

        MLIR_TargetInfo x86 = MLIR_TargetFromTriple(str_lit("x86_64-unknown-linux-gnu"),
            MLIR_FEATURE_SSE2 | MLIR_FEATURE_AVX | MLIR_FEATURE_FMA);
        MLIR_TargetInfo arm = MLIR_TargetFromTriple(str_lit("aarch64-apple-darwin"),
            MLIR_FEATURE_NEON);
        bool target_ok =
            MLIR_TargetIsX86(&x86) && !MLIR_TargetIsX86(&arm) &&
            arm.arch == MLIR_ARCH_AARCH64 &&
            MLIR_TargetHasFeature(ctx, &x86, MLIR_INVALID_HANDLE, MLIR_FEATURE_AVX) &&
            MLIR_TargetHasFeature(ctx, &x86, MLIR_INVALID_HANDLE, MLIR_FEATURE_FMA) &&
            !MLIR_TargetHasFeature(ctx, &x86, MLIR_INVALID_HANDLE, MLIR_FEATURE_AVX512F) &&
            MLIR_TargetHasFeature(ctx, &x86, def_fn, MLIR_FEATURE_AVX512F) &&
            !MLIR_TargetHasFeature(ctx, &x86, def_fn, MLIR_FEATURE_FMA) &&
            MLIR_TargetHasFeature(ctx, &x86, def_fn, MLIR_FEATURE_AVX);

        bool verify_ok = MLIR_VerifyOp(ctx, mod);

        print_string(const_ok && fmf_ok && callee_ok && pt_ok && target_ok && verify_ok
                         ? str_lit("llvm completeness: yes\n")
                         : str_lit("llvm completeness: no\n"));
    }

    // ---- pass and pipeline infrastructure (#355) ----------------------------
    {
        MLIR_OpSpec mspec = {0};
        mspec.type = OP_TYPE_MODULE;
        mspec.n_regions = 1;
        MLIR_OpHandle mod = MLIR_CreateOpFromSpec(ctx, &mspec);
        MLIR_BlockHandle mb = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, MLIR_GetOpRegion(ctx, mod, 0), mb);

        MLIR_TypeHandle void_ty = MLIR_CreateTypeLLVMVoid(ctx);
        MLIR_TypeHandle fty = MLIR_CreateTypeLLVMFunction(ctx, void_ty, NULL, 0, false);

        // One declaration (0 blocks) and two definitions (1 block each).
        MLIR_NamedAttr decl_attrs[2] = {
            MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("pdecl"))),
            MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, fty)),
        };
        MLIR_OpSpec fspec = {0};
        fspec.type = OP_TYPE_LLVM_FUNC;
        fspec.attrs = decl_attrs;
        fspec.n_attrs = 2;
        fspec.n_regions = 1;
        MLIR_AppendBlockOp(ctx, mb, MLIR_CreateOpFromSpec(ctx, &fspec));

        const char *fnames[2] = { "pdef0", "pdef1" };
        for (int i = 0; i < 2; i++) {
            MLIR_NamedAttr fattrs[2] = {
                MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, (string){ (char *)fnames[i], 5 })),
                MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, fty)),
            };
            fspec.attrs = fattrs;
            MLIR_OpHandle fn = MLIR_CreateOpFromSpec(ctx, &fspec);
            MLIR_AppendBlockOp(ctx, mb, fn);
            MLIR_BlockHandle fb = MLIR_CreateBlock(ctx);
            MLIR_AppendRegionBlock(ctx, MLIR_GetOpRegion(ctx, fn, 0), fb);
            MLIR_OpSpec rspec = {0};
            rspec.type = OP_TYPE_LLVM_RETURN;
            MLIR_AppendBlockOp(ctx, fb, MLIR_CreateOpFromSpec(ctx, &rspec));
        }

        PassTestState st = {0};
        MLIR_Pass pipeline[2];
        pipeline[0].name = str_lit("test-mod");
        pipeline[0].anchor = MLIR_ANCHOR_MODULE;
        pipeline[0].run = test_mod_pass;
        pipeline[0].state = &st;
        pipeline[1].name = str_lit("test-fn");
        pipeline[1].anchor = MLIR_ANCHOR_FUNCTION;
        pipeline[1].run = test_fn_pass;
        pipeline[1].state = &st;

        DiagCapture cap = {0};
        MLIR_SetDiagnosticHandler(ctx, capture_diag, &cap);
        MLIR_PipelineOptions popts = {0};
        popts.verify_each = true;
        popts.timing = true;
        popts.print_after = str_lit("test-fn");
        bool run_ok = MLIR_RunPasses(ctx, mod, pipeline, 2, &popts);
        // Expected remarks: 2 timing remarks + 1 print_after remark = 3 diagnostics, 0 errors.
        bool remarks_ok = (cap.count == 3 && !cap.saw_error);

        // Failing pass stops the pipeline before the next pass runs.
        PassTestState st2 = {0};
        MLIR_Pass fail_pipe[2];
        fail_pipe[0].name = str_lit("test-fail");
        fail_pipe[0].anchor = MLIR_ANCHOR_MODULE;
        fail_pipe[0].run = test_fail_pass;
        fail_pipe[0].state = NULL;
        fail_pipe[1].name = str_lit("after-fail");
        fail_pipe[1].anchor = MLIR_ANCHOR_MODULE;
        fail_pipe[1].run = test_mod_pass;
        fail_pipe[1].state = &st2;
        bool fail_stopped = !MLIR_RunPasses(ctx, mod, fail_pipe, 2, NULL) && st2.module_runs == 0;

        // verify_each catches an invalid op introduced by a pass.
        MLIR_Pass bad_pipe[1];
        bad_pipe[0].name = str_lit("break-ir");
        bad_pipe[0].anchor = MLIR_ANCHOR_FUNCTION;
        bad_pipe[0].run = test_break_ir_pass;
        bad_pipe[0].state = NULL;
        MLIR_PipelineOptions vopts = {0};
        vopts.verify_each = true;
        DiagCapture vcap = {0};
        MLIR_SetDiagnosticHandler(ctx, capture_diag, &vcap);
        bool verify_caught = !MLIR_RunPasses(ctx, mod, bad_pipe, 1, &vopts) && vcap.saw_error;
        MLIR_SetDiagnosticHandler(ctx, NULL, NULL);

        bool pass_ok = run_ok && st.module_runs == 1 && st.fn_runs == 2 &&
                       st.saw_scratch == 3 && remarks_ok && fail_stopped && verify_caught;
        print_string(pass_ok ? str_lit("pass pipeline API: yes\n")
                             : str_lit("pass pipeline API: no\n"));
    }

    // Generic printer: nested arrays and escaped string attributes (#299).
    {
        MLIR_TypeHandle i32_ty = MLIR_CreateTypeInteger(ctx, 32, false);
        MLIR_AttributeHandle row0[2] = {
            MLIR_CreateAttributeInteger(ctx, 1, i32_ty),
            MLIR_CreateAttributeInteger(ctx, 2, i32_ty),
        };
        MLIR_AttributeHandle row1[1] = {
            MLIR_CreateAttributeInteger(ctx, 3, i32_ty),
        };
        MLIR_AttributeHandle outer[2] = {
            MLIR_CreateAttributeArray(ctx, row0, 2),
            MLIR_CreateAttributeArray(ctx, row1, 1),
        };
        MLIR_AttributeHandle nested_arr = MLIR_CreateAttributeArray(ctx, outer, 2);
        const char raw_bytes[7] = { 'h', 'i', '\n', '\0', '"', '\\', 'x' };
        string raw_str = { (char *)raw_bytes, 7 };
        MLIR_AttributeHandle esc_str = MLIR_CreateAttributeString(ctx, raw_str);

        MLIR_NamedAttr gattrs[2] = {
            MLIR_Named(MLIR_InternAttrKey(ctx, str_lit("nested")), nested_arr),
            MLIR_Named(MLIR_InternAttrKey(ctx, str_lit("text")), esc_str),
        };
        MLIR_OpSpec gspec = {0};
        gspec.type = MLIR_RegisterOpType(ctx, str_lit("test.generic_attrs"));
        gspec.attrs = gattrs;
        gspec.n_attrs = 2;
        MLIR_OpHandle gop = MLIR_CreateOpFromSpec(ctx, &gspec);
        string printed = print_operation_generic(ctx, 0, gop);
        string want_arr = str_lit("nested = [[1, 2], [3]]");
        string want_str = str_lit("text = \"hi\\0A\\00\\\"\\\\x\"");
        bool arr_ok = false, str_ok = false;
        for (size_t i = 0; i + want_arr.size <= printed.size; i++) {
            if (str_eq((string){ printed.str + i, want_arr.size }, want_arr)) { arr_ok = true; break; }
        }
        for (size_t i = 0; i + want_str.size <= printed.size; i++) {
            if (str_eq((string){ printed.str + i, want_str.size }, want_str)) { str_ok = true; break; }
        }
        print_string((arr_ok && str_ok) ? str_lit("generic printer nested/escaped: yes\n")
                                        : str_lit("generic printer nested/escaped: no\n"));
    }

    MLIR_DestroyContext(ctx);
    arena_destroy(arena);
    return 0;
}

