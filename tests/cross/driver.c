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
#include "mlir_generic_printer.h"

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
        MLIR_CreateAttributeInteger(ctx, str_lit(""), 11, i32_ty),
        MLIR_CreateAttributeInteger(ctx, str_lit(""), 22, i32_ty),
        MLIR_CreateAttributeInteger(ctx, str_lit(""), 33, i32_ty),
    };
    *out_array = MLIR_CreateAttributeArray(ctx, str_lit("arr"), elems, 3);
    // Keys in sorted order: upstream's DictionaryAttr sorts its entries.
    // The key text also lives in this frame: attribute names are copied too.
    char keys[2] = { 'a', 'b' };
    MLIR_AttributeHandle entries[2] = {
        MLIR_CreateAttributeInteger(ctx, (string){ &keys[0], 1 }, 44, i32_ty),
        MLIR_CreateAttributeInteger(ctx, (string){ &keys[1], 1 }, 55, i32_ty),
    };
    *out_dict = MLIR_CreateAttributeDict(ctx, str_lit("dict"), entries, 2);
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
    MLIR_DiagSeverity severity;
    MLIR_LocationHandle loc;
    char text[128];
    size_t len;
} DiagCapture;

static void capture_diag(void *user, MLIR_DiagSeverity severity,
                         MLIR_LocationHandle loc, string message) {
    DiagCapture *c = (DiagCapture *)user;
    c->count++;
    c->severity = severity;
    c->loc = loc;
    c->len = message.size < sizeof(c->text) ? message.size : sizeof(c->text);
    for (size_t i = 0; i < c->len; i++) c->text[i] = message.str[i];
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
        for (size_t i = 0; i < MLIR_GetAttributeArraySize(arr); i++) {
            int64_t v = MLIR_GetAttributeInteger(MLIR_GetAttributeArrayElement(arr, i));
            print_string(format(arena, str_lit(" {}"), v));
        }
        print_string(str_lit("\ndict attr from stack buffer:"));
        for (size_t i = 0; i < MLIR_GetAttributeDictSize(dict); i++) {
            MLIR_AttributeHandle e = MLIR_GetAttributeDictElement(dict, i);
            print_string(format(arena, str_lit(" {}={}"), MLIR_GetAttributeName(e),
                                MLIR_GetAttributeInteger(e)));
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

        MLIR_ValueHandle defv = MLIR_CreateValueBlockArg(
            ctx, str_lit("%d"), 0, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_AppendBlockArg(ctx, entry, defv);
        MLIR_ValueHandle replv = MLIR_CreateValueBlockArg(
            ctx, str_lit("%r"), 1, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_AppendBlockArg(ctx, entry, replv);

        // Build cf.br tgt(defv : i32) — zero regular operands, one successor
        // with one successor-operand.
        MLIR_BlockHandle *succs = arena_new_array(arena, MLIR_BlockHandle, 1);
        succs[0] = tgt;
        MLIR_ValueHandle **sops = arena_new_array(arena, MLIR_ValueHandle *, 1);
        sops[0] = arena_new_array(arena, MLIR_ValueHandle, 1);
        sops[0][0] = defv;
        size_t *snums = arena_new_array(arena, size_t, 1);
        snums[0] = 1;
        MLIR_OpHandle br = MLIR_CreateOpWithSuccessors(
            ctx, OP_TYPE_CF_BR,
            NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
            succs, 1, sops, snums,
            MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, str_lit(""), -1);
        MLIR_AppendBlockOp(ctx, entry, br);

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

    MLIR_ValueHandle arg0 = MLIR_CreateValueBlockArg(
        ctx, str_lit("%arg0"), 0, i32_ty, MLIR_INVALID_HANDLE);
    MLIR_AppendBlockArg(ctx, func_block, arg0);
    MLIR_ValueHandle arg1 = MLIR_CreateValueBlockArg(
        ctx, str_lit("%arg1"), 1, i32_ty, MLIR_INVALID_HANDLE);
    MLIR_AppendBlockArg(ctx, func_block, arg1);

    // %0 = arith.addi(%arg0, %arg1) -> i32
    MLIR_ValueHandle add_result = MLIR_CreateValueOpResult(
        ctx, MLIR_INVALID_HANDLE, 0, i32_ty,
        str_lit(""), MLIR_INVALID_HANDLE);

    MLIR_ValueHandle *add_operands = arena_new_array(arena, MLIR_ValueHandle, 2);
    add_operands[0] = arg0;
    add_operands[1] = arg1;
    MLIR_TypeHandle  *add_result_types = arena_new_array(arena, MLIR_TypeHandle, 1);
    add_result_types[0] = i32_ty;
    MLIR_ValueHandle *add_results = arena_new_array(arena, MLIR_ValueHandle, 1);
    add_results[0] = add_result;

    MLIR_OpHandle add_op = MLIR_CreateOp(
        ctx, OP_TYPE_ARITH_ADDI,
        NULL, 0,
        add_result_types, 1,
        add_results, 1,
        add_operands, 2,
        NULL, 0,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);

    // func.return(%0)
    MLIR_ValueHandle *ret_operands = arena_new_array(arena, MLIR_ValueHandle, 1);
    ret_operands[0] = add_result;
    MLIR_OpHandle ret_op = MLIR_CreateOp(
        ctx, OP_TYPE_FUNC_RETURN,
        NULL, 0, NULL, 0, NULL, 0,
        ret_operands, 1,
        NULL, 0,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);

    MLIR_AppendBlockOp(ctx, func_block, add_op);
    MLIR_AppendBlockOp(ctx, func_block, ret_op);
    MLIR_AppendRegionBlock(ctx, func_region, func_block);

    // ---- func.func op --------------------------------------------------------
    MLIR_AttributeHandle sym_name_attr =
        MLIR_CreateAttributeString(ctx, str_lit("sym_name"), str_lit("add"));
    MLIR_AttributeHandle *func_attrs = arena_new_array(arena, MLIR_AttributeHandle, 1);
    func_attrs[0] = sym_name_attr;
    MLIR_RegionHandle *func_regions = arena_new_array(arena, MLIR_RegionHandle, 1);
    func_regions[0] = func_region;

    MLIR_OpHandle func_op = MLIR_CreateOp(
        ctx, OP_TYPE_FUNC_FUNC,
        func_attrs, 1,
        NULL, 0, NULL, 0, NULL, 0,
        func_regions, 1,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, module_block, func_op);

    // ---- module op ----------------------------------------------------------
    MLIR_RegionHandle *module_regions = arena_new_array(arena, MLIR_RegionHandle, 1);
    module_regions[0] = module_region;
    MLIR_OpHandle module_op = MLIR_CreateOp(
        ctx, OP_TYPE_MODULE,
        NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        module_regions, 1,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);

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
                            MLIR_GetLocationFileFilename(cap.loc),
                            (int64_t)MLIR_GetLocationFileLine(cap.loc),
                            (int64_t)MLIR_GetLocationFileColumn(cap.loc),
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
        MLIR_OpHandle dyn_op = MLIR_CreateOp(ctx, dyn, NULL, 0, NULL, 0, NULL, 0,
                                             NULL, 0, NULL, 0, MLIR_INVALID_HANDLE,
                                             MLIR_INVALID_HANDLE, str_lit(""), -1);
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
        MLIR_OpHandle mod2 = MLIR_CreateOp(ctx, OP_TYPE_MODULE, NULL, 0, NULL, 0, NULL, 0,
                                           NULL, 0, NULL, 0, MLIR_INVALID_HANDLE,
                                           MLIR_INVALID_HANDLE, str_lit(""), -1);
        print_string(format(arena,
            str_lit("builtin.module alias: lookup={} register={} module op={} name={}\n"),
            bm_lookup == OP_TYPE_MODULE ? str_lit("yes") : str_lit("no"),
            bm_reg == OP_TYPE_MODULE ? str_lit("yes") : str_lit("no"),
            MLIR_GetOpType(ctx, mod2) == OP_TYPE_MODULE && MLIR_GetOpType(ctx, module_op) == OP_TYPE_MODULE
                ? str_lit("yes") : str_lit("no"),
            MLIR_GetOpTypeName(ctx, bm_reg)));
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
        for (size_t i = 0; i < N; i++) {
            MLIR_ValueHandle res = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, i32_ty,
                                                            str_lit(""), MLIR_INVALID_HANDLE);
            MLIR_ValueHandle ops2[2] = { last, arg };
            MLIR_TypeHandle rt[1] = { i32_ty };
            MLIR_OpHandle add = MLIR_CreateOp(ctx, OP_TYPE_LLVM_ADD, NULL, 0, rt, 1, &res, 1,
                                              ops2, 2, NULL, 0, MLIR_INVALID_HANDLE,
                                              MLIR_INVALID_HANDLE, str_lit(""), -1);
            MLIR_AppendBlockOp(ctx, src, add);
            if (MLIR_GetOpResult(ctx, add, 0) != res || MLIR_GetValueDefiningOp(ctx, res) != add)
                chain_ok = false;
            last = res;
        }
        MLIR_ValueHandle *succ_ops[1] = { &last };
        size_t n_succ_ops[1] = { 1 };
        MLIR_BlockHandle succs[1] = { dst };
        MLIR_OpHandle br = MLIR_CreateOpWithSuccessors(ctx, OP_TYPE_LLVM_BR, NULL, 0, NULL, 0,
                                                       NULL, 0, NULL, 0, NULL, 0, succs, 1,
                                                       succ_ops, n_succ_ops, MLIR_INVALID_HANDLE,
                                                       MLIR_INVALID_HANDLE, str_lit(""), -1);
        MLIR_AppendBlockOp(ctx, src, br);
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

    MLIR_DestroyContext(ctx);
    arena_destroy(arena);
    return 0;
}

