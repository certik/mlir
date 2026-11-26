// tinyC -> MLIR emitter (LLVM-dialect memory model). Uses ONLY mlir_api.h
// plus corec primitives. No upstream MLIR headers are included from this
// directory.
//
// Memory model:
//   - Every local is one !llvm.ptr produced by `llvm.alloca <count> x ELEM`.
//   - Scalar local: ELEM = i32 / f32; load/store with llvm.load / llvm.store.
//   - Struct local: ELEM = !llvm.struct<"S", (...)>; field access via
//     llvm.getelementptr + llvm.load / llvm.store.
//   - int[N] local: ELEM = !llvm.array<N x i32>; indexed via GEP [0, i].
//   - struct[N] local: ELEM = !llvm.array<N x !llvm.struct<...>>; field via
//     GEP [0, i, field_idx].
//   - int* local: ELEM = !llvm.ptr; *p loads p first to get inner ptr.
//   - struct* local: ELEM = !llvm.ptr; p->f loads first then GEP+load.
//
// ABI:
//   - int / float param: scalar.
//   - struct param: caller allocates a fresh copy and passes !llvm.ptr; the
//     callee binds its sym->addr to that block-arg directly (caller-byval).
//   - struct* param: !llvm.ptr; callee alloca's a local !llvm.ptr slot and
//     stores the block-arg into it.
//   - struct return: function returns void and takes a hidden first
//     !llvm.ptr argument (sret) into which the callee writes.
//
// Control flow is emitted in cf-form (cf.br / cf.cond_br) for if/while/for
// and `break`/`continue`/early-return. Short-circuit && / || still use
// scf.if-with-result inside the expression itself; SCFToCF runs as part of
// MLIR_LowerToLLVMDialect.

#include "tinyc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <stdlib.h>
#include <base/arena.h>
#include <base/format.h>
#include <base/io.h>
#include <base/string.h>

// ----- symbol table -----

typedef struct Sym {
    string name;
    Type   type;                // declared type
    MLIR_ValueHandle addr;      // !llvm.ptr to this local's storage
    StructDef *sdef;            // for TY_STRUCT / TY_PTR_STRUCT / TY_ARRAY_STRUCT
    bool   is_global;           // if true, addr is unused; reload via addressof
    string global_sym;          // sym_name of the llvm.mlir.global (for globals)
    uint32_t address_space;     // LLVM pointer AS for global addressof (0=default)
    bool   wide_slot;           // a char / short / bool local whose address is
                                // never taken, kept in an i32 slot (see
                                // local_slot_type)
    bool   is_ssa;              // immutable scalar/pointer param or local:
                                // `addr` holds the SSA value directly (no alloca)
    struct Sym *next;
} Sym;

typedef struct Scope {
    Sym *head;
    struct Scope *parent;
} Scope;

static Sym *scope_lookup(Scope *s, string name) {
    size_t n = name.size;
    char c0 = n ? name.str[0] : 0;
    for (Scope *cur = s; cur; cur = cur->parent) {
        for (Sym *sy = cur->head; sy; sy = sy->next) {
            if (sy->name.size == n &&
                (n == 0 || (sy->name.str[0] == c0 &&
                 (n == 1 || memcmp(sy->name.str + 1, name.str + 1, n - 1) == 0))))
                return sy;
        }
    }
    return NULL;
}

// ----- emit context -----

typedef struct LoopCtx {
    MLIR_BlockHandle continue_block;
    MLIR_BlockHandle break_block;
    struct LoopCtx *parent;
} LoopCtx;

typedef struct {
    Type        type;          // declared slot type
    StructDef  *sdef;          // resolved if struct/struct*/struct[]
    // ABI: every slot maps to exactly ONE flat MLIR operand/result type.
    // For struct values that's !llvm.ptr (caller-byval); for pointer/scalar
    // it's the natural type. For sret returns the callee's flat return is
    // 0 and a hidden !llvm.ptr is prepended to the parameters.
} SlotInfo;

typedef struct FuncSig {
    string         name;
    Func          *func;
    SlotInfo       ret;
    SlotInfo      *params;
    size_t         n_params;
    bool           sret;             // struct return -> hidden first ptr param
    bool           is_variadic;      // C `f(T, ...)` — emitted as `llvm.func`
                                     // and called via `llvm.call` with a
                                     // `var_callee_type` attribute.
    bool           is_used;          // any reachable call site referenced this
                                     // signature; used to suppress extern
                                     // decls for forward-declared functions
                                     // that are never called (otherwise they
                                     // turn into spurious wasm imports that
                                     // fail to resolve at link time).
    MLIR_TypeHandle *flat_in_tys;    // including sret arg if sret
    size_t         n_flat_in;
    MLIR_TypeHandle *flat_out_tys;   // 0 if void/sret, 1 otherwise
    size_t         n_flat_out;
    MLIR_TypeHandle fn_ty;           // FunctionType (non-variadic) or
                                     // LLVMFunctionType (variadic).
    MLIR_TypeHandle llvm_fn_ty;      // LLVMFunctionType — only set for
                                     // variadic; used as `var_callee_type`.
    bool           cursor_va;        // variadic function DEFINED in this TU:
                                     // lowered with tinyC's portable 8-byte
                                     // cursor va_list (hidden trailing ptr
                                     // param) instead of the target's native
                                     // varargs ABI. Extern variadic callees
                                     // (e.g. host printf) keep cursor_va=false.
    bool           defined_here;     // the body is in this TU, so a char /
                                     // short / bool result is already
                                     // narrowed (by its return statement)
    MLIR_AttributeHandle callee_attr; // cached SymbolRef attr for sig->name
} FuncSig;

typedef struct StructTypeEntry {
    StructDef       *sd;
    MLIR_TypeHandle  ty;
} StructTypeEntry;

typedef struct StringEntry {
    string  bytes;
    string  sym;        // global symbol name like ".str.0"
} StringEntry;

typedef struct {
    uint32_t as;
    MLIR_TypeHandle ptr;
} PtrAsCacheEntry;

typedef struct {
    MLIR_Context *ctx;
    Arena *arena;
    MLIR_TypeHandle i32;
    MLIR_TypeHandle i1;
    MLIR_TypeHandle i64;
    MLIR_TypeHandle i64_signed;
    MLIR_TypeHandle i8;
    MLIR_TypeHandle i16;
    MLIR_TypeHandle f32;
    MLIR_TypeHandle f64;
    MLIR_TypeHandle index;
    MLIR_TypeHandle ptr;             // !llvm.ptr
    MLIR_TypeHandle size_t_ty;       // i32 on wasm32, i64 on native (matches `unsigned long`)
    MLIR_AttributeHandle cmpi_pred_attrs[10]; // cached predicate attrs 0..9
    MLIR_AttributeHandle small_i32_attrs[128]; // cached i32 constant attrs 0..127
    MLIR_AttributeHandle small_i64_attrs[32];  // cached i64 constant attrs 0..31
    MLIR_AttributeHandle gep_0k_attrs[32];     // cached GEP [0, k] attrs 0..31
    MLIR_AttributeHandle gep_0dyn_attr;        // cached GEP [0, dyn] attr
    MLIR_AttributeHandle gep_dyn_attr;         // cached GEP [dyn] attr
    MLIR_TypeHandle      ty_attr_k[128];       // direct-mapped cache: type -> TypeAttr
    MLIR_AttributeHandle ty_attr_v[128];
    bool                 caches_init;
    bool             target_wasm32;  // mirrors program->target_wasm32, cached on E for convenience
    MLIR_BlockHandle cur_block;
    MLIR_BlockHandle entry_block;    // function entry block, alloca insertion point
    MLIR_ValueHandle entry_const_one; // i64 constant 1 in entry_block, shared by all allocas
    size_t entry_alloca_insert_idx;  // next position in entry_block for hoisted allocas
    bool terminated;
    MLIR_RegionHandle func_region;
    MLIR_LocationHandle loc;
    int next_ssa;
    LoopCtx *loops;
    // Linked list of `goto` labels in the currently-emitting function.
    // Pre-populated at emit_func entry with one entry per ST_LABEL found
    // anywhere in the body. Reset to NULL between functions.
    struct LabelBlock {
        string name;
        MLIR_BlockHandle block;
        struct LabelBlock *next;
    } *labels;
    Program *program;
    FuncSig *sigs;
    size_t   n_sigs;
    size_t  *sig_map;       // open-addressing hash table: 0=empty, idx+1 into sigs
    size_t   sig_map_cap;
    StructDef **struct_map; // open-addressing hash table: NULL=empty
    size_t   struct_map_cap;
    ProgramEnum **enum_map; // open-addressing hash table: NULL=empty
    size_t   enum_map_cap;
    bool     enum_map_built;
    Sym    **global_map;    // open-addressing hash table: NULL=empty
    size_t   global_map_cap;
    size_t   global_map_n;
    size_t  *struct_ty_map; // open-addressing hash table: 0=empty, idx+1 into struct_types
    size_t   struct_ty_map_cap;
    size_t   struct_ty_map_n;
    size_t  *str_map;       // open-addressing hash table: 0=empty, idx+1 into strings
    size_t   str_map_cap;
    FuncSig *cur_sig;
    MLIR_ValueHandle cur_sret_ptr;   // for struct-returning functions
    MLIR_ValueHandle cur_va_args;    // cursor_va functions: the hidden
                                     // trailing ptr param (caller-packed
                                     // 8-byte-slot varargs buffer) that
                                     // va_start copies into the va_list
    StructTypeEntry *struct_types;
    size_t           n_struct_types;
    StringEntry     *strings;        // dedup'd string-literal pool
    size_t           n_strings;
    size_t           cap_strings;
    PtrAsCacheEntry *ptr_as_cache;
    size_t           n_ptr_as_cache, cap_ptr_as_cache;
    MLIR_BlockHandle module_block;
    Sym             *globals;        // module-scope symbols

    bool             need_syscall6_stub;   // emit @__tinyc_syscall6 extern decl
    int              cur_line;       // last AST node line entered; used by
                                     // EMIT_ERR for diagnostic line numbers.
    int              err_count;      // count of EMIT_ERR diagnostics
    bool             struct_src_int_ok; // see resolve_struct_source
    // The lvalue of the compound assignment being emitted (`a[i++] += x`,
    // `++*p`) and its already-resolved address. The desugared rhs reads the
    // same Expr node; emit_lvalue returns this instead of evaluating the
    // node (and its side effects) a second time.
    Expr            *compound_lv_expr;
    // An assignment emitted as a statement (or for-loop step): its value is
    // not used, so a store into char / short / bool storage need not
    // re-extend the stored value (the wasm path has no DCE to drop it).
    Expr            *unused_assign;
    // Names whose address `&name` the current function takes anywhere
    // (collected by collect_addr_taken before its body is emitted).
    string          *addr_taken;
    size_t           n_addr_taken, cap_addr_taken;
    // Names assigned or address-taken anywhere in the current function
    // (collected alongside addr_taken). Unmutated scalar/pointer params and
    // initialized locals bind directly to SSA values without alloca/load/store.
    string          *mutated;
    size_t           n_mutated, cap_mutated;
    uint64_t         mutated_bloom;
    bool             has_goto;
    struct LValS    *compound_lv;
    // What is known about the most recently emitted i32 values (see
    // fact_put): lets emit_narrow_i32 / bool conversion fold constants and
    // skip narrowing a value that is already in range.
    // The last value a fact was recorded for, and the fact:
    MLIR_ValueHandle fact_v;
    int              fact_kind;          // FACT_CONST or FACT_RANGE (0: none)
    int              fact_bits;          // FACT_RANGE: fits `bits` bits ...
    bool             fact_unsigned;      // ... zero-extended (else sign-extended)
    int64_t          fact_c;             // FACT_CONST: the value
} E;

enum { FACT_CONST = 1, FACT_RANGE = 2 };

static void fact_put(E *e, MLIR_ValueHandle v, int kind, int bits, bool is_unsigned,
                     int64_t c) {
    e->fact_v = v; e->fact_kind = kind; e->fact_bits = bits;
    e->fact_unsigned = is_unsigned; e->fact_c = c;
}

// The fact kind recorded for `v` (0 if none); its details are in e->fact_*.
static int fact_get(E *e, MLIR_ValueHandle v) {
    return (e->fact_kind && e->fact_v == v) ? e->fact_kind : 0;
}

// All emit-time diagnostics route through this macro so they carry a line
// number. `msg` must be a C string literal (it is concatenated with the
// "tinyc emit error at line {}: " prefix at preprocess time and wrapped in
// str_lit). The line is taken from e->cur_line, which is updated at the
// entry of emit_expr / emit_lvalue / emit_stmt / emit_func / per-struct
// pre-pass loops.
#define EMIT_ERR(e, msg, ...) \
    do { \
        println(str_lit("tinyc emit error at line {}: " msg), \
                (int64_t)((e)->cur_line), ##__VA_ARGS__); \
        (e)->err_count++; \
    } while (0)

// An emit-time warning. Like preprocess.c's diagnostics it goes to stderr,
// since stdout can carry the --emit output.
static void emit_warn(E *e, string msg) {
    string text = format(e->arena, str_lit("tinyc emit warning at line {}: {}\n"),
                         (int64_t)e->cur_line, msg);
    ciovec_t iov;
    iov.buf = text.str;
    iov.buf_len = text.size;
    write_all(PLATFORM_STDERR_FD, &iov, 1);
}

static uint32_t emit_str_hash(string s) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < s.size; i++) {
        h ^= (uint8_t)s.str[i];
        h *= 16777619u;
    }
    return h;
}

static StructDef *find_struct(E *e, string name) {
    if (!e->program) return NULL;
    size_t n = e->program->structs.size;
    if (n == 0) return NULL;
    if (!e->struct_map) {
        size_t cap = 16;
        while (cap < n * 2u + 1u) cap <<= 1;
        e->struct_map = arena_new_array(e->arena, StructDef *, cap);
        memset(e->struct_map, 0, cap * sizeof(StructDef *));
        e->struct_map_cap = cap;
        size_t mask = cap - 1;
        for (size_t i = 0; i < n; i++) {
            StructDef *sd = e->program->structs.data[i];
            size_t slot = emit_str_hash(sd->name) & mask;
            while (e->struct_map[slot]) {
                if (str_eq(e->struct_map[slot]->name, sd->name)) break;
                slot = (slot + 1) & mask;
            }
            if (!e->struct_map[slot]) e->struct_map[slot] = sd;
        }
    }
    size_t mask = e->struct_map_cap - 1;
    size_t slot = emit_str_hash(name) & mask;
    while (e->struct_map[slot]) {
        if (str_eq(e->struct_map[slot]->name, name)) return e->struct_map[slot];
        slot = (slot + 1) & mask;
    }
    return NULL;
}

static MLIR_TypeHandle find_struct_type(E *e, StructDef *sd) {
    if (e->n_struct_types == 0) return MLIR_INVALID_HANDLE;
    if (e->struct_ty_map_n != e->n_struct_types) {
        size_t cap = 16;
        while (cap < e->n_struct_types * 2u + 1u) cap <<= 1;
        e->struct_ty_map = arena_new_array(e->arena, size_t, cap);
        memset(e->struct_ty_map, 0, cap * sizeof(size_t));
        e->struct_ty_map_cap = cap;
        e->struct_ty_map_n = e->n_struct_types;
        size_t mask = cap - 1;
        for (size_t i = 0; i < e->n_struct_types; i++) {
            size_t slot = (((uintptr_t)e->struct_types[i].sd * 2654435761u) >> 4) & mask;
            while (e->struct_ty_map[slot]) slot = (slot + 1) & mask;
            e->struct_ty_map[slot] = i + 1;
        }
    }
    size_t mask = e->struct_ty_map_cap - 1;
    size_t slot = (((uintptr_t)sd * 2654435761u) >> 4) & mask;
    while (e->struct_ty_map[slot]) {
        size_t idx = e->struct_ty_map[slot] - 1;
        if (e->struct_types[idx].sd == sd) return e->struct_types[idx].ty;
        slot = (slot + 1) & mask;
    }
    return MLIR_INVALID_HANDLE;
}

static int struct_field_index(StructDef *sd, string name) {
    size_t n = name.size;
    char c0 = n ? name.str[0] : 0;
    size_t nf = sd->fields.size;
    for (size_t i = 0; i < nf; i++) {
        string fn = sd->fields.data[i].name;
        if (fn.size == n && (n == 0 || (fn.str[0] == c0 &&
            (n == 1 || memcmp(fn.str + 1, name.str + 1, n - 1) == 0))))
            return (int)i;
    }
    return -1;
}

static FuncSig *find_sig(E *e, string name) {
    if (!e->sig_map || e->n_sigs == 0) return NULL;
    size_t mask = e->sig_map_cap - 1;
    size_t slot = emit_str_hash(name) & mask;
    while (e->sig_map[slot]) {
        size_t idx = e->sig_map[slot] - 1;
        if (str_eq(e->sigs[idx].name, name)) {
            e->sigs[idx].is_used = true;
            return &e->sigs[idx];
        }
        slot = (slot + 1) & mask;
    }
    return NULL;
}

// Emit-time enumerator lookup. Resolved AFTER locals/globals/functions
// so that an inner-scope variable with the same name correctly shadows
// the enum constant (e.g. `enum {A=1}; int f(){ int A=2; return A; }`
// must return 2, not 1).
static bool find_enum(E *e, string name, int64_t *out_value) {
    if (!e->program || !e->program->enums) return false;
    if (!e->enum_map_built) {
        e->enum_map_built = true;
        size_t n = 0;
        for (ProgramEnum *pe = e->program->enums; pe; pe = pe->next) n++;
        size_t cap = 16;
        while (cap < n * 2u + 1u) cap <<= 1;
        e->enum_map = arena_new_array(e->arena, ProgramEnum *, cap);
        memset(e->enum_map, 0, cap * sizeof(ProgramEnum *));
        e->enum_map_cap = cap;
        size_t mask = cap - 1;
        for (ProgramEnum *pe = e->program->enums; pe; pe = pe->next) {
            size_t slot = emit_str_hash(pe->name) & mask;
            while (e->enum_map[slot]) {
                if (str_eq(e->enum_map[slot]->name, pe->name)) break;
                slot = (slot + 1) & mask;
            }
            if (!e->enum_map[slot]) e->enum_map[slot] = pe;
        }
    }
    size_t mask = e->enum_map_cap - 1;
    size_t slot = emit_str_hash(name) & mask;
    while (e->enum_map[slot]) {
        if (str_eq(e->enum_map[slot]->name, name)) {
            *out_value = e->enum_map[slot]->value;
            return true;
        }
        slot = (slot + 1) & mask;
    }
    return false;
}

// The MLIR type of a value of kind `k` in registers, in function signatures
// and in calls: char / short / bool values are i32 there (see
// scalar_mem_type for their storage).
static MLIR_TypeHandle scalar_mlir_type(E *e, TypeKind k) {
    if (k == TY_F32) return e->f32;
    if (k == TY_F64) return e->f64;
    if (k == TY_I64) return e->i64;
    if (k == TY_PTR_STRUCT || k == TY_PTR_I32 || k == TY_PTR_CHAR ||
        k == TY_PTR_VOID || k == TY_FNPTR || k == TY_PTR_PTR ||
        k == TY_VA_LIST) return e->ptr;
    return e->i32;
}

// Storage type of an int-sized integer of C type `t` (a scalar, an array
// element or a pointee): i8 for char / bool, i16 for short, else i32.
static MLIR_TypeHandle int_mem_type(E *e, const Type *t) {
    return t->int_bits == 8 ? e->i8 : t->int_bits == 16 ? e->i16 : e->i32;
}

// Storage type of a scalar (local, global, struct field) of type `t`: as
// scalar_mlir_type, but 1 / 2 bytes for char / bool / short, as in C.
static MLIR_TypeHandle scalar_mem_type(E *e, const Type *t) {
    if (t->kind == TY_I32) return int_mem_type(e, t);
    return scalar_mlir_type(e, t->kind);
}

static MLIR_TypeHandle local_slot_type(E *e, string name, const Type *t, bool *wide);

// Element type of an array type `t` (TY_ARRAY_I32 / TY_ARRAY_F32 /
// TY_ARRAY_PTR_*), not counting a second dimension.
static MLIR_TypeHandle array_elem_type(E *e, const Type *t) {
    if (t->kind == TY_ARRAY_F32) return e->f32;
    if (t->kind == TY_ARRAY_PTR_STRUCT || t->kind == TY_ARRAY_PTR_CHAR) return e->ptr;
    if (t->array_elem_is_i64) return e->i64;
    return int_mem_type(e, t);
}

// The whole array type for TY_ARRAY_I32 / TY_ARRAY_F32 / TY_ARRAY_PTR_*:
// `[array_len x elem]`, or `[array_len x [array_len2 x elem]]` for 2D.
static MLIR_TypeHandle array_mlir_type(E *e, const Type *t) {
    MLIR_TypeHandle elem = array_elem_type(e, t);
    if (t->array_len2 != 0)
        elem = MLIR_CreateTypeLLVMArray(e->ctx, elem, (uint64_t)t->array_len2);
    return MLIR_CreateTypeLLVMArray(e->ctx, elem, (uint64_t)t->array_len);
}

static MLIR_LocationHandle eloc(E *e, int line) {
    (void)line;
    return e->loc;
}

static string ssa_name(E *e) {
    int n = e->next_ssa++;
    return format(e->arena, str_lit("%{}"), (int64_t)n);
}

static MLIR_BlockHandle new_cfg_block(E *e) {
    MLIR_BlockHandle b = MLIR_CreateBlock(e->ctx);
    // Append the new block to the region that currently owns
    // `e->cur_block`, not blindly to the function body. The CFG-based
    // emitters (if, while, ternary, ...) can run while emit_expr is
    // recursing inside another op's region (e.g. an scf.if then-block
    // emitted by &&/||): the ternary's new blocks must live in that
    // inner region, otherwise the resulting IR has cf.cond_br ops
    // pointing at successors outside their own region and downstream
    // passes (cf->scf lift, wasmssa lower, ...) reject it.
    MLIR_RegionHandle target = e->func_region;
    if (e->cur_block != MLIR_INVALID_HANDLE) {
        MLIR_RegionHandle r = MLIR_GetBlockParentRegion(e->ctx, e->cur_block);
        if (r != MLIR_INVALID_HANDLE) target = r;
    }
    MLIR_AppendRegionBlock(e->ctx, target, b);
    return b;
}

static MLIR_OpHandle emit_op(E *e,
                             MLIR_OpType type,
                             MLIR_TypeHandle *result_types, size_t n_result_types,
                             MLIR_ValueHandle *results, size_t n_results,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_NamedAttr *attrs, size_t n_attrs,
                             MLIR_RegionHandle *regions, size_t n_regions) {
    MLIR_OpSpec spec = {0};
    spec.type = type;
    spec.operands = operands;
    spec.n_operands = (uint32_t)n_operands;
    spec.result_types = result_types;
    spec.n_results = (uint32_t)n_result_types;
    spec.attrs = attrs;
    spec.n_attrs = (uint32_t)n_attrs;
    spec.regions = regions;
    spec.n_regions = (uint32_t)n_regions;
    spec.loc = eloc(e, 0);
    MLIR_OpHandle op = MLIR_CreateOpFromSpec(e->ctx, &spec);
    if (results) {
        for (size_t i = 0; i < n_results; i++) results[i] = MLIR_GetOpResult(e->ctx, op, i);
    }
    if (e->cur_block) MLIR_AppendBlockOp(e->ctx, e->cur_block, op);
    return op;
}

static MLIR_ValueHandle emit_op1(E *e, MLIR_OpType type, MLIR_TypeHandle rt,
                                 const MLIR_ValueHandle *operands, uint32_t n_operands,
                                 const MLIR_NamedAttr *attrs, uint32_t n_attrs) {
    MLIR_OpSpec spec = {0};
    spec.type = type;
    spec.operands = operands;
    spec.n_operands = n_operands;
    spec.result_types = &rt;
    spec.n_results = 1;
    spec.attrs = attrs;
    spec.n_attrs = n_attrs;
    spec.loc = eloc(e, 0);
    MLIR_OpHandle op = MLIR_CreateOpFromSpec(e->ctx, &spec);
    if (e->cur_block) MLIR_AppendBlockOp(e->ctx, e->cur_block, op);
    return MLIR_GetOpResult(e->ctx, op, 0);
}

// Emit a function-return op. The right op depends on the surrounding
// function dialect: variadic functions are emitted as `llvm.func`, all
// others as `func.func`. Number of operands is 0 (void/sret returns) or 1
// (scalar/pointer return). LLVMReturnOp accepts at most one operand.
static void emit_return_op(E *e, MLIR_ValueHandle *ops, size_t n_ops) {
    if (e->cur_sig && e->cur_sig->is_variadic && !e->cur_sig->cursor_va) {
        emit_op(e, OP_TYPE_LLVM_RETURN,
                NULL, 0, NULL, 0, ops, n_ops, NULL, 0, NULL, 0);
    } else {
        emit_op(e, OP_TYPE_FUNC_RETURN,
                NULL, 0, NULL, 0, ops, n_ops, NULL, 0, NULL, 0);
    }
}

static void emit_branch(E *e, MLIR_BlockHandle target) {
    if (e->terminated) return;
    MLIR_Builder b = MLIR_BuilderAtEnd(e->ctx, e->cur_block);
    b.loc = eloc(e, 0);
    MLIR_BuildBr(&b, OP_TYPE_CF_BR, target, NULL, 0);
    e->terminated = true;
}

static void emit_cond_branch(E *e, MLIR_ValueHandle cond,
                             MLIR_BlockHandle true_b, MLIR_BlockHandle false_b) {
    if (e->terminated) return;
    MLIR_Builder b = MLIR_BuilderAtEnd(e->ctx, e->cur_block);
    b.loc = eloc(e, 0);
    MLIR_BuildCondBr(&b, OP_TYPE_CF_COND_BR, cond, true_b, NULL, 0, false_b, NULL, 0);
    e->terminated = true;
}

// ----- constants -----

static void emit_init_attr_caches(E *e) {
    if (e->caches_init) return;
    e->caches_init = true;
    e->i64_signed = MLIR_CreateTypeInteger(e->ctx, 64, true);
    for (int64_t p = 0; p < 10; p++) {
        e->cmpi_pred_attrs[p] = MLIR_CreateAttributeInteger(e->ctx, p, e->i64_signed);
    }
    for (int64_t v = 0; v < 128; v++) {
        e->small_i32_attrs[v] = MLIR_CreateAttributeInteger(e->ctx, v, e->i32);
    }
    for (int64_t v = 0; v < 32; v++) {
        e->small_i64_attrs[v] = MLIR_CreateAttributeInteger(e->ctx, v, e->i64);
    }
    for (int32_t k = 0; k < 32; k++) {
        int32_t pair[2] = { 0, k };
        e->gep_0k_attrs[k] = MLIR_CreateAttributeDenseI32Array(e->ctx, pair, 2);
    }
    int32_t pair_dyn[2] = { 0, (int32_t)0x80000000 };
    e->gep_0dyn_attr = MLIR_CreateAttributeDenseI32Array(e->ctx, pair_dyn, 2);
    int32_t single_dyn[1] = { (int32_t)0x80000000 };
    e->gep_dyn_attr = MLIR_CreateAttributeDenseI32Array(e->ctx, single_dyn, 1);
}

static MLIR_ValueHandle emit_const_i32(E *e, int64_t v) {
    MLIR_AttributeHandle a;
    if (v >= 0 && v < 128) {
        if (!e->caches_init) emit_init_attr_caches(e);
        a = e->small_i32_attrs[v];
    } else {
        a = MLIR_CreateAttributeInteger(e->ctx, v, e->i32);
    }
    MLIR_NamedAttr val = MLIR_Named(MLIR_KEY_VALUE, a);
    MLIR_ValueHandle r = emit_op1(e, OP_TYPE_ARITH_CONSTANT, e->i32, NULL, 0, &val, 1);
    fact_put(e, r, FACT_CONST, 0, false, (int64_t)(int32_t)v);
    return r;
}

static MLIR_ValueHandle emit_const_f32(E *e, double v) {
    MLIR_NamedAttr val = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeFloat(e->ctx, v, e->f32));
    return emit_op1(e, OP_TYPE_ARITH_CONSTANT, e->f32, NULL, 0, &val, 1);
}

static MLIR_ValueHandle emit_const_f64(E *e, double v) {
    MLIR_NamedAttr val = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeFloat(e->ctx, v, e->f64));
    return emit_op1(e, OP_TYPE_ARITH_CONSTANT, e->f64, NULL, 0, &val, 1);
}

// arith.extf : f32 -> f64
static MLIR_ValueHandle emit_fpext_f32_to_f64(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_ARITH_EXTF, e->f64, &v, 1, NULL, 0);
}

// arith.truncf : f64 -> f32
static MLIR_ValueHandle emit_fptrunc_f64_to_f32(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_ARITH_TRUNCF, e->f32, &v, 1, NULL, 0);
}

static MLIR_ValueHandle emit_const_i64(E *e, int64_t v) {
    MLIR_AttributeHandle a;
    if (v >= 0 && v < 32) {
        if (!e->caches_init) emit_init_attr_caches(e);
        a = e->small_i64_attrs[v];
    } else {
        a = MLIR_CreateAttributeInteger(e->ctx, v, e->i64);
    }
    MLIR_NamedAttr val = MLIR_Named(MLIR_KEY_VALUE, a);
    return emit_op1(e, OP_TYPE_ARITH_CONSTANT, e->i64, NULL, 0, &val, 1);
}

static MLIR_ValueHandle emit_binop(E *e, MLIR_OpType type,
                                   MLIR_TypeHandle rt,
                                   MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle ops[2] = { a, b };
    return emit_op1(e, type, rt, ops, 2, NULL, 0);
}

// Truncate the i32 value `v` to its low `bits` (8 or 16) bits and extend it
// back to i32: zero-extend (andi) for an unsigned type, sign-extend
// (shli + shrsi) otherwise. Narrow integer types are i32 values in tinyc.
static MLIR_ValueHandle emit_narrow_i32(E *e, MLIR_ValueHandle v, int bits,
                                        bool is_unsigned) {
    int f = fact_get(e, v);
    if (f == FACT_CONST) {
        Type t = (Type){0};
        t.int_bits = bits;
        t.int_unsigned = is_unsigned;
        return emit_const_i32(e, tinyc_narrow_const(e->fact_c, t));
    }
    // Already in range: an unsigned b'-bit value fits an unsigned target of
    // b >= b' bits, but a signed target only when b' < b.
    if (f == FACT_RANGE &&
        (is_unsigned ? (e->fact_unsigned && e->fact_bits <= bits)
                     : (e->fact_unsigned ? e->fact_bits < bits
                                         : e->fact_bits <= bits)))
        return v;
    MLIR_ValueHandle r;
    if (is_unsigned) {
        r = emit_binop(e, OP_TYPE_ARITH_ANDI, e->i32,
                       v, emit_const_i32(e, (1 << bits) - 1));
    } else {
        MLIR_ValueHandle sh = emit_const_i32(e, 32 - bits);
        MLIR_ValueHandle hi = emit_binop(e, OP_TYPE_ARITH_SHLI,
                                         e->i32, v, sh);
        r = emit_binop(e, OP_TYPE_ARITH_SHRSI, e->i32, hi, sh);
    }
    fact_put(e, r, FACT_RANGE, bits, is_unsigned, 0);
    return r;
}

static MLIR_ValueHandle emit_cmpi(E *e, int64_t predicate,
                                  MLIR_ValueHandle a, MLIR_ValueHandle b) {
    if (!e->caches_init) emit_init_attr_caches(e);
    MLIR_ValueHandle ops[2] = { a, b };
    MLIR_AttributeHandle pa = (predicate >= 0 && predicate < 10)
        ? e->cmpi_pred_attrs[predicate]
        : MLIR_CreateAttributeInteger(e->ctx, predicate, e->i64_signed);
    MLIR_NamedAttr pred = MLIR_Named(MLIR_KEY_PREDICATE, pa);
    return emit_op1(e, OP_TYPE_ARITH_CMPI, e->i1, ops, 2, &pred, 1);
}

static MLIR_ValueHandle emit_cmpf(E *e, int64_t predicate,
                                  MLIR_ValueHandle a, MLIR_ValueHandle b) {
    if (!e->caches_init) emit_init_attr_caches(e);
    MLIR_ValueHandle ops[2] = { a, b };
    MLIR_AttributeHandle pa = (predicate >= 0 && predicate < 10)
        ? e->cmpi_pred_attrs[predicate]
        : MLIR_CreateAttributeInteger(e->ctx, predicate, e->i64_signed);
    MLIR_NamedAttr pred = MLIR_Named(MLIR_KEY_PREDICATE, pa);
    return emit_op1(e, OP_TYPE_ARITH_CMPF, e->i1, ops, 2, &pred, 1);
}

static MLIR_ValueHandle emit_icmp_ptr(E *e, int64_t predicate,
                                      MLIR_ValueHandle a, MLIR_ValueHandle b) {
    if (!e->caches_init) emit_init_attr_caches(e);
    MLIR_ValueHandle ops[2] = { a, b };
    MLIR_AttributeHandle pa = (predicate >= 0 && predicate < 10)
        ? e->cmpi_pred_attrs[predicate]
        : MLIR_CreateAttributeInteger(e->ctx, predicate, e->i64_signed);
    MLIR_NamedAttr pred = MLIR_Named(MLIR_KEY_PREDICATE, pa);
    return emit_op1(e, OP_TYPE_LLVM_ICMP, e->i1, ops, 2, &pred, 1);
}

static MLIR_ValueHandle emit_null_ptr(E *e) {
    return emit_op1(e, OP_TYPE_LLVM_MLIR_ZERO, e->ptr, NULL, 0, NULL, 0);
}

static MLIR_ValueHandle emit_extsi_i32_to_i64(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_ARITH_EXTSI, e->i64, &v, 1, NULL, 0);
}

// Zero-extend i32 -> i64 (arith.extui), used to widen unsigned operands.
static MLIR_ValueHandle emit_extui_i32_to_i64(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_ARITH_EXTUI, e->i64, &v, 1, NULL, 0);
}

// Reinterpret a pointer as an i64 (llvm.ptrtoint). Used to pass pointer
// arguments to __builtin_syscall6 without requiring an explicit (long) cast.
static MLIR_ValueHandle emit_ptrtoint_i64(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_LLVM_PTRTOINT, e->i64, &v, 1, NULL, 0);
}

// Reinterpret the i64 `v` as a pointer (llvm.inttoptr).
static MLIR_ValueHandle emit_inttoptr_i64(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_LLVM_INTTOPTR, e->ptr, &v, 1, NULL, 0);
}

// Floating-point negation (llvm.fneg): flips the sign bit, so -(+0.0) is
// -0.0. `0.0 - x` would give +0.0 there.
static MLIR_ValueHandle emit_fneg(E *e, MLIR_TypeHandle ft, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_LLVM_FNEG, ft, &v, 1, NULL, 0);
}

// Sign-extend i8 / i16 -> i32 (arith.extsi).
static MLIR_ValueHandle emit_extsi_to_i32(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_ARITH_EXTSI, e->i32, &v, 1, NULL, 0);
}

// Zero-extend i8 / i16 -> i32 (arith.extui), used to widen loads of
// `unsigned char` / `unsigned short` / bool storage.
static MLIR_ValueHandle emit_extui_to_i32(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_ARITH_EXTUI, e->i32, &v, 1, NULL, 0);
}

// Truncate i64 -> i32 (arith.trunci).
static MLIR_ValueHandle emit_trunci_i64_to_i32(E *e, MLIR_ValueHandle v) {
    return emit_op1(e, OP_TYPE_ARITH_TRUNCI, e->i32, &v, 1, NULL, 0);
}

// Truncate any integer to the narrower integer type `ty` (arith.trunci).
static MLIR_ValueHandle emit_trunci_to(E *e, MLIR_ValueHandle v, MLIR_TypeHandle ty) {
    return emit_op1(e, OP_TYPE_ARITH_TRUNCI, ty, &v, 1, NULL, 0);
}

// arith.constant <v> : `ty` (i8 / i16) — narrow stores of constants, e.g.
// char-array string-literal init. `v` is the zero-extended bit pattern
// (0..255 / 0..65535): the backends hold narrow values zero-extended.
static MLIR_ValueHandle emit_const_narrow(E *e, int64_t v, MLIR_TypeHandle ty) {
    MLIR_NamedAttr val = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeInteger(e->ctx, v, ty));
    return emit_op1(e, OP_TYPE_ARITH_CONSTANT, ty, NULL, 0, &val, 1);
}

// A zero of the storage type `ty` (integer, float or pointer).
static MLIR_ValueHandle emit_zero_of(E *e, MLIR_TypeHandle ty) {
    if (ty == e->i8 || ty == e->i16) return emit_const_narrow(e, 0, ty);
    if (ty == e->i64) return emit_const_i64(e, 0);
    if (ty == e->f32) return emit_const_f32(e, 0.0);
    if (ty == e->f64) return emit_const_f64(e, 0.0);
    if (ty == e->ptr) return emit_null_ptr(e);
    return emit_const_i32(e, 0);
}

// Emit `llvm.mlir.addressof @<sym>` -> !llvm.ptr or !llvm.ptr<N>.
static MLIR_TypeHandle ptr_for_as(E *e, uint32_t as) {
    if (as == 0) return e->ptr;
    for (size_t i = 0; i < e->n_ptr_as_cache; i++) {
        if (e->ptr_as_cache[i].as == as) return e->ptr_as_cache[i].ptr;
    }
    MLIR_TypeHandle p = MLIR_CreateTypeLLVMPointerInAddressSpace(e->ctx, as);
    if (e->n_ptr_as_cache == e->cap_ptr_as_cache) {
        size_t nc = e->cap_ptr_as_cache ? e->cap_ptr_as_cache * 2 : 4;
        PtrAsCacheEntry *na = arena_new_array(e->arena, PtrAsCacheEntry, nc);
        for (size_t i = 0; i < e->n_ptr_as_cache; i++) na[i] = e->ptr_as_cache[i];
        e->ptr_as_cache = na;
        e->cap_ptr_as_cache = nc;
    }
    e->ptr_as_cache[e->n_ptr_as_cache++] = (PtrAsCacheEntry){ as, p };
    return p;
}

static MLIR_ValueHandle emit_addressof(E *e, string sym, MLIR_TypeHandle ptr_ty) {
    if (!ptr_ty) ptr_ty = e->ptr;
    MLIR_NamedAttr a = MLIR_Named(MLIR_KEY_GLOBAL_NAME, MLIR_CreateAttributeSymbolRef(
        e->ctx, sym));
    return emit_op1(e, OP_TYPE_LLVM_MLIR_ADDRESSOF, ptr_ty, NULL, 0, &a, 1);
}

// Look up (or create) a string-literal global; return its sym_name (O(1) hash table).
static string intern_string(E *e, string bytes) {
    uint32_t h = emit_str_hash(bytes);
    if (e->str_map_cap) {
        size_t mask = e->str_map_cap - 1;
        size_t slot = h & mask;
        while (e->str_map[slot]) {
            size_t idx = e->str_map[slot] - 1;
            if (e->strings[idx].bytes.size == bytes.size &&
                memcmp(e->strings[idx].bytes.str, bytes.str, bytes.size) == 0) {
                return e->strings[idx].sym;
            }
            slot = (slot + 1) & mask;
        }
    }
    if (e->n_strings == e->cap_strings) {
        size_t nc = e->cap_strings ? e->cap_strings * 2 : 16;
        StringEntry *na = arena_new_array(e->arena, StringEntry, nc);
        if (e->n_strings) memcpy(na, e->strings, e->n_strings * sizeof(StringEntry));
        e->strings = na; e->cap_strings = nc;
    }
    if ((e->n_strings + 1) * 2 > e->str_map_cap) {
        size_t ncap = e->str_map_cap ? e->str_map_cap * 2 : 32;
        size_t *nt = arena_new_array(e->arena, size_t, ncap);
        memset(nt, 0, ncap * sizeof(size_t));
        size_t mask = ncap - 1;
        for (size_t i = 0; i < e->n_strings; i++) {
            size_t slot = emit_str_hash(e->strings[i].bytes) & mask;
            while (nt[slot]) slot = (slot + 1) & mask;
            nt[slot] = i + 1;
        }
        e->str_map = nt;
        e->str_map_cap = ncap;
    }
    char tmp[32];
    tmp[0] = '.'; tmp[1] = 's'; tmp[2] = 't'; tmp[3] = 'r'; tmp[4] = '.';
    size_t v = e->n_strings;
    char rev[24]; int rn = 0;
    if (v == 0) rev[rn++] = '0';
    else { while (v) { rev[rn++] = (char)('0' + (v % 10)); v /= 10; } }
    size_t slen = 5 + (size_t)rn;
    char *sbuf = arena_new_array(e->arena, char, slen + 1);
    memcpy(sbuf, tmp, 5);
    for (int k = 0; k < rn; k++) sbuf[5 + k] = rev[rn - 1 - k];
    sbuf[slen] = '\0';
    string sym = { sbuf, slen };
    size_t idx = e->n_strings++;
    e->strings[idx].bytes = bytes;
    e->strings[idx].sym   = sym;
    size_t mask = e->str_map_cap - 1;
    size_t slot = h & mask;
    while (e->str_map[slot]) slot = (slot + 1) & mask;
    e->str_map[slot] = idx + 1;
    return sym;
}

static MLIR_ValueHandle emit_alloca(E *e, MLIR_TypeHandle elem_ty);
static void emit_store_v(E *e, MLIR_ValueHandle val, MLIR_ValueHandle ptr);
static MLIR_ValueHandle emit_gep(E *e, MLIR_ValueHandle base,
        MLIR_TypeHandle source_elem_ty, const int32_t *raw_idx, size_t n_raw,
        MLIR_ValueHandle *dyn_idx, size_t n_dyn);

// Address of a symbol: either the local alloca slot or a freshly-emitted
// addressof for a module-level global.
static MLIR_ValueHandle sym_addr(E *e, Sym *s) {
    if (s->is_global) {
        return emit_addressof(e, s->global_sym, ptr_for_as(e, s->address_space));
    }
    return s->addr;
}

static MLIR_ValueHandle emit_load_v(E *e, MLIR_ValueHandle ptr, MLIR_TypeHandle elem_ty);

static MLIR_ValueHandle sym_load_ptr(E *e, Sym *s) {
    if (s->is_ssa) return s->addr;
    return emit_load_v(e, sym_addr(e, s), e->ptr);
}

static bool is_mutated(E *e, string name);

static void global_register(E *e, Sym *sy) {
    sy->next = e->globals;
    e->globals = sy;
    if ((e->global_map_n + 1) * 2 > e->global_map_cap) {
        size_t ncap = e->global_map_cap ? e->global_map_cap * 2 : 32;
        Sym **nt = arena_new_array(e->arena, Sym *, ncap);
        memset(nt, 0, ncap * sizeof(Sym *));
        size_t mask = ncap - 1;
        for (size_t i = 0; i < e->global_map_cap; i++) {
            Sym *old = e->global_map[i];
            if (!old) continue;
            size_t slot = emit_str_hash(old->name) & mask;
            while (nt[slot]) slot = (slot + 1) & mask;
            nt[slot] = old;
        }
        e->global_map = nt;
        e->global_map_cap = ncap;
    }
    size_t mask = e->global_map_cap - 1;
    size_t slot = emit_str_hash(sy->name) & mask;
    while (e->global_map[slot]) {
        if (str_eq(e->global_map[slot]->name, sy->name)) {
            e->global_map[slot] = sy;
            return;
        }
        slot = (slot + 1) & mask;
    }
    e->global_map[slot] = sy;
    e->global_map_n++;
}

// Look up a name as a global (O(1) hash table). Returns NULL if not found.
static Sym *global_lookup(E *e, string name) {
    if (!e->global_map_cap) return NULL;
    size_t mask = e->global_map_cap - 1;
    size_t slot = emit_str_hash(name) & mask;
    while (e->global_map[slot]) {
        if (str_eq(e->global_map[slot]->name, name)) return e->global_map[slot];
        slot = (slot + 1) & mask;
    }
    return NULL;
}

// Combined lookup: locals/scope chain first, then module globals.
static Sym *lookup(E *e, Scope *sc, string name) {
    Sym *s = scope_lookup(sc, name);
    if (s) return s;
    return global_lookup(e, name);
}

// ----- LLVM-dialect storage primitives -----

static MLIR_AttributeHandle emit_type_attr(E *e, MLIR_TypeHandle ty) {
    size_t slot = (((uintptr_t)ty * 2654435761u) >> 4) & 127u;
    if (e->ty_attr_k[slot] == ty && ty != MLIR_INVALID_HANDLE)
        return e->ty_attr_v[slot];
    MLIR_AttributeHandle a = MLIR_CreateAttributeType(e->ctx, ty);
    e->ty_attr_k[slot] = ty;
    e->ty_attr_v[slot] = a;
    return a;
}

static MLIR_ValueHandle emit_alloca(E *e, MLIR_TypeHandle elem_ty) {
    // Hoist all `llvm.alloca` ops to the function's entry block so loops
    // (where local declarations sit on each iteration) don't grow the stack
    // unboundedly. We share a single `arith.constant 1 : i64` count operand
    // that lives at the top of the entry block.
    MLIR_ValueHandle r = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle rts[1] = { e->ptr };
    MLIR_ValueHandle ops[1] = { e->entry_const_one };
    MLIR_NamedAttr as[1] = { MLIR_Named(MLIR_KEY_ELEM_TYPE, emit_type_attr(e, elem_ty)) };

    bool hoist = e->entry_block && e->entry_block != e->cur_block;
    MLIR_BlockHandle saved = e->cur_block;
    if (hoist) e->cur_block = MLIR_INVALID_HANDLE;
    MLIR_OpHandle aop = emit_op(e, OP_TYPE_LLVM_ALLOCA,
                                rts, 1, &r, 1, ops, 1, as, 1, NULL, 0);
    if (hoist) {
        MLIR_InsertBlockOpAtIndex(e->ctx, e->entry_block, aop,
                                  e->entry_alloca_insert_idx++);
        e->cur_block = saved;
    } else if (e->cur_block == e->entry_block) {
        e->entry_alloca_insert_idx++;
    }
    return r;
}

static void emit_store_v(E *e, MLIR_ValueHandle val, MLIR_ValueHandle ptr) {
    MLIR_ValueHandle ops[2] = { val, ptr };
    emit_op(e, OP_TYPE_LLVM_STORE,
            NULL, 0, NULL, 0, ops, 2, NULL, 0, NULL, 0);
}

static MLIR_ValueHandle emit_load_v(E *e, MLIR_ValueHandle ptr, MLIR_TypeHandle elem_ty) {
    return emit_op1(e, OP_TYPE_LLVM_LOAD, elem_ty, &ptr, 1, NULL, 0);
}

// llvm.getelementptr base, raw_idx, dyn_idx... -> !llvm.ptr.
// raw_idx[k] is either a constant index or kDynamicIndex (= INT32_MIN) for
// each dynamic slot. dyn_idx[] gives the SSA values for the dynamic slots
// in left-to-right order.
static const int32_t LLVM_GEP_DYN = (int32_t)0x80000000;

static MLIR_ValueHandle emit_gep(E *e,
        MLIR_ValueHandle base,
        MLIR_TypeHandle source_elem_ty,
        const int32_t *raw_idx, size_t n_raw,
        MLIR_ValueHandle *dyn_idx, size_t n_dyn) {
    size_t n_ops = 1 + n_dyn;
    MLIR_ValueHandle ops_stk[8];
    MLIR_ValueHandle *ops = (n_ops <= 8) ? ops_stk
        : arena_new_array(e->arena, MLIR_ValueHandle, n_ops);
    ops[0] = base;
    for (size_t i = 0; i < n_dyn; i++) ops[1 + i] = dyn_idx[i];
    MLIR_AttributeHandle idx_attr;
    if (n_raw == 2 && raw_idx[0] == 0) {
        if (!e->caches_init) emit_init_attr_caches(e);
        if (raw_idx[1] >= 0 && raw_idx[1] < 32) {
            idx_attr = e->gep_0k_attrs[raw_idx[1]];
        } else if (raw_idx[1] == LLVM_GEP_DYN) {
            idx_attr = e->gep_0dyn_attr;
        } else {
            idx_attr = MLIR_CreateAttributeDenseI32Array(e->ctx, raw_idx, n_raw);
        }
    } else if (n_raw == 1 && raw_idx[0] == LLVM_GEP_DYN) {
        if (!e->caches_init) emit_init_attr_caches(e);
        idx_attr = e->gep_dyn_attr;
    } else {
        idx_attr = MLIR_CreateAttributeDenseI32Array(e->ctx, raw_idx, n_raw);
    }
    MLIR_NamedAttr as[2];
    as[0] = MLIR_Named(MLIR_KEY_RAW_CONSTANT_INDICES, idx_attr);
    as[1] = MLIR_Named(MLIR_KEY_ELEM_TYPE, emit_type_attr(e, source_elem_ty));
    return emit_op1(e, OP_TYPE_LLVM_GEP, e->ptr, ops, (uint32_t)n_ops, as, 2);
}

static MLIR_ValueHandle emit_select(E *e, MLIR_TypeHandle rt, MLIR_ValueHandle c,
                                    MLIR_ValueHandle t, MLIR_ValueHandle f) {
    MLIR_ValueHandle ops[3] = { c, t, f };
    return emit_op1(e, OP_TYPE_ARITH_SELECT, rt, ops, 3, NULL, 0);
}

// arith.sitofp / arith.uitofp : iN -> f32/f64, at the integer's own
// width (i32 or i64). `is_unsigned` is the source integer's signedness.
static MLIR_ValueHandle emit_itofp(E *e, MLIR_ValueHandle iv, MLIR_TypeHandle ft,
                                   bool is_unsigned) {
    return emit_op1(e, is_unsigned ? OP_TYPE_ARITH_UITOFP : OP_TYPE_ARITH_SITOFP,
                    ft, &iv, 1, NULL, 0);
}

// arith.fptosi / arith.fptoui : f32/f64 -> `it` (i32 or i64) directly.
// `is_unsigned` is the target integer's signedness.
static MLIR_ValueHandle emit_fptoi(E *e, MLIR_ValueHandle fv, MLIR_TypeHandle it,
                                   bool is_unsigned) {
    return emit_op1(e, is_unsigned ? OP_TYPE_ARITH_FPTOUI : OP_TYPE_ARITH_FPTOSI,
                    it, &fv, 1, NULL, 0);
}

static int64_t cmpi_pred_for(BinOp op, bool is_unsigned) {
    switch (op) {
        case OP_EQ: return 0; case OP_NE: return 1;
        case OP_LT: return is_unsigned ? 6 : 2;
        case OP_LE: return is_unsigned ? 7 : 3;
        case OP_GT: return is_unsigned ? 8 : 4;
        case OP_GE: return is_unsigned ? 9 : 5;
        default:    return 0;
    }
}

static int64_t cmpf_pred_for(BinOp op) {
    switch (op) {
        case OP_EQ: return 1; case OP_NE: return 6;
        case OP_LT: return 4; case OP_LE: return 5;
        case OP_GT: return 2; case OP_GE: return 3;
        default:    return 1;
    }
}

// ----- expression / statement emission -----

typedef struct {
    MLIR_ValueHandle val;
    bool is_float;
    bool is_f64;       // when is_float, the value has TY_F64 (else TY_F32)
    bool is_i64;       // val has MLIR type i64 (TY_I64)
    bool is_ptr;       // val is a !llvm.ptr (struct pointer, int*, char*, ...)
    bool is_str;       // val is a !llvm.ptr to char data (string)
    StructDef *sdef;   // when is_ptr, the StructDef the pointer targets (NULL for null)
    MLIR_TypeHandle ptr_elem;  // pointee element type for non-struct pointers
                               // (INVALID otherwise). Used by pointer arithmetic.
    Type *fnptr_ty;    // when val is a function pointer, its TY_FNPTR signature
                       // (return + parameter types). NULL otherwise.
    bool is_void_ptr;  // val is a `void*` — dereference / indexing /
                       // pointer arithmetic are forbidden on it.
    bool is_unsigned;  // integer val came from an `unsigned` C type; selects
                       // unsigned div/mod/shr (and comparisons) at use sites.
} EVal;

// Whether the integer `v` is the constant 0: an arith.constant 0, possibly
// widened or narrowed (`0`, `0L`, `(long)0`, a folded `sizeof(int) - 4`).
static bool value_is_zero_const(E *e, MLIR_ValueHandle v) {
    for (int depth = 0; depth < 4; depth++) {
        MLIR_OpHandle op = MLIR_GetValueDefiningOp(e->ctx, v);
        if (op == MLIR_INVALID_HANDLE) return false;
        MLIR_OpType ty = MLIR_GetOpType(e->ctx, op);
        if (ty == OP_TYPE_ARITH_CONSTANT) {
            MLIR_AttributeHandle a = MLIR_GetOpAttr(e->ctx, op, MLIR_KEY_VALUE);
            return a != MLIR_INVALID_HANDLE &&
                   MLIR_GetAttributeKind(e->ctx, a) == MLIR_ATTR_KIND_INTEGER &&
                   MLIR_GetAttributeInteger(e->ctx, a) == 0;
        }
        if (ty != OP_TYPE_ARITH_EXTSI && ty != OP_TYPE_ARITH_EXTUI &&
            ty != OP_TYPE_ARITH_TRUNCI)
            return false;
        v = MLIR_GetOpOperand(e->ctx, op, 0);
    }
    return false;
}

// The integer `v` used where a pointer is expected without a cast: a call
// argument, an assignment, an initializer, a return value, a `?:` operand, a
// comparison operand or free's argument. A constant 0 is a null pointer
// constant. C requires a diagnostic for any other integer; like older C
// compilers (and gcc -fpermissive), tinyc warns and converts the value:
// widened to i64 by its signedness, then llvm.inttoptr.
static MLIR_ValueHandle int_to_ptr(E *e, EVal v) {
    if (v.is_ptr || MLIR_GetValueType(e->ctx, v.val) == e->ptr) return v.val;
    if (v.is_float) {
        EMIT_ERR(e, "a floating-point value cannot be converted to a pointer");
        return emit_null_ptr(e);
    }
    if (value_is_zero_const(e, v.val)) return emit_null_ptr(e);
    emit_warn(e, str_lit("integer converted to a pointer without a cast"));
    MLIR_ValueHandle iv = v.val;
    if (!v.is_i64)
        iv = v.is_unsigned ? emit_extui_i32_to_i64(e, iv) : emit_extsi_i32_to_i64(e, iv);
    return emit_inttoptr_i64(e, iv);
}

static EVal emit_expr(E *e, Scope *sc, Expr *ex);
static int64_t type_size(E *e, Type t);
static int64_t type_align(E *e, Type t);
static int64_t c_sizeof_expr(E *e, Scope *sc, Expr *ex);

// EVal.ptr_elem for a value of pointer type `t`: the element type that
// pointer arithmetic steps over and that a dereference loads. INVALID for
// struct / void / function pointers and for non-pointers (pointer arithmetic
// then falls back to the struct type, or to i32).
static MLIR_TypeHandle ptr_elem_of_type(E *e, const Type *t) {
    if (t->kind == TY_PTR_CHAR) return e->i8;
    if (t->kind == TY_PTR_PTR) return e->ptr;
    if (t->kind == TY_PTR_I32) {
        return t->ptr_is_i64 ? e->i64
             : t->ptr_is_f32 ? e->f32
             : t->ptr_is_f64 ? e->f64
             : int_mem_type(e, t);
    }
    return MLIR_INVALID_HANDLE;
}

static bool type_kind_is_ptr(TypeKind k) {
    return k == TY_PTR_I32 || k == TY_PTR_CHAR || k == TY_PTR_STRUCT ||
           k == TY_PTR_VOID || k == TY_PTR_PTR || k == TY_FNPTR;
}

// The EVal of a value `val` of pointer type `*t`, tagged with what the type
// says about the pointee: the element type, the struct, the void / char /
// function-pointer kind, and the pointee's signedness (EVal.is_unsigned
// describes the pointee of a pointer). For pointer values whose producer
// does not tag them itself. (Takes the type by pointer and keeps the EVal
// out of the callers' address: in the self-hosted build both are cheaper.)
static EVal ptr_eval_of_type(E *e, MLIR_ValueHandle val, const Type *t) {
    EVal v = (EVal){0};
    v.val = val;
    v.is_ptr = true;
    v.ptr_elem = ptr_elem_of_type(e, t);
    v.is_str = t->kind == TY_PTR_CHAR;
    v.is_void_ptr = t->kind == TY_PTR_VOID;
    v.is_unsigned = t->int_unsigned;
    if (t->kind == TY_PTR_STRUCT) v.sdef = find_struct(e, t->struct_name);
    if (t->kind == TY_FNPTR) {
        Type *fnty = arena_new(e->arena, Type);
        *fnty = *t;
        v.fnptr_ty = fnty;
    }
    return v;
}

// The size of the element a pointer value (EVal.ptr_elem `t`, EVal.sdef
// `sd`) steps over: the stride of its pointer arithmetic (the element type,
// else the struct, else i32).
static int64_t ptr_elem_size(E *e, MLIR_TypeHandle t, StructDef *sd) {
    if (sd && (t == MLIR_INVALID_HANDLE || t == find_struct_type(e, sd))) {
        Type st = (Type){0};
        st.kind = TY_STRUCT;
        st.struct_name = sd->name;
        return type_size(e, st);
    }
    if (t == e->i8) return 1;
    if (t == e->i16) return 2;
    if (t == e->i64 || t == e->f64) return 8;
    if (t == e->ptr) return e->target_wasm32 ? 4 : 8;
    return 4;
}

// Element type that `pp[i][j]` / `(*pp)[i]` steps over for a T** whose
// single-level pointer type T* is `pointee` (char when unknown).
static MLIR_TypeHandle pointee_elem_type(E *e, Type *pointee) {
    if (!pointee) return e->i8;
    TypeKind pk = pointee->kind;
    if (pk == TY_PTR_CHAR || pk == TY_PTR_I32) return ptr_elem_of_type(e, pointee);
    if (pk == TY_PTR_VOID || pk == TY_PTR_STRUCT || pk == TY_FNPTR || pk == TY_PTR_PTR)
        return e->ptr;
    return e->i8;
}

// Compute the single-blob representation tinyc uses to lay a `union` out
// with all members overlapping at offset 0: an array of `*count` elements
// of `*elem` (an integer type whose width equals the union's alignment),
// sized to cover the union's largest member. Used by type_size,
// init_struct_types, emit_struct_zero and emit_struct_copy_path so that
// they all agree on the union layout.
static void union_blob_layout(E *e, StructDef *sd, MLIR_TypeHandle *elem,
                              int64_t *elem_size, int64_t *count);
static Type infer_expr_type(E *e, Scope *sc, Expr *ex);
static void emit_struct_zero(E *e, MLIR_ValueHandle base, MLIR_TypeHandle source_elem,
                             StructDef *sd, int32_t *prefix, size_t n_prefix);
static void emit_expr_for_side_effect(E *e, Scope *sc, Expr *ex);

static MLIR_ValueHandle emit_expr_i32(E *e, Scope *sc, Expr *ex) {
    EVal v = emit_expr(e, sc, ex);
    if (v.is_float) {
        return emit_fptoi(e, v.val, e->i32, false);
    }
    // Narrow `long long` / `uint64_t` initializers down to i32 when the
    // destination is a default-i32 local. Without this, a declaration
    // like `uint8_t byte = v & 0x7f;` (where `v` is i64) emits an
    // `llvm.store i64, ptr` into a 4-byte slot — the store overflows
    // into the adjacent stack slot, clobbering whatever lives next door
    // (commonly the i64 source variable). Use the existing coerce_eval
    // path to insert an `i32.wrap_i64` (a.k.a. trunc i64 → i32).
    if (v.is_i64) {
        return emit_trunci_i64_to_i32(e, v.val);
    }
    return v.val;
}

static MLIR_ValueHandle coerce_scalar(E *e, EVal v, MLIR_TypeHandle want,
                                      bool want_unsigned);
static MLIR_ValueHandle emit_to_bool_i1(E *e, EVal v);
static MLIR_ValueHandle bool_to_i32(E *e, MLIR_ValueHandle b);
static MLIR_ValueHandle emit_bool_i32(E *e, EVal v);
static bool type_is_narrow_int(Type t);

// Convert `v` to a value of C type `ty` stored as the MLIR scalar type
// `want`, as C does on assignment, initialization, argument passing and
// return: a bool becomes 0 / 1, a char / short value is truncated to its
// width and re-extended by its signedness, and the signedness selects
// fptoui over fptosi. `ty` only needs int_unsigned / int_bits / is_bool.
static MLIR_ValueHandle coerce_eval_to(E *e, EVal v, MLIR_TypeHandle want, Type ty) {
    bool want_int = want == e->i8 || want == e->i16 || want == e->i32 || want == e->i64;
    if (want_int && ty.is_bool) {
        EVal b = (EVal){0};
        b.val = emit_bool_i32(e, v);
        return coerce_scalar(e, b, want, false);
    }
    // A float -> char / short conversion goes through a signed i32 (every
    // in-range value fits); wasm's unsigned truncation would trap on a
    // negative out-of-range value.
    bool narrow = type_is_narrow_int(ty);
    MLIR_ValueHandle r = coerce_scalar(e, v, want, ty.int_unsigned && !narrow);
    if (want == e->i32 && !v.is_ptr && narrow)
        r = emit_narrow_i32(e, r, ty.int_bits, ty.int_unsigned);
    return r;
}

// Width / int <-> float part of coerce_eval_to. `want_unsigned` is the
// signedness of the target C type.
static MLIR_ValueHandle coerce_scalar(E *e, EVal v, MLIR_TypeHandle want,
                                      bool want_unsigned) {
    if (want == e->f64) {
        if (v.is_float) {
            if (v.is_f64) return v.val;
            return emit_fpext_f32_to_f64(e, v.val);
        }
        return emit_itofp(e, v.val, e->f64, v.is_unsigned);
    }
    if (want == e->f32) {
        if (v.is_float) {
            if (v.is_f64) return emit_fptrunc_f64_to_f32(e, v.val);
            return v.val;
        }
        return emit_itofp(e, v.val, e->f32, v.is_unsigned);
    }
    if (v.is_float && (want == e->i32 || want == e->i64)) {
        return emit_fptoi(e, v.val, want, want_unsigned);
    }
    bool want_narrow = want == e->i8 || want == e->i16;
    if (v.is_float && want_narrow) {
        return emit_trunci_to(e, emit_fptoi(e, v.val, e->i32, want_unsigned), want);
    }
    if (want == e->i64) {
        if (v.is_i64) return v.val;
        if (v.is_ptr) return v.val;
        // Use unsigned widening when the source carried an `unsigned`
        // C type — otherwise an i32 value with its top bit set (e.g.
        // a wasm32 `size_t` of 2 GiB) would sign-extend to a huge
        // "negative" u64.
        return v.is_unsigned ? emit_extui_i32_to_i64(e, v.val)
                             : emit_extsi_i32_to_i64(e, v.val);
    }
    if (want == e->i32 && v.is_i64) {
        return emit_trunci_i64_to_i32(e, v.val);
    }
    if (want_narrow) {
        // Storing into char / short / bool storage (i8 / i16): truncate
        // the integer value (an i64 one through i32). A constant folds.
        if (!v.is_i64 && !v.is_ptr) {
            if (fact_get(e, v.val) == FACT_CONST) {
                int64_t c = e->fact_c;
                return emit_const_narrow(e, want == e->i8 ? (int64_t)(uint8_t)c
                                                          : (int64_t)(uint16_t)c, want);
            }
        }
        MLIR_ValueHandle iv = v.is_i64 ? emit_trunci_i64_to_i32(e, v.val) : v.val;
        return emit_trunci_to(e, iv, want);
    }
    if (want == e->ptr) return int_to_ptr(e, v);
    return v.val;
}

static MLIR_ValueHandle emit_to_bool_i1(E *e, EVal v) {
    if (v.is_float) {
        MLIR_ValueHandle z = v.is_f64 ? emit_const_f64(e, 0.0) : emit_const_f32(e, 0.0);
        return emit_cmpf(e, /*one*/ 6, v.val, z);
    }
    if (v.is_ptr) {
        MLIR_ValueHandle z = emit_null_ptr(e);
        return emit_icmp_ptr(e, /*ne*/ 1, v.val, z);
    }
    if (v.is_i64) {
        MLIR_ValueHandle z = emit_const_i64(e, 0);
        return emit_cmpi(e, /*ne*/ 1, v.val, z);
    }
    MLIR_ValueHandle z = emit_const_i32(e, 0);
    return emit_cmpi(e, /*ne*/ 1, v.val, z);
}

static MLIR_ValueHandle bool_to_i32(E *e, MLIR_ValueHandle b) {
    MLIR_ValueHandle one  = emit_const_i32(e, 1);
    MLIR_ValueHandle zero = emit_const_i32(e, 0);
    MLIR_ValueHandle r = emit_select(e, e->i32, b, one, zero);
    fact_put(e, r, FACT_RANGE, 1, true, 0);
    return r;
}

// `v` converted to bool as an i32 0 / 1. A constant folds, and a value
// already known to be 0 / 1 is returned as is.
static MLIR_ValueHandle emit_bool_i32(E *e, EVal v) {
    if (!v.is_float && !v.is_ptr && !v.is_i64) {
        int f = fact_get(e, v.val);
        if (f == FACT_CONST) return emit_const_i32(e, e->fact_c != 0);
        if (f == FACT_RANGE && e->fact_unsigned && e->fact_bits == 1) return v.val;
    }
    return bool_to_i32(e, emit_to_bool_i1(e, v));
}

// ----- LVal: a generalized lvalue described as a GEP-and-load/store target.
//
// The lvalue points at a leaf scalar (i32 or f32). It is described as either
// a direct !llvm.ptr to that scalar (n_const_path == 0 && dyn_index INVALID
// — typical for a simple int/float local or a *p result), or as a base
// !llvm.ptr together with a GEP descent (`source_elem` = type pointed to by
// base, `const_path` / `dyn_index` describe the indices). Resolution emits
// a single GEP and then load/store.

typedef struct LValS {
    MLIR_ValueHandle base_ptr;
    MLIR_TypeHandle  source_elem;     // INVALID if no GEP needed
    int32_t         *const_path;
    size_t           n_const_path;
    MLIR_ValueHandle dyn_index;       // INVALID for purely-static GEP
    MLIR_TypeHandle  elem_ty;         // i32 or f32
    bool             is_unsigned;     // backing C type was `unsigned`
    int              narrow_bits;     // 8 / 16 for a char / short / bool type, else 0
    bool             is_bool;         // backing C type is bool
    bool             in_union;        // a union member: the slot may have been
                                      // written through another member
    bool             wide_slot;       // a char / short / bool local in an i32 slot
} LVal;

// A char / short / bool value (int_bits 8 / 16) is promoted to (signed) int
// in expressions (C integer promotion). For a pointer type this describes
// the pointee.
static bool type_is_narrow_int(Type t) {
    return t.int_bits == 8 || t.int_bits == 16;
}

// A value of type `t` is a char / short / bool integer (not a pointer to
// one), so it promotes to int.
static bool type_promotes_to_int(Type t) {
    return t.kind == TY_I32 && type_is_narrow_int(t);
}

// The C type an lvalue stores, as far as conversions into it care:
// signedness, narrow width and bool.
static Type lval_ctype(LVal lv) {
    Type t = (Type){0};
    t.int_unsigned = lv.is_unsigned;
    t.int_bits = lv.narrow_bits;
    t.is_bool = lv.is_bool;
    return t;
}

static MLIR_ValueHandle lval_address(E *e, LVal lv) {
    if (lv.n_const_path == 0 && lv.dyn_index == MLIR_INVALID_HANDLE) {
        return lv.base_ptr;
    }
    MLIR_ValueHandle dyn[1];
    size_t n_dyn = 0;
    if (lv.dyn_index != MLIR_INVALID_HANDLE) {
        dyn[0] = lv.dyn_index;
        n_dyn = 1;
    }
    return emit_gep(e, lv.base_ptr, lv.source_elem,
                    lv.const_path, lv.n_const_path, dyn, n_dyn);
}

static EVal load_lvalue(E *e, LVal lv) {
    EVal r = {0};
    MLIR_ValueHandle p = lval_address(e, lv);
    r.val = emit_load_v(e, p, lv.elem_ty);
    bool narrow = lv.elem_ty == e->i8 || lv.elem_ty == e->i16;
    if (narrow) {
        // Widen char / short / bool storage to int: zero-extend an unsigned
        // type or bool, sign-extend otherwise so negative values stay
        // negative for subsequent arithmetic. A bool slot holds 0 / 1
        // (every store converts to bool), unless a union member wrote it.
        bool is_bool = lv.is_bool && !lv.in_union;
        bool zext = lv.is_unsigned || lv.is_bool;
        r.val = zext ? emit_extui_to_i32(e, r.val) : emit_extsi_to_i32(e, r.val);
        fact_put(e, r.val, FACT_RANGE, is_bool ? 1 : lv.elem_ty == e->i8 ? 8 : 16,
                 zext, 0);
    } else if (lv.elem_ty == e->i32 && lv.narrow_bits) {
        if (lv.wide_slot) {
            // A char / short / bool local in an i32 slot: no pointer reaches
            // it, and every store converts to its type.
            fact_put(e, r.val, FACT_RANGE, lv.is_bool ? 1 : lv.narrow_bits,
                     lv.is_bool || lv.is_unsigned, 0);
        } else {
            // A char / short / bool type read through i32 storage (the
            // pointee of an untyped pointer expression): narrow the value.
            r.val = emit_narrow_i32(e, r.val, lv.narrow_bits, lv.is_unsigned);
        }
    }
    r.is_float = (lv.elem_ty == e->f32 || lv.elem_ty == e->f64);
    r.is_f64 = (lv.elem_ty == e->f64);
    r.is_i64 = (lv.elem_ty == e->i64);
    r.is_ptr = (lv.elem_ty == e->ptr);
    // A narrow unsigned integer promotes to (signed) int. (For a loaded
    // pointer, is_unsigned describes the pointee and is kept.)
    bool promoted = narrow || (lv.elem_ty == e->i32 && lv.narrow_bits);
    r.is_unsigned = lv.is_unsigned && !promoted;
    r.ptr_elem = MLIR_INVALID_HANDLE;
    return r;
}

// Store `v`, already converted to the lvalue's storage type lv.elem_ty.
static void store_lvalue(E *e, LVal lv, MLIR_ValueHandle v) {
    MLIR_ValueHandle p = lval_address(e, lv);
    emit_store_v(e, v, p);
}

// ----- struct-context walker: produces (base_ptr, source_elem, sd, path)
// for the struct lvalue being descended into (not yet a leaf).

typedef struct {
    bool             ok;
    MLIR_ValueHandle base_ptr;
    MLIR_TypeHandle  source_elem;
    StructDef       *sd;
    int32_t         *const_path;
    size_t           n_const_path;
    size_t           cap_const_path;
    MLIR_ValueHandle dyn_index;
} SCtx;

static void sctx_push(E *e, SCtx *c, int32_t v) {
    if (c->n_const_path == c->cap_const_path) {
        size_t cap = c->cap_const_path ? c->cap_const_path * 2 : 4;
        int32_t *np = arena_new_array(e->arena, int32_t, cap);
        for (size_t i = 0; i < c->n_const_path; i++) np[i] = c->const_path[i];
        c->const_path = np;
        c->cap_const_path = cap;
    }
    c->const_path[c->n_const_path++] = v;
}

static SCtx walk_struct_lhs(E *e, Scope *sc, Expr *ex) {
    e->cur_line = ex->line;
    SCtx r = {0};
    r.dyn_index = MLIR_INVALID_HANDLE;
    if (ex->kind == EX_VAR) {
        Sym *s = lookup(e, sc, ex->name);
        if (!s) { EMIT_ERR(e, "undefined variable: {}", ex->name); return r; }
        if (s->type.kind == TY_STRUCT) {
            r.base_ptr = sym_addr(e, s);
            r.source_elem = find_struct_type(e, s->sdef);
            r.sd = s->sdef;
            sctx_push(e, &r, 0);
            r.ok = (s->sdef != NULL);
            return r;
        }
        if (s->type.kind == TY_PTR_STRUCT) {
            // Load the inner ptr first.
            r.base_ptr = sym_load_ptr(e, s);
            r.source_elem = find_struct_type(e, s->sdef);
            r.sd = s->sdef;
            sctx_push(e, &r, 0);
            r.ok = (s->sdef != NULL);
            return r;
        }
        EMIT_ERR(e, "field/index access on non-struct variable {}", ex->name);
        return r;
    }
    if (ex->kind == EX_DEREF) {
        if (ex->lhs->kind == EX_VAR) {
            Sym *s = lookup(e, sc, ex->lhs->name);
            if (!s || s->type.kind != TY_PTR_STRUCT || !s->sdef) {
                EMIT_ERR(e, "-> requires a struct pointer");
                return r;
            }
            r.base_ptr = sym_load_ptr(e, s);
            r.source_elem = find_struct_type(e, s->sdef);
            r.sd = s->sdef;
            sctx_push(e, &r, 0);
            r.ok = true;
            return r;
        }
        // General case: evaluate the inner expression as a struct pointer.
        // Supports mid-chain walks like p->left->right and (*f(...)).x.
        EVal v = emit_expr(e, sc, ex->lhs);
        if (!v.is_ptr || !v.sdef) {
            EMIT_ERR(e, "-> requires a struct pointer");
            return r;
        }
        r.base_ptr = v.val;
        r.sd = v.sdef;
        r.source_elem = find_struct_type(e, v.sdef);
        sctx_push(e, &r, 0);
        r.ok = true;
        return r;
    }
    if (ex->kind == EX_INDEX && ex->lhs->kind == EX_VAR) {
        Sym *s = lookup(e, sc, ex->lhs->name);
        if (s && s->type.kind == TY_PTR_STRUCT && s->sdef) {
            // `p[i].f` where p is a struct pointer: load p, then GEP by index.
            MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
            r.base_ptr = sym_load_ptr(e, s);
            r.source_elem = find_struct_type(e, s->sdef);
            r.sd = s->sdef;
            sctx_push(e, &r, LLVM_GEP_DYN);
            r.dyn_index = idx_i32;
            r.ok = true;
            return r;
        }
        if (!s || s->type.kind != TY_ARRAY_STRUCT || !s->sdef) {
            EMIT_ERR(e, "arr[i].f requires an array of struct");
            return r;
        }
        MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
        r.base_ptr = sym_addr(e, s);
        // Source elem is the !llvm.array<N x !llvm.struct<...>>.
        MLIR_TypeHandle struct_ty = find_struct_type(e, s->sdef);
        r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, struct_ty, (uint64_t)s->type.array_len);
        r.sd = s->sdef;
        sctx_push(e, &r, 0);                // step into the alloca's one element
        sctx_push(e, &r, LLVM_GEP_DYN);     // dynamic array index
        r.dyn_index = idx_i32;
        r.ok = true;
        return r;
    }
    if (ex->kind == EX_INDEX) {
        // Generic `expr[i].f` where expr evaluates to a struct pointer
        // (e.g. `ht->buckets[i].f`). Evaluate the LHS as an EVal, then
        // GEP into the struct array.
        EVal v = emit_expr(e, sc, ex->lhs);
        if (v.is_ptr && v.sdef) {
            MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
            r.base_ptr = v.val;
            r.source_elem = find_struct_type(e, v.sdef);
            r.sd = v.sdef;
            sctx_push(e, &r, LLVM_GEP_DYN);
            r.dyn_index = idx_i32;
            r.ok = true;
            return r;
        }
        EMIT_ERR(e, "arr[i].f requires an array of struct or struct-pointer base");
        return r;
    }
    if (ex->kind == EX_FIELD) {
        SCtx parent = walk_struct_lhs(e, sc, ex->lhs);
        if (!parent.ok) return r;
        int idx = struct_field_index(parent.sd, ex->name);
        if (idx < 0) {
            EMIT_ERR(e, "unknown struct field {}", ex->name);
            return r;
        }
        Type ft = parent.sd->fields.data[idx].type;
        if (ft.kind != TY_STRUCT) {
            EMIT_ERR(e, "cannot chain field access through scalar field {}", ex->name);
            return r;
        }
        r = parent;
        if (parent.sd->is_union) {
            // Crossing a union boundary into a nested struct member.
            // Emit a partial GEP to offset 0 of the union, then start
            // a fresh GEP rooted at the nested struct's type. This is
            // necessary because tinyc lays unions out as sequential
            // structs, so subsequent indices into the nested struct
            // would otherwise descend into the wrong union member's
            // LLVM type.
            sctx_push(e, &r, 0);
            // The path so far may hold a dynamic index (`gu[k].s.x`).
            MLIR_ValueHandle udyn[1] = {r.dyn_index};
            MLIR_ValueHandle base = emit_gep(e, r.base_ptr, r.source_elem,
                                              r.const_path, r.n_const_path,
                                              udyn, r.dyn_index != MLIR_INVALID_HANDLE ? 1 : 0);
            r.base_ptr = base;
            r.source_elem = find_struct_type(e, find_struct(e, ft.struct_name));
            r.const_path = NULL;
            r.n_const_path = 0;
            r.cap_const_path = 0;
            r.dyn_index = MLIR_INVALID_HANDLE;
            sctx_push(e, &r, 0);
        } else {
            sctx_push(e, &r, idx);
        }
        r.sd = find_struct(e, ft.struct_name);
        r.ok = (r.sd != NULL);
        return r;
    }
    // Fallback: a struct-returning call result, `f(...).field`. The
    // call yields an EVal with is_ptr=true and sdef set; we treat that
    // !llvm.ptr as the struct base.
    if (ex->kind == EX_CALL) {
        EVal v = emit_expr(e, sc, ex);
        if (v.is_ptr && v.sdef) {
            r.base_ptr = v.val;
            r.sd = v.sdef;
            r.source_elem = find_struct_type(e, v.sdef);
            sctx_push(e, &r, 0);
            r.ok = true;
            return r;
        }
    }
    EMIT_ERR(e, "unsupported lvalue base in field access");
    return r;
}

// Type of the struct field an EX_FIELD expression names, determined from the
// declared types alone, i.e. WITHOUT emitting any code. Returns false when it
// cannot be determined this way (the caller then falls back to walking the
// struct base with walk_struct_lhs).
//
// This exists because walk_struct_lhs is not a pure query: it emits the code
// that computes the struct base, including any side effect in it (the `--sp`
// of `stack[--sp].bid`, the call of `f().x`, ...). Callers that only need to
// know the field's type must therefore not use it, or the base expression is
// evaluated once more per query.
static bool field_type_noemit(E *e, Scope *sc, Expr *fe, Type *out) {
    Type st = infer_expr_type(e, sc, fe->lhs);
    if (st.kind != TY_STRUCT && st.kind != TY_PTR_STRUCT) return false;
    if (st.struct_name.size == 0) return false;
    StructDef *sd = find_struct(e, st.struct_name);
    if (!sd) return false;
    int idx = struct_field_index(sd, fe->name);
    if (idx < 0) return false;
    *out = sd->fields.data[idx].type;
    return true;
}

// False when emit_lvalue's struct-field indexing block does not handle
// `<base>.f[i]` (`twice`: `<base>.f[i][j]`), judged from the field's type
// without emitting the base: only an array field, or a pointer field
// indexed once, is handled there. (A separate function keeps the Type
// local out of emit_lvalue.)
static bool field_index_handled(E *e, Scope *sc, Expr *fe, bool twice) {
    Type ft = (Type){0};
    if (!field_type_noemit(e, sc, fe, &ft)) return true;
    TypeKind k = ft.kind;
    if (k == TY_ARRAY_I32 || k == TY_ARRAY_F32 || k == TY_ARRAY_PTR_STRUCT ||
        k == TY_ARRAY_PTR_CHAR) return true;
    return !twice && (k == TY_PTR_I32 || k == TY_PTR_CHAR || k == TY_PTR_VOID ||
                      k == TY_PTR_PTR);
}

// A new expression node `kind` with operands `lhs` / `rhs`, for lowering
// one form to another (e.g. `a[i]` to `*(a + i)`).
static Expr *synth_expr(E *e, ExprKind kind, Expr *lhs, Expr *rhs, int line) {
    Expr *x = arena_new(e->arena, Expr);
    *x = (Expr){0};
    x->kind = kind;
    x->lhs = lhs;
    x->rhs = rhs;
    x->line = line;
    return x;
}

// Record in `r` whether the element it addresses is a char / short / bool
// (and its signedness), from the element type `t`, or from the pointer type
// `t` that points at it (NULL: unknown).
static void lval_set_ctype(LVal *r, Type *t) {
    if (!t) return;
    r->is_unsigned = t->int_unsigned;
    r->narrow_bits = type_is_narrow_int(*t) ? t->int_bits : 0;
    r->is_bool = t->is_bool;
}

static LVal emit_lvalue(E *e, Scope *sc, Expr *ex) {
    if (ex == e->compound_lv_expr) return *e->compound_lv;
    e->cur_line = ex->line;
    LVal r = {0};
    r.source_elem = MLIR_INVALID_HANDLE;
    r.dyn_index = MLIR_INVALID_HANDLE;
    r.elem_ty = e->i32;
    switch (ex->kind) {
        case EX_VAR: {
            Sym *s = lookup(e, sc, ex->name);
            if (!s) {
                EMIT_ERR(e, "undefined variable: {}", ex->name);
                return r;
            }
            r.base_ptr = sym_addr(e, s);
            r.elem_ty = s->wide_slot ? e->i32 : scalar_mem_type(e, &s->type);
            r.wide_slot = s->wide_slot;
            r.is_unsigned = s->type.int_unsigned;
            r.narrow_bits = type_is_narrow_int(s->type) ? s->type.int_bits : 0;
            r.is_bool = s->type.is_bool;
            return r;
        }
        case EX_INDEX: {
            // Struct field array: s.f[i] / p->f[i] (1D) or s.f[i][j] (2D),
            // including chained walks like arr[k].f[i] (parent.dyn_index
            // is non-INVALID then). Emit a single GEP that descends from
            // the parent struct's source_elem all the way to the scalar
            // (or pointer) leaf.
            Expr *field_expr = NULL;
            Expr *idx_a = NULL, *idx_b = NULL;
            if (ex->lhs->kind == EX_FIELD) {
                field_expr = ex->lhs;
                idx_a = ex->rhs;
            } else if (ex->lhs->kind == EX_INDEX &&
                       ex->lhs->lhs->kind == EX_FIELD) {
                field_expr = ex->lhs->lhs;
                idx_a = ex->lhs->rhs;
                idx_b = ex->rhs;
            }
            // Walking the struct base emits it, so first check (without
            // emitting) that this block handles the field: an array field,
            // or a pointer field indexed once. `p->pp[i][j]` on a `T **`
            // field is left to the paths below, which must not see the base
            // evaluated already.
            if (field_expr && !field_index_handled(e, sc, field_expr, idx_b != NULL))
                field_expr = NULL;
            if (field_expr) {
                SCtx parent = walk_struct_lhs(e, sc, field_expr->lhs);
                if (parent.ok) {
                    int fidx = struct_field_index(parent.sd, field_expr->name);
                    if (fidx < 0) {
                        EMIT_ERR(e, "unknown struct field {}",
                                field_expr->name);
                        return r;
                    }
                    Type ft = parent.sd->fields.data[fidx].type;
                    bool is_arr_i32 = (ft.kind == TY_ARRAY_I32);
                    bool is_arr_f32 = (ft.kind == TY_ARRAY_F32);
                    bool is_arr_pst = (ft.kind == TY_ARRAY_PTR_STRUCT);
                    bool is_arr_pch = (ft.kind == TY_ARRAY_PTR_CHAR);
                    if (is_arr_i32 || is_arr_f32 || is_arr_pst || is_arr_pch) {
                        bool is_2d = (ft.array_len2 != 0);
                        if (is_2d && !idx_b) {
                            EMIT_ERR(e, "2D field array requires both indices, e.g. s.m[i][j]");
                            return r;
                        }
                        if (!is_2d && idx_b) {
                            EMIT_ERR(e, "1D field array indexed twice");
                            return r;
                        }
                        if ((is_arr_pst || is_arr_pch) && is_2d) {
                            EMIT_ERR(e, "2D pointer-array fields are not supported");
                            return r;
                        }
                        MLIR_TypeHandle elem = array_elem_type(e, &ft);
                        MLIR_ValueHandle iv = emit_expr_i32(e, sc, idx_a);
                        MLIR_ValueHandle jv = is_2d ? emit_expr_i32(e, sc, idx_b)
                                                   : MLIR_INVALID_HANDLE;
                        if (parent.sd->is_union) {
                            // Every union member is at offset 0 of the
                            // union's blob: index the member's own array
                            // type from the union's address.
                            sctx_push(e, &parent, 0);
                            MLIR_ValueHandle pd[1];
                            size_t n_pd = 0;
                            if (parent.dyn_index != MLIR_INVALID_HANDLE) pd[n_pd++] = parent.dyn_index;
                            MLIR_ValueHandle ubase = emit_gep(e, parent.base_ptr,
                                parent.source_elem, parent.const_path, parent.n_const_path,
                                pd, n_pd);
                            int32_t upath[3] = {0, LLVM_GEP_DYN, LLVM_GEP_DYN};
                            MLIR_ValueHandle udyn[2] = {iv, jv};
                            r.base_ptr = emit_gep(e, ubase, array_mlir_type(e, &ft),
                                                  upath, is_2d ? 3 : 2, udyn, is_2d ? 2 : 1);
                            r.elem_ty = elem;
                            lval_set_ctype(&r, &ft);
                            r.in_union = true;
                            return r;
                        }
                        size_t parent_n = parent.n_const_path;
                        size_t extra = 1 + (is_2d ? 2 : 1);
                        size_t total = parent_n + extra;
                        int32_t *path = arena_new_array(e->arena, int32_t, total);
                        for (size_t k = 0; k < parent_n; k++) path[k] = parent.const_path[k];
                        path[parent_n] = (int32_t)fidx;
                        path[parent_n + 1] = LLVM_GEP_DYN;
                        if (is_2d) path[parent_n + 2] = LLVM_GEP_DYN;
                        size_t n_dyn = (parent.dyn_index != MLIR_INVALID_HANDLE ? 1 : 0)
                                     + 1 + (is_2d ? 1 : 0);
                        MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, n_dyn);
                        size_t di = 0;
                        if (parent.dyn_index != MLIR_INVALID_HANDLE) dyn[di++] = parent.dyn_index;
                        dyn[di++] = iv;
                        if (is_2d) dyn[di++] = jv;
                        MLIR_ValueHandle gep = emit_gep(e, parent.base_ptr,
                            parent.source_elem, path, total, dyn, n_dyn);
                        r.base_ptr = gep;
                        r.elem_ty = elem;
                        r.is_unsigned = ft.int_unsigned;
                        r.narrow_bits = type_is_narrow_int(ft) ? ft.int_bits : 0;
                        r.is_bool = ft.is_bool;
                        return r;
                    }
                    // Pointer field indexed: `s.f[i]` / `p->f[i]` where
                    // `f` is `T *` (TY_PTR_I32 / TY_PTR_CHAR / TY_PTR_VOID
                    // / TY_PTR_PTR). Load the pointer from the field, then
                    // GEP it with the (single) index.
                    bool is_ptr_field = (ft.kind == TY_PTR_I32 ||
                                         ft.kind == TY_PTR_CHAR ||
                                         ft.kind == TY_PTR_VOID ||
                                         ft.kind == TY_PTR_PTR);
                    if (is_ptr_field && idx_a && !idx_b) {
                        size_t parent_n = parent.n_const_path;
                        size_t total = parent_n + 1;
                        int32_t *fpath = arena_new_array(e->arena, int32_t, total);
                        for (size_t k = 0; k < parent_n; k++) fpath[k] = parent.const_path[k];
                        fpath[parent_n] = (int32_t)fidx;
                        size_t n_dyn = (parent.dyn_index != MLIR_INVALID_HANDLE ? 1 : 0);
                        MLIR_ValueHandle *fdyn = arena_new_array(e->arena, MLIR_ValueHandle,
                                                                 n_dyn ? n_dyn : 1);
                        if (n_dyn) fdyn[0] = parent.dyn_index;
                        MLIR_ValueHandle field_addr = emit_gep(e, parent.base_ptr,
                            parent.source_elem, fpath, total, fdyn, n_dyn);
                        MLIR_ValueHandle base = emit_load_v(e, field_addr, e->ptr);
                        MLIR_TypeHandle elem = ptr_elem_of_type(e, &ft);
                        if (elem == MLIR_INVALID_HANDLE) elem = e->i32;
                        MLIR_ValueHandle iv = emit_expr_i32(e, sc, idx_a);
                        int32_t *gpath = arena_new_array(e->arena, int32_t, 1);
                        gpath[0] = LLVM_GEP_DYN;
                        MLIR_ValueHandle *gdyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                        gdyn[0] = iv;
                        r.base_ptr = emit_gep(e, base, elem, gpath, 1, gdyn, 1);
                        r.elem_ty = elem;
                        r.is_unsigned = ft.int_unsigned;
                        r.narrow_bits = type_is_narrow_int(ft) ? ft.int_bits : 0;
                        r.is_bool = ft.is_bool;
                        return r;
                    }
                }
            }
            // Array-of-char-pointers chained indexing:
            //   `const char *T[N]; ... T[i][j]`.
            // Step 1: GEP into the local array at index i to get the slot
            //         that holds the inner char*.
            // Step 2: load that slot to obtain the inner char*.
            // Step 3: GEP char* + j (stride 1) -> ptr-to-char.
            if (ex->lhs->kind == EX_INDEX &&
                ex->lhs->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->lhs->name);
                if (s && (s->type.kind == TY_ARRAY_PTR_CHAR ||
                          s->type.kind == TY_ARRAY_PTR_STRUCT)) {
                    MLIR_ValueHandle i_v = emit_expr_i32(e, sc, ex->lhs->rhs);
                    MLIR_ValueHandle j_v = emit_expr_i32(e, sc, ex->rhs);
                    MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                        e->ctx, e->ptr, (uint64_t)s->type.array_len);
                    int32_t *path1 = arena_new_array(e->arena, int32_t, 2);
                    path1[0] = 0; path1[1] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn1 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn1[0] = i_v;
                    MLIR_ValueHandle slot = emit_gep(e, sym_addr(e, s),
                        arr_ty, path1, 2, dyn1, 1);
                    MLIR_ValueHandle inner = emit_load_v(e, slot, e->ptr);
                    // The element pointer type (`pointee`, when the
                    // declaration kept it) gives the stride and the load.
                    MLIR_TypeHandle elem = s->type.pointee
                        ? pointee_elem_type(e, s->type.pointee)
                        : (s->type.kind == TY_ARRAY_PTR_CHAR) ? e->i8 : e->ptr;
                    int32_t *path2 = arena_new_array(e->arena, int32_t, 1);
                    path2[0] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn2 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn2[0] = j_v;
                    r.base_ptr = emit_gep(e, inner, elem, path2, 1, dyn2, 1);
                    r.elem_ty = elem;
                    lval_set_ctype(&r, s->type.pointee);
                    return r;
                }
            }
            // Pointer-to-pointer chained indexing: pp[i][j] where pp is T**.
            // Step 1: GEP pp+i (stride sizeof(ptr)) -> ptr-to-ptr.
            // Step 2: load that to obtain the inner T*.
            // Step 3: GEP T*+j (stride sizeof(T)) -> ptr-to-T.
            if (ex->lhs->kind == EX_INDEX &&
                ex->lhs->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->lhs->name);
                if (s && s->type.kind == TY_PTR_PTR) {
                    MLIR_ValueHandle i_v = emit_expr_i32(e, sc, ex->lhs->rhs);
                    MLIR_ValueHandle j_v = emit_expr_i32(e, sc, ex->rhs);
                    // pp[i]: GEP at base of pp (which holds T**), stride i.
                    MLIR_ValueHandle base_pp = sym_load_ptr(e, s);
                    int32_t *path1 = arena_new_array(e->arena, int32_t, 1);
                    path1[0] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn1 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn1[0] = i_v;
                    MLIR_ValueHandle slot = emit_gep(e, base_pp, e->ptr, path1, 1, dyn1, 1);
                    // Load T* from that slot.
                    MLIR_ValueHandle inner = emit_load_v(e, slot, e->ptr);
                    // Decide element type of inner (T).
                    MLIR_TypeHandle elem = pointee_elem_type(e, s->type.pointee);
                    int32_t *path2 = arena_new_array(e->arena, int32_t, 1);
                    path2[0] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn2 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn2[0] = j_v;
                    r.base_ptr = emit_gep(e, inner, elem, path2, 1, dyn2, 1);
                    r.elem_ty = elem;
                    lval_set_ctype(&r, s->type.pointee);
                    return r;
                }
            }
            // Multi-dim array: m[i][j] for `int m[N1][N2];`.
            if (ex->lhs->kind == EX_INDEX &&
                ex->lhs->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->lhs->name);
                if (s && s->type.kind == TY_ARRAY_I32 && s->type.array_len2 != 0) {
                    MLIR_ValueHandle i_v = emit_expr_i32(e, sc, ex->lhs->rhs);
                    MLIR_ValueHandle j_v = emit_expr_i32(e, sc, ex->rhs);
                    MLIR_TypeHandle outer_arr = array_mlir_type(e, &s->type);
                    int32_t *path = arena_new_array(e->arena, int32_t, 3);
                    path[0] = 0; path[1] = LLVM_GEP_DYN; path[2] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 2);
                    dyn[0] = i_v; dyn[1] = j_v;
                    MLIR_ValueHandle p = emit_gep(e, sym_addr(e, s), outer_arr,
                                                  path, 3, dyn, 2);
                    r.base_ptr = p;
                    r.elem_ty = array_elem_type(e, &s->type);
                    lval_set_ctype(&r, &s->type);
                    return r;
                }
            }
            if (ex->lhs->kind != EX_VAR) {
                // <struct>.field[s][i] / <ptr>->field[s][i] where field is T**.
                if (ex->lhs->kind == EX_INDEX &&
                    ex->lhs->lhs->kind == EX_FIELD) {
                    Expr *fld = ex->lhs->lhs;
                    Expr *vex = fld->lhs;
                    bool is_arrow = (vex->kind == EX_DEREF && vex->lhs->kind == EX_VAR);
                    bool is_dot = (vex->kind == EX_VAR);
                    if (is_arrow || is_dot) {
                        Expr *vv = is_arrow ? vex->lhs : vex;
                        Sym *vs = lookup(e, sc, vv->name);
                        StructDef *sd = NULL;
                        MLIR_ValueHandle parent_base = MLIR_INVALID_HANDLE;
                        if (vs) {
                            if (is_arrow && vs->type.kind == TY_PTR_STRUCT) {
                                sd = find_struct(e, vs->type.struct_name);
                                parent_base = sym_load_ptr(e, vs);
                            } else if (is_dot && vs->type.kind == TY_STRUCT) {
                                sd = find_struct(e, vs->type.struct_name);
                                parent_base = sym_addr(e, vs);
                            }
                        }
                        if (sd) {
                            size_t fidx = 0; bool found = false;
                            Type fldt = (Type){0};
                            for (size_t k = 0; k < sd->fields.size; k++) {
                                if (str_eq(sd->fields.data[k].name, fld->name)) {
                                    fidx = k; fldt = sd->fields.data[k].type; found = true; break;
                                }
                            }
                            if (found && fldt.kind == TY_PTR_PTR) {
                                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                                int32_t *fpath = arena_new_array(e->arena, int32_t, 2);
                                fpath[0] = 0; fpath[1] = (int32_t)fidx;
                                MLIR_ValueHandle field_addr = emit_gep(e, parent_base,
                                    st_ty, fpath, 2, NULL, 0);
                                MLIR_ValueHandle pp = emit_load_v(e, field_addr, e->ptr);
                                MLIR_ValueHandle sv = emit_expr_i32(e, sc, ex->lhs->rhs);
                                int32_t *p1 = arena_new_array(e->arena, int32_t, 1);
                                p1[0] = LLVM_GEP_DYN;
                                MLIR_ValueHandle *d1 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                                d1[0] = sv;
                                MLIR_ValueHandle slot = emit_gep(e, pp, e->ptr, p1, 1, d1, 1);
                                MLIR_ValueHandle inner = emit_load_v(e, slot, e->ptr);
                                MLIR_TypeHandle elem = pointee_elem_type(e, fldt.pointee);
                                MLIR_ValueHandle iv = emit_expr_i32(e, sc, ex->rhs);
                                int32_t *p2 = arena_new_array(e->arena, int32_t, 1);
                                p2[0] = LLVM_GEP_DYN;
                                MLIR_ValueHandle *d2 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                                d2[0] = iv;
                                r.base_ptr = emit_gep(e, inner, elem, p2, 1, d2, 1);
                                r.elem_ty = elem;
                                lval_set_ctype(&r, fldt.pointee);
                                return r;
                            }
                        }
                    }
                }
                // (*p)[i] where p is T**: load p (get T**), load again (get
                // T*), then GEP by i with stride sizeof(T).
                if (ex->lhs->kind == EX_DEREF &&
                    ex->lhs->lhs->kind == EX_VAR) {
                    Sym *s = lookup(e, sc, ex->lhs->lhs->name);
                    if (s && s->type.kind == TY_PTR_PTR) {
                        MLIR_ValueHandle iv = emit_expr_i32(e, sc, ex->rhs);
                        MLIR_ValueHandle outer = sym_load_ptr(e, s);
                        MLIR_ValueHandle inner = emit_load_v(e, outer, e->ptr);
                        MLIR_TypeHandle elem = pointee_elem_type(e, s->type.pointee);
                        int32_t *gpath = arena_new_array(e->arena, int32_t, 1);
                        gpath[0] = LLVM_GEP_DYN;
                        MLIR_ValueHandle *gdyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                        gdyn[0] = iv;
                        r.base_ptr = emit_gep(e, inner, elem, gpath, 1, gdyn, 1);
                        r.elem_ty = elem;
                        lval_set_ctype(&r, s->type.pointee);
                        return r;
                    }
                }
                // Any other base (a call result, a cast, `(*pp++)`, a string
                // literal, ...): `a[i]` is `*(a + i)`.
                Expr *sum = synth_expr(e, EX_BIN, ex->lhs, ex->rhs, ex->line);
                sum->op = OP_ADD;
                return emit_lvalue(e, sc, synth_expr(e, EX_DEREF, sum, NULL, ex->line));
            }
            Sym *s = lookup(e, sc, ex->lhs->name);
            if (!s) {
                EMIT_ERR(e, "undefined array: {}", ex->lhs->name);
                return r;
            }
            if (s->type.kind == TY_ARRAY_STRUCT) {
                // For lvalue context that takes the *address* of arr[i]
                // (e.g. `&arr[i]`), GEP into the local array. Field access
                // (`arr[i].f`) goes through other paths that build the GEP
                // themselves; we still keep the original guard for that.
                StructDef *sd = find_struct(e, s->type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct in array");
                    return r;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                r.base_ptr = sym_addr(e, s);
                r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, st_ty,
                    (uint64_t)s->type.array_len);
                int32_t *path = arena_new_array(e->arena, int32_t, 2);
                path[0] = 0; path[1] = LLVM_GEP_DYN;
                r.const_path = path; r.n_const_path = 2;
                r.dyn_index = idx_i32;
                r.elem_ty = st_ty;
                return r;
            }
            if (s->type.kind == TY_PTR_STRUCT) {
                // p[i] for a struct pointer parameter / local: load the
                // pointer then GEP by the struct stride.
                StructDef *sd = find_struct(e, s->type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct ptr");
                    return r;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                MLIR_ValueHandle base = sym_load_ptr(e, s);
                int32_t path[1] = { LLVM_GEP_DYN };
                r.base_ptr = emit_gep(e, base, st_ty, path, 1, &idx_i32, 1);
                r.elem_ty = st_ty;
                return r;
            }
            // Pointer indexing: p[i] for int* / char* — GEP %p[%i].
            if (s->type.kind == TY_PTR_I32 || s->type.kind == TY_PTR_CHAR) {
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                MLIR_ValueHandle base = sym_load_ptr(e, s);
                MLIR_TypeHandle elem = ptr_elem_of_type(e, &s->type);
                int32_t path[1] = { LLVM_GEP_DYN };
                r.base_ptr = emit_gep(e, base, elem, path, 1, &idx_i32, 1);
                r.elem_ty = elem;
                r.is_unsigned = s->type.int_unsigned;
                r.narrow_bits = type_is_narrow_int(s->type) ? s->type.int_bits : 0;
                r.is_bool = s->type.is_bool;
                return r;
            }
            // Pointer-to-pointer indexing: pp[i] for T** (e.g. char **argv).
            // Each element is a !llvm.ptr; GEP with stride 8 and load yields
            // a single-level T*.
            if (s->type.kind == TY_PTR_PTR) {
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                MLIR_ValueHandle base = sym_load_ptr(e, s);
                int32_t path[1] = { LLVM_GEP_DYN };
                r.base_ptr = emit_gep(e, base, e->ptr, path, 1, &idx_i32, 1);
                r.elem_ty = e->ptr;
                return r;
            }
            if (s->type.kind == TY_ARRAY_PTR_CHAR ||
                s->type.kind == TY_ARRAY_PTR_STRUCT) {
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                r.base_ptr = sym_addr(e, s);
                r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, e->ptr, (uint64_t)s->type.array_len);
                int32_t *path = arena_new_array(e->arena, int32_t, 2);
                path[0] = 0; path[1] = LLVM_GEP_DYN;
                r.const_path = path; r.n_const_path = 2;
                r.dyn_index = idx_i32;
                r.elem_ty = e->ptr;
                return r;
            }
            if (s->type.kind != TY_ARRAY_I32) {
                EMIT_ERR(e, "indexing of non-array variable");
                return r;
            }
            if (s->type.array_len2 != 0) {
                EMIT_ERR(e, "2D array requires both indices, e.g. m[i][j]");
                return r;
            }
            MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
            MLIR_TypeHandle aelem = array_elem_type(e, &s->type);
            r.base_ptr = sym_addr(e, s);
            r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, aelem, (uint64_t)s->type.array_len);
            int32_t *path = arena_new_array(e->arena, int32_t, 2);
            path[0] = 0; path[1] = LLVM_GEP_DYN;
            r.const_path = path; r.n_const_path = 2;
            r.dyn_index = idx_i32;
            r.elem_ty = aelem;
            r.is_unsigned = s->type.int_unsigned;
            r.narrow_bits = type_is_narrow_int(s->type) ? s->type.int_bits : 0;
            r.is_bool = s->type.is_bool;
            return r;
        }
        case EX_DEREF: {
            // `*arr` of an array of pointers / ints decays to `arr[0]`:
            // the general path below loads it with the element type.
            Sym *as = ex->lhs->kind == EX_VAR ? lookup(e, sc, ex->lhs->name) : NULL;
            bool lhs_is_array = as && (as->type.kind == TY_ARRAY_PTR_CHAR ||
                                       as->type.kind == TY_ARRAY_PTR_STRUCT ||
                                       as->type.kind == TY_ARRAY_I32);
            if (ex->lhs->kind == EX_VAR && !lhs_is_array) {
                Sym *s = lookup(e, sc, ex->lhs->name);
                if (!s || (s->type.kind != TY_PTR_I32 &&
                           s->type.kind != TY_PTR_CHAR &&
                           s->type.kind != TY_PTR_PTR)) {
                    EMIT_ERR(e, "dereference of non-pointer");
                    return r;
                }
                // Load the inner ptr from p's slot.
                r.base_ptr = sym_load_ptr(e, s);
                r.elem_ty = ptr_elem_of_type(e, &s->type);
                r.is_unsigned = s->type.int_unsigned;
                r.narrow_bits = type_is_narrow_int(s->type) ? s->type.int_bits : 0;
                r.is_bool = s->type.is_bool;
                return r;
            }
            // General `*<expr>` form (e.g. *(p+i)). Evaluate the operand
            // as a pointer expression; default elem type to i32.
            EVal v = emit_expr(e, sc, ex->lhs);
            if (!v.is_ptr) {
                EMIT_ERR(e, "dereference of non-pointer expression");
                return r;
            }
            r.base_ptr = v.val;
            r.elem_ty = (v.ptr_elem != MLIR_INVALID_HANDLE) ? v.ptr_elem : e->i32;
            // The pointer's static type says whether the pointee is a char /
            // short / bool and its signedness (e.g. `*(unsigned short *)vp`,
            // `*(p + 1)`: pointer arithmetic does not carry the flag).
            Type pt = infer_expr_type(e, sc, ex->lhs);
            r.is_unsigned = pt.kind != TY_VOID ? pt.int_unsigned : v.is_unsigned;
            r.narrow_bits = type_is_narrow_int(pt) ? pt.int_bits : 0;
            r.is_bool = pt.is_bool;
            return r;
        }
        case EX_FIELD: {
            SCtx parent = walk_struct_lhs(e, sc, ex->lhs);
            if (!parent.ok) return r;
            int idx = struct_field_index(parent.sd, ex->name);
            if (idx < 0) {
                EMIT_ERR(e, "unknown struct field {}", ex->name);
                return r;
            }
            Type ft = parent.sd->fields.data[idx].type;
            if (ft.kind != TY_I32 && ft.kind != TY_I64 && ft.kind != TY_F32 &&
                ft.kind != TY_F64 &&
                ft.kind != TY_PTR_STRUCT && ft.kind != TY_PTR_I32 &&
                ft.kind != TY_PTR_CHAR && ft.kind != TY_PTR_VOID &&
                ft.kind != TY_FNPTR && ft.kind != TY_PTR_PTR) {
                EMIT_ERR(e, "field {} is not a scalar lvalue", ex->name);
                return r;
            }
            sctx_push(e, &parent, parent.sd->is_union ? 0 : idx);
            r.base_ptr = parent.base_ptr;
            r.source_elem = parent.source_elem;
            r.const_path = parent.const_path;
            r.n_const_path = parent.n_const_path;
            r.dyn_index = parent.dyn_index;
            r.elem_ty = scalar_mem_type(e, &ft);
            r.is_unsigned = ft.int_unsigned;
            r.narrow_bits = type_is_narrow_int(ft) ? ft.int_bits : 0;
            r.is_bool = ft.is_bool;
            r.in_union = parent.sd->is_union;
            return r;
        }
        default:
            EMIT_ERR(e, "invalid lvalue");
            return r;
    }
}

static void unify_numeric(E *e, EVal *a, EVal *b) {
    if (a->is_float || b->is_float) {
        // Promote integer side to a matching float type first.
        // The integer converts at its own width and signedness.
        if (!a->is_float) {
            a->val = emit_itofp(e, a->val, b->is_f64 ? e->f64 : e->f32, a->is_unsigned);
            a->is_f64 = b->is_f64;
            a->is_i64 = false; a->is_unsigned = false;
            a->is_float = true;
        }
        if (!b->is_float) {
            b->val = emit_itofp(e, b->val, a->is_f64 ? e->f64 : e->f32, b->is_unsigned);
            b->is_f64 = a->is_f64;
            b->is_i64 = false; b->is_unsigned = false;
            b->is_float = true;
        }
        // If one is f32 and the other f64, fpext the f32 side.
        if (a->is_f64 != b->is_f64) {
            if (!a->is_f64) { a->val = emit_fpext_f32_to_f64(e, a->val); a->is_f64 = true; }
            if (!b->is_f64) { b->val = emit_fpext_f32_to_f64(e, b->val); b->is_f64 = true; }
        }
        return;
    }
    if (a->is_i64 == b->is_i64) {
        bool u = a->is_unsigned || b->is_unsigned;
        a->is_unsigned = u; b->is_unsigned = u;
        return;
    }
    // Mixed widths: the 64-bit type can represent every value of the
    // 32-bit one, so the 64-bit operand's signedness wins (C's usual
    // arithmetic conversions): `-1LL < 1U` stays a signed compare.
    bool u = a->is_i64 ? a->is_unsigned : b->is_unsigned;
    if (!a->is_i64) {
        a->val = a->is_unsigned ? emit_extui_i32_to_i64(e, a->val)
                                : emit_extsi_i32_to_i64(e, a->val);
        a->is_i64 = true;
    }
    if (!b->is_i64) {
        b->val = b->is_unsigned ? emit_extui_i32_to_i64(e, b->val)
                                : emit_extsi_i32_to_i64(e, b->val);
        b->is_i64 = true;
    }
    a->is_unsigned = u; b->is_unsigned = u;
}

// ----- struct-by-pointer helpers -----

// Resolve a struct-typed call argument into the source !llvm.ptr.
// Accepts:
//   - EX_VAR struct          -> sym->addr
//   - EX_VAR struct*         -> load(sym->addr)
//   - EX_ADDR(EX_VAR struct) -> sym->addr
//   - any expression yielding a pointer (struct* field load, malloc cast,
//     null, etc.) -> the EVal's !llvm.ptr value
static void emit_struct_copy(E *e, MLIR_ValueHandle dst, MLIR_ValueHandle src, StructDef *sd);
static MLIR_ValueHandle resolve_struct_source(E *e, Scope *sc, Expr *arg, StructDef **out_sd) {
    // A struct-pointer argument may be an integer (`f(0)`), which converts
    // like any pointer argument. The caller sets the flag for this call
    // only, not for the sources nested in `arg`.
    bool int_ok = e->struct_src_int_ok;
    e->struct_src_int_ok = false;
    Expr *target = arg;
    if (arg->kind == EX_ADDR) {
        // `&<var>` — handled below.
        // `&arr[i]` where arr is array-of-struct: GEP to element address.
        if (arg->lhs->kind == EX_INDEX && arg->lhs->lhs->kind == EX_VAR) {
            Sym *s = lookup(e, sc, arg->lhs->lhs->name);
            if (s && s->type.kind == TY_ARRAY_STRUCT && s->sdef) {
                MLIR_ValueHandle idx = emit_expr_i32(e, sc, arg->lhs->rhs);
                MLIR_TypeHandle st_ty = find_struct_type(e, s->sdef);
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, st_ty, (uint64_t)s->type.array_len);
                int32_t path[2] = {0, LLVM_GEP_DYN};
                MLIR_ValueHandle dyn[1] = {idx};
                MLIR_ValueHandle p = emit_gep(e, sym_addr(e, s), arr_ty, path, 2, dyn, 1);
                *out_sd = s->sdef;
                return p;
            }
            // `&p[i]` where p is struct*: load p, then GEP by i (stride sizeof(struct)).
            if (s && s->type.kind == TY_PTR_STRUCT && s->sdef) {
                MLIR_ValueHandle idx = emit_expr_i32(e, sc, arg->lhs->rhs);
                MLIR_ValueHandle base = sym_load_ptr(e, s);
                MLIR_TypeHandle st_ty = find_struct_type(e, s->sdef);
                int32_t path[1] = {LLVM_GEP_DYN};
                MLIR_ValueHandle dyn[1] = {idx};
                MLIR_ValueHandle p = emit_gep(e, base, st_ty, path, 1, dyn, 1);
                *out_sd = s->sdef;
                return p;
            }
        }
        if (arg->lhs->kind != EX_VAR) {
            // Generic case: `&<lvalue-chain>` where the lvalue resolves
            // to a struct-typed slot reachable via FIELD/ARROW/INDEX
            // descent (e.g. `&st->scopes[i]`).
            Type t = infer_expr_type(e, sc, arg->lhs);
            if (t.kind == TY_STRUCT) {
                StructDef *sd = find_struct(e, t.struct_name);
                if (sd) {
                    SCtx c = walk_struct_lhs(e, sc, arg->lhs);
                    if (c.ok && c.sd == sd) {
                        LVal lv = (LVal){0};
                        lv.base_ptr = c.base_ptr;
                        lv.source_elem = c.source_elem;
                        lv.const_path = c.const_path;
                        lv.n_const_path = c.n_const_path;
                        lv.dyn_index = c.dyn_index;
                        *out_sd = sd;
                        return lval_address(e, lv);
                    }
                }
            }
            EMIT_ERR(e, "&expr in struct context requires a simple variable");
            return MLIR_INVALID_HANDLE;
        }
        target = arg->lhs;
    }
    if (target->kind == EX_VAR) {
        Sym *s = lookup(e, sc, target->name);
        if (s && (s->type.kind == TY_STRUCT || s->type.kind == TY_PTR_STRUCT) && s->sdef) {
            if (arg->kind == EX_ADDR && s->type.kind != TY_STRUCT) {
                EMIT_ERR(e, "&<var> requires a struct local");
                return MLIR_INVALID_HANDLE;
            }
            *out_sd = s->sdef;
            if (s->type.kind == TY_PTR_STRUCT) return sym_load_ptr(e, s);
            return sym_addr(e, s);
        }
    }
    // `*<expr>` where <expr> is a struct pointer: yield the loaded
    // !llvm.ptr value; the caller treats it as the struct source for
    // a by-value copy.
    if (arg->kind == EX_DEREF) {
        Type inner = infer_expr_type(e, sc, arg->lhs);
        if (inner.kind == TY_PTR_STRUCT) {
            StructDef *sd = find_struct(e, inner.struct_name);
            if (sd) {
                EVal pv = emit_expr(e, sc, arg->lhs);
                if (pv.is_ptr) {
                    *out_sd = sd;
                    return pv.val;
                }
            }
        }
    }
    // Struct lvalue chain (e.g. `ht->buckets[i].value` where `.value`
    // is itself a struct). walk_struct_lhs descends FIELD/INDEX/ARROW
    // chains and stops at struct-typed fields, yielding a base_ptr +
    // path that lval_address can resolve to the struct's address.
    if (arg->kind == EX_FIELD || arg->kind == EX_INDEX) {
        Type t = infer_expr_type(e, sc, arg);
        if (t.kind == TY_STRUCT) {
            StructDef *sd = find_struct(e, t.struct_name);
            if (sd) {
                SCtx c = walk_struct_lhs(e, sc, arg);
                if (c.ok && c.sd == sd) {
                    LVal lv = (LVal){0};
                    lv.base_ptr = c.base_ptr;
                    lv.source_elem = c.source_elem;
                    lv.const_path = c.const_path;
                    lv.n_const_path = c.n_const_path;
                    lv.dyn_index = c.dyn_index;
                    *out_sd = sd;
                    return lval_address(e, lv);
                }
            }
        }
    }
    // Ternary returning a struct: `cond ? a : b` where both arms are
    // struct-typed of the same struct. Allocate one slot, copy in each
    // branch, return the slot.
    if (arg->kind == EX_TERNARY) {
        Type t = infer_expr_type(e, sc, arg->rhs);
        if (t.kind == TY_STRUCT) {
            StructDef *sd = find_struct(e, t.struct_name);
            if (sd) {
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle save_ptr = MLIR_INVALID_HANDLE;
                MLIR_BlockHandle save_blk = e->cur_block;
                MLIR_ValueHandle slot = emit_alloca(e, st_ty);
                EVal cv = emit_expr(e, sc, arg->lhs);
                MLIR_ValueHandle cb = emit_to_bool_i1(e, cv);
                MLIR_BlockHandle then_blk = new_cfg_block(e);
                MLIR_BlockHandle else_blk = new_cfg_block(e);
                MLIR_BlockHandle merge_blk = new_cfg_block(e);
                e->cur_block = then_blk; e->terminated = false;
                StructDef *tsd = NULL;
                MLIR_ValueHandle tsrc = resolve_struct_source(e, sc, arg->rhs, &tsd);
                if (tsrc != MLIR_INVALID_HANDLE && tsd == sd) {
                    emit_struct_copy(e, slot, tsrc, sd);
                }
                emit_branch(e, merge_blk);
                e->cur_block = else_blk; e->terminated = false;
                StructDef *esd = NULL;
                MLIR_ValueHandle esrc = resolve_struct_source(e, sc, arg->lvalue, &esd);
                if (esrc != MLIR_INVALID_HANDLE && esd == sd) {
                    emit_struct_copy(e, slot, esrc, sd);
                }
                emit_branch(e, merge_blk);
                e->cur_block = save_blk; e->terminated = false;
                emit_cond_branch(e, cb, then_blk, else_blk);
                e->cur_block = merge_blk; e->terminated = false;
                (void)save_ptr;
                *out_sd = sd;
                return slot;
            }
        }
    }
    // Fallback: evaluate as a generic pointer expression.
    EVal v = emit_expr(e, sc, arg);
    if (!v.is_ptr && int_ok) {
        *out_sd = NULL;
        return int_to_ptr(e, v);
    }
    if (!v.is_ptr) {
        EMIT_ERR(e, "argument is not a struct or struct pointer");
        return MLIR_INVALID_HANDLE;
    }
    *out_sd = v.sdef;
    return v.val;
}

// Recursively copy each leaf of a struct from src to dst (both !llvm.ptr to
// the struct's storage), using the struct's identified type for GEP descent.
static void emit_struct_copy_path(E *e, MLIR_ValueHandle dst, MLIR_ValueHandle src,
                                  MLIR_TypeHandle source_elem, StructDef *sd,
                                  int32_t *prefix, size_t n_prefix) {
    if (sd->is_union) {
        // The union's llvm body is a single blob field (index 0) of
        // `count` alignment-wide integers. Copy the blob element by element.
        MLIR_TypeHandle uelem; int64_t ues, uc;
        union_blob_layout(e, sd, &uelem, &ues, &uc);
        for (int64_t j = 0; j < uc; j++) {
            size_t total = n_prefix + 2;
            int32_t *p2 = arena_new_array(e->arena, int32_t, total);
            for (size_t k = 0; k < n_prefix; k++) p2[k] = prefix[k];
            p2[n_prefix] = 0;
            p2[n_prefix + 1] = (int32_t)j;
            MLIR_ValueHandle sp = emit_gep(e, src, source_elem, p2, total, NULL, 0);
            MLIR_ValueHandle val = emit_load_v(e, sp, uelem);
            MLIR_ValueHandle dp = emit_gep(e, dst, source_elem, p2, total, NULL, 0);
            emit_store_v(e, val, dp);
        }
        return;
    }
    for (size_t i = 0; i < sd->fields.size; i++) {
        Type ft = sd->fields.data[i].type;
        size_t n_path = n_prefix + 1;
        int32_t *path = arena_new_array(e->arena, int32_t, n_path);
        for (size_t k = 0; k < n_prefix; k++) path[k] = prefix[k];
        path[n_prefix] = (int32_t)i;
        if (ft.kind == TY_STRUCT) {
            StructDef *inner = find_struct(e, ft.struct_name);
            emit_struct_copy_path(e, dst, src, source_elem, inner, path, n_path);
        } else if (ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                   ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID ||
                   ft.kind == TY_PTR_PTR || ft.kind == TY_FNPTR) {
            MLIR_ValueHandle sp = emit_gep(e, src, source_elem, path, n_path, NULL, 0);
            MLIR_ValueHandle val = emit_load_v(e, sp, e->ptr);
            MLIR_ValueHandle dp = emit_gep(e, dst, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, val, dp);
        } else if (ft.kind == TY_ARRAY_I32 || ft.kind == TY_ARRAY_F32 ||
                   ft.kind == TY_ARRAY_PTR_STRUCT || ft.kind == TY_ARRAY_PTR_CHAR) {
            bool is_2d = (ft.array_len2 != 0);
            int64_t n1 = ft.array_len;
            int64_t n2 = is_2d ? ft.array_len2 : 1;
            MLIR_TypeHandle elem = array_elem_type(e, &ft);
            for (int64_t a = 0; a < n1; a++) {
                for (int64_t b = 0; b < n2; b++) {
                    size_t inner_n = is_2d ? 2 : 1;
                    size_t total = n_path + inner_n;
                    int32_t *p2 = arena_new_array(e->arena, int32_t, total);
                    for (size_t kk = 0; kk < n_path; kk++) p2[kk] = path[kk];
                    p2[n_path] = (int32_t)a;
                    if (is_2d) p2[n_path + 1] = (int32_t)b;
                    MLIR_ValueHandle sp = emit_gep(e, src, source_elem, p2, total, NULL, 0);
                    MLIR_ValueHandle val = emit_load_v(e, sp, elem);
                    MLIR_ValueHandle dp = emit_gep(e, dst, source_elem, p2, total, NULL, 0);
                    emit_store_v(e, val, dp);
                }
            }
        } else {
            MLIR_TypeHandle elem = scalar_mem_type(e, &ft);
            MLIR_ValueHandle sp = emit_gep(e, src, source_elem, path, n_path, NULL, 0);
            MLIR_ValueHandle val = emit_load_v(e, sp, elem);
            MLIR_ValueHandle dp = emit_gep(e, dst, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, val, dp);
        }
    }
}

static void emit_struct_copy(E *e, MLIR_ValueHandle dst, MLIR_ValueHandle src, StructDef *sd) {
    MLIR_TypeHandle st = find_struct_type(e, sd);
    int32_t *prefix = arena_new_array(e->arena, int32_t, 1);
    prefix[0] = 0;
    emit_struct_copy_path(e, dst, src, st, sd, prefix, 1);
}

// Emit a func.call (non-variadic) or llvm.call (variadic). For
// struct-returning functions a hidden first sret !llvm.ptr operand is
// prepended; the function returns void; out_results[0] (if non-NULL) is
// set to that sret pointer. If out_sret_buf is non-INVALID, that ptr is
// used as the sret buffer instead of allocating a fresh one.
//
// For variadic callees the operand list may have MORE entries than
// sig->n_params; trailing args are appended verbatim (with the standard C
// promotion of f32 to f64 — currently rejected with a clear error since
// tinyC has no double type).
static void emit_flat_call(E *e, Scope *sc, FuncSig *sig, VecExprPtr args,
                           MLIR_ValueHandle *out_results,
                           MLIR_ValueHandle out_sret_buf) {
    if (!sig->is_variadic && args.size != sig->n_params) {
        EMIT_ERR(e, "call to {} arity mismatch", sig->name);
    }
    if (sig->is_variadic && args.size < sig->n_params) {
        EMIT_ERR(e, "call to variadic {} requires at least {} fixed arg(s)",
                 sig->name, (int64_t)sig->n_params);
    }
    size_t n_fixed = sig->n_flat_in;
    // cursor_va: the varargs are packed into a single buffer passed via the
    // hidden trailing ptr param (already counted in n_flat_in), so there are
    // no per-arg operands. Real (extern) variadic calls pass each vararg as an
    // operand of an llvm.call.
    bool real_variadic = sig->is_variadic && !sig->cursor_va;
    size_t n_extra = real_variadic ? (args.size - sig->n_params) : 0;
    size_t n_in = n_fixed + n_extra;
    // Variadic struct args may expand into multiple i64 words; account for that.
    if (real_variadic) {
        for (size_t i = sig->n_params; i < args.size; i++) {
            Type at = infer_expr_type(e, sc, args.data[i]);
            if (at.kind == TY_STRUCT) {
                int64_t bytes = type_size(e, at);
                int64_t words = (bytes + 7) / 8;
                if (words > 1) n_in += (words - 1);
            }
        }
    }
    MLIR_ValueHandle ops_stk[16];
    MLIR_ValueHandle *ops = (n_in <= 16) ? ops_stk
        : arena_new_array(e->arena, MLIR_ValueHandle, n_in);
    size_t op_off = 0;
    MLIR_ValueHandle ret_buf = MLIR_INVALID_HANDLE;
    if (sig->sret) {
        if (out_sret_buf != MLIR_INVALID_HANDLE) {
            ret_buf = out_sret_buf;
        } else {
            MLIR_TypeHandle st = find_struct_type(e, sig->ret.sdef);
            ret_buf = emit_alloca(e, st);
        }
        ops[op_off++] = ret_buf;
    }
    for (size_t i = 0; i < sig->n_params && i < args.size; i++) {
        SlotInfo *p = &sig->params[i];
        if (p->type.kind == TY_STRUCT) {
            // Caller-byval: allocate a fresh copy of the struct, copy from
            // the source, pass !llvm.ptr to the copy.
            StructDef *sd = NULL;
            MLIR_ValueHandle src = resolve_struct_source(e, sc, args.data[i], &sd);
            if (src == MLIR_INVALID_HANDLE || sd != p->sdef) {
                EMIT_ERR(e, "struct call arg type mismatch");
                ops[op_off++] = src;
                continue;
            }
            MLIR_TypeHandle st = find_struct_type(e, sd);
            MLIR_ValueHandle tmp = emit_alloca(e, st);
            emit_struct_copy(e, tmp, src, sd);
            ops[op_off++] = tmp;
        } else if (p->type.kind == TY_PTR_STRUCT) {
            StructDef *sd = NULL;
            e->struct_src_int_ok = true;
            MLIR_ValueHandle src = resolve_struct_source(e, sc, args.data[i], &sd);
            ops[op_off++] = src;
        } else {
            EVal v = emit_expr(e, sc, args.data[i]);
            ops[op_off++] = coerce_eval_to(e, v, scalar_mlir_type(e, p->type.kind),
                                           p->type);
        }
    }
    // Variadic-portion arguments: apply C default argument promotion
    // (float -> double; narrow ints are already modelled as i32).
    if (sig->cursor_va) {
        // Pack each vararg into a uniform 8-byte-slot buffer (tinyC's portable
        // cursor va_list layout) and pass the buffer via the hidden trailing
        // ptr param. Structs occupy ceil(size/8) consecutive slots.
        size_t n_slots = 0;
        for (size_t i = sig->n_params; i < args.size; i++) {
            Type at = infer_expr_type(e, sc, args.data[i]);
            n_slots += (at.kind == TY_STRUCT)
                ? (size_t)((type_size(e, at) + 7) / 8) : 1;
        }
        size_t buf_slots = n_slots ? n_slots : 1;  // valid ptr even with 0 args
        MLIR_TypeHandle buf_ty = MLIR_CreateTypeLLVMArray(e->ctx, e->i64,
                                                          (uint64_t)buf_slots);
        MLIR_ValueHandle buf = emit_alloca(e, buf_ty);
        size_t slot = 0;
        for (size_t i = sig->n_params; i < args.size; i++) {
            Type at = infer_expr_type(e, sc, args.data[i]);
            int32_t path[2] = {0, (int32_t)slot};
            MLIR_ValueHandle slot_addr = emit_gep(e, buf, buf_ty, path, 2, NULL, 0);
            if (at.kind == TY_STRUCT) {
                StructDef *asd = NULL;
                MLIR_ValueHandle src = resolve_struct_source(e, sc, args.data[i], &asd);
                if (src == MLIR_INVALID_HANDLE || !asd) {
                    EMIT_ERR(e, "variadic struct argument: cannot resolve");
                    continue;
                }
                emit_struct_copy(e, slot_addr, src, asd);
                slot += (size_t)((type_size(e, at) + 7) / 8);
            } else {
                EVal v = emit_expr(e, sc, args.data[i]);
                if (v.is_float && !v.is_f64) {
                    v.val = emit_fpext_f32_to_f64(e, v.val);
                    v.is_f64 = true;
                }
                emit_store_v(e, v.val, slot_addr);
                slot += 1;
            }
        }
        ops[op_off++] = buf;  // hidden trailing __va param
    } else {
        // Real (extern) variadic call: pass each promoted vararg as an llvm.call
        // operand. Struct-by-value args are unpacked into i64 words.
        for (size_t i = sig->n_params; i < args.size; i++) {
            Type at = infer_expr_type(e, sc, args.data[i]);
            if (at.kind == TY_STRUCT) {
                StructDef *asd = NULL;
                MLIR_ValueHandle src = resolve_struct_source(e, sc, args.data[i], &asd);
                if (src == MLIR_INVALID_HANDLE || !asd) {
                    EMIT_ERR(e, "variadic struct argument: cannot resolve");
                    continue;
                }
                int64_t bytes = type_size(e, at);
                int64_t words = (bytes + 7) / 8;
                MLIR_TypeHandle warr = MLIR_CreateTypeLLVMArray(e->ctx, e->i64, (uint64_t)words);
                for (int64_t w = 0; w < words; w++) {
                    int32_t path[2] = {0, (int32_t)w};
                    MLIR_ValueHandle wp = emit_gep(e, src, warr, path, 2, NULL, 0);
                    MLIR_ValueHandle wv = emit_load_v(e, wp, e->i64);
                    ops[op_off++] = wv;
                }
                continue;
            }
            EVal v = emit_expr(e, sc, args.data[i]);
            if (v.is_float && !v.is_f64) {
                v.val = emit_fpext_f32_to_f64(e, v.val);
                v.is_f64 = true;
            }
            ops[op_off++] = v.val;
        }
    }
    size_t n_out = sig->n_flat_out;
    MLIR_TypeHandle rts_stk[2];
    MLIR_ValueHandle rs_stk[2];
    MLIR_TypeHandle *rts = (n_out <= 2) ? rts_stk : arena_new_array(e->arena, MLIR_TypeHandle, n_out);
    MLIR_ValueHandle *rs = (n_out <= 2) ? rs_stk  : arena_new_array(e->arena, MLIR_ValueHandle, n_out);
    for (size_t i = 0; i < n_out; i++) {
        rts[i] = sig->flat_out_tys[i];
    }
    if (!sig->callee_attr) {
        sig->callee_attr = MLIR_CreateAttributeSymbolRef(e->ctx, sig->name);
    }
    MLIR_NamedAttr callee_attr = MLIR_Named(MLIR_KEY_CALLEE, sig->callee_attr);
    if (real_variadic) {
        MLIR_NamedAttr var_ty = MLIR_Named(MLIR_KEY_VAR_CALLEE_TYPE, MLIR_CreateAttributeType(
            e->ctx, sig->llvm_fn_ty));
        MLIR_NamedAttr as[2];
        as[0] = callee_attr; as[1] = var_ty;
        emit_op(e, OP_TYPE_LLVM_CALL,
                rts, n_out, rs, n_out, ops, op_off, as, 2, NULL, 0);
    } else {
        emit_op(e, OP_TYPE_FUNC_CALL,
                rts, n_out, rs, n_out, ops, op_off, &callee_attr, 1, NULL, 0);
    }
    if (out_results) {
        if (sig->sret) out_results[0] = ret_buf;
        else for (size_t i = 0; i < n_out; i++) out_results[i] = rs[i];
    }
}

// Bring `v` into the range of the type (is_i64, is_unsigned).
static int64_t fold_norm(int64_t v, bool is_i64, bool is_unsigned) {
    if (is_i64) return v;
    return is_unsigned ? (int64_t)(uint32_t)v : (int64_t)(int32_t)v;
}

static FoldVal fold_int(int64_t v) {
    FoldVal r = {0};
    r.v = (int64_t)(int32_t)v;
    return r;
}

// A `sizeof` result: a size_t (`unsigned long`), unsigned and 64-bit
// except on wasm32.
static FoldVal fold_size(E *e, int64_t v) {
    FoldVal r = {0};
    r.is_i64 = !e->target_wasm32;
    r.is_unsigned = true;
    r.v = fold_norm(v, r.is_i64, true);
    return r;
}

static bool fold_expr(Expr *ex, FoldLeafFn leaf, void *ctx, bool lenient,
                      FoldVal *out);

// Convert two folded operands to their common type (C's usual arithmetic
// conversions).
static void fold_convert(FoldVal *a, FoldVal *b) {
    bool i64 = a->is_i64 || b->is_i64;
    bool u;
    if (a->is_i64 == b->is_i64) u = a->is_unsigned || b->is_unsigned;
    else u = a->is_i64 ? a->is_unsigned : b->is_unsigned;
    a->v = fold_norm(a->v, i64, u); a->is_i64 = i64; a->is_unsigned = u;
    b->v = fold_norm(b->v, i64, u); b->is_i64 = i64; b->is_unsigned = u;
}

// Constant-fold a literal-only integer expression with C semantics: each
// literal has its own width and signedness, and operations follow the
// usual arithmetic conversions. Folded:
//   EX_INT
//   EX_UN: OP_NEG, OP_BNOT, OP_NOT
//   EX_BIN: + - * / % & | ^ << >> < <= > >= == != && ||
//   EX_TERNARY when the condition and both branches fold (the result
//     has the branches' common type, so a float branch blocks folding);
//     the branch not taken, like the right operand of && / || that is
//     not evaluated, only needs an integer constant type, not a value
//     (`1 ? 7 : 1 / 0`)
//   anything else `leaf` folds (sizeof in the emitter, enumerators and
//     an unevaluated sizeof in the parser); `leaf` may be NULL.
//
// Refused (returns false; the emitter falls back to normal emission):
//   - variables, calls, assignments, floats, casts, ... (unless `leaf`
//     folds them)
//   - division / modulo by zero and signed MIN / -1
//   - shifts by a negative count or by the left operand's width or more
// except in an operand that is not evaluated (`lenient`): there they give
// 0, since only the operand's type matters.
bool tinyc_fold(Expr *ex, FoldLeafFn leaf, void *ctx, FoldVal *out) {
    return fold_expr(ex, leaf, ctx, false, out);
}

static bool fold_expr(Expr *ex, FoldLeafFn leaf, void *ctx, bool lenient,
                      FoldVal *out) {
    if (!ex) return false;
    switch (ex->kind) {
        case EX_INT: {
            FoldVal r = {0};
            r.is_i64 = ex->is_i64;
            r.is_unsigned = ex->is_unsigned;
            r.v = fold_norm(ex->int_value, r.is_i64, r.is_unsigned);
            *out = r;
            return true;
        }
        case EX_UN: {
            FoldVal a;
            if (!fold_expr(ex->lhs, leaf, ctx, lenient, &a)) return false;
            switch (ex->op) {
                case OP_NEG:
                    a.v = fold_norm((int64_t)(0 - (uint64_t)a.v), a.is_i64, a.is_unsigned);
                    *out = a; return true;
                case OP_BNOT:
                    a.v = fold_norm(~a.v, a.is_i64, a.is_unsigned);
                    *out = a; return true;
                case OP_NOT:
                    *out = fold_int(a.v == 0 ? 1 : 0); return true;
                default: return false;
            }
        }
        case EX_BIN: {
            FoldVal a, b;
            if (ex->op == OP_SHL || ex->op == OP_SHR) {
                // The result has the left operand's type.
                if (!fold_expr(ex->lhs, leaf, ctx, lenient, &a) ||
                    !fold_expr(ex->rhs, leaf, ctx, lenient, &b))
                    return false;
                int width = a.is_i64 ? 64 : 32;
                if (b.v < 0 || b.v >= width) {
                    if (!lenient) return false;
                    a.v = 0;
                } else if (ex->op == OP_SHL) {
                    a.v = fold_norm((int64_t)((uint64_t)a.v << b.v), a.is_i64, a.is_unsigned);
                } else if (a.is_unsigned) {
                    a.v = (int64_t)((uint64_t)a.v >> b.v);  // zero-extended when 32-bit
                } else {
                    a.v = a.v >> b.v;
                }
                *out = a;
                return true;
            }
            if (ex->op == OP_AND || ex->op == OP_OR) {
                // The right operand is not evaluated when the left one
                // decides the result.
                if (!fold_expr(ex->lhs, leaf, ctx, lenient, &a)) return false;
                bool decided = ex->op == OP_AND ? a.v == 0 : a.v != 0;
                if (!fold_expr(ex->rhs, leaf, ctx, lenient || decided, &b))
                    return false;
                bool t = ex->op == OP_AND ? (a.v != 0 && b.v != 0) : (a.v != 0 || b.v != 0);
                *out = fold_int(t ? 1 : 0);
                return true;
            }
            if (!fold_expr(ex->lhs, leaf, ctx, lenient, &a) ||
                !fold_expr(ex->rhs, leaf, ctx, lenient, &b))
                return false;
            fold_convert(&a, &b);
            bool u = a.is_unsigned;
            uint64_t ua = (uint64_t)a.v, ub = (uint64_t)b.v;
            FoldVal r = a;
            switch (ex->op) {
                case OP_ADD: r.v = (int64_t)(ua + ub); break;
                case OP_SUB: r.v = (int64_t)(ua - ub); break;
                case OP_MUL: r.v = (int64_t)(ua * ub); break;
                case OP_DIV: case OP_MOD: {
                    if (b.v == 0) {
                        if (!lenient) return false;
                        r.v = 0;
                    } else if (u) {
                        r.v = (int64_t)(ex->op == OP_DIV ? ua / ub : ua % ub);
                    } else {
                        int64_t min = a.is_i64 ? (int64_t)0x8000000000000000ull
                                               : -(int64_t)0x80000000ll;
                        if (a.v == min && b.v == -1) {
                            if (!lenient) return false;
                            r.v = 0;
                        } else {
                            r.v = ex->op == OP_DIV ? a.v / b.v : a.v % b.v;
                        }
                    }
                    break;
                }
                case OP_BAND: r.v = a.v & b.v; break;
                case OP_BOR:  r.v = a.v | b.v; break;
                case OP_BXOR: r.v = a.v ^ b.v; break;
                case OP_LT: *out = fold_int(u ? ua <  ub : a.v <  b.v); return true;
                case OP_LE: *out = fold_int(u ? ua <= ub : a.v <= b.v); return true;
                case OP_GT: *out = fold_int(u ? ua >  ub : a.v >  b.v); return true;
                case OP_GE: *out = fold_int(u ? ua >= ub : a.v >= b.v); return true;
                case OP_EQ: *out = fold_int(a.v == b.v); return true;
                case OP_NE: *out = fold_int(a.v != b.v); return true;
                default: return false;
            }
            r.v = fold_norm(r.v, r.is_i64, r.is_unsigned);
            *out = r;
            return true;
        }
        case EX_TERNARY: {
            // The result has the branches' common type, so both must fold:
            // `0 ? 1.5 : 3` is a double, not the int 3. Only the taken
            // branch is evaluated.
            FoldVal c, a, b;
            if (!fold_expr(ex->lhs, leaf, ctx, lenient, &c)) return false;
            bool then = c.v != 0;
            if (!fold_expr(ex->rhs, leaf, ctx, lenient || !then, &a) ||
                !fold_expr(ex->lvalue, leaf, ctx, lenient || then, &b))
                return false;
            fold_convert(&a, &b);
            *out = then ? a : b;
            return true;
        }
        default:
            return leaf && leaf(ctx, ex, lenient, out);
    }
}

// The emitter's leaves for tinyc_fold: enumerators (an `int`) and `sizeof`
// (a size_t), as the emitter types them.
typedef struct {
    E *e;
    Scope *sc;
} EmitFoldCtx;

static bool emit_fold_leaf(void *ctx, Expr *ex, bool lenient, FoldVal *out) {
    EmitFoldCtx *fc = (EmitFoldCtx *)ctx;
    if (ex->kind == EX_VAR) {
        // An enumerator (an `int`) unless a variable shadows it, as in
        // emit_expr's EX_VAR.
        int64_t en_v;
        if (lookup(fc->e, fc->sc, ex->name) || !find_enum(fc->e, ex->name, &en_v))
            return false;
        *out = fold_int(en_v);
        return true;
    }
    if (ex->kind != EX_SIZEOF) return false;
    // An unevaluated `sizeof` only needs its type, a size_t; its operand
    // may not be sizable here (`sizeof g` in a global array bound).
    if (lenient) {
        *out = fold_size(fc->e, 0);
        return true;
    }
    Type ty = (Type){0};
    if (ex->sizeof_is_expr) {
        if (!fc->sc) return false;
        if (ex->lhs && ex->lhs->kind == EX_STR) {
            // sizeof("string literal") yields the array size
            // including the trailing NUL, matching standard C.
            *out = fold_size(fc->e, ex->is_alignof ? 1 : (int64_t)ex->lhs->name.size);
            return true;
        }
        ty = infer_expr_type(fc->e, fc->sc, ex->lhs);
        if (ty.kind == TY_VOID) return false;
        *out = fold_size(fc->e, ex->is_alignof ? type_align(fc->e, ty)
                                               : c_sizeof_expr(fc->e, fc->sc, ex->lhs));
        return true;
    }
    if (!ex->cast_type || ex->cast_type->kind == TY_VOID) return false;
    *out = fold_size(fc->e, ex->is_alignof ? type_align(fc->e, *ex->cast_type)
                                           : type_size(fc->e, *ex->cast_type));
    return true;
}

static bool ast_fold(E *e, Scope *sc, Expr *ex, FoldVal *out) {
    EmitFoldCtx fc;
    fc.e = e;
    fc.sc = sc;
    return tinyc_fold(ex, emit_fold_leaf, &fc, out);
}

static bool ast_fold_int(E *e, Scope *sc, Expr *ex, int64_t *out) {
    FoldVal r;
    if (!ast_fold(e, sc, ex, &r)) return false;
    *out = r.v;
    return true;
}

// Resolve a deferred array-length expression on `t` (set when the
// parser saw `[<const-expr>]`) into the integer `array_len` slot, in
// the given scope. No-op when array_len_expr is NULL or array_len is
// already set. The expression must constant-fold to a positive int32;
// otherwise we leave array_len at 0 and let the caller error.
static void resolve_array_len(E *e, Scope *sc, Type *t) {
    if (!t->array_len_expr) return;
    if (t->array_len > 0) return;
    int64_t v = 0;
    if (!ast_fold_int(e, sc, t->array_len_expr, &v)) {
        EMIT_ERR(e, "array length is not a constant integer expression");
        return;
    }
    if (v <= 0) {
        EMIT_ERR(e, "array length must be positive (got {})", v);
        return;
    }
    t->array_len = v;
}

// The width rank of an integer type: its int_bits, where a plain `int`
// may carry 0 (keyword-declared locals). `long` is 64 also on wasm32.
static int int_type_bits(const Type *t) {
    if (t->int_bits) return t->int_bits;
    return t->kind == TY_I64 ? 64 : 32;
}

static bool type_is_arith(const Type *t) {
    return t->kind == TY_I32 || t->kind == TY_I64 || t->kind == TY_F32 || t->kind == TY_F64;
}

// C's integer promotions: a char / short / bool becomes `int`.
// (The helpers below update Types in place, field by field: copying a whole
// Type is expensive in the self-hosted build.)
static void promote_int_type(Type *t) {
    if (t->kind == TY_I32 && (t->int_bits == 8 || t->int_bits == 16 || t->is_bool)) {
        t->int_bits = 0;
        t->int_unsigned = false;
        t->is_bool = false;
    }
}

// The type of `a op b` (and of `c ? a : b`) under C's usual arithmetic
// conversions, stored into *a. Matches what unify_numeric emits: a double /
// float operand wins; integers are promoted; a 64-bit operand's type wins
// over a 32-bit one; at equal width the higher rank (`long` over `int` on
// wasm32) is taken, unsigned if either operand is.
static void arith_conv_type(Type *a, const Type *b) {
    if (a->kind == TY_F64 || b->kind == TY_F64 || a->kind == TY_F32 || b->kind == TY_F32) {
        a->kind = (a->kind == TY_F64 || b->kind == TY_F64) ? TY_F64 : TY_F32;
        a->int_bits = 0;
        a->int_unsigned = false;
        a->is_bool = false;
        return;
    }
    bool b_promotes = b->kind == TY_I32 &&
                      (b->int_bits == 8 || b->int_bits == 16 || b->is_bool);
    int b_bits = b_promotes ? 32 : int_type_bits(b);
    bool b_unsigned = !b_promotes && b->int_unsigned;
    promote_int_type(a);
    if (a->kind != b->kind) {
        if (b->kind == TY_I64) {
            a->kind = TY_I64;
            a->int_bits = b_bits;
            a->int_unsigned = b_unsigned;
        }
        return;
    }
    if (b_bits > int_type_bits(a)) a->int_bits = b_bits;
    a->int_unsigned = a->int_unsigned || b_unsigned;
}

// True if the i-th `_Generic` association's type matches the controlling
// expression type. struct types match by struct_name; integer types
// match by kind, width and signedness. Pointer kinds match by exact kind.
static bool generic_type_matches(Type a, Type b) {
    if (a.kind != b.kind) return false;
    if (a.kind == TY_STRUCT || a.kind == TY_PTR_STRUCT) {
        return str_eq(a.struct_name, b.struct_name);
    }
    if (a.kind == TY_I32 || a.kind == TY_I64) {
        // A plain `int` may carry int_bits 0 (keyword-declared locals,
        // arithmetic results): it is 32 bits wide, not "any width".
        return int_type_bits(&a) == int_type_bits(&b) &&
               a.int_unsigned == b.int_unsigned;
    }
    return true;
}

static Expr *generic_select(E *e, Scope *sc, Expr *ex) {
    Type ct = infer_expr_type(e, sc, ex->lhs);
    // C array decay: in _Generic, an array operand decays to a pointer.
    if (ct.kind == TY_ARRAY_I32 || ct.kind == TY_ARRAY_F32) {
        Type d = (Type){0};
        d.kind = TY_PTR_CHAR; // char[] is the typical case (string, buffer)
        // For int[] we'd want TY_PTR_I32; but in practice corec uses char[].
        // Still, prefer TY_PTR_I32 if int_bits suggests int.
        if (ct.int_bits != 0 && ct.int_bits != 8) d.kind = TY_PTR_I32;
        ct = d;
    } else if (ct.kind == TY_ARRAY_STRUCT || ct.kind == TY_ARRAY_PTR_STRUCT ||
               ct.kind == TY_ARRAY_PTR_CHAR) {
        Type d = (Type){0};
        d.kind = TY_PTR_VOID;
        ct = d;
    }
    int n = (int)ex->args.size;
    int chosen = -1;
    for (int i = 0; i < n; i++) {
        if (i == ex->generic_default_index) continue;
        if (generic_type_matches(ex->generic_types[i], ct)) {
            chosen = i; break;
        }
    }
    if (chosen < 0) chosen = ex->generic_default_index;
    if (chosen < 0 || chosen >= n) {
        EMIT_ERR(e, "_Generic: no matching association and no default");
        return ex->lhs;
    }
    return ex->args.data[chosen];
}

static EVal emit_expr(E *e, Scope *sc, Expr *ex) {
    e->cur_line = ex->line;
    EVal r = {0};
    if (ex->kind == EX_GENERIC) {
        return emit_expr(e, sc, generic_select(e, sc, ex));
    }
    // `__func__` is a magic predefined identifier carrying the enclosing
    // function's name as a NUL-terminated string literal. Synthesize the
    // string from the current FuncSig at emit time.
    if (ex->kind == EX_VAR &&
        ex->name.size == 8 &&
        memcmp(ex->name.str, "__func__", 8) == 0 &&
        !lookup(e, sc, ex->name)) {
        string fname = e->cur_sig ? e->cur_sig->name : str_lit("");
        // Build a NUL-terminated copy in arena and intern.
        char *buf = arena_alloc(e->arena, fname.size + 1);
        for (size_t i = 0; i < fname.size; i++) buf[i] = fname.str[i];
        buf[fname.size] = '\0';
        string bytes = (string){.str = buf, .size = fname.size + 1};
        string sym = intern_string(e, bytes);
        r.val = emit_addressof(e, sym, e->ptr);
        r.is_ptr = true;
        r.is_str = true;
        r.ptr_elem = e->i8;
        return r;
    }
    // Pre-pass: literal-only integer subtree -> a single arith.constant.
    // Only kicks in for nodes that ast_fold_int actually folds; anything
    // with a side effect or a non-int operand falls through unchanged.
    if (ex->kind == EX_BIN || ex->kind == EX_UN || ex->kind == EX_TERNARY) {
        FoldVal fv;
        if (ast_fold(e, sc, ex, &fv)) {
            r.val = fv.is_i64 ? emit_const_i64(e, fv.v) : emit_const_i32(e, fv.v);
            r.is_i64 = fv.is_i64;
            r.is_unsigned = fv.is_unsigned;
            return r;
        }
    }
    switch (ex->kind) {
        case EX_GENERIC:
            // Handled above; listed here only to silence -Wswitch.
            break;
        case EX_INT:
            if (ex->is_i64) {
                r.val = emit_const_i64(e, ex->int_value);
                r.is_i64 = true;
            } else {
                r.val = emit_const_i32(e, ex->int_value);
            }
            r.is_unsigned = ex->is_unsigned;
            return r;
        case EX_FLOAT:
            if (ex->is_f64) {
                r.val = emit_const_f64(e, ex->float_value);
                r.is_float = true; r.is_f64 = true;
            } else {
                r.val = emit_const_f32(e, ex->float_value);
                r.is_float = true;
            }
            return r;
        case EX_NULL:
            r.val = emit_null_ptr(e); r.is_ptr = true; r.sdef = NULL; return r;
        case EX_VA_ARG: {
            // Portable cursor va_list: the va_list buffer's first pointer
            // slot holds the read cursor (into the caller-packed 8-byte-slot
            // varargs buffer). va_arg loads the value at the cursor, then
            // advances the cursor by one slot (8 bytes) — or by the struct's
            // rounded-up word count. No target-specific va_arg ABI and no
            // external helper: works identically on wasm / x86-64 / arm64.
            if (!ex->lhs || ex->lhs->kind != EX_VAR) {
                EMIT_ERR(e, "va_arg first argument must be a va_list variable");
                r.val = emit_const_i32(e, 0); return r;
            }
            Sym *sy = lookup(e, sc, ex->lhs->name);
            if (!sy || sy->type.kind != TY_VA_LIST) {
                EMIT_ERR(e, "va_arg first argument must be of type va_list");
                r.val = emit_const_i32(e, 0); return r;
            }
            TypeKind tk = ex->cast_type->kind;
            MLIR_ValueHandle ap_ptr = sym_addr(e, sy);
            MLIR_ValueHandle cursor = emit_load_v(e, ap_ptr, e->ptr);
            if (tk == TY_STRUCT) {
                // va_arg of a struct: copy sizeof(struct) bytes (rounded up to
                // a whole number of 8-byte slots) out of the buffer, then
                // advance the cursor past those slots.
                StructDef *sd = find_struct(e, ex->cast_type->struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct in va_arg");
                    r.val = emit_const_i32(e, 0); return r;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle out = emit_alloca(e, st_ty);
                emit_struct_copy(e, out, cursor, sd);
                int64_t words = (type_size(e, *ex->cast_type) + 7) / 8;
                int32_t adv = (int32_t)(words * 8);
                MLIR_ValueHandle nxt = emit_gep(e, cursor, e->i8, &adv, 1, NULL, 0);
                emit_store_v(e, nxt, ap_ptr);
                r.val = out;
                r.is_ptr = true;
                r.sdef = sd;
                return r;
            }
            MLIR_TypeHandle rt;
            switch (tk) {
                case TY_I32:        rt = e->i32; break;
                case TY_I64:        rt = e->i64; break;
                case TY_F64:        rt = e->f64; break;
                case TY_PTR_CHAR:
                case TY_PTR_VOID:
                case TY_PTR_I32:
                case TY_PTR_STRUCT:
                case TY_PTR_PTR:    rt = e->ptr; break;
                default:
                    EMIT_ERR(e, "va_arg only supports int, long, and pointer types");
                    r.val = emit_const_i32(e, 0); return r;
            }
            MLIR_ValueHandle val = emit_load_v(e, cursor, rt);
            int32_t adv = 8;  // one 8-byte slot
            MLIR_ValueHandle nxt = emit_gep(e, cursor, e->i8, &adv, 1, NULL, 0);
            emit_store_v(e, nxt, ap_ptr);
            r.val = val;
            r.is_i64 = (tk == TY_I64);
            r.is_float = (tk == TY_F64);
            r.is_f64 = (tk == TY_F64);
            r.is_ptr = (tk == TY_PTR_CHAR || tk == TY_PTR_VOID || tk == TY_PTR_I32
                        || tk == TY_PTR_STRUCT || tk == TY_PTR_PTR);
            r.is_str = (tk == TY_PTR_CHAR);
            r.is_void_ptr = (tk == TY_PTR_VOID);
            r.ptr_elem = ptr_elem_of_type(e, ex->cast_type);
            return r;
        }
        case EX_STR: {
            // String literal: emit a !llvm.ptr to a deduplicated
            // module-level llvm.mlir.global string buffer. The literal is an
            // array of char that decays to a char*, so `"abc" + 1` and
            // `*("abc" + 1)` step over / load single bytes.
            string sym = intern_string(e, ex->name);
            r.val = emit_addressof(e, sym, e->ptr);
            r.is_ptr = true;
            r.is_str = true;
            r.ptr_elem = e->i8;
            return r;
        }
        case EX_TERNARY: {
            // Lower `c ? a : b` via cf.cond_br + alloca for result.
            EVal cv = emit_expr(e, sc, ex->lhs);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, cv);
            MLIR_BlockHandle then_blk = new_cfg_block(e);
            MLIR_BlockHandle else_blk = new_cfg_block(e);
            MLIR_BlockHandle merge_blk = new_cfg_block(e);

            MLIR_BlockHandle save = e->cur_block;
            e->cur_block = then_blk; e->terminated = false;
            EVal av = emit_expr(e, sc, ex->rhs);
            MLIR_BlockHandle then_tail = e->cur_block;

            e->cur_block = else_blk; e->terminated = false;
            EVal bv = emit_expr(e, sc, ex->lvalue);
            MLIR_BlockHandle else_tail = e->cur_block;

            bool any_float = av.is_float || bv.is_float;
            bool any_f64   = av.is_f64   || bv.is_f64;
            bool any_ptr   = av.is_ptr   || bv.is_ptr;
            bool any_i64   = av.is_i64   || bv.is_i64;
            MLIR_TypeHandle rty = any_ptr ? e->ptr : (any_float ? (any_f64 ? e->f64 : e->f32) : (any_i64 ? e->i64 : e->i32));

            // Alloca in current (pre-branch) block.
            e->cur_block = save; e->terminated = false;
            MLIR_ValueHandle slot = emit_alloca(e, rty);

            // Coerce + store in then-tail.
            e->cur_block = then_tail; e->terminated = false;
            MLIR_ValueHandle ta = any_ptr
                ? int_to_ptr(e, av)
                : coerce_eval_to(e, av, rty, (Type){0});
            emit_store_v(e, ta, slot);
            emit_branch(e, merge_blk);

            e->cur_block = else_tail; e->terminated = false;
            MLIR_ValueHandle tb = any_ptr
                ? int_to_ptr(e, bv)
                : coerce_eval_to(e, bv, rty, (Type){0});
            emit_store_v(e, tb, slot);
            emit_branch(e, merge_blk);

            // From the original block, branch on cb to then/else.
            e->cur_block = save; e->terminated = false;
            emit_cond_branch(e, cb, then_blk, else_blk);

            e->cur_block = merge_blk; e->terminated = false;
            r.val = emit_load_v(e, slot, rty);
            r.is_float = any_float;
            r.is_f64   = any_f64;
            r.is_ptr   = any_ptr;
            r.is_i64   = any_i64;
            // Integer branches: C's usual arithmetic conversions decide the
            // signedness (the 64-bit operand's when the widths differ).
            if (!any_ptr && !any_float) {
                r.is_unsigned = (av.is_i64 == bv.is_i64)
                    ? (av.is_unsigned || bv.is_unsigned)
                    : (av.is_i64 ? av.is_unsigned : bv.is_unsigned);
            }
            r.is_str   = av.is_str && bv.is_str;
            r.sdef     = av.sdef ? av.sdef : bv.sdef;
            // Keep the pointee type of `c ? "ab" : s` so pointer arithmetic
            // on the result steps by the element size. One side may be a
            // null pointer constant (no element type); two different element
            // types leave it unknown.
            if (any_ptr) {
                MLIR_TypeHandle ae = av.is_ptr ? av.ptr_elem : MLIR_INVALID_HANDLE;
                MLIR_TypeHandle be = bv.is_ptr ? bv.ptr_elem : MLIR_INVALID_HANDLE;
                if (ae == MLIR_INVALID_HANDLE) r.ptr_elem = be;
                else if (be == MLIR_INVALID_HANDLE || be == ae) r.ptr_elem = ae;
            }
            return r;
        }
        case EX_SIZEOF: {
            int64_t n = 1;
            if (ex->sizeof_is_expr) {
                if (ex->lhs && ex->lhs->kind == EX_STR) {
                    n = ex->is_alignof ? 1 : (int64_t)ex->lhs->name.size;
                } else if (infer_expr_type(e, sc, ex->lhs).kind == TY_VOID) {
                    EMIT_ERR(e, "sizeof of unsupported expression");
                } else {
                    n = ex->is_alignof ? type_align(e, infer_expr_type(e, sc, ex->lhs))
                                       : c_sizeof_expr(e, sc, ex->lhs);
                }
            } else if (!ex->cast_type || ex->cast_type->kind == TY_VOID) {
                EMIT_ERR(e, "sizeof(void) is not allowed");
            } else {
                n = ex->is_alignof ? type_align(e, *ex->cast_type)
                                   : type_size(e, *ex->cast_type);
            }
            // The result is a size_t (`unsigned long`): unsigned, and 64-bit
            // except on wasm32 (as in ast_fold and infer_expr_type).
            r.is_unsigned = true;
            r.is_i64 = !e->target_wasm32;
            r.val = r.is_i64 ? emit_const_i64(e, n) : emit_const_i32(e, n);
            return r;
        }
        case EX_CAST: {
            EVal v = emit_expr(e, sc, ex->lhs);
            TypeKind ck = ex->cast_type->kind;
            bool ck_is_ptr = (ck == TY_PTR_VOID || ck == TY_PTR_I32 ||
                              ck == TY_PTR_CHAR || ck == TY_PTR_STRUCT ||
                              ck == TY_FNPTR || ck == TY_PTR_PTR);
            // A cast to bool yields 0 or 1 (from an integer, float or
            // pointer operand).
            if (!ck_is_ptr && ex->cast_type->is_bool) {
                r.val = emit_bool_i32(e, v);
                return r;
            }
            // Integer-to-integer cast: convert between i32 and i64; everything
            // else is treated as i32 (narrow types are i32 values, truncated
            // and re-extended to their width below).
            if (!ck_is_ptr && !v.is_ptr) {
                // float-target casts (from int or float).
                if (ck == TY_F64) {
                    v.val = coerce_eval_to(e, v, e->f64, (Type){0});
                    v.is_float = true; v.is_f64 = true; v.is_i64 = false;
                    return v;
                }
                if (ck == TY_F32) {
                    v.val = coerce_eval_to(e, v, e->f32, (Type){0});
                    v.is_float = true; v.is_f64 = false; v.is_i64 = false;
                    return v;
                }
                // int-target casts.
                if (v.is_float) {
                    // float -> int: fptosi / fptoui straight to the
                    // target width, by the target's signedness.
                    v.is_i64 = (ck == TY_I64);
                    // (A char / short target converts through a signed i32,
                    // as in coerce_eval_to.)
                    v.val = emit_fptoi(e, v.val, v.is_i64 ? e->i64 : e->i32,
                                       ex->cast_type->int_unsigned &&
                                       !type_is_narrow_int(*ex->cast_type));
                    // Keep an out-of-range value inside the narrow type.
                    int fnb = ex->cast_type->int_bits;
                    if (fnb == 8 || fnb == 16)
                        v.val = emit_narrow_i32(e, v.val, fnb, ex->cast_type->int_unsigned);
                    // A char / short result is promoted to (signed) int.
                    v.is_unsigned = ex->cast_type->int_unsigned &&
                                    !type_is_narrow_int(*ex->cast_type);
                    v.is_float = false; v.is_f64 = false;
                    return v;
                }
                bool want_i64 = (ck == TY_I64);
                if (want_i64 && !v.is_i64) {
                    // Use unsigned widening when the source carried an
                    // `unsigned` C type — otherwise an i32 value with
                    // its top bit set (e.g. `(int64_t)<uint32_t>` with
                    // bit 31 set) sign-extends to a huge "negative"
                    // i64. The widening decision uses the SOURCE
                    // signedness; the result's signedness is the cast
                    // target's (set below).
                    v.val = v.is_unsigned ? emit_extui_i32_to_i64(e, v.val)
                                          : emit_extsi_i32_to_i64(e, v.val);
                    v.is_i64 = true;
                } else if (!want_i64 && v.is_i64) {
                    v.val = emit_trunci_i64_to_i32(e, v.val);
                    v.is_i64 = false;
                }
                // char / short: keep the low 8 / 16 bits, zero- or
                // sign-extended by the target's signedness.
                int nb = ex->cast_type->int_bits;
                if (!want_i64 && (nb == 8 || nb == 16)) {
                    v.val = emit_narrow_i32(e, v.val, nb, ex->cast_type->int_unsigned);
                }
                // A cast to an unsigned integer type yields an unsigned
                // value: subsequent comparisons / div / shr must use the
                // unsigned form (e.g. `(unsigned long)x >= CONST`). A char /
                // short result is promoted to (signed) int.
                v.is_unsigned = ex->cast_type->int_unsigned &&
                                !type_is_narrow_int(*ex->cast_type);
                return v;
            }
            // Pointer-to-integer cast: emit llvm.ptrtoint, then optionally
            // truncate to i32. Used for `(uintptr_t)p` / `(long)p` / etc.
            if (!ck_is_ptr && v.is_ptr) {
                MLIR_ValueHandle iv = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i64;
                MLIR_ValueHandle *rs = &iv;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v.val;
                emit_op(e, OP_TYPE_LLVM_PTRTOINT,
                        rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
                v.val = iv;
                v.is_ptr = false;
                v.is_void_ptr = false;
                v.sdef = NULL;
                v.is_str = false;
                if (ck == TY_I64) {
                    v.is_i64 = true;
                } else {
                    v.val = emit_trunci_i64_to_i32(e, v.val);
                    v.is_i64 = false;
                }
                // `(uintptr_t)p` / `(unsigned long)p` produce an unsigned
                // integer: select unsigned comparisons / div / shr below.
                v.is_unsigned = ex->cast_type->int_unsigned;
                return v;
            }
            // Pointer-to-pointer cast: opaque !llvm.ptr is universal, so
            // just evaluate the operand. We tag the result type from
            // cast_type for downstream consumers.
            if (!v.is_ptr) {
                // Integer-to-pointer cast: literal 0 -> null pointer
                // (used for `(T*)0` null sentinels in C). Any other
                // integer operand is materialized via llvm.inttoptr
                // (used in corec for `(struct buddy_block *)addr` where
                // `addr` is a uintptr_t).
                if (ex->lhs->kind == EX_INT && ex->lhs->int_value == 0) {
                    v.val = emit_null_ptr(e);
                } else if (!v.is_float) {
                    // Ensure the operand is i64 for inttoptr.
                    MLIR_ValueHandle iv = v.val;
                    if (!v.is_i64) {
                        iv = emit_extsi_i32_to_i64(e, iv);
                    }
                    v.val = emit_inttoptr_i64(e, iv);
                    v.is_i64 = false;
                } else {
                    EMIT_ERR(e, "cast operand is not a pointer");
                }
            }
            v.is_ptr = true;
            v.is_float = false;
            v.is_void_ptr = (ck == TY_PTR_VOID);
            if (ck == TY_PTR_STRUCT) {
                v.sdef = find_struct(e, ex->cast_type->struct_name);
            } else {
                v.sdef = NULL;
            }
            // The element type comes from the target type alone: a
            // `(long long *)vp` steps by 8, a `(struct S *)charp` by the
            // struct size (ptr_elem unset, sdef set), not by the
            // operand's element size.
            v.ptr_elem = ptr_elem_of_type(e, ex->cast_type);
            v.is_str = (ck == TY_PTR_CHAR);
            v.is_unsigned = ex->cast_type->int_unsigned;
            return v;
        }
        case EX_VAR:
        case EX_INDEX:
        case EX_DEREF:
        case EX_FIELD: {
            // Bare function name (no call): decay to a function-pointer
            // value via `llvm.mlir.addressof @sym : !llvm.ptr`. Tag the
            // result with a TY_FNPTR carrying the function's signature so
            // downstream call sites can build the right call_indirect.
            if (ex->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->name);
                if (!s) {
                    FuncSig *fsig = find_sig(e, ex->name);
                    if (fsig) {
                        // Build the function MLIR type for func.constant
                        // using the flat (sret/void-aware) signature so it
                        // matches the actual func.func definition.
                        MLIR_TypeHandle f_ty = MLIR_CreateTypeFunction(e->ctx,
                            fsig->flat_in_tys, fsig->n_flat_in,
                            fsig->flat_out_tys, fsig->n_flat_out);
                        // %fn = func.constant @sym : (T...) -> R
                        MLIR_ValueHandle fn_v = MLIR_INVALID_HANDLE;
                        MLIR_TypeHandle *frts = arena_new_array(e->arena,
                            MLIR_TypeHandle, 1); frts[0] = f_ty;
                        MLIR_ValueHandle *frs = &fn_v;
                        MLIR_NamedAttr va = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeSymbolRef(
                            e->ctx, fsig->name));
                        MLIR_NamedAttr *fas = arena_new_array(e->arena,
                            MLIR_NamedAttr, 1); fas[0] = va;
                        emit_op(e, OP_TYPE_FUNC_CONSTANT,
                                frts, 1, frs, 1, NULL, 0, fas, 1, NULL, 0);
                        // Cast (T...) -> R into !llvm.ptr for uniform storage.
                        MLIR_ValueHandle ptr_v = MLIR_INVALID_HANDLE;
                        MLIR_TypeHandle *crts = arena_new_array(e->arena,
                            MLIR_TypeHandle, 1); crts[0] = e->ptr;
                        MLIR_ValueHandle *crs = &ptr_v;
                        MLIR_ValueHandle *cops = arena_new_array(e->arena,
                            MLIR_ValueHandle, 1); cops[0] = fn_v;
                        emit_op(e, OP_TYPE_UNREALIZED_CONVERSION_CAST,
                                crts, 1, crs, 1, cops, 1, NULL, 0, NULL, 0);
                        r.val = ptr_v;
                        r.is_ptr = true;
                        Type *fnty = arena_new(e->arena, Type);
                        *fnty = (Type){0};
                        fnty->kind = TY_FNPTR;
                        Type *ret_box = arena_new(e->arena, Type);
                        *ret_box = fsig->ret.type;
                        fnty->fnptr_ret = ret_box;
                        fnty->fnptr_nparams = (int)fsig->n_params;
                        fnty->fnptr_params = arena_new_array(e->arena, Type,
                            fnty->fnptr_nparams ? (size_t)fnty->fnptr_nparams : 1);
                        for (size_t i = 0; i < fsig->n_params; i++)
                            fnty->fnptr_params[i] = fsig->params[i].type;
                        r.fnptr_ty = fnty;
                        return r;
                    }
                    // Fall back to enum constants ONLY when both local
                    // scope and the function table miss — this preserves
                    // shadowing by inner-scope variables.
                    int64_t en_v;
                    if (find_enum(e, ex->name, &en_v)) {
                        r.val = emit_const_i32(e, en_v);
                        return r;
                    }
                }
            }
            // Array-to-pointer decay: when an array variable appears as an
            // rvalue (e.g. passed to a pointer-typed function parameter), it
            // decays to a pointer to its first element. The alloca already
            // returns !llvm.ptr; just yield that address.
            if (ex->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->name);
                if (s && (s->type.kind == TY_ARRAY_I32 ||
                          s->type.kind == TY_ARRAY_F32 ||
                          s->type.kind == TY_ARRAY_STRUCT ||
                          s->type.kind == TY_ARRAY_PTR_STRUCT ||
                          s->type.kind == TY_ARRAY_PTR_CHAR)) {
                    r.val = sym_addr(e, s);
                    r.is_ptr = true;
                    if (s->type.kind == TY_ARRAY_STRUCT) {
                        // A struct array steps by the struct (sdef set,
                        // ptr_elem unset).
                        if (s->sdef) r.sdef = s->sdef;
                        else r.ptr_elem = e->i32;
                    } else {
                        r.ptr_elem = array_elem_type(e, &s->type);
                    }
                    return r;
                }
                // va_list value is the buffer pointer itself (the alloca
                // address). Passing va_list as an argument or to va_arg
                // wants this !llvm.ptr, not a load of its content.
                if (s && s->type.kind == TY_VA_LIST) {
                    r.val = sym_addr(e, s);
                    r.is_ptr = true;
                    return r;
                }
                // Direct fast path for scalar/pointer variables (both direct-SSA
                // and mutable alloca/global slots): avoids emit_lvalue + load_lvalue
                // + 2 redundant scope lookups.
                if (s && s->type.kind != TY_STRUCT) {
                    TypeKind tk = s->type.kind;
                    if (s->is_ssa) {
                        r.val = s->addr;
                    } else if (tk == TY_I64) {
                        r.val = emit_load_v(e, sym_addr(e, s), e->i64);
                    } else if (tk == TY_F32) {
                        r.val = emit_load_v(e, sym_addr(e, s), e->f32);
                    } else if (tk == TY_F64) {
                        r.val = emit_load_v(e, sym_addr(e, s), e->f64);
                    } else if (tk == TY_I32) {
                        MLIR_TypeHandle mt = s->wide_slot ? e->i32 : scalar_mem_type(e, &s->type);
                        MLIR_ValueHandle raw = emit_load_v(e, sym_addr(e, s), mt);
                        if (mt == e->i8 || mt == e->i16) {
                            r.val = (s->type.int_unsigned || s->type.is_bool)
                                ? emit_extui_to_i32(e, raw)
                                : emit_extsi_to_i32(e, raw);
                        } else {
                            r.val = raw;
                        }
                    } else {
                        r.val = emit_load_v(e, sym_addr(e, s), e->ptr);
                    }
                    if (tk == TY_I64) {
                        r.is_i64 = true;
                        r.is_unsigned = s->type.int_unsigned;
                    } else if (tk == TY_F32) {
                        r.is_float = true;
                    } else if (tk == TY_F64) {
                        r.is_float = true;
                        r.is_f64 = true;
                    } else if (tk == TY_I32) {
                        r.is_unsigned = s->type.int_unsigned && !type_is_narrow_int(s->type);
                    } else {
                        r.is_ptr = true;
                        r.sdef = s->sdef;
                        if (tk == TY_PTR_CHAR) { r.is_str = true; r.ptr_elem = e->i8; }
                        else if (tk == TY_PTR_I32) r.ptr_elem = ptr_elem_of_type(e, &s->type);
                        else if (tk == TY_PTR_VOID) r.is_void_ptr = true;
                        else if (tk == TY_PTR_PTR) {
                            r.ptr_elem = e->ptr;
                            if (s->type.pointee) {
                                Type *pe = s->type.pointee;
                                if (pe->kind == TY_PTR_CHAR) r.is_str = true;
                                else if (pe->kind == TY_PTR_STRUCT) {
                                    r.sdef = find_struct(e, pe->struct_name);
                                } else if (pe->kind == TY_FNPTR) {
                                    Type *fnty = arena_new(e->arena, Type);
                                    *fnty = *pe;
                                    r.fnptr_ty = fnty;
                                }
                            }
                        } else if (tk == TY_FNPTR) {
                            Type *fnty = arena_new(e->arena, Type);
                            *fnty = s->type;
                            r.fnptr_ty = fnty;
                        }
                    }
                    return r;
                }
            }
            // Array-to-pointer decay for a struct field whose declared
            // type is an array (e.g. `struct S { uint8_t data[64]; };
            // ... s.data + i`). The rvalue is a pointer to the array's
            // first element; carrying the right ptr_elem lets downstream
            // pointer-arithmetic / indexing emit the correct stride.
            //
            // The field type is looked up without emitting code first, so that the
            // struct base is walked (= its code emitted, side effects included)
            // exactly once: here for an array field, or in emit_lvalue below for a
            // scalar one. Walking it "just to look" used to evaluate the base of
            // `stack[--sp].bid` twice (three times for a pointer field).
            Type field_ty = (Type){0};
            bool field_ty_known = ex->kind == EX_FIELD &&
                                  field_type_noemit(e, sc, ex, &field_ty);
            bool field_is_array = field_ty.kind == TY_ARRAY_I32 ||
                                  field_ty.kind == TY_ARRAY_F32 ||
                                  field_ty.kind == TY_ARRAY_STRUCT ||
                                  field_ty.kind == TY_ARRAY_PTR_STRUCT ||
                                  field_ty.kind == TY_ARRAY_PTR_CHAR;
            if (ex->kind == EX_FIELD && (!field_ty_known || field_is_array)) {
                SCtx parent = walk_struct_lhs(e, sc, ex->lhs);
                if (parent.ok) {
                    int fidx = struct_field_index(parent.sd, ex->name);
                    if (fidx >= 0) {
                        Type ft = parent.sd->fields.data[fidx].type;
                        if (ft.kind == TY_ARRAY_I32 ||
                            ft.kind == TY_ARRAY_F32 ||
                            ft.kind == TY_ARRAY_STRUCT ||
                            ft.kind == TY_ARRAY_PTR_STRUCT ||
                            ft.kind == TY_ARRAY_PTR_CHAR) {
                            sctx_push(e, &parent,
                                      parent.sd->is_union ? 0 : fidx);
                            MLIR_ValueHandle dyn[1];
                            size_t n_dyn = 0;
                            if (parent.dyn_index != MLIR_INVALID_HANDLE) {
                                dyn[0] = parent.dyn_index;
                                n_dyn = 1;
                            }
                            MLIR_ValueHandle p = emit_gep(e,
                                parent.base_ptr, parent.source_elem,
                                parent.const_path, parent.n_const_path,
                                dyn, n_dyn);
                            r.val = p;
                            r.is_ptr = true;
                            if (ft.kind == TY_ARRAY_STRUCT &&
                                find_struct(e, ft.struct_name)) {
                                // A struct array steps by the struct
                                // (sdef set, ptr_elem unset).
                                r.sdef = find_struct(e, ft.struct_name);
                            } else {
                                r.ptr_elem = ft.kind == TY_ARRAY_STRUCT
                                    ? e->i32 : array_elem_type(e, &ft);
                            }
                            return r;
                        }
                    }
                }
            }
            LVal lv = emit_lvalue(e, sc, ex);
            EVal v = load_lvalue(e, lv);
            // Tag the resulting pointer with its target StructDef, when
            // determinable, so callers (struct-arg passing, pointer
            // comparisons) can reason about it.
            if (v.is_ptr) {
                if (ex->kind == EX_VAR) {
                    Sym *s = lookup(e, sc, ex->name);
                    if (s) {
                        v.sdef = s->sdef;
                        if (s->type.kind == TY_PTR_CHAR) { v.is_str = true; v.ptr_elem = e->i8; }
                        else if (s->type.kind == TY_PTR_I32)
                            v.ptr_elem = ptr_elem_of_type(e, &s->type);
                        else if (s->type.kind == TY_PTR_VOID) v.is_void_ptr = true;
                        else if (s->type.kind == TY_PTR_PTR) {
                            // Reading the value of a T** yields a T* — tag
                            // the EVal so a subsequent dereference can load
                            // the inner T at the right element type. The
                            // pointee field carries the inner pointer's
                            // element kind.
                            v.ptr_elem = e->ptr;
                            if (s->type.pointee) {
                                Type *pe = s->type.pointee;
                                if (pe->kind == TY_PTR_CHAR) v.is_str = true;
                                else if (pe->kind == TY_PTR_STRUCT) {
                                    v.sdef = find_struct(e, pe->struct_name);
                                } else if (pe->kind == TY_FNPTR) {
                                    Type *fnty = arena_new(e->arena, Type);
                                    *fnty = *pe;
                                    v.fnptr_ty = fnty;
                                }
                            }
                        }
                        else if (s->type.kind == TY_FNPTR) {
                            Type *fnty = arena_new(e->arena, Type);
                            *fnty = s->type;
                            v.fnptr_ty = fnty;
                        }
                    }
                } else if (ex->kind == EX_DEREF && ex->lhs->kind == EX_VAR) {
                    // *pp where pp is T**: the loaded value is a T*. Tag
                    // it from the pointee so further reads / calls work.
                    Sym *s = lookup(e, sc, ex->lhs->name);
                    if (s && s->type.kind == TY_PTR_PTR && s->type.pointee)
                        v = ptr_eval_of_type(e, v.val, s->type.pointee);
                } else if (ex->kind == EX_FIELD) {
                    // Prefer the no-code lookup: walking the struct base again here
                    // would evaluate it (and its side effects) a second time.
                    Type ft = field_ty;
                    bool have_ft = field_ty_known;
                    if (!have_ft) {
                        SCtx parent = walk_struct_lhs(e, sc, ex->lhs);
                        if (parent.ok) {
                            int fidx = struct_field_index(parent.sd, ex->name);
                            if (fidx >= 0) {
                                ft = parent.sd->fields.data[fidx].type;
                                have_ft = true;
                            }
                        }
                    }
                    if (have_ft) {
                        if (ft.kind == TY_PTR_PTR) {
                            v.ptr_elem = e->ptr;
                            if (ft.pointee) {
                                Type *pe = ft.pointee;
                                if (pe->kind == TY_PTR_STRUCT) {
                                    v.sdef = find_struct(e, pe->struct_name);
                                } else if (pe->kind == TY_FNPTR) {
                                    Type *fnty = arena_new(e->arena, Type);
                                    *fnty = *pe;
                                    v.fnptr_ty = fnty;
                                }
                            }
                        } else if (type_kind_is_ptr(ft.kind)) {
                            v = ptr_eval_of_type(e, v.val, &ft);
                        }
                    }
                } else if (ex->kind == EX_INDEX) {
                    // Reading an element of a struct-field array of
                    // pointers: tag the result so downstream consumers
                    // (-> chains) see the right element kind.
                    Expr *fe = NULL;
                    if (ex->lhs->kind == EX_FIELD) fe = ex->lhs;
                    else if (ex->lhs->kind == EX_INDEX &&
                             ex->lhs->lhs->kind == EX_FIELD) fe = ex->lhs->lhs;
                    if (fe) {
                        Type fe_ty = (Type){0};
                        bool fe_ty_known = field_type_noemit(e, sc, fe, &fe_ty);
                        // Prefer the no-code lookup: walking the struct base again here
                        // would evaluate it (and its side effects) a second time.
                        Type ft = fe_ty;
                        bool have_ft = fe_ty_known;
                        if (!have_ft) {
                            SCtx parent = walk_struct_lhs(e, sc, fe->lhs);
                            if (parent.ok) {
                                int fidx = struct_field_index(parent.sd, fe->name);
                                if (fidx >= 0) {
                                    ft = parent.sd->fields.data[fidx].type;
                                    have_ft = true;
                                }
                            }
                        }
                        if (have_ft) {
                            if ((ft.kind == TY_ARRAY_PTR_CHAR ||
                                 ft.kind == TY_ARRAY_PTR_STRUCT) && ft.pointee) {
                                // An element of a field `T *f[N]`: a T *.
                                v = ptr_eval_of_type(e, v.val, ft.pointee);
                            } else if (ft.kind == TY_ARRAY_PTR_CHAR) {
                                v.is_str = true;
                                v.ptr_elem = e->i8;
                            } else if (ft.kind == TY_ARRAY_PTR_STRUCT) {
                                v.sdef = find_struct(e, ft.struct_name);
                            } else if (ft.kind == TY_PTR_PTR && ft.pointee) {
                                // Indexing a struct field of type T**:
                                // the loaded element is a single T*.
                                // Tag with the inner type so '->'
                                // chains and string ops work.
                                v = ptr_eval_of_type(e, v.val, ft.pointee);
                            }
                        }
                    }
                    // Local variable indexed by a single subscript:
                    //   `a[i]` where `a` is `char **` / `char *arr[N]` /
                    //   `struct T *arr[N]`. The loaded element is a single
                    //   T*; without tagging `v.ptr_elem`, subsequent
                    //   pointer arithmetic (`a[i] + n`) falls back to the
                    //   `e->i32` default and GEPs with the wrong stride
                    //   (4 bytes per `n` instead of 1 byte per `n` for
                    //   char*) — observed during stage-3 self-host as the
                    //   driver.c `--export=NAME` parser advancing 36
                    //   bytes past the prefix instead of 9.
                    if (ex->lhs->kind == EX_VAR) {
                        Sym *vs = lookup(e, sc, ex->lhs->name);
                        if (vs) {
                            if (vs->type.kind == TY_PTR_PTR && vs->type.pointee) {
                                v = ptr_eval_of_type(e, v.val, vs->type.pointee);
                            } else if ((vs->type.kind == TY_ARRAY_PTR_CHAR ||
                                        vs->type.kind == TY_ARRAY_PTR_STRUCT) &&
                                       vs->type.pointee) {
                                // An element of `T *arr[N]`: a T *.
                                v = ptr_eval_of_type(e, v.val, vs->type.pointee);
                            } else if (vs->type.kind == TY_ARRAY_PTR_CHAR) {
                                v.is_str = true;
                                v.ptr_elem = e->i8;
                            } else if (vs->type.kind == TY_ARRAY_PTR_STRUCT) {
                                v.sdef = find_struct(e, vs->type.struct_name);
                            }
                        }
                    }
                }
                // Any other loaded pointer (e.g. `*pp++` or `*(pp + 1)` with
                // a `char **pp`): the expression's static type gives the
                // pointee.
                if (v.ptr_elem == MLIR_INVALID_HANDLE && !v.sdef && !v.fnptr_ty &&
                    !v.is_void_ptr) {
                    Type t = infer_expr_type(e, sc, ex);
                    if (type_kind_is_ptr(t.kind)) v = ptr_eval_of_type(e, v.val, &t);
                }
            }
            return v;
        }
        case EX_ADDR: {
            // &x: result is the !llvm.ptr to x's storage. Used as the
            // RHS of `int* p = &x;` style decls and for &a[0]; treated as
            // a pseudo-rvalue whose `val` is a !llvm.ptr SSA value.
            if (ex->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->name);
                if (!s) {
                    // &funcname: explicit address-of-function. Same as
                    // bare function-name decay.
                    FuncSig *fsig = find_sig(e, ex->lhs->name);
                    if (fsig) {
                        return emit_expr(e, sc, ex->lhs);
                    }
                    EMIT_ERR(e, "undefined variable: {}", ex->lhs->name);
                    return r;
                }
                r.val = sym_addr(e, s);
                r.is_ptr = true;
                if (s->type.kind == TY_I32) r.ptr_elem = int_mem_type(e, &s->type);
                else if (s->type.kind == TY_ARRAY_I32) r.ptr_elem = array_elem_type(e, &s->type);
                else if (s->type.kind == TY_I64) r.ptr_elem = e->i64;
                else if (s->type.kind == TY_F32) r.ptr_elem = e->f32;
                else if (s->type.kind == TY_F64) r.ptr_elem = e->f64;
                else if (s->type.kind == TY_PTR_I32 || s->type.kind == TY_PTR_CHAR ||
                         s->type.kind == TY_PTR_VOID || s->type.kind == TY_PTR_STRUCT ||
                         s->type.kind == TY_FNPTR || s->type.kind == TY_PTR_PTR) {
                    // &p where p is itself a pointer: yield a T** (storage
                    // !llvm.ptr) whose deref loads another !llvm.ptr.
                    r.ptr_elem = e->ptr;
                }
                else if (s->type.kind == TY_STRUCT) {
                    // &s where s is a struct value: the result is a real
                    // struct pointer. Carry the struct sdef + element type
                    // so downstream `(&s)->field` accesses (and the general
                    // `EX_DEREF` walk) can resolve the field chain.
                    r.sdef = s->sdef;
                    if (s->sdef) r.ptr_elem = find_struct_type(e, s->sdef);
                }
                return r;
            }
            if (ex->lhs->kind == EX_INDEX) {
                // &p->arr[i] / &(s.f)[i]: address of a struct element via
                // a chain. Use resolve_struct_source so the GEP descends
                // through the arrow/field chain instead of expecting a
                // simple-array base.
                Type lt = infer_expr_type(e, sc, ex->lhs);
                if (lt.kind == TY_STRUCT) {
                    StructDef *sd = find_struct(e, lt.struct_name);
                    if (sd) {
                        StructDef *src_sd = NULL;
                        MLIR_ValueHandle p = resolve_struct_source(
                            e, sc, ex->lhs, &src_sd);
                        if (p != MLIR_INVALID_HANDLE && src_sd == sd) {
                            r.val = p;
                            r.is_ptr = true;
                            r.sdef = sd;
                            return r;
                        }
                    }
                }
                // &arr[i] -> GEP address.
                LVal lv = emit_lvalue(e, sc, ex->lhs);
                r.val = lval_address(e, lv);
                r.is_ptr = true;
                r.ptr_elem = lv.elem_ty;
                return r;
            }
            // &<struct field>: take the address of any struct lvalue
            // (including struct-typed fields like `&buckets[i].value`).
            // For struct-valued lvalues we use walk_struct_lhs + GEP.
            if (ex->lhs->kind == EX_FIELD || ex->lhs->kind == EX_DEREF) {
                Type lt = infer_expr_type(e, sc, ex->lhs);
                if (lt.kind == TY_STRUCT) {
                    StructDef *sd = find_struct(e, lt.struct_name);
                    if (sd) {
                        StructDef *src_sd = NULL;
                        MLIR_ValueHandle p = resolve_struct_source(
                            e, sc, ex->lhs, &src_sd);
                        if (p != MLIR_INVALID_HANDLE && src_sd == sd) {
                            r.val = p;
                            r.is_ptr = true;
                            r.sdef = sd;
                            return r;
                        }
                    }
                }
                // Scalar field: fall back to emit_lvalue.
                LVal lv = emit_lvalue(e, sc, ex->lhs);
                r.val = lval_address(e, lv);
                r.is_ptr = true;
                r.ptr_elem = lv.elem_ty;
                return r;
            }
            EMIT_ERR(e, "unsupported & operand");
            return r;
        }
        case EX_ASSIGN: {
            // Struct rhs from a struct-returning call: q = f(args);
            if (ex->lvalue->kind == EX_VAR && ex->rhs_assign->kind == EX_CALL) {
                Sym *target = lookup(e, sc, ex->lvalue->name);
                FuncSig *sig = find_sig(e, ex->rhs_assign->callee);
                if (target && target->type.kind == TY_STRUCT && target->sdef &&
                    sig && sig->ret.type.kind == TY_STRUCT &&
                    sig->ret.sdef == target->sdef) {
                    // Pass &q as the sret slot; no extra copy needed.
                    emit_flat_call(e, sc, sig, ex->rhs_assign->args, NULL, sym_addr(e, target));
                    return r;
                }
            }
            // Struct-typed assignment: `lhs = rhs;` where both sides are
            // struct lvalues (e.g. `buckets[i].value = existing_value;`).
            // Resolve both addresses and emit a struct copy.
            {
                Type lt = infer_expr_type(e, sc, ex->lvalue);
                if (lt.kind == TY_STRUCT) {
                    StructDef *sd = find_struct(e, lt.struct_name);
                    if (sd) {
                        StructDef *src_sd = NULL;
                        StructDef *dst_sd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(
                            e, sc, ex->rhs_assign, &src_sd);
                        MLIR_ValueHandle dst = resolve_struct_source(
                            e, sc, ex->lvalue, &dst_sd);
                        if (src != MLIR_INVALID_HANDLE &&
                            dst != MLIR_INVALID_HANDLE &&
                            src_sd == sd && dst_sd == sd) {
                            emit_struct_copy(e, dst, src, sd);
                            r.is_ptr = true; r.sdef = sd; r.val = dst;
                            return r;
                        }
                    }
                }
            }
            LVal lv = emit_lvalue(e, sc, ex->lvalue);
            if (ex->is_post_step) {
                EVal old = load_lvalue(e, lv);
                bool sub = (ex->rhs_assign && ex->rhs_assign->op == OP_SUB);
                if (lv.elem_ty == e->ptr) {
                    // Pointer postfix `p++` / `p--`: step by sizeof(*p)
                    // via GEP, store new pointer, return the OLD pointer.
                    Type lt = infer_expr_type(e, sc, ex->lvalue);
                    old = ptr_eval_of_type(e, old.val, &lt);
                    MLIR_TypeHandle pe = old.ptr_elem;
                    if (pe == MLIR_INVALID_HANDLE)
                        pe = old.sdef ? find_struct_type(e, old.sdef) : e->i32;
                    int32_t step = sub ? -1 : 1;
                    MLIR_ValueHandle idx = emit_const_i32(e, step);
                    int32_t *path = arena_new_array(e->arena, int32_t, 1);
                    path[0] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn[0] = idx;
                    MLIR_ValueHandle stepped = emit_gep(e, old.val, pe, path, 1, dyn, 1);
                    store_lvalue(e, lv, stepped);
                    old.is_ptr = true;
                    return old;
                }
                MLIR_TypeHandle ity = old.is_i64 ? e->i64 : e->i32;
                MLIR_ValueHandle one_v = old.is_i64 ? emit_const_i64(e, 1)
                                                    : emit_const_i32(e, 1);
                MLIR_OpType opty = sub ? OP_TYPE_ARITH_SUBI : OP_TYPE_ARITH_ADDI;
                MLIR_ValueHandle stepped = emit_binop(e, opty, ity,
                                                       old.val, one_v);
                MLIR_ValueHandle stored = coerce_eval_to(e,
                    (EVal){.val = stepped, .is_i64 = old.is_i64}, lv.elem_ty,
                    lval_ctype(lv));
                store_lvalue(e, lv, stored);
                return old;
            }
            EVal v;
            if (ex->is_compound) {
                // `lhs op= rhs` / `++lhs`: rhs_assign is `lhs op rhs` with the
                // same `lhs` node. Resolve the address once and let the read
                // of `lhs` inside rhs_assign reuse it (see emit_lvalue).
                LVal at = lv;
                at.base_ptr = lval_address(e, lv);
                at.source_elem = MLIR_INVALID_HANDLE;
                at.const_path = NULL;
                at.n_const_path = 0;
                at.dyn_index = MLIR_INVALID_HANDLE;
                lv = at;
                Expr *saved_expr = e->compound_lv_expr;
                LVal *saved_lv = e->compound_lv;
                e->compound_lv_expr = ex->lvalue;
                e->compound_lv = &at;
                v = emit_expr(e, sc, ex->rhs_assign);
                e->compound_lv_expr = saved_expr;
                e->compound_lv = saved_lv;
            } else {
                v = emit_expr(e, sc, ex->rhs_assign);
            }
            v.val = coerce_eval_to(e, v, lv.elem_ty, lval_ctype(lv));
            // `p = x` has p's type, not x's: `q = p = i` and `if ((p = i))`
            // must not convert the result again, and `(p = x)[1]` / `+ 1`
            // step by p's element.
            if (lv.elem_ty == e->ptr) {
                Type lt = infer_expr_type(e, sc, ex->lvalue);
                if (type_kind_is_ptr(lt.kind)) v = ptr_eval_of_type(e, v.val, &lt);
                else v.is_ptr = true;
            }
            bool narrow = lv.elem_ty == e->i8 || lv.elem_ty == e->i16;
            // The assignment's value has the lvalue's (promoted) type.
            if (lv.elem_ty == e->i32 || lv.elem_ty == e->i64 || narrow)
                v.is_unsigned = lv.is_unsigned && !narrow && !lv.narrow_bits;
            v.is_float = (lv.elem_ty == e->f32 || lv.elem_ty == e->f64);
            v.is_f64 = (lv.elem_ty == e->f64);
            v.is_i64 = (lv.elem_ty == e->i64);
            store_lvalue(e, lv, v.val);
            if (narrow && e->unused_assign != ex) {
                // The stored value is i8 / i16, but the EVal we hand back to
                // the surrounding expression is the promoted i32. Match the
                // load path: zero-extend an unsigned type or bool,
                // sign-extend otherwise.
                bool zext = lv.is_unsigned || lv.is_bool;
                v.val = zext ? emit_extui_to_i32(e, v.val) : emit_extsi_to_i32(e, v.val);
                fact_put(e, v.val, FACT_RANGE,
                         lv.is_bool ? 1 : lv.elem_ty == e->i8 ? 8 : 16, zext, 0);
            }
            return v;
        }
        case EX_UN: {
            EVal a = emit_expr(e, sc, ex->lhs);
            if (ex->op == OP_NEG) {
                if (a.is_float) {
                    MLIR_TypeHandle ft = a.is_f64 ? e->f64 : e->f32;
                    r.val = emit_fneg(e, ft, a.val);
                    r.is_float = true; r.is_f64 = a.is_f64;
                } else {
                    MLIR_ValueHandle z = a.is_i64 ? emit_const_i64(e, 0) : emit_const_i32(e, 0);
                    MLIR_TypeHandle ity = a.is_i64 ? e->i64 : e->i32;
                    r.val = emit_binop(e, OP_TYPE_ARITH_SUBI, ity, z, a.val);
                    r.is_i64 = a.is_i64;
                    r.is_unsigned = a.is_unsigned;
                }
                return r;
            }
            if (ex->op == OP_BNOT) {
                if (a.is_float) {
                    EMIT_ERR(e, "~ not supported on floats");
                    r.val = emit_const_i32(e, 0);
                    return r;
                }
                if (a.is_i64) {
                    MLIR_ValueHandle ones = emit_const_i64(e, -1);
                    r.val = emit_binop(e, OP_TYPE_ARITH_XORI,
                                       e->i64, a.val, ones);
                    r.is_i64 = true;
                    r.is_unsigned = a.is_unsigned;
                    return r;
                }
                MLIR_ValueHandle ones = emit_const_i32(e, -1);
                r.val = emit_binop(e, OP_TYPE_ARITH_XORI,
                                   e->i32, a.val, ones);
                r.is_unsigned = a.is_unsigned;
                return r;
            }
            MLIR_ValueHandle b = emit_to_bool_i1(e, a);
            r.val = emit_select(e, e->i32, b, emit_const_i32(e, 0), emit_const_i32(e, 1));
            fact_put(e, r.val, FACT_RANGE, 1, true, 0);
            return r;
        }
        case EX_BIN: {
            if (ex->op == OP_AND || ex->op == OP_OR) {
                EVal a = emit_expr(e, sc, ex->lhs);
                MLIR_ValueHandle ab = emit_to_bool_i1(e, a);

                MLIR_RegionHandle then_r = MLIR_CreateRegion(e->ctx);
                MLIR_BlockHandle then_b = MLIR_CreateBlock(e->ctx);
                MLIR_AppendRegionBlock(e->ctx, then_r, then_b);
                MLIR_RegionHandle else_r = MLIR_CreateRegion(e->ctx);
                MLIR_BlockHandle else_b = MLIR_CreateBlock(e->ctx);
                MLIR_AppendRegionBlock(e->ctx, else_r, else_b);

                MLIR_ValueHandle if_res = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
                MLIR_ValueHandle *rs = &if_res;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = ab;
                MLIR_RegionHandle *regs = arena_new_array(e->arena, MLIR_RegionHandle, 2);
                regs[0] = then_r; regs[1] = else_r;
                emit_op(e, OP_TYPE_SCF_IF,
                        rts, 1, rs, 1, ops, 1, NULL, 0, regs, 2);

                MLIR_BlockHandle saved = e->cur_block;
                bool saved_term = e->terminated;

                e->cur_block = then_b; e->terminated = false;
                MLIR_ValueHandle then_val;
                if (ex->op == OP_AND) {
                    EVal b = emit_expr(e, sc, ex->rhs);
                    then_val = bool_to_i32(e, emit_to_bool_i1(e, b));
                } else {
                    then_val = emit_const_i32(e, 1);
                }
                MLIR_ValueHandle *ty = arena_new_array(e->arena, MLIR_ValueHandle, 1); ty[0] = then_val;
                emit_op(e, OP_TYPE_SCF_YIELD,
                        NULL, 0, NULL, 0, ty, 1, NULL, 0, NULL, 0);

                e->cur_block = else_b; e->terminated = false;
                MLIR_ValueHandle else_val;
                if (ex->op == OP_AND) {
                    else_val = emit_const_i32(e, 0);
                } else {
                    EVal b = emit_expr(e, sc, ex->rhs);
                    else_val = bool_to_i32(e, emit_to_bool_i1(e, b));
                }
                MLIR_ValueHandle *ey = arena_new_array(e->arena, MLIR_ValueHandle, 1); ey[0] = else_val;
                emit_op(e, OP_TYPE_SCF_YIELD,
                        NULL, 0, NULL, 0, ey, 1, NULL, 0, NULL, 0);
                e->cur_block = saved; e->terminated = saved_term;
                r.val = if_res;
                return r;
            }
            EVal a = emit_expr(e, sc, ex->lhs);
            EVal b = emit_expr(e, sc, ex->rhs);
            // Pointer comparisons (eq/ne) use llvm.icmp on !llvm.ptr. An
            // integer operand converts to a pointer, as in gcc: `p == 0` is
            // a null check, `p == k` compares p with (T *)k.
            if ((a.is_ptr || b.is_ptr) &&
                (ex->op == OP_EQ || ex->op == OP_NE)) {
                MLIR_ValueHandle av = int_to_ptr(e, a);
                MLIR_ValueHandle bv = int_to_ptr(e, b);
                int64_t pred = (ex->op == OP_EQ) ? 0 : 1;
                MLIR_ValueHandle cmp = emit_icmp_ptr(e, pred, av, bv);
                r.val = bool_to_i32(e, cmp);
                return r;
            }
            // Pointer ordering comparisons (<, <=, >, >=) use llvm.icmp with
            // unsigned predicates (addresses are unsigned: on wasm32 a heap
            // above 2 GiB would compare wrongly as signed). An integer
            // operand converts to a pointer, as for `==`.
            if ((a.is_ptr || b.is_ptr) &&
                (ex->op == OP_LT || ex->op == OP_LE ||
                 ex->op == OP_GT || ex->op == OP_GE)) {
                MLIR_ValueHandle av = int_to_ptr(e, a);
                MLIR_ValueHandle bv = int_to_ptr(e, b);
                MLIR_ValueHandle cmp = emit_icmp_ptr(e, cmpi_pred_for(ex->op, true), av, bv);
                r.val = bool_to_i32(e, cmp);
                return r;
            }
            // Pointer arithmetic: p + i, i + p (-> !llvm.ptr via GEP).
            if (ex->op == OP_ADD && (a.is_ptr || b.is_ptr) && !(a.is_ptr && b.is_ptr)) {
                if (a.is_void_ptr || b.is_void_ptr) {
                    EMIT_ERR(e, "pointer arithmetic on 'void*' is not allowed");
                }
                EVal pv = a.is_ptr ? a : b;
                EVal iv = a.is_ptr ? b : a;
                MLIR_TypeHandle elem = (pv.ptr_elem != MLIR_INVALID_HANDLE) ? pv.ptr_elem
                    : (pv.sdef ? find_struct_type(e, pv.sdef) : e->i32);
                if (elem == MLIR_INVALID_HANDLE) elem = e->i32;
                MLIR_ValueHandle idx = iv.is_float ? emit_const_i32(e, 0) : iv.val;
                int32_t *path = arena_new_array(e->arena, int32_t, 1); path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1); dyn[0] = idx;
                r.val = emit_gep(e, pv.val, elem, path, 1, dyn, 1);
                r.is_ptr = true;
                r.ptr_elem = pv.ptr_elem;
                r.sdef = pv.sdef;
                r.is_str = pv.is_str;
                return r;
            }
            // Pointer minus integer: p - i -> GEP %p[-i].
            if (ex->op == OP_SUB && a.is_ptr && !b.is_ptr) {
                if (a.is_void_ptr) {
                    EMIT_ERR(e, "pointer arithmetic on 'void*' is not allowed");
                }
                MLIR_TypeHandle elem = (a.ptr_elem != MLIR_INVALID_HANDLE) ? a.ptr_elem
                    : (a.sdef ? find_struct_type(e, a.sdef) : e->i32);
                if (elem == MLIR_INVALID_HANDLE) elem = e->i32;
                // Negate the index at its own width (`p - n` with a 64-bit n).
                MLIR_ValueHandle zero = b.is_i64 ? emit_const_i64(e, 0) : emit_const_i32(e, 0);
                MLIR_ValueHandle neg = emit_binop(e, OP_TYPE_ARITH_SUBI,
                                                  b.is_i64 ? e->i64 : e->i32, zero, b.val);
                int32_t *path = arena_new_array(e->arena, int32_t, 1); path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1); dyn[0] = neg;
                r.val = emit_gep(e, a.val, elem, path, 1, dyn, 1);
                r.is_ptr = true;
                r.ptr_elem = a.ptr_elem;
                r.sdef = a.sdef;
                r.is_str = a.is_str;
                return r;
            }
            // Pointer minus pointer (T* - T*) -> ptrdiff_t element count.
            if (ex->op == OP_SUB && a.is_ptr && b.is_ptr) {
                if (a.is_void_ptr || b.is_void_ptr) {
                    EMIT_ERR(e, "pointer arithmetic on 'void*' is not allowed");
                }
                // ptrtoint both, then (ai - bi) / sizeof(elem).
                MLIR_ValueHandle ai = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *rts1 = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts1[0] = e->i64;
                MLIR_ValueHandle *rs1 = &ai;
                MLIR_ValueHandle *ops1 = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops1[0] = a.val;
                emit_op(e, OP_TYPE_LLVM_PTRTOINT,
                        rts1, 1, rs1, 1, ops1, 1, NULL, 0, NULL, 0);
                MLIR_ValueHandle bi = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *rts2 = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts2[0] = e->i64;
                MLIR_ValueHandle *rs2 = &bi;
                MLIR_ValueHandle *ops2 = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops2[0] = b.val;
                emit_op(e, OP_TYPE_LLVM_PTRTOINT,
                        rts2, 1, rs2, 1, ops2, 1, NULL, 0, NULL, 0);
                MLIR_ValueHandle diff = emit_binop(e, OP_TYPE_ARITH_SUBI, e->i64, ai, bi);
                // One side may carry no element type (an untagged pointer
                // value); the other side then decides.
                bool a_known = a.ptr_elem != MLIR_INVALID_HANDLE || a.sdef;
                int64_t es = a_known ? ptr_elem_size(e, a.ptr_elem, a.sdef)
                                     : ptr_elem_size(e, b.ptr_elem, b.sdef);
                MLIR_ValueHandle szc = emit_const_i64(e, es);
                MLIR_ValueHandle q = emit_binop(e, OP_TYPE_ARITH_DIVSI, e->i64, diff, szc);
                // The result is a ptrdiff_t (`long`): i64, or i32 on wasm32.
                r.is_i64 = !e->target_wasm32;
                r.val = r.is_i64 ? q : emit_trunci_i64_to_i32(e, q);
                return r;
            }
            // A shift has the (promoted) left operand's type: the right
            // operand's signedness must not make `x >> 2u` logical.
            bool lhs_unsigned = a.is_unsigned;
            if ((ex->op == OP_SHL || ex->op == OP_SHR) && !a.is_float && !b.is_float &&
                !a.is_i64 && b.is_i64) {
                b.val = emit_trunci_i64_to_i32(e, b.val);
                b.is_i64 = false;
            }
            unify_numeric(e, &a, &b);
            bool flt = a.is_float;
            bool flt64 = a.is_f64;
            bool i64m = a.is_i64;
            bool unsgn = (ex->op == OP_SHL || ex->op == OP_SHR) ? lhs_unsigned
                                                               : a.is_unsigned;
            MLIR_TypeHandle ity = i64m ? e->i64 : e->i32;
            MLIR_TypeHandle fty = flt64 ? e->f64 : e->f32;
            switch (ex->op) {
                case OP_ADD:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_ADDF, fty, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_ADDI, ity, a.val, b.val);
                    r.is_float = flt; r.is_f64 = flt && flt64; r.is_i64 = i64m && !flt; r.is_unsigned = unsgn && !flt; return r;
                case OP_SUB:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_SUBF, fty, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_SUBI, ity, a.val, b.val);
                    r.is_float = flt; r.is_f64 = flt && flt64; r.is_i64 = i64m && !flt; r.is_unsigned = unsgn && !flt; return r;
                case OP_MUL:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_MULF, fty, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_MULI, ity, a.val, b.val);
                    r.is_float = flt; r.is_f64 = flt && flt64; r.is_i64 = i64m && !flt; r.is_unsigned = unsgn && !flt; return r;
                case OP_DIV:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_DIVF, fty, a.val, b.val)
                                : (unsgn ? emit_binop(e, OP_TYPE_ARITH_DIVUI, ity, a.val, b.val)
                                         : emit_binop(e, OP_TYPE_ARITH_DIVSI, ity, a.val, b.val));
                    r.is_float = flt; r.is_f64 = flt && flt64; r.is_i64 = i64m && !flt; r.is_unsigned = unsgn && !flt; return r;
                case OP_MOD:
                    if (flt) {
                        EMIT_ERR(e, "% not supported on floats");
                        r.val = emit_const_f32(e, 0.0); r.is_float = true; return r;
                    }
                    r.val = unsgn
                        ? emit_binop(e, OP_TYPE_ARITH_REMUI, ity, a.val, b.val)
                        : emit_binop(e, OP_TYPE_ARITH_REMSI, ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_LT: case OP_LE: case OP_GT: case OP_GE:
                case OP_EQ: case OP_NE: {
                    MLIR_ValueHandle cmp = flt
                        ? emit_cmpf(e, cmpf_pred_for(ex->op), a.val, b.val)
                        : emit_cmpi(e, cmpi_pred_for(ex->op, unsgn), a.val, b.val);
                    r.val = bool_to_i32(e, cmp);
                    return r;
                }
                case OP_BAND:
                    if (flt) { EMIT_ERR(e, "& not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_ANDI, ity, a.val, b.val);
                    if (!i64m) {
                        // `x & 0xFF`: the result fits the mask's width.
                        int64_t m = -1;
                        if (fact_get(e, b.val) == FACT_CONST ||
                            fact_get(e, a.val) == FACT_CONST) m = e->fact_c;
                        if (m >= 0 && m <= 0xFFFF) {
                            int bits = 1;
                            while (bits < 16 && (m >> bits) != 0) bits++;
                            fact_put(e, r.val, FACT_RANGE, bits, true, 0);
                        }
                    }
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_BOR:
                    if (flt) { EMIT_ERR(e, "| not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_ORI, ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_BXOR:
                    if (flt) { EMIT_ERR(e, "^ not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_XORI, ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_SHL:
                    if (flt) { EMIT_ERR(e, "<< not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_SHLI, ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_SHR:
                    if (flt) { EMIT_ERR(e, ">> not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = unsgn
                        ? emit_binop(e, OP_TYPE_ARITH_SHRUI, ity, a.val, b.val)
                        : emit_binop(e, OP_TYPE_ARITH_SHRSI, ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                default: break;
            }
            r.val = emit_const_i32(e, 0);
            return r;
        }
        case EX_CALL: {
            // Indirect call through a callable expression captured in
            // `ex->lhs` (e.g. `(*fp)(args)` produced by parse_postfix). The
            // expression must yield an EVal with is_ptr=true and a
            // fnptr_ty signature. We branch into the same call_indirect
            // emission used for the named-fnptr-variable case below.
            Type *indirect_fnty = NULL;
            MLIR_ValueHandle indirect_callee_ptr = MLIR_INVALID_HANDLE;
            if (ex->callee.size == 0 && ex->lhs) {
                EVal cv = emit_expr(e, sc, ex->lhs);
                if (!cv.is_ptr || !cv.fnptr_ty || !cv.fnptr_ty->fnptr_ret) {
                    EMIT_ERR(e, "callee expression is not a function pointer");
                    r.val = emit_const_i32(e, 0);
                    return r;
                }
                indirect_fnty = cv.fnptr_ty;
                indirect_callee_ptr = cv.val;
            }
            // Built-in va_start/va_end/va_copy: lower to LLVM intrinsics
            // operating on the !llvm.ptr that backs the va_list buffer.
            // The first argument must be a va_list lvalue; we take its
            // address (the alloca pointer) directly.
            char callee_c0 = (!indirect_fnty && ex->callee.size > 0) ? ex->callee.str[0] : 0;
            if ((callee_c0 == 'v' || callee_c0 == '_') &&
                                 (str_eq(ex->callee, str_lit("va_start"))
                                 || str_eq(ex->callee, str_lit("va_end"))
                                 || str_eq(ex->callee, str_lit("va_copy"))
                                 || str_eq(ex->callee, str_lit("__builtin_va_start"))
                                 || str_eq(ex->callee, str_lit("__builtin_va_end"))
                                 || str_eq(ex->callee, str_lit("__builtin_va_copy")))) {
                bool is_start = str_eq(ex->callee, str_lit("va_start")) ||
                                str_eq(ex->callee, str_lit("__builtin_va_start"));
                bool is_copy  = str_eq(ex->callee, str_lit("va_copy")) ||
                                str_eq(ex->callee, str_lit("__builtin_va_copy"));
                size_t expected = is_start ? 2 : (is_copy ? 2 : 1);
                if (ex->args.size != expected) {
                    EMIT_ERR(e, "{} expects {} argument(s)", ex->callee, (int64_t)expected);
                    r.val = emit_const_i32(e, 0); return r;
                }
                Expr *ap = ex->args.data[0];
                if (ap->kind != EX_VAR) {
                    EMIT_ERR(e, "{} first argument must be a va_list variable", ex->callee);
                    r.val = emit_const_i32(e, 0); return r;
                }
                Sym *sy = lookup(e, sc, ap->name);
                if (!sy || sy->type.kind != TY_VA_LIST) {
                    EMIT_ERR(e, "{} first argument must be of type va_list", ex->callee);
                    r.val = emit_const_i32(e, 0); return r;
                }
                MLIR_ValueHandle ap_ptr = sym_addr(e, sy);
                // Cursor va_list model: the va_list buffer's first pointer
                // slot holds the read cursor. va_start seeds it from the
                // hidden __va param (the caller-packed varargs buffer);
                // va_copy duplicates the cursor; va_end is a no-op.
                if (is_copy) {
                    Expr *src = ex->args.data[1];
                    if (src->kind != EX_VAR) {
                        EMIT_ERR(e, "va_copy second argument must be a va_list variable");
                        r.val = emit_const_i32(e, 0); return r;
                    }
                    Sym *ssy = lookup(e, sc, src->name);
                    if (!ssy || ssy->type.kind != TY_VA_LIST) {
                        EMIT_ERR(e, "va_copy second argument must be of type va_list");
                        r.val = emit_const_i32(e, 0); return r;
                    }
                    MLIR_ValueHandle src_ptr = sym_addr(e, ssy);
                    MLIR_ValueHandle cur = emit_load_v(e, src_ptr, e->ptr);
                    emit_store_v(e, cur, ap_ptr);
                } else if (is_start) {
                    if (e->cur_va_args == MLIR_INVALID_HANDLE) {
                        EMIT_ERR(e, "va_start outside a variadic function");
                        r.val = emit_const_i32(e, 0); return r;
                    }
                    emit_store_v(e, e->cur_va_args, ap_ptr);
                }
                // va_end: nothing to release in the cursor model.
                r.val = emit_const_i32(e, 0);
                return r;
            }
            // Built-in __builtin_sqrt / __builtin_sqrtf: lower to
            // llvm.intr.sqrt on f64 / f32 respectively.
            if (callee_c0 == '_' &&
                (str_eq(ex->callee, str_lit("__builtin_sqrt")) ||
                 str_eq(ex->callee, str_lit("__builtin_sqrtf")))) {
                bool is_f32 = str_eq(ex->callee, str_lit("__builtin_sqrtf"));
                if (ex->args.size != 1) {
                    EMIT_ERR(e, "{} expects 1 argument", ex->callee);
                    r.val = is_f32 ? emit_const_f32(e, 0.0) : emit_const_f64(e, 0.0);
                    r.is_float = true; r.is_f64 = !is_f32; return r;
                }
                EVal a = emit_expr(e, sc, ex->args.data[0]);
                MLIR_TypeHandle ft = is_f32 ? e->f32 : e->f64;
                if (!a.is_float) {
                    a.val = emit_itofp(e, a.val, ft, a.is_unsigned);
                } else {
                    if (is_f32 && a.is_f64) {
                        // Truncate f64->f32 (rare here since arg should match).
                        // Fall back: just reuse value if compiler can't match.
                    } else if (!is_f32 && !a.is_f64) {
                        a.val = emit_fpext_f32_to_f64(e, a.val);
                    }
                }
                MLIR_ValueHandle res = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = ft;
                MLIR_ValueHandle *rs = &res;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = a.val;
                emit_op(e, OP_TYPE_LLVM_INTR_SQRT,
                        rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
                r.val = res; r.is_float = true; r.is_f64 = !is_f32; return r;
            }
            // Built-in __builtin_unreachable(): marks an unreachable point
            // (e.g. after a noreturn syscall). tinyC does no flow analysis on
            // it, so treat it as a no-op — any code the C author placed after
            // it is already dead. Returns a dummy i32 for expression contexts.
            if (callee_c0 == '_' && str_eq(ex->callee, str_lit("__builtin_unreachable"))) {
                if (ex->args.size != 0) {
                    EMIT_ERR(e, "__builtin_unreachable expects 0 arguments");
                }
                r.val = emit_const_i32(e, 0); return r;
            }
            // Built-in __builtin_syscall6(num, a1, a2, a3, a4, a5, a6): a raw
            // OS syscall. Lowers to `func.call @__tinyc_syscall6` (a stub the
            // native llvm->aarch64 backend synthesises as a `svc` trap). All
            // seven operands and the result are `long` (i64). This is the
            // primitive corec's platform_*.c builds raw syscalls from; it has
            // meaning only for native targets (there are no raw syscalls on
            // wasm — use the WASI platform there).
            if (callee_c0 == '_' && str_eq(ex->callee, str_lit("__builtin_syscall6"))) {
                if (ex->args.size != 7) {
                    EMIT_ERR(e, "__builtin_syscall6 expects 7 arguments (num + 6)");
                    r.val = emit_const_i64(e, 0); r.is_i64 = true; return r;
                }
                e->need_syscall6_stub = true;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 7);
                for (size_t k = 0; k < 7; k++) {
                    EVal av = emit_expr(e, sc, ex->args.data[k]);
                    ops[k] = av.is_ptr ? emit_ptrtoint_i64(e, av.val)
                                       : coerce_eval_to(e, av, e->i64, (Type){0});
                }
                MLIR_ValueHandle res = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i64;
                MLIR_ValueHandle *rs = &res;
                MLIR_NamedAttr ca = MLIR_Named(MLIR_KEY_CALLEE, MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("__tinyc_syscall6")));
                MLIR_NamedAttr *as = arena_new_array(e->arena, MLIR_NamedAttr, 1); as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL,
                        rts, 1, rs, 1, ops, 7, as, 1, NULL, 0);
                r.val = res; r.is_i64 = true; return r;
            }
            // Built-in __builtin_wasm_memory_size(0) and
            // __builtin_wasm_memory_grow(0, n): lower to the
            // LLVM-style wasm intrinsics that the
            // llvm-dialect -> wasmssa pass recognizes. The first
            // argument (memory index) must be the integer literal 0;
            // we don't model multi-memory wasm yet so we just verify
            // and drop it. Both intrinsics produce i32 results.
            if (callee_c0 == '_' &&
                (str_eq(ex->callee, str_lit("__builtin_wasm_memory_size")) ||
                 str_eq(ex->callee, str_lit("__builtin_wasm_memory_grow")))) {
                bool is_grow = str_eq(ex->callee, str_lit("__builtin_wasm_memory_grow"));
                size_t want = is_grow ? 2 : 1;
                if (ex->args.size != want) {
                    EMIT_ERR(e, "{} expects {} argument(s)", ex->callee,
                             want == 1 ? str_lit("1") : str_lit("2"));
                    r.val = emit_const_i32(e, 0); return r;
                }
                // Argument 0 must be the immediate memory index. Only
                // 0 (default linear memory) is supported.
                Expr *aidx = ex->args.data[0];
                if (aidx->kind != EX_INT || aidx->int_value != 0) {
                    EMIT_ERR(e, "{}: first arg must be the constant 0",
                             ex->callee);
                }
                MLIR_TypeHandle rty = e->i32;
                MLIR_ValueHandle res = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = rty;
                MLIR_ValueHandle *rs = &res;
                if (is_grow) {
                    EVal pages = emit_expr(e, sc, ex->args.data[1]);
                    MLIR_ValueHandle p32 = coerce_eval_to(e, pages, e->i32, (Type){.int_unsigned = true});
                    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = p32;
                    emit_op(e, OP_TYPE_LLVM_INTR_WASM_MEMORY_GROW,
                            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
                } else {
                    emit_op(e, OP_TYPE_LLVM_INTR_WASM_MEMORY_SIZE,
                            rts, 1, rs, 1, NULL, 0, NULL, 0, NULL, 0);
                }
                r.val = res; return r;
            }
            // Built-in malloc(size) -> !llvm.ptr. The libc signature
            // takes i64, so accept any integer expression (i32, i64,
            // pointer-sized) and coerce up to i64. Using emit_expr +
            // coerce_eval avoids emitting a no-op `arith.extsi i64->i64`
            // when the source is already size_t-sized (e.g. `size_t n;
            // malloc(n)` on a 64-bit host).
            if (callee_c0 == 'm' && str_eq(ex->callee, str_lit("malloc"))) {
                // If the user provided their own `extern void *malloc(...);`
                // prototype, mark that signature as referenced so the
                // forward-decl loop in emit_program emits a proper
                // `func.func` declaration for it. The special-case path
                // here bypasses find_sig (which is what normally tracks
                // usage), so without this we'd drop the user decl AND
                // skip our auto-decl (because have_user_malloc is true),
                // leaving the @malloc call unresolved.
                (void)find_sig(e, ex->callee);
                if (ex->args.size != 1) {
                    EMIT_ERR(e, "malloc expects 1 argument");
                    r.val = emit_null_ptr(e); r.is_ptr = true; return r;
                }
                EVal sz_v = emit_expr(e, sc, ex->args.data[0]);
                MLIR_ValueHandle sz_coerced = coerce_eval_to(e, sz_v, e->size_t_ty, (Type){.int_unsigned = true});
                MLIR_ValueHandle res = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
                MLIR_ValueHandle *rs = &res;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = sz_coerced;
                MLIR_NamedAttr ca = MLIR_Named(MLIR_KEY_CALLEE, MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("malloc")));
                MLIR_NamedAttr *as = arena_new_array(e->arena, MLIR_NamedAttr, 1); as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL,
                        rts, 1, rs, 1, ops, 1, as, 1, NULL, 0);
                r.val = res; r.is_ptr = true; return r;
            }
            if (callee_c0 == 'f' && str_eq(ex->callee, str_lit("free"))) {
                (void)find_sig(e, ex->callee);
                if (ex->args.size != 1) {
                    EMIT_ERR(e, "free expects 1 argument");
                    return r;
                }
                EVal pv = emit_expr(e, sc, ex->args.data[0]);
                MLIR_ValueHandle p = int_to_ptr(e, pv);
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = p;
                MLIR_NamedAttr ca = MLIR_Named(MLIR_KEY_CALLEE, MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("free")));
                MLIR_NamedAttr *as = arena_new_array(e->arena, MLIR_NamedAttr, 1); as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL,
                        NULL, 0, NULL, 0, ops, 1, as, 1, NULL, 0);
                r.val = emit_const_i32(e, 0);
                return r;
            }
            FuncSig *sig = indirect_fnty ? NULL : find_sig(e, ex->callee);
            if (!sig) {
                Type *fnty = NULL;
                MLIR_ValueHandle callee_ptr = MLIR_INVALID_HANDLE;
                if (indirect_fnty) {
                    fnty = indirect_fnty;
                    callee_ptr = indirect_callee_ptr;
                } else {
                    // Indirect call through a function-pointer value bound to
                    // `ex->callee` (local var, parameter, or module global).
                    Sym *sy = lookup(e, sc, ex->callee);
                    if (!sy || sy->type.kind != TY_FNPTR || !sy->type.fnptr_ret) {
                        EMIT_ERR(e, "undefined function: {}", ex->callee);
                        r.val = emit_const_i32(e, 0);
                        return r;
                    }
                    fnty = &sy->type;
                    // Load the !llvm.ptr from the variable's storage.
                    callee_ptr = sym_load_ptr(e, sy);
                }
                size_t na = ex->args.size;
                if ((int)na != fnty->fnptr_nparams) {
                    EMIT_ERR(e, "indirect call to {} arity mismatch",
                            ex->callee);
                }
                // Struct-returning indirect call uses sret ABI: a hidden
                // first !llvm.ptr param receives the buffer the callee
                // writes the struct into; the call itself returns void.
                bool ret_is_struct = (fnty->fnptr_ret->kind == TY_STRUCT);
                bool ret_is_void = (fnty->fnptr_ret->kind == TY_VOID);
                StructDef *ret_sd = NULL;
                MLIR_TypeHandle ret_st_ty = MLIR_INVALID_HANDLE;
                MLIR_ValueHandle sret_buf = MLIR_INVALID_HANDLE;
                if (ret_is_struct) {
                    ret_sd = find_struct(e, fnty->fnptr_ret->struct_name);
                    if (!ret_sd) {
                        EMIT_ERR(e, "unknown struct return type {}",
                                 fnty->fnptr_ret->struct_name);
                        r.val = emit_const_i32(e, 0);
                        return r;
                    }
                    ret_st_ty = find_struct_type(e, ret_sd);
                    sret_buf = emit_alloca(e, ret_st_ty);
                }
                // Build the function MLIR type that func.call_indirect needs.
                size_t np = (size_t)fnty->fnptr_nparams;
                size_t n_in_tys = np + (ret_is_struct ? 1 : 0);
                MLIR_TypeHandle *in_tys = arena_new_array(e->arena, MLIR_TypeHandle,
                                                           n_in_tys ? n_in_tys : 1);
                size_t in_off = 0;
                if (ret_is_struct) in_tys[in_off++] = e->ptr;
                for (size_t i = 0; i < np; i++) {
                    if (fnty->fnptr_params[i].kind == TY_STRUCT ||
                        fnty->fnptr_params[i].kind == TY_PTR_STRUCT)
                        in_tys[in_off++] = e->ptr;
                    else
                        in_tys[in_off++] = scalar_mlir_type(e, fnty->fnptr_params[i].kind);
                }
                MLIR_TypeHandle rty = ret_is_struct
                    ? e->i32 /* unused */
                    : scalar_mlir_type(e, fnty->fnptr_ret->kind);
                size_t n_out_tys = (ret_is_struct || ret_is_void) ? 0 : 1;
                MLIR_TypeHandle *out_tys = n_out_tys ? &rty : NULL;
                MLIR_TypeHandle f_ty = MLIR_CreateTypeFunction(e->ctx,
                    in_tys, n_in_tys, out_tys, n_out_tys);
                // Cast !llvm.ptr -> function type so func.call_indirect's
                // verifier is happy. The reconcile-unrealized-casts pass
                // eliminates this after func-to-llvm lowering.
                MLIR_ValueHandle callee_v = MLIR_INVALID_HANDLE;
                MLIR_TypeHandle *crts = arena_new_array(e->arena, MLIR_TypeHandle, 1);
                crts[0] = f_ty;
                MLIR_ValueHandle *crs = &callee_v;
                MLIR_ValueHandle *cops = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                cops[0] = callee_ptr;
                emit_op(e, OP_TYPE_UNREALIZED_CONVERSION_CAST,
                        crts, 1, crs, 1, cops, 1, NULL, 0, NULL, 0);

                size_t n_ops = 1 + na + (ret_is_struct ? 1 : 0);
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle,
                                                         n_ops ? n_ops : 1);
                size_t op_off = 0;
                ops[op_off++] = callee_v;
                if (ret_is_struct) ops[op_off++] = sret_buf;
                for (size_t i = 0; i < na; i++) {
                    Type pty = (i < (size_t)fnty->fnptr_nparams)
                        ? fnty->fnptr_params[i] : (Type){0};
                    if (pty.kind == TY_STRUCT) {
                        StructDef *psd = find_struct(e, pty.struct_name);
                        StructDef *asd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(e, sc,
                                ex->args.data[i], &asd);
                        if (src == MLIR_INVALID_HANDLE || (psd && asd != psd)) {
                            EMIT_ERR(e, "indirect call struct arg type mismatch");
                            ops[op_off++] = src;
                            continue;
                        }
                        MLIR_TypeHandle st = find_struct_type(e, psd);
                        MLIR_ValueHandle tmp = emit_alloca(e, st);
                        emit_struct_copy(e, tmp, src, psd);
                        ops[op_off++] = tmp;
                    } else if (pty.kind == TY_PTR_STRUCT) {
                        StructDef *asd = NULL;
                        e->struct_src_int_ok = true;
                        MLIR_ValueHandle src = resolve_struct_source(e, sc,
                                ex->args.data[i], &asd);
                        ops[op_off++] = src;
                    } else {
                        EVal av = emit_expr(e, sc, ex->args.data[i]);
                        TypeKind want = (i < (size_t)fnty->fnptr_nparams)
                            ? fnty->fnptr_params[i].kind : TY_I32;
                        Type want_t = (i < (size_t)fnty->fnptr_nparams)
                            ? fnty->fnptr_params[i] : (Type){0};
                        ops[op_off++] = coerce_eval_to(e, av, scalar_mlir_type(e, want),
                                                       want_t);
                    }
                }
                size_t n_rts = (ret_is_struct || ret_is_void) ? 0 : 1;
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1);
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                if (n_rts) {
                    rts[0] = rty;
                }
                emit_op(e, OP_TYPE_FUNC_CALL_INDIRECT,
                        rts, n_rts, rs, n_rts, ops, n_ops, NULL, 0, NULL, 0);
                if (ret_is_struct) {
                    r.val = sret_buf;
                    r.is_ptr = true;
                    r.sdef = ret_sd;
                    return r;
                }
                if (ret_is_void) {
                    r.val = emit_const_i32(e, 0);
                    return r;
                }
                r.val = rs[0];
                // The callee may be compiled elsewhere (see the direct call).
                if (!e->target_wasm32 && type_promotes_to_int(*fnty->fnptr_ret))
                    r.val = emit_narrow_i32(e, r.val, fnty->fnptr_ret->int_bits,
                                            fnty->fnptr_ret->is_bool ||
                                            fnty->fnptr_ret->int_unsigned);
                r.is_float = (fnty->fnptr_ret->kind == TY_F32 || fnty->fnptr_ret->kind == TY_F64);
                r.is_f64 = (fnty->fnptr_ret->kind == TY_F64);
                r.is_i64 = (fnty->fnptr_ret->kind == TY_I64);
                r.is_unsigned = fnty->fnptr_ret->int_unsigned &&
                                !type_promotes_to_int(*fnty->fnptr_ret);
                if (rty == e->ptr) return ptr_eval_of_type(e, rs[0], fnty->fnptr_ret);
                return r;
            }
            if (sig->ret.type.kind == TY_STRUCT) {
                MLIR_TypeHandle st = find_struct_type(e, sig->ret.sdef);
                MLIR_ValueHandle tmp = emit_alloca(e, st);
                emit_flat_call(e, sc, sig, ex->args, NULL, tmp);
                r.val = tmp;
                r.is_ptr = true;
                r.sdef = sig->ret.sdef;
                return r;
            }
            if (sig->ret.type.kind == TY_VOID) {
                EMIT_ERR(e, "void-returning call has no value (use as a statement)");
                emit_flat_call(e, sc, sig, ex->args, NULL, MLIR_INVALID_HANDLE);
                r.val = emit_const_i32(e, 0);
                return r;
            }
            MLIR_ValueHandle *results = arena_new_array(e->arena, MLIR_ValueHandle, 1);
            emit_flat_call(e, sc, sig, ex->args, results, MLIR_INVALID_HANDLE);
            r.val = results[0];
            r.is_float = (sig->ret.type.kind == TY_F32 || sig->ret.type.kind == TY_F64);
            r.is_f64 = (sig->ret.type.kind == TY_F64);
            r.is_i64 = (sig->ret.type.kind == TY_I64);
            r.is_unsigned = sig->ret.type.int_unsigned &&
                            !type_promotes_to_int(sig->ret.type);
            if (sig->defined_here && type_promotes_to_int(sig->ret.type))
                fact_put(e, r.val, FACT_RANGE,
                         sig->ret.type.is_bool ? 1 : sig->ret.type.int_bits,
                         sig->ret.type.is_bool || sig->ret.type.int_unsigned, 0);
            else if (!e->target_wasm32 && type_promotes_to_int(sig->ret.type))
                // A function compiled elsewhere, e.g. by a C compiler: the
                // x86-64 / AArch64 ABIs leave the bits above a char / short /
                // bool result unspecified (on wasm the callee extends).
                r.val = emit_narrow_i32(e, r.val, sig->ret.type.int_bits,
                                        sig->ret.type.is_bool || sig->ret.type.int_unsigned);
            if (sig->ret.type.kind == TY_PTR_STRUCT) {
                r.is_ptr = true;
                r.sdef = sig->ret.sdef;
            }
            if (sig->ret.type.kind == TY_PTR_I32 ||
                sig->ret.type.kind == TY_PTR_CHAR ||
                sig->ret.type.kind == TY_PTR_VOID ||
                sig->ret.type.kind == TY_PTR_PTR ||
                sig->ret.type.kind == TY_FNPTR) {
                return ptr_eval_of_type(e, results[0], &sig->ret.type);
            }
            return r;
        }
        case EX_COMPOUND: {
            EVal r = (EVal){0};
            Type t = *ex->cast_type;
            if (t.kind == TY_ARRAY_STRUCT) {
                // Array-of-struct compound literal `(S[]){{...}, {...}, ...}`.
                // Each arg is itself an EX_COMPOUND for a single struct
                // element. Length defaults to the number of args when the
                // bracket was empty (`(S[]){...}`).
                StructDef *sd = find_struct(e, t.struct_name);
                if (!sd) { EMIT_ERR(e, "unknown struct in array compound literal"); return r; }
                int64_t n = (int64_t)ex->args.size;
                if (t.array_len > 0) {
                    if (t.array_len < n) n = t.array_len;
                } else {
                    t.array_len = n;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, st_ty, (uint64_t)t.array_len);
                MLIR_ValueHandle addr = emit_alloca(e, arr_ty);
                for (int64_t i = 0; i < n; i++) {
                    Expr *elem = ex->args.data[i];
                    int32_t path[2] = {0, (int32_t)i};
                    MLIR_ValueHandle p = emit_gep(e, addr, arr_ty, path, 2, NULL, 0);
                    if (elem->kind == EX_COMPOUND) {
                        // Fill element struct fields directly (avoid
                        // emitting a second alloca + memcpy).
                        size_t fn = elem->args.size;
                        if (fn > sd->fields.size) fn = sd->fields.size;
                        // Zero first.
                        int32_t pz[1] = {0};
                        emit_struct_zero(e, p, st_ty, sd, pz, 1);
                        for (size_t fi = 0; fi < fn; fi++) {
                            Type ft = sd->fields.data[fi].type;
                            int32_t fpath[2] = {0, (int32_t)fi};
                            MLIR_ValueHandle fp = emit_gep(e, p, st_ty, fpath, 2, NULL, 0);
                            EVal vv = emit_expr(e, sc, elem->args.data[fi]);
                            MLIR_TypeHandle want = scalar_mem_type(e, &ft);
                            MLIR_ValueHandle sv;
                            if (ft.kind == TY_PTR_I32 || ft.kind == TY_PTR_CHAR ||
                                ft.kind == TY_PTR_VOID || ft.kind == TY_PTR_STRUCT ||
                                ft.kind == TY_FNPTR || ft.kind == TY_PTR_PTR) {
                                sv = vv.val;
                            } else {
                                sv = coerce_eval_to(e, vv, want, ft);
                            }
                            emit_store_v(e, sv, fp);
                        }
                    } else {
                        // Allow a struct-typed expression as an element
                        // (e.g. an existing struct local) by copying.
                        StructDef *esd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(e, sc, elem, &esd);
                        if (src != MLIR_INVALID_HANDLE && esd == sd) {
                            emit_struct_copy(e, p, src, sd);
                        } else {
                            EMIT_ERR(e, "array compound literal element type mismatch");
                        }
                    }
                }
                r.val = addr;
                r.is_ptr = true;
                r.sdef = sd;
                return r;
            }
            if (t.kind == TY_STRUCT) {
                StructDef *sd = find_struct(e, t.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct in compound literal");
                    return r;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle addr = emit_alloca(e, st_ty);
                int32_t prefix0[1] = {0};
                emit_struct_zero(e, addr, st_ty, sd, prefix0, 1);
                size_t n = ex->args.size;
                if (n > sd->fields.size) {
                    EMIT_ERR(e, "too many initializers in compound literal");
                    n = sd->fields.size;
                }
                for (size_t i = 0; i < n; i++) {
                    size_t fi = i;
                    if (ex->compound_field_names &&
                        ex->compound_field_names[i].size > 0) {
                        bool found = false;
                        for (size_t k = 0; k < sd->fields.size; k++) {
                            if (str_eq(sd->fields.data[k].name,
                                       ex->compound_field_names[i])) {
                                fi = k; found = true; break;
                            }
                        }
                        if (!found) {
                            EMIT_ERR(e, "unknown field '{}' in designated initializer",
                                     ex->compound_field_names[i]);
                            continue;
                        }
                    }
                    Type ft = sd->fields.data[fi].type;
                    int32_t path[2] = {0, (int32_t)fi};
                    MLIR_ValueHandle p =
                        emit_gep(e, addr, st_ty, path, 2, NULL, 0);
                    if (ft.kind == TY_STRUCT) {
                        StructDef *fsd = find_struct(e, ft.struct_name);
                        if (!fsd) {
                            EMIT_ERR(e, "unknown struct type for compound-literal field");
                            continue;
                        }
                        Expr *ae = ex->args.data[i];
                        if (ae->kind == EX_INT && ae->int_value == 0) {
                            // Already zero-initialized by emit_struct_zero above.
                            continue;
                        }
                        // Nested aggregate `{ ... }` with no explicit
                        // cast_type: propagate the surrounding field's
                        // struct type so emit_expr resolves it.
                        if (ae->kind == EX_COMPOUND &&
                            ae->cast_type->kind == TY_VOID) {
                            *ae->cast_type = ft;
                        }
                        StructDef *src_sd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(e, sc, ae, &src_sd);
                        if (src != MLIR_INVALID_HANDLE && src_sd == fsd) {
                            emit_struct_copy(e, p, src, fsd);
                        } else {
                            EVal vv = emit_expr(e, sc, ae);
                            if (vv.is_ptr && vv.val != MLIR_INVALID_HANDLE) {
                                emit_struct_copy(e, p, vv.val, fsd);
                            } else {
                                EMIT_ERR(e, "compound-literal struct field needs a struct value");
                            }
                        }
                        continue;
                    }
                    EVal v = emit_expr(e, sc, ex->args.data[i]);
                    MLIR_TypeHandle want = scalar_mem_type(e, &ft);
                    MLIR_ValueHandle sv;
                    if (ft.kind == TY_PTR_I32 || ft.kind == TY_PTR_CHAR ||
                        ft.kind == TY_PTR_VOID || ft.kind == TY_PTR_STRUCT ||
                        ft.kind == TY_FNPTR || ft.kind == TY_PTR_PTR) {
                        sv = v.val;
                    } else {
                        sv = coerce_eval_to(e, v, want, ft);
                    }
                    emit_store_v(e, sv, p);
                }
                r.val = addr;
                r.is_ptr = true;
                r.sdef = sd;
                return r;
            }
            if (ex->args.size != 1) {
                EMIT_ERR(e, "scalar compound literal needs exactly one value");
                return r;
            }
            EVal v = emit_expr(e, sc, ex->args.data[0]);
            if (t.kind == TY_I32) {
                r.val = coerce_eval_to(e, v, e->i32, t);
                return r;
            }
            if (t.kind == TY_I64) {
                r.val = coerce_eval_to(e, v, e->i64, t);
                r.is_i64 = true;
                return r;
            }
            if (t.kind == TY_F32) {
                r.val = coerce_eval_to(e, v, e->f32, (Type){0});
                r.is_float = true;
                return r;
            }
            if (t.kind == TY_PTR_I32 || t.kind == TY_PTR_CHAR ||
                t.kind == TY_PTR_VOID || t.kind == TY_PTR_STRUCT ||
                t.kind == TY_PTR_PTR || t.kind == TY_FNPTR) {
                r = v;
                r.is_ptr = true;
                r.is_void_ptr = (t.kind == TY_PTR_VOID);
                if (t.kind == TY_PTR_STRUCT) {
                    r.sdef = find_struct(e, t.struct_name);
                }
                return r;
            }
            EMIT_ERR(e, "unsupported type in compound literal");
            return r;
        }
    }
    r.val = emit_const_i32(e, 0);
    return r;
}

static void emit_block(E *e, Scope *sc, VecStmtPtr body);

// Zero-initialize a struct local field-by-field.
static void emit_struct_zero(E *e, MLIR_ValueHandle base, MLIR_TypeHandle source_elem,
                             StructDef *sd, int32_t *prefix, size_t n_prefix) {
    if (sd->is_union) {
        // The union's llvm body is a single blob field (index 0) of
        // `count` alignment-wide integers. Zero each blob element.
        MLIR_TypeHandle uelem; int64_t ues, uc;
        union_blob_layout(e, sd, &uelem, &ues, &uc);
        MLIR_ValueHandle z = emit_zero_of(e, uelem);
        for (int64_t j = 0; j < uc; j++) {
            size_t total = n_prefix + 2;
            int32_t *p2 = arena_new_array(e->arena, int32_t, total);
            for (size_t k = 0; k < n_prefix; k++) p2[k] = prefix[k];
            p2[n_prefix] = 0;
            p2[n_prefix + 1] = (int32_t)j;
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, p2, total, NULL, 0);
            emit_store_v(e, z, p);
        }
        return;
    }
    for (size_t i = 0; i < sd->fields.size; i++) {
        Type ft = sd->fields.data[i].type;
        size_t n_path = n_prefix + 1;
        int32_t *path = arena_new_array(e->arena, int32_t, n_path);
        for (size_t k = 0; k < n_prefix; k++) path[k] = prefix[k];
        path[n_prefix] = (int32_t)i;
        if (ft.kind == TY_STRUCT) {
            StructDef *inner = find_struct(e, ft.struct_name);
            emit_struct_zero(e, base, source_elem, inner, path, n_path);
        } else if (ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                   ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID ||
                   ft.kind == TY_PTR_PTR || ft.kind == TY_FNPTR) {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, emit_null_ptr(e), p);
        } else if (ft.kind == TY_I64) {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, emit_const_i64(e, 0), p);
        } else if (ft.kind == TY_F64) {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, emit_const_f64(e, 0.0), p);
        } else if (ft.kind == TY_ARRAY_I32 || ft.kind == TY_ARRAY_F32 ||
                   ft.kind == TY_ARRAY_PTR_STRUCT || ft.kind == TY_ARRAY_PTR_CHAR) {
            // Zero each element with explicit GEPs. Cheap and avoids
            // reaching for llvm.memset.
            bool is_2d = (ft.array_len2 != 0);
            int64_t n1 = ft.array_len;
            int64_t n2 = is_2d ? ft.array_len2 : 1;
            MLIR_ValueHandle z = emit_zero_of(e, array_elem_type(e, &ft));
            for (int64_t a = 0; a < n1; a++) {
                for (int64_t b = 0; b < n2; b++) {
                    size_t inner_n = is_2d ? 2 : 1;
                    size_t total = n_path + inner_n;
                    int32_t *p2 = arena_new_array(e->arena, int32_t, total);
                    for (size_t kk = 0; kk < n_path; kk++) p2[kk] = path[kk];
                    p2[n_path] = (int32_t)a;
                    if (is_2d) p2[n_path + 1] = (int32_t)b;
                    MLIR_ValueHandle p = emit_gep(e, base, source_elem, p2, total, NULL, 0);
                    emit_store_v(e, z, p);
                }
            }
        } else {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, emit_zero_of(e, scalar_mem_type(e, &ft)), p);
        }
    }
}

// Emit `ex` for its side effects only — the result value is discarded.
// Handles a few "void-typed" expression forms (void calls, `(void)X`,
// and `cond ? a : b` where both arms are themselves statement-level)
// without trying to compute a result value; falls back to plain
// `emit_expr` for the general case. This is what makes idioms like
//     ((cond) ? (void)0 : abort())   // as produced by assert()
// compile cleanly: emit_expr would otherwise error on the void-returning
// arm "void-returning call has no value (use as a statement)".
static bool expr_is_void_statement(E *e, Expr *ex) {
    if (!ex) return false;
    if (ex->kind == EX_CALL) {
        FuncSig *sig = find_sig(e, ex->callee);
        return sig && (sig->ret.type.kind == TY_VOID ||
                       sig->ret.type.kind == TY_STRUCT);
    }
    if (ex->kind == EX_CAST && ex->cast_type->kind == TY_VOID) return true;
    if (ex->kind == EX_TERNARY) {
        return expr_is_void_statement(e, ex->rhs) &&
               expr_is_void_statement(e, ex->lvalue);
    }
    return false;
}

static void emit_expr_for_side_effect(E *e, Scope *sc, Expr *ex) {
    if (e->terminated) return;
    if (ex->kind == EX_CALL) {
        FuncSig *sig = find_sig(e, ex->callee);
        if (sig && sig->ret.type.kind == TY_STRUCT) {
            emit_flat_call(e, sc, sig, ex->args, NULL, MLIR_INVALID_HANDLE);
            return;
        }
        if (sig && sig->ret.type.kind == TY_VOID) {
            emit_flat_call(e, sc, sig, ex->args, NULL, MLIR_INVALID_HANDLE);
            return;
        }
    }
    if (ex->kind == EX_CAST && ex->cast_type->kind == TY_VOID) {
        emit_expr_for_side_effect(e, sc, ex->lhs);
        return;
    }
    if (ex->kind == EX_TERNARY &&
        expr_is_void_statement(e, ex->rhs) &&
        expr_is_void_statement(e, ex->lvalue)) {
        // `c ? a : b` as a statement where both arms are void: lower as
        // an if/else branch without an alloca/load for the result.
        EVal cv = emit_expr(e, sc, ex->lhs);
        MLIR_ValueHandle cb = emit_to_bool_i1(e, cv);
        MLIR_BlockHandle then_blk = new_cfg_block(e);
        MLIR_BlockHandle else_blk = new_cfg_block(e);
        MLIR_BlockHandle merge_blk = new_cfg_block(e);
        emit_cond_branch(e, cb, then_blk, else_blk);

        e->cur_block = then_blk; e->terminated = false;
        emit_expr_for_side_effect(e, sc, ex->rhs);
        if (!e->terminated) emit_branch(e, merge_blk);

        e->cur_block = else_blk; e->terminated = false;
        emit_expr_for_side_effect(e, sc, ex->lvalue);
        if (!e->terminated) emit_branch(e, merge_blk);

        e->cur_block = merge_blk; e->terminated = false;
        return;
    }
    Expr *saved_unused = e->unused_assign;
    e->unused_assign = ex->kind == EX_ASSIGN ? ex : NULL;
    (void)emit_expr(e, sc, ex);
    e->unused_assign = saved_unused;
}

static void emit_stmt(E *e, Scope *sc, Stmt *st);

// Unsized `T arr[] = {a, b, ...}`: infer the length from the aggregate
// initializer's argument count.
static void infer_unsized_array_len(E *e, Stmt *st) {
    if ((st->decl_type.kind == TY_ARRAY_I32 ||
         st->decl_type.kind == TY_ARRAY_PTR_CHAR ||
         st->decl_type.kind == TY_ARRAY_PTR_STRUCT ||
         st->decl_type.kind == TY_ARRAY_STRUCT) &&
        st->decl_type.array_len == 0 &&
        !st->decl_type.array_len_expr &&
        st->decl_init && st->decl_init->kind == EX_COMPOUND) {
        int64_t n = (int64_t)st->decl_init->args.size;
        if (n <= 0) {
            EMIT_ERR(e, "unsized array '{}[]' requires a non-empty initializer", st->decl_name);
            n = 1;
        }
        st->decl_type.array_len = n;
    }
}

// Bring the variables a declaration statement declares into scope `sc`
// with their storage, but without running their initializers. Any other
// statement is ignored. Used for the unreachable statements before a
// switch's first case label.
static void declare_without_init(E *e, Scope *sc, Stmt *st) {
    if (st->kind == ST_BLOCK && st->block_no_scope) {
        for (size_t i = 0; i < st->block_body.size; i++)
            declare_without_init(e, sc, st->block_body.data[i]);
        return;
    }
    if (st->kind != ST_DECL) return;
    // A `static` is initialized before the program starts, so its
    // initializer still applies here. (A scalar static was promoted to a
    // global by the parser and has no initializer left on the Stmt.)
    if (st->decl_is_static) {
        emit_stmt(e, sc, st);
        return;
    }
    // The initializer of an unsized array also gives its length.
    infer_unsized_array_len(e, st);
    Expr *init = st->decl_init;
    st->decl_init = NULL;
    emit_stmt(e, sc, st);
    st->decl_init = init;
}

static void emit_stmt(E *e, Scope *sc, Stmt *st) {
    if (e->terminated && st->kind != ST_LABEL) return;
    e->cur_line = st->line;
    switch (st->kind) {
        case ST_EXPR: {
            emit_expr_for_side_effect(e, sc, st->expr);
            return;
        }
        case ST_DECL: {
            Sym *sy = arena_new(e->arena, Sym);
            *sy = (Sym){0};
            sy->name = st->decl_name;
            sy->type = st->decl_type;

            // Function-local `static`: the parser has already registered
            // a module-scope global for this declaration. Bind the local
            // name to that global; do not allocate a stack slot and do
            // not re-emit the initializer (the Global captures it).
            if (st->decl_static_global_sym.size > 0) {
                sy->is_global = true;
                sy->global_sym = st->decl_static_global_sym;
                sy->next = sc->head;
                sc->head = sy;
                return;
            }

            infer_unsized_array_len(e, st);
            // Constant-expression array length, e.g. `T arr[sizeof(x)/sizeof(x[0])]`.
            // Fold against the current scope so sizeof of in-scope locals works.
            resolve_array_len(e, sc, &st->decl_type);
            sy->type.array_len = st->decl_type.array_len;

            // C semantics: an identifier's scope begins immediately after its
            // declarator, so it is visible inside its own initializer (e.g.
            // `T *p = alloc(sizeof(*p))`). Register the symbol now, before the
            // initializer is emitted, so type/size queries on it resolve
            // correctly. (The static-global case above has already returned.)
            sy->next = sc->head;
            sc->head = sy;

            if (st->decl_type.kind == TY_VA_LIST) {
                // Allocate a 32-byte buffer (sufficient for x86_64-SysV
                // and aarch64 va_list layouts on Linux/macOS/Windows). We
                // hand out the pointer; va_start/va_arg/va_end consume it.
                MLIR_TypeHandle buf_ty = MLIR_CreateTypeLLVMArray(e->ctx, e->i8, 32);
                sy->addr = emit_alloca(e, buf_ty);
                break;
            }
            if (st->decl_type.kind == TY_PTR_I32 || st->decl_type.kind == TY_PTR_VOID ||
                st->decl_type.kind == TY_PTR_PTR) {
                bool direct_ssa = !e->has_goto && !st->decl_is_static &&
                                  st->decl_init != NULL && !is_mutated(e, st->decl_name);
                if (direct_ssa) {
                    EVal iv = emit_expr(e, sc, st->decl_init);
                    if (iv.is_ptr) {
                        sy->addr = iv.val;
                    } else if (st->decl_init->kind == EX_INT && st->decl_init->int_value == 0) {
                        sy->addr = emit_null_ptr(e);
                    } else {
                        EMIT_ERR(e, "pointer initializer must be a pointer expression");
                        sy->addr = emit_null_ptr(e);
                    }
                    sy->is_ssa = true;
                } else {
                    sy->addr = emit_alloca(e, e->ptr);
                    if (st->decl_init) {
                        EVal iv = emit_expr(e, sc, st->decl_init);
                        if (iv.is_ptr) {
                            emit_store_v(e, iv.val, sy->addr);
                        } else if (st->decl_init->kind == EX_INT && st->decl_init->int_value == 0) {
                            // Null-pointer constant: `T *p = 0;` / `T *p = NULL;`.
                            emit_store_v(e, emit_null_ptr(e), sy->addr);
                        } else {
                            EMIT_ERR(e, "pointer initializer must be a pointer expression");
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_PTR_CHAR) {
                bool direct_ssa = !e->has_goto && !st->decl_is_static &&
                                  st->decl_init != NULL && !is_mutated(e, st->decl_name);
                if (direct_ssa) {
                    EVal iv = emit_expr(e, sc, st->decl_init);
                    sy->addr = int_to_ptr(e, iv);
                    sy->is_ssa = true;
                } else {
                    sy->addr = emit_alloca(e, e->ptr);
                    if (st->decl_init) {
                        EVal iv = emit_expr(e, sc, st->decl_init);
                        MLIR_ValueHandle p = int_to_ptr(e, iv);
                        emit_store_v(e, p, sy->addr);
                    } else {
                        emit_store_v(e, emit_null_ptr(e), sy->addr);
                    }
                }
            } else if (st->decl_type.kind == TY_FNPTR) {
                bool direct_ssa = !e->has_goto && !st->decl_is_static &&
                                  st->decl_init != NULL && !is_mutated(e, st->decl_name);
                if (direct_ssa) {
                    EVal iv = emit_expr(e, sc, st->decl_init);
                    sy->addr = int_to_ptr(e, iv);
                    sy->is_ssa = true;
                } else {
                    sy->addr = emit_alloca(e, e->ptr);
                    if (st->decl_init) {
                        EVal iv = emit_expr(e, sc, st->decl_init);
                        MLIR_ValueHandle p = int_to_ptr(e, iv);
                        emit_store_v(e, p, sy->addr);
                    } else {
                        emit_store_v(e, emit_null_ptr(e), sy->addr);
                    }
                }
            } else if (st->decl_type.kind == TY_PTR_STRUCT) {
                StructDef *sd = find_struct(e, st->decl_type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct type {}", st->decl_type.struct_name);
                    return;
                }
                sy->sdef = sd;
                bool direct_ssa = !e->has_goto && !st->decl_is_static &&
                                  st->decl_init != NULL && !is_mutated(e, st->decl_name);
                if (direct_ssa) {
                    if (st->decl_init->kind == EX_ADDR &&
                        st->decl_init->lhs->kind == EX_VAR) {
                        Sym *target = lookup(e, sc, st->decl_init->lhs->name);
                        if (!target || target->type.kind != TY_STRUCT || target->sdef != sd) {
                            EMIT_ERR(e, "struct* target type mismatch");
                            sy->addr = emit_null_ptr(e);
                        } else {
                            sy->addr = sym_addr(e, target);
                        }
                    } else {
                        EVal iv = emit_expr(e, sc, st->decl_init);
                        sy->addr = int_to_ptr(e, iv);
                    }
                    sy->is_ssa = true;
                } else {
                    sy->addr = emit_alloca(e, e->ptr);
                    if (st->decl_init) {
                        if (st->decl_init->kind == EX_ADDR &&
                            st->decl_init->lhs->kind == EX_VAR) {
                            Sym *target = lookup(e, sc, st->decl_init->lhs->name);
                            if (!target || target->type.kind != TY_STRUCT || target->sdef != sd) {
                                EMIT_ERR(e, "struct* target type mismatch");
                            } else {
                                emit_store_v(e, sym_addr(e, target), sy->addr);
                            }
                        } else {
                            // Allow any expression yielding a pointer (e.g., a
                            // function call returning struct*, or a cast).
                            EVal iv = emit_expr(e, sc, st->decl_init);
                            MLIR_ValueHandle p = int_to_ptr(e, iv);
                            emit_store_v(e, p, sy->addr);
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_ARRAY_I32) {
                MLIR_TypeHandle elem_ty = array_elem_type(e, &st->decl_type);
                MLIR_TypeHandle arr_ty = array_mlir_type(e, &st->decl_type);
                sy->addr = emit_alloca(e, arr_ty);
                if (st->decl_init) {
                    // `char arr[N] = "string literal"` — copy the string
                    // bytes (lexer already includes a trailing NUL) into
                    // the local buffer, zero-padding any remaining bytes.
                    if (st->decl_init->kind == EX_STR &&
                        elem_ty == e->i8 &&
                        st->decl_type.array_len2 == 0) {
                        string lit = st->decl_init->name;
                        int64_t alen = st->decl_type.array_len;
                        if ((int64_t)lit.size > alen) {
                            EMIT_ERR(e, "string initializer ({} bytes) too long for char[{}]",
                                     (int64_t)lit.size, alen);
                        }
                        int64_t lim = (int64_t)lit.size < alen ? (int64_t)lit.size : alen;
                        for (int64_t k = 0; k < lim; k++) {
                            MLIR_ValueHandle bv = emit_const_narrow(e, (int64_t)(uint8_t)lit.str[k], e->i8);
                            int32_t path[2] = {0, (int32_t)k};
                            MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                            emit_store_v(e, bv, p);
                        }
                        if (lim < alen) {
                            MLIR_ValueHandle z8 = emit_const_narrow(e, 0, e->i8);
                            for (int64_t k = lim; k < alen; k++) {
                                int32_t path[2] = {0, (int32_t)k};
                                MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                                emit_store_v(e, z8, p);
                            }
                        }
                        return;
                    }
                    // Support `= {0}` (zero-initialize) by zeroing every
                    // element. Also support a positional list `{v0,v1,...}`
                    // for 1-D arrays where each value is an int expression.
                    bool is_zero_init = false;
                    bool is_list_init = false;
                    if (st->decl_init->kind == EX_COMPOUND) {
                        size_t n = st->decl_init->args.size;
                        is_zero_init = (n == 0);
                        if (n == 1) {
                            Expr *a = st->decl_init->args.data[0];
                            if (a->kind == EX_INT && a->int_value == 0) {
                                is_zero_init = true;
                            }
                        }
                        if (!is_zero_init && st->decl_type.array_len2 == 0) {
                            is_list_init = true;
                        }
                    }
                    if (!is_zero_init && !is_list_init) {
                        EMIT_ERR(e, "array initializers are not supported");
                    } else if (is_list_init) {
                        size_t n = st->decl_init->args.size;
                        if ((int64_t)n != st->decl_type.array_len) {
                            EMIT_ERR(e, "int array initializer count {} does not match length {}",
                                     (int64_t)n, st->decl_type.array_len);
                        }
                        int64_t lim = (int64_t)n < st->decl_type.array_len ? (int64_t)n : st->decl_type.array_len;
                        for (int64_t a = 0; a < lim; a++) {
                            // The element store width must match the declared
                            // element type, so the store doesn't overflow the
                            // slot and clobber adjacent memory (which on wasm32
                            // can corrupt the immediately-following rodata page).
                            MLIR_ValueHandle iv = coerce_eval_to(e,
                                emit_expr(e, sc, st->decl_init->args.data[a]),
                                elem_ty, st->decl_type);
                            int32_t path[2] = {0, (int32_t)a};
                            MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                            emit_store_v(e, iv, p);
                        }
                    } else {
                        int64_t n1 = st->decl_type.array_len;
                        int64_t n2 = st->decl_type.array_len2 ? st->decl_type.array_len2 : 1;
                        bool is_2d = (st->decl_type.array_len2 != 0);
                        // Use a zero value whose width matches the element
                        // type so the store doesn't overflow narrower slots.
                        MLIR_ValueHandle z = emit_zero_of(e, elem_ty);
                        for (int64_t a = 0; a < n1; a++) {
                            for (int64_t b = 0; b < n2; b++) {
                                int32_t path[3];
                                size_t np = 0;
                                path[np++] = 0;
                                path[np++] = (int32_t)a;
                                if (is_2d) path[np++] = (int32_t)b;
                                MLIR_ValueHandle p = emit_gep(
                                    e, sy->addr, arr_ty, path, np, NULL, 0);
                                emit_store_v(e, z, p);
                            }
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_ARRAY_PTR_CHAR ||
                       st->decl_type.kind == TY_ARRAY_PTR_STRUCT) {
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, e->ptr, (uint64_t)st->decl_type.array_len);
                sy->addr = emit_alloca(e, arr_ty);
                if (st->decl_init) {
                    if (st->decl_init->kind != EX_COMPOUND) {
                        EMIT_ERR(e, "char* array initializer must be an aggregate '{...}'");
                    } else {
                        size_t n = st->decl_init->args.size;
                        // `= {0}` zero-init shorthand.
                        bool is_zero = (n == 0) ||
                                       (n == 1 &&
                                        st->decl_init->args.data[0]->kind == EX_INT &&
                                        st->decl_init->args.data[0]->int_value == 0);
                        if (is_zero) {
                            MLIR_ValueHandle nullp = emit_null_ptr(e);
                            for (int64_t a = 0; a < st->decl_type.array_len; a++) {
                                int32_t path[2] = {0, (int32_t)a};
                                MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                                emit_store_v(e, nullp, p);
                            }
                        } else {
                            if ((int64_t)n != st->decl_type.array_len) {
                                EMIT_ERR(e, "char* array initializer count {} does not match length {}",
                                         (int64_t)n, st->decl_type.array_len);
                            }
                            int64_t lim = (int64_t)n < st->decl_type.array_len ? (int64_t)n : st->decl_type.array_len;
                            for (int64_t a = 0; a < lim; a++) {
                                EVal iv = emit_expr(e, sc, st->decl_init->args.data[a]);
                                MLIR_ValueHandle p_val = int_to_ptr(e, iv);
                                int32_t path[2] = {0, (int32_t)a};
                                MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                                emit_store_v(e, p_val, p);
                            }
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_ARRAY_STRUCT) {
                StructDef *sd = find_struct(e, st->decl_type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct type {}", st->decl_type.struct_name);
                    return;
                }
                sy->sdef = sd;
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                // Allow `S name[] = { ... };` — infer length from the
                // initializer count when no explicit dimension was given.
                int64_t alen = st->decl_type.array_len;
                if (st->decl_init &&
                    st->decl_init->kind == EX_COMPOUND &&
                    alen == 0 && !st->decl_type.array_len_expr) {
                    alen = (int64_t)st->decl_init->args.size;
                    st->decl_type.array_len = alen;
                    sy->type.array_len = alen;
                }
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, st_ty, (uint64_t)alen);
                sy->addr = emit_alloca(e, arr_ty);
                if (st->decl_init) {
                    if (st->decl_init->kind != EX_COMPOUND) {
                        EMIT_ERR(e, "array-of-struct initializer must be a brace list");
                    } else {
                        size_t n = st->decl_init->args.size;
                        if ((int64_t)n > alen) {
                            EMIT_ERR(e, "too many array-of-struct initializers");
                            n = (size_t)alen;
                        }
                        for (size_t k = 0; k < n; k++) {
                            Expr *elem = st->decl_init->args.data[k];
                            int32_t epath[2] = {0, (int32_t)k};
                            MLIR_ValueHandle pelem = emit_gep(
                                e, sy->addr, arr_ty, epath, 2, NULL, 0);
                            // Zero this slot first so any unspecified
                            // fields read back as 0 / null. emit_struct_zero
                            // descends through nested struct fields.
                            int32_t *zprefix = arena_new_array(e->arena, int32_t, 1);
                            zprefix[0] = 0;
                            emit_struct_zero(e, pelem, st_ty, sd, zprefix, 1);
                            if (elem->kind == EX_INT && elem->int_value == 0) {
                                continue;
                            }
                            if (elem->kind == EX_COMPOUND) {
                                // Element is a nested brace list — fill in
                                // each field positionally / by designator.
                                size_t fn = elem->args.size;
                                if (fn > sd->fields.size) {
                                    EMIT_ERR(e, "too many initializers for "
                                                "array element {}", (int64_t)k);
                                    fn = sd->fields.size;
                                }
                                for (size_t i = 0; i < fn; i++) {
                                    int fidx = (int)i;
                                    if (elem->compound_field_names) {
                                        string fname = elem->compound_field_names[i];
                                        if (fname.size != 0) {
                                            fidx = struct_field_index(sd, fname);
                                            if (fidx < 0) {
                                                EMIT_ERR(e, "unknown struct field "
                                                            "{} in initializer", fname);
                                                continue;
                                            }
                                        }
                                    }
                                    Type ft = sd->fields.data[fidx].type;
                                    int32_t fp[2] = {0, (int32_t)fidx};
                                    MLIR_ValueHandle pf = emit_gep(
                                        e, pelem, st_ty, fp, 2, NULL, 0);
                                    Expr *fe = elem->args.data[i];
                                    if (ft.kind == TY_STRUCT) {
                                        StructDef *fsd = find_struct(e, ft.struct_name);
                                        if (!fsd) {
                                            EMIT_ERR(e, "unknown struct type "
                                                        "for nested-struct field "
                                                        "initializer");
                                            continue;
                                        }
                                        if (fe->kind == EX_INT && fe->int_value == 0) {
                                            continue;
                                        }
                                        if (fe->kind == EX_COMPOUND &&
                                            fe->cast_type->kind == TY_VOID) {
                                            *fe->cast_type = ft;
                                        }
                                        StructDef *src_sd = NULL;
                                        MLIR_ValueHandle src = resolve_struct_source(
                                            e, sc, fe, &src_sd);
                                        if (src != MLIR_INVALID_HANDLE &&
                                            src_sd == fsd) {
                                            emit_struct_copy(e, pf, src, fsd);
                                        } else {
                                            EVal vv = emit_expr(e, sc, fe);
                                            if (vv.is_ptr && vv.val != MLIR_INVALID_HANDLE) {
                                                emit_struct_copy(e, pf, vv.val, fsd);
                                            } else {
                                                EMIT_ERR(e, "nested-struct field "
                                                            "initializer needs a "
                                                            "struct value");
                                            }
                                        }
                                        continue;
                                    }
                                    EVal v = emit_expr(e, sc, fe);
                                    MLIR_ValueHandle sv;
                                    if (ft.kind == TY_PTR_I32 ||
                                        ft.kind == TY_PTR_CHAR ||
                                        ft.kind == TY_PTR_VOID ||
                                        ft.kind == TY_PTR_STRUCT ||
                                        ft.kind == TY_FNPTR ||
                                        ft.kind == TY_PTR_PTR) {
                                        sv = int_to_ptr(e, v);
                                    } else if (ft.kind == TY_I64) {
                                        sv = coerce_eval_to(e, v, e->i64, ft);
                                    } else if (ft.kind == TY_F64) {
                                        sv = coerce_eval_to(e, v, e->f64, (Type){0});
                                    } else {
                                        sv = coerce_eval_to(
                                            e, v, scalar_mem_type(e, &ft),
                                            ft);
                                    }
                                    emit_store_v(e, sv, pf);
                                }
                            } else {
                                // Element is a single struct value (e.g.
                                // copy from another struct lvalue): use
                                // resolve_struct_source + emit_struct_copy.
                                StructDef *src_sd = NULL;
                                MLIR_ValueHandle src = resolve_struct_source(
                                    e, sc, elem, &src_sd);
                                if (src != MLIR_INVALID_HANDLE && src_sd == sd) {
                                    emit_struct_copy(e, pelem, src, sd);
                                } else {
                                    EMIT_ERR(e, "array-of-struct initializer "
                                                "element must be a brace list "
                                                "or struct value");
                                }
                            }
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_STRUCT) {
                StructDef *sd = find_struct(e, st->decl_type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct type {}", st->decl_type.struct_name);
                    return;
                }
                sy->sdef = sd;
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                sy->addr = emit_alloca(e, st_ty);
                int32_t *prefix = arena_new_array(e->arena, int32_t, 1);
                prefix[0] = 0;
                emit_struct_zero(e, sy->addr, st_ty, sd, prefix, 1);
                if (st->decl_init) {
                    // Struct rhs from a struct-returning call: `S x = f(...)`.
                    if (st->decl_init->kind == EX_CALL) {
                        FuncSig *sig = find_sig(e, st->decl_init->callee);
                        if (sig && sig->ret.type.kind == TY_STRUCT &&
                            sig->ret.sdef == sd) {
                            emit_flat_call(e, sc, sig, st->decl_init->args, NULL, sy->addr);
                            return;
                        }
                    }
                    // Struct rhs from another struct lvalue or *p:
                    // `S x = other;` / `S x = ht->buckets[i].field;` /
                    // `S x = *p;` — copy the struct field-by-field.
                    if (st->decl_init->kind != EX_COMPOUND) {
                        StructDef *src_sd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(
                            e, sc, st->decl_init, &src_sd);
                        if (src != MLIR_INVALID_HANDLE && src_sd == sd) {
                            emit_struct_copy(e, sy->addr, src, sd);
                            return;
                        }
                    }
                    if (st->decl_init->kind != EX_COMPOUND ||
                        st->decl_init->cast_type->kind != TY_STRUCT ||
                        !str_eq(st->decl_init->cast_type->struct_name,
                                st->decl_type.struct_name)) {
                        EMIT_ERR(e, "struct initializer must be a "
                                    "matching compound literal");
                    } else {
                        size_t n = st->decl_init->args.size;
                        if (n > sd->fields.size) {
                            EMIT_ERR(e, "too many initializers");
                            n = sd->fields.size;
                        }
                        for (size_t i = 0; i < n; i++) {
                            // Map positional or designated initializer
                            // entry to the target field index. For
                            // designated entries (.fname = v), look up
                            // fname in the struct's field list.
                            int fidx = (int)i;
                            if (st->decl_init->compound_field_names) {
                                string fname = st->decl_init->compound_field_names[i];
                                if (fname.size != 0) {
                                    fidx = struct_field_index(sd, fname);
                                    if (fidx < 0) {
                                        EMIT_ERR(e, "unknown struct field {} in initializer", fname);
                                        continue;
                                    }
                                }
                            }
                            Type ft = sd->fields.data[fidx].type;
                            int32_t path[2] = {0, (int32_t)fidx};
                            MLIR_ValueHandle p = emit_gep(
                                e, sy->addr, st_ty, path, 2, NULL, 0);
                            if (ft.kind == TY_STRUCT) {
                                // Nested struct field: copy from the
                                // initializer's source struct field-by-field
                                // (a single emit_store_v would only copy the
                                // first 8 bytes).
                                StructDef *fsd = find_struct(e, ft.struct_name);
                                if (!fsd) {
                                    EMIT_ERR(e, "unknown struct type for "
                                                "nested-struct field "
                                                "initializer");
                                    continue;
                                }
                                Expr *ae = st->decl_init->args.data[i];
                                if (ae->kind == EX_INT && ae->int_value == 0) {
                                    continue; // already zero-initialised
                                }
                                // Nested aggregate `{ ... }` for this
                                // struct field: parser leaves cast_type
                                // unset (TY_VOID); fill it in here so
                                // emit_expr's EX_COMPOUND path resolves
                                // the field's struct type.
                                if (ae->kind == EX_COMPOUND &&
                                    ae->cast_type->kind == TY_VOID) {
                                    *ae->cast_type = ft;
                                }
                                StructDef *src_sd = NULL;
                                MLIR_ValueHandle src = resolve_struct_source(
                                    e, sc, ae, &src_sd);
                                if (src != MLIR_INVALID_HANDLE && src_sd == fsd) {
                                    emit_struct_copy(e, p, src, fsd);
                                } else {
                                    EVal vv = emit_expr(e, sc, ae);
                                    if (vv.is_ptr && vv.val != MLIR_INVALID_HANDLE) {
                                        emit_struct_copy(e, p, vv.val, fsd);
                                    } else {
                                        EMIT_ERR(e, "nested-struct field "
                                                    "initializer needs a "
                                                    "struct value");
                                    }
                                }
                                continue;
                            }
                            EVal v = emit_expr(
                                e, sc, st->decl_init->args.data[i]);
                            MLIR_ValueHandle sv;
                            if (ft.kind == TY_PTR_I32 ||
                                ft.kind == TY_PTR_CHAR ||
                                ft.kind == TY_PTR_VOID ||
                                ft.kind == TY_PTR_STRUCT ||
                                ft.kind == TY_FNPTR ||
                                ft.kind == TY_PTR_PTR) {
                                sv = int_to_ptr(e, v);
                            } else if (ft.kind == TY_I64) {
                                sv = coerce_eval_to(e, v, e->i64, ft);
                            } else if (ft.kind == TY_F64) {
                                sv = coerce_eval_to(e, v, e->f64, (Type){0});
                            } else {
                                sv = coerce_eval_to(
                                    e, v, scalar_mem_type(e, &ft),
                                    ft);
                            }
                            emit_store_v(e, sv, p);
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_I64) {
                if (!e->has_goto && !st->decl_is_static &&
                    st->decl_init != NULL && !is_mutated(e, st->decl_name)) {
                    EVal v = emit_expr(e, sc, st->decl_init);
                    sy->addr = coerce_eval_to(e, v, e->i64, st->decl_type);
                    sy->is_ssa = true;
                } else {
                    sy->addr = emit_alloca(e, e->i64);
                    if (st->decl_init) {
                        EVal v = emit_expr(e, sc, st->decl_init);
                        MLIR_ValueHandle iv = coerce_eval_to(e, v, e->i64,
                                                             st->decl_type);
                        emit_store_v(e, iv, sy->addr);
                    } else {
                        emit_store_v(e, emit_const_i64(e, 0), sy->addr);
                    }
                }
            } else if (st->decl_type.kind == TY_F32) {
                if (!e->has_goto && !st->decl_is_static &&
                    st->decl_init != NULL && !is_mutated(e, st->decl_name)) {
                    EVal v = emit_expr(e, sc, st->decl_init);
                    sy->addr = coerce_eval_to(e, v, e->f32, (Type){0});
                    sy->is_ssa = true;
                } else {
                    sy->addr = emit_alloca(e, e->f32);
                    if (st->decl_init) {
                        EVal v = emit_expr(e, sc, st->decl_init);
                        MLIR_ValueHandle iv = coerce_eval_to(e, v, e->f32, (Type){0});
                        emit_store_v(e, iv, sy->addr);
                    } else {
                        emit_store_v(e, emit_const_f32(e, 0.0), sy->addr);
                    }
                }
            } else if (st->decl_type.kind == TY_F64) {
                if (!e->has_goto && !st->decl_is_static &&
                    st->decl_init != NULL && !is_mutated(e, st->decl_name)) {
                    EVal v = emit_expr(e, sc, st->decl_init);
                    sy->addr = coerce_eval_to(e, v, e->f64, (Type){0});
                    sy->is_ssa = true;
                } else {
                    sy->addr = emit_alloca(e, e->f64);
                    if (st->decl_init) {
                        EVal v = emit_expr(e, sc, st->decl_init);
                        MLIR_ValueHandle iv = coerce_eval_to(e, v, e->f64, (Type){0});
                        emit_store_v(e, iv, sy->addr);
                    } else {
                        emit_store_v(e, emit_const_f64(e, 0.0), sy->addr);
                    }
                }
            } else {
                if (!e->has_goto && !st->decl_is_static &&
                    st->decl_init != NULL && !is_mutated(e, st->decl_name)) {
                    sy->wide_slot = type_is_narrow_int(st->decl_type);
                    sy->addr = coerce_eval_to(e, emit_expr(e, sc, st->decl_init),
                                              e->i32, st->decl_type);
                    sy->is_ssa = true;
                } else {
                    MLIR_TypeHandle mt = local_slot_type(e, st->decl_name, &st->decl_type,
                                                         &sy->wide_slot);
                    sy->addr = emit_alloca(e, mt);
                    if (st->decl_init) {
                        MLIR_ValueHandle v = coerce_eval_to(e, emit_expr(e, sc, st->decl_init),
                                                            mt, st->decl_type);
                        emit_store_v(e, v, sy->addr);
                    } else {
                        emit_store_v(e, emit_zero_of(e, mt), sy->addr);
                    }
                }
            }
            return;
        }
        case ST_RETURN: {
            FuncSig *sig = e->cur_sig;
            bool is_void_fn = (sig && sig->ret.type.kind == TY_VOID);
            if (is_void_fn) {
                if (st->expr) {
                    EMIT_ERR(e, "void function cannot return a value");
                }
                emit_return_op(e, NULL, 0);
                e->terminated = true;
                return;
            }
            if (!st->expr) {
                EMIT_ERR(e, "non-void function requires a return value");
                emit_return_op(e, NULL, 0);
                e->terminated = true;
                return;
            }
            if (sig && sig->ret.type.kind == TY_STRUCT) {
                StructDef *want = sig->ret.sdef;
                MLIR_ValueHandle out = e->cur_sret_ptr;
                if (st->expr->kind == EX_CALL) {
                    FuncSig *csig = find_sig(e, st->expr->callee);
                    if (!csig || csig->ret.type.kind != TY_STRUCT || csig->ret.sdef != want) {
                        EMIT_ERR(e, "returned call type mismatch");
                    } else {
                        emit_flat_call(e, sc, csig, st->expr->args, NULL, out);
                    }
                } else if (st->expr->kind == EX_VAR) {
                    Sym *s = lookup(e, sc, st->expr->name);
                    if (!s || s->type.kind != TY_STRUCT || s->sdef != want) {
                        EMIT_ERR(e, "returned struct type mismatch");
                    } else {
                        emit_struct_copy(e, out, sym_addr(e, s), want);
                    }
                } else if (st->expr->kind == EX_COMPOUND &&
                           st->expr->cast_type->kind == TY_STRUCT) {
                    EVal cv = emit_expr(e, sc, st->expr);
                    if (cv.sdef != want) {
                        EMIT_ERR(e, "returned compound literal type mismatch");
                    } else {
                        emit_struct_copy(e, out, cv.val, want);
                    }
                } else {
                    StructDef *src_sd = NULL;
                    MLIR_ValueHandle src = resolve_struct_source(e, sc, st->expr, &src_sd);
                    if (src != MLIR_INVALID_HANDLE && src_sd == want) {
                        emit_struct_copy(e, out, src, want);
                    } else {
                        EMIT_ERR(e, "struct return must be a variable or struct call");
                    }
                }
                emit_return_op(e, NULL, 0);
                e->terminated = true;
                return;
            }
            MLIR_TypeHandle want_ty;
            if (sig && sig->ret.type.kind == TY_F32) want_ty = e->f32;
            else if (sig && sig->ret.type.kind == TY_F64) want_ty = e->f64;
            else if (sig && sig->ret.type.kind == TY_I64) want_ty = e->i64;
            else if (sig && (sig->ret.type.kind == TY_PTR_STRUCT ||
                             sig->ret.type.kind == TY_PTR_I32 ||
                             sig->ret.type.kind == TY_PTR_CHAR ||
                             sig->ret.type.kind == TY_PTR_VOID ||
                             sig->ret.type.kind == TY_FNPTR ||
                             sig->ret.type.kind == TY_PTR_PTR)) want_ty = e->ptr;
            else want_ty = e->i32;
            EVal v = emit_expr(e, sc, st->expr);
            MLIR_ValueHandle ret_v;
            if (want_ty == e->ptr) {
                ret_v = int_to_ptr(e, v);
            } else {
                ret_v = coerce_eval_to(e, v, want_ty,
                                       sig ? sig->ret.type : (Type){0});
            }
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = ret_v;
            emit_return_op(e, ops, 1);
            e->terminated = true;
            return;
        }
        case ST_BREAK: {
            if (!e->loops) { EMIT_ERR(e, "break outside of loop"); return; }
            emit_branch(e, e->loops->break_block);
            return;
        }
        case ST_CONTINUE: {
            if (!e->loops) { EMIT_ERR(e, "continue outside of loop"); return; }
            emit_branch(e, e->loops->continue_block);
            return;
        }
        case ST_LABEL: {
            struct LabelBlock *lb = e->labels;
            while (lb && !str_eq(lb->name, st->label_name)) lb = lb->next;
            if (!lb) {
                EMIT_ERR(e, "internal: label not pre-registered");
                return;
            }
            // Block was already appended to the func region in
            // collect_labels; just terminate the predecessor and
            // continue emission inside the label block.
            if (!e->terminated) emit_branch(e, lb->block);
            e->cur_block = lb->block;
            e->terminated = false;
            return;
        }
        case ST_GOTO: {
            struct LabelBlock *lb = e->labels;
            while (lb && !str_eq(lb->name, st->label_name)) lb = lb->next;
            if (!lb) {
                EMIT_ERR(e, "goto: undefined label '{}'", st->label_name);
                return;
            }
            emit_branch(e, lb->block);
            return;
        }
        case ST_BLOCK: {
            if (st->block_no_scope) {
                for (size_t i = 0; i < st->block_body.size && !e->terminated; i++)
                    emit_stmt(e, sc, st->block_body.data[i]);
                return;
            }
            Scope inner = (Scope){.head = NULL, .parent = sc};
            emit_block(e, &inner, st->block_body);
            return;
        }
        case ST_IF: {
            EVal c = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, c);

            MLIR_BlockHandle then_b = new_cfg_block(e);
            MLIR_BlockHandle else_b = st->has_else ? new_cfg_block(e) : MLIR_INVALID_HANDLE;
            MLIR_BlockHandle end_b  = new_cfg_block(e);
            emit_cond_branch(e, cb, then_b, st->has_else ? else_b : end_b);

            e->cur_block = then_b; e->terminated = false;
            { Scope inner = (Scope){.head = NULL, .parent = sc};
              for (size_t i = 0; i < st->then_body.size && !e->terminated; i++)
                  emit_stmt(e, &inner, st->then_body.data[i]); }
            emit_branch(e, end_b);

            if (st->has_else) {
                e->cur_block = else_b; e->terminated = false;
                Scope inner = (Scope){.head = NULL, .parent = sc};
                for (size_t i = 0; i < st->else_body.size && !e->terminated; i++)
                    emit_stmt(e, &inner, st->else_body.data[i]);
                emit_branch(e, end_b);
            }

            e->cur_block = end_b; e->terminated = false;
            return;
        }
        case ST_WHILE: {
            MLIR_BlockHandle hdr_b  = new_cfg_block(e);
            MLIR_BlockHandle body_b = new_cfg_block(e);
            MLIR_BlockHandle exit_b = new_cfg_block(e);
            emit_branch(e, hdr_b);

            e->cur_block = hdr_b; e->terminated = false;
            EVal c = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, c);
            emit_cond_branch(e, cb, body_b, exit_b);

            e->cur_block = body_b; e->terminated = false;
            LoopCtx lc = (LoopCtx){.continue_block = hdr_b, .break_block = exit_b, .parent = e->loops};
            e->loops = &lc;
            { Scope inner = (Scope){.head = NULL, .parent = sc};
              for (size_t i = 0; i < st->while_body.size && !e->terminated; i++)
                  emit_stmt(e, &inner, st->while_body.data[i]); }
            e->loops = lc.parent;
            emit_branch(e, hdr_b);

            e->cur_block = exit_b; e->terminated = false;
            return;
        }
        case ST_DO_WHILE: {
            // do { body } while (cond); — body executes once before the
            // condition is tested. `continue` jumps to the cond-check.
            MLIR_BlockHandle body_b = new_cfg_block(e);
            MLIR_BlockHandle cond_b = new_cfg_block(e);
            MLIR_BlockHandle exit_b = new_cfg_block(e);
            emit_branch(e, body_b);

            e->cur_block = body_b; e->terminated = false;
            LoopCtx lc = (LoopCtx){.continue_block = cond_b, .break_block = exit_b, .parent = e->loops};
            e->loops = &lc;
            { Scope inner = (Scope){.head = NULL, .parent = sc};
              for (size_t i = 0; i < st->while_body.size && !e->terminated; i++)
                  emit_stmt(e, &inner, st->while_body.data[i]); }
            e->loops = lc.parent;
            emit_branch(e, cond_b);

            e->cur_block = cond_b; e->terminated = false;
            EVal c = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, c);
            emit_cond_branch(e, cb, body_b, exit_b);

            e->cur_block = exit_b; e->terminated = false;
            return;
        }
        case ST_FOR: {
            // The init declaration gets its own scope (`inner`), which the
            // condition and the step see. The body is a nested block scope,
            // so a body declaration that reuses the loop variable's name
            // shadows it only inside the body.
            Scope inner = (Scope){.head = NULL, .parent = sc};
            if (st->for_init) emit_stmt(e, &inner, st->for_init);
            if (e->terminated) return;

            MLIR_BlockHandle hdr_b  = new_cfg_block(e);
            MLIR_BlockHandle body_b = new_cfg_block(e);
            MLIR_BlockHandle step_b = new_cfg_block(e);
            MLIR_BlockHandle exit_b = new_cfg_block(e);
            emit_branch(e, hdr_b);

            e->cur_block = hdr_b; e->terminated = false;
            MLIR_ValueHandle cb;
            if (st->cond) {
                EVal c = emit_expr(e, &inner, st->cond);
                cb = emit_to_bool_i1(e, c);
            } else {
                MLIR_ValueHandle one = emit_const_i32(e, 1);
                cb = emit_cmpi(e, /*ne*/ 1, one, emit_const_i32(e, 0));
            }
            emit_cond_branch(e, cb, body_b, exit_b);

            e->cur_block = body_b; e->terminated = false;
            LoopCtx lc = (LoopCtx){.continue_block = step_b, .break_block = exit_b, .parent = e->loops};
            e->loops = &lc;
            { Scope body = (Scope){.head = NULL, .parent = &inner};
              for (size_t i = 0; i < st->for_body.size && !e->terminated; i++)
                  emit_stmt(e, &body, st->for_body.data[i]); }
            e->loops = lc.parent;
            emit_branch(e, step_b);

            e->cur_block = step_b; e->terminated = false;
            for (size_t i = 0; i < st->for_steps.size; i++) {
                emit_expr_for_side_effect(e, &inner, st->for_steps.data[i]);
            }
            emit_branch(e, hdr_b);

            e->cur_block = exit_b; e->terminated = false;
            return;
        }
        case ST_SWITCH: {
            // Lower switch as a chain of cf.cond_br tests followed by case
            // bodies that fall through to the next case unless terminated by
            // a `break`. Multiple SwitchCase entries pointing at the same
            // body_start share one CFG block.
            EVal cv = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cond_v = cv.val;

            size_t nc = st->switch_cases.size;
            // Statements before the first case label never run, but a
            // declaration among them is in scope for the rest of the body
            // (`switch (x) { int t; case 1: t = ...; }`). Declare those
            // names without running their initializers; skip the rest.
            Scope inner = (Scope){.head = NULL, .parent = sc};
            int64_t first_case = nc ? st->switch_cases.data[0].body_start
                                    : (int64_t)st->block_body.size;
            for (int64_t k = 0; k < first_case; k++)
                declare_without_init(e, &inner, st->block_body.data[k]);

            MLIR_BlockHandle exit_b = new_cfg_block(e);

            // Per-case body block, one per unique body_start.
            MLIR_BlockHandle *case_blocks = arena_new_array(e->arena, MLIR_BlockHandle, nc ? nc : 1);
            int64_t *body_start = arena_new_array(e->arena, int64_t, nc ? nc : 1);
            for (size_t i = 0; i < nc; i++) {
                int64_t bs = st->switch_cases.data[i].body_start;
                body_start[i] = bs;
                case_blocks[i] = MLIR_INVALID_HANDLE;
                for (size_t j = 0; j < i; j++) {
                    if (body_start[j] == bs) { case_blocks[i] = case_blocks[j]; break; }
                }
                if (case_blocks[i] == MLIR_INVALID_HANDLE) case_blocks[i] = new_cfg_block(e);
            }
            // Locate default block (if any) and emit the test cascade.
            MLIR_BlockHandle default_b = MLIR_INVALID_HANDLE;
            for (size_t i = 0; i < nc; i++) {
                if (st->switch_cases.data[i].is_default) { default_b = case_blocks[i]; break; }
            }
            for (size_t i = 0; i < nc; i++) {
                SwitchCase *c = &st->switch_cases.data[i];
                if (c->is_default) continue;
                MLIR_ValueHandle val = cv.is_i64 ? emit_const_i64(e, c->value)
                                                 : emit_const_i32(e, c->value);
                MLIR_ValueHandle eq = emit_cmpi(e, /*eq*/ 0, cond_v, val);
                MLIR_BlockHandle next_test = new_cfg_block(e);
                emit_cond_branch(e, eq, case_blocks[i], next_test);
                e->cur_block = next_test; e->terminated = false;
            }
            emit_branch(e, default_b != MLIR_INVALID_HANDLE ? default_b : exit_b);

            // Determine the contiguous stmt range for each case block.
            // Inherit `continue` target from any enclosing loop.
            MLIR_BlockHandle cont_target = e->loops ? e->loops->continue_block : MLIR_INVALID_HANDLE;
            LoopCtx lc = (LoopCtx){.continue_block = cont_target, .break_block = exit_b, .parent = e->loops};
            e->loops = &lc;
            for (size_t i = 0; i < nc; i++) {
                // Skip duplicate entries pointing at a body_start already emitted.
                bool dup = false;
                for (size_t j = 0; j < i; j++) if (body_start[j] == body_start[i]) { dup = true; break; }
                if (dup) continue;
                e->cur_block = case_blocks[i]; e->terminated = false;
                int64_t end = (int64_t)st->block_body.size;
                // Find the next case's body_start after this one.
                for (size_t j = 0; j < nc; j++) {
                    if (body_start[j] > body_start[i] && body_start[j] < end)
                        end = body_start[j];
                }
                for (int64_t k = body_start[i]; k < end && !e->terminated; k++) {
                    emit_stmt(e, &inner, st->block_body.data[k]);
                }
                // Fall through to the next case block in source order, or exit.
                if (!e->terminated) {
                    MLIR_BlockHandle nxt = exit_b;
                    int64_t best = -1;
                    for (size_t j = 0; j < nc; j++) {
                        if (body_start[j] > body_start[i] &&
                            (best < 0 || body_start[j] < best)) {
                            best = body_start[j];
                            nxt = case_blocks[j];
                        }
                    }
                    emit_branch(e, nxt);
                }
            }
            e->loops = lc.parent;
            e->cur_block = exit_b; e->terminated = false;
            return;
        }
    }
}

static void emit_block(E *e, Scope *sc, VecStmtPtr body) {
    for (size_t i = 0; i < body.size; i++) {
        Stmt *st = body.data[i];
        if (e->terminated && st->kind != ST_LABEL) continue;
        emit_stmt(e, sc, st);
    }
}

// ----- function & module -----

static void slot_resolve(E *e, Type ty, SlotInfo *out) {
    out->type = ty;
    out->sdef = NULL;
    if (ty.kind == TY_STRUCT || ty.kind == TY_PTR_STRUCT) {
        out->sdef = find_struct(e, ty.struct_name);
        if (!out->sdef) {
            EMIT_ERR(e, "unknown struct type {}", ty.struct_name);
        }
    } else if (ty.kind != TY_I32 && ty.kind != TY_I64 && ty.kind != TY_F32 &&
               ty.kind != TY_F64 &&
               ty.kind != TY_PTR_I32 && ty.kind != TY_PTR_CHAR &&
               ty.kind != TY_PTR_VOID && ty.kind != TY_VOID &&
               ty.kind != TY_FNPTR && ty.kind != TY_PTR_PTR &&
               ty.kind != TY_VA_LIST) {
        EMIT_ERR(e, "unsupported type in function signature");
    }
}

static MLIR_TypeHandle slot_param_type(E *e, SlotInfo *p) {
    if (p->type.kind == TY_STRUCT || p->type.kind == TY_PTR_STRUCT) return e->ptr;
    return scalar_mlir_type(e, p->type.kind);
}

static void build_signatures(E *e) {
    e->n_sigs = e->program->funcs.size;
    e->sigs = arena_new_array(e->arena, FuncSig, e->n_sigs ? e->n_sigs : 1);
    for (size_t f_i = 0; f_i < e->n_sigs; f_i++) {
        Func *f = e->program->funcs.data[f_i];
        FuncSig *sig = &e->sigs[f_i];
        *sig = (FuncSig){0};
        sig->name = f->name;
        sig->func = f;
        sig->is_variadic = f->is_variadic;
        slot_resolve(e, f->return_type, &sig->ret);
        sig->n_params = f->params.size;
        sig->params = arena_new_array(e->arena, SlotInfo, sig->n_params ? sig->n_params : 1);
        for (size_t i = 0; i < sig->n_params; i++) {
            slot_resolve(e, f->params.data[i].type, &sig->params[i]);
        }
        sig->sret = (sig->ret.type.kind == TY_STRUCT);
        // Note: variadic + sret is allowed. The sret pointer is the first
        // argument, followed by fixed args, then variadic args. The function
        // returns void via llvm.return.
        bool is_void = (sig->ret.type.kind == TY_VOID);
        // A variadic function lowered with tinyC's portable 8-byte cursor
        // va_list takes one extra hidden trailing `ptr` parameter (the
        // caller-packed varargs buffer). Use the cursor model only for
        // variadic functions DEFINED in this TU (tinyc va_list / wasm path).
        // Extern variadic callees (libc `open`, `printf`, `fcntl`, ...) lower
        // as real `llvm.func` variadic + `llvm.call` and bypass `func.func`.
        // is_forward distinguishes the cases: a prototype ends with ';' and
        // stays forward-only; a later definition replaces it and enables (A).
        // Extern decls never get a tinyc body, so they always take (B).
        //
        // Gate (B) on program->host_varargs (--host-varargs): native tests that
        // link host libc need real ABI for forward decls. wasm/ELF self-host
        // compiles each .c to a separate .wasm.o without --host-varargs; forward
        // decls there must keep cursor_va so calls match tinyc-defined printf
        // etc. in other objects (restores pre-fix wasm self-host behavior).
        sig->cursor_va = sig->is_variadic && f &&
            (!f->is_forward || !e->program->host_varargs);
        sig->defined_here = f && !f->is_forward;
        size_t in_total = sig->n_params + (sig->sret ? 1 : 0) +
                          (sig->cursor_va ? 1 : 0);
        size_t out_total = (sig->sret || is_void) ? 0 : 1;
        sig->flat_in_tys  = arena_new_array(e->arena, MLIR_TypeHandle, in_total ? in_total : 1);
        sig->flat_out_tys = arena_new_array(e->arena, MLIR_TypeHandle, out_total ? out_total : 1);
        size_t off = 0;
        if (sig->sret) sig->flat_in_tys[off++] = e->ptr;
        for (size_t i = 0; i < sig->n_params; i++) {
            sig->flat_in_tys[off++] = slot_param_type(e, &sig->params[i]);
        }
        if (sig->cursor_va) sig->flat_in_tys[off++] = e->ptr;  // hidden __va
        sig->n_flat_in = off;
        if (!sig->sret && !is_void) {
            sig->flat_out_tys[0] = scalar_mlir_type(e, sig->ret.type.kind);
            sig->n_flat_out = 1;
        } else {
            sig->n_flat_out = 0;
        }
        sig->fn_ty = MLIR_CreateTypeFunction(e->ctx,
            sig->flat_in_tys, sig->n_flat_in, sig->flat_out_tys, sig->n_flat_out);
        // Always compute llvm_fn_ty too — needed for variadic emit and for
        // static functions which we emit as `llvm.func` with internal linkage.
        // cursor_va functions are NOT real LLVM varargs (the hidden ptr param
        // already carries the args), so their llvm_fn_ty is non-variadic.
        {
            MLIR_TypeHandle ret_ty = (sig->n_flat_out == 1)
                ? sig->flat_out_tys[0]
                : MLIR_CreateTypeLLVMVoid(e->ctx);
            sig->llvm_fn_ty = MLIR_CreateTypeLLVMFunction(e->ctx,
                ret_ty, sig->flat_in_tys, sig->n_flat_in,
                /*is_var_arg=*/sig->is_variadic && !sig->cursor_va);
        }
    }
    if (e->n_sigs > 0) {
        size_t cap = 16;
        while (cap < e->n_sigs * 2u + 1u) cap <<= 1;
        e->sig_map = arena_new_array(e->arena, size_t, cap);
        memset(e->sig_map, 0, cap * sizeof(size_t));
        e->sig_map_cap = cap;
        size_t mask = cap - 1;
        for (size_t i = 0; i < e->n_sigs; i++) {
            size_t slot = emit_str_hash(e->sigs[i].name) & mask;
            while (e->sig_map[slot]) {
                if (str_eq(e->sigs[e->sig_map[slot] - 1].name, e->sigs[i].name)) break;
                slot = (slot + 1) & mask;
            }
            if (!e->sig_map[slot]) e->sig_map[slot] = i + 1;
        }
    }
}

static void record_mutated(E *e, string name) {
    if (name.size == 0) return;
    uint32_t bit = ((uint32_t)(unsigned char)name.str[0] * 31u + (uint32_t)name.size) & 63u;
    e->mutated_bloom |= (1ull << bit);
    if (e->n_mutated == e->cap_mutated) {
        size_t cap = e->cap_mutated ? e->cap_mutated * 2 : 16;
        string *na = arena_new_array(e->arena, string, cap);
        for (size_t i = 0; i < e->n_mutated; i++) na[i] = e->mutated[i];
        e->mutated = na;
        e->cap_mutated = cap;
    }
    e->mutated[e->n_mutated++] = name;
}

static bool is_mutated(E *e, string name) {
    if (e->n_mutated == 0 || name.size == 0) return false;
    uint32_t bit = ((uint32_t)(unsigned char)name.str[0] * 31u + (uint32_t)name.size) & 63u;
    if (((e->mutated_bloom >> bit) & 1u) == 0) return false;
    size_t n = name.size;
    char c0 = name.str[0];
    for (size_t i = 0; i < e->n_mutated; i++) {
        string m = e->mutated[i];
        if (m.size == n && m.str[0] == c0 && (n == 1 || memcmp(m.str + 1, name.str + 1, n - 1) == 0))
            return true;
    }
    return false;
}

// Record every `&name` and assignment target `name = ...` in `ex` / `st`.
static void collect_addr_taken_expr(E *e, Expr *ex) {
    if (!ex) return;
    if (ex->kind == EX_ADDR && ex->lhs && ex->lhs->kind == EX_VAR) {
        if (e->n_addr_taken == e->cap_addr_taken) {
            size_t cap = e->cap_addr_taken ? e->cap_addr_taken * 2 : 8;
            string *na = arena_new_array(e->arena, string, cap);
            for (size_t i = 0; i < e->n_addr_taken; i++) na[i] = e->addr_taken[i];
            e->addr_taken = na;
            e->cap_addr_taken = cap;
        }
        e->addr_taken[e->n_addr_taken++] = ex->lhs->name;
        record_mutated(e, ex->lhs->name);
    } else if (ex->kind == EX_ASSIGN && ex->lvalue && ex->lvalue->kind == EX_VAR) {
        record_mutated(e, ex->lvalue->name);
    }
    collect_addr_taken_expr(e, ex->lhs);
    collect_addr_taken_expr(e, ex->rhs);
    collect_addr_taken_expr(e, ex->lvalue);
    collect_addr_taken_expr(e, ex->rhs_assign);
    for (size_t i = 0; i < ex->args.size; i++) collect_addr_taken_expr(e, ex->args.data[i]);
}

static void collect_addr_taken(E *e, VecStmtPtr body);

static void collect_addr_taken_stmt(E *e, Stmt *st) {
    if (!st) return;
    if (st->kind == ST_LABEL) {
        struct LabelBlock *lb = arena_new(e->arena, struct LabelBlock);
        lb->name = st->label_name;
        lb->block = MLIR_CreateBlock(e->ctx);
        MLIR_AppendRegionBlock(e->ctx, e->func_region, lb->block);
        lb->next = e->labels;
        e->labels = lb;
    } else if (st->kind == ST_GOTO || st->kind == ST_SWITCH) {
        e->has_goto = true;
    }
    collect_addr_taken_expr(e, st->expr);
    collect_addr_taken_expr(e, st->decl_init);
    collect_addr_taken_expr(e, st->cond);
    collect_addr_taken(e, st->then_body);
    collect_addr_taken(e, st->else_body);
    collect_addr_taken(e, st->while_body);
    collect_addr_taken_stmt(e, st->for_init);
    for (size_t i = 0; i < st->for_steps.size; i++)
        collect_addr_taken_expr(e, st->for_steps.data[i]);
    collect_addr_taken(e, st->for_body);
    collect_addr_taken(e, st->block_body);
}

static void collect_addr_taken(E *e, VecStmtPtr body) {
    for (size_t i = 0; i < body.size; i++) collect_addr_taken_stmt(e, body.data[i]);
}

// Storage type of the local (or parameter) `name` of type `t`. A char /
// short / bool whose address the function never takes is kept in an i32 slot
// (*wide set): no pointer can observe its size, every store converts the
// value to its type and a load needs no extension. Otherwise its 1- / 2-byte
// storage (scalar_mem_type).
static MLIR_TypeHandle local_slot_type(E *e, string name, const Type *t, bool *wide) {
    MLIR_TypeHandle mt = scalar_mem_type(e, t);
    *wide = false;
    if (mt != e->i8 && mt != e->i16) return mt;
    for (size_t i = 0; i < e->n_addr_taken; i++)
        if (str_eq(e->addr_taken[i], name)) return mt;
    *wide = true;
    return e->i32;
}

static MLIR_OpHandle emit_func(E *e, Func *f) {
    e->cur_line = f->line;
    FuncSig *sig = find_sig(e, f->name);

    MLIR_RegionHandle body_r = MLIR_CreateRegion(e->ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(e->ctx);
    MLIR_AppendRegionBlock(e->ctx, body_r, entry);

    Scope sc = (Scope){.head = NULL, .parent = NULL};
    MLIR_ValueHandle flat_args_stk[16];
    MLIR_ValueHandle *flat_args = (sig->n_flat_in <= 16) ? flat_args_stk
        : arena_new_array(e->arena, MLIR_ValueHandle, sig->n_flat_in);
    for (size_t i = 0; i < sig->n_flat_in; i++) {
        flat_args[i] = MLIR_AddBlockArgument(e->ctx, entry,
                                             sig->flat_in_tys[i], eloc(e, 0));
    }

    MLIR_BlockHandle saved = e->cur_block;
    MLIR_RegionHandle saved_region = e->func_region;
    FuncSig *saved_sig = e->cur_sig;
    MLIR_ValueHandle saved_sret = e->cur_sret_ptr;
    MLIR_ValueHandle saved_va_args = e->cur_va_args;
    e->cur_block = entry;
    e->entry_block = entry;
    e->entry_const_one = emit_const_i64(e, 1);
    // entry_const_one is op #0 in entry; allocas hoisted from inner blocks
    // get inserted starting at index 1 to ensure they dominate all uses.
    e->entry_alloca_insert_idx = 1;
    e->func_region = body_r;
    e->cur_sig = sig;
    e->cur_sret_ptr = MLIR_INVALID_HANDLE;
    e->cur_va_args = MLIR_INVALID_HANDLE;
    e->terminated = false;
    e->next_ssa = 0;
    e->labels = NULL;
    // Single pre-pass: collect ST_LABEL blocks, &name addr-taken vars, and
    // mutated vars in one AST walk.
    e->n_addr_taken = 0;
    e->n_mutated = 0;
    e->mutated_bloom = 0;
    e->has_goto = false;
    collect_addr_taken(e, f->body);

    size_t arg_off = 0;
    if (sig->sret) {
        e->cur_sret_ptr = flat_args[arg_off++];
    }

    // cursor_va functions take a hidden trailing ptr param (the caller-packed
    // 8-byte-slot varargs buffer); va_start copies it into the va_list.
    if (sig->cursor_va) {
        e->cur_va_args = flat_args[sig->n_flat_in - 1];
    }

    for (size_t i = 0; i < sig->n_params; i++) {
        SlotInfo *p = &sig->params[i];
        Sym *sy = arena_new(e->arena, Sym);
        *sy = (Sym){0};
        sy->name = f->params.data[i].name;
        sy->type = p->type;
        MLIR_ValueHandle blk = flat_args[arg_off++];
        if (p->type.kind == TY_STRUCT) {
            // Caller-byval: the block-arg IS our copy. Bind directly.
            sy->sdef = p->sdef;
            sy->addr = blk;
        } else if (p->type.kind == TY_VA_LIST) {
            // va_list is passed as the alloca pointer of the caller's
            // 32-byte slot. Reuse it directly so va_arg / va_end operate
            // on the caller's slot.
            sy->addr = blk;
        } else if (!is_mutated(e, sy->name)) {
            // Unmutated scalar/pointer parameter: bind directly to block-arg SSA
            // value without emitting alloca + store + loads.
            if (p->type.kind == TY_PTR_STRUCT) sy->sdef = p->sdef;
            sy->wide_slot = (p->type.kind == TY_I32 && type_is_narrow_int(p->type));
            sy->addr = blk;
            sy->is_ssa = true;
        } else if (p->type.kind == TY_PTR_STRUCT) {
            sy->sdef = p->sdef;
            sy->addr = emit_alloca(e, e->ptr);
            emit_store_v(e, blk, sy->addr);
        } else if (p->type.kind == TY_F32) {
            sy->addr = emit_alloca(e, e->f32);
            emit_store_v(e, blk, sy->addr);
        } else if (p->type.kind == TY_F64) {
            sy->addr = emit_alloca(e, e->f64);
            emit_store_v(e, blk, sy->addr);
        } else if (p->type.kind == TY_I64) {
            sy->addr = emit_alloca(e, e->i64);
            emit_store_v(e, blk, sy->addr);
        } else if (p->type.kind == TY_PTR_I32 || p->type.kind == TY_PTR_CHAR ||
                   p->type.kind == TY_PTR_VOID || p->type.kind == TY_FNPTR ||
                   p->type.kind == TY_PTR_PTR) {
            sy->addr = emit_alloca(e, e->ptr);
            emit_store_v(e, blk, sy->addr);
        } else {
            // A char / short / bool parameter arrives as an i32 (in range:
            // the caller converted it); see local_slot_type for its storage.
            MLIR_TypeHandle mt = local_slot_type(e, sy->name, &p->type, &sy->wide_slot);
            sy->addr = emit_alloca(e, mt);
            emit_store_v(e, mt == e->i32 ? blk : emit_trunci_to(e, blk, mt), sy->addr);
        }
        sy->next = sc.head;
        sc.head = sy;
    }

    emit_block(e, &sc, f->body);
    if (!e->terminated) {
        if (sig->sret || sig->ret.type.kind == TY_VOID) {
            emit_return_op(e, NULL, 0);
        } else {
            MLIR_TypeHandle ty = sig->flat_out_tys[0];
            MLIR_ValueHandle v = (ty == e->f32) ? emit_const_f32(e, 0.0)
                : (ty == e->f64) ? emit_const_f64(e, 0.0)
                : (ty == e->i64) ? emit_const_i64(e, 0)
                : (ty == e->ptr) ? emit_null_ptr(e)
                : emit_const_i32(e, 0);
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
            emit_return_op(e, ops, 1);
        }
        e->terminated = true;
    }
    e->cur_block = saved;
    e->entry_block = MLIR_INVALID_HANDLE;
    e->func_region = saved_region;
    e->cur_sig = saved_sig;
    e->cur_sret_ptr = saved_sret;
    e->cur_va_args = saved_va_args;

    MLIR_NamedAttr sym_name = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(e->ctx, f->name));
    // Variadic functions are emitted as `llvm.func` with an LLVMFunctionType
    // attribute (which carries the var_arg flag); non-variadic ones stay as
    // `func.func` with a regular FunctionType. cursor_va functions are a
    // defined variadic lowered with a hidden ptr param — they are NOT real
    // LLVM varargs, so they emit as a plain `func.func`. Static functions add
    // an `llvm.linkage = #llvm.linkage<internal>` attribute that propagates to
    // LLVM IR through convert-func-to-llvm.
    bool real_variadic = sig->is_variadic && !sig->cursor_va;
    MLIR_TypeHandle ft = real_variadic ? sig->llvm_fn_ty : sig->fn_ty;
    MLIR_NamedAttr fn_ty_attr = MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(e->ctx, ft));
    // Up to 4 attrs: sym_name, function_type, llvm.linkage, wasm.export_name.
    MLIR_NamedAttr attrs[4];
    attrs[0] = sym_name; attrs[1] = fn_ty_attr;
    int n_attrs = 2;
    if (f->is_static) {
        attrs[n_attrs++] = MLIR_Named(MLIR_KEY_LLVM_LINKAGE, MLIR_CreateAttributeLLVMLinkageInternal(e->ctx));
    }
    // `__attribute__((__export_name__("foo")))` on a function definition
    // re-exports the function under a different wasm export name (used
    // e.g. by corec's wasm_buddy_alloc / wasm_buddy_free helpers).
    if (f->wasm_export_name.size > 0) {
        attrs[n_attrs++] = MLIR_Named(MLIR_KEY_WASM_EXPORT_NAME, MLIR_CreateAttributeString(e->ctx,
                                                      f->wasm_export_name));
    }
    MLIR_RegionHandle regs[1] = { body_r };
    MLIR_OpSpec fn_spec = {0};
    fn_spec.type = real_variadic ? OP_TYPE_LLVM_FUNC : OP_TYPE_FUNC_FUNC;
    fn_spec.attrs = attrs;
    fn_spec.n_attrs = (uint32_t)n_attrs;
    fn_spec.regions = regs;
    fn_spec.n_regions = 1;
    fn_spec.loc = eloc(e, 0);
    return MLIR_CreateOpFromSpec(e->ctx, &fn_spec);
}

// Compute byte size per a fixed layout: i32/f32 = 4, ptr = 8, struct =
// padded sum of fields (round each field to its alignment, struct align =
// max field align, total rounded up to that). Used for sizeof().
static int64_t type_align(E *e, Type t);

// Infer the static C-level Type of an expression for `sizeof(<expr>)`.
// We do NOT emit any operations — we walk the AST and look up symbols /
// struct fields by name. Returns TY_VOID to signal "unsupported" (the
// caller emits an error). Side-effecting expressions are still safe
// because we never evaluate them.
// A pointer or an array (which decays to one in arithmetic).
static bool type_is_pointerish(Type t) {
    switch (t.kind) {
        case TY_PTR_I32: case TY_PTR_CHAR: case TY_PTR_STRUCT: case TY_PTR_VOID:
        case TY_PTR_PTR: case TY_FNPTR: case TY_ARRAY_I32: case TY_ARRAY_F32:
        case TY_ARRAY_PTR_STRUCT: case TY_ARRAY_PTR_CHAR: case TY_ARRAY_STRUCT:
            return true;
        default:
            return false;
    }
}

// Decay an array type to the pointer type it converts to; other types are
// unchanged.
static void decay_array_type(E *e, Type *t) {
    TypeKind k = t->kind;
    if ((k == TY_ARRAY_PTR_CHAR || k == TY_ARRAY_PTR_STRUCT) && t->pointee) {
        // An array of pointers built by make_ptr_array keeps its element
        // pointer type.
        Type *pe = t->pointee;
        *t = (Type){0};
        t->kind = TY_PTR_PTR;
        t->pointee = pe;
        return;
    }
    if (k == TY_ARRAY_PTR_CHAR || k == TY_ARRAY_PTR_STRUCT) {
        // The element pointer type. The parser uses TY_ARRAY_PTR_STRUCT for
        // every non-char pointer array and keeps the element's other
        // fields: a struct name, a function-pointer signature, or the
        // TY_PTR_I32 flags (int / long long / float / double / void ...).
        Type *pe = arena_new(e->arena, Type);
        *pe = *t;
        pe->kind = k == TY_ARRAY_PTR_CHAR ? TY_PTR_CHAR
                 : t->struct_name.size ? TY_PTR_STRUCT
                 : t->fnptr_ret ? TY_FNPTR
                 : TY_PTR_I32;
        pe->array_len = 0;
        pe->array_len2 = 0;
        pe->array_len_expr = NULL;
        *t = (Type){0};
        t->kind = TY_PTR_PTR;
        t->pointee = pe;
        return;
    }
    if (k == TY_ARRAY_I32) {
        // A char element (int_bits 8, not bool) decays to a `char *`; a
        // short / bool element keeps its width on the TY_PTR_I32.
        t->kind = t->int_bits == 8 && !t->is_bool ? TY_PTR_CHAR : TY_PTR_I32;
        t->ptr_is_i64 = t->array_elem_is_i64;
    } else if (k == TY_ARRAY_F32) {
        t->kind = TY_PTR_I32;
        t->ptr_is_f32 = true;
    } else if (k == TY_ARRAY_STRUCT) {
        t->kind = TY_PTR_STRUCT;
    } else {
        return;
    }
    t->array_len = 0;
    t->array_len2 = 0;
    t->array_len_expr = NULL;
    t->array_elem_is_i64 = false;
}

// `NULL` or `(void *)0`: a pointer-typed null pointer constant.
static bool expr_is_null_ptr_const(Expr *x) {
    if (x->kind == EX_NULL) return true;
    return x->kind == EX_CAST && x->cast_type->kind == TY_PTR_VOID && x->lhs &&
           x->lhs->kind == EX_INT && x->lhs->int_value == 0;
}

// The int / char / short / bool type of an element of the integer array or
// the pointee of the integer pointer `t` (a char for a `char *`).
static Type int_elem_ctype(const Type *t) {
    Type r = (Type){0};
    r.kind = TY_I32;
    r.int_bits = t->int_bits;
    r.int_unsigned = t->int_unsigned;
    r.is_bool = t->is_bool;
    if (t->kind == TY_PTR_CHAR) r.int_bits = 8;
    return r;
}

static Type infer_expr_type(E *e, Scope *sc, Expr *ex) {
    Type t = (Type){0};
    switch (ex->kind) {
        case EX_INT:
            if (ex->is_i64) { t.kind = TY_I64; t.int_bits = 64; }
            else            { t.kind = TY_I32; t.int_bits = 32; }
            t.int_unsigned = ex->is_unsigned;
            return t;
        case EX_FLOAT: t.kind = ex->is_f64 ? TY_F64 : TY_F32; return t;
        case EX_STR:   t.kind = TY_PTR_CHAR; return t;
        case EX_NULL:  t.kind = TY_PTR_VOID; return t;
        case EX_VAR: {
            Sym *s = lookup(e, sc, ex->name);
            if (s) return s->type;
            int64_t en;
            if (find_enum(e, ex->name, &en)) { t.kind = TY_I32; return t; }
            return t;
        }
        case EX_DEREF: {
            Type inner = infer_expr_type(e, sc, ex->lhs);
            if (inner.kind == TY_PTR_I32 || inner.kind == TY_PTR_CHAR) {
                // The pointee, with its width and signedness (as EX_INDEX).
                if (inner.ptr_is_f32) { t.kind = TY_F32; return t; }
                if (inner.ptr_is_f64) { t.kind = TY_F64; return t; }
                if (inner.ptr_is_i64) {
                    t.kind = TY_I64;
                    t.int_bits = inner.int_bits; t.int_unsigned = inner.int_unsigned;
                    return t;
                }
                return int_elem_ctype(&inner);
            }
            if ((inner.kind == TY_ARRAY_I32 || inner.kind == TY_ARRAY_F32) &&
                inner.array_len2) {
                // `*m` of a 2D array is its first row, a 1D array.
                inner.array_len = inner.array_len2;
                inner.array_len2 = 0;
                inner.array_len_expr = NULL;
                return inner;
            }
            if (inner.kind == TY_ARRAY_F32) { t.kind = TY_F32; return t; }
            if (inner.kind == TY_ARRAY_I32) {
                if (inner.array_elem_is_i64) { t.kind = TY_I64; return t; }
                return int_elem_ctype(&inner);
            }
            // `*arr` / `*(arr + k)` on an array of pointers: its element.
            if ((inner.kind == TY_ARRAY_PTR_CHAR || inner.kind == TY_ARRAY_PTR_STRUCT) &&
                inner.pointee)
                return *inner.pointee;
            if (inner.kind == TY_ARRAY_PTR_CHAR) {
                t.kind = TY_PTR_CHAR;
                t.int_bits = inner.int_bits; t.int_unsigned = inner.int_unsigned;
                return t;
            }
            if (inner.kind == TY_ARRAY_PTR_STRUCT) {
                t.kind = TY_PTR_STRUCT; t.struct_name = inner.struct_name; return t;
            }
            if (inner.kind == TY_ARRAY_STRUCT) {
                t.kind = TY_STRUCT; t.struct_name = inner.struct_name; return t;
            }
            if (inner.kind == TY_PTR_PTR && inner.pointee) return *inner.pointee;
            if (inner.kind == TY_PTR_STRUCT) {
                t.kind = TY_STRUCT; t.struct_name = inner.struct_name; return t;
            }
            return t;
        }
        case EX_ADDR: {
            Type inner = infer_expr_type(e, sc, ex->lhs);
            if (inner.kind == TY_I32) {
                t = int_elem_ctype(&inner);
                t.kind = TY_PTR_I32;
                return t;
            }
            if (inner.kind == TY_I64) { t.kind = TY_PTR_I32; t.ptr_is_i64 = true; return t; }
            if (inner.kind == TY_F32) { t.kind = TY_PTR_I32; t.ptr_is_f32 = true; return t; }
            if (inner.kind == TY_F64) { t.kind = TY_PTR_I32; t.ptr_is_f64 = true; return t; }
            if (inner.kind == TY_STRUCT) {
                t.kind = TY_PTR_STRUCT; t.struct_name = inner.struct_name;
                return t;
            }
            // For other base kinds we report a generic pointer; sizeof
            // is the same regardless.
            t.kind = TY_PTR_VOID; return t;
        }
        case EX_FIELD: {
            Type st = infer_expr_type(e, sc, ex->lhs);
            StructDef *sd = NULL;
            if (st.kind == TY_STRUCT)      sd = find_struct(e, st.struct_name);
            else if (st.kind == TY_PTR_STRUCT) sd = find_struct(e, st.struct_name);
            if (!sd) return t;
            for (size_t i = 0; i < sd->fields.size; i++) {
                if (str_eq(sd->fields.data[i].name, ex->name))
                    return sd->fields.data[i].type;
            }
            return t;
        }
        case EX_INDEX: {
            Type base = infer_expr_type(e, sc, ex->lhs);
            if ((base.kind == TY_ARRAY_I32 || base.kind == TY_ARRAY_F32) && base.array_len2) {
                // A row `m[i]` of a 2D array is a 1D array.
                base.array_len = base.array_len2;
                base.array_len2 = 0;
                base.array_len_expr = NULL;
                return base;
            }
            if (base.kind == TY_ARRAY_I32 || base.kind == TY_PTR_I32) {
                if (base.kind == TY_PTR_I32 && base.ptr_is_f32) { t.kind = TY_F32; return t; }
                if (base.kind == TY_PTR_I32 && base.ptr_is_f64) { t.kind = TY_F64; return t; }
                if ((base.kind == TY_ARRAY_I32 && base.array_elem_is_i64) ||
                    (base.kind == TY_PTR_I32 && base.ptr_is_i64)) {
                    t.kind = TY_I64;
                    t.int_bits = base.int_bits; t.int_unsigned = base.int_unsigned;
                    return t;
                }
                return int_elem_ctype(&base);
            }
            if (base.kind == TY_PTR_CHAR) return int_elem_ctype(&base);
            if ((base.kind == TY_ARRAY_PTR_CHAR || base.kind == TY_ARRAY_PTR_STRUCT) &&
                base.pointee)
                return *base.pointee;
            if (base.kind == TY_ARRAY_PTR_CHAR) {
                t.kind = TY_PTR_CHAR;
                t.int_bits = base.int_bits; t.int_unsigned = base.int_unsigned;
                return t;
            }
            if (base.kind == TY_ARRAY_PTR_STRUCT) {
                t.kind = TY_PTR_STRUCT; t.struct_name = base.struct_name; return t;
            }
            if (base.kind == TY_ARRAY_F32) { t.kind = TY_F32; return t; }
            if (base.kind == TY_ARRAY_STRUCT) {
                t.kind = TY_STRUCT; t.struct_name = base.struct_name; return t;
            }
            if (base.kind == TY_PTR_STRUCT) {
                t.kind = TY_STRUCT; t.struct_name = base.struct_name; return t;
            }
            if (base.kind == TY_PTR_PTR && base.pointee) return *base.pointee;
            return t;
        }
        case EX_CALL: {
            FuncSig *sig = find_sig(e, ex->callee);
            if (sig) return sig->ret.type;
            // Indirect call through a fnptr-typed local/param/global.
            Sym *sy = lookup(e, sc, ex->callee);
            if (sy && sy->type.kind == TY_FNPTR && sy->type.fnptr_ret) {
                return *sy->type.fnptr_ret;
            }
            // Call through a callable expression (`s.f(x)`, `(*fp)(x)`).
            if (ex->callee.size == 0 && ex->lhs) {
                Type ft = infer_expr_type(e, sc, ex->lhs);
                if (ft.kind == TY_FNPTR && ft.fnptr_ret) return *ft.fnptr_ret;
            }
            return t;
        }
        case EX_CAST: return *ex->cast_type;
        case EX_COMPOUND: return *ex->cast_type;
        case EX_VA_ARG: return *ex->cast_type;
        case EX_SIZEOF:
            // size_t is `unsigned long` (corec/base/types.h, and clang):
            // typed as the parser types `unsigned long`, with 32-bit storage
            // on wasm32.
            t.kind = e->target_wasm32 ? TY_I32 : TY_I64;
            t.int_bits = 64;
            t.int_unsigned = true;
            return t;
        case EX_BIN: {
            // A comparison or logical operator gives an `int`.
            if (ex->op == OP_LT || ex->op == OP_LE || ex->op == OP_GT || ex->op == OP_GE ||
                ex->op == OP_EQ || ex->op == OP_NE || ex->op == OP_AND || ex->op == OP_OR)
                return t;
            Type a = infer_expr_type(e, sc, ex->lhs);
            Type b = infer_expr_type(e, sc, ex->rhs);
            // Pointer arithmetic (`p + i`, `i + p`, `p - i`) has the pointer's
            // type, with an array operand decayed (`sizeof(arr + 1)` is a
            // pointer's size); `p - q` is a ptrdiff_t (`long`), which tinyC
            // emits as an i64, or an i32 on wasm32.
            bool ap = type_is_pointerish(a), bp = type_is_pointerish(b);
            if (ap && bp) {
                t.kind = e->target_wasm32 ? TY_I32 : TY_I64;
                t.int_bits = 64;
                return t;
            }
            if (ap || bp) {
                Type *pt = ap ? &a : &b;
                decay_array_type(e, pt);
                return *pt;
            }
            // A shift has the promoted left operand's type; other operators
            // apply the usual arithmetic conversions.
            if (ex->op == OP_SHL || ex->op == OP_SHR) promote_int_type(&a);
            else if (type_is_arith(&a) && type_is_arith(&b)) arith_conv_type(&a, &b);
            return a;
        }
        case EX_UN:
        case EX_ASSIGN: {
            // `!x` is an `int`; `-x` / `~x` have x's promoted type; an
            // assignment has the lvalue's type.
            if (ex->kind == EX_UN && ex->op == OP_NOT) return t;
            Type a = infer_expr_type(e, sc, ex->lhs ? ex->lhs : ex->lvalue);
            if (ex->kind == EX_UN) promote_int_type(&a);
            return a;
        }
        case EX_TERNARY: {
            // For `c ? a : b` the result type is the common type of the
            // then/else branches (`a` / `b`), NOT the condition `c`.
            // tinyc's AST stores: lhs=cond, rhs=then, lvalue=else.
            // Arithmetic branches get the usual arithmetic conversions
            // (`1 ? 1 : 2LL` is a long long); an integer, `NULL` or
            // `(void *)0` next to a pointer is a null pointer constant, so
            // the result has the pointer's type; otherwise the then-branch
            // (rhs) decides, falling back
            // to lvalue or lhs only if rhs is missing (defensive).
            Expr *val = ex->rhs ? ex->rhs : (ex->lvalue ? ex->lvalue : ex->lhs);
            Type a = infer_expr_type(e, sc, val);
            if (ex->rhs && ex->lvalue) {
                Type b = infer_expr_type(e, sc, ex->lvalue);
                bool an = type_is_arith(&a), bn = type_is_arith(&b);
                if (an && bn) arith_conv_type(&a, &b);
                else if ((an || expr_is_null_ptr_const(ex->rhs)) && !bn &&
                         b.kind != TY_VOID) a = b;
            }
            // An array branch decays (`sizeof(z ? arr : p)` is a pointer's).
            decay_array_type(e, &a);
            return a;
        }
        default: return t;
    }
}

static int64_t type_size(E *e, Type t) {
    int64_t ptr_sz = e->target_wasm32 ? 4 : 8;
    if (t.kind == TY_I32) return tinyc_int_size(&t);
    if (t.kind == TY_I64) return 8;
    if (t.kind == TY_F32) return 4;
    if (t.kind == TY_F64) return 8;
    if (t.kind == TY_PTR_I32 || t.kind == TY_PTR_STRUCT) return ptr_sz;
    if (t.kind == TY_PTR_CHAR || t.kind == TY_PTR_VOID || t.kind == TY_FNPTR) return ptr_sz;
    if (t.kind == TY_PTR_PTR) return ptr_sz;
    if (t.kind == TY_ARRAY_I32)
        return tinyc_int_size(&t) * t.array_len * (t.array_len2 ? t.array_len2 : 1);
    if (t.kind == TY_ARRAY_F32) return 4 * t.array_len * (t.array_len2 ? t.array_len2 : 1);
    if (t.kind == TY_ARRAY_PTR_STRUCT || t.kind == TY_ARRAY_PTR_CHAR)
        return ptr_sz * t.array_len;
    if (t.kind == TY_STRUCT) {
        StructDef *sd = find_struct(e, t.struct_name);
        if (!sd) return 0;
        if (sd->is_union) {
            MLIR_TypeHandle elem; int64_t es, c;
            union_blob_layout(e, sd, &elem, &es, &c);
            return es * c;
        }
        int64_t off = 0, max_align = 1;
        for (size_t i = 0; i < sd->fields.size; i++) {
            Type ft = sd->fields.data[i].type;
            int64_t fa = type_align(e, ft);
            int64_t fs = type_size(e, ft);
            if (fa > max_align) max_align = fa;
            // round off up to fa
            off = (off + fa - 1) / fa * fa;
            off += fs;
        }
        if (max_align > 0) off = (off + max_align - 1) / max_align * max_align;
        return off;
    }
    if (t.kind == TY_ARRAY_STRUCT) {
        // Inline the TY_STRUCT branch by name rather than recursing
        // with a Type-by-value parameter — passing a Type with
        // `kind == TY_STRUCT` re-into `type_size` (which itself takes
        // `Type` by value) hit a clang -Os miscompile on wasm32-wasi
        // that returned 0 for the array's element size.
        StructDef *sd = find_struct(e, t.struct_name);
        if (!sd) return 0;
        if (sd->is_union) {
            MLIR_TypeHandle elem; int64_t es, c;
            union_blob_layout(e, sd, &elem, &es, &c);
            return es * c * t.array_len;
        }
        int64_t off = 0, max_align = 1;
        for (size_t i = 0; i < sd->fields.size; i++) {
            Type ft = sd->fields.data[i].type;
            int64_t fa = type_align(e, ft);
            int64_t fs = type_size(e, ft);
            if (fa > max_align) max_align = fa;
            off = (off + fa - 1) / fa * fa;
            off += fs;
        }
        if (max_align > 0) off = (off + max_align - 1) / max_align * max_align;
        return off * t.array_len;
    }
    return 0;
}
// C-level `sizeof` of an expression operand.
static int64_t c_sizeof_expr(E *e, Scope *sc, Expr *ex) {
    return type_size(e, infer_expr_type(e, sc, ex));
}
static int64_t type_align(E *e, Type t) {
    int64_t ptr_a = e->target_wasm32 ? 4 : 8;
    if (t.kind == TY_I32 || t.kind == TY_ARRAY_I32) return tinyc_int_size(&t);
    if (t.kind == TY_F32) return 4;
    if (t.kind == TY_I64 || t.kind == TY_F64) return 8;
    if (t.kind == TY_PTR_I32 || t.kind == TY_PTR_STRUCT) return ptr_a;
    if (t.kind == TY_PTR_CHAR || t.kind == TY_PTR_VOID || t.kind == TY_FNPTR) return ptr_a;
    if (t.kind == TY_PTR_PTR) return ptr_a;
    if (t.kind == TY_ARRAY_F32) return 4;
    if (t.kind == TY_ARRAY_PTR_STRUCT || t.kind == TY_ARRAY_PTR_CHAR) return ptr_a;
    if (t.kind == TY_STRUCT) {
        StructDef *sd = find_struct(e, t.struct_name);
        if (!sd) return 1;
        int64_t a = 1;
        for (size_t i = 0; i < sd->fields.size; i++) {
            int64_t fa = type_align(e, sd->fields.data[i].type);
            if (fa > a) a = fa;
        }
        return a;
    }
    if (t.kind == TY_ARRAY_STRUCT) {
        Type elt = (Type){.kind = TY_STRUCT, .struct_name = t.struct_name};
        return type_align(e, elt);
    }
    return 1;
}

static void union_blob_layout(E *e, StructDef *sd, MLIR_TypeHandle *elem,
                              int64_t *elem_size, int64_t *count) {
    int64_t max_sz = 0, max_al = 1;
    for (size_t i = 0; i < sd->fields.size; i++) {
        Type ft = sd->fields.data[i].type;
        int64_t s = type_size(e, ft);
        int64_t a = type_align(e, ft);
        if (s > max_sz) max_sz = s;
        if (a > max_al) max_al = a;
    }
    MLIR_TypeHandle et;
    int64_t es;
    if (max_al >= 8) { et = e->i64; es = 8; }
    else if (max_al >= 4) { et = e->i32; es = 4; }
    else if (max_al >= 2) { et = e->i16; es = 2; }
    else { et = e->i8; es = 1; }
    int64_t c = (max_sz + es - 1) / es;
    if (c < 1) c = 1;
    *elem = et;
    *elem_size = es;
    *count = c;
}


// only; pointer fields don't count). Standard 3-color DFS.
static bool struct_cycle_dfs(E *e, StructDef *sd, int *color, StructDef **stack, int depth) {
    e->cur_line = sd->line;
    size_t idx = 0;
    for (; idx < e->n_struct_types && e->struct_types[idx].sd != sd; idx++) {}
    if (idx >= e->n_struct_types) return false;
    if (color[idx] == 1) {
        EMIT_ERR(e, "by-value cyclic struct definition involving '{}'",
                sd->name);
        return true;
    }
    if (color[idx] == 2) return false;
    color[idx] = 1;
    for (size_t i = 0; i < sd->fields.size; i++) {
        Type ft = sd->fields.data[i].type;
        if (ft.kind == TY_STRUCT) {
            StructDef *inner = find_struct(e, ft.struct_name);
            if (inner && struct_cycle_dfs(e, inner, color, stack, depth + 1)) return true;
        }
    }
    color[idx] = 2;
    return false;
}

static void check_struct_cycles(E *e) {
    if (e->n_struct_types == 0) return;
    int *color = arena_new_array(e->arena, int, e->n_struct_types);
    for (size_t i = 0; i < e->n_struct_types; i++) color[i] = 0;
    StructDef **stack = arena_new_array(e->arena, StructDef *, e->n_struct_types);
    for (size_t i = 0; i < e->n_struct_types; i++) {
        if (color[i] == 0) {
            e->cur_line = e->struct_types[i].sd->line;
            if (struct_cycle_dfs(e, e->struct_types[i].sd, color, stack, 0)) {
                EMIT_ERR(e, "aborting due to cyclic struct definition");
                exit(1);
            }
        }
    }
}

// Pre-pass: register identified !llvm.struct types for each StructDef and
// then set their bodies (allowing already-registered types to be looked up
// when nested by-value structs reference each other).
static void init_struct_types(E *e) {
    e->n_struct_types = e->program->structs.size;
    e->struct_types = arena_new_array(e->arena, StructTypeEntry,
                                      e->n_struct_types ? e->n_struct_types : 1);
    for (size_t i = 0; i < e->n_struct_types; i++) {
        StructDef *sd = e->program->structs.data[i];
        e->struct_types[i].sd = sd;
        e->struct_types[i].ty = MLIR_CreateTypeLLVMStructIdentified(e->ctx, sd->name);
    }
    for (size_t i = 0; i < e->n_struct_types; i++) {
        StructDef *sd = e->struct_types[i].sd;
        e->cur_line = sd->line;
        if (sd->is_union) {
            // A union overlaps all members at offset 0. Emit its body as a
            // single blob field (array of alignment-wide integers) sized to
            // the largest member, so the llvm.struct size equals max(member
            // sizes) rather than their sum. All union member accesses are
            // normalised to offset 0 elsewhere (see walk_struct_lhs).
            MLIR_TypeHandle uelem; int64_t ues, uc;
            union_blob_layout(e, sd, &uelem, &ues, &uc);
            MLIR_TypeHandle *ubody = arena_new_array(e->arena, MLIR_TypeHandle, 1);
            ubody[0] = MLIR_CreateTypeLLVMArray(e->ctx, uelem, (uint64_t)uc);
            MLIR_SetTypeLLVMStructBody(e->ctx, e->struct_types[i].ty, ubody, 1);
            continue;
        }
        size_t n = sd->fields.size;
        MLIR_TypeHandle *body = arena_new_array(e->arena, MLIR_TypeHandle, n ? n : 1);
        for (size_t k = 0; k < n; k++) {
            Type ft = sd->fields.data[k].type;
            if (ft.kind == TY_I32 || ft.kind == TY_I64 || ft.kind == TY_F32 ||
                ft.kind == TY_F64 || ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID ||
                ft.kind == TY_PTR_PTR || ft.kind == TY_FNPTR) {
                body[k] = scalar_mem_type(e, &ft);
            } else if (ft.kind == TY_STRUCT) {
                StructDef *inner = find_struct(e, ft.struct_name);
                MLIR_TypeHandle t = find_struct_type(e, inner);
                body[k] = t;
            } else if (ft.kind == TY_ARRAY_I32 || ft.kind == TY_ARRAY_F32 ||
                       ft.kind == TY_ARRAY_PTR_STRUCT || ft.kind == TY_ARRAY_PTR_CHAR) {
                body[k] = array_mlir_type(e, &ft);
            } else {
                EMIT_ERR(e, "unsupported struct field type in {}", sd->name);
                body[k] = e->i32;
            }
        }
        MLIR_SetTypeLLVMStructBody(e->ctx, e->struct_types[i].ty, body, n);
    }
}

// ----- File-scope aggregate initializers -----
//
// A global struct, or array of structs, gets an initializer region that
// builds its constant value from `llvm.mlir.zero` with one `llvm.insertvalue`
// of an `llvm.mlir.constant` per non-zero scalar. For `struct S g = {1, 2.5};`
//   %0 = llvm.mlir.zero : !llvm.struct<"S", (i32, f64)>
//   %1 = llvm.mlir.constant(1 : i32) : i32
//   %2 = llvm.insertvalue %1, %0[0] : !llvm.struct<"S", (i32, f64)>
//   %3 = llvm.mlir.constant(2.5 : f64) : f64
//   %4 = llvm.insertvalue %3, %2[1] : !llvm.struct<"S", (i32, f64)>
//   llvm.return %4 : !llvm.struct<"S", (i32, f64)>
// Each backend folds the chain back into the global's data (see
// llvm_global_init_bytes). Every scalar must be a constant: an integer
// constant expression, a floating literal, or a cast of one; a pointer only
// a null pointer constant. The values are collected first, so a designator
// can override an earlier one, and braces may be elided as in C.

#define GINIT_MAX_DEPTH 16

// One write of an initializer at position `path`, in source order: a
// scalar (`val`, or INVALID for a 0 that replaces an earlier value), or
// the clearing of a whole member that a designator re-initializes. The
// last write to a position wins (see ginit_emit).
typedef struct {
    int64_t path[GINIT_MAX_DEPTH];
    size_t depth;
    MLIR_TypeHandle ty;
    MLIR_AttributeHandle val;
    bool clear;
} GInitSlot;

typedef struct {
    E *e;
    int64_t path[GINIT_MAX_DEPTH];    // position of the member being set
    size_t depth;
    GInitSlot *slots;                 // the writes so far
    size_t n, cap;
    bool overriding;                  // after a designator: writes may repeat
    int64_t top_count;                // elements of an unsized array
} GInit;

// A constant scalar initializer: an integer of a C type, or a float.
typedef struct {
    bool is_float;
    double f;
    FoldVal i;
} GConst;

static double gconst_to_float(GConst c, bool to_f32) {
    double f = c.f;
    if (!c.is_float) {
        f = (c.i.is_unsigned && c.i.is_i64) ? (double)(uint64_t)c.i.v : (double)c.i.v;
    }
    return to_f32 ? (double)(float)f : f;
}

// `v` wrapped into `bits` bits, held sign- or zero-extended.
static int64_t wrap_bits(int64_t v, int bits, bool is_unsigned) {
    if (bits >= 64) return v;
    uint64_t mask = ((uint64_t)1 << bits) - 1;
    uint64_t u = (uint64_t)v & mask;
    if (!is_unsigned && (u >> (bits - 1)) != 0) u |= ~mask;
    return (int64_t)u;
}

// Convert `c` to the integer C type `t` as C does: a float truncates toward
// zero, a bool becomes 0 / 1, and the value wraps into the type's width.
static int64_t gconst_to_int(GConst c, Type t) {
    if (t.is_bool) return c.is_float ? (c.f != 0.0) : (c.i.v != 0);
    int64_t v = c.i.v;
    if (c.is_float) {
        v = (t.int_unsigned && c.f >= 9223372036854775808.0)
            ? (int64_t)(uint64_t)c.f : (int64_t)c.f;
    }
    if (type_is_narrow_int(t)) return tinyc_narrow_const(v, t);
    return wrap_bits(v, t.kind == TY_I64 ? 64 : 32, t.int_unsigned);
}

// Evaluate a constant scalar initializer.
static bool gconst_eval(E *e, Expr *ex, GConst *out) {
    *out = (GConst){0};
    if (!ex) return false;
    if (ex->kind == EX_FLOAT) {
        out->is_float = true;
        out->f = ex->float_value;
        return true;
    }
    if (ex->kind == EX_NULL) return true;
    if (ex->kind == EX_UN && ex->op == OP_NEG) {
        GConst a;
        if (gconst_eval(e, ex->lhs, &a) && a.is_float) {
            out->is_float = true;
            // Not `-a.f`: tinyC compiles a float negation as `0 - x`, which
            // loses the sign of -0.0 when tinyC compiles itself.
            out->f = a.f * -1.0;
            return true;
        }
    }
    if (ex->kind == EX_CAST) {
        GConst a;
        if (!gconst_eval(e, ex->lhs, &a)) return false;
        Type t = *ex->cast_type;
        if (t.kind == TY_F32 || t.kind == TY_F64) {
            out->is_float = true;
            out->f = gconst_to_float(a, t.kind == TY_F32);
            return true;
        }
        if (t.kind == TY_I32 || t.kind == TY_I64) {
            out->i.v = gconst_to_int(a, t);
            out->i.is_i64 = t.kind == TY_I64;
            out->i.is_unsigned = t.int_unsigned && !type_is_narrow_int(t);
            return true;
        }
        // A pointer cast keeps the integer (only 0 is accepted).
        if (a.is_float) return false;
        *out = a;
        return true;
    }
    return ast_fold(e, NULL, ex, &out->i);
}

// True when `ex` is a constant zero (`0`, `NULL`, `0.0`).
static bool gconst_is_zero(E *e, Expr *ex) {
    GConst c;
    if (!gconst_eval(e, ex, &c)) return false;
    return c.is_float ? c.f == 0.0 : c.i.v == 0;
}

// Record a write at the current path: the constant `val` of scalar type
// `ty` (INVALID: 0), or with `clear` the zeroing of the whole member.
static void ginit_write(GInit *gi, MLIR_TypeHandle ty, MLIR_AttributeHandle val,
                        bool clear) {
    if (gi->n == gi->cap) {
        size_t cap = gi->cap ? gi->cap * 2 : 16;
        GInitSlot *ns = arena_new_array(gi->e->arena, GInitSlot, cap);
        for (size_t i = 0; i < gi->n; i++) ns[i] = gi->slots[i];
        gi->slots = ns;
        gi->cap = cap;
    }
    GInitSlot *sl = &gi->slots[gi->n++];
    for (size_t k = 0; k < gi->depth; k++) sl->path[k] = gi->path[k];
    sl->depth = gi->depth;
    sl->ty = ty;
    sl->val = val;
    sl->clear = clear;
}

static void ginit_set(GInit *gi, MLIR_TypeHandle ty, MLIR_AttributeHandle val) {
    ginit_write(gi, ty, val, false);
}

// A designator re-initializes a member that may already be set
// (`{.in = {1, 2}, .in = {3}}`): zero all of it.
static void ginit_clear(GInit *gi) {
    ginit_write(gi, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, true);
}

// Order writes by path (a member before what is inside it), then by
// source order.
static bool ginit_before(GInitSlot *a, size_t ia, GInitSlot *b, size_t ib) {
    size_t n = a->depth < b->depth ? a->depth : b->depth;
    for (size_t k = 0; k < n; k++) {
        if (a->path[k] != b->path[k]) return a->path[k] < b->path[k];
    }
    if (a->depth != b->depth) return a->depth < b->depth;
    return ia < ib;
}

static bool ginit_same_path(GInitSlot *a, GInitSlot *b) {
    if (a->depth != b->depth) return false;
    for (size_t k = 0; k < a->depth; k++) {
        if (a->path[k] != b->path[k]) return false;
    }
    return true;
}

// Emit the values that survive into the initializer region: one
// llvm.mlir.constant + llvm.insertvalue each, starting from `agg`. A
// scalar survives unless a later write has its path, or a later clear
// covers it. Sorting the writes by path makes that one pass.
static MLIR_ValueHandle ginit_emit(GInit *gi, MLIR_TypeHandle agg_ty, MLIR_ValueHandle agg) {
    E *e = gi->e;
    size_t n = gi->n;
    if (n == 0) return agg;
    // Bottom-up merge sort of the write indices.
    size_t *ord = arena_new_array(e->arena, size_t, n);
    size_t *tmp = arena_new_array(e->arena, size_t, n);
    for (size_t i = 0; i < n; i++) ord[i] = i;
    for (size_t w = 1; w < n; w *= 2) {
        for (size_t lo = 0; lo < n; lo += 2 * w) {
            size_t mid = lo + w < n ? lo + w : n;
            size_t hi = lo + 2 * w < n ? lo + 2 * w : n;
            size_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {
                if (ginit_before(&gi->slots[ord[j]], ord[j], &gi->slots[ord[i]], ord[i]))
                    tmp[k++] = ord[j++];
                else
                    tmp[k++] = ord[i++];
            }
            while (i < mid) tmp[k++] = ord[i++];
            while (j < hi) tmp[k++] = ord[j++];
        }
        size_t *t = ord; ord = tmp; tmp = t;
    }
    // The clears whose member contains the current write (the slot of
    // each), with the latest source position among them and the ones
    // outside them.
    size_t cidx[GINIT_MAX_DEPTH + 1];
    size_t cseq[GINIT_MAX_DEPTH + 1];
    size_t nc = 0;
    for (size_t r = 0; r < n; r++) {
        size_t i = ord[r];
        GInitSlot *sl = &gi->slots[i];
        while (nc > 0) {
            GInitSlot *cl = &gi->slots[cidx[nc - 1]];
            bool prefix = cl->depth <= sl->depth;
            for (size_t q = 0; prefix && q < cl->depth; q++) {
                if (cl->path[q] != sl->path[q]) prefix = false;
            }
            if (prefix) break;
            nc--;
        }
        if (sl->clear) {
            size_t seq = nc > 0 && cseq[nc - 1] > i ? cseq[nc - 1] : i;
            // A clear of the same member replaces the previous one.
            if (nc > 0 && gi->slots[cidx[nc - 1]].depth == sl->depth) nc--;
            cidx[nc] = i;
            cseq[nc] = seq;
            nc++;
            continue;
        }
        if (sl->val == MLIR_INVALID_HANDLE) continue;
        if (nc > 0 && cseq[nc - 1] > i) continue;
        if (r + 1 < n && ginit_same_path(sl, &gi->slots[ord[r + 1]])) continue;
        MLIR_ValueHandle c = MLIR_INVALID_HANDLE;
        MLIR_TypeHandle *crts = arena_new_array(e->arena, MLIR_TypeHandle, 1); crts[0] = sl->ty;
        MLIR_ValueHandle *crs = &c;
        MLIR_NamedAttr *cas = arena_new_array(e->arena, MLIR_NamedAttr, 1);
        cas[0] = MLIR_Named(MLIR_KEY_VALUE, sl->val);
        emit_op(e, OP_TYPE_LLVM_MLIR_CONSTANT,
                crts, 1, crs, 1, NULL, 0, cas, 1, NULL, 0);
        MLIR_ValueHandle rv = MLIR_INVALID_HANDLE;
        MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = agg_ty;
        MLIR_ValueHandle *rs = &rv;
        MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 2);
        ops[0] = agg;
        ops[1] = c;
        MLIR_NamedAttr *as = arena_new_array(e->arena, MLIR_NamedAttr, 1);
        as[0] = MLIR_Named(MLIR_KEY_POSITION, MLIR_CreateAttributeDenseI64Array(e->ctx,
                                                  sl->path, sl->depth));
        emit_op(e, OP_TYPE_LLVM_INSERTVALUE,
                rts, 1, rs, 1, ops, 2, as, 1, NULL, 0);
        agg = rv;
    }
    return agg;
}

static bool ginit_push(GInit *gi, int64_t idx) {
    if (gi->depth == GINIT_MAX_DEPTH) {
        EMIT_ERR(gi->e, "global initializer nests too deeply");
        return false;
    }
    gi->path[gi->depth++] = idx;
    return true;
}

// Set the scalar of C type `t`, stored as `ty`, at the current path.
static void ginit_scalar(GInit *gi, Type t, MLIR_TypeHandle ty, Expr *ex) {
    E *e = gi->e;
    GConst c;
    bool ok = gconst_eval(e, ex, &c);
    if (ty == e->ptr) {
        if (!ok || c.is_float || c.i.v != 0)
            EMIT_ERR(e, "only a null pointer can initialize a pointer in a global aggregate");
        // After a designator, a null pointer may replace an earlier value.
        else if (gi->overriding)
            ginit_set(gi, ty, MLIR_INVALID_HANDLE);
        return;
    }
    if (!ok) {
        EMIT_ERR(e, "global initializer element is not a constant");
        return;
    }
    if (ty == e->f32 || ty == e->f64) {
        ginit_set(gi, ty, MLIR_CreateAttributeFloat(e->ctx,
                                                    gconst_to_float(c, ty == e->f32), ty));
        return;
    }
    // Convert to the C type, then hold the value sign-extended from the
    // storage width (i8 for char / bool, i16 for short).
    int64_t v = wrap_bits(gconst_to_int(c, t),
                          ty == e->i8 ? 8 : ty == e->i16 ? 16 : ty == e->i64 ? 64 : 32, false);
    if (v != 0)
        ginit_set(gi, ty, MLIR_CreateAttributeInteger(e->ctx, v, ty));
    else if (gi->overriding)
        ginit_set(gi, ty, MLIR_INVALID_HANDLE);  // may replace an earlier value
}

// The members of an aggregate initialized from a list: the fields of a
// struct, the elements of an array (the rows of a 2-D array).
typedef struct {
    StructDef *sd;          // struct members, or NULL for an array
    int64_t count;          // number of members (-1: unbounded, `a[] = ...`)
} GMembers;

static bool ginit_is_aggregate(Type t) {
    return t.kind == TY_STRUCT || t.kind == TY_ARRAY_I32 || t.kind == TY_ARRAY_F32 ||
           t.kind == TY_ARRAY_STRUCT || t.kind == TY_ARRAY_PTR_CHAR ||
           t.kind == TY_ARRAY_PTR_STRUCT;
}

// The C type of member `k` of the aggregate `t`, and its MLIR storage type
// when it is a scalar (i8 / i16 for char, bool and short).
static Type ginit_member_type(GInit *gi, Type t, GMembers *m, int64_t k,
                              MLIR_TypeHandle *ty) {
    E *e = gi->e;
    Type mt = (Type){0};
    *ty = MLIR_INVALID_HANDLE;
    if (m->sd) {
        mt = m->sd->fields.data[k].type;
        if (!ginit_is_aggregate(mt)) {
            bool p = mt.kind == TY_PTR_I32 || mt.kind == TY_PTR_CHAR ||
                     mt.kind == TY_PTR_STRUCT || mt.kind == TY_PTR_VOID ||
                     mt.kind == TY_PTR_PTR || mt.kind == TY_FNPTR;
            *ty = p ? e->ptr
                : mt.kind == TY_I32 ? int_mem_type(e, &mt)
                : scalar_mlir_type(e, mt.kind);
        }
        return mt;
    }
    if (t.kind == TY_ARRAY_STRUCT) {
        mt.kind = TY_STRUCT;
        mt.struct_name = t.struct_name;
    } else if (t.array_len2 != 0) {
        mt = t;
        mt.array_len = t.array_len2;
        mt.array_len2 = 0;
    } else if (t.kind == TY_ARRAY_I32) {
        mt.kind = t.array_elem_is_i64 ? TY_I64 : TY_I32;
        mt.int_bits = t.array_elem_is_i64 ? 64
                    : type_is_narrow_int(t) ? t.int_bits : 32;
        mt.int_unsigned = t.int_unsigned;
        mt.is_bool = t.is_bool;
        *ty = array_elem_type(e, &t);
    } else if (t.kind == TY_ARRAY_F32) {
        mt.kind = TY_F32;
        *ty = e->f32;
    } else {
        mt.kind = TY_PTR_VOID;  // TY_ARRAY_PTR_CHAR / TY_ARRAY_PTR_STRUCT
        *ty = e->ptr;
    }
    return mt;
}

static void ginit_value(GInit *gi, Type t, MLIR_TypeHandle ty, Expr *ex);

// Initialize the members of the aggregate `t` at the current path from the
// list `list`, starting at list->args[*pos]. A braced list (`braced`) is
// the aggregate's own and must not have more initializers than members;
// otherwise the aggregate takes as many initializers as it needs from an
// enclosing list (brace elision: `{1, 2, 3}` for `struct {struct {int a,
// b;} in; int c;}`) and stops at a designator, which belongs to the
// enclosing list.
static void ginit_members_in(GInit *gi, Type t, Expr *list, size_t *pos, bool braced);

static void ginit_members(GInit *gi, Type t, Expr *list, size_t *pos, bool braced) {
    // A braced list starts a fresh subobject: the enclosing list cleared
    // it if it was set before.
    bool overriding = gi->overriding;
    if (braced) gi->overriding = false;
    ginit_members_in(gi, t, list, pos, braced);
    gi->overriding = overriding;
}

static void ginit_members_in(GInit *gi, Type t, Expr *list, size_t *pos, bool braced) {
    E *e = gi->e;
    GMembers m;
    m.sd = NULL;
    m.count = t.array_len;
    if (t.kind == TY_STRUCT) {
        m.sd = find_struct(e, t.struct_name);
        if (!m.sd) { EMIT_ERR(e, "unknown struct {} in global initializer", t.struct_name); return; }
        m.count = (int64_t)m.sd->fields.size;
        if (m.sd->is_union) {
            // Only zero is supported: consume one initializer (or the
            // whole braced list).
            size_t end = braced ? list->args.size : *pos + 1;
            for (; *pos < end && *pos < list->args.size; (*pos)++) {
                if (!gconst_is_zero(e, list->args.data[*pos]))
                    EMIT_ERR(e, "initializing a union in a global is not supported");
            }
            return;
        }
    } else if (gi->depth == 0 && t.array_len == 0 && !t.array_len_expr) {
        m.count = -1;  // `T a[] = {...}`: sized by its initializers
    }
    size_t start = *pos;
    int64_t k = 0;
    while (*pos < list->args.size) {
        Expr *a = list->args.data[*pos];
        string dn = list->compound_field_names ? list->compound_field_names[*pos]
                                               : (string){0};
        if (dn.size > 0 && (*pos != start || braced)) {
            if (!braced) return;
            if (!m.sd) {
                EMIT_ERR(e, "designated initializer for an array element");
                return;
            }
            bool found = false;
            for (size_t f = 0; f < m.sd->fields.size; f++) {
                if (str_eq(m.sd->fields.data[f].name, dn)) {
                    k = (int64_t)f; found = true; break;
                }
            }
            if (!found) {
                EMIT_ERR(e, "unknown field '{}' in designated initializer", dn);
                return;
            }
            gi->overriding = true;  // members may be set twice from here
        }
        if (m.count >= 0 && k >= m.count) {
            if (braced) EMIT_ERR(e, "too many initializers in a global aggregate");
            return;
        }
        if (!ginit_push(gi, k)) return;
        MLIR_TypeHandle mty;
        Type mt = ginit_member_type(gi, t, &m, k, &mty);
        bool char_array = mt.kind == TY_ARRAY_I32 && mt.array_len2 == 0 &&
                          mt.int_bits == 8 && !mt.is_bool;
        bool elided = ginit_is_aggregate(mt) && a->kind != EX_COMPOUND &&
                      !(char_array && a->kind == EX_STR);
        // A braced list or a string re-initializes the whole member (C11
        // 6.7.9p19); elided initializers only replace the scalars they set.
        if (gi->overriding && !elided) ginit_clear(gi);
        if (elided) {
            size_t before = *pos;
            int errs = e->err_count;
            ginit_members(gi, mt, list, pos, false);
            if (*pos == before) {
                // It has no scalars; report that once, innermost.
                if (e->err_count == errs)
                    EMIT_ERR(e, "initializer for an aggregate with no elements needs braces");
                gi->depth--;
                return;
            }
        } else {
            ginit_value(gi, mt, mty, a);
            (*pos)++;
        }
        gi->depth--;
        k++;
        if (gi->depth == 0 && m.count < 0 && k > gi->top_count) gi->top_count = k;
    }
}

// Set the value of C type `t` (stored as `ty` when a scalar) at the
// current path from the single initializer `ex`.
static void ginit_value(GInit *gi, Type t, MLIR_TypeHandle ty, Expr *ex) {
    E *e = gi->e;
    if (!ginit_is_aggregate(t)) {
        ginit_scalar(gi, t, ty, ex);
        return;
    }
    if (ex->kind == EX_COMPOUND) {
        size_t pos = 0;
        ginit_members(gi, t, ex, &pos, true);
        return;
    }
    // `char s[N] = "..."`: the string's bytes, cut or zero-padded to N.
    MLIR_TypeHandle ety;
    GMembers m;
    m.sd = NULL;
    m.count = t.array_len;
    Type et = ginit_member_type(gi, t, &m, 0, &ety);
    if (ex->kind == EX_STR && t.kind == TY_ARRAY_I32 && t.array_len2 == 0 &&
        et.int_bits == 8) {
        for (int64_t k = 0; k < t.array_len && k < (int64_t)ex->name.size; k++) {
            int64_t b = tinyc_narrow_const((int64_t)(uint8_t)ex->name.str[k], et);
            if (b == 0) continue;
            if (!ginit_push(gi, k)) return;
            ginit_set(gi, ety, MLIR_CreateAttributeInteger(e->ctx, b, ety));
            gi->depth--;
        }
        return;
    }
    EMIT_ERR(e, "an aggregate in a global initializer needs braces");
}

// Walk g->init_expr into `gi` (before the global's type is final: an
// unsized `T a[] = {...}` gets its length here).
static void ginit_collect(E *e, Global *g, GInit *gi) {
    *gi = (GInit){0};
    gi->e = e;
    ginit_value(gi, g->type, MLIR_INVALID_HANDLE, g->init_expr);
    if (g->type.kind != TY_STRUCT && g->type.array_len == 0 && !g->type.array_len_expr)
        g->type.array_len = gi->top_count;
}

// Emit the global `g` of the aggregate type `gty`: zero, or the values
// collected in `gi` (NULL: zero).
static MLIR_OpHandle emit_global_aggregate(E *e, Global *g, MLIR_TypeHandle gty, GInit *gi) {
    MLIR_BlockHandle init_blk = MLIR_INVALID_HANDLE;
    MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(e->ctx, g->name, gty,
        /*is_constant=*/g->is_const, /*init_kind=*/2, 0, 0.0, &init_blk, e->loc);
    MLIR_BlockHandle save_blk = e->cur_block;
    bool save_term = e->terminated;
    e->cur_block = init_blk;
    e->terminated = false;
    MLIR_ValueHandle zv = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle *zrts = arena_new_array(e->arena, MLIR_TypeHandle, 1); zrts[0] = gty;
    MLIR_ValueHandle *zrs = &zv;
    emit_op(e, OP_TYPE_LLVM_MLIR_ZERO,
            zrts, 1, zrs, 1, NULL, 0, NULL, 0, NULL, 0);
    MLIR_ValueHandle v = gi ? ginit_emit(gi, gty, zv) : zv;
    MLIR_ValueHandle *rops = arena_new_array(e->arena, MLIR_ValueHandle, 1);
    rops[0] = v;
    emit_op(e, OP_TYPE_LLVM_RETURN,
            NULL, 0, NULL, 0, rops, 1, NULL, 0, NULL, 0);
    e->cur_block = save_blk;
    e->terminated = save_term;
    return gop;
}

static int g_last_emit_errors = 0;

MLIR_OpHandle tinyc_emit_module(MLIR_Context *ctx, Program *program) {
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    E e = (E){0};
    e.ctx = ctx;
    e.arena = arena;
    e.i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    e.i1  = MLIR_CreateTypeInteger(ctx, 1, false);
    e.i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    e.i8  = MLIR_CreateTypeInteger(ctx, 8, true);
    e.i16 = MLIR_CreateTypeInteger(ctx, 16, true);
    e.f32 = MLIR_CreateTypeFloat(ctx, 32, false);
    e.f64 = MLIR_CreateTypeFloat(ctx, 64, false);
    e.index = MLIR_CreateTypeIndex(ctx);
    e.ptr = MLIR_CreateTypeLLVMPointer(ctx);
    e.target_wasm32 = program->target_wasm32;
    e.size_t_ty = program->target_wasm32 ? e.i32 : e.i64;
    e.loc = MLIR_CreateLocationUnknown(ctx, str_lit(""));
    e.next_ssa = 0;
    e.cur_block = MLIR_INVALID_HANDLE;
    e.func_region = MLIR_INVALID_HANDLE;
    e.terminated = false;
    e.loops = NULL;
    e.program = program;
    e.cur_sig = NULL;
    e.cur_sret_ptr = MLIR_INVALID_HANDLE;

    init_struct_types(&e);
    check_struct_cycles(&e);
    build_signatures(&e);

    MLIR_RegionHandle mr = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle mb = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, mr, mb);
    e.module_block = mb;
    MLIR_RegionHandle mregs[1] = { mr };
    MLIR_OpSpec mod_spec = {0};
    mod_spec.type = OP_TYPE_MODULE;
    mod_spec.regions = mregs;
    mod_spec.n_regions = 1;
    mod_spec.loc = e.loc;
    MLIR_OpHandle module = MLIR_CreateOpFromSpec(ctx, &mod_spec);

    // Pre-pass: emit module-level globals. For char* globals initialized
    // by a string literal, emit the string-literal global first, then the
    // pointer global with an initializer region that takes the address.
    for (size_t i = 0; i < program->globals.size; i++) {
        Global *g = &program->globals.data[i];
        e.cur_line = g->line;
        // Resolve a deferred array length (e.g. `[20 + 1]` or `[N]` with an
        // enumerator N) with no scope (file-scope constant expressions
        // only) before the symbol copies the type, so `sizeof` sees it.
        if (g->type.array_len_expr) {
            int64_t v = 0;
            if (!ast_fold_int(&e, NULL, g->type.array_len_expr, &v)) {
                EMIT_ERR(&e, "non-constant global array length");
            } else if (v <= 0) {
                EMIT_ERR(&e, "array length must be positive (got {})", v);
                v = 0;
            }
            g->type.array_len = v;
            // The parser packed the initializer without knowing the
            // length: cut or zero-pad its bytes to it.
            if (g->init_array_data.size > 0 && g->type.kind == TY_ARRAY_I32) {
                size_t es = (size_t)tinyc_int_size(&g->type);
                size_t n = (size_t)v * es;
                char *buf = arena_alloc(arena, n ? n : 1);
                for (size_t k = 0; k < n; k++)
                    buf[k] = k < g->init_array_data.size ? g->init_array_data.str[k] : 0;
                g->init_array_data = (string){.str = buf, .size = n};
            }
        }
        // A struct (array) initializer; an unsized `T a[] = {...}` gets its
        // length here.
        GInit *gi = NULL;
        if (g->init_expr) {
            gi = arena_new(arena, GInit);
            ginit_collect(&e, g, gi);
        }
        Sym *sy = arena_new(arena, Sym);
        *sy = (Sym){0};
        sy->name = g->name;
        sy->type = g->type;
        sy->is_global = true;
        sy->global_sym = g->name;
        sy->address_space = g->address_space;
        // Record the struct sdef on the symbol so indexed/field accesses
        // through the global (e.g. `g_arr[i].field`) can resolve fields.
        if (g->type.kind == TY_PTR_STRUCT || g->type.kind == TY_ARRAY_STRUCT
                || g->type.kind == TY_STRUCT) {
            sy->sdef = find_struct(&e, g->type.struct_name);
        }
        global_register(&e, sy);

        if (g->is_extern) {
            // `extern T x;` — emit external linkage with no initializer.
            // Used for libc symbols like `stdout` resolved by the linker.
            MLIR_TypeHandle gty;
            if (g->type.kind == TY_F32) {
                gty = e.f32;
            } else if (g->type.kind == TY_F64) {
                gty = e.f64;
            } else if (g->type.kind == TY_PTR_CHAR || g->type.kind == TY_PTR_I32 ||
                       g->type.kind == TY_PTR_STRUCT || g->type.kind == TY_PTR_VOID ||
                       g->type.kind == TY_FNPTR || g->type.kind == TY_PTR_PTR) {
                gty = e.ptr;
            } else if (g->type.kind == TY_I32) {
                gty = int_mem_type(&e, &g->type);
            } else {
                gty = e.i32;
            }
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, gty,
                /*is_constant=*/g->is_const,
                /*init_kind=*/4, 0, 0.0, NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_PTR_CHAR || g->type.kind == TY_PTR_I32 ||
                   g->type.kind == TY_PTR_STRUCT || g->type.kind == TY_PTR_VOID ||
                   g->type.kind == TY_FNPTR || g->type.kind == TY_PTR_PTR) {
            string str_sym = (string){0};
            if (g->has_init && g->init_str.size > 0) {
                str_sym = intern_string(&e, g->init_str);
            } else if (g->has_init && (g->init_int != 0 || g->init_float != 0.0)) {
                // `T *g = 5;` / `(T *)5`: not supported (it used to become
                // the null pointer, #365).
                EMIT_ERR(&e, "only a null pointer or a string can initialize a global pointer");
            }
            // Emit the !llvm.ptr global with an initializer region that
            // either returns the string addr or a null pointer.
            MLIR_BlockHandle init_blk = MLIR_INVALID_HANDLE;
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.ptr,
                /*is_constant=*/g->is_const,
                /*init_kind=*/2, 0, 0.0, &init_blk, e.loc);
            // Build initializer body: %0 = addressof @str ; llvm.return %0
            MLIR_BlockHandle save_blk = e.cur_block;
            bool save_term = e.terminated;
            e.cur_block = init_blk;
            e.terminated = false;
            MLIR_ValueHandle init_v;
            if (str_sym.size > 0) {
                init_v = emit_addressof(&e, str_sym, e.ptr);
            } else {
                init_v = emit_null_ptr(&e);
            }
            MLIR_ValueHandle *rops = arena_new_array(arena, MLIR_ValueHandle, 1);
            rops[0] = init_v;
            emit_op(&e, OP_TYPE_LLVM_RETURN,
                    NULL, 0, NULL, 0, rops, 1, NULL, 0, NULL, 0);
            e.cur_block = save_blk;
            e.terminated = save_term;
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_F32) {
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.f32,
                /*is_constant=*/g->is_const,
                /*init_kind=*/1, 0, g->has_init ? g->init_float : 0.0,
                NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_F64) {
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.f64,
                /*is_constant=*/g->is_const,
                /*init_kind=*/1, 0, g->has_init ? g->init_float : 0.0,
                NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_STRUCT) {
            StructDef *sd = find_struct(&e, g->type.struct_name);
            if (!sd) { EMIT_ERR(&e, "unknown struct in global"); continue; }
            MLIR_AppendBlockOp(ctx, mb,
                emit_global_aggregate(&e, g, find_struct_type(&e, sd), gi));
        } else if (g->type.kind == TY_ARRAY_STRUCT ||
                   g->type.kind == TY_ARRAY_I32 ||
                   g->type.kind == TY_ARRAY_PTR_CHAR ||
                   g->type.kind == TY_ARRAY_PTR_STRUCT) {
            // Aggregate global (its deferred array length is resolved
            // above).
            MLIR_TypeHandle elem;
            if (g->type.kind == TY_ARRAY_STRUCT) {
                StructDef *sd = find_struct(&e, g->type.struct_name);
                if (!sd) { EMIT_ERR(&e, "unknown struct in global array"); continue; }
                elem = find_struct_type(&e, sd);
            } else if (g->type.kind == TY_ARRAY_PTR_CHAR ||
                       g->type.kind == TY_ARRAY_PTR_STRUCT) {
                elem = e.ptr;
            } else {
                elem = array_elem_type(&e, &g->type);
            }
            MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                ctx, elem, (uint64_t)g->type.array_len);
            MLIR_OpHandle gop;
            string bytes = g->init_array_data;
            if (g->init_str.size > 0) {
                // `char s[N] = "..."`: the string's bytes, cut or
                // zero-padded to N.
                if (g->type.kind != TY_ARRAY_I32 || g->type.int_bits != 8 || g->type.is_bool ||
                    g->type.array_len2 != 0) {
                    EMIT_ERR(&e, "a string can only initialize a char array");
                }
                size_t n = (size_t)g->type.array_len;
                char *buf = arena_alloc(arena, n ? n : 1);
                bool any = false;
                for (size_t k = 0; k < n; k++) {
                    buf[k] = k < g->init_str.size ? g->init_str.str[k] : 0;
                    if (buf[k]) any = true;
                }
                bytes = any ? (string){.str = buf, .size = n} : (string){0};
            }
            if (bytes.size > 0 && g->type.kind == TY_ARRAY_I32) {
                // Non-zero aggregate initializer: emit the raw bytes
                // as the global's `value` STRING attribute. lower_global
                // (wasm) and emit_global (native) both interpret it as
                // packed little-endian element bytes.
                gop = MLIR_CreateLLVMGlobalArrayInit(ctx, g->name, arr_ty,
                    /*is_constant=*/g->is_const, bytes, e.loc);
            } else {
                gop = emit_global_aggregate(&e, g, arr_ty, gi);
            }
            // Record the struct sdef on the symbol for indexed access.
            if (g->type.kind == TY_ARRAY_STRUCT) {
                sy->sdef = find_struct(&e, g->type.struct_name);
            }
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_I64) {
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.i64,
                /*is_constant=*/g->is_const,
                /*init_kind=*/0, g->has_init ? g->init_int : 0, 0.0,
                NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else {
            // i32 (default for plain int); a char / short / bool initializer
            // is converted to the declared type.
            if (g->init_str.size > 0)
                EMIT_ERR(&e, "a string can only initialize a char array or a pointer");
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name,
                g->type.kind == TY_I32 ? int_mem_type(&e, &g->type) : e.i32,
                /*is_constant=*/g->is_const,
                /*init_kind=*/0,
                g->has_init ? tinyc_narrow_const(g->init_int, g->type) : 0, 0.0,
                NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        }
    }

    for (size_t i = 0; i < program->funcs.size; i++) {
        if (program->funcs.data[i]->is_forward) continue;
        MLIR_OpHandle fn = emit_func(&e, program->funcs.data[i]);
        MLIR_AppendBlockOp(ctx, mb, fn);
    }

    // Emit extern declarations for any forward-declared functions that
    // never received a definition. Without these, `func.call` / `llvm.call`
    // ops referring to them would fail symbol verification. Variadic decls
    // become `llvm.func` (so the variadic flag is preserved); non-variadic
    // ones use `func.func` matching the existing malloc/free pattern.
    for (size_t i = 0; i < program->funcs.size; i++) {
        Func *fwd = program->funcs.data[i];
        if (!fwd->is_forward) continue;
        FuncSig *sig = NULL;
        if (e.sig_map_cap > 0) {
            size_t mask = e.sig_map_cap - 1;
            size_t slot = emit_str_hash(fwd->name) & mask;
            while (e.sig_map[slot] != 0) {
                size_t idx = e.sig_map[slot] - 1;
                if (str_eq(e.sigs[idx].name, fwd->name)) { sig = &e.sigs[idx]; break; }
                slot = (slot + 1) & mask;
            }
        }
        if (!sig) continue;
        // Skip extern decls for forward-declared functions nobody
        // actually called. Without this, every prototype reachable
        // through an `#include` becomes a wasm import that the link
        // step can't resolve.
        if (!sig->is_used) continue;
        MLIR_NamedAttr a0 = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, fwd->name));
        bool decl_real_variadic = sig->is_variadic && !sig->cursor_va;
        MLIR_TypeHandle ft = decl_real_variadic ? sig->llvm_fn_ty : sig->fn_ty;
        MLIR_NamedAttr a1 = MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, ft));
        MLIR_NamedAttr a2 = MLIR_Named(MLIR_KEY_SYM_VISIBILITY, MLIR_CreateAttributeString(ctx, str_lit("private")));
        // Forward the `__attribute__((__import_module__("...")))` /
        // `__import_name__("...")` annotations onto the MLIR func op
        // so the wasm binary emitter can place this import into the
        // requested module (e.g. WASI's `wasi_snapshot_preview1`).
        size_t n_xtra = 0;
        if (fwd->wasm_import_module.size > 0) n_xtra++;
        if (fwd->wasm_import_name.size > 0) n_xtra++;
        MLIR_NamedAttr *attrs = arena_new_array(arena, MLIR_NamedAttr, 3 + n_xtra);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        size_t na = 3;
        if (fwd->wasm_import_module.size > 0) {
            attrs[na++] = MLIR_Named(MLIR_KEY_WASM_IMPORT_MODULE, MLIR_CreateAttributeString(ctx,
                                                    fwd->wasm_import_module));
        }
        if (fwd->wasm_import_name.size > 0) {
            attrs[na++] = MLIR_Named(MLIR_KEY_WASM_IMPORT_NAME, MLIR_CreateAttributeString(ctx,
                                                    fwd->wasm_import_name));
        }
        MLIR_OpSpec decl_spec = {0};
        decl_spec.type = decl_real_variadic ? OP_TYPE_LLVM_FUNC : OP_TYPE_FUNC_FUNC;
        decl_spec.attrs = attrs;
        decl_spec.n_attrs = (uint32_t)na;
        decl_spec.n_regions = 1;
        decl_spec.loc = e.loc;
        MLIR_OpHandle decl = MLIR_CreateOpFromSpec(ctx, &decl_spec);
        MLIR_AppendBlockOp(ctx, mb, decl);
    }

    // Emit `malloc`/`free` extern declarations at module scope so that
    // user code calling them links against libc — unless the user has
    // already provided their own forward declaration above (which would
    // otherwise create duplicate symbols and fail symbol verification).
    bool have_user_malloc = false, have_user_free = false;
    for (size_t i = 0; i < program->funcs.size; i++) {
        Func *fwd = program->funcs.data[i];
        // Skip the auto-decl if the user already has a forward decl OR a
        // full definition (corec-stdlib's stdlib/stdlib.c defines them).
        if (str_eq(fwd->name, str_lit("malloc"))) have_user_malloc = true;
        else if (str_eq(fwd->name, str_lit("free"))) have_user_free = true;
    }
    if (!have_user_malloc) {
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 1); ins[0] = e.size_t_ty;
        MLIR_TypeHandle *outs = arena_new_array(arena, MLIR_TypeHandle, 1); outs[0] = e.ptr;
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 1, outs, 1);
        MLIR_NamedAttr a0 = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("malloc")));
        MLIR_NamedAttr a1 = MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, fty));
        MLIR_NamedAttr a2 = MLIR_Named(MLIR_KEY_SYM_VISIBILITY, MLIR_CreateAttributeString(ctx, str_lit("private")));
        MLIR_NamedAttr *attrs = arena_new_array(arena, MLIR_NamedAttr, 3);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        MLIR_OpSpec decl_spec = {0};
        decl_spec.type = OP_TYPE_FUNC_FUNC;
        decl_spec.attrs = attrs;
        decl_spec.n_attrs = 3;
        decl_spec.n_regions = 1;
        decl_spec.loc = e.loc;
        MLIR_OpHandle decl = MLIR_CreateOpFromSpec(ctx, &decl_spec);
        MLIR_AppendBlockOp(ctx, mb, decl);
    }
    if (!have_user_free) {
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 1); ins[0] = e.ptr;
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 1, NULL, 0);
        MLIR_NamedAttr a0 = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("free")));
        MLIR_NamedAttr a1 = MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, fty));
        MLIR_NamedAttr a2 = MLIR_Named(MLIR_KEY_SYM_VISIBILITY, MLIR_CreateAttributeString(ctx, str_lit("private")));
        MLIR_NamedAttr *attrs = arena_new_array(arena, MLIR_NamedAttr, 3);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        MLIR_OpSpec decl_spec = {0};
        decl_spec.type = OP_TYPE_FUNC_FUNC;
        decl_spec.attrs = attrs;
        decl_spec.n_attrs = 3;
        decl_spec.n_regions = 1;
        decl_spec.loc = e.loc;
        MLIR_OpHandle decl = MLIR_CreateOpFromSpec(ctx, &decl_spec);
        MLIR_AppendBlockOp(ctx, mb, decl);
    }

    // __tinyc_syscall6 — the raw-syscall intrinsic stub. Declared (i64 x7) ->
    // i64; the native llvm->aarch64 backend synthesises its `svc` body. Only
    // emitted when user code used __builtin_syscall6.
    if (e.need_syscall6_stub) {
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 7);
        for (size_t k = 0; k < 7; k++) ins[k] = e.i64;
        MLIR_TypeHandle *outs = arena_new_array(arena, MLIR_TypeHandle, 1); outs[0] = e.i64;
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 7, outs, 1);
        MLIR_NamedAttr a0 = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("__tinyc_syscall6")));
        MLIR_NamedAttr a1 = MLIR_Named(MLIR_KEY_FUNCTION_TYPE, MLIR_CreateAttributeType(ctx, fty));
        MLIR_NamedAttr a2 = MLIR_Named(MLIR_KEY_SYM_VISIBILITY, MLIR_CreateAttributeString(ctx, str_lit("private")));
        MLIR_NamedAttr *attrs = arena_new_array(arena, MLIR_NamedAttr, 3);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        MLIR_OpSpec decl_spec = {0};
        decl_spec.type = OP_TYPE_FUNC_FUNC;
        decl_spec.attrs = attrs;
        decl_spec.n_attrs = 3;
        decl_spec.n_regions = 1;
        decl_spec.loc = e.loc;
        MLIR_OpHandle decl = MLIR_CreateOpFromSpec(ctx, &decl_spec);
        MLIR_AppendBlockOp(ctx, mb, decl);
    }

    // Emit any string-literal globals collected by EX_STR.
    for (size_t i = 0; i < e.n_strings; i++) {
        StringEntry *se = &e.strings[i];
        MLIR_OpHandle gop = MLIR_CreateLLVMGlobalString(
            ctx, se->sym, se->bytes, e.loc);
        MLIR_AppendBlockOp(ctx, mb, gop);
    }

    g_last_emit_errors = e.err_count;
    return module;
}

int tinyc_last_emit_errors(void) { return g_last_emit_errors; }
