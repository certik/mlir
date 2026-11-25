// Upstream LLVM/MLIR-backed implementation of the public API in mlir_api.h.
//
// All structural queries (getOperand/getResult/getAttrs/getRegions/...)
// go directly through the upstream MLIR C++ API. MLIR_Context is a subclass
// of mlir::MLIRContext and owns every piece of auxiliary state (there are no
// Zero-copy handle representation: ops, blocks, regions, values, types,
// attributes, and locations are raw MLIR pointers (or opaque pointers).
// The only side tables are for optional parser/printer register names and
// per-op user-attribute snapshots.
// exists because:
//   1. The C API creates `MLIR_ValueHandle` for a future block arg
//      *before* the block exists (CreateValueBlockArg, then later
//      AppendBlockArg), so the handle has to be an indirection slot.
//   2. Upstream MLIR does not track per-Value register names; the
//      printer's MLIR_GetValueRegisterName needs them for block args.
//
// All other handle types map directly to upstream pointers / opaque
// pointers, with no parallel storage.
//
// Coverage is the surface needed by tests/cross/driver.c. APIs outside
// that surface are stubbed with UNIMPLEMENTED().

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/IR/Module.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/Host.h"

extern "C" {
#include "mlir_api.h"
#include "mlir_api_upstream.h"
#include "mlir_op_names.h"
#include "mlir_parser.h"
#include "mlir_passes.h"
#include "mlir_generic_printer.h"
#include <platform/platform.h>
#include "mlir_printer.h"
}

// Contract checks (mlir_api.h, "Diagnostics"): with -DMLIR_API_CHECKS a
// violated precondition aborts; otherwise the checks compile to nothing.
#ifdef MLIR_API_CHECKS
#define API_CHECK(cond, fn, what)                                                   \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::fprintf(stderr, "MLIR API contract violation in %s: %s\n", fn, what); \
            std::abort();                                                           \
        }                                                                           \
    } while (0)
#else
#define API_CHECK(cond, fn, what) ((void)0)
#endif

#define UNIMPLEMENTED()                                                   \
    do {                                                                  \
        std::fprintf(stderr, "%s unimplemented (upstream)\n", __func__);  \
        std::abort();                                                     \
    } while (0)

namespace {

// The dialects every context loads.
mlir::DialectRegistry contextRegistry() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect,
                    mlir::affine::AffineDialect,
                    mlir::func::FuncDialect,
                    mlir::gpu::GPUDialect,
                    mlir::index::IndexDialect,
                    mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect,
                    mlir::memref::MemRefDialect,
                    mlir::tensor::TensorDialect,
                    mlir::vector::VectorDialect,
                    mlir::LLVM::LLVMDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::math::MathDialect>();
    return registry;
}

// Value handles are zero-copy: mlir::Value::getAsOpaquePointer() directly.
inline MLIR_ValueHandle valH(mlir::Value v) {
    return reinterpret_cast<MLIR_ValueHandle>(v.getAsOpaquePointer());
}
inline mlir::Value valF(MLIR_ValueHandle h) {
    return mlir::Value::getFromOpaquePointer(reinterpret_cast<void *>(h));
}
} // namespace

// The context. It is the mlir::MLIRContext itself (a subclass), so the
// context of any op, value, type or attribute is one getContext() and a
// static_cast away: functions that take only a handle reach the context's
// caches without a lookup. Everything the backend keeps lives here; there is
// no process-wide state.
struct MLIR_Context : public mlir::MLIRContext {
    explicit MLIR_Context(const mlir::DialectRegistry &registry)
        : mlir::MLIRContext(registry), unknownLoc(mlir::UnknownLoc::get(this)) {}

    Arena *arena = nullptr;  // current IR arena (owned by the caller)
    MLIR_DiagnosticHandler diagFn = nullptr;
    void *diagUser = nullptr;

    // Optional register names for parser/printer (MLIR_SetValueRegisterName).
    // Empty for all compiler/pass pipelines.
    llvm::DenseMap<const void *, std::string> valueNames;
    // Lazy per-op attribute cache populated only on demand when a printer
    // calls MLIR_GetOpNumAttributes / MLIR_GetOpAttribute / MLIR_GetOpAttributeKey.
    // Empty during all non-printing compiler/pass pipelines.
    std::unordered_map<const mlir::Operation *, std::vector<mlir::NamedAttribute>> opAttrCache;
    // Cached UnknownLoc so op creation never calls UnknownLoc::get(&ctx).
    mlir::Location unknownLoc;
    // Attribute keys (mlir_api.h): the names, the StringAttr of each key
    // (as an opaque pointer, created on first use) and the reverse map.
    // Static keys (1..MLIR_KEY_COUNT-1) are cached in a fixed array for a
    // single-load fast path in MLIR_GetOpAttr.
    AttrKeyTable keyTable;
    const void *staticKeyNames[MLIR_KEY_COUNT] = {};
    std::vector<const void *> keyNames;
    llvm::DenseMap<const void *, MLIR_AttrKey> keyOfName;
    // MLIR_GetOpAttr's i32 `case_values` conversions (caseValuesAsI32).
    llvm::DenseMap<const void *, const void *> caseValuesI32;
    // Op types (doc/mlir_api_design.md section 2.5). opTypeTable holds the
    // names of the static and dynamic types (in `persistent`); opNames[t] is
    // the mlir::OperationName of type t (as an opaque pointer), so creation
    // does no string work; opTypes maps it back (MLIR_GetOpType).
    Arena *persistent = nullptr;
    OpTypeTable opTypeTable;
    std::vector<const void *> opNames;
    llvm::DenseMap<const void *, MLIR_OpType> opTypes;
    struct OpTypeCacheEntry {
        const void *ptr = nullptr;
        MLIR_OpType type = OP_TYPE_UNREGISTERED;
    };
    OpTypeCacheEntry opTypeCache[64];
    // Printed form of each type (MLIR_GetTypeString).
    llvm::DenseMap<const void *, std::string> typeStrings;
    // Strings returned by MLIR_GetAttributeString that are not stored in an
    // attribute (`@sym` for symbol refs, the printed form of other kinds).
    // Their storage lives as long as the context.
    llvm::StringSet<> attrStrings;
    // Hidden block that owns block arguments created before their block
    // (MLIR_CreateValueBlockArg).
    mlir::Block *holderBlock = nullptr;
    // The IR nobody else owns: ops, blocks and regions created through the
    // API (or detached by it) that are not inside a parent. An object leaves
    // its set when it is inserted into a parent or erased, and joins it when
    // the API detaches it, so the sets only ever hold live objects that no
    // pass can see. MLIR_DestroyContext frees them, and with them all IR.
    llvm::DenseSet<mlir::Operation *> rootOps;
    llvm::DenseSet<mlir::Block *> rootBlocks;
    llvm::DenseSet<mlir::Region *> rootRegions;

    ~MLIR_Context() {
        delete holderBlock;
        if (persistent) arena_destroy(persistent);
    }
};

namespace {

inline MLIR_Context *ctxOf(mlir::MLIRContext *m) { return static_cast<MLIR_Context *>(m); }
inline MLIR_Context *ctxOf(mlir::Operation *op) { return ctxOf(op->getContext()); }

// An attribute handle is the mlir::Attribute's storage pointer: attributes
// are uniqued in the MLIRContext, so equal values are equal handles.
uintptr_t attrH(mlir::Attribute a) { return reinterpret_cast<uintptr_t>(a.getAsOpaquePointer()); }
mlir::Attribute attrF(uintptr_t h) {
    return mlir::Attribute::getFromOpaquePointer(reinterpret_cast<const void *>(h));
}

#if defined(__GNUC__) || defined(__clang__)
#define MLIR_UP_NOINLINE __attribute__((noinline))
#define MLIR_UP_LIKELY(x) __builtin_expect(!!(x), 1)
#define MLIR_UP_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define MLIR_UP_NOINLINE
#define MLIR_UP_LIKELY(x) (x)
#define MLIR_UP_UNLIKELY(x) (x)
#endif

// Slow path for dynamic or uncached attribute keys.
MLIR_UP_NOINLINE mlir::StringAttr keyNameSlow(MLIR_Context *c, MLIR_AttrKey key) {
    if (key >= c->keyNames.size()) c->keyNames.resize(c->keyTable.n_keys, nullptr);
    if (!c->keyNames[key]) {
        string n = attr_key_table_name(&c->keyTable, key);
        mlir::StringAttr a = mlir::StringAttr::get(c, llvm::StringRef(n.str, n.size));
        c->keyNames[key] = a.getAsOpaquePointer();
        c->keyOfName[a.getAsOpaquePointer()] = key;
        if (key < MLIR_KEY_COUNT) c->staticKeyNames[key] = a.getAsOpaquePointer();
    }
    return mlir::StringAttr::getFromOpaquePointer(c->keyNames[key]);
}

// The StringAttr of `key`. Static keys (1..MLIR_KEY_COUNT-1) are pre-populated
// in MLIR_CreateContext, so the hot path is a single fixed-array load.
inline mlir::StringAttr keyName(MLIR_Context *c, MLIR_AttrKey key) {
    API_CHECK(key != MLIR_KEY_INVALID && key < c->keyTable.n_keys, "MLIR attribute key",
              "not a key of this context");
    if (MLIR_UP_LIKELY(key < MLIR_KEY_COUNT))
        return mlir::StringAttr::getFromOpaquePointer(c->staticKeyNames[key]);
    if (MLIR_UP_LIKELY(key < c->keyNames.size() && c->keyNames[key] != nullptr))
        return mlir::StringAttr::getFromOpaquePointer(c->keyNames[key]);
    return keyNameSlow(c, key);
}

// The key of the attribute name `name` (interned on first use).
MLIR_AttrKey keyOf(MLIR_Context *c, mlir::StringAttr name) {
    auto it = c->keyOfName.find(name.getAsOpaquePointer());
    if (it != c->keyOfName.end()) return it->second;
    llvm::StringRef n = name.getValue();
    string sn;
    sn.str = const_cast<char *>(n.data());
    sn.size = n.size();
    MLIR_AttrKey k = attr_key_table_intern(&c->keyTable, sn);
    c->keyOfName[name.getAsOpaquePointer()] = k;
    return k;
}

// Forget every per-op entry of `op` and of the ops nested in it, before
// they are destroyed: the allocator may reuse their addresses for new ops,
// and a stale entry would then describe the wrong op.
void forgetErasedOp(MLIR_Context *c, mlir::Operation *op) {
    bool has_attrs = !c->opAttrCache.empty();
    bool has_names = !c->valueNames.empty();
    if (!has_attrs && !has_names) return;
    op->walk([&](mlir::Operation *o) {
        if (has_attrs) c->opAttrCache.erase(o);
        if (has_names) {
            for (mlir::Value r : o->getResults()) c->valueNames.erase(r.getAsOpaquePointer());
            for (mlir::Region &region : o->getRegions())
                for (mlir::Block &b : region)
                    for (mlir::Value a : b.getArguments()) c->valueNames.erase(a.getAsOpaquePointer());
        }
    });
}

template <class T> uintptr_t H(T *p) { return reinterpret_cast<uintptr_t>(p); }
template <class T> T *F(uintptr_t h) { return reinterpret_cast<T *>(h); }

// Under MLIR_API_CHECKS every IR getter checks that its handle belongs to the
// context it is given (the native backend resolves handles through the
// context, so passing another context's handle is a contract violation on
// both backends). A detached block or region, or a value whose op does not
// exist yet, has no context to compare with.
#ifdef MLIR_API_CHECKS
static void checkOwnOp(MLIR_Context *ctx, MLIR_OpHandle h, const char *fn) {
    if (h != MLIR_INVALID_HANDLE)
        API_CHECK(F<mlir::Operation>(h)->getContext() == ctx, fn, "the op belongs to another context");
}
static void checkOwnRegion(MLIR_Context *ctx, MLIR_RegionHandle h, const char *fn) {
    if (h == MLIR_INVALID_HANDLE) return;
    if (mlir::Operation *p = F<mlir::Region>(h)->getParentOp())
        API_CHECK(p->getContext() == ctx, fn, "the region belongs to another context");
}
static void checkOwnBlock(MLIR_Context *ctx, MLIR_BlockHandle h, const char *fn) {
    if (h == MLIR_INVALID_HANDLE) return;
    if (mlir::Operation *p = F<mlir::Block>(h)->getParentOp())
        API_CHECK(p->getContext() == ctx, fn, "the block belongs to another context");
}
static void checkOwnValue(MLIR_Context *ctx, MLIR_ValueHandle h, const char *fn) {
    if (h == MLIR_INVALID_HANDLE) return;
    mlir::Value v = valF(h);
    if (v) API_CHECK(v.getContext() == ctx, fn, "the value belongs to another context");
}
#define CHECK_OWN_OP(c, h, fn) checkOwnOp((c), (h), (fn))
#define CHECK_OWN_REGION(c, h, fn) checkOwnRegion((c), (h), (fn))
#define CHECK_OWN_BLOCK(c, h, fn) checkOwnBlock((c), (h), (fn))
#define CHECK_OWN_VALUE(c, h, fn) checkOwnValue((c), (h), (fn))
#else
#define CHECK_OWN_OP(c, h, fn) ((void)0)
#define CHECK_OWN_REGION(c, h, fn) ((void)0)
#define CHECK_OWN_BLOCK(c, h, fn) ((void)0)
#define CHECK_OWN_VALUE(c, h, fn) ((void)0)
#endif

uintptr_t typeH(mlir::Type t) { return reinterpret_cast<uintptr_t>(t.getAsOpaquePointer()); }
mlir::Type typeF(uintptr_t h) {
    return mlir::Type::getFromOpaquePointer(reinterpret_cast<const void *>(h));
}
// A type handle of `ctx` (the type getters' context check).
mlir::Type typeOf(MLIR_Context *ctx, uintptr_t h, const char *fn) {
    mlir::Type t = typeF(h);
#ifdef MLIR_API_CHECKS
    if (t) API_CHECK(t.getContext() == ctx, fn, "the type belongs to another context");
#else
    (void)ctx; (void)fn;
#endif
    return t;
}
uintptr_t locH(mlir::Location loc) { return reinterpret_cast<uintptr_t>(loc.getAsOpaquePointer()); }
mlir::Location locF(uintptr_t h) {
    return mlir::Location::getFromOpaquePointer(reinterpret_cast<const void *>(h));
}

// Records `type`'s mlir::OperationName in both directions. The C side
// names OP_TYPE_MODULE `module`; upstream calls the op `builtin.module`.
void bindOpName(MLIR_Context *c, MLIR_OpType type) {
    string s = op_type_table_name(&c->opTypeTable, type);
    llvm::StringRef name = type == OP_TYPE_MODULE ? llvm::StringRef("builtin.module")
                                                  : llvm::StringRef(s.str, s.size);
    const void *key = mlir::OperationName(name, c).getAsOpaquePointer();
    // Never rebind a name that already stands for a type: the op type of
    // every op of that name would change. (Aliases such as `builtin.module`
    // resolve to their static type before they get here.)
    auto ins = c->opTypes.try_emplace(key, type);
    API_CHECK(ins.second || ins.first->second == type, "MLIR_RegisterOpType",
              "op name already bound to another op type");
    if (!ins.second && ins.first->second != type) return;
    if ((size_t)type >= c->opNames.size()) c->opNames.resize((size_t)type + 1, nullptr);
    c->opNames[type] = key;
}

// The type of `name`, registering a dynamic type for a name `c` has not seen
// (MLIR_RegisterOpType).
MLIR_OpType registerOpType(MLIR_Context *c, llvm::StringRef name) {
    string s{const_cast<char *>(name.data()), name.size()};
    MLIR_OpType t = op_type_table_register(&c->opTypeTable, s);
    if (t != OP_TYPE_UNREGISTERED &&
        ((size_t)t >= c->opNames.size() || !c->opNames[t]))
        bindOpName(c, t);
    return t;
}

// Copy a std::string into the C-API arena so the returned `string` is
// stable for the lifetime of the printer.
string mkArenaString(MLIR_Context *ctx, llvm::StringRef src) {
    Arena *a = ctx->arena;
    auto *p = (char *)arena_alloc(a, src.size());
    if (src.size() > 0) std::memcpy(p, src.data(), src.size());
    string s; s.str = p; s.size = src.size();
    return s;
}

string mkRefString(llvm::StringRef src) {
    string s;
    s.str  = const_cast<char *>(src.data());
    s.size = src.size();
    return s;
}

} // namespace

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

namespace {

void defaultDiagnosticHandler(MLIR_DiagSeverity severity, MLIR_LocationHandle loc,
                              string message) {
    const char *sev = "error";
    if (severity == MLIR_DIAG_WARNING) sev = "warning";
    else if (severity == MLIR_DIAG_NOTE) sev = "note";
    else if (severity == MLIR_DIAG_REMARK) sev = "remark";
    if (loc != MLIR_INVALID_HANDLE) {
        mlir::Location l = locF(loc);
        if (auto fl = llvm::dyn_cast<mlir::FileLineColLoc>(l)) {
            llvm::StringRef file = fl.getFilename().getValue();
            std::fprintf(stderr, "%.*s:%u:%u: %s: %.*s\n", (int)file.size(), file.data(),
                         fl.getLine(), fl.getColumn(), sev,
                         (int)message.size, message.str);
            return;
        }
        if (!llvm::isa<mlir::UnknownLoc>(l)) {
            // Other locations (names, call sites, fused) print as MLIR's own
            // default handler prints them, e.g. `loc("name")`.
            std::string text;
            llvm::raw_string_ostream os(text);
            l.print(os);
            os.flush();
            std::fprintf(stderr, "%s: %s: %.*s\n", text.c_str(), sev,
                         (int)message.size, message.str);
            return;
        }
    }
    std::fprintf(stderr, "%s: %.*s\n", sev, (int)message.size, message.str);
}

void emitDiagnostic(MLIR_Context *c, MLIR_DiagSeverity severity,
                    MLIR_LocationHandle loc, llvm::StringRef text) {
    string message = mkRefString(text);
    if (c->diagFn) c->diagFn(c->diagUser, severity, loc, message);
    else defaultDiagnosticHandler(severity, loc, message);
}

// Forward an mlir::Diagnostic (parser, verifier, passes) to the context's
// handler; its notes follow as MLIR_DIAG_NOTE.
void forwardDiagnostic(MLIR_Context *c, mlir::Diagnostic &diag) {
    MLIR_DiagSeverity sev = MLIR_DIAG_ERROR;
    switch (diag.getSeverity()) {
        case mlir::DiagnosticSeverity::Error:   sev = MLIR_DIAG_ERROR; break;
        case mlir::DiagnosticSeverity::Warning: sev = MLIR_DIAG_WARNING; break;
        case mlir::DiagnosticSeverity::Note:    sev = MLIR_DIAG_NOTE; break;
        case mlir::DiagnosticSeverity::Remark:  sev = MLIR_DIAG_REMARK; break;
    }
    emitDiagnostic(c, sev, locH(diag.getLocation()), diag.str());
    for (mlir::Diagnostic &note : diag.getNotes())
        emitDiagnostic(c, MLIR_DIAG_NOTE, locH(note.getLocation()), note.str());
}

} // namespace

extern "C" MLIR_Context *MLIR_CreateContext(const MLIR_ContextOptions *options) {
    // no_def_use_tracking does not apply: upstream always tracks uses.
    auto *c = new MLIR_Context(contextRegistry());
    c->loadAllAvailableDialects();
    c->allowUnregisteredDialects(true);
    c->getDiagEngine().registerHandler([c](mlir::Diagnostic &diag) {
        forwardDiagnostic(c, diag);
        return mlir::success();
    });
    if (options) c->arena = options->arena;
    c->persistent = arena_create(16 * 1024);
    op_type_table_init(&c->opTypeTable, c->persistent);
    attr_key_table_init(&c->keyTable, c->persistent);
    c->keyNames.assign((size_t)MLIR_KEY_COUNT, nullptr);
    for (uint32_t k = 1; k < (uint32_t)MLIR_KEY_COUNT; k++) (void)keyNameSlow(c, (MLIR_AttrKey)k);
    c->opNames.assign((size_t)OP_TYPE_COUNT, nullptr);
    for (int t = 1; t < (int)OP_TYPE_COUNT; t++) bindOpName(c, (MLIR_OpType)t);
#ifdef MLIR_API_CHECKS
    static bool traits_checked = false;
    if (!traits_checked) {
        traits_checked = true;
        bool all_ok = true;
        for (int t = 1; t < (int)OP_TYPE_COUNT; t++) {
            mlir::OperationName opName = mlir::OperationName::getFromOpaquePointer(c->opNames[t]);
            auto reg = opName.getRegisteredInfo();
            if (!reg) continue;
            MLIR_OpTraits tr = MLIR_GetOpTypeTraits((MLIR_OpType)t);
            bool upTerm = reg->hasTrait<mlir::OpTrait::IsTerminator>();
            bool ourTerm = (tr & MLIR_T_TERMINATOR) != 0;
            if (upTerm != ourTerm) {
                fprintf(stderr, "trait mismatch for %s: terminator ours=%d upstream=%d\n",
                        opName.getStringRef().str().c_str(), (int)ourTerm, (int)upTerm);
                all_ok = false;
            }
            bool upComm = reg->hasTrait<mlir::OpTrait::IsCommutative>();
            bool ourComm = (tr & MLIR_T_COMMUTATIVE) != 0;
            if (upComm && !ourComm) {
                fprintf(stderr, "trait mismatch for %s: upstream commutative but not MLIR_T_COMMUTATIVE\n",
                        opName.getStringRef().str().c_str());
                all_ok = false;
            }
            bool upConst = reg->hasTrait<mlir::OpTrait::ConstantLike>();
            bool ourConst = (tr & MLIR_T_CONSTANT) != 0;
            if (upConst != ourConst) {
                fprintf(stderr, "trait mismatch for %s: constant ours=%d upstream=%d\n",
                        opName.getStringRef().str().c_str(), (int)ourConst, (int)upConst);
                all_ok = false;
            }
            bool upRet = reg->hasTrait<mlir::OpTrait::ReturnLike>();
            bool ourRet = (tr & MLIR_T_RETURN_LIKE) != 0;
            if (upRet != ourRet) {
                fprintf(stderr, "trait mismatch for %s: return_like ours=%d upstream=%d\n",
                        opName.getStringRef().str().c_str(), (int)ourRet, (int)upRet);
                all_ok = false;
            }
            if (!upTerm && !reg->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>() &&
                !(tr & MLIR_T_SINGLE_BLOCK)) {
                bool upSpec = reg->hasTrait<mlir::OpTrait::AlwaysSpeculatableImplTrait>();
                bool ourPure = (tr & MLIR_T_PURE) != 0;
                if (upSpec && !ourPure) {
                    fprintf(stderr, "trait mismatch for %s: upstream speculatable but not MLIR_T_PURE\n",
                            opName.getStringRef().str().c_str());
                    all_ok = false;
                }
                if (ourPure && !reg->hasInterface<mlir::MemoryEffectOpInterface>() && !upConst) {
                    fprintf(stderr, "trait mismatch for %s: MLIR_T_PURE but no MemoryEffectOpInterface upstream\n",
                            opName.getStringRef().str().c_str());
                    all_ok = false;
                }
            }
        }
        assert(all_ok);
    }
#endif
    return c;
}
// Frees all IR of the context: the parentless ops, blocks and regions own the
// rest. References between them (operands, successors, the holder block's
// arguments) are dropped first, so they can be destroyed in any order.
extern "C" void MLIR_DestroyContext(MLIR_Context *cctx) {
    if (!cctx) return;
    for (mlir::Operation *op : cctx->rootOps) op->dropAllReferences();
    for (mlir::Block *b : cctx->rootBlocks) b->dropAllReferences();
    for (mlir::Region *r : cctx->rootRegions) r->dropAllReferences();
    for (mlir::Operation *op : cctx->rootOps) op->destroy();
    for (mlir::Block *b : cctx->rootBlocks) delete b;
    for (mlir::Region *r : cctx->rootRegions) delete r;
    delete cctx;
}
extern "C" void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena) { ctx->arena = arena; }
extern "C" Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx) { return ctx->arena; }
// Upstream always tracks uses; the setting is accepted and ignored.
extern "C" void MLIR_SetContextDefUseTracking(MLIR_Context *, bool) {}
extern "C" bool MLIR_HasContextDefUseTracking(MLIR_Context *) { return true; }
extern "C" void MLIR_SetContextExpectedOps(MLIR_Context *, size_t) {}

extern "C" bool MLIR_VerifyOp(MLIR_Context *cctx, MLIR_OpHandle op) {
    API_CHECK(cctx != nullptr, "MLIR_VerifyOp", "context is NULL");
    API_CHECK(op != MLIR_INVALID_HANDLE, "MLIR_VerifyOp", "op handle is invalid");
    return mlir::succeeded(mlir::verify(reinterpret_cast<mlir::Operation *>(op)));
}

extern "C" void MLIR_SetDiagnosticHandler(MLIR_Context *cctx, MLIR_DiagnosticHandler fn,
                                          void *user) {
    cctx->diagFn = fn;
    cctx->diagUser = fn ? user : nullptr;
}
extern "C" void MLIR_EmitDiagnostic(MLIR_Context *cctx, MLIR_DiagSeverity severity,
                                    MLIR_LocationHandle loc, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string buf(n > 0 ? (size_t)n : 0, '\0');
    if (n > 0) std::vsnprintf(&buf[0], (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    emitDiagnostic(cctx, severity, loc, buf);
}

// -----------------------------------------------------------------------------
// Region / Block
// -----------------------------------------------------------------------------

extern "C" MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *cctx) {
    auto *r = new mlir::Region();
    cctx->rootRegions.insert(r);
    return H(r);
}
extern "C" MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *cctx) {
    auto *b = new mlir::Block();
    cctx->rootBlocks.insert(b);
    return H(b);
}
extern "C" void MLIR_AppendRegionBlock(MLIR_Context *cctx, MLIR_RegionHandle r,
                                        MLIR_BlockHandle b) {
    auto *blk = F<mlir::Block>(b);
    API_CHECK(!blk->getParent(), "MLIR_AppendRegionBlock", "block is already in a region");
    cctx->rootBlocks.erase(blk);
    F<mlir::Region>(r)->push_back(blk);
}
extern "C" void MLIR_AppendBlockOp(MLIR_Context *cctx, MLIR_BlockHandle b,
                                    MLIR_OpHandle op) {
    auto *target = F<mlir::Block>(b);
    auto *o = F<mlir::Operation>(op);
    API_CHECK(!o->getBlock(), "MLIR_AppendBlockOp", "op is already in a block (use MLIR_MoveOpToBlockEnd)");
    cctx->rootOps.erase(o);
    target->push_back(o);
}
extern "C" void MLIR_InsertBlockOpAtIndex(MLIR_Context *cctx,
                                          MLIR_BlockHandle b,
                                          MLIR_OpHandle op, size_t idx) {
    auto *block = F<mlir::Block>(b);
    auto *o = F<mlir::Operation>(op);
    API_CHECK(!o->getBlock(), "MLIR_InsertBlockOpAtIndex", "op is already in a block (detach it first)");
    cctx->rootOps.erase(o);
    auto &ops = block->getOperations();
    size_t n = 0;
    for (auto it = ops.begin(); it != ops.end(); ++it, ++n) {
        if (n == idx) { ops.insert(it, o); return; }
    }
    block->push_back(o);
}
extern "C" void MLIR_AppendBlockArg(MLIR_Context *cctx, MLIR_BlockHandle bh,
                                     MLIR_ValueHandle vh) {
    auto *block = F<mlir::Block>(bh);
    mlir::Value oldVal = valF(vh);
    auto realArg = block->addArgument(oldVal.getType(), oldVal.getLoc());
    auto it = cctx->valueNames.find(oldVal.getAsOpaquePointer());
    if (it != cctx->valueNames.end()) {
        std::string nm = std::move(it->second);
        cctx->valueNames.erase(it);
        cctx->valueNames[realArg.getAsOpaquePointer()] = std::move(nm);
    }
    oldVal.replaceAllUsesWith(realArg);
}

// -----------------------------------------------------------------------------
// Op specification & Builder API
// -----------------------------------------------------------------------------

static mlir::Operation *createOpFromSpecImpl(MLIR_Context *cctx, const MLIR_OpSpec *spec,
                                             MLIR_LocationHandle locOverride) {
    MLIR_Context &ctx = *cctx;
    MLIR_OpType type = spec->type;
    API_CHECK(type != OP_TYPE_UNREGISTERED && (size_t)type < ctx.opNames.size() &&
                  ctx.opNames[type],
              "MLIR_CreateOpFromSpec", "not an op type of this context");

    MLIR_LocationHandle locHnd = (spec->loc != MLIR_INVALID_HANDLE) ? spec->loc : locOverride;
    mlir::Location loc = (locHnd == MLIR_INVALID_HANDLE) ? ctx.unknownLoc : locF(locHnd);
    mlir::OperationState state(loc, mlir::OperationName::getFromOpaquePointer(ctx.opNames[type]));

    for (uint32_t i = 0; i < spec->n_operands; i++) {
        state.addOperands(valF(spec->operands[i]));
    }
    for (uint32_t i = 0; i < spec->n_results; i++) {
        state.addTypes(typeF(spec->result_types[i]));
    }
    if (spec->n_attrs > 0) {
        bool is_switch = type == OP_TYPE_CF_SWITCH || type == OP_TYPE_LLVM_SWITCH;
        for (uint32_t i = 0; i < spec->n_attrs; i++) {
            mlir::StringAttr name = keyName(cctx, spec->attrs[i].key);
            mlir::Attribute val = attrF(spec->attrs[i].value);
            if (is_switch && spec->attrs[i].key == MLIR_KEY_CASE_VALUES) {
                if (auto arr = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(val)) {
                    auto i32Ty = mlir::IntegerType::get(&ctx, 32);
                    auto shaped = mlir::RankedTensorType::get(
                        {(int64_t)arr.size()}, i32Ty);
                    val = mlir::DenseIntElementsAttr::get(
                        shaped, llvm::ArrayRef<int32_t>(arr.asArrayRef().data(),
                                                        arr.size()));
                }
            }
            state.addAttribute(name, val);
        }
    }
    if (spec->n_regions > 0) {
        if (spec->regions) {
            for (uint32_t i = 0; i < spec->n_regions; i++) {
                cctx->rootRegions.erase(F<mlir::Region>(spec->regions[i]));
                state.addRegion(std::unique_ptr<mlir::Region>(F<mlir::Region>(spec->regions[i])));
            }
        } else {
            for (uint32_t i = 0; i < spec->n_regions; i++) {
                state.addRegion();
            }
        }
    }
    if (spec->n_successors > 0) {
        uint32_t n_succ_ops = 0;
        if (spec->successor_operand_counts) {
            for (uint32_t s = 0; s < spec->n_successors; s++)
                n_succ_ops += spec->successor_operand_counts[s];
        }
        API_CHECK(n_succ_ops <= spec->n_operands, "MLIR_CreateOpFromSpec",
                  "successor operand counts exceed n_operands");
        uint32_t n_regular = spec->n_operands - n_succ_ops;
        for (uint32_t i = 0; i < spec->n_successors; i++) {
            state.addSuccessors(F<mlir::Block>(spec->successors[i]));
        }
        if (type == OP_TYPE_CF_COND_BR || type == OP_TYPE_LLVM_COND_BR) {
            llvm::SmallVector<int32_t, 4> seg;
            seg.push_back((int32_t)n_regular);
            for (uint32_t s = 0; s < spec->n_successors; s++) {
                seg.push_back((int32_t)(spec->successor_operand_counts ? spec->successor_operand_counts[s] : 0));
            }
            state.addAttribute("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(&ctx, seg));
        } else if (type == OP_TYPE_CF_SWITCH || type == OP_TYPE_LLVM_SWITCH) {
            uint32_t n_default = spec->successor_operand_counts ? spec->successor_operand_counts[0] : 0;
            int32_t n_case_total = 0;
            llvm::SmallVector<int32_t, 4> case_seg;
            for (uint32_t s = 1; s < spec->n_successors; s++) {
                int32_t c = (int32_t)(spec->successor_operand_counts ? spec->successor_operand_counts[s] : 0);
                case_seg.push_back(c);
                n_case_total += c;
            }
            llvm::SmallVector<int32_t, 3> seg;
            seg.push_back((int32_t)n_regular);
            seg.push_back((int32_t)n_default);
            seg.push_back(n_case_total);
            state.addAttribute("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(&ctx, seg));
            state.addAttribute("case_operand_segments",
                mlir::DenseI32ArrayAttr::get(&ctx, case_seg));
        }
    }

    return mlir::Operation::create(state);
}

extern "C" MLIR_OpHandle MLIR_CreateOpFromSpec(MLIR_Context *cctx, const MLIR_OpSpec *spec) {
    API_CHECK(cctx != nullptr, "MLIR_CreateOpFromSpec", "context is NULL");
    API_CHECK(spec != nullptr, "MLIR_CreateOpFromSpec", "spec is NULL");
    mlir::Operation *op = createOpFromSpecImpl(cctx, spec, MLIR_INVALID_HANDLE);
    cctx->rootOps.insert(op);
    return H(op);
}

extern "C" void MLIR_InsertOpBefore(MLIR_Context *cctx, MLIR_BlockHandle bh,
                                    MLIR_OpHandle oh, MLIR_OpHandle before_h) {
    CHECK_OWN_BLOCK(cctx, bh, "MLIR_InsertOpBefore");
    CHECK_OWN_OP(cctx, oh, "MLIR_InsertOpBefore");
    auto *block = F<mlir::Block>(bh);
    auto *op = F<mlir::Operation>(oh);
    if (op->getBlock()) {
        op->remove();
    } else {
        cctx->rootOps.erase(op);
    }
    if (before_h == MLIR_INVALID_HANDLE) {
        block->push_back(op);
    } else {
        CHECK_OWN_OP(cctx, before_h, "MLIR_InsertOpBefore");
        auto *bef = F<mlir::Operation>(before_h);
        API_CHECK(bef->getBlock() == block, "MLIR_InsertOpBefore",
                  "before op is not in target block");
        block->getOperations().insert(mlir::Block::iterator(bef), op);
    }
}

extern "C" MLIR_Builder MLIR_BuilderAfter(MLIR_Context *cctx, MLIR_OpHandle oh) {
    CHECK_OWN_OP(cctx, oh, "MLIR_BuilderAfter");
    auto *op = F<mlir::Operation>(oh);
    mlir::Block *block = op->getBlock();
    API_CHECK(block != nullptr, "MLIR_BuilderAfter", "op has no parent block");
    auto it = std::next(mlir::Block::iterator(op));
    MLIR_Builder res;
    res.ctx = cctx;
    res.block = H(block);
    res.before = (it == block->end()) ? MLIR_INVALID_HANDLE : H(&*it);
    res.loc = MLIR_INVALID_HANDLE;
    return res;
}

extern "C" MLIR_OpHandle MLIR_BuildOp(MLIR_Builder *b, const MLIR_OpSpec *spec) {
    API_CHECK(b != nullptr && b->ctx != nullptr && spec != nullptr, "MLIR_BuildOp", "NULL argument");
    mlir::Operation *op = createOpFromSpecImpl(b->ctx, spec, b->loc);
    if (b->block != MLIR_INVALID_HANDLE) {
        CHECK_OWN_BLOCK(b->ctx, b->block, "MLIR_BuildOp");
        auto *block = F<mlir::Block>(b->block);
        if (b->before == MLIR_INVALID_HANDLE) {
            block->push_back(op);
        } else {
            CHECK_OWN_OP(b->ctx, b->before, "MLIR_BuildOp");
            auto *bef = F<mlir::Operation>(b->before);
            API_CHECK(bef->getBlock() == block, "MLIR_BuildOp",
                      "before op is not in target block");
            block->getOperations().insert(mlir::Block::iterator(bef), op);
        }
    } else {
        b->ctx->rootOps.insert(op);
    }
    return H(op);
}

extern "C" MLIR_ValueHandle MLIR_BuildBinary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y) {
    MLIR_ValueHandle ops[2] = { x, y };
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = ops;
    s.n_operands = 2;
    s.result_types = &rt;
    s.n_results = 1;
    return valH(F<mlir::Operation>(MLIR_BuildOp(b, &s))->getResult(0));
}

extern "C" MLIR_ValueHandle MLIR_BuildUnary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x) {
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = &x;
    s.n_operands = 1;
    s.result_types = &rt;
    s.n_results = 1;
    return valH(F<mlir::Operation>(MLIR_BuildOp(b, &s))->getResult(0));
}

extern "C" void MLIR_GetOpSuccessorOperandRange(MLIR_Context *ctx, MLIR_OpHandle h,
                                                size_t succ, size_t *first, size_t *count) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpSuccessorOperandRange");
    auto *op = F<mlir::Operation>(h);
    API_CHECK(succ < op->getNumSuccessors(), "MLIR_GetOpSuccessorOperandRange", "index out of range");
    if (auto br = mlir::dyn_cast<mlir::BranchOpInterface>(op)) {
        size_t total_succ = 0;
        size_t before_succ = 0;
        size_t cnt = 0;
        for (size_t s = 0; s < op->getNumSuccessors(); s++) {
            size_t n = br.getSuccessorOperands(s).size();
            if (s < succ) before_succ += n;
            if (s == succ) cnt = n;
            total_succ += n;
        }
        size_t base = (op->getNumOperands() - total_succ) + before_succ;
        if (first) *first = base;
        if (count) *count = cnt;
        return;
    }
    if (first) *first = op->getNumOperands();
    if (count) *count = 0;
}

extern "C" void MLIR_SetOpTrailingComment(MLIR_Context *ctx, MLIR_OpHandle h, string) {
    CHECK_OWN_OP(ctx, h, "MLIR_SetOpTrailingComment");
}

extern "C" void MLIR_SetValueRegisterName(MLIR_Context *ctx, MLIR_ValueHandle h, string name) {
    CHECK_OWN_VALUE(ctx, h, "MLIR_SetValueRegisterName");
    if (h == MLIR_INVALID_HANDLE) return;
    mlir::Value v = valF(h);
    if (name.size == 0) {
        ctx->valueNames.erase(v.getAsOpaquePointer());
    } else {
        ctx->valueNames[v.getAsOpaquePointer()].assign(name.str, name.size);
    }
}

extern "C" void MLIR_SetOpAttr(MLIR_Context *cctx, MLIR_OpHandle oh, MLIR_AttrKey key,
                               MLIR_AttributeHandle ah) {
    CHECK_OWN_OP(cctx, oh, "MLIR_SetOpAttr");
    auto *op = F<mlir::Operation>(oh);
    mlir::NamedAttribute na(keyName(cctx, key), attrF(ah));
    op->setAttr(na.getName(), na.getValue());
    if (!cctx->opAttrCache.empty()) {
        auto it = cctx->opAttrCache.find(op);
        if (it != cctx->opAttrCache.end()) {
            for (auto &existing : it->second) {
                if (existing.getName() == na.getName()) {
                    existing = na;
                    return;
                }
            }
            it->second.push_back(na);
        }
    }
}

// -----------------------------------------------------------------------------
// Op accessors — straight passthrough to upstream
// -----------------------------------------------------------------------------

static MLIR_UP_NOINLINE MLIR_OpType getOpTypeSlow(MLIR_Context *c, mlir::OperationName name,
                                                  MLIR_Context::OpTypeCacheEntry &slot) {
    const void *p = name.getAsOpaquePointer();
    auto it = c->opTypes.find(p);
    MLIR_OpType t = (it != c->opTypes.end()) ? it->second : registerOpType(c, name.getStringRef());
    slot.ptr = p;
    slot.type = t;
    return t;
}

extern "C" MLIR_OpType MLIR_GetOpType(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpType");
    auto *op = F<mlir::Operation>(h);
    auto name = op->getName();
    const void *p = name.getAsOpaquePointer();
    auto &slot = ctx->opTypeCache[(reinterpret_cast<uintptr_t>(p) >> 4) & 63];
    if (MLIR_UP_LIKELY(slot.ptr == p)) return slot.type;
    return getOpTypeSlow(ctx, name, slot);
}
extern "C" MLIR_LocationHandle MLIR_GetOpLocation(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpLocation");
    return locH(F<mlir::Operation>(h)->getLoc());
}
extern "C" string MLIR_GetOpTrailingComment(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpTrailingComment");
    string s; s.str = nullptr; s.size = 0; return s;
}

static MLIR_UP_NOINLINE size_t getOpNumOperandsSlow(mlir::Operation *op) {
    if (auto br = mlir::dyn_cast<mlir::BranchOpInterface>(op)) {
        // Exclude successor operands; expose only "regular" (non-forwarded)
        // operands so the API matches the native backend convention.
        size_t total = op->getNumOperands();
        for (unsigned s = 0; s < op->getNumSuccessors(); s++) {
            total -= br.getSuccessorOperands(s).size();
        }
        return total;
    }
    return op->getNumOperands();
}

extern "C" size_t MLIR_GetOpNumOperands(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpNumOperands");
    auto *op = F<mlir::Operation>(h);
    if (MLIR_UP_LIKELY(op->getNumSuccessors() == 0)) return op->getNumOperands();
    return getOpNumOperandsSlow(op);
}
extern "C" MLIR_ValueHandle MLIR_GetOpOperand(MLIR_Context *ctx, MLIR_OpHandle h, size_t i) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpOperand");
    API_CHECK(h != MLIR_INVALID_HANDLE, "MLIR_GetOpOperand", "op handle is NULL");
    API_CHECK(i < MLIR_GetOpNumOperands(ctx, h), "MLIR_GetOpOperand", "index out of range");
    return valH(F<mlir::Operation>(h)->getOperand(i));
}
extern "C" size_t MLIR_GetOpNumResults(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpNumResults");
    return F<mlir::Operation>(h)->getNumResults();
}
extern "C" MLIR_ValueHandle MLIR_GetOpResult(MLIR_Context *ctx, MLIR_OpHandle h, size_t i) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpResult");
    API_CHECK(h != MLIR_INVALID_HANDLE, "MLIR_GetOpResult", "op handle is NULL");
    API_CHECK(i < F<mlir::Operation>(h)->getNumResults(), "MLIR_GetOpResult", "index out of range");
    return valH(F<mlir::Operation>(h)->getResult(i));
}
extern "C" size_t MLIR_GetOpNumResultTypes(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpNumResultTypes");
    return F<mlir::Operation>(h)->getNumResults();
}
extern "C" MLIR_TypeHandle MLIR_GetOpResultType(MLIR_Context *ctx, MLIR_OpHandle h, size_t i) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpResultType");
    return typeH(F<mlir::Operation>(h)->getResult(i).getType());
}
// op->getAttrs() may return an ArrayRef whose elements are not stable across
// calls (especially on newer MLIR where some "inherent" attributes are kept
// in inline property storage and the dictionary view is rebuilt on demand).
// Cache a stable copy on first access so MLIR_AttributeHandle pointers stay
// valid for the rest of the lowering pass.
//
// The cache is also seeded by MLIR_CreateOp with the exact user-passed
// attribute list, which is what we want to expose to consumers of the
// classic-form text printer: they should see only attrs the user wrote, not
// every default-valued inherent property MLIR keeps in typed storage (e.g.
// arith.addi's `overflowFlags = #arith.overflow<none>`).
//
// The MLIR_EraseOp path invalidates this cache, so the seeded snapshot can
// never be shadowed by a freshly-allocated op reusing a stale Operation*
// slab address.
static bool isDefaultInherentAttr(const mlir::NamedAttribute &na) {
    llvm::StringRef k = na.getName().getValue();
    if (k == "operandSegmentSizes" || k == "case_operand_segments") return true;
    mlir::Attribute v = na.getValue();
    if (k == "overflowFlags") {
        if (auto a = mlir::dyn_cast<mlir::arith::IntegerOverflowFlagsAttr>(v))
            return a.getValue() == mlir::arith::IntegerOverflowFlags::none;
        if (auto a = mlir::dyn_cast<mlir::LLVM::IntegerOverflowFlagsAttr>(v))
            return a.getValue() == mlir::LLVM::IntegerOverflowFlags::none;
    }
    if (k == "fastmath") {
        if (auto a = mlir::dyn_cast<mlir::arith::FastMathFlagsAttr>(v))
            return a.getValue() == mlir::arith::FastMathFlags::none;
    }
    if (k == "fastmathFlags") {
        if (auto a = mlir::dyn_cast<mlir::LLVM::FastmathFlagsAttr>(v))
            return a.getValue() == mlir::LLVM::FastmathFlags::none;
    }
    return false;
}

static std::vector<mlir::NamedAttribute> &
ensureCachedAttrs(mlir::Operation *op) {
    auto &m = ctxOf(op)->opAttrCache;
    auto it = m.find(op);
    if (it != m.end()) return it->second;
    auto &v = m[op];
    for (const mlir::NamedAttribute &na : op->getAttrs()) {
        if (!isDefaultInherentAttr(na)) v.push_back(na);
    }
    return v;
}
extern "C" size_t MLIR_GetOpNumAttributes(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpNumAttributes");
    auto *op = F<mlir::Operation>(h);
    return ensureCachedAttrs(op).size();
}
extern "C" MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_Context *ctx, MLIR_OpHandle h, size_t i) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpAttribute");
    API_CHECK(i < MLIR_GetOpNumAttributes(ctx, h), "MLIR_GetOpAttribute", "index out of range");
    auto *op = F<mlir::Operation>(h);
    return attrH(ensureCachedAttrs(op)[i].getValue());
}
extern "C" MLIR_AttrKey MLIR_GetOpAttributeKey(MLIR_Context *ctx, MLIR_OpHandle h, size_t i) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpAttributeKey");
    API_CHECK(i < MLIR_GetOpNumAttributes(ctx, h), "MLIR_GetOpAttributeKey", "index out of range");
    return keyOf(ctx, ensureCachedAttrs(F<mlir::Operation>(h))[i].getName());
}
// A switch's i32 `case_values` (a DenseIntElementsAttr) as the
// DenseI32ArrayAttr the API creates, converted once per distinct value.
static mlir::Attribute caseValuesAsI32(MLIR_Context &mctx, mlir::DenseIntElementsAttr cv) {
    auto it = mctx.caseValuesI32.find(cv.getAsOpaquePointer());
    if (it != mctx.caseValuesI32.end()) return mlir::Attribute::getFromOpaquePointer(it->second);
    llvm::SmallVector<int32_t, 8> vals;
    vals.reserve(cv.getNumElements());
    for (auto apv : cv.getValues<llvm::APInt>())
        vals.push_back((int32_t)apv.getSExtValue());
    mlir::Attribute a = mlir::DenseI32ArrayAttr::get(&mctx, vals);
    mctx.caseValuesI32[cv.getAsOpaquePointer()] = a.getAsOpaquePointer();
    return a;
}

static MLIR_UP_NOINLINE MLIR_AttributeHandle getOpAttrSpecial(MLIR_Context *ctx,
                                                              mlir::Operation *op,
                                                              MLIR_AttrKey key) {
    if (key == MLIR_KEY_CALLEE || key == MLIR_KEY_VAR_CALLEE_TYPE) {
        if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
            if (key == MLIR_KEY_CALLEE) {
                if (auto c = call.getCalleeAttr()) return attrH(c);
                return MLIR_INVALID_HANDLE;
            }
            if (auto t = call.getVarCalleeTypeAttr()) return attrH(t);
            return MLIR_INVALID_HANDLE;
        }
    } else if (key == MLIR_KEY_CASE_VALUES) {
        if (auto sw = mlir::dyn_cast<mlir::cf::SwitchOp>(op)) {
            if (auto cv = sw.getCaseValuesAttr()) {
                if (cv.getElementType().isInteger(32)) return attrH(caseValuesAsI32(*ctx, cv));
                return attrH(cv);
            }
        }
        if (auto sw = mlir::dyn_cast<mlir::LLVM::SwitchOp>(op)) {
            if (auto cv = sw.getCaseValuesAttr()) {
                if (cv.getElementType().isInteger(32)) return attrH(caseValuesAsI32(*ctx, cv));
                return attrH(cv);
            }
        }
    }
    mlir::StringAttr nameAttr = keyName(ctx, key);
    if (auto a = op->getAttr(nameAttr)) return attrH(a);
    return MLIR_INVALID_HANDLE;
}

// Inherent attributes (properties) included: Operation::getAttr reads them.
extern "C" MLIR_AttributeHandle MLIR_GetOpAttr(MLIR_Context *ctx, MLIR_OpHandle h,
                                               MLIR_AttrKey key) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpAttr");
    auto *op = F<mlir::Operation>(h);
    if (MLIR_UP_UNLIKELY(key == MLIR_KEY_CALLEE || key == MLIR_KEY_VAR_CALLEE_TYPE ||
                         key == MLIR_KEY_CASE_VALUES))
        return getOpAttrSpecial(ctx, op, key);
    mlir::StringAttr nameAttr = keyName(ctx, key);
    if (auto a = op->getAttr(nameAttr)) return attrH(a);
    return MLIR_INVALID_HANDLE;
}
extern "C" size_t MLIR_GetOpNumRegions(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpNumRegions");
    return F<mlir::Operation>(h)->getNumRegions();
}
extern "C" MLIR_RegionHandle MLIR_GetOpRegion(MLIR_Context *ctx, MLIR_OpHandle h, size_t i) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpRegion");
    API_CHECK(i < F<mlir::Operation>(h)->getNumRegions(), "MLIR_GetOpRegion", "index out of range");
    return H(&F<mlir::Operation>(h)->getRegion(i));
}

extern "C" size_t MLIR_GetOpNumSuccessors(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpNumSuccessors");
    return F<mlir::Operation>(h)->getNumSuccessors();
}
extern "C" MLIR_BlockHandle MLIR_GetOpSuccessor(MLIR_Context *ctx, MLIR_OpHandle h, size_t i) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpSuccessor");
    API_CHECK(i < F<mlir::Operation>(h)->getNumSuccessors(), "MLIR_GetOpSuccessor", "index out of range");
    return H(F<mlir::Operation>(h)->getSuccessor(i));
}
extern "C" size_t MLIR_GetOpNumSuccessorOperands(MLIR_Context *ctx, MLIR_OpHandle h, size_t s) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpNumSuccessorOperands");
    auto *op = F<mlir::Operation>(h);
    if (auto br = mlir::dyn_cast<mlir::BranchOpInterface>(op)) {
        return br.getSuccessorOperands(s).size();
    }
    return 0;
}
extern "C" MLIR_ValueHandle MLIR_GetOpSuccessorOperand(MLIR_Context *ctx, MLIR_OpHandle h, size_t s, size_t i) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpSuccessorOperand");
    API_CHECK(i < MLIR_GetOpNumSuccessorOperands(ctx, h, s), "MLIR_GetOpSuccessorOperand", "index out of range");
    auto *op = F<mlir::Operation>(h);
    if (auto br = mlir::dyn_cast<mlir::BranchOpInterface>(op)) {
        auto ops = br.getSuccessorOperands(s);
        return valH(ops[i]);
    }
    return MLIR_INVALID_HANDLE;
}

// -----------------------------------------------------------------------------
// O(1) Traversal iterators (#352)
// -----------------------------------------------------------------------------

extern "C" MLIR_OpHandle MLIR_GetBlockFirstOp(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    CHECK_OWN_BLOCK(ctx, bh, "MLIR_GetBlockFirstOp");
    auto *b = F<mlir::Block>(bh);
    return b->empty() ? MLIR_INVALID_HANDLE : H(&b->front());
}

extern "C" MLIR_OpHandle MLIR_GetBlockLastOp(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    CHECK_OWN_BLOCK(ctx, bh, "MLIR_GetBlockLastOp");
    auto *b = F<mlir::Block>(bh);
    return b->empty() ? MLIR_INVALID_HANDLE : H(&b->back());
}

extern "C" MLIR_OpHandle MLIR_GetOpNext(MLIR_Context *ctx, MLIR_OpHandle oh) {
    CHECK_OWN_OP(ctx, oh, "MLIR_GetOpNext");
    auto *op = F<mlir::Operation>(oh);
    mlir::Block *b = op->getBlock();
    if (MLIR_UP_UNLIKELY(!b)) return MLIR_INVALID_HANDLE;
    auto it = std::next(mlir::Block::iterator(op));
    return (it == b->end()) ? MLIR_INVALID_HANDLE : H(&*it);
}

extern "C" MLIR_OpHandle MLIR_GetOpPrev(MLIR_Context *ctx, MLIR_OpHandle oh) {
    CHECK_OWN_OP(ctx, oh, "MLIR_GetOpPrev");
    auto *op = F<mlir::Operation>(oh);
    mlir::Block *b = op->getBlock();
    if (MLIR_UP_UNLIKELY(!b)) return MLIR_INVALID_HANDLE;
    auto it = mlir::Block::iterator(op);
    return (it == b->begin()) ? MLIR_INVALID_HANDLE : H(&*std::prev(it));
}

extern "C" MLIR_BlockOpIter MLIR_BlockOpsBegin(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    CHECK_OWN_BLOCK(ctx, bh, "MLIR_BlockOpsBegin");
    auto *b = F<mlir::Block>(bh);
    MLIR_BlockOpIter it;
    it.cur = reinterpret_cast<uintptr_t>(b->begin().getNodePtr());
    it.end = reinterpret_cast<uintptr_t>(b->end().getNodePtr());
    return it;
}

extern "C" MLIR_OpHandle MLIR_BlockOpIterNext(MLIR_Context *ctx, MLIR_BlockOpIter *it) {
    (void)ctx;
    if (it->cur == it->end) return MLIR_INVALID_HANDLE;
    using NodePtr = decltype(std::declval<mlir::Block::iterator>().getNodePtr());
    mlir::Block::iterator bit(*reinterpret_cast<NodePtr>(it->cur));
    mlir::Operation *op = &*bit;
    ++bit;
    it->cur = reinterpret_cast<uintptr_t>(bit.getNodePtr());
    return H(op);
}

extern "C" MLIR_OpHandle MLIR_GetParentOp(MLIR_Context *ctx, MLIR_OpHandle oh) {
    CHECK_OWN_OP(ctx, oh, "MLIR_GetParentOp");
    auto *p = F<mlir::Operation>(oh)->getParentOp();
    return p ? H(p) : MLIR_INVALID_HANDLE;
}

extern "C" MLIR_BlockHandle MLIR_GetRegionFirstBlock(MLIR_Context *ctx, MLIR_RegionHandle rh) {
    CHECK_OWN_REGION(ctx, rh, "MLIR_GetRegionFirstBlock");
    auto *r = F<mlir::Region>(rh);
    return r->empty() ? MLIR_INVALID_HANDLE : H(&r->front());
}

extern "C" MLIR_BlockHandle MLIR_GetRegionLastBlock(MLIR_Context *ctx, MLIR_RegionHandle rh) {
    CHECK_OWN_REGION(ctx, rh, "MLIR_GetRegionLastBlock");
    auto *r = F<mlir::Region>(rh);
    return r->empty() ? MLIR_INVALID_HANDLE : H(&r->back());
}

extern "C" MLIR_BlockHandle MLIR_GetBlockNext(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    CHECK_OWN_BLOCK(ctx, bh, "MLIR_GetBlockNext");
    auto *b = F<mlir::Block>(bh);
    if (!b->getParent()) return MLIR_INVALID_HANDLE;
    auto *next = b->getNextNode();
    return next ? H(next) : MLIR_INVALID_HANDLE;
}

extern "C" MLIR_BlockHandle MLIR_GetBlockPrev(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    CHECK_OWN_BLOCK(ctx, bh, "MLIR_GetBlockPrev");
    auto *b = F<mlir::Block>(bh);
    if (!b->getParent()) return MLIR_INVALID_HANDLE;
    auto *prev = b->getPrevNode();
    return prev ? H(prev) : MLIR_INVALID_HANDLE;
}

extern "C" MLIR_UseHandle MLIR_GetValueFirstUse(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    CHECK_OWN_VALUE(ctx, vh, "MLIR_GetValueFirstUse");
    mlir::Value v = valF(vh);
    if (!v || v.use_empty()) return MLIR_INVALID_HANDLE;
    return H(&*v.use_begin());
}

extern "C" MLIR_UseHandle MLIR_GetUseNext(MLIR_Context *, MLIR_UseHandle use) {
    if (!use) return MLIR_INVALID_HANDLE;
    auto *next = F<mlir::OpOperand>(use)->getNextOperandUsingThisValue();
    return next ? H(next) : MLIR_INVALID_HANDLE;
}

extern "C" MLIR_OpHandle MLIR_GetUseOwner(MLIR_Context *, MLIR_UseHandle use) {
    API_CHECK(use != MLIR_INVALID_HANDLE, "MLIR_GetUseOwner", "invalid use handle");
    return H(F<mlir::OpOperand>(use)->getOwner());
}

extern "C" size_t MLIR_GetUseOperandIndex(MLIR_Context *, MLIR_UseHandle use) {
    API_CHECK(use != MLIR_INVALID_HANDLE, "MLIR_GetUseOperandIndex", "invalid use handle");
    return (size_t)F<mlir::OpOperand>(use)->getOperandNumber();
}

extern "C" bool MLIR_ValueHasOneUse(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    CHECK_OWN_VALUE(ctx, vh, "MLIR_ValueHasOneUse");
    mlir::Value v = valF(vh);
    return v && v.hasOneUse();
}

extern "C" bool MLIR_IsValueUnused(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    CHECK_OWN_VALUE(ctx, vh, "MLIR_IsValueUnused");
    mlir::Value v = valF(vh);
    return !v || v.use_empty();
}

extern "C" MLIR_PredHandle MLIR_GetBlockFirstPred(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    CHECK_OWN_BLOCK(ctx, bh, "MLIR_GetBlockFirstPred");
    auto *b = F<mlir::Block>(bh);
    if (b->use_empty()) return MLIR_INVALID_HANDLE;
    return H(&*b->use_begin());
}

extern "C" MLIR_PredHandle MLIR_GetPredNext(MLIR_Context *, MLIR_PredHandle pred) {
    if (!pred) return MLIR_INVALID_HANDLE;
    auto *next = F<mlir::BlockOperand>(pred)->getNextOperandUsingThisValue();
    return next ? H(next) : MLIR_INVALID_HANDLE;
}

extern "C" MLIR_OpHandle MLIR_GetPredTerminator(MLIR_Context *, MLIR_PredHandle pred) {
    API_CHECK(pred != MLIR_INVALID_HANDLE, "MLIR_GetPredTerminator", "invalid pred handle");
    return H(F<mlir::BlockOperand>(pred)->getOwner());
}

extern "C" MLIR_BlockHandle MLIR_GetPredBlock(MLIR_Context *, MLIR_PredHandle pred) {
    API_CHECK(pred != MLIR_INVALID_HANDLE, "MLIR_GetPredBlock", "invalid pred handle");
    auto *b = F<mlir::BlockOperand>(pred)->getOwner()->getBlock();
    return b ? H(b) : MLIR_INVALID_HANDLE;
}

extern "C" size_t MLIR_GetPredSuccessorIndex(MLIR_Context *, MLIR_PredHandle pred) {
    API_CHECK(pred != MLIR_INVALID_HANDLE, "MLIR_GetPredSuccessorIndex", "invalid pred handle");
    return (size_t)F<mlir::BlockOperand>(pred)->getOperandNumber();
}

// -----------------------------------------------------------------------------
// Region / Block accessors
// -----------------------------------------------------------------------------

extern "C" size_t MLIR_GetRegionNumBlocks(MLIR_Context *ctx, MLIR_RegionHandle h) {
    CHECK_OWN_REGION(ctx, h, "MLIR_GetRegionNumBlocks");
    auto *r = F<mlir::Region>(h);
    return std::distance(r->begin(), r->end());
}
extern "C" MLIR_BlockHandle MLIR_GetRegionBlock(MLIR_Context *ctx, MLIR_RegionHandle h, size_t i) {
    CHECK_OWN_REGION(ctx, h, "MLIR_GetRegionBlock");
    auto *r = F<mlir::Region>(h);
    // The range check rides along the O(i) walk instead of counting the
    // blocks first, which would make an index loop O(n^2) in checked builds.
    auto it = r->begin();
    for (size_t k = 0; k < i; k++) {
        API_CHECK(it != r->end(), "MLIR_GetRegionBlock", "index out of range");
        ++it;
    }
    API_CHECK(it != r->end(), "MLIR_GetRegionBlock", "index out of range");
    return H(&*it);
}

extern "C" size_t MLIR_GetBlockNumOps(MLIR_Context *ctx, MLIR_BlockHandle h) {
    CHECK_OWN_BLOCK(ctx, h, "MLIR_GetBlockNumOps");
    auto *b = F<mlir::Block>(h);
    return std::distance(b->begin(), b->end());
}
extern "C" MLIR_OpHandle MLIR_GetBlockOp(MLIR_Context *ctx, MLIR_BlockHandle h, size_t i) {
    CHECK_OWN_BLOCK(ctx, h, "MLIR_GetBlockOp");
    auto *b = F<mlir::Block>(h);
    API_CHECK(!b->empty(), "MLIR_GetBlockOp", "index out of range");
    auto it = b->begin();
    std::advance(it, i);
    return H(&*it);
}
extern "C" size_t MLIR_GetBlockNumArgs(MLIR_Context *ctx, MLIR_BlockHandle h) {
    CHECK_OWN_BLOCK(ctx, h, "MLIR_GetBlockNumArgs");
    return F<mlir::Block>(h)->getNumArguments();
}
extern "C" MLIR_ValueHandle MLIR_GetBlockArg(MLIR_Context *ctx, MLIR_BlockHandle h, size_t i) {
    CHECK_OWN_BLOCK(ctx, h, "MLIR_GetBlockArg");
    API_CHECK(i < F<mlir::Block>(h)->getNumArguments(), "MLIR_GetBlockArg", "index out of range");
    return valH(F<mlir::Block>(h)->getArgument(i));
}

extern "C" size_t MLIR_GetBlockIndex(MLIR_Context *ctx, MLIR_BlockHandle h) {
    CHECK_OWN_BLOCK(ctx, h, "MLIR_GetBlockIndex");
    auto *b = F<mlir::Block>(h);
    auto *r = b->getParent();
    if (!r) return (size_t)-1;
    size_t idx = 0;
    for (auto &it : *r) {
        if (&it == b) return idx;
        idx++;
    }
    return (size_t)-1;
}

// -----------------------------------------------------------------------------
// Value
// -----------------------------------------------------------------------------

extern "C" MLIR_ValueHandle MLIR_CreateValueBlockArg(
    MLIR_Context *cctx, string register_name, uint32_t /*idx*/,
    MLIR_TypeHandle type, MLIR_LocationHandle loc) {
    mlir::Type ty = (type != MLIR_INVALID_HANDLE)
        ? typeF(type) : mlir::Type(mlir::NoneType::get(cctx));
    mlir::Location mloc = (loc != MLIR_INVALID_HANDLE)
        ? locF(loc) : mlir::Location(mlir::UnknownLoc::get(cctx));
    if (!cctx->holderBlock) cctx->holderBlock = new mlir::Block();
    auto arg = cctx->holderBlock->addArgument(ty, mloc);
    if (register_name.size > 0) {
        cctx->valueNames[arg.getAsOpaquePointer()].assign(register_name.str, register_name.size);
    }
    return valH(arg);
}
extern "C" MLIR_ValueKind MLIR_GetValueKind(MLIR_Context *ctx, MLIR_ValueHandle h) {
    CHECK_OWN_VALUE(ctx, h, "MLIR_GetValueKind");
    auto v = valF(h);
    return llvm::isa<mlir::BlockArgument>(v) ? MLIR_VALUE_BLOCK_ARG : MLIR_VALUE_OP_RESULT;
}
extern "C" MLIR_TypeHandle MLIR_GetValueType(MLIR_Context *ctx, MLIR_ValueHandle h) {
    CHECK_OWN_VALUE(ctx, h, "MLIR_GetValueType");
    return typeH(valF(h).getType());
}
extern "C" string MLIR_GetValueRegisterName(MLIR_Context *ctx, MLIR_ValueHandle h) {
    CHECK_OWN_VALUE(ctx, h, "MLIR_GetValueRegisterName");
    if (h == MLIR_INVALID_HANDLE || ctx->valueNames.empty()) return str_lit("");
    auto it = ctx->valueNames.find(valF(h).getAsOpaquePointer());
    if (it == ctx->valueNames.end()) return str_lit("");
    return mkRefString(it->second);
}
extern "C" uint32_t MLIR_GetValueResultIndex(MLIR_Context *ctx, MLIR_ValueHandle h) {
    CHECK_OWN_VALUE(ctx, h, "MLIR_GetValueResultIndex");
    auto v = valF(h);
    if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(v)) return ba.getArgNumber();
    if (auto opr = llvm::dyn_cast<mlir::OpResult>(v)) return opr.getResultNumber();
    return 0;
}
extern "C" MLIR_OpHandle MLIR_GetValueDefiningOp(MLIR_Context *ctx, MLIR_ValueHandle h) {
    CHECK_OWN_VALUE(ctx, h, "MLIR_GetValueDefiningOp");
    auto v = valF(h);
    if (auto opr = llvm::dyn_cast<mlir::OpResult>(v)) return H(opr.getOwner());
    return MLIR_INVALID_HANDLE;
}

// -----------------------------------------------------------------------------
// Type
// -----------------------------------------------------------------------------

extern "C" MLIR_TypeHandle MLIR_CreateTypeInteger(MLIR_Context *cctx, uint32_t width,
                                                   bool /*is_signed*/) {
    return typeH(mlir::IntegerType::get(cctx, width));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *cctx, uint32_t width, bool is_bfloat) {
    MLIR_Context &ctx = *cctx;
    if (is_bfloat) return typeH(mlir::BFloat16Type::get(&ctx));
    switch (width) {
        case 16: return typeH(mlir::Float16Type::get(&ctx));
        case 32: return typeH(mlir::Float32Type::get(&ctx));
        case 64: return typeH(mlir::Float64Type::get(&ctx));
        default: return typeH(mlir::Float32Type::get(&ctx));
    }
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *cctx) {
    return typeH(mlir::IndexType::get(cctx));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *cctx) {
    return typeH(mlir::NoneType::get(cctx));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *cctx, const int64_t *shape,
                                                  size_t rank, MLIR_TypeHandle element_type) {
    MLIR_Context &ctx = *cctx;
    mlir::Type elem = (element_type == MLIR_INVALID_HANDLE)
                          ? mlir::Type(mlir::Float32Type::get(&ctx))
                          : typeF(element_type);
    if (shape == nullptr) {
        // Native impl uses shape==NULL to signal "shape not known"; print as
        // a rank-0 tensor (tensor<elem>) to match its textual output, not as
        // an unranked tensor (tensor<*xelem>).
        return typeH(mlir::RankedTensorType::get({}, elem));
    }
    llvm::SmallVector<int64_t, 8> dims;
    dims.reserve(rank);
    for (size_t i = 0; i < rank; i++) {
        int64_t d = shape[i];
        dims.push_back(d < 0 ? mlir::ShapedType::kDynamic : d);
    }
    return typeH(mlir::RankedTensorType::get(dims, elem));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *cctx, const int64_t *shape,
                                                  size_t rank, MLIR_TypeHandle element_type) {
    MLIR_Context &ctx = *cctx;
    mlir::Type elem = (element_type == MLIR_INVALID_HANDLE)
                          ? mlir::Type(mlir::Float32Type::get(&ctx))
                          : typeF(element_type);
    if (shape == nullptr) {
        return typeH(mlir::MemRefType::get({}, elem));
    }
    llvm::SmallVector<int64_t, 8> dims;
    dims.reserve(rank);
    for (size_t i = 0; i < rank; i++) {
        int64_t d = shape[i];
        dims.push_back(d < 0 ? mlir::ShapedType::kDynamic : d);
    }
    return typeH(mlir::MemRefType::get(dims, elem));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *ctx, MLIR_TypeHandle element,
                                                   bool has_addr, uint32_t addr) {
    // Upstream has no built-in dialect-agnostic pointer type. Use an opaque
    // type tagged "!tt.ptr<elem[, addr]>" so the printed form matches the
    // native backend.
    MLIR_Context &mctx = *ctx;
    std::string data = "ptr";
    auto elem = typeF(element);
    if (elem) {
        std::string ebuf;
        llvm::raw_string_ostream eos(ebuf);
        elem.print(eos);
        eos.flush();
        if (has_addr) {
            data += "<" + ebuf + ", " + std::to_string((unsigned)addr) + ">";
        } else {
            data += "<" + ebuf + ">";
        }
    }
    (void)ctx;
    return typeH(mlir::OpaqueType::get(mlir::StringAttr::get(&mctx, "tt"), data));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *cctx, string name) {
    MLIR_Context &ctx = *cctx;
    return typeH(mlir::OpaqueType::get(mlir::StringAttr::get(&ctx, "?"),
                                        llvm::StringRef(name.str, name.size)));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeFunction(MLIR_Context *cctx,
                                                    const MLIR_TypeHandle *inputs, size_t n_inputs,
                                                    const MLIR_TypeHandle *results, size_t n_results) {
    MLIR_Context &ctx = *cctx;
    llvm::SmallVector<mlir::Type, 4> in;
    in.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; i++) in.push_back(typeF(inputs[i]));
    llvm::SmallVector<mlir::Type, 4> out;
    out.reserve(n_results);
    for (size_t i = 0; i < n_results; i++) out.push_back(typeF(results[i]));
    return typeH(mlir::FunctionType::get(&ctx, in, out));
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeVector(MLIR_Context *cctx, const int64_t *shape,
                                                  size_t rank, MLIR_TypeHandle element_type) {
    API_CHECK(shape != nullptr && rank > 0, "MLIR_CreateTypeVector", "a vector needs a shape");
    API_CHECK(element_type != MLIR_INVALID_HANDLE, "MLIR_CreateTypeVector", "element type is NULL");
    llvm::SmallVector<int64_t, 4> dims(shape, shape + rank);
    return typeH(mlir::VectorType::get(dims, typeOf(cctx, element_type, "MLIR_CreateTypeVector")));
}

extern "C" MLIR_TypeKind MLIR_GetTypeKind(MLIR_Context *ctx, MLIR_TypeHandle h) {
    mlir::Type t = typeOf(ctx, h, "MLIR_GetTypeKind");
    if (!t) return MLIR_TYPE_KIND_INVALID;
    if (llvm::isa<mlir::IntegerType>(t)) return MLIR_TYPE_KIND_INTEGER;
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(t)) return MLIR_TYPE_KIND_LLVM_POINTER;
    if (llvm::isa<mlir::BFloat16Type>(t)) return MLIR_TYPE_KIND_BF16;
    if (llvm::isa<mlir::FloatType>(t)) return MLIR_TYPE_KIND_FLOAT;
    if (llvm::isa<mlir::IndexType>(t)) return MLIR_TYPE_KIND_INDEX;
    if (llvm::isa<mlir::LLVM::LLVMStructType>(t)) return MLIR_TYPE_KIND_LLVM_STRUCT;
    if (llvm::isa<mlir::LLVM::LLVMArrayType>(t)) return MLIR_TYPE_KIND_LLVM_ARRAY;
    if (llvm::isa<mlir::LLVM::LLVMFunctionType>(t)) return MLIR_TYPE_KIND_LLVM_FUNCTION;
    if (llvm::isa<mlir::LLVM::LLVMVoidType>(t)) return MLIR_TYPE_KIND_LLVM_VOID;
    if (llvm::isa<mlir::FunctionType>(t)) return MLIR_TYPE_KIND_FUNCTION;
    if (llvm::isa<mlir::VectorType>(t)) return MLIR_TYPE_KIND_VECTOR;
    if (llvm::isa<mlir::RankedTensorType>(t)) return MLIR_TYPE_KIND_TENSOR;
    if (llvm::isa<mlir::MemRefType>(t)) return MLIR_TYPE_KIND_MEMREF;
    if (llvm::isa<mlir::NoneType>(t)) return MLIR_TYPE_KIND_NONE;
    return MLIR_TYPE_KIND_OPAQUE;
}
extern "C" uint32_t MLIR_GetTypeWidth(MLIR_Context *ctx, MLIR_TypeHandle h) {
    mlir::Type t = typeOf(ctx, h, "MLIR_GetTypeWidth");
    if (auto it = llvm::dyn_cast_or_null<mlir::IntegerType>(t)) return it.getWidth();
    if (auto ft = llvm::dyn_cast_or_null<mlir::FloatType>(t)) return ft.getWidth();
    return 0;
}
extern "C" MLIR_TypeHandle MLIR_GetTypeElement(MLIR_Context *ctx, MLIR_TypeHandle h) {
    mlir::Type t = typeOf(ctx, h, "MLIR_GetTypeElement");
    if (auto at = llvm::dyn_cast_or_null<mlir::LLVM::LLVMArrayType>(t)) return typeH(at.getElementType());
    if (llvm::isa_and_nonnull<mlir::VectorType, mlir::RankedTensorType, mlir::MemRefType>(t))
        return typeH(llvm::cast<mlir::ShapedType>(t).getElementType());
    return MLIR_INVALID_HANDLE;
}
extern "C" uint64_t MLIR_GetTypeNumElements(MLIR_Context *ctx, MLIR_TypeHandle h) {
    mlir::Type t = typeOf(ctx, h, "MLIR_GetTypeNumElements");
    if (auto at = llvm::dyn_cast_or_null<mlir::LLVM::LLVMArrayType>(t)) return at.getNumElements();
    if (auto vt = llvm::dyn_cast_or_null<mlir::VectorType>(t)) return vt.getNumElements();
    return 0;
}
extern "C" size_t MLIR_GetTypeShapeRank(MLIR_Context *ctx, MLIR_TypeHandle h) {
    mlir::Type t = typeOf(ctx, h, "MLIR_GetTypeShapeRank");
    if (llvm::isa_and_nonnull<mlir::VectorType, mlir::RankedTensorType, mlir::MemRefType>(t))
        return llvm::cast<mlir::ShapedType>(t).getRank();
    return 0;
}
extern "C" int64_t MLIR_GetTypeShapeDim(MLIR_Context *ctx, MLIR_TypeHandle h, size_t idx) {
    API_CHECK(idx < MLIR_GetTypeShapeRank(ctx, h), "MLIR_GetTypeShapeDim", "index out of range");
    int64_t d = llvm::cast<mlir::ShapedType>(typeOf(ctx, h, "MLIR_GetTypeShapeDim")).getDimSize(idx);
    return mlir::ShapedType::isDynamic(d) ? -1 : d;
}
extern "C" size_t MLIR_GetTypeFunctionNumInputs(MLIR_Context *ctx, MLIR_TypeHandle h) {
    auto t = typeOf(ctx, h, "MLIR_GetTypeFunctionNumInputs");
    if (auto ft = llvm::dyn_cast_or_null<mlir::FunctionType>(t)) return ft.getNumInputs();
    if (auto ft = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFunctionType>(t)) return ft.getNumParams();
    return 0;
}
extern "C" MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_Context *ctx, MLIR_TypeHandle h, size_t idx) {
    API_CHECK(idx < MLIR_GetTypeFunctionNumInputs(ctx, h), "MLIR_GetTypeFunctionInput", "index out of range");
    auto t = typeOf(ctx, h, "MLIR_GetTypeFunctionInput");
    if (auto ft = llvm::dyn_cast_or_null<mlir::FunctionType>(t)) {
        if (idx >= ft.getNumInputs()) return MLIR_INVALID_HANDLE;
        return typeH(ft.getInput(idx));
    }
    if (auto ft = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFunctionType>(t)) {
        if (idx >= ft.getNumParams()) return MLIR_INVALID_HANDLE;
        return typeH(ft.getParamType(idx));
    }
    return MLIR_INVALID_HANDLE;
}
extern "C" size_t MLIR_GetTypeFunctionNumResults(MLIR_Context *ctx, MLIR_TypeHandle h) {
    auto t = typeOf(ctx, h, "MLIR_GetTypeFunctionNumResults");
    if (auto ft = llvm::dyn_cast_or_null<mlir::FunctionType>(t)) return ft.getNumResults();
    if (auto ft = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFunctionType>(t)) {
        // LLVMFunctionType has exactly one return type; "void" counts as 0
        // results to mirror mlir::FunctionType semantics for callers that
        // want to spell `declare void` etc.
        return llvm::isa<mlir::LLVM::LLVMVoidType>(ft.getReturnType()) ? 0 : 1;
    }
    return 0;
}
extern "C" MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_Context *ctx, MLIR_TypeHandle h, size_t idx) {
    API_CHECK(idx < MLIR_GetTypeFunctionNumResults(ctx, h), "MLIR_GetTypeFunctionResult", "index out of range");
    auto t = typeOf(ctx, h, "MLIR_GetTypeFunctionResult");
    if (auto ft = llvm::dyn_cast_or_null<mlir::FunctionType>(t)) {
        if (idx >= ft.getNumResults()) return MLIR_INVALID_HANDLE;
        return typeH(ft.getResult(idx));
    }
    if (auto ft = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFunctionType>(t)) {
        if (idx != 0) return MLIR_INVALID_HANDLE;
        if (llvm::isa<mlir::LLVM::LLVMVoidType>(ft.getReturnType())) return MLIR_INVALID_HANDLE;
        return typeH(ft.getReturnType());
    }
    return MLIR_INVALID_HANDLE;
}

extern "C" bool MLIR_GetTypeFunctionIsVarArg(MLIR_Context *ctx, MLIR_TypeHandle h) {
    auto t = typeOf(ctx, h, "MLIR_GetTypeFunctionIsVarArg");
    if (auto ft = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFunctionType>(t)) {
        return ft.isVarArg();
    }
    return false;
}

extern "C" size_t MLIR_GetTypeLLVMStructNumFields(MLIR_Context *ctx, MLIR_TypeHandle h) {
    auto st = llvm::dyn_cast_or_null<mlir::LLVM::LLVMStructType>(typeOf(ctx, h, "MLIR_GetTypeLLVMStructNumFields"));
    if (!st) return 0;
    return st.getBody().size();
}
extern "C" MLIR_TypeHandle MLIR_GetTypeLLVMStructField(MLIR_Context *ctx, MLIR_TypeHandle h, size_t idx) {
    API_CHECK(idx < MLIR_GetTypeLLVMStructNumFields(ctx, h), "MLIR_GetTypeLLVMStructField", "index out of range");
    auto st = llvm::dyn_cast_or_null<mlir::LLVM::LLVMStructType>(typeOf(ctx, h, "MLIR_GetTypeLLVMStructField"));
    if (!st || idx >= st.getBody().size()) return MLIR_INVALID_HANDLE;
    return typeH(st.getBody()[idx]);
}
extern "C" string MLIR_GetTypeLLVMStructName(MLIR_Context *ctx, MLIR_TypeHandle h) {
    auto st = llvm::dyn_cast_or_null<mlir::LLVM::LLVMStructType>(typeOf(ctx, h, "MLIR_GetTypeLLVMStructName"));
    if (!st || !st.isIdentified()) return str_lit("");
    // The name is owned by the MLIRContext's uniquer, so it outlives every
    // IR arena.
    llvm::StringRef n = st.getName();
    string r;
    r.str = const_cast<char *>(n.data());
    r.size = n.size();
    return r;
}
extern "C" bool MLIR_IsTypeLLVMStructOpaque(MLIR_Context *ctx, MLIR_TypeHandle h) {
    auto st = llvm::dyn_cast_or_null<mlir::LLVM::LLVMStructType>(typeOf(ctx, h, "MLIR_IsTypeLLVMStructOpaque"));
    return st && st.isOpaque();
}

extern "C" uint32_t MLIR_GetTypeLLVMPointerAddressSpace(MLIR_Context *ctx, MLIR_TypeHandle h) {
    auto p = llvm::dyn_cast_or_null<mlir::LLVM::LLVMPointerType>(typeOf(ctx, h, "MLIR_GetTypeLLVMPointerAddressSpace"));
    return p ? p.getAddressSpace() : 0;
}

extern "C" string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle h) {
    auto t = typeF(h);
    // Normalize opaque "unknown" types (dialect `?`, type `unknown`) to the
    // single token "unknown" so that the native and upstream backends emit
    // identical strings.
    if (auto opaq = llvm::dyn_cast<mlir::OpaqueType>(t)) {
        if (opaq.getDialectNamespace() == "?" && opaq.getTypeData() == "unknown") {
            return mkArenaString(ctx, std::string("unknown"));
        }
    }
    // Cache the printed form keyed by the interned Type pointer. mlir::Type
    // is uniqued in the MLIRContext, so the same logical type always maps
    // to the same opaque pointer — repeated GetTypeString on the same type
    // (extremely common during wasm lowering: every operand and result of
    // every op queries its type's name) becomes a hash lookup + arena copy.
    //
    // We keep the canonical std::string in this long-lived cache and copy
    // it into the (per-compile) arena on each hit, because the arena may
    // be reset between compiles even though the MLIRContext lives on.
    auto &typeStrings = ctx->typeStrings;
    const void *key = t.getAsOpaquePointer();
    auto it = typeStrings.find(key);
    if (it != typeStrings.end()) {
        return mkArenaString(ctx, it->second);
    }
    std::string buf;
    llvm::raw_string_ostream os(buf);
    t.print(os);
    os.flush();
    typeStrings[key] = buf;
    return mkArenaString(ctx, buf);
}

extern "C" string MLIR_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle h) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    mlir::OpPrintingFlags flags;
    flags.assumeVerified();
    F<mlir::Operation>(h)->print(os, flags);
    os.flush();
    buf.push_back('\n');
    return mkArenaString(ctx, buf);
}

extern "C" MLIR_OpHandle MLIR_ParseTextUpstream(MLIR_Context *ctx, string text) {
    MLIR_Context &mctx = *ctx;
    llvm::StringRef src(text.str, text.size);
    mlir::ParserConfig config(&mctx);
    mlir::OwningOpRef<mlir::ModuleOp> mod =
        mlir::parseSourceString<mlir::ModuleOp>(src, config);
    if (!mod) return MLIR_INVALID_HANDLE;
    // Release ownership; the operation lives in the MLIRContext arena.
    mlir::Operation *op = mod.release().getOperation();
    ctx->rootOps.insert(op);
    return H(op);
}

// -----------------------------------------------------------------------------
// Attribute — the handle is the mlir::Attribute (see attrH)
// -----------------------------------------------------------------------------

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *cctx, int64_t value, MLIR_TypeHandle type) {
    (void)cctx;
    return attrH(mlir::IntegerAttr::get(typeF(type), value));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *cctx, double value, MLIR_TypeHandle type) {
    (void)cctx;
    return attrH(mlir::FloatAttr::get(typeF(type), value));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *cctx, bool value) {
    return attrH(mlir::BoolAttr::get(cctx, value));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *cctx, string value) {
    return attrH(mlir::StringAttr::get(cctx, llvm::StringRef(value.str, value.size)));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeLLVMLinkageInternal(MLIR_Context *cctx) {
    return attrH(mlir::LLVM::LinkageAttr::get(cctx, mlir::LLVM::Linkage::Internal));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *cctx,
                                                           const MLIR_AttributeHandle *elements,
                                                           size_t count) {
    llvm::SmallVector<mlir::Attribute, 8> values;
    values.reserve(count);
    for (size_t i = 0; i < count; i++) values.push_back(attrF(elements[i]));
    return attrH(mlir::ArrayAttr::get(cctx, values));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *cctx,
                                                          const MLIR_NamedAttr *entries,
                                                          size_t count) {
    llvm::SmallVector<mlir::NamedAttribute, 8> v;
    v.reserve(count);
    for (size_t i = 0; i < count; i++)
        v.push_back(mlir::NamedAttribute(keyName(cctx, entries[i].key), attrF(entries[i].value)));
    return attrH(mlir::DictionaryAttr::get(cctx, v));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeType(MLIR_Context *cctx, MLIR_TypeHandle type) {
    (void)cctx;
    return attrH(mlir::TypeAttr::get(typeF(type)));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeSymbolRef(MLIR_Context *cctx, string value) {
    return attrH(mlir::FlatSymbolRefAttr::get(cctx, llvm::StringRef(value.str, value.size)));
}

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeDenseI32Array(MLIR_Context *cctx,
                                                                    const int32_t *values, size_t count) {
    llvm::SmallVector<int32_t, 8> v(values, values + count);
    return attrH(mlir::DenseI32ArrayAttr::get(cctx, v));
}

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeDenseI64Array(MLIR_Context *cctx,
                                                                    const int64_t *values, size_t count) {
    llvm::SmallVector<int64_t, 8> v(values, values + count);
    return attrH(mlir::DenseI64ArrayAttr::get(cctx, v));
}

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeDistinct(MLIR_Context *cctx,
                                                              MLIR_AttributeHandle referenced) {
    (void)cctx;
    return attrH(mlir::DistinctAttr::create(attrF(referenced)));
}

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeFastMath(MLIR_Context *cctx, MLIR_FastMath fmf) {
    auto flags = static_cast<mlir::LLVM::FastmathFlags>(fmf & MLIR_FMF_FAST);
    return attrH(mlir::LLVM::FastmathFlagsAttr::get(cctx, flags));
}

extern "C" MLIR_FastMath MLIR_GetAttributeFastMath(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    (void)ctx;
    if (h == MLIR_INVALID_HANDLE) return MLIR_FMF_NONE;
    mlir::Attribute a = attrF(h);
    if (auto fmf = llvm::dyn_cast_or_null<mlir::LLVM::FastmathFlagsAttr>(a))
        return static_cast<MLIR_FastMath>(fmf.getValue()) & MLIR_FMF_FAST;
    return MLIR_FMF_NONE;
}

extern "C" MLIR_FastMath MLIR_GetOpFastMath(MLIR_Context *ctx, MLIR_OpHandle h) {
    CHECK_OWN_OP(ctx, h, "MLIR_GetOpFastMath");
    auto *op = F<mlir::Operation>(h);
    if (auto fmi = mlir::dyn_cast<mlir::LLVM::FastmathFlagsInterface>(op)) {
        auto attr = fmi.getFastmathAttr();
        if (attr) return static_cast<MLIR_FastMath>(attr.getValue()) & MLIR_FMF_FAST;
        return MLIR_FMF_NONE;
    }
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, h, MLIR_KEY_FASTMATH_FLAGS);
    return MLIR_GetAttributeFastMath(ctx, a);
}

extern "C" void MLIR_SetOpFastMath(MLIR_Context *ctx, MLIR_OpHandle h, MLIR_FastMath fmf) {
    MLIR_SetOpAttr(ctx, h, MLIR_KEY_FASTMATH_FLAGS,
                   MLIR_CreateAttributeFastMath(ctx, fmf));
}

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeSplat(MLIR_Context *cctx,
                                                          MLIR_TypeHandle vec_type,
                                                          MLIR_AttributeHandle scalar_attr) {
    (void)cctx;
    auto shaped = llvm::cast<mlir::ShapedType>(typeF(vec_type));
    return attrH(mlir::SplatElementsAttr::get(shaped, attrF(scalar_attr)));
}

static bool upstream_lane_in_bounds(mlir::Type ty, int64_t lane) {
    if (lane < 0) return true;
    if (auto st = llvm::dyn_cast<mlir::ShapedType>(ty)) {
        if (!st.hasStaticShape()) return false;
        return (int64_t)lane < st.getNumElements();
    }
    return lane == 0;
}

extern "C" bool MLIR_MatchConstantFloat(MLIR_Context *ctx, MLIR_ValueHandle v,
                                        int64_t lane, double *out) {
    if (v == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetValueKind(ctx, v) != MLIR_VALUE_OP_RESULT) return false;
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
    if (def == MLIR_INVALID_HANDLE) return false;
    auto *op = F<mlir::Operation>(def);
    MLIR_OpType t = MLIR_GetOpType(ctx, def);
    if (t == OP_TYPE_LLVM_MLIR_ZERO) {
        mlir::Type vty = valF(v).getType();
        mlir::Type ety = vty;
        if (auto st = llvm::dyn_cast<mlir::ShapedType>(vty)) ety = st.getElementType();
        if (llvm::isa<mlir::FloatType>(ety) && upstream_lane_in_bounds(vty, lane)) {
            if (out) *out = 0.0;
            return true;
        }
        return false;
    }
    if (!MLIR_OpHasTrait(ctx, def, MLIR_T_CONSTANT)) return false;
    mlir::Attribute a = op->getAttr("value");
    if (!a) return false;
    if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(a)) {
        if (lane > 0) return false;
        if (out) *out = fa.getValueAsDouble();
        return true;
    }
    if (auto da = llvm::dyn_cast<mlir::DenseFPElementsAttr>(a)) {
        if (da.isSplat()) {
            if (!upstream_lane_in_bounds(da.getType(), lane)) return false;
            if (out) *out = da.getSplatValue<llvm::APFloat>().convertToDouble();
            return true;
        }
        if (lane < 0 || lane >= da.getNumElements()) return false;
        if (out) *out = (*(da.getValues<llvm::APFloat>().begin() + lane)).convertToDouble();
        return true;
    }
    return false;
}

extern "C" bool MLIR_MatchConstantInt(MLIR_Context *ctx, MLIR_ValueHandle v,
                                      int64_t lane, int64_t *out) {
    if (v == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetValueKind(ctx, v) != MLIR_VALUE_OP_RESULT) return false;
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
    if (def == MLIR_INVALID_HANDLE) return false;
    auto *op = F<mlir::Operation>(def);
    MLIR_OpType t = MLIR_GetOpType(ctx, def);
    if (t == OP_TYPE_LLVM_MLIR_ZERO) {
        mlir::Type vty = valF(v).getType();
        mlir::Type ety = vty;
        if (auto st = llvm::dyn_cast<mlir::ShapedType>(vty)) ety = st.getElementType();
        if ((llvm::isa<mlir::IntegerType>(ety) || llvm::isa<mlir::IndexType>(ety)) &&
            upstream_lane_in_bounds(vty, lane)) {
            if (out) *out = 0;
            return true;
        }
        return false;
    }
    if (!MLIR_OpHasTrait(ctx, def, MLIR_T_CONSTANT)) return false;
    mlir::Attribute a = op->getAttr("value");
    if (!a) return false;
    if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(a)) {
        if (lane > 0) return false;
        if (out) *out = ia.getValue().getSExtValue();
        return true;
    }
    if (auto da = llvm::dyn_cast<mlir::DenseIntElementsAttr>(a)) {
        if (da.isSplat()) {
            if (!upstream_lane_in_bounds(da.getType(), lane)) return false;
            if (out) *out = da.getSplatValue<llvm::APInt>().getSExtValue();
            return true;
        }
        if (lane < 0 || lane >= da.getNumElements()) return false;
        if (out) *out = (*(da.getValues<llvm::APInt>().begin() + lane)).getSExtValue();
        return true;
    }
    return false;
}

extern "C" bool MLIR_IsConstantZero(MLIR_Context *ctx, MLIR_ValueHandle v, int64_t lane) {
    if (v == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetValueKind(ctx, v) == MLIR_VALUE_OP_RESULT) {
        MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
        if (def != MLIR_INVALID_HANDLE) {
            MLIR_OpType t = MLIR_GetOpType(ctx, def);
            if (t == OP_TYPE_LLVM_MLIR_ZERO || t == OP_TYPE_LLVM_MLIR_NULL)
                return upstream_lane_in_bounds(valF(v).getType(), lane);
        }
    }
    int64_t iv = 0;
    if (MLIR_MatchConstantInt(ctx, v, lane, &iv)) return iv == 0;
    double dv = 0.0;
    if (MLIR_MatchConstantFloat(ctx, v, lane, &dv)) return dv == 0.0;
    return false;
}

extern "C" size_t MLIR_GetAttributeDenseSize(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    (void)ctx;
    mlir::Attribute a = attrF(h);
    if (auto d = llvm::dyn_cast_or_null<mlir::DenseI32ArrayAttr>(a)) return d.size();
    if (auto d = llvm::dyn_cast_or_null<mlir::DenseI64ArrayAttr>(a)) return d.size();
    if (auto d = llvm::dyn_cast_or_null<mlir::DenseIntElementsAttr>(a)) return d.getNumElements();
    return 0;
}
extern "C" int64_t MLIR_GetAttributeDenseElement(MLIR_Context *ctx, MLIR_AttributeHandle h, size_t i) {
    API_CHECK(i < MLIR_GetAttributeDenseSize(ctx, h), "MLIR_GetAttributeDenseElement", "index out of range");
    mlir::Attribute a = attrF(h);
    if (auto d = llvm::dyn_cast<mlir::DenseI32ArrayAttr>(a)) return d[i];
    if (auto d = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(a)) return d[i];
    auto d = llvm::cast<mlir::DenseIntElementsAttr>(a);
    return (*(d.getValues<llvm::APInt>().begin() + i)).getSExtValue();
}

extern "C" MLIR_AttrKey MLIR_InternAttrKey(MLIR_Context *ctx, string name) {
    API_CHECK(name.size > 0, "MLIR_InternAttrKey", "empty attribute name");
    return attr_key_table_intern(&ctx->keyTable, name);
}
extern "C" MLIR_AttrKey MLIR_LookupAttrKey(MLIR_Context *ctx, string name) {
    return attr_key_table_lookup(&ctx->keyTable, name);
}
extern "C" string MLIR_GetAttrKeyName(MLIR_Context *ctx, MLIR_AttrKey key) {
    return attr_key_table_name(&ctx->keyTable, key);
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *cctx) {
    return MLIR_CreateTypeLLVMPointerInAddressSpace(cctx, 0);
}

// Adds address space to the pointer type, e.g. !llvm.ptr<3> for address space 3.
extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMPointerInAddressSpace(
    MLIR_Context *cctx, uint32_t address_space) {
    MLIR_Context &ctx = *cctx;
    return typeH(mlir::LLVM::LLVMPointerType::get(&ctx, address_space));
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMStructIdentified(MLIR_Context *cctx, string name) {
    MLIR_Context &ctx = *cctx;
    return typeH(mlir::LLVM::LLVMStructType::getIdentified(&ctx,
                     llvm::StringRef(name.str, name.size)));
}

extern "C" void MLIR_SetTypeLLVMStructBody(MLIR_Context *cctx, MLIR_TypeHandle struct_ty,
                                            const MLIR_TypeHandle *fields, size_t n_fields) {
    auto t = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(typeF(struct_ty));
    if (!t) {
        MLIR_EmitDiagnostic(cctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                            "MLIR_SetTypeLLVMStructBody: not an LLVM struct type");
        return;
    }
    llvm::SmallVector<mlir::Type, 8> body;
    body.reserve(n_fields);
    for (size_t i = 0; i < n_fields; i++) body.push_back(typeF(fields[i]));
    if (llvm::failed(t.setBody(body, /*isPacked=*/false))) {
        MLIR_EmitDiagnostic(cctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                            "MLIR_SetTypeLLVMStructBody: setBody failed (already set?)");
    }
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMArray(MLIR_Context *cctx, MLIR_TypeHandle elem,
                                                     uint64_t count) {
    return typeH(mlir::LLVM::LLVMArrayType::get(typeF(elem), count));
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMFunction(MLIR_Context *cctx,
                                                       MLIR_TypeHandle result,
                                                       const MLIR_TypeHandle *inputs,
                                                       size_t n_inputs,
                                                       bool is_var_arg) {
    llvm::SmallVector<mlir::Type, 8> in;
    in.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; i++) in.push_back(typeF(inputs[i]));
    return typeH(mlir::LLVM::LLVMFunctionType::get(typeF(result), in, is_var_arg));
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMVoid(MLIR_Context *cctx) {
    MLIR_Context &ctx = *cctx;
    return typeH(mlir::LLVM::LLVMVoidType::get(&ctx));
}

extern "C" MLIR_OpHandle MLIR_CreateLLVMGlobalString(MLIR_Context *cctx,
                                                     string sym_name,
                                                     string bytes,
                                                     MLIR_LocationHandle location) {
    MLIR_Context &ctx = *cctx;
    mlir::Location loc = (location == MLIR_INVALID_HANDLE)
                             ? mlir::Location(mlir::UnknownLoc::get(&ctx))
                             : locF(location);
    auto i8 = mlir::IntegerType::get(&ctx, 8);
    auto arrTy = mlir::LLVM::LLVMArrayType::get(i8, bytes.size);
    auto strAttr = mlir::StringAttr::get(&ctx,
        llvm::StringRef(bytes.str, bytes.size));
    mlir::OpBuilder b(&ctx);
    auto op = b.create<mlir::LLVM::GlobalOp>(
        loc, arrTy, /*isConstant=*/true,
        mlir::LLVM::Linkage::Private,
        llvm::StringRef(sym_name.str, sym_name.size),
        strAttr, /*alignment=*/0);
    cctx->rootOps.insert(op.getOperation());
    return H(op.getOperation());
}

extern "C" MLIR_OpHandle MLIR_CreateLLVMGlobal(MLIR_Context *cctx,
                                               string sym_name,
                                               MLIR_TypeHandle elem_ty,
                                               bool is_constant,
                                               int init_kind,
                                               int64_t init_int,
                                               double init_float,
                                               MLIR_BlockHandle *out_init_block,
                                               MLIR_LocationHandle location) {
    MLIR_Context &ctx = *cctx;
    mlir::Location loc = (location == MLIR_INVALID_HANDLE)
                             ? mlir::Location(mlir::UnknownLoc::get(&ctx))
                             : locF(location);
    mlir::Type ty = typeF(elem_ty);
    mlir::Attribute initAttr;
    if (init_kind == 0) {
        initAttr = mlir::IntegerAttr::get(ty, init_int);
    } else if (init_kind == 1) {
        initAttr = mlir::FloatAttr::get(ty, init_float);
    }
    mlir::LLVM::Linkage linkage = (init_kind == 4)
                                      ? mlir::LLVM::Linkage::External
                                      : mlir::LLVM::Linkage::Internal;
    mlir::OpBuilder b(&ctx);
    auto op = b.create<mlir::LLVM::GlobalOp>(
        loc, ty, is_constant,
        linkage,
        llvm::StringRef(sym_name.str, sym_name.size),
        initAttr, /*alignment=*/0);
    if (init_kind == 2) {
        // Caller will populate the entry block.
        mlir::Region &region = op.getInitializerRegion();
        mlir::Block *entry = new mlir::Block();
        region.push_back(entry);
        if (out_init_block) *out_init_block = H(entry);
    } else if (out_init_block) {
        *out_init_block = MLIR_INVALID_HANDLE;
    }
    cctx->rootOps.insert(op.getOperation());
    return H(op.getOperation());
}

extern "C" MLIR_OpHandle MLIR_CreateLLVMGlobalArrayInit(MLIR_Context *cctx,
                                                        string sym_name,
                                                        MLIR_TypeHandle array_ty,
                                                        bool is_constant,
                                                        string bytes,
                                                        MLIR_LocationHandle location) {
    MLIR_Context &ctx = *cctx;
    mlir::Location loc = (location == MLIR_INVALID_HANDLE)
                             ? mlir::Location(mlir::UnknownLoc::get(&ctx))
                             : locF(location);
    mlir::Type arrTy = typeF(array_ty);
    // Build a region initializer with a chain of `llvm.insertvalue`s
    // off a `llvm.mlir.undef` so the result matches the global's array
    // type. MLIR's LLVM dialect only accepts a `StringAttr` value for
    // !llvm.array<N x i8>; for any other element type we *must* go
    // through a region initializer.
    mlir::OpBuilder b(&ctx);
    auto op = b.create<mlir::LLVM::GlobalOp>(
        loc, arrTy, is_constant,
        mlir::LLVM::Linkage::Internal,
        llvm::StringRef(sym_name.str, sym_name.size),
        /*value=*/mlir::Attribute(), /*alignment=*/0);
    mlir::Region &region = op.getInitializerRegion();
    mlir::Block *entry = new mlir::Block();
    region.push_back(entry);
    b.setInsertionPointToStart(entry);
    auto arr = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(arrTy);
    if (!arr) {
        // Fallback to zero-init.
        auto zero = b.create<mlir::LLVM::ZeroOp>(loc, arrTy);
        b.create<mlir::LLVM::ReturnOp>(loc, zero.getRes());
        cctx->rootOps.insert(op.getOperation());
        return H(op.getOperation());
    }
    mlir::Type elemTy = arr.getElementType();
    uint64_t n = arr.getNumElements();
    uint64_t elem_size = 0;
    if (auto it = llvm::dyn_cast<mlir::IntegerType>(elemTy)) {
        elem_size = (uint64_t)((it.getWidth() + 7) / 8);
    }
    if (elem_size == 0 || bytes.size != n * elem_size) {
        auto zero = b.create<mlir::LLVM::ZeroOp>(loc, arrTy);
        b.create<mlir::LLVM::ReturnOp>(loc, zero.getRes());
        cctx->rootOps.insert(op.getOperation());
        return H(op.getOperation());
    }
    mlir::Value cur = b.create<mlir::LLVM::UndefOp>(loc, arrTy).getRes();
    const unsigned char *raw = (const unsigned char *)bytes.str;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t v = 0;
        for (uint64_t b2 = 0; b2 < elem_size; b2++) {
            v |= (uint64_t)raw[i * elem_size + b2] << (8 * b2);
        }
        // Sign-extend small element types to their natural width before
        // building the IntegerAttr.
        int64_t sv;
        if (elem_size == 1) sv = (int64_t)(int8_t)v;
        else if (elem_size == 2) sv = (int64_t)(int16_t)v;
        else if (elem_size == 4) sv = (int64_t)(int32_t)v;
        else sv = (int64_t)v;
        auto cst = b.create<mlir::LLVM::ConstantOp>(
            loc, elemTy, mlir::IntegerAttr::get(elemTy, sv));
        cur = b.create<mlir::LLVM::InsertValueOp>(
            loc, cur, cst.getRes(), llvm::ArrayRef<int64_t>{(int64_t)i});
    }
    b.create<mlir::LLVM::ReturnOp>(loc, cur);
    cctx->rootOps.insert(op.getOperation());
    return H(op.getOperation());
}

extern "C" MLIR_AttrKind MLIR_GetAttributeKind(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    API_CHECK(h != MLIR_INVALID_HANDLE, "MLIR_GetAttributeKind", "attribute handle is NULL");
    auto value = attrF(h);
    if (llvm::isa<mlir::StringAttr>(value))  return MLIR_ATTR_KIND_STRING;
    if (llvm::isa<mlir::FlatSymbolRefAttr>(value)) return MLIR_ATTR_KIND_STRING;
    if (llvm::isa<mlir::BoolAttr>(value))    return MLIR_ATTR_KIND_BOOL;
    if (llvm::isa<mlir::IntegerAttr>(value)) return MLIR_ATTR_KIND_INTEGER;
    if (llvm::isa<mlir::FloatAttr>(value))   return MLIR_ATTR_KIND_FLOAT;
    if (llvm::isa<mlir::ArrayAttr>(value))   return MLIR_ATTR_KIND_ARRAY;
    if (llvm::isa<mlir::DictionaryAttr>(value)) return MLIR_ATTR_KIND_DICT;
    if (llvm::isa<mlir::TypeAttr>(value))    return MLIR_ATTR_KIND_TYPE;
    return MLIR_ATTR_KIND_OTHER;
}
extern "C" int64_t MLIR_GetAttributeInteger(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    API_CHECK(h != MLIR_INVALID_HANDLE && llvm::isa<mlir::IntegerAttr>(attrF(h)), "MLIR_GetAttributeInteger", "not an integer or bool attribute");
    return llvm::cast<mlir::IntegerAttr>(attrF(h)).getInt();
}
extern "C" double MLIR_GetAttributeFloat(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    API_CHECK(h != MLIR_INVALID_HANDLE && llvm::isa<mlir::FloatAttr>(attrF(h)), "MLIR_GetAttributeFloat", "not a float attribute");
    return llvm::cast<mlir::FloatAttr>(attrF(h)).getValueAsDouble();
}
extern "C" MLIR_TypeHandle MLIR_GetAttributeType(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    auto attr = attrF(h);
    if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(attr)) return typeH(ia.getType());
    if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(attr))   return typeH(fa.getType());
    return MLIR_INVALID_HANDLE;
}
extern "C" MLIR_TypeHandle MLIR_GetAttributeTypeValue(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    auto attr = attrF(h);
    if (auto ta = llvm::dyn_cast<mlir::TypeAttr>(attr)) return typeH(ta.getValue());
    return MLIR_INVALID_HANDLE;
}
extern "C" bool MLIR_GetAttributeBool(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    API_CHECK(h != MLIR_INVALID_HANDLE && llvm::isa<mlir::BoolAttr>(attrF(h)), "MLIR_GetAttributeBool", "not a bool attribute");
    return llvm::cast<mlir::BoolAttr>(attrF(h)).getValue();
}
extern "C" string MLIR_GetAttributeString(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    API_CHECK(MLIR_GetAttributeKind(ctx, h) == MLIR_ATTR_KIND_STRING || MLIR_GetAttributeKind(ctx, h) == MLIR_ATTR_KIND_OTHER, "MLIR_GetAttributeString", "not a string attribute");
    auto value = attrF(h);
    if (auto s = llvm::dyn_cast<mlir::StringAttr>(value)) return mkRefString(s.getValue());
    // Other kinds have no stored text: return `@name` for a symbol
    // reference and the printed form otherwise (e.g. `#llvm.linkage<internal>`
    // for LLVM_LinkageAttr; the native backend stores such payloads as
    // strings and returns them verbatim). The text is interned in the
    // context, so it lives as long as the attribute.
    std::string buf;
    if (auto sr = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(value)) {
        buf = "@";
        buf += sr.getValue().str();
    } else {
        llvm::raw_string_ostream os(buf);
        value.print(os);
    }
    auto &strings = ctxOf(value.getContext())->attrStrings;
    return mkRefString(strings.insert(buf).first->getKey());
}
extern "C" string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    attrF(h).print(os);
    os.flush();
    return mkArenaString(ctx, buf);
}
extern "C" size_t MLIR_GetAttributeArraySize(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    API_CHECK(h != MLIR_INVALID_HANDLE && llvm::isa<mlir::ArrayAttr>(attrF(h)), "MLIR_GetAttributeArraySize", "not an array attribute");
    return llvm::cast<mlir::ArrayAttr>(attrF(h)).size();
}
extern "C" MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_Context *ctx, MLIR_AttributeHandle h, size_t i) {
    API_CHECK(i < MLIR_GetAttributeArraySize(ctx, h), "MLIR_GetAttributeArrayElement", "index out of range");
    return attrH(llvm::cast<mlir::ArrayAttr>(attrF(h))[i]);
}
extern "C" size_t MLIR_GetAttributeDictSize(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    API_CHECK(h != MLIR_INVALID_HANDLE && llvm::isa<mlir::DictionaryAttr>(attrF(h)), "MLIR_GetAttributeDictSize", "not a dictionary attribute");
    return llvm::cast<mlir::DictionaryAttr>(attrF(h)).size();
}
extern "C" MLIR_AttrKey MLIR_GetAttributeDictKey(MLIR_Context *ctx, MLIR_AttributeHandle h, size_t i) {
    API_CHECK(i < MLIR_GetAttributeDictSize(ctx, h), "MLIR_GetAttributeDictKey", "index out of range");
    return keyOf(ctx, llvm::cast<mlir::DictionaryAttr>(attrF(h)).getValue()[i].getName());
}
extern "C" MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_Context *ctx, MLIR_AttributeHandle h, size_t i) {
    API_CHECK(i < MLIR_GetAttributeDictSize(ctx, h), "MLIR_GetAttributeDictElement", "index out of range");
    return attrH(llvm::cast<mlir::DictionaryAttr>(attrF(h)).getValue()[i].getValue());
}

// -----------------------------------------------------------------------------
// Location
// -----------------------------------------------------------------------------
//
// MLIR_LocationHandle is the opaque pointer of mlir::Location. Information
// the upstream Location system cannot represent (e.g. our `original_text`
// blob, or the parser's `#locN` reference id) is dropped on the floor;
// printing locations through parser_upstream is therefore expected to be
// less faithful than the native parser.

namespace {
} // namespace

extern "C" MLIR_LocationHandle MLIR_CreateLocationUnknown(MLIR_Context *cctx, string /*orig*/) {
    return locH(cctx->unknownLoc);
}
extern "C" MLIR_LocationHandle MLIR_CreateLocationFile(MLIR_Context *cctx, string filename,
                                                        int line, int column) {
    MLIR_Context &ctx = *cctx;
    return locH(mlir::FileLineColLoc::get(
        &ctx, llvm::StringRef(filename.str, filename.size),
        (unsigned)line, (unsigned)column));
}
extern "C" MLIR_LocationHandle MLIR_CreateLocationName(MLIR_Context *cctx, string name) {
    MLIR_Context &ctx = *cctx;
    return locH(mlir::NameLoc::get(
        mlir::StringAttr::get(&ctx, llvm::StringRef(name.str, name.size))));
}
extern "C" MLIR_LocationHandle MLIR_CreateLocationRef(MLIR_Context *cctx, int /*ref_id*/) {
    return locH(mlir::UnknownLoc::get(cctx));
}

extern "C" MLIR_LocationKind MLIR_GetLocationKind(MLIR_Context *ctx, MLIR_LocationHandle h) {
    if (h == MLIR_INVALID_HANDLE) return MLIR_LOC_UNKNOWN;
    auto loc = locF(h);
    if (llvm::isa<mlir::FileLineColLoc>(loc)) return MLIR_LOC_FILE;
    if (llvm::isa<mlir::NameLoc>(loc))        return MLIR_LOC_NAME;
    if (llvm::isa<mlir::CallSiteLoc>(loc))    return MLIR_LOC_CALLSITE;
    if (llvm::isa<mlir::FusedLoc>(loc))       return MLIR_LOC_FUSED;
    return MLIR_LOC_UNKNOWN;
}
extern "C" string MLIR_GetLocationOriginalText(MLIR_Context *, MLIR_LocationHandle) {
    return mkRefString(llvm::StringRef());
}
extern "C" string MLIR_GetLocationFileFilename(MLIR_Context *ctx, MLIR_LocationHandle h) {
    if (auto fl = llvm::dyn_cast<mlir::FileLineColLoc>(locF(h)))
        return mkRefString(fl.getFilename().getValue());
    return mkRefString(llvm::StringRef());
}
extern "C" int MLIR_GetLocationFileLine(MLIR_Context *ctx, MLIR_LocationHandle h) {
    if (auto fl = llvm::dyn_cast<mlir::FileLineColLoc>(locF(h)))
        return (int)fl.getLine();
    return 0;
}
extern "C" int MLIR_GetLocationFileColumn(MLIR_Context *ctx, MLIR_LocationHandle h) {
    if (auto fl = llvm::dyn_cast<mlir::FileLineColLoc>(locF(h)))
        return (int)fl.getColumn();
    return 0;
}
extern "C" string MLIR_GetLocationName(MLIR_Context *ctx, MLIR_LocationHandle h) {
    if (auto nl = llvm::dyn_cast<mlir::NameLoc>(locF(h)))
        return mkRefString(nl.getName().getValue());
    return mkRefString(llvm::StringRef());
}

extern "C" MLIR_LocationHandle MLIR_GetValueLocation(MLIR_Context *ctx, MLIR_ValueHandle h) {
    CHECK_OWN_VALUE(ctx, h, "MLIR_GetValueLocation");
    if (h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto v = valF(h);
    if (!v) return MLIR_INVALID_HANDLE;
    return locH(v.getLoc());
}

// -----------------------------------------------------------------------------
// Misc
// -----------------------------------------------------------------------------

extern "C" MLIR_OpType MLIR_RegisterOpType(MLIR_Context *ctx, string name) {
    API_CHECK(ctx != nullptr, "MLIR_RegisterOpType", "context is NULL");
    API_CHECK(name.size > 0, "MLIR_RegisterOpType", "empty op name");
    return registerOpType(ctx, llvm::StringRef(name.str, name.size));
}
extern "C" MLIR_OpType MLIR_LookupOpType(MLIR_Context *ctx, string name) {
    API_CHECK(ctx != nullptr, "MLIR_LookupOpType", "context is NULL");
    return op_type_table_lookup(&ctx->opTypeTable, name);
}
extern "C" string MLIR_GetOpTypeName(MLIR_Context *ctx, MLIR_OpType type) {
    API_CHECK(ctx != nullptr, "MLIR_GetOpTypeName", "context is NULL");
    return op_type_table_name(&ctx->opTypeTable, type);
}

// -----------------------------------------------------------------------------
// IR mutation primitives (Stage B / Stage C use only)
// -----------------------------------------------------------------------------

extern "C" void MLIR_ReplaceAllUsesOfValue(MLIR_Context *cctx,
                                           MLIR_ValueHandle old_h,
                                           MLIR_ValueHandle new_h) {
    (void)cctx;
    if (old_h == MLIR_INVALID_HANDLE || new_h == MLIR_INVALID_HANDLE) return;
    mlir::Value oldV = valF(old_h);
    mlir::Value newV = valF(new_h);
    if (!oldV || !newV) return;
    oldV.replaceAllUsesWith(newV);
}

namespace {
#ifdef MLIR_API_CHECKS
// Erasing requires that nothing outside the erased subtree uses a value or a
// block defined inside it (mlir_api.h): such a use would point into freed
// memory, and MLIR_DestroyContext would write through it when it drops the
// references of the parentless IR. `inside(owner)` says whether a using op is
// in the subtree. O(size of the subtree), checked builds only.
template <class Inside>
bool blockUsedOutside(mlir::Block &b, Inside inside);
template <class Inside>
bool opUsedOutside(mlir::Operation *o, Inside inside) {
    for (mlir::Value r : o->getResults())
        for (mlir::OpOperand &u : r.getUses())
            if (!inside(u.getOwner())) return true;
    for (mlir::Region &region : o->getRegions())
        for (mlir::Block &b : region)
            if (blockUsedOutside(b, inside)) return true;
    return false;
}
template <class Inside>
bool blockUsedOutside(mlir::Block &b, Inside inside) {
    for (mlir::BlockArgument a : b.getArguments())
        for (mlir::OpOperand &u : a.getUses())
            if (!inside(u.getOwner())) return true;
    for (mlir::BlockOperand &u : b.getUses())
        if (!inside(u.getOwner())) return true;
    for (mlir::Operation &o : b)
        if (opUsedOutside(&o, inside)) return true;
    return false;
}
#endif

} // namespace

// Destroys the op (and the ops nested in it), as mlir::Operation::erase does.
// The handle is dead afterwards, as on the native backend (mlir_api.h).
extern "C" void MLIR_EraseOp(MLIR_Context *cctx, MLIR_OpHandle op_h) {
    API_CHECK(op_h != MLIR_INVALID_HANDLE, "MLIR_EraseOp", "op handle is NULL");
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
#ifdef MLIR_API_CHECKS
    API_CHECK(!opUsedOutside(op, [op](mlir::Operation *owner) { return op->isAncestor(owner); }),
              "MLIR_EraseOp", "a value or block defined in the op or nested in it is used outside it");
#endif
    if (!cctx->opAttrCache.empty() || !cctx->valueNames.empty())
        forgetErasedOp(cctx, op);
    if (!op->getBlock())
        cctx->rootOps.erase(op);
    op->erase();
}

extern "C" void MLIR_DetachOp(MLIR_Context *cctx, MLIR_OpHandle op_h) {
    API_CHECK(op_h != MLIR_INVALID_HANDLE, "MLIR_DetachOp", "op handle is NULL");
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    auto *blk = op->getBlock();
    if (!blk) return;
    op->remove();
    cctx->rootOps.insert(op);
}

extern "C" MLIR_RegionHandle MLIR_TakeOpRegion(MLIR_Context *cctx,
                                                MLIR_OpHandle op_h,
                                                size_t idx) {
    if (op_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *op = F<mlir::Operation>(op_h);
    if (idx >= op->getNumRegions()) return MLIR_INVALID_HANDLE;
    auto *out = new mlir::Region();
    out->takeBody(op->getRegion(idx));
    cctx->rootRegions.insert(out);
    return H(out);
}

extern "C" MLIR_BlockHandle MLIR_GetOpParentBlock(MLIR_Context *ctx, MLIR_OpHandle op_h) {
    CHECK_OWN_OP(ctx, op_h, "MLIR_GetOpParentBlock");
    if (op_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *op = F<mlir::Operation>(op_h);
    auto *blk = op->getBlock();
    return blk ? H(blk) : MLIR_INVALID_HANDLE;
}

extern "C" MLIR_RegionHandle MLIR_GetBlockParentRegion(MLIR_Context *ctx, MLIR_BlockHandle blk_h) {
    CHECK_OWN_BLOCK(ctx, blk_h, "MLIR_GetBlockParentRegion");
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    auto *r = blk->getParent();
    return r ? H(r) : MLIR_INVALID_HANDLE;
}

extern "C" void MLIR_MoveOpToBlockEnd(MLIR_Context *cctx, MLIR_OpHandle op_h,
                                       MLIR_BlockHandle dest_h) {
    if (op_h == MLIR_INVALID_HANDLE || dest_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    auto *dest = F<mlir::Block>(dest_h);
    if (op->getBlock()) {
        op->moveBefore(dest, dest->end());
    } else {
        // Operation::moveBefore needs a current block.
        cctx->rootOps.erase(op);
        dest->push_back(op);
    }
}

extern "C" void MLIR_MoveBlockToRegionEnd(MLIR_Context *cctx, MLIR_BlockHandle blk_h,
                                          MLIR_RegionHandle dest_h) {
    if (blk_h == MLIR_INVALID_HANDLE || dest_h == MLIR_INVALID_HANDLE) return;
    auto *blk = F<mlir::Block>(blk_h);
    auto *dest = F<mlir::Region>(dest_h);
    // Block::moveBefore requires the block to already have a parent region;
    // for blocks that don't, just splice into dest directly.
    if (blk->getParent() == nullptr) {
        cctx->rootBlocks.erase(blk);
        dest->push_back(blk);
    } else {
        blk->moveBefore(dest, dest->end());
    }
}

extern "C" void MLIR_SetOpOperand(MLIR_Context *cctx, MLIR_OpHandle op_h,
                                  size_t idx, MLIR_ValueHandle v_h) {
    (void)cctx;
    if (op_h == MLIR_INVALID_HANDLE || v_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    if (idx >= op->getNumOperands()) return;
    mlir::Value v = valF(v_h);
    if (!v) return;
    op->setOperand(static_cast<unsigned>(idx), v);
}

extern "C" void MLIR_SetOpSuccessor(MLIR_Context *cctx, MLIR_OpHandle op_h,
                                    size_t succ_idx, MLIR_BlockHandle blk_h) {
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    if (succ_idx >= op->getNumSuccessors()) return;
    auto *blk = (blk_h == MLIR_INVALID_HANDLE) ? nullptr : F<mlir::Block>(blk_h);
    op->setSuccessor(blk, static_cast<unsigned>(succ_idx));
}

extern "C" void MLIR_SetOpSuccessorOperands(MLIR_Context *cctx, MLIR_OpHandle op_h,
                                            size_t succ_idx,
                                            const MLIR_ValueHandle *values, size_t n) {
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(op);
    if (!branch) return;
    if (succ_idx >= op->getNumSuccessors()) return;
    mlir::SuccessorOperands sops = branch.getSuccessorOperands(static_cast<unsigned>(succ_idx));
    mlir::MutableOperandRange mut = sops.getMutableForwardedOperands();
    llvm::SmallVector<mlir::Value, 8> vals;
    vals.reserve(n);
    for (size_t i = 0; i < n; i++) {
        if (values && values[i] != MLIR_INVALID_HANDLE) {
            vals.push_back(valF(values[i]));
        } else {
            vals.push_back(mlir::Value());
        }
    }
    mut.assign(vals);
}

extern "C" void MLIR_EraseBlock(MLIR_Context *cctx, MLIR_BlockHandle blk_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return;
    auto *blk = F<mlir::Block>(blk_h);
#ifdef MLIR_API_CHECKS
    // No branch outside the block may still target it (mlir_api.h).
    for (mlir::BlockOperand &u : blk->getUses())
        API_CHECK(blk->findAncestorOpInBlock(*u.getOwner()) != nullptr, "MLIR_EraseBlock",
                  "the block is still a successor of a block outside it");
#endif
    if (blk->getParent() != nullptr) {
        // Detach without freeing contained ops/args; the context owns it now.
        blk->getParent()->getBlocks().remove(blk);
        cctx->rootBlocks.insert(blk);
    }
}

extern "C" void MLIR_InsertRegionBlockAfter(MLIR_Context *cctx, MLIR_RegionHandle region_h,
                                            MLIR_BlockHandle blk_h,
                                            MLIR_BlockHandle after_h) {
    if (region_h == MLIR_INVALID_HANDLE || blk_h == MLIR_INVALID_HANDLE) return;
    auto *region = F<mlir::Region>(region_h);
    auto *blk = F<mlir::Block>(blk_h);
    // Detach block from any current region first.
    if (blk->getParent() != nullptr)
        blk->getParent()->getBlocks().remove(blk);
    cctx->rootBlocks.erase(blk);
    if (after_h == MLIR_INVALID_HANDLE) {
        region->getBlocks().push_front(blk);
        return;
    }
    auto *after = F<mlir::Block>(after_h);
    // Insert right after `after`.
    auto it = std::next(after->getIterator());
    region->getBlocks().insert(it, blk);
}

extern "C" void MLIR_InsertRegionBlockBefore(MLIR_Context *cctx, MLIR_RegionHandle region_h,
                                             MLIR_BlockHandle blk_h,
                                             MLIR_BlockHandle before_h) {
    if (region_h == MLIR_INVALID_HANDLE || blk_h == MLIR_INVALID_HANDLE) return;
    auto *region = F<mlir::Region>(region_h);
    auto *blk = F<mlir::Block>(blk_h);
    if (blk->getParent() != nullptr)
        blk->getParent()->getBlocks().remove(blk);
    cctx->rootBlocks.erase(blk);
    if (before_h == MLIR_INVALID_HANDLE) {
        region->push_back(blk);
        return;
    }
    auto *before = F<mlir::Block>(before_h);
    region->getBlocks().insert(before->getIterator(), blk);
}

extern "C" MLIR_OpHandle MLIR_GetBlockTerminator(MLIR_Context *ctx, MLIR_BlockHandle blk_h) {
    CHECK_OWN_BLOCK(ctx, blk_h, "MLIR_GetBlockTerminator");
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    if (blk->empty()) return MLIR_INVALID_HANDLE;
    return H(&blk->back());
}

extern "C" MLIR_OpHandle MLIR_GetBlockParentOp(MLIR_Context *ctx, MLIR_BlockHandle blk_h) {
    CHECK_OWN_BLOCK(ctx, blk_h, "MLIR_GetBlockParentOp");
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    auto *r = blk->getParent();
    if (!r) return MLIR_INVALID_HANDLE;
    auto *op = r->getParentOp();
    return op ? H(op) : MLIR_INVALID_HANDLE;
}

extern "C" bool MLIR_IsBlockEntry(MLIR_Context *ctx, MLIR_BlockHandle blk_h) {
    CHECK_OWN_BLOCK(ctx, blk_h, "MLIR_IsBlockEntry");
    if (blk_h == MLIR_INVALID_HANDLE) return false;
    auto *blk = F<mlir::Block>(blk_h);
    return blk->isEntryBlock();
}

extern "C" size_t MLIR_GetBlockNumPredecessors(MLIR_Context *ctx, MLIR_BlockHandle blk_h) {
    CHECK_OWN_BLOCK(ctx, blk_h, "MLIR_GetBlockNumPredecessors");
    if (blk_h == MLIR_INVALID_HANDLE) return 0;
    auto *blk = F<mlir::Block>(blk_h);
    size_t n = 0;
    for (auto it = blk->pred_begin(); it != blk->pred_end(); ++it) ++n;
    return n;
}

extern "C" MLIR_BlockHandle MLIR_GetBlockPredecessor(MLIR_Context *ctx, MLIR_BlockHandle blk_h,
                                                    size_t idx,
                                                    size_t *out_succ_idx) {
    CHECK_OWN_BLOCK(ctx, blk_h, "MLIR_GetBlockPredecessor");
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    size_t i = 0;
    for (auto it = blk->pred_begin(); it != blk->pred_end(); ++it, ++i) {
        if (i == idx) {
            if (out_succ_idx) *out_succ_idx = it.getSuccessorIndex();
            return H(*it);
        }
    }
    if (out_succ_idx) *out_succ_idx = SIZE_MAX;
    return MLIR_INVALID_HANDLE;
}

extern "C" MLIR_ValueHandle MLIR_AddBlockArgument(MLIR_Context *cctx,
                                                  MLIR_BlockHandle blk_h,
                                                  MLIR_TypeHandle type_h,
                                                  MLIR_LocationHandle loc_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    mlir::Type ty = typeF(type_h);
    mlir::Location loc = (loc_h == MLIR_INVALID_HANDLE)
        ? mlir::Location(mlir::UnknownLoc::get(cctx))
        : locF(loc_h);
    auto arg = blk->addArgument(ty, loc);
    return valH(arg);
}

extern "C" void MLIR_EraseBlockArguments(MLIR_Context *cctx,
                                          MLIR_BlockHandle blk_h,
                                          size_t start, size_t count) {
    if (blk_h == MLIR_INVALID_HANDLE || count == 0) return;
    auto *blk = F<mlir::Block>(blk_h);
    if (!cctx->valueNames.empty()) {
        for (size_t i = 0; i < count; i++) {
            unsigned argIdx = static_cast<unsigned>(start + i);
            if (argIdx >= blk->getNumArguments()) break;
            cctx->valueNames.erase(blk->getArgument(argIdx).getAsOpaquePointer());
        }
    }
    blk->eraseArguments(static_cast<unsigned>(start),
                        static_cast<unsigned>(count));
}

extern "C" MLIR_BlockHandle MLIR_GetValueParentBlock(MLIR_Context *ctx, MLIR_ValueHandle v_h) {
    CHECK_OWN_VALUE(ctx, v_h, "MLIR_GetValueParentBlock");
    if (v_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    mlir::Value v = valF(v_h);
    if (!v) return MLIR_INVALID_HANDLE;
    auto *blk = v.getParentBlock();
    return blk ? H(blk) : MLIR_INVALID_HANDLE;
}

extern "C" size_t MLIR_GetValueNumUses(MLIR_Context *cctx, MLIR_ValueHandle v_h) {
    (void)cctx;
    if (v_h == MLIR_INVALID_HANDLE) return 0;
    mlir::Value v = valF(v_h);
    if (!v) return 0;
    size_t n = 0;
    for (auto &u : v.getUses()) { (void)u; ++n; }
    return n;
}

extern "C" MLIR_OpHandle MLIR_GetValueUseOwner(MLIR_Context *cctx,
                                                MLIR_ValueHandle v_h,
                                                size_t idx,
                                                size_t *out_operand_idx) {
    (void)cctx;
    if (v_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    mlir::Value v = valF(v_h);
    if (!v) return MLIR_INVALID_HANDLE;
    size_t i = 0;
    for (auto &u : v.getUses()) {
        if (i == idx) {
            if (out_operand_idx) *out_operand_idx = u.getOperandNumber();
            return H(u.getOwner());
        }
        ++i;
    }
    if (out_operand_idx) *out_operand_idx = SIZE_MAX;
    return MLIR_INVALID_HANDLE;
}

extern "C" void MLIR_AppendOpSuccessorOperand(MLIR_Context *cctx, MLIR_OpHandle op_h,
                                               size_t succ_idx,
                                               MLIR_ValueHandle v_h) {
    (void)cctx;
    if (op_h == MLIR_INVALID_HANDLE || v_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(op);
    if (!branch) return;
    if (succ_idx >= op->getNumSuccessors()) return;
    mlir::Value v = valF(v_h);
    if (!v) return;
    mlir::SuccessorOperands sops =
        branch.getSuccessorOperands(static_cast<unsigned>(succ_idx));
    mlir::MutableOperandRange mut = sops.getMutableForwardedOperands();
    mut.append({v});
}

extern "C" void MLIR_SpliceBlockOps(MLIR_Context *cctx, MLIR_BlockHandle dst_h,
                                     MLIR_BlockHandle src_h) {
    (void)cctx;
    if (dst_h == MLIR_INVALID_HANDLE || src_h == MLIR_INVALID_HANDLE) return;
    auto *dst = F<mlir::Block>(dst_h);
    auto *src = F<mlir::Block>(src_h);
    dst->getOperations().splice(dst->end(), src->getOperations());
}

// -----------------------------------------------------------------------------
// Upstream-only lowering / translation entry points (the *Upstream
// siblings of the agnostic functions in mlir_lower_to_llvm.c,
// mlir_translate_to_llvm_ir.c, mlir_translate_to_wasm.c). Each runs
// upstream MLIR's pass pipeline / translator / LLVM target machine.
// -----------------------------------------------------------------------------

// Forward declaration: applies upstream's LiftControlFlowToSCF transform to
// `llvm.func` bodies (the stock pass only walks `func.func`). Implemented
// further down in this file.
static void liftLLVMFuncCFGToSCF(mlir::ModuleOp module);

// Rewrite tinyc's `llvm.intr.wasm.memory.size` / `llvm.intr.wasm.memory.grow`
// ops into upstream LLVM dialect's `llvm.call_intrinsic` form. Upstream has
// no such ops, so they are unregistered there; tinyc emits them (and
// drops the memory-index argument, since only the default linear memory is
// modeled); the native llvm->wasmssa pipeline recognizes the unregistered
// names directly, but upstream's LLVM-IR translator only knows the
// registered LLVM dialect ops. Rewriting to `llvm.call_intrinsic` lets
// `translateModuleToLLVMIR` emit the actual `llvm.wasm.memory.size` /
// `llvm.wasm.memory.grow` IR intrinsics.
static void rewriteWasmMemoryIntrinsicsForUpstream(mlir::ModuleOp module) {
    using namespace mlir;
    llvm::SmallVector<Operation *> to_erase;
    MLIR_Context *c = ctxOf(module.getOperation());
    const void *size_name = c->opNames[OP_TYPE_LLVM_INTR_WASM_MEMORY_SIZE];
    const void *grow_name = c->opNames[OP_TYPE_LLVM_INTR_WASM_MEMORY_GROW];
    module.walk([&](Operation *op) {
        const void *name = op->getName().getAsOpaquePointer();
        bool is_size = (name == size_name);
        bool is_grow = (name == grow_name);
        if (!is_size && !is_grow) return;
        if (op->getNumResults() != 1) return;
        OpBuilder b(op);
        Location loc = op->getLoc();
        Type i32 = b.getI32Type();
        // Both intrinsics take the memory index (always 0) as their first
        // argument; tinyc dropped it, so prepend a fresh `llvm.mlir.constant
        // 0 : i32` here.
        Value zero = b.create<LLVM::ConstantOp>(
            loc, i32, b.getI32IntegerAttr(0));
        llvm::SmallVector<Value> args;
        args.push_back(zero);
        for (Value v : op->getOperands()) args.push_back(v);
        llvm::StringRef intrin = is_size
            ? llvm::StringRef("llvm.wasm.memory.size")
            : llvm::StringRef("llvm.wasm.memory.grow");
        auto call = b.create<LLVM::CallIntrinsicOp>(
            loc, op->getResult(0).getType(), b.getStringAttr(intrin),
            args, LLVM::FastmathFlagsAttr{});
        op->getResult(0).replaceAllUsesWith(call.getResult(0));
        to_erase.push_back(op);
    });
    for (Operation *op : to_erase) {
        forgetErasedOp(c, op);
        op->erase();
    }
}

// After the LLVM-lowering pass pipeline runs, convert our MLIR-level
// `wasm.import_module` / `wasm.import_name` / `wasm.export_name`
// attributes on `llvm.func` ops into the upstream LLVM dialect's
// `passthrough` attribute. `passthrough` is the canonical mechanism for
// forwarding arbitrary LLVM IR function attribute strings through
// `translateModuleToLLVMIR`; LLVM's WebAssembly backend reads
// `wasm-import-module` / `wasm-import-name` / `wasm-export-name`
// function attributes to place imports in the correct WASI module
// (`wasi_snapshot_preview1`) and to honor explicit exports. Without
// this conversion, the upstream translator drops these on the floor
// and all imports fall back to module `env`. (Clang itself attaches
// these same strings when compiling `__attribute__((import_module(...)))`.)
static void liftWasmImportExportAttrsToPassthrough(mlir::ModuleOp module) {
    using namespace mlir;
    module.walk([&](LLVM::LLVMFuncOp fn) {
        struct KV { llvm::StringRef key; llvm::StringRef val; };
        llvm::SmallVector<KV, 3> kvs;
        if (auto a = fn->getAttrOfType<StringAttr>("wasm.import_module"))
            kvs.push_back({"wasm-import-module", a.getValue()});
        if (auto a = fn->getAttrOfType<StringAttr>("wasm.import_name"))
            kvs.push_back({"wasm-import-name", a.getValue()});
        if (auto a = fn->getAttrOfType<StringAttr>("wasm.export_name"))
            kvs.push_back({"wasm-export-name", a.getValue()});
        if (kvs.empty()) return;
        auto &mctx = *module.getContext();
        llvm::SmallVector<Attribute, 4> elements;
        // Preserve any existing passthrough entries (the pass pipeline
        // does not currently add any, but be defensive).
        if (auto existing = fn.getPassthroughAttr()) {
            for (Attribute e : existing) elements.push_back(e);
        }
        for (const KV &kv : kvs) {
            Attribute pair_elts[2] = {
                StringAttr::get(&mctx, kv.key),
                StringAttr::get(&mctx, kv.val),
            };
            elements.push_back(ArrayAttr::get(&mctx, pair_elts));
        }
        fn.setPassthroughAttr(ArrayAttr::get(&mctx, elements));
    });
}

extern "C" bool MLIR_LowerToLLVMDialectUpstream(MLIR_Context *ctx,
                                                MLIR_OpHandle module_h) {
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_LowerToLLVMDialectUpstream: handle is not a ModuleOp");
        return false;
    }
    MLIR_Context &mctx = *ctx;
    // Only run vector-to-LLVM if the module actually contains a vector op.
    // The dialect-conversion infrastructure used by that pass performs
    // identical-block merging that miscompiles deeply-nested if/else
    // cascades over scalar code (LLVM 19 bug). Skipping the pass when
    // there is no vector op to lower avoids the miscompile entirely.
    bool has_vector_op = false;
    module.walk([&](mlir::Operation *o) {
        if (o->getDialect() && o->getDialect()->getNamespace() == "vector") {
            has_vector_op = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    // Drop unreachable blocks before lowering. The conversion driver used
    // by createConvertVectorToLLVMPass performs this DCE as a side effect;
    // when we skip that pass, convert-cf-to-llvm leaves unreachable
    // `cf.br`s behind which then fail LLVM-IR translation.
    {
        mlir::IRRewriter rewriter(&mctx);
        (void)mlir::eraseUnreachableBlocks(rewriter, module->getRegions());
    }
    // Lift any CFG (cf.cond_br/cf.br) inside structured ops (scf.if/scf.for/
    // scf.while) up to nested SCF form first. tinyc's `&&` / `||` lowering
    // builds an `scf.if` whose then/else regions evaluate the rhs, and if
    // that rhs itself emits CFG (e.g. a nested ternary), the result is a
    // multi-block `scf.if` region — which violates `scf.if`'s SizedRegion<1>
    // invariant and trips `createConvertSCFToCFPass` later in the pipeline.
    // Running the lift first rewrites the multi-block region into a nested
    // single-block `scf.if`, restoring `scf.if`'s invariant.
    liftLLVMFuncCFGToSCF(module);
    // Rewrite tinyc's unregistered wasm memory intrinsics (used by the
    // upstream wasm flow, which goes through this same upstream-llvm
    // pipeline before handing off to LLVM's wasm target). Cheap walk on
    // non-wasm builds since no such ops are present.
    rewriteWasmMemoryIntrinsicsForUpstream(module);
    mlir::PassManager pm(&mctx);
    pm.addPass(mlir::createLiftControlFlowToSCFPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    if (has_vector_op) {
        pm.addPass(mlir::createConvertVectorToLLVMPass());
    }
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    // The lift-to-SCF pass above can introduce `ub.poison` values for
    // SSA edges it cannot prove a concrete value for; convert them to
    // `llvm.poison` so LLVM-IR translation accepts the module.
    pm.addPass(mlir::createUBToLLVMConversionPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(module))) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_LowerToLLVMDialectUpstream: pass pipeline failed");
        return false;
    }
    liftWasmImportExportAttrsToPassthrough(module);
    return true;
}

// For the native wasm backend, MLIR's `--lift-cf-to-scf` only walks
// `func.func` ops, so any `llvm.func` body containing cf.br/cf.cond_br is
// left unstructured and our wasmssa lowerer cannot handle it. tinyc emits
// variadic functions (e.g. `int sum(int n, ...)`) as `llvm.func` because
// `func.func` has no var-arg representation, which is exactly the case that
// trips the wasm pipeline.
//
// Run the same CFG-to-SCF lifting that the stock pass applies to func.func,
// but extend it to llvm.func by subclassing the transformation so it can
// also emit an `llvm.return` (with poison operands) when an unreachable
// terminator is needed.
namespace {
class CFGToSCFForWasm : public mlir::ControlFlowToSCFTransformation {
public:
    // Override switch-flag generation to use i32 instead of `index`. The
    // wasm backend has no native `index` type and rejects
    // arith.index_cast(ui), which the stock implementation introduces to
    // bridge the index discriminator with cf.switch's i32-typed cases.
    // Using i32 throughout side-steps the cast entirely.
    mlir::Value getCFGSwitchValue(mlir::Location loc, mlir::OpBuilder &builder,
                                  unsigned value) override {
        return builder.create<mlir::arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(static_cast<int32_t>(value)));
    }

    void createCFGSwitchOp(mlir::Location loc, mlir::OpBuilder &builder,
                           mlir::Value flag,
                           llvm::ArrayRef<unsigned> caseValues,
                           mlir::BlockRange caseDestinations,
                           llvm::ArrayRef<mlir::ValueRange> caseArguments,
                           mlir::Block *defaultDest,
                           mlir::ValueRange defaultArgs) override {
        builder.create<mlir::cf::SwitchOp>(
            loc, flag, defaultDest, defaultArgs,
            llvm::to_vector_of<int32_t>(caseValues),
            caseDestinations, caseArguments);
    }

    mlir::FailureOr<mlir::Operation *>
    createUnreachableTerminator(mlir::Location loc, mlir::OpBuilder &builder,
                                mlir::Region &region) override {
        mlir::Operation *parentOp = region.getParentOp();
        if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(parentOp)) {
            return mlir::ControlFlowToSCFTransformation::
                createUnreachableTerminator(loc, builder, region);
        }
        if (auto llvmFn = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(parentOp)) {
            auto fnTy = llvmFn.getFunctionType();
            llvm::SmallVector<mlir::Value> operands;
            mlir::Type retTy = fnTy.getReturnType();
            if (!llvm::isa<mlir::LLVM::LLVMVoidType>(retTy))
                operands.push_back(getUndefValue(loc, builder, retTy));
            return builder.create<mlir::LLVM::ReturnOp>(loc, operands)
                .getOperation();
        }
        return mlir::emitError(loc, "Cannot create unreachable terminator for '")
               << parentOp->getName() << "'";
    }
};
}

static void liftLLVMFuncCFGToSCF(mlir::ModuleOp module) {
    using namespace mlir;
    CFGToSCFForWasm transformation;
    // Walk both func.func and llvm.func bodies so the wasm-flavored
    // overrides (i32 switch values, llvm.return for unreachable) are
    // applied uniformly. The stock LiftControlFlowToSCF pass only
    // visits func.func, and would otherwise lift those bodies with
    // index-typed switch values that the wasm backend rejects.
    llvm::SmallVector<Operation *> fns;
    module.walk([&](Operation *fn) {
        if (auto fOp = llvm::dyn_cast<func::FuncOp>(fn)) {
            if (!fOp.getBody().empty()) fns.push_back(fn);
        } else if (auto lOp = llvm::dyn_cast<LLVM::LLVMFuncOp>(fn)) {
            if (!lOp.getBody().empty()) fns.push_back(fn);
        }
    });
    for (auto *fn : fns) {
        DominanceInfo domInfo(fn);
        // Post-order so nested regions are lifted first, matching the stock
        // LiftControlFlowToSCF pass.
        fn->walk<WalkOrder::PostOrder>([&](Operation *innerOp) {
            for (Region &reg : innerOp->getRegions())
                (void)transformCFGToSCF(reg, transformation, domInfo);
        });
    }
}

// Agnostic-API: lift cf -> scf in `module`. Used by the wasm pipeline
// regardless of which lowering binary is doing the work. Implemented by
// reusing upstream MLIR's `transformCFGToSCF` algorithm, applied to both
// func.func bodies (via the stock `LiftControlFlowToSCFPass`) and
// llvm.func bodies (via `liftLLVMFuncCFGToSCF`).
extern "C" bool MLIR_LiftCfToScfNative(MLIR_Context *ctx,
                                       MLIR_OpHandle module,
                                       const MLIR_LiftCfToScfOptions *options);

extern "C" bool MLIR_LiftCfToScf(MLIR_Context *ctx, MLIR_OpHandle module_h,
                                 const MLIR_LiftCfToScfOptions *options) {
    if (module_h == MLIR_INVALID_HANDLE) return false;
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_LiftCfToScf: handle is not a ModuleOp");
        return false;
    }
    MLIR_Context &mctx = *ctx;
    {
        mlir::IRRewriter rewriter(&mctx);
        (void)mlir::eraseUnreachableBlocks(rewriter, module->getRegions());
    }
    // Route the native (agnostic C) lift port for both API impls. The two
    // impls used to diverge here: native ran `MLIR_LiftCfToScfNative` while
    // upstream ran `liftLLVMFuncCFGToSCF` + `createLiftControlFlowToSCFPass`.
    // Both lifts are individually valid, but they produce structurally
    // different SCF (different scf.if result threading, different number
    // of trivially-dead consts). When the native wasm pipeline
    // (`mlir_llvm_to_wasmssa.c`) consumes the upstream-lifted IR, the
    // resulting wasm has different carrier counts and local layouts;
    // accumulated across all of tinyc's source files this produces a
    // self-hosted `tinyc.wasm` that miscompiles its own input
    // (e.g. `int main(){return 42;}` lowers to `func.return(NULL_OPERAND)`).
    // The agnostic lifter only relies on MLIR_GetOpRegion / MLIR_GetBlockOp
    // / MLIR_CreateOp etc, all of which dispatch through the upstream API
    // impl correctly. Running the same lifter on both sides keeps the wasm
    // output bit-identical between the `native` and `upstream_api` configs
    // for the simple cases we test, and functionally correct for all of
    // tinyc's self-host.
    return MLIR_LiftCfToScfNative(ctx, module_h, options);
}

// Host codegen (native llc / x86 ELF) cannot lower wasm memory intrinsics.
// Expand them to the same @__wasm_mem_pages load/store model used by the
// native mlir_translate_to_llvm_ir.c path.
static void expandWasmMemoryIntrinsicsForHost(llvm::Module &M) {
    llvm::LLVMContext &C = M.getContext();
    llvm::Type *I32 = llvm::Type::getInt32Ty(C);
    llvm::Type *I64 = llvm::Type::getInt64Ty(C);
    llvm::GlobalVariable *pages = nullptr;
    auto getPages = [&]() -> llvm::GlobalVariable * {
        if (pages) return pages;
        if (auto *existing = M.getGlobalVariable("__wasm_mem_pages"))
            return pages = existing;
        pages = new llvm::GlobalVariable(
            M, I32, false, llvm::GlobalValue::InternalLinkage,
            llvm::ConstantInt::get(I32, 0), "__wasm_mem_pages");
        pages->setAlignment(llvm::Align(4));
        return pages;
    };

    llvm::SmallVector<llvm::Instruction *, 8> erase;
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        for (llvm::BasicBlock &BB : F) {
            for (llvm::Instruction &I : BB) {
                auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
                if (!CI) continue;
                llvm::Function *Callee = CI->getCalledFunction();
                if (!Callee) continue;
                llvm::StringRef name = Callee->getName();
                bool is_size = name.starts_with("llvm.wasm.memory.size");
                bool is_grow = name.starts_with("llvm.wasm.memory.grow");
                if (!is_size && !is_grow) continue;

                llvm::IRBuilder<> B(CI);
                llvm::Value *ptr = getPages();
                if (is_size) {
                    llvm::Value *load = B.CreateLoad(I32, ptr);
                    CI->replaceAllUsesWith(load);
                    erase.push_back(CI);
                    continue;
                }
                llvm::Value *delta = CI->getArgOperand(CI->arg_size() - 1);
                llvm::Value *old = B.CreateLoad(I32, ptr);
                llvm::Value *newv = B.CreateAdd(old, delta);
                llvm::Value *new64 = B.CreateZExt(newv, I64);
                llvm::Value *ok = B.CreateICmpULE(
                    new64, llvm::ConstantInt::get(I64, 65536));
                llvm::Value *stored = B.CreateSelect(ok, newv, old);
                B.CreateStore(stored, ptr);
                llvm::Value *res = B.CreateSelect(
                    ok, old, llvm::ConstantInt::getSigned(I32, -1));
                CI->replaceAllUsesWith(res);
                erase.push_back(CI);
            }
        }
    }
    for (llvm::Instruction *I : erase) I->eraseFromParent();
}

extern "C" string MLIR_TranslateModuleToLLVMIRUpstream(MLIR_Context *ctx,
                                                       MLIR_OpHandle module_h) {
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_TranslateModuleToLLVMIRUpstream: not a ModuleOp");
        return mkRefString(llvm::StringRef());
    }
    // Register the LLVM-IR translation interfaces (idempotent).
    mlir::registerBuiltinDialectTranslation(*ctx);
    mlir::registerLLVMDialectTranslation(*ctx);

    llvm::LLVMContext llctx;
    auto llmod = mlir::translateModuleToLLVMIR(module, llctx);
    if (!llmod) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_TranslateModuleToLLVMIRUpstream: translation failed");
        return mkRefString(llvm::StringRef());
    }
    expandWasmMemoryIntrinsicsForHost(*llmod);
    std::string out;
    llvm::raw_string_ostream os(out);
    llmod->print(os, nullptr);
    os.flush();
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char *buf = (char*)arena_alloc(arena, out.size());
    std::memcpy(buf, out.data(), out.size());
    string result;
    result.str = buf;
    result.size = out.size();
    return result;
}

// Translate a `builtin.module` op already lowered to the LLVM dialect
// into a wasm32-wasi relocatable object file using LLVM's WebAssembly
// target. The returned bytes are NOT a runnable wasm module — they
// still need to be linked (typically with `wasm-ld`) against any
// runtime/imports the program uses.
extern "C" string MLIR_TranslateModuleToWasmUpstream(MLIR_Context *ctx,
                                                     MLIR_OpHandle module_h) {
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_TranslateModuleToWasmUpstream: handle is not a ModuleOp");
        return mkRefString(llvm::StringRef());
    }

    llvm::LLVMContext llctx;
    std::unique_ptr<llvm::Module> llmod;
    {
        mlir::registerBuiltinDialectTranslation(*ctx);
        mlir::registerLLVMDialectTranslation(*ctx);
        llmod = mlir::translateModuleToLLVMIR(module, llctx);
        if (!llmod) {
            MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                         "MLIR_TranslateModuleToWasmUpstream: translation to LLVM IR failed");
            return mkRefString(llvm::StringRef());
        }
    }

    // Initialize the WebAssembly target. Idempotent: LLVM guards these
    // with internal flags so repeated calls are cheap.
    LLVMInitializeWebAssemblyTargetInfo();
    LLVMInitializeWebAssemblyTarget();
    LLVMInitializeWebAssemblyTargetMC();
    LLVMInitializeWebAssemblyAsmParser();
    LLVMInitializeWebAssemblyAsmPrinter();

    const std::string triple = "wasm32-wasi";
    llmod->setTargetTriple(triple);

    std::string err;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, err);
    if (!target) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_TranslateModuleToWasmUpstream: lookupTarget failed: %s",
                     err.c_str());
        return mkRefString(llvm::StringRef());
    }

    llvm::TargetOptions opts;
    auto rm = std::optional<llvm::Reloc::Model>();
    auto cm = std::optional<llvm::CodeModel::Model>();
    std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
        triple, /*CPU*/ "generic", /*Features*/ "", opts, rm, cm));
    if (!tm) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_TranslateModuleToWasmUpstream: createTargetMachine failed");
        return mkRefString(llvm::StringRef());
    }
    llmod->setDataLayout(tm->createDataLayout());

    // Emit a wasm32 relocatable object file (ELF-like wasm object) into
    // an in-memory buffer. Final linking with wasm-ld is the caller's
    // responsibility.
    llvm::SmallVector<char, 0> obj_bytes;
    llvm::raw_svector_ostream obj_os(obj_bytes);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, obj_os, /*DwoOut*/ nullptr,
                                llvm::CodeGenFileType::ObjectFile)) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_INVALID_HANDLE,
                     "MLIR_TranslateModuleToWasmUpstream: target cannot emit object file");
        return mkRefString(llvm::StringRef());
    }
    pm.run(*llmod);

    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char *buf = (char *)arena_alloc(arena, obj_bytes.size());
    std::memcpy(buf, obj_bytes.data(), obj_bytes.size());
    string result;
    result.str = buf;
    result.size = obj_bytes.size();
    return result;
}

// -----------------------------------------------------------------------------
// Pass and pipeline infrastructure (#355): CPassAdapter on mlir::PassManager
// -----------------------------------------------------------------------------

namespace {

struct CPassShared {
    MLIR_Context *cctx;
    MLIR_Pass pass;
    const MLIR_TargetInfo *target;
    Arena *scratch;
    arena_pos_t scratchBase;
    bool timing;
    uint64_t elapsedNs = 0;
};

struct CPassAdapter
    : public mlir::PassWrapper<CPassAdapter, mlir::OperationPass<>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CPassAdapter)

    std::shared_ptr<CPassShared> shared;
    std::string passName;

    explicit CPassAdapter(std::shared_ptr<CPassShared> s)
        : shared(std::move(s)),
          passName(shared->pass.name.str, shared->pass.name.size) {}

    llvm::StringRef getArgument() const override { return passName; }
    llvm::StringRef getDescription() const override { return passName; }
    std::unique_ptr<mlir::Pass> clonePass() const override {
        return std::make_unique<CPassAdapter>(shared);
    }

    void runOnOperation() override {
        mlir::Operation *op = getOperation();
        if (shared->pass.anchor == MLIR_ANCHOR_FUNCTION) {
            if (op->getNumRegions() == 0 || op->getRegion(0).empty()) return;
        }
        MLIR_PassContext pc;
        pc.ctx = shared->cctx;
        pc.target = shared->target;
        pc.scratch = shared->scratch;
        uint64_t t0 = shared->timing ? platform_clock_ns() : 0;
        bool ok = shared->pass.run(&pc, H(op), shared->pass.state);
        if (shared->timing) shared->elapsedNs += (platform_clock_ns() - t0);
        arena_reset(shared->scratch, shared->scratchBase);
        if (!ok) signalPassFailure();
    }
};

static bool upstream_should_print_after(string print_after, string name) {
    if (print_after.size == 0) return false;
    return str_eq(print_after, str_lit("*")) || str_eq(print_after, name);
}

} // namespace

extern "C" bool MLIR_RunPasses(MLIR_Context *ctx, MLIR_OpHandle module_h,
                               const MLIR_Pass *passes, size_t n,
                               const MLIR_PipelineOptions *opts) {
    API_CHECK(ctx != nullptr, "MLIR_RunPasses", "context is NULL");
    API_CHECK(module_h != MLIR_INVALID_HANDLE, "MLIR_RunPasses", "module handle is NULL");
    if (n == 0) return true;

    const MLIR_TargetInfo *target = opts ? opts->target : nullptr;
    bool verify_each = opts ? opts->verify_each : false;
    bool timing = opts ? opts->timing : false;
    string print_after = opts ? opts->print_after : str_lit("");

    Arena *scratch = arena_create(256 * 1024);
    arena_pos_t scratch_base = arena_get_pos(scratch);
    auto *modOp = F<mlir::Operation>(module_h);

    // C passes share `state` and `scratch` without locking, so run
    // single-threaded.
    bool was_mt = ctx->isMultithreadingEnabled();
    if (was_mt) ctx->disableMultithreading();

    bool all_ok = true;
    for (size_t i = 0; i < n; i++) {
        auto shared = std::make_shared<CPassShared>();
        shared->cctx = ctx;
        shared->pass = passes[i];
        shared->target = target;
        shared->scratch = scratch;
        shared->scratchBase = scratch_base;
        shared->timing = timing;

        mlir::PassManager pm(ctx);
        pm.enableVerifier(verify_each);
        if (passes[i].anchor == MLIR_ANCHOR_MODULE) {
            pm.addPass(std::make_unique<CPassAdapter>(shared));
        } else {
            pm.nest<mlir::LLVM::LLVMFuncOp>().addPass(std::make_unique<CPassAdapter>(shared));
            pm.nest<mlir::func::FuncOp>().addPass(std::make_unique<CPassAdapter>(shared));
        }

        if (mlir::failed(pm.run(modOp))) {
            all_ok = false;
            break;
        }
        if (timing) {
            MLIR_EmitDiagnostic(ctx, MLIR_DIAG_REMARK, MLIR_INVALID_HANDLE,
                                "pass %.*s: %.3f ms",
                                (int)passes[i].name.size, passes[i].name.str,
                                (double)shared->elapsedNs * 1e-6);
        }
        if (upstream_should_print_after(print_after, passes[i].name)) {
            string ir = print_operation_generic(ctx, 0, module_h);
            MLIR_EmitDiagnostic(ctx, MLIR_DIAG_REMARK, MLIR_INVALID_HANDLE,
                                "// -----// IR Dump After %.*s //----- //\n%.*s",
                                (int)passes[i].name.size, passes[i].name.str,
                                (int)ir.size, ir.str);
        }
    }

    if (was_mt) ctx->enableMultithreading();
    arena_destroy(scratch);
    return all_ok;
}
