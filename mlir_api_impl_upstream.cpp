// Upstream LLVM/MLIR-backed implementation of the public API in mlir_api.h.
//
// All structural queries (getOperand/getResult/getAttrs/getRegions/...)
// go directly through the upstream MLIR C++ API. The only auxiliary
// state we keep is a small map from `mlir::Value` to a heap-allocated
// `ValueBox`, which exists because:
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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "llvm/Support/raw_ostream.h"

extern "C" {
#include "mlir_api.h"
#include "mlir_op_names.h"
}

#define UNIMPLEMENTED()                                                   \
    do {                                                                  \
        std::fprintf(stderr, "%s unimplemented (upstream)\n", __func__);  \
        std::abort();                                                     \
    } while (0)

namespace {

struct UpstreamCtx {
    mlir::MLIRContext mctx;
    UpstreamCtx() { mctx.allowUnregisteredDialects(true); }
};
UpstreamCtx &globalCtx() { static UpstreamCtx g; return g; }

// One ValueBox per logical mlir::Value. Provides a stable handle and
// (for block args) carries the register name that upstream doesn't
// track natively.
struct ValueBox {
    mlir::Value v;                  // bound after AppendBlockArg / Op create
    std::string name;               // register name; empty for op results
    mlir::Type pendingType;         // used until AppendBlockArg binds v
    bool deferredBlockArg = false;
};

std::unordered_map<const void *, ValueBox *> &valueIndex() {
    static std::unordered_map<const void *, ValueBox *> m;
    return m;
}

ValueBox *boxFor(mlir::Value v) {
    auto &m = valueIndex();
    auto it = m.find(v.getAsOpaquePointer());
    if (it != m.end()) return it->second;
    auto *box = new ValueBox();
    box->v = v;
    m[v.getAsOpaquePointer()] = box;
    return box;
}

template <class T> uintptr_t H(T *p) { return reinterpret_cast<uintptr_t>(p); }
template <class T> T *F(uintptr_t h) { return reinterpret_cast<T *>(h); }

uintptr_t typeH(mlir::Type t) { return reinterpret_cast<uintptr_t>(t.getAsOpaquePointer()); }
mlir::Type typeF(uintptr_t h) {
    return mlir::Type::getFromOpaquePointer(reinterpret_cast<const void *>(h));
}

const char *opNameForType(MLIR_OpType t) {
    switch (t) {
        case OP_TYPE_MODULE:        return "builtin.module";
        case OP_TYPE_ARITH_ADDI:    return "arith.addi";
        case OP_TYPE_FUNC_FUNC:     return "func.func";
        case OP_TYPE_FUNC_RETURN:   return "func.return";
        default:                    return nullptr;
    }
}

MLIR_OpType opTypeFromName(llvm::StringRef name) {
    if (name == "builtin.module") return OP_TYPE_MODULE;
    if (name == "arith.addi")     return OP_TYPE_ARITH_ADDI;
    if (name == "func.func")      return OP_TYPE_FUNC_FUNC;
    if (name == "func.return")    return OP_TYPE_FUNC_RETURN;
    return OP_TYPE_UNREGISTERED;
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

extern "C" void MLIR_InitApi(MLIR_Context *, MLIR_OpHandle) {}
extern "C" void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena) { ctx->arena = arena; }
extern "C" Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx) { return ctx->arena; }

// -----------------------------------------------------------------------------
// Region / Block
// -----------------------------------------------------------------------------

extern "C" MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *) {
    return H(new mlir::Region());
}
extern "C" MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *) {
    return H(new mlir::Block());
}
extern "C" void MLIR_AppendRegionBlock(MLIR_Context *, MLIR_RegionHandle r,
                                        MLIR_BlockHandle b) {
    F<mlir::Region>(r)->push_back(F<mlir::Block>(b));
}
extern "C" void MLIR_AppendBlockOp(MLIR_Context *, MLIR_BlockHandle b,
                                    MLIR_OpHandle op) {
    F<mlir::Block>(b)->push_back(F<mlir::Operation>(op));
}
extern "C" void MLIR_AppendBlockArg(MLIR_Context *, MLIR_BlockHandle bh,
                                     MLIR_ValueHandle vh) {
    auto *block = F<mlir::Block>(bh);
    auto *box = F<ValueBox>(vh);
    if (!box->deferredBlockArg) {
        std::fprintf(stderr, "AppendBlockArg: handle is not a deferred block arg\n");
        std::abort();
    }
    auto arg = block->addArgument(box->pendingType,
                                   mlir::UnknownLoc::get(&globalCtx().mctx));
    box->v = arg;
    box->deferredBlockArg = false;
    valueIndex()[arg.getAsOpaquePointer()] = box;
}

// -----------------------------------------------------------------------------
// Op construction
// -----------------------------------------------------------------------------

extern "C" MLIR_OpHandle MLIR_CreateOp(
    MLIR_Context *, MLIR_OpType type, string opname,
    MLIR_AttributeHandle *attrs, size_t n_attrs,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_LocationHandle, MLIR_LocationHandle, string, int64_t) {
    auto &ctx = globalCtx().mctx;

    std::string nm(opname.str, opname.size);
    if (nm.empty()) {
        const char *fallback = opNameForType(type);
        nm = fallback ? fallback : "unknown";
    } else if (type == OP_TYPE_MODULE && nm == "module") {
        nm = "builtin.module";
    }

    mlir::OperationState state(mlir::UnknownLoc::get(&ctx),
                               llvm::StringRef(nm));

    for (size_t i = 0; i < n_operands; i++) {
        state.addOperands(F<ValueBox>(operands[i])->v);
    }
    for (size_t i = 0; i < n_result_types; i++) {
        state.addTypes(typeF(result_types[i]));
    }
    for (size_t i = 0; i < n_attrs; i++) {
        const auto *na = F<mlir::NamedAttribute>(attrs[i]);
        state.addAttribute(na->getName(), na->getValue());
    }
    for (size_t i = 0; i < n_regions; i++) {
        state.addRegion(std::unique_ptr<mlir::Region>(F<mlir::Region>(regions[i])));
    }
    mlir::Operation *op = mlir::Operation::create(state);

    // Bind user-supplied result handles to the freshly-created OpResults.
    for (size_t i = 0; i < n_results; i++) {
        auto v = op->getResult(i);
        auto *box = F<ValueBox>(results[i]);
        box->v = v;
        valueIndex()[v.getAsOpaquePointer()] = box;
    }

    return H(op);
}

extern "C" void MLIR_AppendOpAttribute(MLIR_Context *, MLIR_OpHandle oh,
                                        MLIR_AttributeHandle ah) {
    auto *op = F<mlir::Operation>(oh);
    const auto *na = F<mlir::NamedAttribute>(ah);
    op->setAttr(na->getName(), na->getValue());
}

// -----------------------------------------------------------------------------
// Op accessors — straight passthrough to upstream
// -----------------------------------------------------------------------------

extern "C" MLIR_OpType MLIR_GetOpType(MLIR_OpHandle h) {
    return opTypeFromName(F<mlir::Operation>(h)->getName().getStringRef());
}
extern "C" string MLIR_GetOpName(MLIR_OpHandle h) {
    auto sr = F<mlir::Operation>(h)->getName().getStringRef();
    // The C side prints OP_TYPE_MODULE as "module"; strip "builtin." prefix
    // for the canonical builtin module op so output matches.
    if (sr == "builtin.module") return mkRefString("module");
    return mkRefString(sr);
}
extern "C" string MLIR_GetOpName_string(MLIR_OpHandle h) { return MLIR_GetOpName(h); }
extern "C" MLIR_LocationHandle MLIR_GetOpLocation(MLIR_OpHandle) { return MLIR_INVALID_HANDLE; }
extern "C" string MLIR_GetOpTrailingComment(MLIR_OpHandle) {
    string s; s.str = nullptr; s.size = 0; return s;
}
extern "C" int64_t MLIR_GetOpSourceLineStart(MLIR_OpHandle) { return -1; }
extern "C" MLIR_LocationHandle MLIR_GetOpUnnumberedLocationDef(MLIR_OpHandle) {
    return MLIR_INVALID_HANDLE;
}

extern "C" size_t MLIR_GetOpNumOperands(MLIR_OpHandle h) {
    return F<mlir::Operation>(h)->getNumOperands();
}
extern "C" MLIR_ValueHandle MLIR_GetOpOperand(MLIR_OpHandle h, size_t i) {
    return H(boxFor(F<mlir::Operation>(h)->getOperand(i)));
}
extern "C" size_t MLIR_GetOpNumResults(MLIR_OpHandle h) {
    return F<mlir::Operation>(h)->getNumResults();
}
extern "C" MLIR_ValueHandle MLIR_GetOpResult(MLIR_OpHandle h, size_t i) {
    return H(boxFor(F<mlir::Operation>(h)->getResult(i)));
}
extern "C" size_t MLIR_GetOpNumResultTypes(MLIR_OpHandle h) {
    return F<mlir::Operation>(h)->getNumResults();
}
extern "C" MLIR_TypeHandle MLIR_GetOpResult_type(MLIR_OpHandle h, size_t i) {
    return typeH(F<mlir::Operation>(h)->getResult(i).getType());
}
extern "C" size_t MLIR_GetOpNumAttributes(MLIR_OpHandle h) {
    return F<mlir::Operation>(h)->getAttrs().size();
}
extern "C" MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle h, size_t i) {
    auto attrs = F<mlir::Operation>(h)->getAttrs();
    return reinterpret_cast<uintptr_t>(&attrs[i]);
}
extern "C" size_t MLIR_GetOpNumRegions(MLIR_OpHandle h) {
    return F<mlir::Operation>(h)->getNumRegions();
}
extern "C" MLIR_RegionHandle MLIR_GetOpRegion(MLIR_OpHandle h, size_t i) {
    return H(&F<mlir::Operation>(h)->getRegion(i));
}

// -----------------------------------------------------------------------------
// Region / Block accessors
// -----------------------------------------------------------------------------

extern "C" size_t MLIR_GetRegionNumBlocks(MLIR_RegionHandle h) {
    auto *r = F<mlir::Region>(h);
    return std::distance(r->begin(), r->end());
}
extern "C" MLIR_BlockHandle MLIR_GetRegionBlock(MLIR_RegionHandle h, size_t i) {
    auto *r = F<mlir::Region>(h);
    auto it = r->begin();
    std::advance(it, i);
    return H(&*it);
}

extern "C" size_t MLIR_GetBlockNumOps(MLIR_BlockHandle h) {
    auto *b = F<mlir::Block>(h);
    return std::distance(b->begin(), b->end());
}
extern "C" MLIR_OpHandle MLIR_GetBlockOp(MLIR_BlockHandle h, size_t i) {
    auto *b = F<mlir::Block>(h);
    auto it = b->begin();
    std::advance(it, i);
    return H(&*it);
}
extern "C" size_t MLIR_GetBlockNumArgs(MLIR_BlockHandle h) {
    return F<mlir::Block>(h)->getNumArguments();
}
extern "C" MLIR_ValueHandle MLIR_GetBlockArg(MLIR_BlockHandle h, size_t i) {
    return H(boxFor(F<mlir::Block>(h)->getArgument(i)));
}

// -----------------------------------------------------------------------------
// Value
// -----------------------------------------------------------------------------

extern "C" MLIR_ValueHandle MLIR_CreateValueBlockArg(
    MLIR_Context *, string register_name, uint32_t /*idx*/,
    MLIR_TypeHandle type, MLIR_LocationHandle) {
    auto *box = new ValueBox();
    box->name.assign(register_name.str, register_name.size);
    box->pendingType = typeF(type);
    box->deferredBlockArg = true;
    return H(box);
}
extern "C" MLIR_ValueHandle MLIR_CreateValueOpResult(
    MLIR_Context *, MLIR_OpHandle, uint32_t /*idx*/, MLIR_TypeHandle,
    string register_name, MLIR_LocationHandle) {
    auto *box = new ValueBox();
    box->name.assign(register_name.str, register_name.size);
    return H(box);
}

extern "C" MLIR_ValueKind MLIR_GetValueKind(MLIR_ValueHandle h) {
    auto v = F<ValueBox>(h)->v;
    return llvm::isa<mlir::BlockArgument>(v) ? BLOCK_ARG : OP_RESULT;
}
extern "C" MLIR_TypeHandle MLIR_GetValueType(MLIR_ValueHandle h) {
    return typeH(F<ValueBox>(h)->v.getType());
}
extern "C" string MLIR_GetValueRegisterName(MLIR_ValueHandle h) {
    return mkRefString(F<ValueBox>(h)->name);
}
extern "C" uint32_t MLIR_GetValueResultIndex(MLIR_ValueHandle h) {
    auto v = F<ValueBox>(h)->v;
    if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(v)) return ba.getArgNumber();
    if (auto opr = llvm::dyn_cast<mlir::OpResult>(v)) return opr.getResultNumber();
    return 0;
}
extern "C" MLIR_OpHandle MLIR_GetValueDefiningOp(MLIR_ValueHandle h) {
    auto v = F<ValueBox>(h)->v;
    if (auto opr = llvm::dyn_cast<mlir::OpResult>(v)) return H(opr.getOwner());
    return MLIR_INVALID_HANDLE;
}

// -----------------------------------------------------------------------------
// Type
// -----------------------------------------------------------------------------

extern "C" MLIR_TypeHandle MLIR_CreateTypeInteger(MLIR_Context *, uint32_t width,
                                                   bool /*is_signed*/) {
    return typeH(mlir::IntegerType::get(&globalCtx().mctx, width));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *, uint32_t, bool) { UNIMPLEMENTED(); }
extern "C" MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *) {
    return typeH(mlir::IndexType::get(&globalCtx().mctx));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *) { UNIMPLEMENTED(); }
extern "C" MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *, const int64_t *, size_t, MLIR_TypeHandle) { UNIMPLEMENTED(); }
extern "C" MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *, const int64_t *, size_t, MLIR_TypeHandle) { UNIMPLEMENTED(); }
extern "C" MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *, MLIR_TypeHandle, bool, uint32_t) { UNIMPLEMENTED(); }
extern "C" MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *, string) { UNIMPLEMENTED(); }
extern "C" void MLIR_SetTypeIntegerProperties(MLIR_TypeHandle, uint32_t, bool) { UNIMPLEMENTED(); }
extern "C" void MLIR_SetTypeFloatProperties(MLIR_TypeHandle, uint32_t, bool) { UNIMPLEMENTED(); }
extern "C" void MLIR_SetTypeTensorProperties(MLIR_TypeHandle, const int64_t *, size_t, MLIR_TypeHandle) { UNIMPLEMENTED(); }
extern "C" void MLIR_SetTypeMemrefProperties(MLIR_TypeHandle, const int64_t *, size_t, MLIR_TypeHandle) { UNIMPLEMENTED(); }
extern "C" void MLIR_SetTypePointerProperties(MLIR_TypeHandle, MLIR_TypeHandle, bool, uint32_t) { UNIMPLEMENTED(); }

extern "C" bool MLIR_IsTypeInteger(MLIR_TypeHandle h) { return llvm::isa<mlir::IntegerType>(typeF(h)); }
extern "C" bool MLIR_IsTypeFloat(MLIR_TypeHandle h)   { return llvm::isa<mlir::FloatType>(typeF(h)); }
extern "C" bool MLIR_IsTypeTensor(MLIR_TypeHandle)    { return false; }
extern "C" bool MLIR_IsTypeMemref(MLIR_TypeHandle)    { return false; }
extern "C" bool MLIR_IsTypePointer(MLIR_TypeHandle)   { return false; }
extern "C" bool MLIR_IsTypeIndex(MLIR_TypeHandle h)   { return llvm::isa<mlir::IndexType>(typeF(h)); }
extern "C" bool MLIR_IsTypeUnknown(MLIR_TypeHandle)   { return false; }
extern "C" bool MLIR_IsTypeOpaque(MLIR_TypeHandle)    { return false; }

extern "C" string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle h) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    typeF(h).print(os);
    os.flush();
    return mkArenaString(ctx, buf);
}

// -----------------------------------------------------------------------------
// Attribute — handle is a heap mlir::NamedAttribute*
// -----------------------------------------------------------------------------

static MLIR_AttributeHandle makeNamedAttr(llvm::StringRef name, mlir::Attribute value) {
    auto &ctx = globalCtx().mctx;
    auto *na = new mlir::NamedAttribute(mlir::StringAttr::get(&ctx, name), value);
    return reinterpret_cast<uintptr_t>(na);
}

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *, string name, int64_t value) {
    auto &ctx = globalCtx().mctx;
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), value));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *, string name, double value) {
    auto &ctx = globalCtx().mctx;
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::FloatAttr::get(mlir::Float64Type::get(&ctx), value));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *, string name, bool value) {
    auto &ctx = globalCtx().mctx;
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::BoolAttr::get(&ctx, value));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *, string name, string value) {
    auto &ctx = globalCtx().mctx;
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::StringAttr::get(&ctx, llvm::StringRef(value.str, value.size)));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *, string,
                                                           MLIR_AttributeHandle *, size_t) { UNIMPLEMENTED(); }
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *, string,
                                                          MLIR_AttributeHandle *, size_t) { UNIMPLEMENTED(); }

extern "C" MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle h) {
    auto value = F<mlir::NamedAttribute>(h)->getValue();
    if (llvm::isa<mlir::StringAttr>(value))  return MLIR_ATTR_KIND_STRING;
    if (llvm::isa<mlir::BoolAttr>(value))    return MLIR_ATTR_KIND_BOOL;
    if (llvm::isa<mlir::IntegerAttr>(value)) return MLIR_ATTR_KIND_INTEGER;
    if (llvm::isa<mlir::FloatAttr>(value))   return MLIR_ATTR_KIND_FLOAT;
    if (llvm::isa<mlir::ArrayAttr>(value))   return MLIR_ATTR_KIND_ARRAY;
    if (llvm::isa<mlir::DictionaryAttr>(value)) return MLIR_ATTR_KIND_DICT;
    return MLIR_ATTR_KIND_STRING;
}
extern "C" string MLIR_GetAttributeName(MLIR_AttributeHandle h) {
    return mkRefString(F<mlir::NamedAttribute>(h)->getName().getValue());
}
extern "C" int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::IntegerAttr>(F<mlir::NamedAttribute>(h)->getValue()).getInt();
}
extern "C" double MLIR_GetAttributeFloat(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::FloatAttr>(F<mlir::NamedAttribute>(h)->getValue()).getValueAsDouble();
}
extern "C" bool MLIR_GetAttributeBool(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::BoolAttr>(F<mlir::NamedAttribute>(h)->getValue()).getValue();
}
extern "C" string MLIR_GetAttributeString(MLIR_AttributeHandle h) {
    return mkRefString(llvm::cast<mlir::StringAttr>(F<mlir::NamedAttribute>(h)->getValue()).getValue());
}
extern "C" size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::ArrayAttr>(F<mlir::NamedAttribute>(h)->getValue()).size();
}
extern "C" MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle, size_t) { UNIMPLEMENTED(); }
extern "C" size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::DictionaryAttr>(F<mlir::NamedAttribute>(h)->getValue()).size();
}
extern "C" MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle, size_t) { UNIMPLEMENTED(); }

// -----------------------------------------------------------------------------
// Misc
// -----------------------------------------------------------------------------

extern "C" string MLIR_MLIR_OpTypeToString(MLIR_OpType type) {
    return op_type_to_string(type);
}

extern "C" size_t MLIR_GetLocationMapSize(const MLIR_LocationMap *) { return 0; }
extern "C" size_t MLIR_CollectLocationMap(const MLIR_LocationMap *, string *,
                                           MLIR_LocationHandle *, size_t) {
    return 0;
}

extern "C" int app_main(void);
extern "C" void platform_init(int argc, char **argv);
int main(int /*argc*/, char ** /*argv*/) {
    platform_init(0, nullptr);
    return app_main();
}
