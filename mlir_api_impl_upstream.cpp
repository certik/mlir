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
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/AsmParser/AsmParser.h"
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

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

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
    UpstreamCtx() {
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
        mctx.appendDialectRegistry(registry);
        mctx.loadAllAvailableDialects();
        mctx.allowUnregisteredDialects(true);
    }
};
UpstreamCtx &globalCtx() { static UpstreamCtx g; return g; }

// One ValueBox per logical mlir::Value. Provides a stable handle and
// (for block args) carries the register name that upstream doesn't
// track natively.
struct ValueBox {
    mlir::Value v;                  // bound after AppendBlockArg / Op create
    std::string name;               // register name; empty for op results
    mlir::Type pendingType;         // used until AppendBlockArg binds v
    mlir::Location pending_loc{mlir::UnknownLoc::get(&globalCtx().mctx)};
    bool deferredBlockArg = false;
};

std::unordered_map<const void *, ValueBox *> &valueIndex() {
    static std::unordered_map<const void *, ValueBox *> m;
    return m;
}

// Side-map: per-Operation snapshot of the named attributes the user
// supplied via MLIR_CreateOp / MLIR_AppendOpAttribute. Used by
// MLIR_GetOpNumAttributes / MLIR_GetOpAttribute so that dialect-injected
// inherent attributes (e.g. arith.addi's default
// `overflowFlags = #arith.overflow<none>` property) don't leak through
// the API and cause spurious diffs against the dialect-agnostic native
// backend.
std::unordered_map<const mlir::Operation *, std::vector<mlir::NamedAttribute>> &
opUserAttrs() {
    static std::unordered_map<const mlir::Operation *,
                              std::vector<mlir::NamedAttribute>> m;
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
uintptr_t locH(mlir::Location loc) { return reinterpret_cast<uintptr_t>(loc.getAsOpaquePointer()); }
mlir::Location locF(uintptr_t h) {
    return mlir::Location::getFromOpaquePointer(reinterpret_cast<const void *>(h));
}

MLIR_OpType opTypeFromName(llvm::StringRef name) {
    if (name == "builtin.module") return OP_TYPE_MODULE;
    // Walk the canonical name table backwards. OP_TYPE_UNREGISTERED is the
    // fallback.
    for (int t = 0; t < 256; t++) {
        string s = op_type_to_string((MLIR_OpType)t);
        if (s.size == 0) continue;
        if (name.size() == s.size && std::memcmp(name.data(), s.str, s.size) == 0)
            return (MLIR_OpType)t;
    }
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
    auto realArg = block->addArgument(box->pendingType, box->pending_loc);
    // If we have a stand-in arg from a hidden holder block, redirect uses.
    if (box->v) {
        auto oldOpaque = box->v.getAsOpaquePointer();
        valueIndex().erase(oldOpaque);
        box->v.replaceAllUsesWith(realArg);
    }
    box->v = realArg;
    box->deferredBlockArg = false;
    valueIndex()[realArg.getAsOpaquePointer()] = box;
}

// -----------------------------------------------------------------------------
// Op construction
// -----------------------------------------------------------------------------

extern "C" MLIR_OpHandle MLIR_CreateOpWithSuccessors(
    MLIR_Context *, MLIR_OpType type, string opname,
    MLIR_AttributeHandle *attrs, size_t n_attrs,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_BlockHandle *successors, size_t n_successors,
    MLIR_ValueHandle **successor_operands,
    size_t *n_successor_operands,
    MLIR_LocationHandle location, MLIR_LocationHandle, string, int64_t) {
    auto &ctx = globalCtx().mctx;

    std::string nm(opname.str, opname.size);
    if (nm.empty()) {
        string s = op_type_to_string(type);
        if (s.size > 0) nm.assign(s.str, s.size);
        else nm = "unknown";
    }
    if (type == OP_TYPE_MODULE && nm == "module") {
        nm = "builtin.module";
    }

    mlir::Location loc = (location == MLIR_INVALID_HANDLE)
                             ? mlir::Location(mlir::UnknownLoc::get(&ctx))
                             : locF(location);
    mlir::OperationState state(loc, llvm::StringRef(nm));

    for (size_t i = 0; i < n_operands; i++) {
        state.addOperands(F<ValueBox>(operands[i])->v);
    }
    // For branch-like ops, successor operands are appended after regular
    // operands; the BranchOpInterface uses operandSegmentSizes to slice them.
    for (size_t s = 0; s < n_successors; s++) {
        size_t n = n_successor_operands ? n_successor_operands[s] : 0;
        for (size_t i = 0; i < n; i++) {
            state.addOperands(F<ValueBox>(successor_operands[s][i])->v);
        }
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
    for (size_t i = 0; i < n_successors; i++) {
        state.addSuccessors(F<mlir::Block>(successors[i]));
    }

    // Snapshot user attrs BEFORE we synthesize operandSegmentSizes — the
    // generic printer iterates user attrs and we don't want that internal
    // attribute leaking into output.
    std::vector<mlir::NamedAttribute> userAttrs(state.attributes.begin(),
                                                state.attributes.end());

    if (n_successors > 0 && n_successor_operands) {
        if (nm == "cf.cond_br") {
            llvm::SmallVector<int32_t, 4> seg;
            seg.push_back((int32_t)n_operands);
            for (size_t s = 0; s < n_successors; s++) {
                seg.push_back((int32_t)n_successor_operands[s]);
            }
            state.addAttribute("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(&ctx, seg));
        }
    }

    mlir::Operation *op = mlir::Operation::create(state);
    opUserAttrs()[op] = std::move(userAttrs);

    for (size_t i = 0; i < n_results; i++) {
        auto v = op->getResult(i);
        auto *box = F<ValueBox>(results[i]);
        box->v = v;
        valueIndex()[v.getAsOpaquePointer()] = box;
    }

    return H(op);
}

extern "C" MLIR_OpHandle MLIR_CreateOp(
    MLIR_Context *ctx, MLIR_OpType type, string opname,
    MLIR_AttributeHandle *attrs, size_t n_attrs,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_LocationHandle location, MLIR_LocationHandle u, string tc, int64_t sl) {
    return MLIR_CreateOpWithSuccessors(
        ctx, type, opname, attrs, n_attrs,
        result_types, n_result_types,
        results, n_results,
        operands, n_operands,
        regions, n_regions,
        nullptr, 0, nullptr, nullptr,
        location, u, tc, sl);
}

extern "C" void MLIR_AppendOpAttribute(MLIR_Context *, MLIR_OpHandle oh,
                                        MLIR_AttributeHandle ah) {
    auto *op = F<mlir::Operation>(oh);
    const auto *na = F<mlir::NamedAttribute>(ah);
    op->setAttr(na->getName(), na->getValue());
    auto &m = opUserAttrs();
    auto it = m.find(op);
    if (it != m.end()) {
        // Replace if name already present, else append.
        for (auto &existing : it->second) {
            if (existing.getName() == na->getName()) {
                existing = *na;
                return;
            }
        }
        it->second.push_back(*na);
    }
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
    auto *op = F<mlir::Operation>(h);
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
    auto *op = F<mlir::Operation>(h);
    auto &m = opUserAttrs();
    auto it = m.find(op);
    if (it != m.end()) return it->second.size();
    return op->getAttrs().size();
}
extern "C" MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle h, size_t i) {
    auto *op = F<mlir::Operation>(h);
    auto &m = opUserAttrs();
    auto it = m.find(op);
    if (it != m.end()) return reinterpret_cast<uintptr_t>(&it->second[i]);
    auto attrs = op->getAttrs();
    return reinterpret_cast<uintptr_t>(&attrs[i]);
}
extern "C" size_t MLIR_GetOpNumRegions(MLIR_OpHandle h) {
    return F<mlir::Operation>(h)->getNumRegions();
}
extern "C" MLIR_RegionHandle MLIR_GetOpRegion(MLIR_OpHandle h, size_t i) {
    return H(&F<mlir::Operation>(h)->getRegion(i));
}

extern "C" size_t MLIR_GetOpNumSuccessors(MLIR_OpHandle h) {
    return F<mlir::Operation>(h)->getNumSuccessors();
}
extern "C" MLIR_BlockHandle MLIR_GetOpSuccessor(MLIR_OpHandle h, size_t i) {
    return H(F<mlir::Operation>(h)->getSuccessor(i));
}
extern "C" size_t MLIR_GetOpNumSuccessorOperands(MLIR_OpHandle h, size_t s) {
    auto *op = F<mlir::Operation>(h);
    if (auto br = mlir::dyn_cast<mlir::BranchOpInterface>(op)) {
        return br.getSuccessorOperands(s).size();
    }
    return 0;
}
extern "C" MLIR_ValueHandle MLIR_GetOpSuccessorOperand(MLIR_OpHandle h, size_t s, size_t i) {
    auto *op = F<mlir::Operation>(h);
    if (auto br = mlir::dyn_cast<mlir::BranchOpInterface>(op)) {
        auto ops = br.getSuccessorOperands(s);
        return H(boxFor(ops[i]));
    }
    return MLIR_INVALID_HANDLE;
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

extern "C" size_t MLIR_GetBlockIndex(MLIR_BlockHandle h) {
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
    MLIR_Context *, string register_name, uint32_t /*idx*/,
    MLIR_TypeHandle type, MLIR_LocationHandle loc) {
    auto *box = new ValueBox();
    box->name.assign(register_name.str, register_name.size);
    box->pendingType = typeF(type);
    if (loc != MLIR_INVALID_HANDLE) box->pending_loc = locF(loc);
    box->deferredBlockArg = true;
    // Eagerly bind to a hidden holder block so the value can be used as an
    // operand before AppendBlockArg moves it onto the real block. Upstream
    // MLIR doesn't allow moving a BlockArgument between blocks, so on
    // AppendBlockArg we add a fresh arg to the real block and RAUW.
    static thread_local mlir::Block *holder = nullptr;
    if (!holder) holder = new mlir::Block();
    auto arg = holder->addArgument(box->pendingType, box->pending_loc);
    box->v = arg;
    valueIndex()[arg.getAsOpaquePointer()] = box;
    return H(box);
}
extern "C" MLIR_ValueHandle MLIR_CreateValueOpResult(
    MLIR_Context *, MLIR_OpHandle oh, uint32_t idx, MLIR_TypeHandle,
    string register_name, MLIR_LocationHandle) {
    auto *box = new ValueBox();
    box->name.assign(register_name.str, register_name.size);
    if (oh != MLIR_INVALID_HANDLE) {
        auto *op = F<mlir::Operation>(oh);
        if (idx < op->getNumResults()) {
            auto v = op->getResult(idx);
            box->v = v;
            valueIndex()[v.getAsOpaquePointer()] = box;
        }
    }
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
extern "C" MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *, uint32_t width, bool is_bfloat) {
    auto &ctx = globalCtx().mctx;
    if (is_bfloat) return typeH(mlir::BFloat16Type::get(&ctx));
    switch (width) {
        case 16: return typeH(mlir::Float16Type::get(&ctx));
        case 32: return typeH(mlir::Float32Type::get(&ctx));
        case 64: return typeH(mlir::Float64Type::get(&ctx));
        default: return typeH(mlir::Float32Type::get(&ctx));
    }
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *) {
    return typeH(mlir::IndexType::get(&globalCtx().mctx));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *) {
    return typeH(mlir::NoneType::get(&globalCtx().mctx));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *, const int64_t *shape,
                                                  size_t rank, MLIR_TypeHandle element_type) {
    auto &ctx = globalCtx().mctx;
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
extern "C" MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *, const int64_t *shape,
                                                  size_t rank, MLIR_TypeHandle element_type) {
    auto &ctx = globalCtx().mctx;
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
    auto &mctx = globalCtx().mctx;
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
extern "C" MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *, string name) {
    auto &ctx = globalCtx().mctx;
    return typeH(mlir::OpaqueType::get(mlir::StringAttr::get(&ctx, "?"),
                                        llvm::StringRef(name.str, name.size)));
}
extern "C" MLIR_TypeHandle MLIR_CreateTypeFunction(MLIR_Context *,
                                                    const MLIR_TypeHandle *inputs, size_t n_inputs,
                                                    const MLIR_TypeHandle *results, size_t n_results) {
    auto &ctx = globalCtx().mctx;
    llvm::SmallVector<mlir::Type, 4> in;
    in.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; i++) in.push_back(typeF(inputs[i]));
    llvm::SmallVector<mlir::Type, 4> out;
    out.reserve(n_results);
    for (size_t i = 0; i < n_results; i++) out.push_back(typeF(results[i]));
    return typeH(mlir::FunctionType::get(&ctx, in, out));
}
extern "C" void MLIR_SetTypeIntegerProperties(MLIR_TypeHandle, uint32_t, bool) {}
extern "C" void MLIR_SetTypeFloatProperties(MLIR_TypeHandle, uint32_t, bool) {}
extern "C" void MLIR_SetTypeTensorProperties(MLIR_TypeHandle, const int64_t *, size_t, MLIR_TypeHandle) {}
extern "C" void MLIR_SetTypeMemrefProperties(MLIR_TypeHandle, const int64_t *, size_t, MLIR_TypeHandle) {}
extern "C" void MLIR_SetTypePointerProperties(MLIR_TypeHandle, MLIR_TypeHandle, bool, uint32_t) {}

extern "C" bool MLIR_IsTypeInteger(MLIR_TypeHandle h) { return llvm::isa<mlir::IntegerType>(typeF(h)); }
extern "C" bool MLIR_IsTypeFloat(MLIR_TypeHandle h)   { return llvm::isa<mlir::FloatType>(typeF(h)); }
extern "C" bool MLIR_IsTypeTensor(MLIR_TypeHandle h)  { return llvm::isa<mlir::TensorType>(typeF(h)); }
extern "C" bool MLIR_IsTypeMemref(MLIR_TypeHandle h)  { return llvm::isa<mlir::BaseMemRefType>(typeF(h)); }
extern "C" bool MLIR_IsTypePointer(MLIR_TypeHandle h) {
    auto opaq = llvm::dyn_cast<mlir::OpaqueType>(typeF(h));
    return opaq && opaq.getDialectNamespace() == "tt";
}
extern "C" bool MLIR_IsTypeIndex(MLIR_TypeHandle h)   { return llvm::isa<mlir::IndexType>(typeF(h)); }
extern "C" bool MLIR_IsTypeUnknown(MLIR_TypeHandle h) { return llvm::isa<mlir::NoneType>(typeF(h)); }
extern "C" bool MLIR_IsTypeOpaque(MLIR_TypeHandle h)  {
    auto opaq = llvm::dyn_cast<mlir::OpaqueType>(typeF(h));
    return opaq && opaq.getDialectNamespace() != "tt";
}
extern "C" bool MLIR_IsTypeFunction(MLIR_TypeHandle h) { return llvm::isa<mlir::FunctionType>(typeF(h)); }
extern "C" size_t MLIR_GetTypeFunctionNumInputs(MLIR_TypeHandle h) {
    auto ft = llvm::dyn_cast<mlir::FunctionType>(typeF(h));
    return ft ? ft.getNumInputs() : 0;
}
extern "C" MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_TypeHandle h, size_t idx) {
    auto ft = llvm::dyn_cast<mlir::FunctionType>(typeF(h));
    if (!ft || idx >= ft.getNumInputs()) return MLIR_INVALID_HANDLE;
    return typeH(ft.getInput(idx));
}
extern "C" size_t MLIR_GetTypeFunctionNumResults(MLIR_TypeHandle h) {
    auto ft = llvm::dyn_cast<mlir::FunctionType>(typeF(h));
    return ft ? ft.getNumResults() : 0;
}
extern "C" MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_TypeHandle h, size_t idx) {
    auto ft = llvm::dyn_cast<mlir::FunctionType>(typeF(h));
    if (!ft || idx >= ft.getNumResults()) return MLIR_INVALID_HANDLE;
    return typeH(ft.getResult(idx));
}

extern "C" MLIR_TypeHandle MLIR_GetTypeShapedElement(MLIR_TypeHandle h) {
    auto t = typeF(h);
    if (auto st = llvm::dyn_cast<mlir::ShapedType>(t)) {
        return typeH(st.getElementType());
    }
    return MLIR_INVALID_HANDLE;
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
    std::string buf;
    llvm::raw_string_ostream os(buf);
    t.print(os);
    os.flush();
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
    (void)ctx;
    auto &mctx = globalCtx().mctx;
    llvm::StringRef src(text.str, text.size);
    mlir::ParserConfig config(&mctx);
    mlir::OwningOpRef<mlir::ModuleOp> mod =
        mlir::parseSourceString<mlir::ModuleOp>(src, config);
    if (!mod) return MLIR_INVALID_HANDLE;
    // Release ownership; the operation lives in the MLIRContext arena.
    mlir::Operation *op = mod.release().getOperation();
    return reinterpret_cast<uintptr_t>(op);
}

// -----------------------------------------------------------------------------
// Attribute — handle is a heap mlir::NamedAttribute*
// -----------------------------------------------------------------------------

static MLIR_AttributeHandle makeNamedAttr(llvm::StringRef name, mlir::Attribute value) {
    auto &ctx = globalCtx().mctx;
    auto *na = new mlir::NamedAttribute(mlir::StringAttr::get(&ctx, name), value);
    return reinterpret_cast<uintptr_t>(na);
}

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *, string name, int64_t value, MLIR_TypeHandle type) {
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::IntegerAttr::get(typeF(type), value));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *, string name, double value, MLIR_TypeHandle type) {
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::FloatAttr::get(typeF(type), value));
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
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *, string name,
                                                           MLIR_AttributeHandle *elements,
                                                           size_t count) {
    auto &ctx = globalCtx().mctx;
    llvm::SmallVector<mlir::Attribute, 8> values;
    values.reserve(count);
    for (size_t i = 0; i < count; i++) {
        values.push_back(F<mlir::NamedAttribute>(elements[i])->getValue());
    }
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::ArrayAttr::get(&ctx, values));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *, string name,
                                                          MLIR_AttributeHandle *elements,
                                                          size_t count) {
    auto &ctx = globalCtx().mctx;
    llvm::SmallVector<mlir::NamedAttribute, 8> entries;
    entries.reserve(count);
    for (size_t i = 0; i < count; i++) {
        entries.push_back(*F<mlir::NamedAttribute>(elements[i]));
    }
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::DictionaryAttr::get(&ctx, entries));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeType(MLIR_Context *, string name, MLIR_TypeHandle type) {
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::TypeAttr::get(typeF(type)));
}
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeSymbolRef(MLIR_Context *, string name, string value) {
    auto &ctx = globalCtx().mctx;
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::FlatSymbolRefAttr::get(&ctx,
                             llvm::StringRef(value.str, value.size)));
}

extern "C" MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle h) {
    auto value = F<mlir::NamedAttribute>(h)->getValue();
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
extern "C" string MLIR_GetAttributeName(MLIR_AttributeHandle h) {
    return mkRefString(F<mlir::NamedAttribute>(h)->getName().getValue());
}
extern "C" int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::IntegerAttr>(F<mlir::NamedAttribute>(h)->getValue()).getInt();
}
extern "C" double MLIR_GetAttributeFloat(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::FloatAttr>(F<mlir::NamedAttribute>(h)->getValue()).getValueAsDouble();
}
extern "C" MLIR_TypeHandle MLIR_GetAttributeType(MLIR_AttributeHandle h) {
    auto attr = F<mlir::NamedAttribute>(h)->getValue();
    if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(attr)) return typeH(ia.getType());
    if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(attr))   return typeH(fa.getType());
    return MLIR_INVALID_HANDLE;
}
extern "C" MLIR_TypeHandle MLIR_GetAttributeTypeValue(MLIR_AttributeHandle h) {
    auto attr = F<mlir::NamedAttribute>(h)->getValue();
    if (auto ta = llvm::dyn_cast<mlir::TypeAttr>(attr)) return typeH(ta.getValue());
    return MLIR_INVALID_HANDLE;
}
extern "C" bool MLIR_GetAttributeBool(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::BoolAttr>(F<mlir::NamedAttribute>(h)->getValue()).getValue();
}
extern "C" string MLIR_GetAttributeString(MLIR_AttributeHandle h) {
    auto value = F<mlir::NamedAttribute>(h)->getValue();
    if (auto s = llvm::dyn_cast<mlir::StringAttr>(value)) return mkRefString(s.getValue());
    if (auto sr = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(value)) {
        // Return `@name`. Interned via a thread-local map so the storage
        // outlives this call without needing an external arena.
        static thread_local std::unordered_map<std::string, std::string> cache;
        auto rootRef = sr.getValue();
        std::string key(rootRef.data(), rootRef.size());
        auto it = cache.find(key);
        if (it == cache.end()) {
            it = cache.emplace(key, std::string("@") + key).first;
        }
        const std::string &v = it->second;
        string out; out.str = const_cast<char *>(v.data()); out.size = v.size();
        return out;
    }
    return mkRefString(llvm::cast<mlir::StringAttr>(value).getValue());
}
extern "C" string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle h) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    F<mlir::NamedAttribute>(h)->getValue().print(os);
    os.flush();
    return mkArenaString(ctx, buf);
}
extern "C" size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::ArrayAttr>(F<mlir::NamedAttribute>(h)->getValue()).size();
}
extern "C" MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle h, size_t i) {
    auto arr = llvm::cast<mlir::ArrayAttr>(F<mlir::NamedAttribute>(h)->getValue());
    auto &ctx = globalCtx().mctx;
    auto *na = new mlir::NamedAttribute(mlir::StringAttr::get(&ctx, ""), arr[i]);
    return reinterpret_cast<uintptr_t>(na);
}
extern "C" size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle h) {
    return llvm::cast<mlir::DictionaryAttr>(F<mlir::NamedAttribute>(h)->getValue()).size();
}
extern "C" MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle h, size_t i) {
    auto dict = llvm::cast<mlir::DictionaryAttr>(F<mlir::NamedAttribute>(h)->getValue());
    auto entries = dict.getValue();
    return reinterpret_cast<uintptr_t>(&entries[i]);
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

extern "C" MLIR_LocationHandle MLIR_CreateLocationUnknown(MLIR_Context *, string /*orig*/) {
    return locH(mlir::UnknownLoc::get(&globalCtx().mctx));
}
extern "C" MLIR_LocationHandle MLIR_CreateLocationFile(MLIR_Context *, string filename,
                                                        int line, int column) {
    auto &ctx = globalCtx().mctx;
    return locH(mlir::FileLineColLoc::get(
        &ctx, llvm::StringRef(filename.str, filename.size),
        (unsigned)line, (unsigned)column));
}
extern "C" MLIR_LocationHandle MLIR_CreateLocationName(MLIR_Context *, string name) {
    auto &ctx = globalCtx().mctx;
    return locH(mlir::NameLoc::get(
        mlir::StringAttr::get(&ctx, llvm::StringRef(name.str, name.size))));
}
extern "C" MLIR_LocationHandle MLIR_CreateLocationRef(MLIR_Context *, int /*ref_id*/) {
    return locH(mlir::UnknownLoc::get(&globalCtx().mctx));
}

extern "C" MLIR_LocationKind MLIR_GetLocationKind(MLIR_LocationHandle h) {
    if (h == MLIR_INVALID_HANDLE) return MLIR_LOC_UNKNOWN;
    auto loc = locF(h);
    if (llvm::isa<mlir::FileLineColLoc>(loc)) return MLIR_LOC_FILE;
    if (llvm::isa<mlir::NameLoc>(loc))        return MLIR_LOC_NAME;
    if (llvm::isa<mlir::CallSiteLoc>(loc))    return MLIR_LOC_CALLSITE;
    if (llvm::isa<mlir::FusedLoc>(loc))       return MLIR_LOC_FUSED;
    return MLIR_LOC_UNKNOWN;
}
extern "C" string MLIR_GetLocationOriginalText(MLIR_LocationHandle) {
    return mkRefString(llvm::StringRef());
}
extern "C" string MLIR_GetLocationFileFilename(MLIR_LocationHandle h) {
    if (auto fl = llvm::dyn_cast<mlir::FileLineColLoc>(locF(h)))
        return mkRefString(fl.getFilename().getValue());
    return mkRefString(llvm::StringRef());
}
extern "C" int MLIR_GetLocationFileLine(MLIR_LocationHandle h) {
    if (auto fl = llvm::dyn_cast<mlir::FileLineColLoc>(locF(h)))
        return (int)fl.getLine();
    return 0;
}
extern "C" int MLIR_GetLocationFileColumn(MLIR_LocationHandle h) {
    if (auto fl = llvm::dyn_cast<mlir::FileLineColLoc>(locF(h)))
        return (int)fl.getColumn();
    return 0;
}
extern "C" string MLIR_GetLocationName(MLIR_LocationHandle h) {
    if (auto nl = llvm::dyn_cast<mlir::NameLoc>(locF(h)))
        return mkRefString(nl.getName().getValue());
    return mkRefString(llvm::StringRef());
}
extern "C" int MLIR_GetLocationRefId(MLIR_LocationHandle) { return 0; }

extern "C" MLIR_LocationHandle MLIR_GetValueLocation(MLIR_ValueHandle h) {
    if (h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto v = F<ValueBox>(h)->v;
    if (!v) return MLIR_INVALID_HANDLE;
    return locH(v.getLoc());
}

// -----------------------------------------------------------------------------
// Misc
// -----------------------------------------------------------------------------

extern "C" string MLIR_MLIR_OpTypeToString(MLIR_OpType type) {
    return op_type_to_string(type);
}

// -----------------------------------------------------------------------------
// Lowering to LLVM dialect + translation to LLVM IR text
// -----------------------------------------------------------------------------

extern "C" bool MLIR_LowerToLLVMDialect(MLIR_Context *, MLIR_OpHandle module_h) {
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        std::fprintf(stderr, "MLIR_LowerToLLVMDialect: handle is not a ModuleOp\n");
        return false;
    }
    auto &ctx = globalCtx().mctx;
    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertVectorToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(module))) {
        std::fprintf(stderr, "MLIR_LowerToLLVMDialect: pass pipeline failed\n");
        return false;
    }
    return true;
}

extern "C" string MLIR_TranslateModuleToLLVMIR(MLIR_Context *ctx, MLIR_OpHandle module_h) {
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        std::fprintf(stderr, "MLIR_TranslateModuleToLLVMIR: not a ModuleOp\n");
        return mkRefString(llvm::StringRef());
    }
    // Register the LLVM-IR translation interfaces (idempotent).
    mlir::registerBuiltinDialectTranslation(globalCtx().mctx);
    mlir::registerLLVMDialectTranslation(globalCtx().mctx);

    llvm::LLVMContext llctx;
    auto llmod = mlir::translateModuleToLLVMIR(module, llctx);
    if (!llmod) {
        std::fprintf(stderr, "MLIR_TranslateModuleToLLVMIR: translation failed\n");
        return mkRefString(llvm::StringRef());
    }
    std::string out;
    llvm::raw_string_ostream os(out);
    llmod->print(os, nullptr);
    os.flush();
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char *buf = (char*)arena_alloc(arena, out.size());
    std::memcpy(buf, out.data(), out.size());
    return (string){.str = buf, .size = out.size()};
}
