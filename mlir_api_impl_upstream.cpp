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
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/Host.h"

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

// Separate stash for MLIR_GetOpAttributeByName, which sometimes needs to
// materialise a NamedAttribute that doesn't live in `op->getAttrs()` (e.g.
// LLVM::CallOp's inherent `callee` / `var_callee_type` properties). Using
// unique_ptr keeps element addresses stable across subsequent push_backs
// so previously-returned MLIR_AttributeHandle pointers stay valid.
//
// Keyed by op so it can be cleared alongside opUserAttrs() when ops are
// erased by a conversion pass, avoiding the unbounded process-lifetime
// growth a single thread_local stash would suffer in a long-running host.
std::unordered_map<const mlir::Operation *,
                   std::vector<std::unique_ptr<mlir::NamedAttribute>>> &
opByNameAttrs() {
    static std::unordered_map<const mlir::Operation *,
                              std::vector<std::unique_ptr<mlir::NamedAttribute>>> m;
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
    // Lazily-populated reverse map. Built once on first call from the
    // canonical name table (the linear scan we used to do here was hot
    // in profiles — ~5% of total time during wasm lowering).
    static llvm::DenseMap<llvm::StringRef, MLIR_OpType> *map = nullptr;
    if (!map) {
        map = new llvm::DenseMap<llvm::StringRef, MLIR_OpType>();
        map->reserve(256);
        for (int t = 0; t < 256; t++) {
            string s = op_type_to_string((MLIR_OpType)t);
            if (s.size == 0) continue;
            // The canonical table entries are static string literals, so
            // we can safely take a StringRef without copying.
            (*map)[llvm::StringRef(s.str, s.size)] = (MLIR_OpType)t;
        }
        // Special case kept for parity with the old linear-scan path.
        (*map)[llvm::StringRef("builtin.module")] = OP_TYPE_MODULE;
    }
    auto it = map->find(name);
    if (it != map->end()) return it->second;
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

// Small LRU cache for MLIR_GetBlockOp / MLIR_GetBlockNumOps. mlir::Block
// stores its ops in an intrusive linked list, so `std::advance(begin(), i)`
// is O(i) and a natural `for (i=0; i<n; i++) GetBlockOp(b, i)` loop is
// O(n^2). The cache lets sequential and locally-bounded accesses
// (forward by one, backward by one, small jumps near the last position)
// run in amortized O(1) by walking `Operation::getNextNode()` from the
// last position instead of restarting at `begin()`.
//
// Multi-slot, not single-slot: the wasm lowering walks the IR recursively
// (walk_block -> walk_op -> walk_block -> ...) AND interleaves iteration
// over one block with mutations of another block (read from `mb`, append
// to `out_body`). A single slot — or coarse-grained "nuke everything"
// invalidation — would be clobbered on every mutation. NUM_SLOTS is sized
// to comfortably exceed any nesting depth we hit in practice.
//
// Invalidation is per-block: each API entry point that mutates the op list
// of block B (append, insert, erase, splice) clears only B's slot. Erase
// also clears any slot whose cached `op` pointer matches the erased op
// (so it never dangles past the erase). Upstream passes that run on the
// IR (MLIR_LowerToLLVMDialect*, MLIR_LiftCfToScf) call `resetAll()` to
// nuke everything, because they can arbitrarily rewrite the IR.
struct BlockOpIterSlot {
    mlir::Block *block = nullptr;
    size_t idx = 0;
    mlir::Operation *op = nullptr;
    size_t cached_size = SIZE_MAX;  // SIZE_MAX = unknown; valid otherwise
};
struct BlockOpIterCache {
    static constexpr size_t NUM_SLOTS = 16;
    BlockOpIterSlot slots[NUM_SLOTS];
    size_t next_evict = 0;  // round-robin replacement.

    BlockOpIterSlot *find(mlir::Block *b) {
        for (size_t i = 0; i < NUM_SLOTS; i++) {
            if (slots[i].block == b && slots[i].op != nullptr) return &slots[i];
        }
        return nullptr;
    }
    BlockOpIterSlot *findOrEvict(mlir::Block *b) {
        if (auto *s = find(b)) return s;
        for (size_t i = 0; i < NUM_SLOTS; i++) {
            if (slots[i].block == nullptr) {
                slots[i] = {};
                return &slots[i];
            }
        }
        auto *s = &slots[next_evict];
        next_evict = (next_evict + 1) % NUM_SLOTS;
        *s = {};
        return s;
    }
    void invalidateBlock(mlir::Block *b) {
        if (!b) return;
        for (size_t i = 0; i < NUM_SLOTS; i++) {
            if (slots[i].block == b) slots[i] = {};
        }
    }
    void invalidateOp(mlir::Operation *op) {
        if (!op) return;
        for (size_t i = 0; i < NUM_SLOTS; i++) {
            if (slots[i].op == op) slots[i] = {};
        }
    }
    void resetAll() {
        for (size_t i = 0; i < NUM_SLOTS; i++) slots[i] = {};
        next_evict = 0;
    }
};
inline BlockOpIterCache &blockOpCache() {
    static BlockOpIterCache c;
    return c;
}
inline void resetBlockOpCache() {
    blockOpCache().resetAll();
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
    auto *target = F<mlir::Block>(b);
    auto *o = F<mlir::Operation>(op);
    auto &cache = blockOpCache();
    // op may be migrating from another block — invalidate its old slot.
    if (auto *src = o->getBlock()) cache.invalidateBlock(src);
    cache.invalidateBlock(target);
    target->push_back(o);
}
extern "C" void MLIR_InsertBlockOpBeforeTerminator(MLIR_Context *,
                                                   MLIR_BlockHandle b,
                                                   MLIR_OpHandle op) {
    auto *block = F<mlir::Block>(b);
    auto *o = F<mlir::Operation>(op);
    auto &cache = blockOpCache();
    if (auto *src = o->getBlock()) cache.invalidateBlock(src);
    cache.invalidateBlock(block);
    if (!block->empty() && block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        block->getOperations().insert(mlir::Block::iterator(&block->back()), o);
    } else {
        block->push_back(o);
    }
}
extern "C" void MLIR_InsertBlockOpAtIndex(MLIR_Context *,
                                          MLIR_BlockHandle b,
                                          MLIR_OpHandle op, size_t idx) {
    auto *block = F<mlir::Block>(b);
    auto *o = F<mlir::Operation>(op);
    auto &cache = blockOpCache();
    if (auto *src = o->getBlock()) cache.invalidateBlock(src);
    cache.invalidateBlock(block);
    auto &ops = block->getOperations();
    size_t n = 0;
    for (auto it = ops.begin(); it != ops.end(); ++it, ++n) {
        if (n == idx) { ops.insert(it, o); return; }
    }
    block->push_back(o);
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
        mlir::Attribute val = na->getValue();
        // cf.switch's `case_values` is declared as `AnyIntElementsAttr`
        // (DenseIntElementsAttr) in upstream MLIR. The agnostic C lifter
        // creates it as DenseI32ArrayAttr (since it has no concept of
        // shaped types). Upstream silently drops the property when the
        // attribute type doesn't match, which then makes the
        // intermediate cf.switch fail to lift to scf.index_switch.
        // Convert here so the typed property storage accepts it.
        if (nm == "cf.switch" && na->getName().strref() == "case_values") {
            if (auto arr = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(val)) {
                auto i32Ty = mlir::IntegerType::get(&ctx, 32);
                auto shaped = mlir::RankedTensorType::get(
                    {(int64_t)arr.size()}, i32Ty);
                val = mlir::DenseIntElementsAttr::get(
                    shaped, llvm::ArrayRef<int32_t>(arr.asArrayRef().data(),
                                                    arr.size()));
            }
        }
        state.addAttribute(na->getName(), val);
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
        } else if (nm == "cf.switch") {
            // cf.switch has 3 operand segments: flag (always 1), default
            // operands (successor 0), and case operands (sum across
            // successors 1..N-1). It also needs case_operand_segments —
            // one int per case telling BranchOpInterface how to slice
            // the flat case-operand range per case.
            size_t n_default = (n_successors >= 1) ? n_successor_operands[0] : 0;
            int32_t n_case_total = 0;
            llvm::SmallVector<int32_t, 4> case_seg;
            for (size_t s = 1; s < n_successors; s++) {
                int32_t c = (int32_t)n_successor_operands[s];
                case_seg.push_back(c);
                n_case_total += c;
            }
            llvm::SmallVector<int32_t, 3> seg;
            seg.push_back((int32_t)n_operands);
            seg.push_back((int32_t)n_default);
            seg.push_back(n_case_total);
            state.addAttribute("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(&ctx, seg));
            state.addAttribute("case_operand_segments",
                mlir::DenseI32ArrayAttr::get(&ctx, case_seg));
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
    auto *op = F<mlir::Operation>(h);
    // mlir::OperationName::getAsOpaquePointer() returns the interned
    // Impl pointer — unique per registered op name. Cache the
    // (name pointer -> MLIR_OpType) mapping so the common case is a
    // single DenseMap lookup instead of a 256-entry string scan.
    auto name = op->getName();
    const void *key = name.getAsOpaquePointer();
    static llvm::DenseMap<const void *, MLIR_OpType> *nameCache = nullptr;
    if (!nameCache) nameCache = new llvm::DenseMap<const void *, MLIR_OpType>();
    auto it = nameCache->find(key);
    if (it != nameCache->end()) return it->second;
    MLIR_OpType t = opTypeFromName(name.getStringRef());
    (*nameCache)[key] = t;
    return t;
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
static std::vector<mlir::NamedAttribute> &
ensureCachedAttrs(mlir::Operation *op) {
    auto &m = opUserAttrs();
    auto it = m.find(op);
    if (it != m.end()) return it->second;
    auto attrs = op->getAttrs();
    auto &v = m[op];
    v.assign(attrs.begin(), attrs.end());
    return v;
}
extern "C" size_t MLIR_GetOpNumAttributes(MLIR_OpHandle h) {
    auto *op = F<mlir::Operation>(h);
    return ensureCachedAttrs(op).size();
}
extern "C" MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle h, size_t i) {
    auto *op = F<mlir::Operation>(h);
    return reinterpret_cast<uintptr_t>(&ensureCachedAttrs(op)[i]);
}
extern "C" MLIR_AttributeHandle MLIR_GetOpAttributeByName(MLIR_OpHandle h,
                                                          const char *name) {
    auto *op = F<mlir::Operation>(h);
    auto &keep = opByNameAttrs()[op];
    auto &mctx = globalCtx().mctx;
    auto stash = [&](mlir::Attribute a) -> MLIR_AttributeHandle {
        keep.emplace_back(std::make_unique<mlir::NamedAttribute>(
            mlir::StringAttr::get(op->getContext(), name), a));
        return reinterpret_cast<uintptr_t>(keep.back().get());
    };
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
        llvm::StringRef nm(name);
        if (nm == "callee") {
            if (auto c = call.getCalleeAttr()) return stash(c);
            return MLIR_INVALID_HANDLE;
        }
        if (nm == "var_callee_type") {
            if (auto t = call.getVarCalleeTypeAttr()) return stash(t);
            return MLIR_INVALID_HANDLE;
        }
    }
    // cf.switch stores case_values as a DenseIntElementsAttr (typed
    // property). The agnostic-API consumers (e.g. the C CF-to-SCF lifter)
    // pass it in / read it out as a DenseI32ArrayAttr and parse the
    // resulting `array<i32: V0, V1, ...>` text format. Round-trip the
    // attribute back to DenseI32ArrayAttr here so reads under upstream
    // produce the same text format as native.
    if (mlir::isa<mlir::cf::SwitchOp>(op) &&
        llvm::StringRef(name) == "case_values") {
        if (auto sw = mlir::dyn_cast<mlir::cf::SwitchOp>(op)) {
            if (auto cv = sw.getCaseValuesAttr()) {
                llvm::SmallVector<int32_t, 8> vals;
                vals.reserve(cv.getNumElements());
                for (auto apv : cv.getValues<llvm::APInt>())
                    vals.push_back((int32_t)apv.getSExtValue());
                return stash(mlir::DenseI32ArrayAttr::get(&mctx, vals));
            }
        }
    }
    if (auto a = op->getAttr(name)) return stash(a);
    auto dict = op->getAttrDictionary();
    if (auto na = dict.getNamed(name)) {
        keep.emplace_back(std::make_unique<mlir::NamedAttribute>(*na));
        return reinterpret_cast<uintptr_t>(keep.back().get());
    }
    return MLIR_INVALID_HANDLE;
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
    auto &cache = blockOpCache();
    if (auto *s = cache.find(b)) {
        if (s->cached_size != SIZE_MAX) return s->cached_size;
        size_t n = std::distance(b->begin(), b->end());
        s->cached_size = n;
        return n;
    }
    return std::distance(b->begin(), b->end());
}
extern "C" MLIR_OpHandle MLIR_GetBlockOp(MLIR_BlockHandle h, size_t i) {
    auto *b = F<mlir::Block>(h);
    auto &cache = blockOpCache();
    if (auto *s = cache.find(b)) {
        if (i == s->idx) {
            return H(s->op);
        }
        if (i == s->idx + 1) {
            auto *nx = s->op->getNextNode();
            if (nx) {
                s->idx = i;
                s->op = nx;
                return H(nx);
            }
        }
        if (s->idx > 0 && i + 1 == s->idx) {
            auto *pv = s->op->getPrevNode();
            if (pv) {
                s->idx = i;
                s->op = pv;
                return H(pv);
            }
        }
        // Small jumps near the cached position: walk via getNextNode /
        // getPrevNode from the cache instead of restarting from begin().
        // The 64-step cap keeps worst-case behavior bounded if the caller
        // does scattered access.
        if (i > s->idx && i - s->idx < 64) {
            auto *cur = s->op;
            size_t delta = i - s->idx;
            for (size_t k = 0; k < delta && cur; k++) cur = cur->getNextNode();
            if (cur) {
                s->idx = i;
                s->op = cur;
                return H(cur);
            }
        }
        if (i < s->idx && s->idx - i < 64) {
            auto *cur = s->op;
            size_t delta = s->idx - i;
            for (size_t k = 0; k < delta && cur; k++) cur = cur->getPrevNode();
            if (cur) {
                s->idx = i;
                s->op = cur;
                return H(cur);
            }
        }
    }
    // Fallback: scan from begin and seed the cache.
    auto it = b->begin();
    std::advance(it, i);
    mlir::Operation *op = &*it;
    auto *s = cache.findOrEvict(b);
    s->block = b;
    s->idx = i;
    s->op = op;
    s->cached_size = SIZE_MAX;  // unknown; will be filled by NumOps if called
    return H(op);
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
    auto t = typeF(h);
    if (auto ft = llvm::dyn_cast<mlir::FunctionType>(t)) return ft.getNumInputs();
    if (auto ft = llvm::dyn_cast<mlir::LLVM::LLVMFunctionType>(t)) return ft.getNumParams();
    return 0;
}
extern "C" MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_TypeHandle h, size_t idx) {
    auto t = typeF(h);
    if (auto ft = llvm::dyn_cast<mlir::FunctionType>(t)) {
        if (idx >= ft.getNumInputs()) return MLIR_INVALID_HANDLE;
        return typeH(ft.getInput(idx));
    }
    if (auto ft = llvm::dyn_cast<mlir::LLVM::LLVMFunctionType>(t)) {
        if (idx >= ft.getNumParams()) return MLIR_INVALID_HANDLE;
        return typeH(ft.getParamType(idx));
    }
    return MLIR_INVALID_HANDLE;
}
extern "C" size_t MLIR_GetTypeFunctionNumResults(MLIR_TypeHandle h) {
    auto t = typeF(h);
    if (auto ft = llvm::dyn_cast<mlir::FunctionType>(t)) return ft.getNumResults();
    if (auto ft = llvm::dyn_cast<mlir::LLVM::LLVMFunctionType>(t)) {
        // LLVMFunctionType has exactly one return type; "void" counts as 0
        // results to mirror mlir::FunctionType semantics for callers that
        // want to spell `declare void` etc.
        return llvm::isa<mlir::LLVM::LLVMVoidType>(ft.getReturnType()) ? 0 : 1;
    }
    return 0;
}
extern "C" MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_TypeHandle h, size_t idx) {
    auto t = typeF(h);
    if (auto ft = llvm::dyn_cast<mlir::FunctionType>(t)) {
        if (idx >= ft.getNumResults()) return MLIR_INVALID_HANDLE;
        return typeH(ft.getResult(idx));
    }
    if (auto ft = llvm::dyn_cast<mlir::LLVM::LLVMFunctionType>(t)) {
        if (idx != 0) return MLIR_INVALID_HANDLE;
        if (llvm::isa<mlir::LLVM::LLVMVoidType>(ft.getReturnType())) return MLIR_INVALID_HANDLE;
        return typeH(ft.getReturnType());
    }
    return MLIR_INVALID_HANDLE;
}

extern "C" bool MLIR_GetTypeFunctionIsVarArg(MLIR_TypeHandle h) {
    auto t = typeF(h);
    if (auto ft = llvm::dyn_cast<mlir::LLVM::LLVMFunctionType>(t)) {
        return ft.isVarArg();
    }
    return false;
}

extern "C" MLIR_TypeHandle MLIR_GetTypeShapedElement(MLIR_TypeHandle h) {
    auto t = typeF(h);
    if (auto st = llvm::dyn_cast<mlir::ShapedType>(t)) {
        return typeH(st.getElementType());
    }
    return MLIR_INVALID_HANDLE;
}

extern "C" bool MLIR_IsTypeLLVMStruct(MLIR_TypeHandle h) {
    return llvm::isa<mlir::LLVM::LLVMStructType>(typeF(h));
}
extern "C" size_t MLIR_GetTypeLLVMStructNumFields(MLIR_TypeHandle h) {
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(typeF(h));
    if (!st) return 0;
    return st.getBody().size();
}
extern "C" MLIR_TypeHandle MLIR_GetTypeLLVMStructField(MLIR_TypeHandle h, size_t idx) {
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(typeF(h));
    if (!st || idx >= st.getBody().size()) return MLIR_INVALID_HANDLE;
    return typeH(st.getBody()[idx]);
}
extern "C" bool MLIR_IsTypeLLVMArray(MLIR_TypeHandle h) {
    return llvm::isa<mlir::LLVM::LLVMArrayType>(typeF(h));
}
extern "C" MLIR_TypeHandle MLIR_GetTypeLLVMArrayElement(MLIR_TypeHandle h) {
    auto at = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(typeF(h));
    if (!at) return MLIR_INVALID_HANDLE;
    return typeH(at.getElementType());
}
extern "C" uint64_t MLIR_GetTypeLLVMArrayNumElements(MLIR_TypeHandle h) {
    auto at = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(typeF(h));
    if (!at) return 0;
    return at.getNumElements();
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
    static llvm::DenseMap<const void *, std::string> *typeStringCache = nullptr;
    if (!typeStringCache)
        typeStringCache = new llvm::DenseMap<const void *, std::string>();
    const void *key = t.getAsOpaquePointer();
    auto it = typeStringCache->find(key);
    if (it != typeStringCache->end()) {
        return mkArenaString(ctx, it->second);
    }
    std::string buf;
    llvm::raw_string_ostream os(buf);
    t.print(os);
    os.flush();
    (*typeStringCache)[key] = buf;
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
extern "C" MLIR_AttributeHandle MLIR_CreateAttributeLLVMLinkageInternal(MLIR_Context *, string name) {
    auto &ctx = globalCtx().mctx;
    auto attr = mlir::LLVM::LinkageAttr::get(&ctx, mlir::LLVM::Linkage::Internal);
    return makeNamedAttr(llvm::StringRef(name.str, name.size), attr);
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

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeDenseI32Array(MLIR_Context *, string name,
                                                                    const int32_t *values, size_t count) {
    auto &ctx = globalCtx().mctx;
    llvm::SmallVector<int32_t, 8> v(values, values + count);
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::DenseI32ArrayAttr::get(&ctx, v));
}

extern "C" MLIR_AttributeHandle MLIR_CreateAttributeDenseI64Array(MLIR_Context *, string name,
                                                                    const int64_t *values, size_t count) {
    auto &ctx = globalCtx().mctx;
    llvm::SmallVector<int64_t, 8> v(values, values + count);
    return makeNamedAttr(llvm::StringRef(name.str, name.size),
                         mlir::DenseI64ArrayAttr::get(&ctx, v));
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *) {
    auto &ctx = globalCtx().mctx;
    return typeH(mlir::LLVM::LLVMPointerType::get(&ctx));
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMStructIdentified(MLIR_Context *, string name) {
    auto &ctx = globalCtx().mctx;
    return typeH(mlir::LLVM::LLVMStructType::getIdentified(&ctx,
                     llvm::StringRef(name.str, name.size)));
}

extern "C" void MLIR_SetTypeLLVMStructBody(MLIR_Context *, MLIR_TypeHandle struct_ty,
                                            const MLIR_TypeHandle *fields, size_t n_fields) {
    auto t = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(typeF(struct_ty));
    if (!t) {
        std::fprintf(stderr, "MLIR_SetTypeLLVMStructBody: not an LLVM struct type\n");
        return;
    }
    llvm::SmallVector<mlir::Type, 8> body;
    body.reserve(n_fields);
    for (size_t i = 0; i < n_fields; i++) body.push_back(typeF(fields[i]));
    if (llvm::failed(t.setBody(body, /*isPacked=*/false))) {
        std::fprintf(stderr, "MLIR_SetTypeLLVMStructBody: setBody failed (already set?)\n");
    }
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMArray(MLIR_Context *, MLIR_TypeHandle elem,
                                                     uint64_t count) {
    return typeH(mlir::LLVM::LLVMArrayType::get(typeF(elem), count));
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMFunction(MLIR_Context *,
                                                       MLIR_TypeHandle result,
                                                       const MLIR_TypeHandle *inputs,
                                                       size_t n_inputs,
                                                       bool is_var_arg) {
    llvm::SmallVector<mlir::Type, 8> in;
    in.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; i++) in.push_back(typeF(inputs[i]));
    return typeH(mlir::LLVM::LLVMFunctionType::get(typeF(result), in, is_var_arg));
}

extern "C" MLIR_TypeHandle MLIR_CreateTypeLLVMVoid(MLIR_Context *) {
    auto &ctx = globalCtx().mctx;
    return typeH(mlir::LLVM::LLVMVoidType::get(&ctx));
}

extern "C" MLIR_OpHandle MLIR_CreateLLVMGlobalString(MLIR_Context *,
                                                     string sym_name,
                                                     string bytes,
                                                     MLIR_LocationHandle location) {
    auto &ctx = globalCtx().mctx;
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
    return H(op.getOperation());
}

extern "C" MLIR_OpHandle MLIR_CreateLLVMGlobal(MLIR_Context *,
                                               string sym_name,
                                               MLIR_TypeHandle elem_ty,
                                               bool is_constant,
                                               int init_kind,
                                               int64_t init_int,
                                               double init_float,
                                               MLIR_BlockHandle *out_init_block,
                                               MLIR_LocationHandle location) {
    auto &ctx = globalCtx().mctx;
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
    return H(op.getOperation());
}

extern "C" MLIR_OpHandle MLIR_CreateLLVMGlobalArrayInit(MLIR_Context *,
                                                        string sym_name,
                                                        MLIR_TypeHandle array_ty,
                                                        bool is_constant,
                                                        string bytes,
                                                        MLIR_LocationHandle location) {
    auto &ctx = globalCtx().mctx;
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
    return H(op.getOperation());
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
    // Fallback: return the printed assembly form (e.g. `#llvm.linkage<internal>`
    // for `LLVM_LinkageAttr`). The agnostic impl stores attribute payloads as
    // strings and returns them verbatim, so callers expect the printed form
    // for non-string attrs; this matches that contract. Intern the printed
    // string so the storage outlives the call.
    static thread_local std::unordered_map<std::string, std::string> printed_cache;
    std::string buf;
    {
        llvm::raw_string_ostream os(buf);
        value.print(os);
    }
    auto it = printed_cache.find(buf);
    if (it == printed_cache.end()) {
        it = printed_cache.emplace(buf, buf).first;
    }
    const std::string &v = it->second;
    string out; out.str = const_cast<char *>(v.data()); out.size = v.size();
    return out;
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
// IR mutation primitives (Stage B / Stage C use only)
// -----------------------------------------------------------------------------

extern "C" void MLIR_ReplaceAllUsesOfValue(MLIR_Context *,
                                           MLIR_ValueHandle old_h,
                                           MLIR_ValueHandle new_h) {
    if (old_h == MLIR_INVALID_HANDLE || new_h == MLIR_INVALID_HANDLE) return;
    auto *oldBox = F<ValueBox>(old_h);
    auto *newBox = F<ValueBox>(new_h);
    if (!oldBox->v || !newBox->v) return;
    oldBox->v.replaceAllUsesWith(newBox->v);
}

extern "C" void MLIR_EraseOp(MLIR_Context *, MLIR_OpHandle op_h) {
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    auto &cache = blockOpCache();
    if (auto *blk = op->getBlock()) cache.invalidateBlock(blk);
    cache.invalidateOp(op);  // cached pointer would dangle past erase
    // Drop any cached attribute snapshot for this Operation* — once the
    // op is erased the MLIR allocator may reuse the same address for a
    // freshly-constructed op (potentially of a different kind), and the
    // stale userAttrs snapshot would shadow the new op's real attrs.
    opUserAttrs().erase(op);
    opByNameAttrs().erase(op);
    op->erase();
}

extern "C" void MLIR_SetOpRegion(MLIR_Context *, MLIR_OpHandle op_h,
                                 size_t idx, MLIR_RegionHandle region_h) {
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    if (idx >= op->getNumRegions()) return;
    auto *newRegion = F<mlir::Region>(region_h);
    auto &dst = op->getRegion(idx);
    // Blocks (and their op lists) are moved between regions. The blocks
    // themselves are unchanged, so their cached op pointers stay valid —
    // but to be safe against any block whose identity changes, clear
    // cache entries for all blocks in either region.
    auto &cache = blockOpCache();
    for (auto &b : dst) cache.invalidateBlock(&b);
    for (auto &b : *newRegion) cache.invalidateBlock(&b);
    dst.takeBody(*newRegion);
    delete newRegion;
}

extern "C" MLIR_RegionHandle MLIR_TakeOpRegion(MLIR_Context *,
                                                MLIR_OpHandle op_h,
                                                size_t idx) {
    if (op_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *op = F<mlir::Operation>(op_h);
    if (idx >= op->getNumRegions()) return MLIR_INVALID_HANDLE;
    auto &cache = blockOpCache();
    for (auto &b : op->getRegion(idx)) cache.invalidateBlock(&b);
    auto *out = new mlir::Region();
    out->takeBody(op->getRegion(idx));
    return H(out);
}

extern "C" MLIR_BlockHandle MLIR_GetOpParentBlock(MLIR_OpHandle op_h) {
    if (op_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *op = F<mlir::Operation>(op_h);
    auto *blk = op->getBlock();
    return blk ? H(blk) : MLIR_INVALID_HANDLE;
}

extern "C" size_t MLIR_GetBlockOpIndex(MLIR_BlockHandle blk_h,
                                       MLIR_OpHandle op_h) {
    if (blk_h == MLIR_INVALID_HANDLE || op_h == MLIR_INVALID_HANDLE)
        return SIZE_MAX;
    auto *blk = F<mlir::Block>(blk_h);
    auto *op  = F<mlir::Operation>(op_h);
    size_t i = 0;
    for (auto &o : blk->getOperations()) {
        if (&o == op) return i;
        i++;
    }
    return SIZE_MAX;
}

extern "C" MLIR_RegionHandle MLIR_GetBlockParentRegion(MLIR_BlockHandle blk_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    auto *r = blk->getParent();
    return r ? H(r) : MLIR_INVALID_HANDLE;
}

extern "C" void MLIR_MoveOpToBlockEnd(MLIR_Context *, MLIR_OpHandle op_h,
                                       MLIR_BlockHandle dest_h) {
    if (op_h == MLIR_INVALID_HANDLE || dest_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    auto *dest = F<mlir::Block>(dest_h);
    auto &cache = blockOpCache();
    if (auto *src = op->getBlock()) cache.invalidateBlock(src);
    cache.invalidateBlock(dest);
    op->moveBefore(dest, dest->end());
}

extern "C" void MLIR_MoveBlockToRegionEnd(MLIR_Context *, MLIR_BlockHandle blk_h,
                                          MLIR_RegionHandle dest_h) {
    if (blk_h == MLIR_INVALID_HANDLE || dest_h == MLIR_INVALID_HANDLE) return;
    auto *blk = F<mlir::Block>(blk_h);
    auto *dest = F<mlir::Region>(dest_h);
    // Block::moveBefore requires the block to already have a parent region;
    // for blocks that don't, just splice into dest directly.
    if (blk->getParent() == nullptr) {
        dest->push_back(blk);
    } else {
        blk->moveBefore(dest, dest->end());
    }
}

extern "C" void MLIR_SetOpOperand(MLIR_Context *, MLIR_OpHandle op_h,
                                  size_t idx, MLIR_ValueHandle v_h) {
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    if (idx >= op->getNumOperands()) return;
    auto *box = F<ValueBox>(v_h);
    if (!box || !box->v) return;
    op->setOperand(static_cast<unsigned>(idx), box->v);
}

extern "C" void MLIR_SetOpSuccessor(MLIR_Context *, MLIR_OpHandle op_h,
                                    size_t succ_idx, MLIR_BlockHandle blk_h) {
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    if (succ_idx >= op->getNumSuccessors()) return;
    auto *blk = (blk_h == MLIR_INVALID_HANDLE) ? nullptr : F<mlir::Block>(blk_h);
    op->setSuccessor(blk, static_cast<unsigned>(succ_idx));
}

extern "C" void MLIR_SetOpSuccessorOperands(MLIR_Context *, MLIR_OpHandle op_h,
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
            auto *box = F<ValueBox>(values[i]);
            vals.push_back(box ? box->v : mlir::Value());
        } else {
            vals.push_back(mlir::Value());
        }
    }
    mut.assign(vals);
}

extern "C" void MLIR_EraseBlock(MLIR_Context *, MLIR_BlockHandle blk_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return;
    auto *blk = F<mlir::Block>(blk_h);
    blockOpCache().invalidateBlock(blk);
    if (blk->getParent() != nullptr) {
        // Detach without freeing contained ops/args.
        blk->getParent()->getBlocks().remove(blk);
    }
}

extern "C" void MLIR_InsertRegionBlockAfter(MLIR_Context *, MLIR_RegionHandle region_h,
                                            MLIR_BlockHandle blk_h,
                                            MLIR_BlockHandle after_h) {
    if (region_h == MLIR_INVALID_HANDLE || blk_h == MLIR_INVALID_HANDLE) return;
    auto *region = F<mlir::Region>(region_h);
    auto *blk = F<mlir::Block>(blk_h);
    // Detach block from any current region first.
    if (blk->getParent() != nullptr)
        blk->getParent()->getBlocks().remove(blk);
    if (after_h == MLIR_INVALID_HANDLE) {
        region->getBlocks().push_front(blk);
        return;
    }
    auto *after = F<mlir::Block>(after_h);
    // Insert right after `after`.
    auto it = std::next(after->getIterator());
    region->getBlocks().insert(it, blk);
}

extern "C" void MLIR_InsertRegionBlockBefore(MLIR_Context *, MLIR_RegionHandle region_h,
                                             MLIR_BlockHandle blk_h,
                                             MLIR_BlockHandle before_h) {
    if (region_h == MLIR_INVALID_HANDLE || blk_h == MLIR_INVALID_HANDLE) return;
    auto *region = F<mlir::Region>(region_h);
    auto *blk = F<mlir::Block>(blk_h);
    if (blk->getParent() != nullptr)
        blk->getParent()->getBlocks().remove(blk);
    if (before_h == MLIR_INVALID_HANDLE) {
        region->push_back(blk);
        return;
    }
    auto *before = F<mlir::Block>(before_h);
    region->getBlocks().insert(before->getIterator(), blk);
}

extern "C" MLIR_OpHandle MLIR_GetBlockTerminator(MLIR_BlockHandle blk_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    if (blk->empty()) return MLIR_INVALID_HANDLE;
    return H(&blk->back());
}

extern "C" MLIR_OpHandle MLIR_GetBlockParentOp(MLIR_BlockHandle blk_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    auto *r = blk->getParent();
    if (!r) return MLIR_INVALID_HANDLE;
    auto *op = r->getParentOp();
    return op ? H(op) : MLIR_INVALID_HANDLE;
}

extern "C" bool MLIR_BlockIsEntry(MLIR_BlockHandle blk_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return false;
    auto *blk = F<mlir::Block>(blk_h);
    return blk->isEntryBlock();
}

extern "C" size_t MLIR_GetBlockNumPredecessors(MLIR_BlockHandle blk_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return 0;
    auto *blk = F<mlir::Block>(blk_h);
    size_t n = 0;
    for (auto it = blk->pred_begin(); it != blk->pred_end(); ++it) ++n;
    return n;
}

extern "C" MLIR_BlockHandle MLIR_GetBlockPredecessor(MLIR_BlockHandle blk_h,
                                                    size_t idx,
                                                    size_t *out_succ_idx) {
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

extern "C" MLIR_ValueHandle MLIR_AddBlockArgument(MLIR_Context *,
                                                  MLIR_BlockHandle blk_h,
                                                  MLIR_TypeHandle type_h,
                                                  MLIR_LocationHandle loc_h) {
    if (blk_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *blk = F<mlir::Block>(blk_h);
    mlir::Type ty = typeF(type_h);
    mlir::Location loc = (loc_h == MLIR_INVALID_HANDLE)
        ? mlir::Location(mlir::UnknownLoc::get(&globalCtx().mctx))
        : locF(loc_h);
    auto arg = blk->addArgument(ty, loc);
    return H(boxFor(arg));
}

extern "C" void MLIR_EraseBlockArguments(MLIR_Context *,
                                          MLIR_BlockHandle blk_h,
                                          size_t start, size_t count) {
    if (blk_h == MLIR_INVALID_HANDLE || count == 0) return;
    auto *blk = F<mlir::Block>(blk_h);
    // Clear cached ValueBox entries for the args being erased before
    // they are destroyed.
    auto &idx = valueIndex();
    for (size_t i = 0; i < count; i++) {
        unsigned argIdx = static_cast<unsigned>(start + i);
        if (argIdx >= blk->getNumArguments()) break;
        idx.erase(blk->getArgument(argIdx).getAsOpaquePointer());
    }
    blk->eraseArguments(static_cast<unsigned>(start),
                        static_cast<unsigned>(count));
}

extern "C" MLIR_BlockHandle MLIR_GetValueParentBlock(MLIR_ValueHandle v_h) {
    if (v_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *box = F<ValueBox>(v_h);
    if (!box->v) return MLIR_INVALID_HANDLE;
    auto *blk = box->v.getParentBlock();
    return blk ? H(blk) : MLIR_INVALID_HANDLE;
}

extern "C" size_t MLIR_GetValueNumUses(MLIR_Context *, MLIR_ValueHandle v_h) {
    if (v_h == MLIR_INVALID_HANDLE) return 0;
    auto *box = F<ValueBox>(v_h);
    if (!box->v) return 0;
    size_t n = 0;
    for (auto &u : box->v.getUses()) { (void)u; ++n; }
    return n;
}

extern "C" MLIR_OpHandle MLIR_GetValueUseOwner(MLIR_Context *,
                                                MLIR_ValueHandle v_h,
                                                size_t idx,
                                                size_t *out_operand_idx) {
    if (v_h == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    auto *box = F<ValueBox>(v_h);
    if (!box->v) return MLIR_INVALID_HANDLE;
    size_t i = 0;
    for (auto &u : box->v.getUses()) {
        if (i == idx) {
            if (out_operand_idx) *out_operand_idx = u.getOperandNumber();
            return H(u.getOwner());
        }
        ++i;
    }
    if (out_operand_idx) *out_operand_idx = SIZE_MAX;
    return MLIR_INVALID_HANDLE;
}

extern "C" void MLIR_SetOpOperands(MLIR_Context *, MLIR_OpHandle op_h,
                                    const MLIR_ValueHandle *values, size_t n) {
    if (op_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    llvm::SmallVector<mlir::Value, 8> vs;
    vs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        if (values && values[i] != MLIR_INVALID_HANDLE) {
            auto *box = F<ValueBox>(values[i]);
            vs.push_back(box ? box->v : mlir::Value());
        } else {
            vs.push_back(mlir::Value());
        }
    }
    op->setOperands(vs);
}

extern "C" void MLIR_AppendOpSuccessorOperand(MLIR_Context *, MLIR_OpHandle op_h,
                                               size_t succ_idx,
                                               MLIR_ValueHandle v_h) {
    if (op_h == MLIR_INVALID_HANDLE || v_h == MLIR_INVALID_HANDLE) return;
    auto *op = F<mlir::Operation>(op_h);
    auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(op);
    if (!branch) return;
    if (succ_idx >= op->getNumSuccessors()) return;
    auto *box = F<ValueBox>(v_h);
    if (!box || !box->v) return;
    mlir::SuccessorOperands sops =
        branch.getSuccessorOperands(static_cast<unsigned>(succ_idx));
    mlir::MutableOperandRange mut = sops.getMutableForwardedOperands();
    mut.append({box->v});
}

extern "C" void MLIR_SpliceBlockOps(MLIR_Context *, MLIR_BlockHandle dst_h,
                                     MLIR_BlockHandle src_h) {
    if (dst_h == MLIR_INVALID_HANDLE || src_h == MLIR_INVALID_HANDLE) return;
    auto *dst = F<mlir::Block>(dst_h);
    auto *src = F<mlir::Block>(src_h);
    auto &cache = blockOpCache();
    cache.invalidateBlock(dst);
    cache.invalidateBlock(src);
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

// Rewrite tinyc's unregistered `llvm.intr.wasm.memory.size` /
// `llvm.intr.wasm.memory.grow` ops into upstream LLVM dialect's
// `llvm.call_intrinsic` form. tinyc emits these as unregistered ops (and
// drops the memory-index argument, since only the default linear memory is
// modeled); the native llvm->wasmssa pipeline recognizes the unregistered
// names directly, but upstream's LLVM-IR translator only knows the
// registered LLVM dialect ops. Rewriting to `llvm.call_intrinsic` lets
// `translateModuleToLLVMIR` emit the actual `llvm.wasm.memory.size` /
// `llvm.wasm.memory.grow` IR intrinsics.
static void rewriteWasmMemoryIntrinsicsForUpstream(mlir::ModuleOp module) {
    using namespace mlir;
    llvm::SmallVector<Operation *> to_erase;
    module.walk([&](Operation *op) {
        llvm::StringRef name = op->getName().getStringRef();
        bool is_size = (name == "llvm.intr.wasm.memory.size");
        bool is_grow = (name == "llvm.intr.wasm.memory.grow");
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
        opUserAttrs().erase(op);
        opByNameAttrs().erase(op);
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
    (void)ctx;
    resetBlockOpCache();
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        std::fprintf(stderr,
                     "MLIR_LowerToLLVMDialectUpstream: handle is not a ModuleOp\n");
        return false;
    }
    auto &mctx = globalCtx().mctx;
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
        std::fprintf(stderr,
                     "MLIR_LowerToLLVMDialectUpstream: pass pipeline failed\n");
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
                                       MLIR_OpHandle module);

extern "C" bool MLIR_LiftCfToScf(MLIR_Context *ctx, MLIR_OpHandle module_h) {
    if (module_h == MLIR_INVALID_HANDLE) return false;
    resetBlockOpCache();
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        std::fprintf(stderr,
                     "MLIR_LiftCfToScf: handle is not a ModuleOp\n");
        return false;
    }
    auto &mctx = globalCtx().mctx;
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
    resetBlockOpCache();
    return MLIR_LiftCfToScfNative(ctx, module_h);
}

extern "C" bool MLIR_LowerToLLVMDialectForWasmUpstream(MLIR_Context *ctx,
                                                       MLIR_OpHandle module_h) {
    (void)ctx;
    resetBlockOpCache();
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        std::fprintf(stderr,
                     "MLIR_LowerToLLVMDialectForWasmUpstream: handle is not a ModuleOp\n");
        return false;
    }
    auto &mctx = globalCtx().mctx;
    // The opUserAttrs / opByNameAttrs caches are keyed by Operation* and
    // may contain stale entries for ops about to be erased by the
    // conversion passes below. Allocator reuse can produce false hits
    // later; clear them before lowering.
    opUserAttrs().clear();
    opByNameAttrs().clear();
    bool has_vector_op = false;
    module.walk([&](mlir::Operation *o) {
        if (o->getDialect() && o->getDialect()->getNamespace() == "vector") {
            has_vector_op = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    {
        mlir::IRRewriter rewriter(&mctx);
        (void)mlir::eraseUnreachableBlocks(rewriter, module->getRegions());
    }
    // Lift CF to SCF on both func.func and llvm.func bodies. tinyc emits
    // variadic functions as llvm.func (since func.func has no var-arg form),
    // and the stock `LiftControlFlowToSCFPass` only walks func.func; do it
    // ourselves so the wasm backend can see structured loops/ifs.
    liftLLVMFuncCFGToSCF(module);
    mlir::PassManager pm(&mctx);
    pm.addPass(mlir::createLiftControlFlowToSCFPass());
    if (has_vector_op) {
        pm.addPass(mlir::createConvertVectorToLLVMPass());
    }
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(module))) {
        std::fprintf(stderr,
                     "MLIR_LowerToLLVMDialectForWasmUpstream: pass pipeline failed\n");
        return false;
    }
    // Drop any cached attrs from before / during the pipeline: the ops they
    // were keyed on may have been erased and the addresses recycled.
    opUserAttrs().clear();
    opByNameAttrs().clear();
    return true;
}

extern "C" string MLIR_TranslateModuleToLLVMIRUpstream(MLIR_Context *ctx,
                                                       MLIR_OpHandle module_h) {
    auto *op = F<mlir::Operation>(module_h);
    auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
    if (!module) {
        std::fprintf(stderr,
                     "MLIR_TranslateModuleToLLVMIRUpstream: not a ModuleOp\n");
        return mkRefString(llvm::StringRef());
    }
    // Register the LLVM-IR translation interfaces (idempotent).
    mlir::registerBuiltinDialectTranslation(globalCtx().mctx);
    mlir::registerLLVMDialectTranslation(globalCtx().mctx);

    llvm::LLVMContext llctx;
    auto llmod = mlir::translateModuleToLLVMIR(module, llctx);
    if (!llmod) {
        std::fprintf(stderr,
                     "MLIR_TranslateModuleToLLVMIRUpstream: translation failed\n");
        return mkRefString(llvm::StringRef());
    }
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
        std::fprintf(stderr,
                     "MLIR_TranslateModuleToWasmUpstream: handle is not a ModuleOp\n");
        return mkRefString(llvm::StringRef());
    }

    llvm::LLVMContext llctx;
    std::unique_ptr<llvm::Module> llmod;
    {
        mlir::registerBuiltinDialectTranslation(globalCtx().mctx);
        mlir::registerLLVMDialectTranslation(globalCtx().mctx);
        llmod = mlir::translateModuleToLLVMIR(module, llctx);
        if (!llmod) {
            std::fprintf(stderr,
                         "MLIR_TranslateModuleToWasmUpstream: translation to LLVM IR failed\n");
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
        std::fprintf(stderr,
                     "MLIR_TranslateModuleToWasmUpstream: lookupTarget failed: %s\n",
                     err.c_str());
        return mkRefString(llvm::StringRef());
    }

    llvm::TargetOptions opts;
    auto rm = std::optional<llvm::Reloc::Model>();
    auto cm = std::optional<llvm::CodeModel::Model>();
    std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
        triple, /*CPU*/ "generic", /*Features*/ "", opts, rm, cm));
    if (!tm) {
        std::fprintf(stderr,
                     "MLIR_TranslateModuleToWasmUpstream: createTargetMachine failed\n");
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
        std::fprintf(stderr,
                     "MLIR_TranslateModuleToWasmUpstream: target cannot emit object file\n");
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
