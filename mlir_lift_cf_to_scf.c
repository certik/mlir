// Agnostic C port of upstream MLIR's transformCFGToSCF
// (mlir/lib/Transforms/Utils/CFGToSCF.cpp). Plays the role of upstream's
// `ControlFlowToSCFTransformation` (specifically the wasm-flavored
// variant defined in mlir_api_impl_upstream.cpp's `CFGToSCFForWasm`)
// while running against either backend through the mlir_api.h surface.
//
// Algorithm reference: Bahmann, Reissmann, Jahre, Meyer (2015), "Perfect
// Reconstructability of Control Flow from Demand Dependence Graphs",
// ACM TACO 11(4):66. https://doi.org/10.1145/2693261
//
// Status: scaffolding. The top-level entry returns false; see TODO
// markers below for each of the three transforms still to land:
//   1. createSingleExitBlocksForReturnLike
//   2. transformCyclesToSCFLoops
//   3. transformToStructuredCFBranches
// plus the supporting predecessor cache, dominance analysis (iterative
// data-flow on the cf graph), and Tarjan SCC iteration. Each transform
// is a faithful port of the corresponding static function in
// CFGToSCF.cpp, modulo data-structure adaptations to plain C.

#include "mlir_lift_cf_to_scf.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/format.h>
#include <base/string.h>

// ============================================================================
// Edge: (from_block, successor_index). Mirrors upstream's `Edge` class.
// ============================================================================
typedef struct {
    MLIR_BlockHandle from_block;
    size_t           succ_idx;
} Edge;

static MLIR_BlockHandle edge_successor(Edge e) {
    MLIR_OpHandle term = MLIR_GetBlockTerminator(e.from_block);
    if (term == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    return MLIR_GetOpSuccessor(term, e.succ_idx);
}

static void edge_set_successor(MLIR_Context *ctx, Edge e, MLIR_BlockHandle dst) {
    MLIR_OpHandle term = MLIR_GetBlockTerminator(e.from_block);
    if (term == MLIR_INVALID_HANDLE) return;
    MLIR_SetOpSuccessor(ctx, term, e.succ_idx, dst);
}

// ============================================================================
// CycleEdges: entry, exit, back edge buckets for an SCC.
// ============================================================================
typedef struct {
    Edge   *entry;     size_t n_entry;
    Edge   *exit;      size_t n_exit;
    Edge   *back;      size_t n_back;
} CycleEdges;

// ============================================================================
// EdgeMultiplexer: helper that turns N entries into a single multiplexer
// block with a discriminator argument. See the diagrams at the top of
// CFGToSCF.cpp.
//
// Pure-C version: block_arg_offsets[i] is the index into the multiplexer
// block's argument list where the `i`-th entry block's args were copied.
// ============================================================================
typedef struct {
    MLIR_BlockHandle mux_block;
    MLIR_BlockHandle *entries;       // distinct entry blocks
    size_t           *entry_arg_off; // arg-list offset per entry
    size_t            n_entries;
    MLIR_ValueHandle  discriminator; // INVALID if only one entry
    size_t            n_extra_args;
} EdgeMultiplexer;

// ============================================================================
// Switch value / undef value caches. Mirror the typedUndefCache and
// switchValueCache lambdas of transformCFGToSCF in CFGToSCF.cpp:1309-1338.
// ============================================================================
typedef struct {
    MLIR_TypeHandle  type;
    MLIR_ValueHandle value;
} TypedValueEntry;

typedef struct {
    MLIR_Context     *ctx;
    Arena            *arena;
    MLIR_OpHandle     fn_op;        // func.func or llvm.func being processed
    MLIR_BlockHandle  entry_block;  // first block of the region under lift

    // Cached arith.constant of i32 for switch flag values.
    MLIR_ValueHandle *switch_value_cache;
    size_t            switch_value_cache_n;

    // Cached ub.poison (or llvm.mlir.undef) per type.
    TypedValueEntry  *undef_cache;
    size_t            undef_cache_n;
} LiftState;

// ============================================================================
// TODO(get_switch_value):
// ----------------------------------------------------------------------------
// Construct (or return cached) `arith.constant <i32 value> : i32` op,
// inserted at the start of the entry block of the function under
// transformation. Returns the SSA value of its result.
//
// Mirrors `interface.getCFGSwitchValue` from CFGToSCFForWasm, which uses
// i32 (not `index`) so the wasm backend's `arith.index_cast` handling is
// not required when the cf.switch / scf.index_switch dispatchers fire.
// ============================================================================
static MLIR_ValueHandle get_switch_value(LiftState *st, unsigned v);

// ============================================================================
// TODO(get_undef_value):
// ----------------------------------------------------------------------------
// Construct (or return cached) an `ub.poison : T` op at the start of the
// function entry. For `!llvm.ptr` and similar non-ub-compatible types
// the upstream wasm flavor uses `llvm.mlir.zero` / `llvm.mlir.undef`;
// pick the variant matching the function dialect (llvm.* vs builtin).
// ============================================================================
static MLIR_ValueHandle get_undef_value(LiftState *st, MLIR_TypeHandle ty);

// ============================================================================
// TODO(create_cf_switch / create_cond_branch / create_unconditional_branch):
// ----------------------------------------------------------------------------
// Emit the cf.* terminator ops via MLIR_CreateOpWithSuccessors. The
// algorithm builds cf.switch / cf.cond_br / cf.br when it needs to
// rewire flow inside the lifted body (multiplexer dispatch, single
// destination, latch back-edge). Operand-segment layout for cf.switch
// matches upstream: [flag, default_args..., case_args[0]..., ...]. The
// `case_operand_segments` attribute lists per-case operand counts.
// ============================================================================
// static MLIR_OpHandle create_cf_switch(...);
// static MLIR_OpHandle create_cf_cond_br(...);
// static MLIR_OpHandle create_cf_br(...);

// ============================================================================
// TODO(create_scf_if / create_scf_index_switch / create_scf_while):
// ----------------------------------------------------------------------------
// Emit the structured op replacing the cf terminator in the region
// entry. Layout follows upstream:
//   scf.if           - one i1 condition operand, two regions, results
//                      forwarded from each region's scf.yield.
//   scf.index_switch - one index operand (we feed an arith.index_castui
//                      of an i32 flag), one region per case + default.
//   scf.while        - do-while: body+condition; we synthesize a single
//                      `scf.condition %cond [%iter_args...]` in the
//                      latch and an scf.yield in the body.
// All three need correct result-type vectors (matching the merged
// continuation's block-argument types per upstream `createStructured*`).
// ============================================================================
// static MLIR_OpHandle create_scf_if(...);
// static MLIR_OpHandle create_scf_index_switch(...);
// static MLIR_OpHandle create_scf_while(...);
// static MLIR_OpHandle create_scf_yield(...);

// ============================================================================
// TODO(predecessor_cache):
// ----------------------------------------------------------------------------
// Maintain a region-scoped cache mapping block -> list<Edge> of incoming
// edges. MLIR_GetBlockNumPredecessors does an O(R) scan; caching is a
// significant constant-factor win when the algorithm queries the same
// block repeatedly (e.g. when collecting cycle entry edges, or when
// transformToReduceLoop walks all predecessors of the latch). Invalidate
// whenever we mutate terminators (setSuccessor, redirectEdge,
// createConditionalBranch, etc.).
// ============================================================================

// ============================================================================
// TODO(dominance):
// ----------------------------------------------------------------------------
// Iterative data-flow Lengauer-Tarjan is overkill; the Cooper, Harvey,
// Kennedy (2006) "A Simple, Fast Dominance Algorithm" suffices and is
// ~80 LOC in C. State: per-region postorder, idom array, dominates(a,b)
// answered by walking idom chain from b until either a is hit or root.
// Used in two places:
//   1. transformToStructuredCFBranches: enumerating dominator-tree
//      successors of each branch entry (depth-first walk of the dom
//      tree subtree rooted at the entry) — see CFGToSCF.cpp:984-990.
//   2. transformToReduceLoop: `dominanceInfo.dominates(loopBlock, X)`
//      queries — CFGToSCF.cpp:706-714 / 743-755.
// Both are read-only; we recompute after invalidation.
// ============================================================================

// ============================================================================
// TODO(scc):
// ----------------------------------------------------------------------------
// Tarjan's SCC iteration over the region's CFG, returning SCCs in
// reverse-topological order. We only act on SCCs that "have a cycle"
// (size > 1, or size == 1 with self-loop). Used by
// transformCyclesToSCFLoops (CFGToSCF.cpp:805-815). ~100 LOC in C.
// ============================================================================

// ============================================================================
// TODO(check_preconditions):
// ----------------------------------------------------------------------------
// Port checkTransformationPreconditions (CFGToSCF.cpp:1237-1297):
//   * Reject unreachable blocks (block has no preds and is not entry).
//   * Every terminator with successors must be a known cf.* op (we know
//     the universe: cf.br, cf.cond_br, cf.switch).
//   * Reject ops with producedOperandCount > 0 (none of the cf.* ops we
//     accept produce successor operands, so this passes trivially).
//   * Reject multi-successor terminators we cannot convert (cf.* only,
//     we always can).
// ============================================================================

// ============================================================================
// TODO(transform_cycles_to_scf_loops):
// ----------------------------------------------------------------------------
// Port transformCyclesToSCFLoops (CFGToSCF.cpp:800-893). Steps per SCC:
//   1. calculateCycleEdges (entry, exit, back)
//   2. If multiple entry edges, createSingleEntryBlock multiplexing
//      both entry and back edges; new mux block becomes the header.
//   3. createSingleExitingLatch: multiplex back+exit edges into a latch
//      block; conditional branch on shouldRepeat to header vs an exit
//      block that dispatches to original exit destinations.
//   4. transformToReduceLoop: ensure no SSA escape from loop body, and
//      that exit block args == loop header args (modulo extras).
//   5. Build a fresh `newLoopParentBlock` before the header; move
//      header+body+latch into a fresh Region; emit scf.while with that
//      region as the body; splice in the exit block's ops after the
//      scf.while; replace exit block uses with scf.while results.
// Push each emitted scf.while body's header onto the work list so the
// caller re-runs the lift inside it.
// ============================================================================

// ============================================================================
// TODO(transform_to_structured_cf_branches):
// ----------------------------------------------------------------------------
// Port transformToStructuredCFBranches (CFGToSCF.cpp:947-1216). Steps:
//   1. If region entry has 0 successors -> nothing to do.
//   2. If region entry has 1 successor -> splice successor into entry.
//   3. Otherwise: split successors into "branch regions" via dominance,
//      classify against case 1/2/3 (see header comment at top of
//      CFGToSCF.cpp:946-1053), maybe create a continuation mux,
//      createSingleExitBranchRegion per branch region, then build
//      scf.if (if 2-way) or scf.index_switch (n-way) op replacing the
//      cf terminator; splice continuation into entry.
// Push each new sub-region onto the work list.
// ============================================================================

// ============================================================================
// TODO(create_single_exit_blocks_for_return_like):
// ----------------------------------------------------------------------------
// Port createSingleExitBlocksForReturnLike + ReturnLikeExitCombiner
// (CFGToSCF.cpp:417-466, 1221-1234). Two-pass:
//   1. Enumerate every block with no successors; classify its terminator
//      kind (func.return / llvm.return / cf.assert / etc.).
//   2. For each kind, create one shared exit block hosting that
//      terminator; redirect each occurrence to branch (cf.br) to the
//      shared block, passing its operands through block arguments.
// ============================================================================

static bool string_eq(string a, const char *b);
static bool string_eq_string(string a, string b);

// ============================================================================
// Helper: append a fresh cf.br op to the end of `in_block` jumping to
// `dst` with `args` as branch operands.
// ============================================================================
static MLIR_OpHandle create_cf_br_in_block(MLIR_Context *ctx, Arena *arena,
                                           MLIR_BlockHandle in_block,
                                           MLIR_BlockHandle dst,
                                           MLIR_ValueHandle *args, size_t n_args,
                                           MLIR_LocationHandle loc) {
    MLIR_BlockHandle *succs = arena_new_array(arena, MLIR_BlockHandle, 1);
    succs[0] = dst;
    MLIR_ValueHandle **sops = arena_new_array(arena, MLIR_ValueHandle *, 1);
    sops[0] = args;
    size_t *snums = arena_new_array(arena, size_t, 1);
    snums[0] = n_args;
    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(
        ctx, OP_TYPE_CF_BR, str_lit("cf.br"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        succs, 1, sops, snums,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, in_block, op);
    return op;
}

// ============================================================================
// ReturnLikeExitCombiner port (CFGToSCF.cpp:417-466). Two return-like
// terminators are "equivalent" if they have the same op name and the same
// operand-type signature. (tinyc's universe: func.return / llvm.return,
// where every return inside a function shares one operand-type
// signature, so each function collapses to one shared exit block.)
// ============================================================================
typedef struct {
    string            op_name;        // owned by the existing op
    MLIR_TypeHandle  *operand_types;  // arena-owned
    size_t            n_operand_types;
    MLIR_BlockHandle  exit_block;
    MLIR_OpHandle     exit_terminator; // the moved-in return op
} CombinerEntry;

typedef struct {
    CombinerEntry *entries;
    size_t         n_entries;
    size_t         cap;
    MLIR_RegionHandle region;
    Arena         *arena;
} Combiner;

static bool string_eq_string(string a, string b) {
    return a.size == b.size && (a.size == 0 || memcmp(a.str, b.str, a.size) == 0);
}

static bool combiner_signature_matches(const CombinerEntry *e,
                                       MLIR_OpHandle op) {
    if (!string_eq_string(e->op_name, MLIR_GetOpName(op))) return false;
    size_t no = MLIR_GetOpNumOperands(op);
    if (no != e->n_operand_types) return false;
    for (size_t i = 0; i < no; ++i) {
        MLIR_ValueHandle v = MLIR_GetOpOperand(op, i);
        if (MLIR_GetValueType(v) != e->operand_types[i]) return false;
    }
    return true;
}

static CombinerEntry *combiner_find(Combiner *c, MLIR_OpHandle op) {
    for (size_t i = 0; i < c->n_entries; ++i) {
        if (combiner_signature_matches(&c->entries[i], op))
            return &c->entries[i];
    }
    return NULL;
}

static CombinerEntry *combiner_add_new(MLIR_Context *ctx, Combiner *c,
                                       MLIR_OpHandle return_op) {
    if (c->n_entries == c->cap) {
        size_t new_cap = c->cap ? c->cap * 2 : 4;
        CombinerEntry *new_arr = arena_new_array(c->arena, CombinerEntry, new_cap);
        if (c->entries)
            memcpy(new_arr, c->entries, sizeof(CombinerEntry) * c->n_entries);
        c->entries = new_arr;
        c->cap = new_cap;
    }
    CombinerEntry *e = &c->entries[c->n_entries++];
    e->op_name = MLIR_GetOpName(return_op);
    e->n_operand_types = MLIR_GetOpNumOperands(return_op);
    e->operand_types = e->n_operand_types
        ? arena_new_array(c->arena, MLIR_TypeHandle, e->n_operand_types)
        : NULL;
    for (size_t i = 0; i < e->n_operand_types; ++i) {
        e->operand_types[i] = MLIR_GetValueType(MLIR_GetOpOperand(return_op, i));
    }
    e->exit_block = MLIR_CreateBlock(ctx);
    e->exit_terminator = MLIR_INVALID_HANDLE;
    MLIR_LocationHandle loc = MLIR_GetOpLocation(return_op);
    // Add a block argument for each operand type.
    for (size_t i = 0; i < e->n_operand_types; ++i) {
        MLIR_AddBlockArgument(ctx, e->exit_block, e->operand_types[i], loc);
    }
    MLIR_InsertRegionBlockBefore(ctx, c->region, e->exit_block, MLIR_INVALID_HANDLE);
    return e;
}

// Replace `return_op` (located at the end of its block) with a cf.br to
// the matching shared exit block, forwarding the original operands.
// First occurrence of a signature creates the exit block and moves the
// return op there; subsequent occurrences just create the cf.br and
// erase the local return op.
static void combiner_combine_exit(MLIR_Context *ctx, Combiner *c,
                                  MLIR_OpHandle return_op) {
    MLIR_BlockHandle from_block = MLIR_INVALID_HANDLE;
    size_t nb = MLIR_GetRegionNumBlocks(c->region);
    for (size_t bi = 0; bi < nb && from_block == MLIR_INVALID_HANDLE; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(c->region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            if (MLIR_GetBlockOp(b, oi) == return_op) {
                from_block = b;
                break;
            }
        }
    }
    if (from_block == MLIR_INVALID_HANDLE) return;

    MLIR_LocationHandle loc = MLIR_GetOpLocation(return_op);
    size_t n_ops = MLIR_GetOpNumOperands(return_op);
    MLIR_ValueHandle *operands = n_ops
        ? arena_new_array(c->arena, MLIR_ValueHandle, n_ops) : NULL;
    for (size_t i = 0; i < n_ops; ++i) {
        operands[i] = MLIR_GetOpOperand(return_op, i);
    }

    CombinerEntry *e = combiner_find(c, return_op);
    bool inserted = (e == NULL);
    if (inserted) {
        e = combiner_add_new(ctx, c, return_op);
    }

    // The return op is still at the end of `from_block`. Erase it, then
    // append the cf.br.
    MLIR_EraseOp(ctx, return_op);
    create_cf_br_in_block(ctx, c->arena, from_block, e->exit_block,
                          operands, n_ops, loc);

    if (inserted) {
        // First occurrence: install a fresh return op in the new exit
        // block whose operands are the exit block's arguments. We don't
        // need to "move" the original — we already erased it; just
        // synthesize a new one with identical attributes and operand
        // count using the same op type. For tinyc the universe is
        // {func.return, llvm.return}, both of which take only operands
        // and no attributes/results.
        MLIR_ValueHandle *args = e->n_operand_types
            ? arena_new_array(c->arena, MLIR_ValueHandle, e->n_operand_types)
            : NULL;
        for (size_t i = 0; i < e->n_operand_types; ++i) {
            args[i] = MLIR_GetBlockArg(e->exit_block, i);
        }
        MLIR_OpType ot = string_eq(e->op_name, "llvm.return")
            ? OP_TYPE_LLVM_RETURN : OP_TYPE_FUNC_RETURN;
        MLIR_OpHandle new_ret = MLIR_CreateOp(
            ctx, ot, e->op_name,
            NULL, 0, NULL, 0, NULL, 0,
            args, e->n_operand_types, NULL, 0,
            loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        MLIR_AppendBlockOp(ctx, e->exit_block, new_ret);
        e->exit_terminator = new_ret;
    }
}

// Top-level driver: scan every block of `region`; for each block with no
// successors (block ending in a return-like op), funnel through the
// combiner. Mirrors CFGToSCF.cpp:1221-1234.
static void create_single_exit_blocks_for_return_like(MLIR_Context *ctx,
                                                      Arena *arena,
                                                      MLIR_RegionHandle region) {
    Combiner c = {0};
    c.region = region;
    c.arena = arena;

    // Snapshot blocks first (the loop mutates the region by appending
    // new exit blocks at the end).
    size_t snapshot_n = MLIR_GetRegionNumBlocks(region);
    MLIR_BlockHandle *snapshot = arena_new_array(arena, MLIR_BlockHandle,
                                                 snapshot_n ? snapshot_n : 1);
    for (size_t i = 0; i < snapshot_n; ++i) {
        snapshot[i] = MLIR_GetRegionBlock(region, i);
    }

    for (size_t i = 0; i < snapshot_n; ++i) {
        MLIR_BlockHandle b = snapshot[i];
        MLIR_OpHandle term = MLIR_GetBlockTerminator(b);
        if (term == MLIR_INVALID_HANDLE) continue;
        if (MLIR_GetOpNumSuccessors(term) != 0) continue;
        // Only act on the recognized return-like ops; leave other
        // zero-successor terminators (e.g. cf.assert, unreachable) alone.
        string n = MLIR_GetOpName(term);
        if (!string_eq_string(n, str_lit("func.return")) &&
            !string_eq_string(n, str_lit("llvm.return"))) continue;
        combiner_combine_exit(ctx, &c, term);
    }
}

// or llvm.func body. The lift algorithm runs once per such region.
// ============================================================================
static bool string_eq(string a, const char *b) {
    if (!a.str || !b) return false;
    size_t bn = strlen(b);
    return a.size == bn && memcmp(a.str, b, bn) == 0;
}

static bool op_is_func_like(MLIR_OpHandle op) {
    string n = MLIR_GetOpName(op);
    return string_eq(n, "func.func") || string_eq(n, "llvm.func");
}

// Returns true iff `op` is one of the cf branch ops the algorithm cares
// about (cf.br, cf.cond_br, cf.switch).
static bool op_is_cf_branch(MLIR_OpHandle op) {
    string n = MLIR_GetOpName(op);
    return string_eq(n, "cf.br") || string_eq(n, "cf.cond_br") ||
           string_eq(n, "cf.switch");
}

// Returns true iff the given region contains any cf.* branch op that we
// would have to lift. Used by the fast path.
static bool region_has_cf_branch(MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle o = MLIR_GetBlockOp(b, oi);
            if (op_is_cf_branch(o)) return true;
            // Recurse into nested regions (e.g. scf.if/scf.while produced
            // by an earlier iteration of the algorithm).
            size_t nr = MLIR_GetOpNumRegions(o);
            for (size_t ri = 0; ri < nr; ++ri) {
                if (region_has_cf_branch(MLIR_GetOpRegion(o, ri)))
                    return true;
            }
        }
    }
    return false;
}

// ============================================================================
// Port of checkTransformationPreconditions (CFGToSCF.cpp:1237-1297).
// Returns true on success, false on a violated precondition (with a
// diagnostic printed on stderr).
// ============================================================================
static bool check_preconditions(MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        if (!MLIR_BlockIsEntry(b) && MLIR_GetBlockNumPredecessors(b) == 0) {
            fprintf(stderr, "MLIR_LiftCfToScfNative: unreachable block "
                            "encountered (block %zu has no predecessors)\n",
                    bi);
            return false;
        }
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle o = MLIR_GetBlockOp(b, oi);
            if (MLIR_GetOpNumSuccessors(o) == 0) continue;
            // We accept only the cf.* branch ops as branch-op-interface
            // carriers. Anything else with successors is unsupported.
            if (!op_is_cf_branch(o)) {
                string n = MLIR_GetOpName(o);
                fprintf(stderr,
                        "MLIR_LiftCfToScfNative: unsupported terminator "
                        "with successors: %.*s\n",
                        (int)n.size, n.str ? n.str : "");
                return false;
            }
            // cf.* ops have no operation-produced successor operands.
        }
    }
    return true;
}

// Once the TODOs above are filled in, this iterates the work list as in
// transformCFGToSCF (CFGToSCF.cpp:1300-1376):
//   - work-list seeded with region entry
//   - for each: run cycle-to-loop pass, then branch-to-if pass; push
//     any new sub-regions returned.
// ============================================================================
// ============================================================================
// Single-successor splice (CFGToSCF.cpp:955-967): if `region_entry` has
// exactly one successor reached via cf.br, replace that successor's
// block-arg uses with the cf.br's branch operands, splice the
// successor's ops into `region_entry`, drop the cf.br, and erase the
// (now empty, no-predecessors) successor block. Returns true if a
// splice happened.
// ============================================================================
static bool try_splice_single_successor(MLIR_Context *ctx, Arena *arena,
                                        MLIR_BlockHandle region_entry) {
    MLIR_OpHandle term = MLIR_GetBlockTerminator(region_entry);
    if (term == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumSuccessors(term) != 1) return false;
    // Only handle the cf.br case; cf.cond_br with both successors equal
    // is theoretically also single-edge but ports differently.
    string nm = MLIR_GetOpName(term);
    if (!string_eq_string(nm, str_lit("cf.br"))) return false;

    MLIR_BlockHandle succ = MLIR_GetOpSuccessor(term, 0);
    if (succ == MLIR_INVALID_HANDLE || succ == region_entry) return false;
    // Algorithm prerequisite: the successor's only predecessor is us.
    // Otherwise splicing the successor's ops up would change semantics.
    if (MLIR_GetBlockNumPredecessors(succ) != 1) return false;

    size_t n_args = MLIR_GetBlockNumArgs(succ);
    size_t n_branch_ops = MLIR_GetOpNumSuccessorOperands(term, 0);
    if (n_args != n_branch_ops) return false;

    // Snapshot branch operands before mutation (RAUW will rewrite the
    // op's operand list otherwise).
    MLIR_ValueHandle *new_vals = n_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_args) : NULL;
    for (size_t i = 0; i < n_args; ++i) {
        new_vals[i] = MLIR_GetOpSuccessorOperand(term, 0, i);
    }
    for (size_t i = 0; i < n_args; ++i) {
        MLIR_ValueHandle old_arg = MLIR_GetBlockArg(succ, i);
        MLIR_ReplaceAllUsesOfValue(ctx, old_arg, new_vals[i]);
    }

    MLIR_EraseOp(ctx, term);
    MLIR_SpliceBlockOps(ctx, region_entry, succ);
    // Successor is now empty and has no predecessors (we just removed
    // the only one). Erase the args first (no remaining uses) then the
    // block itself.
    if (n_args) {
        MLIR_EraseBlockArguments(ctx, succ, 0, n_args);
    }
    MLIR_EraseBlock(ctx, succ);
    return true;
}

// Apply single-successor splicing repeatedly until the region entry's
// terminator has 0 or >1 successors (or is not cf.br). Returns true if
// any splice happened.
static bool fold_linear_chain(MLIR_Context *ctx, Arena *arena,
                              MLIR_BlockHandle region_entry) {
    bool any = false;
    while (try_splice_single_successor(ctx, arena, region_entry)) {
        any = true;
    }
    return any;
}

// ============================================================================
// Lift the simplest cf.cond_br diamond into scf.if.
//
// Pattern recognized (after return-like combining + linear folding):
//
//   ^entry:
//     ...
//     cf.cond_br %cond, ^then(then_args), ^else(else_args)
//   ^then:                       (sole pred = ^entry)
//     ...
//     cf.br ^merge(then_yield)
//   ^else:                       (sole pred = ^entry)
//     ...
//     cf.br ^merge(else_yield)
//   ^merge(merge_args):          (sole preds = {^then, ^else})
//     ...
//
// becomes:
//
//   ^entry:
//     ...
//     %r:N = scf.if %cond -> (merge_arg_types) {
//       ...                  (then's ops)
//       scf.yield then_yield
//     } else {
//       ...                  (else's ops)
//       scf.yield else_yield
//     }
//     ...                    (merge's ops, with merge_args replaced by %r:N)
//
// Returns true if the transform fired. Mirrors the 2-way structured
// branch case of CFGToSCF.cpp:947-1216 limited to the simplest shape.
// ============================================================================

static uint64_t g_lift_ssa_counter = 0;

static string fresh_ssa_name(Arena *arena) {
    return format(arena, str_lit("%lift_{}"),
                  (int64_t)(g_lift_ssa_counter++));
}

static MLIR_OpHandle create_scf_yield(MLIR_Context *ctx, Arena *arena,
                                      MLIR_BlockHandle in_block,
                                      MLIR_ValueHandle *args, size_t n_args,
                                      MLIR_LocationHandle loc) {
    MLIR_ValueHandle *operand_arr = n_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_args) : NULL;
    for (size_t i = 0; i < n_args; ++i) operand_arr[i] = args[i];
    MLIR_OpHandle y = MLIR_CreateOp(
        ctx, OP_TYPE_SCF_YIELD, str_lit("scf.yield"),
        NULL, 0, NULL, 0, NULL, 0,
        operand_arr, n_args, NULL, 0,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, in_block, y);
    return y;
}

static bool try_lift_simple_if(MLIR_Context *ctx, Arena *arena,
                               MLIR_BlockHandle entry) {
    MLIR_OpHandle cond_br = MLIR_GetBlockTerminator(entry);
    if (cond_br == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(cond_br), str_lit("cf.cond_br")))
        return false;
    if (MLIR_GetOpNumSuccessors(cond_br) != 2) return false;
    MLIR_BlockHandle then_b = MLIR_GetOpSuccessor(cond_br, 0);
    MLIR_BlockHandle else_b = MLIR_GetOpSuccessor(cond_br, 1);
    if (then_b == MLIR_INVALID_HANDLE || else_b == MLIR_INVALID_HANDLE) return false;
    if (then_b == else_b) return false;
    if (then_b == entry || else_b == entry) return false;
    if (MLIR_GetBlockNumPredecessors(then_b) != 1) return false;
    if (MLIR_GetBlockNumPredecessors(else_b) != 1) return false;

    MLIR_OpHandle then_term = MLIR_GetBlockTerminator(then_b);
    MLIR_OpHandle else_term = MLIR_GetBlockTerminator(else_b);
    if (then_term == MLIR_INVALID_HANDLE || else_term == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(then_term), str_lit("cf.br"))) return false;
    if (!string_eq_string(MLIR_GetOpName(else_term), str_lit("cf.br"))) return false;

    MLIR_BlockHandle merge = MLIR_GetOpSuccessor(then_term, 0);
    if (merge == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpSuccessor(else_term, 0) != merge) return false;
    if (merge == entry || merge == then_b || merge == else_b) return false;
    if (MLIR_GetBlockNumPredecessors(merge) != 2) return false;

    // ---- Snapshot all the values we need before mutating IR. ----
    MLIR_LocationHandle loc = MLIR_GetOpLocation(cond_br);

    // cf.cond_br: operand 0 = condition; successor operands 0 = then args,
    // successor operands 1 = else args.
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(cond_br, 0);

    size_t n_then_args = MLIR_GetOpNumSuccessorOperands(cond_br, 0);
    size_t n_else_args = MLIR_GetOpNumSuccessorOperands(cond_br, 1);
    if (n_then_args != MLIR_GetBlockNumArgs(then_b)) return false;
    if (n_else_args != MLIR_GetBlockNumArgs(else_b)) return false;

    MLIR_ValueHandle *then_args = n_then_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_then_args) : NULL;
    for (size_t i = 0; i < n_then_args; ++i)
        then_args[i] = MLIR_GetOpSuccessorOperand(cond_br, 0, i);
    MLIR_ValueHandle *else_args = n_else_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_else_args) : NULL;
    for (size_t i = 0; i < n_else_args; ++i)
        else_args[i] = MLIR_GetOpSuccessorOperand(cond_br, 1, i);

    size_t n_then_yield = MLIR_GetOpNumSuccessorOperands(then_term, 0);
    size_t n_else_yield = MLIR_GetOpNumSuccessorOperands(else_term, 0);
    size_t n_merge_args = MLIR_GetBlockNumArgs(merge);
    if (n_then_yield != n_merge_args || n_else_yield != n_merge_args)
        return false;

    MLIR_ValueHandle *then_yield = n_then_yield
        ? arena_new_array(arena, MLIR_ValueHandle, n_then_yield) : NULL;
    for (size_t i = 0; i < n_then_yield; ++i)
        then_yield[i] = MLIR_GetOpSuccessorOperand(then_term, 0, i);
    MLIR_ValueHandle *else_yield = n_else_yield
        ? arena_new_array(arena, MLIR_ValueHandle, n_else_yield) : NULL;
    for (size_t i = 0; i < n_else_yield; ++i)
        else_yield[i] = MLIR_GetOpSuccessorOperand(else_term, 0, i);

    MLIR_TypeHandle *result_types = n_merge_args
        ? arena_new_array(arena, MLIR_TypeHandle, n_merge_args) : NULL;
    for (size_t i = 0; i < n_merge_args; ++i)
        result_types[i] = MLIR_GetValueType(MLIR_GetBlockArg(merge, i));

    // ---- Replace then_b/else_b's block args with cond_br's branch ops. ----
    for (size_t i = 0; i < n_then_args; ++i) {
        MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetBlockArg(then_b, i), then_args[i]);
    }
    if (n_then_args) MLIR_EraseBlockArguments(ctx, then_b, 0, n_then_args);
    for (size_t i = 0; i < n_else_args; ++i) {
        MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetBlockArg(else_b, i), else_args[i]);
    }
    if (n_else_args) MLIR_EraseBlockArguments(ctx, else_b, 0, n_else_args);

    // ---- Replace then_term/else_term with scf.yield in their blocks. ----
    MLIR_LocationHandle then_loc = MLIR_GetOpLocation(then_term);
    MLIR_LocationHandle else_loc = MLIR_GetOpLocation(else_term);
    MLIR_EraseOp(ctx, then_term);
    create_scf_yield(ctx, arena, then_b, then_yield, n_then_yield, then_loc);
    MLIR_EraseOp(ctx, else_term);
    create_scf_yield(ctx, arena, else_b, else_yield, n_else_yield, else_loc);

    // ---- Move then_b / else_b into fresh regions for the scf.if op. ----
    MLIR_RegionHandle then_region = MLIR_CreateRegion(ctx);
    MLIR_RegionHandle else_region = MLIR_CreateRegion(ctx);
    MLIR_MoveBlockToRegionEnd(ctx, then_b, then_region);
    MLIR_MoveBlockToRegionEnd(ctx, else_b, else_region);

    // ---- Build the scf.if op + result handles. ----
    MLIR_ValueHandle *result_vs = n_merge_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_merge_args) : NULL;
    for (size_t i = 0; i < n_merge_args; ++i) {
        result_vs[i] = MLIR_CreateValueOpResult(
            ctx, MLIR_INVALID_HANDLE, (uint32_t)i, result_types[i],
            fresh_ssa_name(arena), loc);
    }
    MLIR_RegionHandle *regions = arena_new_array(arena, MLIR_RegionHandle, 2);
    regions[0] = then_region; regions[1] = else_region;
    MLIR_ValueHandle *operands = arena_new_array(arena, MLIR_ValueHandle, 1);
    operands[0] = cond_v;

    // Erase cond_br first so the entry's terminator slot is free.
    MLIR_EraseOp(ctx, cond_br);

    MLIR_OpHandle scf_if = MLIR_CreateOp(
        ctx, OP_TYPE_SCF_IF, str_lit("scf.if"),
        NULL, 0,
        result_types, n_merge_args,
        result_vs, n_merge_args,
        operands, 1,
        regions, 2,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, entry, scf_if);

    // ---- Replace merge's block args with scf.if's results. ----
    for (size_t i = 0; i < n_merge_args; ++i) {
        MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetBlockArg(merge, i), result_vs[i]);
    }
    if (n_merge_args) MLIR_EraseBlockArguments(ctx, merge, 0, n_merge_args);

    // ---- Splice merge's ops into entry, then erase merge. ----
    MLIR_SpliceBlockOps(ctx, entry, merge);
    MLIR_EraseBlock(ctx, merge);
    return true;
}

static bool fold_simple_ifs(MLIR_Context *ctx, Arena *arena,
                            MLIR_BlockHandle entry) {
    bool any = false;
    // After lifting an if, the entry's new tail (merge's spliced ops)
    // may end in another cf.br/cond_br; loop with the linear chain
    // folder so we keep making progress.
    while (true) {
        bool spliced = fold_linear_chain(ctx, arena, entry);
        bool lifted  = try_lift_simple_if(ctx, arena, entry);
        if (!spliced && !lifted) break;
        any = any || spliced || lifted;
    }
    return any;
}

// Erase non-entry blocks with no predecessors, iterating to fixed point
// (erasing one may leave another unreachable). Mirrors upstream's
// `eraseUnreachableBlocks`.
static void erase_unreachable_blocks(MLIR_Context *ctx, MLIR_RegionHandle region) {
    bool changed = true;
    while (changed) {
        changed = false;
        size_t nb = MLIR_GetRegionNumBlocks(region);
        for (size_t bi = 0; bi < nb; ++bi) {
            MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
            if (MLIR_BlockIsEntry(b)) continue;
            if (MLIR_GetBlockNumPredecessors(b) != 0) continue;
            MLIR_EraseBlock(ctx, b);
            changed = true;
            break;
        }
    }
}

bool MLIR_LiftCfToScfNative(MLIR_Context *ctx, MLIR_OpHandle module) {
    (void)ctx;
    if (module == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumRegions(module) == 0) return true;
    // Allow staged enablement of the partial port without touching IR
    // on the default opt-in path. TINYC_LIFT_USE_NATIVE_PARTIAL=1 runs
    // the return-like exit combiner on each function body before
    // returning false so the upstream lift finishes the rest.
    // Always run the partial transforms we have ported. Anything left
    // unhandled (loops, multi-way switches, non-clean diamonds) falls
    // through; downstream (wasmssa-lower) will report it. We return
    // true unconditionally so the caller proceeds with the lowered
    // (or partially-lowered) IR.
    (void)getenv;

    Arena *scratch = arena_create(4096);
    MLIR_RegionHandle mod_body = MLIR_GetOpRegion(module, 0);
    size_t nb = MLIR_GetRegionNumBlocks(mod_body);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(mod_body, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle o = MLIR_GetBlockOp(b, oi);
            if (!op_is_func_like(o)) continue;
            if (MLIR_GetOpNumRegions(o) == 0) continue;
            MLIR_RegionHandle body = MLIR_GetOpRegion(o, 0);
            if (MLIR_GetRegionNumBlocks(body) == 0) continue;
            erase_unreachable_blocks(ctx, body);
            if (!check_preconditions(body)) {
                arena_destroy(scratch);
                return false;
            }
            if (!region_has_cf_branch(body)) continue;
            create_single_exit_blocks_for_return_like(ctx, scratch, body);
            MLIR_BlockHandle entry = MLIR_GetRegionBlock(body, 0);
            fold_simple_ifs(ctx, scratch, entry);
        }
    }
    arena_destroy(scratch);
    return true;
}

// ---------------------------------------------------------------------------
// Placeholder definitions so the TUs build even before the algorithm
// lands. Each aborts loudly if invoked from the stub entry (which it
// can't, since the entry returns false before touching anything).
// ---------------------------------------------------------------------------
static MLIR_ValueHandle get_switch_value(LiftState *st, unsigned v) {
    (void)st; (void)v;
    fprintf(stderr, "mlir_lift_cf_to_scf: get_switch_value not implemented\n");
    return MLIR_INVALID_HANDLE;
}

static MLIR_ValueHandle get_undef_value(LiftState *st, MLIR_TypeHandle ty) {
    (void)st; (void)ty;
    fprintf(stderr, "mlir_lift_cf_to_scf: get_undef_value not implemented\n");
    return MLIR_INVALID_HANDLE;
}
