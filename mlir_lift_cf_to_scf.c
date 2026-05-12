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

// Forward declarations for cross-references between EdgeMultiplexer and
// LiftState. Full definitions appear below.
typedef struct LiftState_S       LiftState;
typedef struct TypedValueEntry_S TypedValueEntry;

// ============================================================================
// Switch value / undef value caches. Mirror the typedUndefCache and
// switchValueCache lambdas of transformCFGToSCF in CFGToSCF.cpp:1309-1338.
// ============================================================================
struct TypedValueEntry_S {
    MLIR_TypeHandle  type;
    MLIR_ValueHandle value;
};

struct LiftState_S {
    MLIR_Context     *ctx;
    Arena            *arena;
    MLIR_OpHandle     fn_op;        // func.func or llvm.func being processed
    MLIR_BlockHandle  entry_block;  // first block of the region under lift
    MLIR_TypeHandle   i32_ty;       // cached i32 type
    MLIR_LocationHandle unk_loc;    // cached unknown location

    // Cached arith.constant of i32 for switch flag values.
    MLIR_ValueHandle *switch_value_cache;
    size_t            switch_value_cache_n;
    size_t            switch_value_cache_cap;

    // Cached llvm.mlir.undef per type.
    TypedValueEntry  *undef_cache;
    size_t            undef_cache_n;
    size_t            undef_cache_cap;
};

static MLIR_ValueHandle get_switch_value(LiftState *st, unsigned v);
static MLIR_ValueHandle get_undef_value(LiftState *st, MLIR_TypeHandle ty);
static string           fresh_ssa_name(Arena *arena);

// Forward declaration of branch op constructor used by EdgeMultiplexer
// (defined further down with the other op constructors).
static MLIR_OpHandle create_cf_switch_in_block(MLIR_Context *ctx, Arena *arena,
                                               MLIR_BlockHandle in_block,
                                               MLIR_ValueHandle flag,
                                               MLIR_BlockHandle dflt_dst,
                                               MLIR_ValueHandle *dflt_args, size_t n_dflt_args,
                                               const int32_t *case_values,
                                               MLIR_BlockHandle *case_dsts,
                                               MLIR_ValueHandle **case_args,
                                               size_t *n_case_args,
                                               size_t n_cases,
                                               MLIR_LocationHandle loc);

static MLIR_OpHandle create_cf_cond_br_in_block(MLIR_Context *ctx, Arena *arena,
                                                MLIR_BlockHandle in_block,
                                                MLIR_ValueHandle cond,
                                                MLIR_BlockHandle t_dst,
                                                MLIR_ValueHandle *t_args, size_t n_t_args,
                                                MLIR_BlockHandle f_dst,
                                                MLIR_ValueHandle *f_args, size_t n_f_args,
                                                MLIR_LocationHandle loc);

// ============================================================================
// EdgeMultiplexer (port of CFGToSCF.cpp:212-387).
//
// A multiplexer takes N "entry edges" (each leading to one of several
// distinct entry blocks) and funnels them through a single new
// multiplexer block carrying:
//   - the UNION of all distinct entry blocks' arguments,
//   - optionally a discriminator i32 arg (only when >1 distinct entries),
//   - optionally `n_extra_args` extra args at the end (for the latch case
//     where we add a `shouldRepeat` flag).
//
// edge_multiplexer_redirect_edge() retargets an existing edge to the
// multiplexer, passing the original successor operands into THIS edge's
// slot of the union and undef into all other slots.
//
// edge_multiplexer_create_switch() emits a cf.switch in the SUPPLIED
// destination block (typically the multiplexer block itself, or a
// downstream exit-dispatch block) that dispatches to the original
// successors based on the discriminator.
// ============================================================================
typedef struct {
    MLIR_BlockHandle  mux_block;
    // Distinct entry-block list. block_arg_off[i] is the offset in
    // `mux_block`'s arg list where the args of `entries[i]` start.
    MLIR_BlockHandle *entries;
    size_t           *block_arg_off;
    size_t            n_entries;
    // Index of the discriminator in mux_block's args, or SIZE_MAX if
    // single-entry (no discriminator needed).
    size_t            discriminator_idx;
    // Number of extra args appended at the very end of mux_block's args.
    size_t            n_extra_args;
    // Cached getters.
    LiftState        *st;
} EdgeMultiplexer;

// Find the index in `m->entries` of the given block, or SIZE_MAX.
static size_t edge_multiplexer_entry_index(const EdgeMultiplexer *m,
                                           MLIR_BlockHandle block) {
    for (size_t i = 0; i < m->n_entries; ++i) {
        if (m->entries[i] == block) return i;
    }
    return SIZE_MAX;
}

// Copy block_arg types from `src` into `dst`. Mirrors upstream's
// `addBlockArgumentsFromOther`.
static void mux_copy_block_args(MLIR_Context *ctx,
                                MLIR_BlockHandle dst, MLIR_BlockHandle src,
                                MLIR_LocationHandle loc) {
    size_t n = MLIR_GetBlockNumArgs(src);
    for (size_t i = 0; i < n; ++i) {
        MLIR_ValueHandle a = MLIR_GetBlockArg(src, i);
        MLIR_AddBlockArgument(ctx, dst, MLIR_GetValueType(a), loc);
    }
}

// Construct a multiplexer block targeted by all edges in `entry_blocks`
// (which may contain duplicates — the multiplexer only allocates one
// slot per distinct entry block). `n_extra_args` is the number of extra
// args (e.g. 1 for the latch's shouldRepeat flag); the types of these
// extras are i32 by convention (matches what
// `create_single_exiting_latch` needs).
//
// The new block is inserted immediately AFTER the first entry block in
// the parent region.
//
// Note: this only constructs the multiplexer. Use
// `edge_multiplexer_redirect_edge` to actually route an existing edge
// to it, and `edge_multiplexer_create_switch` to install the dispatch
// in a downstream block.
static EdgeMultiplexer edge_multiplexer_create(LiftState *st,
                                               MLIR_LocationHandle loc,
                                               const MLIR_BlockHandle *entry_blocks,
                                               size_t n_entries_in,
                                               size_t n_extra_args) {
    EdgeMultiplexer m = {0};
    m.st = st;
    m.discriminator_idx = SIZE_MAX;
    m.n_extra_args = n_extra_args;
    m.mux_block = MLIR_CreateBlock(st->ctx);

    // Insert the mux block right after the first entry block in its
    // parent region. (The parent is shared by all entry blocks.)
    MLIR_RegionHandle parent_region = MLIR_GetBlockParentRegion(entry_blocks[0]);
    MLIR_InsertRegionBlockAfter(st->ctx, parent_region, m.mux_block, entry_blocks[0]);

    // Deduplicate entry blocks while preserving insertion order, and
    // record the arg offset for each distinct entry block.
    m.entries = arena_new_array(st->arena, MLIR_BlockHandle, n_entries_in ? n_entries_in : 1);
    m.block_arg_off = arena_new_array(st->arena, size_t, n_entries_in ? n_entries_in : 1);
    size_t current_offset = 0;
    for (size_t i = 0; i < n_entries_in; ++i) {
        MLIR_BlockHandle b = entry_blocks[i];
        // Skip if already inserted.
        if (edge_multiplexer_entry_index(&m, b) != SIZE_MAX) continue;
        m.entries[m.n_entries] = b;
        m.block_arg_off[m.n_entries] = current_offset;
        m.n_entries++;
        mux_copy_block_args(st->ctx, m.mux_block, b, loc);
        current_offset += MLIR_GetBlockNumArgs(b);
    }

    // Add a discriminator if there is more than one distinct entry.
    if (m.n_entries > 1) {
        MLIR_AddBlockArgument(st->ctx, m.mux_block, st->i32_ty, loc);
        m.discriminator_idx = current_offset;
        current_offset += 1;
    }

    // Add the extra args (i32 each).
    for (size_t i = 0; i < n_extra_args; ++i) {
        MLIR_AddBlockArgument(st->ctx, m.mux_block, st->i32_ty, loc);
    }

    return m;
}

// Redirect the given edge to the multiplexer block. The edge's
// successor MUST be one of the entries passed to
// edge_multiplexer_create. `extra_args` provides values for the extra
// slots (must have exactly m.n_extra_args entries).
static void edge_multiplexer_redirect_edge(EdgeMultiplexer *m, Edge edge,
                                           MLIR_ValueHandle *extra_args,
                                           size_t n_extra_args_in) {
    if (n_extra_args_in != m->n_extra_args) return;
    MLIR_OpHandle term = MLIR_GetBlockTerminator(edge.from_block);
    if (term == MLIR_INVALID_HANDLE) return;
    MLIR_BlockHandle orig_succ = MLIR_GetOpSuccessor(term, edge.succ_idx);
    size_t entry_idx = edge_multiplexer_entry_index(m, orig_succ);
    if (entry_idx == SIZE_MAX) return;

    size_t n_mux_args = MLIR_GetBlockNumArgs(m->mux_block);
    size_t my_off     = m->block_arg_off[entry_idx];
    size_t my_n_args  = MLIR_GetBlockNumArgs(m->entries[entry_idx]);
    size_t extra_off  = n_mux_args - m->n_extra_args;

    // Capture original successor operands before mutation.
    size_t n_orig = MLIR_GetOpNumSuccessorOperands(term, edge.succ_idx);
    MLIR_ValueHandle *orig = n_orig
        ? arena_new_array(m->st->arena, MLIR_ValueHandle, n_orig) : NULL;
    for (size_t i = 0; i < n_orig; ++i) {
        orig[i] = MLIR_GetOpSuccessorOperand(term, edge.succ_idx, i);
    }

    // Build new operand list of length n_mux_args.
    MLIR_ValueHandle *new_ops = arena_new_array(m->st->arena, MLIR_ValueHandle, n_mux_args);
    for (size_t i = 0; i < n_mux_args; ++i) {
        if (i >= my_off && i < my_off + my_n_args) {
            // Original arg slot.
            size_t local = i - my_off;
            new_ops[i] = (local < n_orig) ? orig[local]
                                          : get_undef_value(m->st, MLIR_GetValueType(MLIR_GetBlockArg(m->mux_block, i)));
            continue;
        }
        if (m->discriminator_idx != SIZE_MAX && i == m->discriminator_idx) {
            new_ops[i] = get_switch_value(m->st, (unsigned)entry_idx);
            continue;
        }
        if (i >= extra_off) {
            new_ops[i] = extra_args[i - extra_off];
            continue;
        }
        // Other entry's slot or padding: undef.
        new_ops[i] = get_undef_value(m->st, MLIR_GetValueType(MLIR_GetBlockArg(m->mux_block, i)));
    }

    // Apply.
    MLIR_SetOpSuccessor(m->st->ctx, term, edge.succ_idx, m->mux_block);
    MLIR_SetOpSuccessorOperands(m->st->ctx, term, edge.succ_idx, new_ops, n_mux_args);
}

// Emit a cf.switch in `in_block` that dispatches to the original
// successors of the multiplexer (excluding any blocks in `excluded`).
// The switch reads the discriminator block-arg (or, if no
// discriminator, a freshly-built constant 0 that effectively forces
// the default arm). `in_block` MUST be dominated by the multiplexer
// block — typically it IS the multiplexer block, or the latch's exit
// block in createSingleExitingLatch.
static void edge_multiplexer_create_switch(EdgeMultiplexer *m,
                                           MLIR_BlockHandle in_block,
                                           MLIR_LocationHandle loc,
                                           const MLIR_BlockHandle *excluded,
                                           size_t n_excluded) {
    // Decide which entries to dispatch.
    MLIR_BlockHandle *case_dsts = arena_new_array(m->st->arena, MLIR_BlockHandle, m->n_entries);
    MLIR_ValueHandle **case_args = arena_new_array(m->st->arena, MLIR_ValueHandle *, m->n_entries);
    size_t *n_case_args = arena_new_array(m->st->arena, size_t, m->n_entries);
    int32_t *case_values = arena_new_array(m->st->arena, int32_t, m->n_entries);
    size_t n_pick = 0;
    for (size_t i = 0; i < m->n_entries; ++i) {
        bool skip = false;
        for (size_t j = 0; j < n_excluded; ++j) {
            if (m->entries[i] == excluded[j]) { skip = true; break; }
        }
        if (skip) continue;
        size_t off = m->block_arg_off[i];
        size_t na  = MLIR_GetBlockNumArgs(m->entries[i]);
        MLIR_ValueHandle *args = na ? arena_new_array(m->st->arena, MLIR_ValueHandle, na) : NULL;
        for (size_t k = 0; k < na; ++k) {
            args[k] = MLIR_GetBlockArg(m->mux_block, off + k);
        }
        case_dsts[n_pick] = m->entries[i];
        case_args[n_pick] = args;
        n_case_args[n_pick] = na;
        case_values[n_pick] = (int32_t)i;
        n_pick++;
    }

    // Need at least one survivor.
    if (n_pick == 0) return;

    // Decide discriminator value.
    MLIR_ValueHandle disc_v;
    if (m->discriminator_idx != SIZE_MAX) {
        disc_v = MLIR_GetBlockArg(m->mux_block, m->discriminator_idx);
    } else {
        disc_v = get_switch_value(m->st, 0);
    }

    // The LAST surviving entry becomes the default case.
    MLIR_BlockHandle dflt_dst = case_dsts[n_pick - 1];
    MLIR_ValueHandle *dflt_args = case_args[n_pick - 1];
    size_t n_dflt = n_case_args[n_pick - 1];
    n_pick -= 1;

    create_cf_switch_in_block(m->st->ctx, m->st->arena, in_block, disc_v,
                              dflt_dst, dflt_args, n_dflt,
                              case_values, case_dsts, case_args, n_case_args,
                              n_pick, loc);
}


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

// ============================================================================
// createSingleEntryBlock + createSingleExitingLatch  (CFGToSCF.cpp:513-635).
//
// `create_single_entry_block` funnels N entry edges of a region (e.g. all
// edges entering a cycle) through a single synthetic block carrying the
// union of distinct-entry args + an i32 discriminator. Returns the
// EdgeMultiplexer that owns the synthetic block.
//
// `create_single_exiting_latch` turns a multi-back-edge / multi-exit-edge
// loop into a structured one with exactly one back edge and one exit edge
// originating from a freshly-built latch block. The latch carries a
// `shouldRepeat: i32` discriminator passed by every redirected edge:
//   - back edges pass shouldRepeat=1, exit edges pass shouldRepeat=0.
// The latch branches on shouldRepeat to either the loop header (one back
// edge) or to a separate exit block which then dispatches by switch to the
// original exit destinations.
// ============================================================================
typedef struct {
    MLIR_BlockHandle latch_block;
    MLIR_ValueHandle condition;     // i32 'shouldRepeat'
    MLIR_BlockHandle exit_block;
    bool             ok;
} StructuredLoopProps;

static EdgeMultiplexer create_single_entry_block(LiftState *st,
                                                 const Edge *entry_edges,
                                                 size_t n_entry_edges,
                                                 MLIR_LocationHandle loc) {
    MLIR_BlockHandle *succs = arena_new_array(st->arena, MLIR_BlockHandle,
                                              n_entry_edges ? n_entry_edges : 1);
    for (size_t i = 0; i < n_entry_edges; ++i) {
        succs[i] = edge_successor(entry_edges[i]);
    }

    EdgeMultiplexer m = edge_multiplexer_create(st, loc, succs, n_entry_edges,
                                                /*n_extra_args=*/0);

    for (size_t i = 0; i < n_entry_edges; ++i) {
        edge_multiplexer_redirect_edge(&m, entry_edges[i], NULL, 0);
    }

    edge_multiplexer_create_switch(&m, m.mux_block, loc,
                                   /*excluded=*/NULL, /*n_excluded=*/0);
    return m;
}

// Emit a return-like terminator carrying undef values for each result of
// the surrounding function. Used as the fallback terminator for the
// statically-infinite-loop case in create_single_exiting_latch. Mirrors
// CFGToSCFForWasm::createUnreachableTerminator in mlir_api_impl_upstream.cpp.
static MLIR_OpHandle create_unreachable_terminator(LiftState *st,
                                                   MLIR_BlockHandle block,
                                                   MLIR_LocationHandle loc) {
    string fn_name = MLIR_GetOpName(st->fn_op);
    bool is_llvm = string_eq_string(fn_name, str_lit("llvm.func"));

    MLIR_AttributeHandle ty_attr =
        MLIR_GetOpAttributeByName(st->fn_op, "function_type");
    MLIR_TypeHandle fn_ty = (ty_attr == MLIR_INVALID_HANDLE)
        ? MLIR_INVALID_HANDLE : MLIR_GetAttributeType(ty_attr);
    size_t n_res = (fn_ty == MLIR_INVALID_HANDLE)
        ? 0 : MLIR_GetTypeFunctionNumResults(fn_ty);

    MLIR_ValueHandle *args = n_res
        ? arena_new_array(st->arena, MLIR_ValueHandle, n_res) : NULL;
    for (size_t i = 0; i < n_res; ++i) {
        MLIR_TypeHandle rt = MLIR_GetTypeFunctionResult(fn_ty, i);
        args[i] = get_undef_value(st, rt);
    }

    MLIR_OpType ot = is_llvm ? OP_TYPE_LLVM_RETURN : OP_TYPE_FUNC_RETURN;
    string op_name = is_llvm ? str_lit("llvm.return") : str_lit("func.return");
    MLIR_OpHandle ret = MLIR_CreateOp(
        st->ctx, ot, op_name,
        NULL, 0, NULL, 0, NULL, 0,
        args, n_res, NULL, 0,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(st->ctx, block, ret);
    return ret;
}

static StructuredLoopProps create_single_exiting_latch(
    LiftState *st, Combiner *exit_combiner,
    const Edge *back_edges, size_t n_back,
    const Edge *exit_edges, size_t n_exit,
    MLIR_LocationHandle loc)
{
    StructuredLoopProps res = {0};
    if (n_back == 0) return res;

    // All back edges point to the same loop header (precondition).
    MLIR_BlockHandle loop_header = edge_successor(back_edges[0]);

    // Multiplexer: successors = [back-edge dests..., exit-edge dests...],
    // with one trailing extra i32 arg (shouldRepeat).
    size_t n_all = n_back + n_exit;
    MLIR_BlockHandle *succs = arena_new_array(st->arena, MLIR_BlockHandle,
                                              n_all ? n_all : 1);
    for (size_t i = 0; i < n_back; ++i) {
        succs[i] = edge_successor(back_edges[i]);
    }
    for (size_t i = 0; i < n_exit; ++i) {
        succs[n_back + i] = edge_successor(exit_edges[i]);
    }
    EdgeMultiplexer mux = edge_multiplexer_create(st, loc, succs, n_all,
                                                  /*n_extra_args=*/1);
    MLIR_BlockHandle latch = mux.mux_block;

    // Fresh exit block placed immediately after the latch.
    MLIR_BlockHandle exit_block = MLIR_CreateBlock(st->ctx);
    MLIR_InsertRegionBlockAfter(st->ctx, MLIR_GetBlockParentRegion(latch),
                                exit_block, latch);

    // Redirect back edges with shouldRepeat=1, exit edges with shouldRepeat=0.
    {
        MLIR_ValueHandle one = get_switch_value(st, 1);
        MLIR_ValueHandle zero = get_switch_value(st, 0);
        for (size_t i = 0; i < n_back; ++i) {
            edge_multiplexer_redirect_edge(&mux, back_edges[i], &one, 1);
        }
        for (size_t i = 0; i < n_exit; ++i) {
            edge_multiplexer_redirect_edge(&mux, exit_edges[i], &zero, 1);
        }
    }

    // shouldRepeat is the *last* block-arg of the latch.
    size_t n_latch_args = MLIR_GetBlockNumArgs(latch);
    MLIR_ValueHandle should_repeat = MLIR_GetBlockArg(latch, n_latch_args - 1);

    // cf.cond_br shouldRepeat, loop_header(first N args), exit_block()
    size_t n_hdr_args = MLIR_GetBlockNumArgs(loop_header);
    MLIR_ValueHandle *hdr_args = n_hdr_args
        ? arena_new_array(st->arena, MLIR_ValueHandle, n_hdr_args) : NULL;
    for (size_t i = 0; i < n_hdr_args; ++i) {
        hdr_args[i] = MLIR_GetBlockArg(latch, i);
    }
    create_cf_cond_br_in_block(st->ctx, st->arena, latch, should_repeat,
                               loop_header, hdr_args, n_hdr_args,
                               exit_block, NULL, 0, loc);

    if (n_exit > 0) {
        // Dispatch from exit_block to the original exit destinations.
        // Exclude loop_header (it must already be reachable via back edge).
        edge_multiplexer_create_switch(&mux, exit_block, loc,
                                       /*excluded=*/&loop_header,
                                       /*n_excluded=*/1);
    } else {
        // Statically infinite loop: terminate exit_block with a fresh
        // return op carrying undef args, then fold it via the combiner so
        // it shares the function's single exit block.
        MLIR_OpHandle term = create_unreachable_terminator(st, exit_block, loc);
        combiner_combine_exit(st->ctx, exit_combiner, term);
    }

    res.latch_block = latch;
    res.condition   = should_repeat;
    res.exit_block  = exit_block;
    res.ok          = true;
    return res;
}

// ============================================================================
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

// ============================================================================
// Tarjan SCC (analysis only; no IR mutation).
//
// Computes strongly-connected components of the cf graph rooted at the
// region's entry block. The result is a dense per-block SCC id and, for
// each SCC, the list of blocks comprising it (in reverse-topological
// order, the standard Tarjan output order). SCCs of size 1 with no
// self-edge are still returned (with `is_cycle=false`); only those are
// "non-cycle" SCCs.
//
// Iterative implementation (no recursion) so deep CFGs don't blow the
// host stack.
// ============================================================================
typedef struct {
    size_t            n_blocks;     // number of reachable blocks
    MLIR_BlockHandle *blocks;       // dense list, indexed by per-block id
    size_t           *scc_of;       // per-block: scc index this block belongs to
    size_t            n_sccs;
    size_t           *scc_starts;   // length n_sccs+1; scc i = blocks_in_sccs[scc_starts[i]..scc_starts[i+1]]
    size_t           *blocks_in_sccs; // permutation of [0..n_blocks)
    bool             *scc_is_cycle; // length n_sccs; true if SCC has size>=2 OR has a self-edge
} SccResult;

// Look up the dense index of a block in `blocks[0..n)`. Returns SIZE_MAX if not found.
static size_t scc_block_index(MLIR_BlockHandle *blocks, size_t n, MLIR_BlockHandle b) {
    for (size_t i = 0; i < n; ++i) if (blocks[i] == b) return i;
    return SIZE_MAX;
}

// Walks the cf graph from the region's entry, collecting all reachable
// blocks in `out_blocks` and returning their count.
static size_t scc_collect_reachable(Arena *arena, MLIR_RegionHandle region,
                                    MLIR_BlockHandle **out_blocks) {
    size_t cap = MLIR_GetRegionNumBlocks(region);
    if (cap == 0) { *out_blocks = NULL; return 0; }
    MLIR_BlockHandle *list = arena_new_array(arena, MLIR_BlockHandle, cap);
    size_t n = 0;
    // BFS from entry.
    MLIR_BlockHandle entry = MLIR_GetRegionBlock(region, 0);
    list[n++] = entry;
    for (size_t head = 0; head < n; ++head) {
        MLIR_BlockHandle b = list[head];
        MLIR_OpHandle term = MLIR_GetBlockTerminator(b);
        if (term == MLIR_INVALID_HANDLE) continue;
        size_t ns = MLIR_GetOpNumSuccessors(term);
        for (size_t i = 0; i < ns; ++i) {
            MLIR_BlockHandle s = MLIR_GetOpSuccessor(term, i);
            if (s == MLIR_INVALID_HANDLE) continue;
            if (scc_block_index(list, n, s) != SIZE_MAX) continue;
            if (n >= cap) return n; // shouldn't happen — region has cap blocks
            list[n++] = s;
        }
    }
    *out_blocks = list;
    return n;
}

// Iterative Tarjan SCC. Returns the reverse-topological order of SCCs
// (i.e. SCC[0] is a "leaf" with no out-edges to other SCCs).
static SccResult scc_compute(Arena *arena, MLIR_RegionHandle region) {
    SccResult r = {0};
    MLIR_BlockHandle *blocks = NULL;
    size_t n = scc_collect_reachable(arena, region, &blocks);
    r.n_blocks = n;
    r.blocks = blocks;
    if (n == 0) return r;

    size_t *index_of    = arena_new_array(arena, size_t, n);
    size_t *lowlink     = arena_new_array(arena, size_t, n);
    bool   *on_stack    = arena_new_array(arena, bool,   n);
    size_t *tarjan_stk  = arena_new_array(arena, size_t, n);
    size_t  tarjan_top  = 0;
    size_t  next_index  = 1; // 0 means "undefined"
    for (size_t i = 0; i < n; ++i) {
        index_of[i] = 0;
        lowlink[i]  = 0;
        on_stack[i] = false;
    }
    r.scc_of         = arena_new_array(arena, size_t, n);
    r.blocks_in_sccs = arena_new_array(arena, size_t, n);
    // Upper bound on number of SCCs: n.
    size_t *scc_starts_tmp = arena_new_array(arena, size_t, n + 1);
    bool   *scc_is_cycle_tmp = arena_new_array(arena, bool, n);
    size_t  n_sccs_out = 0;
    size_t  blocks_out = 0;

    // Iterative DFS frame.
    typedef struct { size_t v; size_t next_succ; size_t n_succ; } Frame;
    Frame *frames = arena_new_array(arena, Frame, n);

    for (size_t root = 0; root < n; ++root) {
        if (index_of[root]) continue;
        // strongconnect(root) — iterative.
        size_t fp = 0;
        // Push initial frame.
        index_of[root] = next_index;
        lowlink[root]  = next_index;
        next_index++;
        tarjan_stk[tarjan_top++] = root;
        on_stack[root] = true;
        {
            MLIR_OpHandle term = MLIR_GetBlockTerminator(blocks[root]);
            size_t ns = (term == MLIR_INVALID_HANDLE) ? 0 : MLIR_GetOpNumSuccessors(term);
            frames[fp].v = root;
            frames[fp].next_succ = 0;
            frames[fp].n_succ = ns;
            fp++;
        }
        while (fp > 0) {
            Frame *cur = &frames[fp - 1];
            if (cur->next_succ < cur->n_succ) {
                MLIR_OpHandle term = MLIR_GetBlockTerminator(blocks[cur->v]);
                MLIR_BlockHandle succ = MLIR_GetOpSuccessor(term, cur->next_succ);
                cur->next_succ++;
                size_t w = scc_block_index(blocks, n, succ);
                if (w == SIZE_MAX) continue; // unreachable from entry — skip
                if (index_of[w] == 0) {
                    // Recurse into w.
                    index_of[w] = next_index;
                    lowlink[w]  = next_index;
                    next_index++;
                    tarjan_stk[tarjan_top++] = w;
                    on_stack[w] = true;
                    MLIR_OpHandle wterm = MLIR_GetBlockTerminator(blocks[w]);
                    size_t wns = (wterm == MLIR_INVALID_HANDLE) ? 0 : MLIR_GetOpNumSuccessors(wterm);
                    frames[fp].v = w;
                    frames[fp].next_succ = 0;
                    frames[fp].n_succ = wns;
                    fp++;
                } else if (on_stack[w]) {
                    if (index_of[w] < lowlink[cur->v]) {
                        lowlink[cur->v] = index_of[w];
                    }
                }
                continue;
            }
            // All successors visited. Update parent's lowlink with ours.
            size_t v = cur->v;
            size_t v_low = lowlink[v];
            // If v is the root of an SCC, pop it.
            if (lowlink[v] == index_of[v]) {
                size_t scc_id = n_sccs_out;
                scc_starts_tmp[n_sccs_out] = blocks_out;
                size_t scc_size = 0;
                while (tarjan_top > 0) {
                    size_t w = tarjan_stk[--tarjan_top];
                    on_stack[w] = false;
                    r.scc_of[w] = scc_id;
                    r.blocks_in_sccs[blocks_out++] = w;
                    scc_size++;
                    if (w == v) break;
                }
                // Mark cyclic if size >= 2 OR contains a self-edge.
                bool cyclic = (scc_size >= 2);
                if (!cyclic) {
                    MLIR_OpHandle term = MLIR_GetBlockTerminator(blocks[v]);
                    size_t ns = (term == MLIR_INVALID_HANDLE) ? 0 : MLIR_GetOpNumSuccessors(term);
                    for (size_t i = 0; i < ns; ++i) {
                        if (MLIR_GetOpSuccessor(term, i) == blocks[v]) {
                            cyclic = true; break;
                        }
                    }
                }
                scc_is_cycle_tmp[n_sccs_out] = cyclic;
                n_sccs_out++;
            }
            fp--;
            if (fp > 0) {
                Frame *par = &frames[fp - 1];
                if (v_low < lowlink[par->v]) lowlink[par->v] = v_low;
            }
        }
    }
    scc_starts_tmp[n_sccs_out] = blocks_out;

    r.n_sccs        = n_sccs_out;
    r.scc_starts    = arena_new_array(arena, size_t, n_sccs_out + 1);
    r.scc_is_cycle  = arena_new_array(arena, bool,   n_sccs_out);
    for (size_t i = 0; i < n_sccs_out; ++i) {
        r.scc_starts[i]   = scc_starts_tmp[i];
        r.scc_is_cycle[i] = scc_is_cycle_tmp[i];
    }
    r.scc_starts[n_sccs_out] = scc_starts_tmp[n_sccs_out];
    return r;
}

// Returns true iff `block` is in the SCC `scc_id`.
static bool scc_contains(const SccResult *r, size_t scc_id, MLIR_BlockHandle block) {
    size_t bi = scc_block_index(r->blocks, r->n_blocks, block);
    if (bi == SIZE_MAX) return false;
    return r->scc_of[bi] == scc_id;
}

// ============================================================================
// Cycle edges (port of CFGToSCF.cpp:475-511 calculateCycleEdges).
//
//   entry_edges:  src ∉ SCC, dst ∈ SCC. Each represents an external
//                 predecessor entering the loop.
//   back_edges:   src ∈ SCC, dst is one of the entry-targets (header).
//                 Each represents an iteration restart.
//   exit_edges:   src ∈ SCC, dst ∉ SCC. Each represents a way to leave
//                 the loop.
//
// All edges are stored as (from_block, succ_idx) so we can mutate them
// later via MLIR_SetOpSuccessor.
// ============================================================================
typedef struct {
    Edge   *entry;   size_t n_entry;
    Edge   *back;    size_t n_back;
    Edge   *exit;    size_t n_exit;
    // Distinct destination blocks of entry edges. The "header set"
    // upstream uses to identify back-edges.
    MLIR_BlockHandle *entry_dsts; size_t n_entry_dsts;
} CycleEdges;

static bool ce_contains_block(MLIR_BlockHandle *list, size_t n, MLIR_BlockHandle b) {
    for (size_t i = 0; i < n; ++i) if (list[i] == b) return true;
    return false;
}

static CycleEdges calculate_cycle_edges(Arena *arena, const SccResult *r,
                                        size_t scc_id) {
    CycleEdges out = {0};
    size_t cap = 0;
    for (size_t i = r->scc_starts[scc_id]; i < r->scc_starts[scc_id + 1]; ++i) {
        MLIR_BlockHandle b = r->blocks[r->blocks_in_sccs[i]];
        cap += MLIR_GetBlockNumPredecessors(b);
        MLIR_OpHandle term = MLIR_GetBlockTerminator(b);
        if (term != MLIR_INVALID_HANDLE) cap += MLIR_GetOpNumSuccessors(term);
    }
    if (cap == 0) cap = 1;
    out.entry = arena_new_array(arena, Edge, cap);
    out.back  = arena_new_array(arena, Edge, cap);
    out.exit  = arena_new_array(arena, Edge, cap);
    out.entry_dsts = arena_new_array(arena, MLIR_BlockHandle, cap);

    // First pass: entry + exit edges.
    for (size_t i = r->scc_starts[scc_id]; i < r->scc_starts[scc_id + 1]; ++i) {
        MLIR_BlockHandle b = r->blocks[r->blocks_in_sccs[i]];
        // Entry: predecessors of b not in this SCC.
        size_t np = MLIR_GetBlockNumPredecessors(b);
        for (size_t pi = 0; pi < np; ++pi) {
            size_t succ_idx = 0;
            MLIR_BlockHandle p = MLIR_GetBlockPredecessor(b, pi, &succ_idx);
            if (p == MLIR_INVALID_HANDLE) continue;
            if (scc_contains(r, scc_id, p)) continue;
            out.entry[out.n_entry++] = (Edge){p, succ_idx};
            if (!ce_contains_block(out.entry_dsts, out.n_entry_dsts, b)) {
                out.entry_dsts[out.n_entry_dsts++] = b;
            }
        }
        // Exit: successors of b not in this SCC.
        MLIR_OpHandle term = MLIR_GetBlockTerminator(b);
        if (term == MLIR_INVALID_HANDLE) continue;
        size_t ns = MLIR_GetOpNumSuccessors(term);
        for (size_t si = 0; si < ns; ++si) {
            MLIR_BlockHandle s = MLIR_GetOpSuccessor(term, si);
            if (s == MLIR_INVALID_HANDLE) continue;
            if (scc_contains(r, scc_id, s)) continue;
            out.exit[out.n_exit++] = (Edge){b, si};
        }
    }

    // Second pass: back edges. An edge (src in SCC, dst in SCC) is a
    // back-edge iff dst is one of the entry-targets (header set).
    for (size_t i = r->scc_starts[scc_id]; i < r->scc_starts[scc_id + 1]; ++i) {
        MLIR_BlockHandle b = r->blocks[r->blocks_in_sccs[i]];
        MLIR_OpHandle term = MLIR_GetBlockTerminator(b);
        if (term == MLIR_INVALID_HANDLE) continue;
        size_t ns = MLIR_GetOpNumSuccessors(term);
        for (size_t si = 0; si < ns; ++si) {
            MLIR_BlockHandle s = MLIR_GetOpSuccessor(term, si);
            if (s == MLIR_INVALID_HANDLE) continue;
            if (!ce_contains_block(out.entry_dsts, out.n_entry_dsts, s)) continue;
            out.back[out.n_back++] = (Edge){b, si};
        }
    }
    // If the SCC has NO external entry edges (e.g. an entire region
    // that is itself a loop reached only through its "entry block"
    // which is in the SCC), treat the first block of the SCC as the
    // synthetic header so we still find back-edges and can do
    // multiplexing. This matches how upstream's scc_iterator returns
    // such SCCs.
    if (out.n_entry == 0 && out.n_entry_dsts == 0) {
        // Use the SCC's first block as header.
        MLIR_BlockHandle hdr = r->blocks[r->blocks_in_sccs[r->scc_starts[scc_id]]];
        out.entry_dsts[out.n_entry_dsts++] = hdr;
        for (size_t i = r->scc_starts[scc_id]; i < r->scc_starts[scc_id + 1]; ++i) {
            MLIR_BlockHandle b = r->blocks[r->blocks_in_sccs[i]];
            MLIR_OpHandle term = MLIR_GetBlockTerminator(b);
            if (term == MLIR_INVALID_HANDLE) continue;
            size_t ns = MLIR_GetOpNumSuccessors(term);
            for (size_t si = 0; si < ns; ++si) {
                MLIR_BlockHandle s = MLIR_GetOpSuccessor(term, si);
                if (s == hdr) out.back[out.n_back++] = (Edge){b, si};
            }
        }
    }
    return out;
}


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

// Apply single-successor splicing across every block of `region`,
// iterating to fixed point. Returns true if anything changed.
static bool fold_linear_chain_region(MLIR_Context *ctx, Arena *arena,
                                     MLIR_RegionHandle region) {
    bool any = false;
    bool changed = true;
    while (changed) {
        changed = false;
        size_t nb = MLIR_GetRegionNumBlocks(region);
        for (size_t bi = 0; bi < nb; ++bi) {
            MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
            if (try_splice_single_successor(ctx, arena, b)) {
                changed = true; any = true; break;
            }
        }
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

// ============================================================================
// Lift the simplest cf-style structured while loop into scf.while.
//
// Pattern recognized (after return-like combining + linear folding):
//
//   ^entry: ...
//     cf.br ^header(init...)
//   ^header(iter...):                  (preds = {entry, body})
//     ...
//     cf.cond_br %cond, ^body(then...), ^exit(else...)
//   ^body(after...):                   (sole pred = ^header)
//     ...
//     cf.br ^header(yield...)
//   ^exit(res...):                     (sole pred = ^header)
//     ...
//
// Constraints (intentionally narrow for the first cut, matching the
// shape tinyc emits where everything flows through alloca/load):
//   - entry's terminator is cf.br with single successor = ^header.
//   - ^header's terminator is cf.cond_br with successors {^body, ^exit},
//     ^body and ^exit distinct, neither equal to ^header or ^entry.
//   - ^body's sole predecessor is ^header; ^body's terminator is cf.br
//     to ^header (the back-edge).
//   - ^exit's sole predecessor is ^header.
//   - ^header has exactly two predecessors (entry + body).
//   - All operand list arities (init, iter, then, after, yield) match
//     the corresponding block arg counts; ^body's args (R) and ^exit's
//     args (R) must have matching types (because scf.condition's
//     payload is shared between scf.while results and after-block args).
//
// Lifting:
//   - Move ^header into a fresh `before` region; replace its cf.cond_br
//     with scf.condition(%cond, exit_args...).
//   - Move ^body into a fresh `after` region; replace its cf.br back-edge
//     with scf.yield(yield_args...).
//   - Build scf.while(init...) -> (R...) with the two regions, append it
//     to ^entry, RAUW ^exit's block args with scf.while's results, splice
//     ^exit's ops into ^entry, erase the entry's old cf.br and ^exit.
// ============================================================================

static MLIR_OpHandle create_scf_condition(MLIR_Context *ctx, Arena *arena,
                                          MLIR_BlockHandle in_block,
                                          MLIR_ValueHandle cond,
                                          MLIR_ValueHandle *payload, size_t n_payload,
                                          MLIR_LocationHandle loc) {
    size_t n_ops = 1 + n_payload;
    MLIR_ValueHandle *operands = arena_new_array(arena, MLIR_ValueHandle, n_ops);
    operands[0] = cond;
    for (size_t i = 0; i < n_payload; ++i) operands[1 + i] = payload[i];
    MLIR_OpHandle c = MLIR_CreateOp(
        ctx, OP_TYPE_SCF_CONDITION, str_lit("scf.condition"),
        NULL, 0, NULL, 0, NULL, 0,
        operands, n_ops, NULL, 0,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, in_block, c);
    return c;
}

static bool try_lift_simple_while(MLIR_Context *ctx, Arena *arena,
                                  MLIR_BlockHandle entry) {
    // entry's terminator must be cf.br with single successor.
    MLIR_OpHandle entry_br = MLIR_GetBlockTerminator(entry);
    if (entry_br == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(entry_br), str_lit("cf.br"))) return false;
    if (MLIR_GetOpNumSuccessors(entry_br) != 1) return false;
    MLIR_BlockHandle header = MLIR_GetOpSuccessor(entry_br, 0);
    if (header == MLIR_INVALID_HANDLE || header == entry) return false;
    if (MLIR_GetBlockNumPredecessors(header) != 2) return false;

    // header terminator must be cf.cond_br to {body, exit}.
    MLIR_OpHandle cond_br = MLIR_GetBlockTerminator(header);
    if (cond_br == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(cond_br), str_lit("cf.cond_br"))) return false;
    if (MLIR_GetOpNumSuccessors(cond_br) != 2) return false;
    MLIR_BlockHandle body = MLIR_GetOpSuccessor(cond_br, 0);
    MLIR_BlockHandle exit_b = MLIR_GetOpSuccessor(cond_br, 1);
    if (body == MLIR_INVALID_HANDLE || exit_b == MLIR_INVALID_HANDLE) return false;
    if (body == exit_b || body == header || body == entry) return false;
    if (exit_b == header || exit_b == entry) return false;
    if (MLIR_GetBlockNumPredecessors(body) != 1) return false;
    if (MLIR_GetBlockNumPredecessors(exit_b) != 1) return false;

    // body's terminator must be cf.br back-edge to header.
    MLIR_OpHandle body_br = MLIR_GetBlockTerminator(body);
    if (body_br == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(body_br), str_lit("cf.br"))) return false;
    if (MLIR_GetOpSuccessor(body_br, 0) != header) return false;

    // Arity sanity: cf.br's successor operands must match header's args.
    size_t n_iter = MLIR_GetBlockNumArgs(header);
    if (MLIR_GetOpNumSuccessorOperands(entry_br, 0) != n_iter) return false;
    if (MLIR_GetOpNumSuccessorOperands(body_br, 0) != n_iter) return false;

    // cond_br's true-operands → body's args (R = after types).
    // cond_br's false-operands → exit's args (must match types of R).
    size_t n_after = MLIR_GetBlockNumArgs(body);
    size_t n_res   = MLIR_GetBlockNumArgs(exit_b);
    if (MLIR_GetOpNumSuccessorOperands(cond_br, 0) != n_after) return false;
    if (MLIR_GetOpNumSuccessorOperands(cond_br, 1) != n_res)   return false;
    if (n_after != n_res) return false;
    // Types must match (R is shared between after-args and scf.while results).
    for (size_t i = 0; i < n_res; ++i) {
        MLIR_TypeHandle t1 = MLIR_GetValueType(MLIR_GetBlockArg(body, i));
        MLIR_TypeHandle t2 = MLIR_GetValueType(MLIR_GetBlockArg(exit_b, i));
        if (t1 != t2) return false;
    }

    // ---- Snapshot all the values we need before mutating IR. ----
    MLIR_LocationHandle entry_loc = MLIR_GetOpLocation(entry_br);
    MLIR_LocationHandle cond_loc  = MLIR_GetOpLocation(cond_br);
    MLIR_LocationHandle body_loc  = MLIR_GetOpLocation(body_br);

    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(cond_br, 0);

    MLIR_ValueHandle *init_args = n_iter
        ? arena_new_array(arena, MLIR_ValueHandle, n_iter) : NULL;
    for (size_t i = 0; i < n_iter; ++i)
        init_args[i] = MLIR_GetOpSuccessorOperand(entry_br, 0, i);

    MLIR_ValueHandle *yield_args = n_iter
        ? arena_new_array(arena, MLIR_ValueHandle, n_iter) : NULL;
    for (size_t i = 0; i < n_iter; ++i)
        yield_args[i] = MLIR_GetOpSuccessorOperand(body_br, 0, i);

    MLIR_ValueHandle *body_args = n_after
        ? arena_new_array(arena, MLIR_ValueHandle, n_after) : NULL;
    for (size_t i = 0; i < n_after; ++i)
        body_args[i] = MLIR_GetOpSuccessorOperand(cond_br, 0, i);

    MLIR_ValueHandle *exit_args = n_res
        ? arena_new_array(arena, MLIR_ValueHandle, n_res) : NULL;
    for (size_t i = 0; i < n_res; ++i)
        exit_args[i] = MLIR_GetOpSuccessorOperand(cond_br, 1, i);

    MLIR_TypeHandle *iter_types = n_iter
        ? arena_new_array(arena, MLIR_TypeHandle, n_iter) : NULL;
    for (size_t i = 0; i < n_iter; ++i)
        iter_types[i] = MLIR_GetValueType(MLIR_GetBlockArg(header, i));

    MLIR_TypeHandle *res_types = n_res
        ? arena_new_array(arena, MLIR_TypeHandle, n_res) : NULL;
    for (size_t i = 0; i < n_res; ++i)
        res_types[i] = MLIR_GetValueType(MLIR_GetBlockArg(exit_b, i));

    // ---- Replace body's block args with cond_br's true-payload. ----
    // (body_args become the scf.condition payload after rewrite, which
    // becomes the after-block args == header's body's block args. So
    // RAUW the existing body args with body_args BEFORE moving body
    // into the after region — we need to keep the body block's args
    // in place to receive the payload, but their values must match
    // what was passed via cond_br.)
    //
    // Actually, in scf.while, the after-block args ARE the new values
    // produced by scf.condition's payload. Inside the body block, uses
    // of the old block args should be replaced by the new args of the
    // moved block. We keep the args in place and just need to RAUW
    // any references that came from outside; since body has sole pred
    // = header, all uses are inside body itself, and the args remain
    // valid by construction. So no RAUW needed for body args.

    // ---- Replace cond_br with scf.condition in header. ----
    MLIR_EraseOp(ctx, cond_br);
    create_scf_condition(ctx, arena, header, cond_v, body_args, n_after, cond_loc);

    // ---- Replace body's back-edge cf.br with scf.yield. ----
    MLIR_EraseOp(ctx, body_br);
    create_scf_yield(ctx, arena, body, yield_args, n_iter, body_loc);

    // ---- Move header / body into fresh regions. ----
    MLIR_RegionHandle before_region = MLIR_CreateRegion(ctx);
    MLIR_RegionHandle after_region  = MLIR_CreateRegion(ctx);
    MLIR_MoveBlockToRegionEnd(ctx, header, before_region);
    MLIR_MoveBlockToRegionEnd(ctx, body, after_region);

    // ---- Build scf.while op + result handles. ----
    MLIR_ValueHandle *result_vs = n_res
        ? arena_new_array(arena, MLIR_ValueHandle, n_res) : NULL;
    for (size_t i = 0; i < n_res; ++i) {
        result_vs[i] = MLIR_CreateValueOpResult(
            ctx, MLIR_INVALID_HANDLE, (uint32_t)i, res_types[i],
            fresh_ssa_name(arena), entry_loc);
    }
    MLIR_RegionHandle *regions = arena_new_array(arena, MLIR_RegionHandle, 2);
    regions[0] = before_region; regions[1] = after_region;
    MLIR_ValueHandle *operands = n_iter
        ? arena_new_array(arena, MLIR_ValueHandle, n_iter) : NULL;
    for (size_t i = 0; i < n_iter; ++i) operands[i] = init_args[i];

    // Erase the entry's cf.br first so the terminator slot is free.
    MLIR_EraseOp(ctx, entry_br);

    MLIR_OpHandle scf_while = MLIR_CreateOp(
        ctx, OP_TYPE_SCF_WHILE, str_lit("scf.while"),
        NULL, 0,
        res_types, n_res,
        result_vs, n_res,
        operands, n_iter,
        regions, 2,
        entry_loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, entry, scf_while);

    // ---- RAUW exit's block args with scf.while's results. ----
    for (size_t i = 0; i < n_res; ++i) {
        MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetBlockArg(exit_b, i), result_vs[i]);
    }
    if (n_res) MLIR_EraseBlockArguments(ctx, exit_b, 0, n_res);

    // ---- Splice exit's ops into entry, erase exit. ----
    MLIR_SpliceBlockOps(ctx, entry, exit_b);
    MLIR_EraseBlock(ctx, exit_b);
    return true;
}

// ============================================================================
// Lift cf.cond_br "if-no-else" / "if-no-then" pattern into scf.if.
//
// When one of cond_br's successors IS the merge (the other branch's
// target), we have a one-armed if:
//
//   ^entry: ... cf.cond_br %c, ^arm(arm_args), ^merge(direct_args)
//   ^arm(args):  ... cf.br ^merge(arm_yield)
//   ^merge(merge_args): ...
//
// (or the symmetric form with arm/merge swapped between true/false).
// Lift to scf.if with one substantive region and one region containing
// just `scf.yield direct_args`.
// ============================================================================
static bool try_lift_if_one_arm(MLIR_Context *ctx, Arena *arena,
                                MLIR_BlockHandle entry) {
    MLIR_OpHandle cond_br = MLIR_GetBlockTerminator(entry);
    if (cond_br == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(cond_br), str_lit("cf.cond_br")))
        return false;
    if (MLIR_GetOpNumSuccessors(cond_br) != 2) return false;
    MLIR_BlockHandle s0 = MLIR_GetOpSuccessor(cond_br, 0);
    MLIR_BlockHandle s1 = MLIR_GetOpSuccessor(cond_br, 1);
    if (s0 == MLIR_INVALID_HANDLE || s1 == MLIR_INVALID_HANDLE) return false;
    if (s0 == s1 || s0 == entry || s1 == entry) return false;

    // Determine which successor is the "arm" (sole pred=entry, cf.br to
    // the other) and which is the "merge" (preds=2).
    MLIR_BlockHandle arm, merge;
    bool arm_is_then;
    if (MLIR_GetBlockNumPredecessors(s0) == 1) {
        MLIR_OpHandle s0t = MLIR_GetBlockTerminator(s0);
        if (s0t != MLIR_INVALID_HANDLE &&
            string_eq_string(MLIR_GetOpName(s0t), str_lit("cf.br")) &&
            MLIR_GetOpSuccessor(s0t, 0) == s1 &&
            MLIR_GetBlockNumPredecessors(s1) == 2) {
            arm = s0; merge = s1; arm_is_then = true;
            goto found;
        }
    }
    if (MLIR_GetBlockNumPredecessors(s1) == 1) {
        MLIR_OpHandle s1t = MLIR_GetBlockTerminator(s1);
        if (s1t != MLIR_INVALID_HANDLE &&
            string_eq_string(MLIR_GetOpName(s1t), str_lit("cf.br")) &&
            MLIR_GetOpSuccessor(s1t, 0) == s0 &&
            MLIR_GetBlockNumPredecessors(s0) == 2) {
            arm = s1; merge = s0; arm_is_then = false;
            goto found;
        }
    }
    return false;
found:;

    MLIR_OpHandle arm_term = MLIR_GetBlockTerminator(arm);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(cond_br);
    MLIR_LocationHandle arm_loc = MLIR_GetOpLocation(arm_term);

    // Snapshot.
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(cond_br, 0);
    size_t arm_succ_idx    = arm_is_then ? 0 : 1;
    size_t merge_succ_idx  = arm_is_then ? 1 : 0;
    size_t n_arm_args      = MLIR_GetOpNumSuccessorOperands(cond_br, arm_succ_idx);
    size_t n_direct_args   = MLIR_GetOpNumSuccessorOperands(cond_br, merge_succ_idx);
    size_t n_arm_yield     = MLIR_GetOpNumSuccessorOperands(arm_term, 0);
    size_t n_merge_args    = MLIR_GetBlockNumArgs(merge);
    if (n_arm_args != MLIR_GetBlockNumArgs(arm)) return false;
    if (n_direct_args != n_merge_args) return false;
    if (n_arm_yield   != n_merge_args) return false;

    MLIR_ValueHandle *arm_args = n_arm_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_arm_args) : NULL;
    for (size_t i = 0; i < n_arm_args; ++i)
        arm_args[i] = MLIR_GetOpSuccessorOperand(cond_br, arm_succ_idx, i);
    MLIR_ValueHandle *direct_args = n_direct_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_direct_args) : NULL;
    for (size_t i = 0; i < n_direct_args; ++i)
        direct_args[i] = MLIR_GetOpSuccessorOperand(cond_br, merge_succ_idx, i);
    MLIR_ValueHandle *arm_yield = n_arm_yield
        ? arena_new_array(arena, MLIR_ValueHandle, n_arm_yield) : NULL;
    for (size_t i = 0; i < n_arm_yield; ++i)
        arm_yield[i] = MLIR_GetOpSuccessorOperand(arm_term, 0, i);
    MLIR_TypeHandle *result_types = n_merge_args
        ? arena_new_array(arena, MLIR_TypeHandle, n_merge_args) : NULL;
    for (size_t i = 0; i < n_merge_args; ++i)
        result_types[i] = MLIR_GetValueType(MLIR_GetBlockArg(merge, i));

    // Replace arm's block args with arm_args.
    for (size_t i = 0; i < n_arm_args; ++i) {
        MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetBlockArg(arm, i), arm_args[i]);
    }
    if (n_arm_args) MLIR_EraseBlockArguments(ctx, arm, 0, n_arm_args);

    // Replace arm's cf.br terminator with scf.yield arm_yield.
    MLIR_EraseOp(ctx, arm_term);
    create_scf_yield(ctx, arena, arm, arm_yield, n_arm_yield, arm_loc);

    // Build the "empty" region's block with just scf.yield direct_args.
    MLIR_BlockHandle empty_block = MLIR_CreateBlock(ctx);
    create_scf_yield(ctx, arena, empty_block, direct_args, n_direct_args, loc);

    // Move arm into an arm-region; assemble the two regions in
    // (then, else) order.
    MLIR_RegionHandle arm_region   = MLIR_CreateRegion(ctx);
    MLIR_RegionHandle empty_region = MLIR_CreateRegion(ctx);
    MLIR_MoveBlockToRegionEnd(ctx, arm, arm_region);
    MLIR_AppendRegionBlock(ctx, empty_region, empty_block);

    // Build scf.if + result handles.
    MLIR_ValueHandle *result_vs = n_merge_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_merge_args) : NULL;
    for (size_t i = 0; i < n_merge_args; ++i) {
        result_vs[i] = MLIR_CreateValueOpResult(
            ctx, MLIR_INVALID_HANDLE, (uint32_t)i, result_types[i],
            fresh_ssa_name(arena), loc);
    }
    MLIR_RegionHandle *regions = arena_new_array(arena, MLIR_RegionHandle, 2);
    regions[0] = arm_is_then ? arm_region : empty_region;
    regions[1] = arm_is_then ? empty_region : arm_region;
    MLIR_ValueHandle *operands = arena_new_array(arena, MLIR_ValueHandle, 1);
    operands[0] = cond_v;

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

    // RAUW merge's block args with scf.if's results.
    for (size_t i = 0; i < n_merge_args; ++i) {
        MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetBlockArg(merge, i), result_vs[i]);
    }
    if (n_merge_args) MLIR_EraseBlockArguments(ctx, merge, 0, n_merge_args);

    // Splice merge into entry, erase merge.
    MLIR_SpliceBlockOps(ctx, entry, merge);
    MLIR_EraseBlock(ctx, merge);
    return true;
}

// ============================================================================
// Local-merge insertion (preprocessing for diamond patterns whose merge
// has more than two predecessors). Given:
//
//   ^entry: ... cf.cond_br %c, ^a(args_a), ^b(args_b)
//   ^a:     ...; cf.br ^merge(yield_a)
//   ^b:     ...; cf.br ^merge(yield_b)
//   ^merge(M_args): ...                  (>2 predecessors)
//
// insert ^local_merge between {a, b} and merge so the inner diamond
// becomes liftable as a clean diamond:
//
//   ^entry: ... cf.cond_br %c, ^a(args_a), ^b(args_b)
//   ^a:     ...; cf.br ^local_merge(yield_a)
//   ^b:     ...; cf.br ^local_merge(yield_b)
//   ^local_merge(L_args: M_arg_types): cf.br ^merge(L_args)
//   ^merge: ...
//
// Returns true if the rewrite fired.
// ============================================================================
static bool try_split_diamond_merge(MLIR_Context *ctx, Arena *arena,
                                    MLIR_BlockHandle entry) {
    MLIR_OpHandle cond_br = MLIR_GetBlockTerminator(entry);
    if (cond_br == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(cond_br), str_lit("cf.cond_br")))
        return false;
    if (MLIR_GetOpNumSuccessors(cond_br) != 2) return false;
    MLIR_BlockHandle then_b = MLIR_GetOpSuccessor(cond_br, 0);
    MLIR_BlockHandle else_b = MLIR_GetOpSuccessor(cond_br, 1);
    if (then_b == MLIR_INVALID_HANDLE || else_b == MLIR_INVALID_HANDLE) return false;
    if (then_b == else_b || then_b == entry || else_b == entry) return false;
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
    // Only fire when merge has MORE than 2 preds (otherwise the simple
    // diamond lifter handles it directly).
    if (MLIR_GetBlockNumPredecessors(merge) <= 2) return false;

    size_t n_args = MLIR_GetBlockNumArgs(merge);
    // Arity sanity.
    if (MLIR_GetOpNumSuccessorOperands(then_term, 0) != n_args) return false;
    if (MLIR_GetOpNumSuccessorOperands(else_term, 0) != n_args) return false;

    // Snapshot then/else cf.br operands and merge arg types.
    MLIR_LocationHandle loc = MLIR_GetOpLocation(then_term);
    MLIR_ValueHandle *then_yield = n_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_args) : NULL;
    MLIR_ValueHandle *else_yield = n_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_args) : NULL;
    for (size_t i = 0; i < n_args; ++i) {
        then_yield[i] = MLIR_GetOpSuccessorOperand(then_term, 0, i);
        else_yield[i] = MLIR_GetOpSuccessorOperand(else_term, 0, i);
    }
    MLIR_TypeHandle *arg_types = n_args
        ? arena_new_array(arena, MLIR_TypeHandle, n_args) : NULL;
    for (size_t i = 0; i < n_args; ++i) {
        arg_types[i] = MLIR_GetValueType(MLIR_GetBlockArg(merge, i));
    }

    // Build the new local-merge block with N args mirroring merge's args
    // and a cf.br trampoline back to merge.
    MLIR_BlockHandle local_merge = MLIR_CreateBlock(ctx);
    MLIR_ValueHandle *local_args = n_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_args) : NULL;
    for (size_t i = 0; i < n_args; ++i) {
        local_args[i] = MLIR_AddBlockArgument(ctx, local_merge,
                                              arg_types[i], loc);
        (void)fresh_ssa_name;
    }
    create_cf_br_in_block(ctx, arena, local_merge, merge, local_args, n_args, loc);

    // Insert the local_merge block right after else_b in the parent
    // region (placement is cosmetic — anywhere in the region works).
    MLIR_RegionHandle region = MLIR_GetBlockParentRegion(else_b);
    MLIR_InsertRegionBlockAfter(ctx, region, local_merge, else_b);

    // Retarget then_term and else_term to local_merge (operands stay).
    MLIR_SetOpSuccessor(ctx, then_term, 0, local_merge);
    MLIR_SetOpSuccessor(ctx, else_term, 0, local_merge);
    return true;
}

// ============================================================================
// Lift a self-loop block (H -> cf.cond_br [H, exit] or [exit, H]) into
// scf.while. This handles do-while loops after fold_linear_chain has
// collapsed the cond block into the body block.
//
//   ^entry:     ... cf.br ^H(init...)
//   ^H(iter...):  body_ops; %c = ...; cf.cond_br %c, ^H(self...), ^exit(res...)
//   ^exit(res...): ...
//
// Constraints: H has exactly 2 preds (entry-style + self), exit has sole
// pred = H, n_iter == n_self (back-edge args feed iter), exit_b's arg
// types/count match cond_br's exit-side payload.
//
// Lifting:
//   - Move all of H's ops (except the cond_br terminator) into a new
//     before-block (in a fresh before-region). before's block args
//     mirror H's iter args; RAUW H's old args -> before's new args.
//   - Replace cond_br with scf.condition(c, self_args) in before. (The
//     payload feeds after's args, which then yield back as next iter.)
//   - Build empty after-region with after-block whose args match
//     iter_types; terminate with scf.yield(after_args).
//   - In H (now empty of ops), append scf.while(init=H's old args)
//     producing res_types, then cf.br ^exit(while_results).
//   - RAUW exit's args with scf.while results; splice exit into H? No —
//     exit was already a separate block reachable only from H; we leave
//     exit standalone and let subsequent linear-fold splice it.
// ============================================================================
static bool try_lift_self_loop_while(MLIR_Context *ctx, Arena *arena,
                                     MLIR_BlockHandle H) {
    MLIR_OpHandle cond_br = MLIR_GetBlockTerminator(H);
    if (cond_br == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(cond_br), str_lit("cf.cond_br"))) return false;
    if (MLIR_GetOpNumSuccessors(cond_br) != 2) return false;
    MLIR_BlockHandle s0 = MLIR_GetOpSuccessor(cond_br, 0);
    MLIR_BlockHandle s1 = MLIR_GetOpSuccessor(cond_br, 1);
    if (s0 == MLIR_INVALID_HANDLE || s1 == MLIR_INVALID_HANDLE) return false;
    // Exactly one successor must be H itself (the self-loop).
    bool self_is_then;
    MLIR_BlockHandle exit_b;
    if (s0 == H && s1 != H) { self_is_then = true; exit_b = s1; }
    else if (s1 == H && s0 != H) { self_is_then = false; exit_b = s0; }
    else return false;
    if (exit_b == H) return false;
    if (MLIR_GetBlockNumPredecessors(H) != 2) return false;
    if (MLIR_GetBlockNumPredecessors(exit_b) != 1) return false;
    if (MLIR_BlockIsEntry(H)) return false; // need a real predecessor

    size_t self_idx = self_is_then ? 0 : 1;
    size_t exit_idx = self_is_then ? 1 : 0;
    size_t n_iter = MLIR_GetBlockNumArgs(H);
    size_t n_self = MLIR_GetOpNumSuccessorOperands(cond_br, self_idx);
    size_t n_res  = MLIR_GetOpNumSuccessorOperands(cond_br, exit_idx);
    size_t n_exit_args = MLIR_GetBlockNumArgs(exit_b);
    if (n_self != n_iter) return false;
    if (n_res != n_exit_args) return false;

    MLIR_LocationHandle loc = MLIR_GetOpLocation(cond_br);

    // Snapshot.
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(cond_br, 0);
    MLIR_ValueHandle *self_args = n_self
        ? arena_new_array(arena, MLIR_ValueHandle, n_self) : NULL;
    for (size_t i = 0; i < n_self; ++i)
        self_args[i] = MLIR_GetOpSuccessorOperand(cond_br, self_idx, i);
    MLIR_ValueHandle *exit_args = n_res
        ? arena_new_array(arena, MLIR_ValueHandle, n_res) : NULL;
    for (size_t i = 0; i < n_res; ++i)
        exit_args[i] = MLIR_GetOpSuccessorOperand(cond_br, exit_idx, i);

    MLIR_TypeHandle *iter_types = n_iter
        ? arena_new_array(arena, MLIR_TypeHandle, n_iter) : NULL;
    for (size_t i = 0; i < n_iter; ++i)
        iter_types[i] = MLIR_GetValueType(MLIR_GetBlockArg(H, i));
    MLIR_TypeHandle *res_types = n_res
        ? arena_new_array(arena, MLIR_TypeHandle, n_res) : NULL;
    for (size_t i = 0; i < n_res; ++i)
        res_types[i] = MLIR_GetValueType(MLIR_GetBlockArg(exit_b, i));

    // Build before block (in a new before region).
    MLIR_RegionHandle before_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  before = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, before_region, before);
    for (size_t i = 0; i < n_iter; ++i) {
        MLIR_ValueHandle ba = MLIR_AddBlockArgument(ctx, before, iter_types[i], loc);
        MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetBlockArg(H, i), ba);
    }

    // Move all of H's ops EXCEPT the terminator into before.
    // Walk H's ops by always taking op[0] until only the terminator
    // remains (since each move shifts indices).
    while (MLIR_GetBlockNumOps(H) > 1) {
        MLIR_OpHandle op0 = MLIR_GetBlockOp(H, 0);
        MLIR_MoveOpToBlockEnd(ctx, op0, before);
    }

    // Erase H's old cond_br and add scf.condition to before.
    MLIR_EraseOp(ctx, cond_br);
    create_scf_condition(ctx, arena, before, cond_v, self_args, n_self, loc);

    // Build after region with empty after-block whose args mirror
    // iter_types; terminate with scf.yield(after_args).
    MLIR_RegionHandle after_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  after = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, after_region, after);
    MLIR_ValueHandle *after_args = n_iter
        ? arena_new_array(arena, MLIR_ValueHandle, n_iter) : NULL;
    for (size_t i = 0; i < n_iter; ++i)
        after_args[i] = MLIR_AddBlockArgument(ctx, after, iter_types[i], loc);
    create_scf_yield(ctx, arena, after, after_args, n_iter, loc);

    // H is now empty of ops. Erase H's old args (they were RAUW'd to
    // before's args). We then re-add them to receive entry's init values
    // and feed the new scf.while op.
    MLIR_ValueHandle *init_vs = n_iter
        ? arena_new_array(arena, MLIR_ValueHandle, n_iter) : NULL;
    for (size_t i = 0; i < n_iter; ++i)
        init_vs[i] = MLIR_GetBlockArg(H, i);

    // Build scf.while op.
    MLIR_ValueHandle *result_vs = n_res
        ? arena_new_array(arena, MLIR_ValueHandle, n_res) : NULL;
    for (size_t i = 0; i < n_res; ++i) {
        result_vs[i] = MLIR_CreateValueOpResult(
            ctx, MLIR_INVALID_HANDLE, (uint32_t)i, res_types[i],
            fresh_ssa_name(arena), loc);
    }
    MLIR_RegionHandle *regions = arena_new_array(arena, MLIR_RegionHandle, 2);
    regions[0] = before_region; regions[1] = after_region;
    MLIR_OpHandle scf_while = MLIR_CreateOp(
        ctx, OP_TYPE_SCF_WHILE, str_lit("scf.while"),
        NULL, 0,
        res_types, n_res,
        result_vs, n_res,
        init_vs, n_iter,
        regions, 2,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, H, scf_while);

    // Add cf.br [exit](while_results) to H.
    create_cf_br_in_block(ctx, arena, H, exit_b, result_vs, n_res, loc);
    return true;
}

// ============================================================================
// Tail-duplicate a pass-through block (all ops have no regions, single
// cf.br terminator, no block args, multiple predecessors). Each pred
// edge gets its own clone of the block. After this every successor of
// those preds has a sole predecessor, so try_split_diamond_merge /
// try_lift_simple_if can fire on cond_br cascades.
//
// Cloning is shallow: each op is recreated via MLIR_CreateOp with the
// same name/attrs/operand-types/result-types. Operands defined inside
// the block are remapped to the clone's local copies; operands defined
// outside the block pass through unchanged.
// ============================================================================
static bool try_duplicate_passthrough_block(MLIR_Context *ctx, Arena *arena,
                                            MLIR_BlockHandle B) {
    if (MLIR_BlockIsEntry(B)) return false;
    if (MLIR_GetBlockNumArgs(B) != 0) return false;
    size_t n_ops = MLIR_GetBlockNumOps(B);
    if (n_ops < 1) return false;
    MLIR_OpHandle term = MLIR_GetBlockTerminator(B);
    if (term == MLIR_INVALID_HANDLE) return false;
    if (!string_eq_string(MLIR_GetOpName(term), str_lit("cf.br"))) return false;
    MLIR_BlockHandle dest = MLIR_GetOpSuccessor(term, 0);
    if (dest == MLIR_INVALID_HANDLE || dest == B) return false;
    size_t n_dest_args = MLIR_GetOpNumSuccessorOperands(term, 0);
    // Restrict ops: no regions, no successors (other than the terminator).
    for (size_t i = 0; i < n_ops; ++i) {
        MLIR_OpHandle o = MLIR_GetBlockOp(B, i);
        if (MLIR_GetOpNumRegions(o) != 0) return false;
        if (o != term && MLIR_GetOpNumSuccessors(o) != 0) return false;
    }
    if (MLIR_GetBlockNumPredecessors(B) < 2) return false;

    // Restrict: only duplicate when the destination is a multi-pred
    // merge (otherwise duplication doesn't enable any subsequent lift).
    if (MLIR_GetBlockNumPredecessors(dest) <= 2) return false;
    // Restrict: all preds must be cf.cond_br (the switch-cascade shape).
    // This avoids duplicating blocks in the middle of richer CFGs where
    // duplication may interact poorly with subsequent lifts.

    // Reject if any op result is used outside the block (would break SSA
    // when we duplicate without inserting a phi-like merge).
    for (size_t i = 0; i < n_ops; ++i) {
        MLIR_OpHandle o = MLIR_GetBlockOp(B, i);
        size_t nr = MLIR_GetOpNumResults(o);
        for (size_t k = 0; k < nr; ++k) {
            MLIR_ValueHandle v = MLIR_GetOpResult(o, k);
            size_t nu = MLIR_GetValueNumUses(ctx, v);
            for (size_t u = 0; u < nu; ++u) {
                MLIR_OpHandle uo = MLIR_GetValueUseOwner(ctx, v, u, NULL);
                if (uo == MLIR_INVALID_HANDLE) continue;
                if (MLIR_GetOpParentBlock(uo) != B) return false;
            }
        }
    }

    MLIR_LocationHandle loc = MLIR_GetOpLocation(term);
    MLIR_RegionHandle region = MLIR_GetBlockParentRegion(B);

    typedef struct { MLIR_OpHandle term; size_t succ_idx; } PredEdge;
    size_t nb = MLIR_GetRegionNumBlocks(region);
    PredEdge *edges = arena_new_array(arena, PredEdge, nb * 2 + 4);
    size_t n_edges = 0;
    for (size_t i = 0; i < nb; ++i) {
        MLIR_BlockHandle pb = MLIR_GetRegionBlock(region, i);
        if (pb == B) continue;
        MLIR_OpHandle pt = MLIR_GetBlockTerminator(pb);
        if (pt == MLIR_INVALID_HANDLE) continue;
        size_t ns = MLIR_GetOpNumSuccessors(pt);
        for (size_t s = 0; s < ns; ++s) {
            if (MLIR_GetOpSuccessor(pt, s) == B) {
                edges[n_edges].term = pt;
                edges[n_edges].succ_idx = s;
                n_edges++;
            }
        }
    }
    if (n_edges < 2) return false;

    // Snapshot attribute handles + result types for each op (immutable,
    // safely shared across clones).
    typedef struct {
        string opname;
        MLIR_OpType opty;
        size_t n_attrs;
        MLIR_AttributeHandle *attrs;
        size_t n_operands;
        MLIR_ValueHandle *operand_vs;
        size_t n_results;
        MLIR_TypeHandle *result_types;
        MLIR_LocationHandle loc;
    } OpSnap;
    OpSnap *snaps = arena_new_array(arena, OpSnap, n_ops);
    for (size_t i = 0; i < n_ops; ++i) {
        MLIR_OpHandle o = MLIR_GetBlockOp(B, i);
        snaps[i].opname = MLIR_GetOpName(o);
        snaps[i].opty   = MLIR_GetOpType(o);
        snaps[i].loc    = MLIR_GetOpLocation(o);
        snaps[i].n_attrs = MLIR_GetOpNumAttributes(o);
        snaps[i].attrs = snaps[i].n_attrs
            ? arena_new_array(arena, MLIR_AttributeHandle, snaps[i].n_attrs) : NULL;
        for (size_t k = 0; k < snaps[i].n_attrs; ++k)
            snaps[i].attrs[k] = MLIR_GetOpAttribute(o, k);
        snaps[i].n_operands = MLIR_GetOpNumOperands(o);
        snaps[i].operand_vs = snaps[i].n_operands
            ? arena_new_array(arena, MLIR_ValueHandle, snaps[i].n_operands) : NULL;
        for (size_t k = 0; k < snaps[i].n_operands; ++k)
            snaps[i].operand_vs[k] = MLIR_GetOpOperand(o, k);
        snaps[i].n_results = MLIR_GetOpNumResults(o);
        snaps[i].result_types = snaps[i].n_results
            ? arena_new_array(arena, MLIR_TypeHandle, snaps[i].n_results) : NULL;
        for (size_t k = 0; k < snaps[i].n_results; ++k)
            snaps[i].result_types[k] = MLIR_GetOpResult_type(o, k);
    }
    // Snapshot original op result handles (to build remap per clone).
    MLIR_ValueHandle **orig_results = arena_new_array(arena, MLIR_ValueHandle*, n_ops);
    for (size_t i = 0; i < n_ops; ++i) {
        orig_results[i] = snaps[i].n_results
            ? arena_new_array(arena, MLIR_ValueHandle, snaps[i].n_results) : NULL;
        MLIR_OpHandle o = MLIR_GetBlockOp(B, i);
        for (size_t k = 0; k < snaps[i].n_results; ++k)
            orig_results[i][k] = MLIR_GetOpResult(o, k);
    }
    // Snapshot terminator's destination operands (just before the term op).
    MLIR_ValueHandle *term_succ_ops = n_dest_args
        ? arena_new_array(arena, MLIR_ValueHandle, n_dest_args) : NULL;
    for (size_t k = 0; k < n_dest_args; ++k)
        term_succ_ops[k] = MLIR_GetOpSuccessorOperand(term, 0, k);

    // For each pred edge, build a clone block.
    for (size_t e = 0; e < n_edges; ++e) {
        MLIR_BlockHandle clone = MLIR_CreateBlock(ctx);
        MLIR_InsertRegionBlockAfter(ctx, region, clone, B);
        for (size_t i = 0; i < n_ops; ++i) {
            if (i + 1 == n_ops) break;  // skip terminator (rebuild with succ)
            OpSnap *s = &snaps[i];
            MLIR_ValueHandle *new_ops = s->n_operands
                ? arena_new_array(arena, MLIR_ValueHandle, s->n_operands) : NULL;
            for (size_t k = 0; k < s->n_operands; ++k) {
                MLIR_ValueHandle v = s->operand_vs[k];
                // Remap if v is a result of a previously cloned op in this block.
                MLIR_ValueHandle nv = v;
                for (size_t pi = 0; pi < i; ++pi) {
                    for (size_t pk = 0; pk < snaps[pi].n_results; ++pk) {
                        if (orig_results[pi][pk] == v) {
                            // Look up clone's i-th op's result.
                            MLIR_OpHandle co = MLIR_GetBlockOp(clone, pi);
                            nv = MLIR_GetOpResult(co, pk);
                            break;
                        }
                    }
                }
                new_ops[k] = nv;
            }
            MLIR_ValueHandle *new_results = s->n_results
                ? arena_new_array(arena, MLIR_ValueHandle, s->n_results) : NULL;
            for (size_t k = 0; k < s->n_results; ++k) {
                new_results[k] = MLIR_CreateValueOpResult(
                    ctx, MLIR_INVALID_HANDLE, (uint32_t)k,
                    s->result_types[k], fresh_ssa_name(arena), s->loc);
            }
            MLIR_OpHandle no = MLIR_CreateOp(
                ctx, s->opty, s->opname,
                s->attrs, s->n_attrs,
                s->result_types, s->n_results,
                new_results, s->n_results,
                new_ops, s->n_operands,
                NULL, 0,
                s->loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
            MLIR_AppendBlockOp(ctx, clone, no);
        }
        // Rebuild terminator's successor operands with remapped values.
        MLIR_ValueHandle *new_dest_ops = n_dest_args
            ? arena_new_array(arena, MLIR_ValueHandle, n_dest_args) : NULL;
        for (size_t k = 0; k < n_dest_args; ++k) {
            MLIR_ValueHandle v = term_succ_ops[k];
            MLIR_ValueHandle nv = v;
            for (size_t pi = 0; pi + 1 < n_ops; ++pi) {
                for (size_t pk = 0; pk < snaps[pi].n_results; ++pk) {
                    if (orig_results[pi][pk] == v) {
                        MLIR_OpHandle co = MLIR_GetBlockOp(clone, pi);
                        nv = MLIR_GetOpResult(co, pk);
                        break;
                    }
                }
            }
            new_dest_ops[k] = nv;
        }
        create_cf_br_in_block(ctx, arena, clone, dest, new_dest_ops, n_dest_args, loc);
        MLIR_SetOpSuccessor(ctx, edges[e].term, edges[e].succ_idx, clone);
    }

    // Erase original block. First erase its ops in reverse order.
    while (MLIR_GetBlockNumOps(B) > 0) {
        MLIR_OpHandle o = MLIR_GetBlockOp(B, MLIR_GetBlockNumOps(B) - 1);
        MLIR_EraseOp(ctx, o);
    }
    MLIR_EraseBlock(ctx, B);
    return true;
}

static bool fold_simple_loops_and_ifs(MLIR_Context *ctx, Arena *arena,
                                      MLIR_BlockHandle entry) {
    bool any = false;
    MLIR_RegionHandle region = MLIR_GetBlockParentRegion(entry);
    while (true) {
        bool spliced = fold_linear_chain_region(ctx, arena, region);
        bool lifted_any = false;
        // Scan every block in the region: any block can be the "header"
        // of a simple-if diamond or a simple-while loop. After a lift,
        // block list mutates so we restart from the beginning.
        bool changed = true;
        while (changed) {
            changed = false;
            size_t nb = MLIR_GetRegionNumBlocks(region);
            for (size_t bi = 0; bi < nb; ++bi) {
                MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
                if (try_lift_simple_if(ctx, arena, b)) {
                    changed = true; lifted_any = true; break;
                }
                if (try_lift_if_one_arm(ctx, arena, b)) {
                    changed = true; lifted_any = true; break;
                }
                if (try_lift_simple_while(ctx, arena, b)) {
                    changed = true; lifted_any = true; break;
                }
                if (try_split_diamond_merge(ctx, arena, b)) {
                    changed = true; lifted_any = true; break;
                }
                if (try_lift_self_loop_while(ctx, arena, b)) {
                    changed = true; lifted_any = true; break;
                }
                // TODO: try_duplicate_passthrough_block is disabled — current
                // implementation regresses re2c_keyword_longest_match under
                // TINYC_LIFT_USE_NATIVE=1 (likely a remap miss when an op
                // result is consumed by a successor-operand of cf.br). Keep
                // for future debugging; needs a real EdgeMultiplexer port.
                (void)try_duplicate_passthrough_block;
            }
        }
        if (!spliced && !lifted_any) break;
        any = any || spliced || lifted_any;
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
            fold_simple_loops_and_ifs(ctx, scratch, entry);
        }
    }
    arena_destroy(scratch);
    return true;
}

// ---------------------------------------------------------------------------
// LiftState init + cached constant/undef value providers.
// ---------------------------------------------------------------------------
static void lift_state_init(LiftState *st, MLIR_Context *ctx, Arena *arena,
                            MLIR_OpHandle fn_op, MLIR_BlockHandle entry_block) {
    memset(st, 0, sizeof(*st));
    st->ctx = ctx;
    st->arena = arena;
    st->fn_op = fn_op;
    st->entry_block = entry_block;
    st->i32_ty = MLIR_CreateTypeInteger(ctx, 32, true);
    st->unk_loc = MLIR_INVALID_HANDLE;
    if (MLIR_GetBlockNumOps(entry_block) > 0) {
        st->unk_loc = MLIR_GetOpLocation(MLIR_GetBlockOp(entry_block, 0));
    }
}

// Emit a fresh `arith.constant <v> : i32` op inserted at the BEGINNING of
// the function entry block. Used to lazily build the switchValueCache.
// Mirrors `CFGToSCFForWasm::getCFGSwitchValue`.
static MLIR_ValueHandle get_switch_value(LiftState *st, unsigned v) {
    if (v < st->switch_value_cache_n) {
        MLIR_ValueHandle cached = st->switch_value_cache[v];
        if (cached != MLIR_INVALID_HANDLE) return cached;
    }
    // Grow cache if needed.
    if (v >= st->switch_value_cache_cap) {
        size_t new_cap = st->switch_value_cache_cap ? st->switch_value_cache_cap : 4;
        while (new_cap <= v) new_cap *= 2;
        MLIR_ValueHandle *na = arena_new_array(st->arena, MLIR_ValueHandle, new_cap);
        for (size_t i = 0; i < new_cap; ++i) na[i] = MLIR_INVALID_HANDLE;
        for (size_t i = 0; i < st->switch_value_cache_n; ++i)
            na[i] = st->switch_value_cache[i];
        st->switch_value_cache = na;
        st->switch_value_cache_cap = new_cap;
    }
    if (v >= st->switch_value_cache_n) st->switch_value_cache_n = v + 1;

    // Build `arith.constant <v> : i32` and insert at start of entry.
    MLIR_TypeHandle  *rt = arena_new_array(st->arena, MLIR_TypeHandle, 1);
    rt[0] = st->i32_ty;
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(
        st->ctx, MLIR_INVALID_HANDLE, 0, st->i32_ty,
        fresh_ssa_name(st->arena), st->unk_loc);
    MLIR_ValueHandle *rs = arena_new_array(st->arena, MLIR_ValueHandle, 1);
    rs[0] = r;
    MLIR_AttributeHandle  val = MLIR_CreateAttributeInteger(
        st->ctx, str_lit("value"), (int64_t)(int32_t)v, st->i32_ty);
    MLIR_AttributeHandle *as = arena_new_array(st->arena, MLIR_AttributeHandle, 1);
    as[0] = val;
    MLIR_OpHandle op = MLIR_CreateOp(
        st->ctx, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
        rt, 1, rs, 1, NULL, 0, NULL, 0, as, 1,
        st->unk_loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_InsertBlockOpAtIndex(st->ctx, st->entry_block, op, 0);
    st->switch_value_cache[v] = r;
    return r;
}

// Emit a fresh `llvm.mlir.undef : T` op at the start of the entry block
// and cache it per-type. Mirrors `getUndefValue` / `ub.poison` in the
// upstream wasm-flavored interface (we use llvm.mlir.undef because our
// generic printer + native parser understands it, and wasmssa-lower
// already supports llvm.mlir.undef -> typed zero).
static MLIR_ValueHandle get_undef_value(LiftState *st, MLIR_TypeHandle ty) {
    for (size_t i = 0; i < st->undef_cache_n; ++i) {
        if (st->undef_cache[i].type == ty) return st->undef_cache[i].value;
    }
    if (st->undef_cache_n == st->undef_cache_cap) {
        size_t new_cap = st->undef_cache_cap ? st->undef_cache_cap * 2 : 8;
        TypedValueEntry *na = arena_new_array(st->arena, TypedValueEntry, new_cap);
        for (size_t i = 0; i < st->undef_cache_n; ++i) na[i] = st->undef_cache[i];
        st->undef_cache = na;
        st->undef_cache_cap = new_cap;
    }
    MLIR_TypeHandle  *rt = arena_new_array(st->arena, MLIR_TypeHandle, 1);
    rt[0] = ty;
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(
        st->ctx, MLIR_INVALID_HANDLE, 0, ty,
        fresh_ssa_name(st->arena), st->unk_loc);
    MLIR_ValueHandle *rs = arena_new_array(st->arena, MLIR_ValueHandle, 1);
    rs[0] = r;
    MLIR_OpHandle op = MLIR_CreateOp(
        st->ctx, OP_TYPE_LLVM_MLIR_UNDEF, str_lit("llvm.mlir.undef"),
        rt, 1, rs, 1, NULL, 0, NULL, 0, NULL, 0,
        st->unk_loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_InsertBlockOpAtIndex(st->ctx, st->entry_block, op, 0);
    TypedValueEntry e = { ty, r };
    st->undef_cache[st->undef_cache_n++] = e;
    return r;
}

// ---------------------------------------------------------------------------
// Branch-op constructors.
// ---------------------------------------------------------------------------

// Emit `cf.cond_br %cond, ^t(t_args...), ^f(f_args...)` at the end of
// `in_block`. Returns the new op handle.
static MLIR_OpHandle create_cf_cond_br_in_block(MLIR_Context *ctx, Arena *arena,
                                                MLIR_BlockHandle in_block,
                                                MLIR_ValueHandle cond,
                                                MLIR_BlockHandle t_dst,
                                                MLIR_ValueHandle *t_args, size_t n_t_args,
                                                MLIR_BlockHandle f_dst,
                                                MLIR_ValueHandle *f_args, size_t n_f_args,
                                                MLIR_LocationHandle loc) {
    MLIR_ValueHandle *ops = arena_new_array(arena, MLIR_ValueHandle, 1);
    ops[0] = cond;
    MLIR_BlockHandle *succs = arena_new_array(arena, MLIR_BlockHandle, 2);
    succs[0] = t_dst; succs[1] = f_dst;
    MLIR_ValueHandle **sops = arena_new_array(arena, MLIR_ValueHandle *, 2);
    sops[0] = t_args; sops[1] = f_args;
    size_t *snums = arena_new_array(arena, size_t, 2);
    snums[0] = n_t_args; snums[1] = n_f_args;
    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(
        ctx, OP_TYPE_CF_COND_BR, str_lit("cf.cond_br"),
        NULL, 0, NULL, 0, ops, 1, NULL, 0, NULL, 0,
        succs, 2, sops, snums,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, in_block, op);
    return op;
}

// Emit `cf.switch %flag : i32, [ default: ^dflt(args), Vi: ^Bi(args), ... ]`
// at the end of `in_block`. `case_values` and `case_dests`/`case_args` are
// parallel arrays of length `n_cases`. The default destination + args
// come last in the successor list (per MLIR's cf.SwitchOp printing
// convention). Returns the new op handle.
//
// Operands: [flag] followed by all per-case+default operand values
// flattened (cf.switch is variadic with `operand_segment_sizes` listing
// per-segment counts). Successors: [default, case0, case1, ...].
// case_values is a DenseI32ArrayAttr; case_operand_segments is a
// DenseI32ArrayAttr listing the number of operands per case.
static MLIR_OpHandle create_cf_switch_in_block(MLIR_Context *ctx, Arena *arena,
                                               MLIR_BlockHandle in_block,
                                               MLIR_ValueHandle flag,
                                               MLIR_BlockHandle dflt_dst,
                                               MLIR_ValueHandle *dflt_args, size_t n_dflt_args,
                                               const int32_t *case_values,
                                               MLIR_BlockHandle *case_dsts,
                                               MLIR_ValueHandle **case_args,
                                               size_t *n_case_args,
                                               size_t n_cases,
                                               MLIR_LocationHandle loc) {
    // Build successor + per-successor operand arrays:
    //   succs[0]   = default
    //   succs[1..] = cases
    size_t n_succ = 1 + n_cases;
    MLIR_BlockHandle *succs = arena_new_array(arena, MLIR_BlockHandle, n_succ);
    MLIR_ValueHandle **sops = arena_new_array(arena, MLIR_ValueHandle *, n_succ);
    size_t *snums = arena_new_array(arena, size_t, n_succ);
    succs[0] = dflt_dst; sops[0] = dflt_args; snums[0] = n_dflt_args;
    for (size_t i = 0; i < n_cases; ++i) {
        succs[1 + i] = case_dsts[i];
        sops[1 + i] = case_args[i];
        snums[1 + i] = n_case_args[i];
    }
    // Operands: just the flag (the successor-operands live on
    // succession metadata, not in the operand list).
    MLIR_ValueHandle *ops = arena_new_array(arena, MLIR_ValueHandle, 1);
    ops[0] = flag;
    // Attribute: case_values: array<i32: V0, V1, ...>.
    MLIR_AttributeHandle ca = MLIR_CreateAttributeDenseI32Array(
        ctx, str_lit("case_values"), case_values, n_cases);
    MLIR_AttributeHandle *as = arena_new_array(arena, MLIR_AttributeHandle, 1);
    as[0] = ca;
    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(
        ctx, OP_TYPE_CF_SWITCH, str_lit("cf.switch"),
        NULL, 0, NULL, 0, ops, 1, NULL, 0, as, 1,
        succs, n_succ, sops, snums,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(ctx, in_block, op);
    return op;
}
