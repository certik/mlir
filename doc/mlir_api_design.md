# MLIR API review and design rules

Status: **proposal** (#330, roadmap #328 item A1). Nothing in this document is
implemented yet; it changes no code. It inventories `mlir_api.h` as of
`48dd697`, states the rules the API should follow before it freezes, and ends
with an ordered list of follow-up issues that replace roadmap items A2–A12.

File references use these abbreviations:

| Prefix | File |
|---|---|
| `h:` | `mlir_api.h` |
| `c:` | `mlir_api_impl.c` (native backend) |
| `u:` | `mlir_api_impl_upstream.cpp` (upstream backend) |
| `rsqrt:` / `red:` | LFortran `src/libasr/codegen/llvm_rsqrt.cpp` / `llvm_reductions.cpp` (main branch, fetched 2026-11-21) |

All other references are `file:line` relative to the repository root.

## 0. Summary

The API works, and its shape is close to what we want: C, opaque handles, an
op enum, arena-allocated IR, one header with two implementations. The problems
are concentrated in a few places:

1. **Strings on hot paths.** Attribute lookup is by `const char *` name
   (`GetOpAttributeByName`: 80 direct call sites, plus about 360 more through
   per-file `attr_i`/`at_i` helpers, 195 of them in `mlir_aarch64_asm.c`).
   Backends classify scalar types by *formatting the type to a string and
   parsing it* (`mlir_llvm_to_x64.c:323`, `mlir_llvm_to_aarch64.c:468`,
   `mlir_llvm_to_wasmssa.c:129`, `mlir_llvm_type_layout.h:117`). Dense arrays
   are stored as text on the native backend and re-parsed in six places
   (§6 N16). Op creation still takes a name string.
2. **Upstream indirection that is not light.** Every operand/result access
   goes through a global `unordered_map` to a heap `ValueBox`; every
   `GetOpAttributeByName` heap-allocates; region/block indexing is linear
   (§5).
3. **No traversal, trait, pass, or LLVM-level vocabulary.** Passes iterate by
   index (`GetRegionBlock(r, i)` is O(i) upstream), keep their own copies of
   op facts (`mlir_llvm_op_traits.h`, `ssa_op_has_result`,
   `ag_is_commutative`, `lc_is_pure`), and rebuild use counts by hand because
   def-use tracking is optional. Vector types, fast-math flags, shuffles,
   intrinsics, splat constants, target triple/features are all missing, so
   neither LFortran pass can be written today (§4).
4. **Surface debt.** 23 of the 176 functions have no caller outside the
   implementations; 35 more are used only by the parser/printers; five are
   unsound (`SetType*Properties` mutate interned types). Several functions
   behave differently on the two backends (erase, `GetOpLocation`,
   `IsTypeFunction`, `GetAttributeAsString`).

Main recommendations:

- **Handles:** keep the `MLIR_*Handle` names but make each one a pointer to a
  distinct incomplete struct (`typedef struct MLIR_OpImpl *MLIR_OpHandle;`).
  Zero cost on every ABI, `==`/`NULL`/hashing keep working, mixing kinds is a
  compiler diagnostic (§2.2).
- **Identity:** one X-macro file `mlir_ops.def` generates the enum, name
  array, dialect ranges, and trait table. Creation takes only the enum.
  Unknown names get dynamic ids registered in the context; no op stores a name
  (§2.5, #51, #145, #140).
- **Attributes:** attribute handles become *values* (no name); names become
  `MLIR_AttrKey` integers (registered keys from a `.def` file plus interned
  keys). `MLIR_GetOpAttr(op, key)` is a u32 scan natively and an
  allocation-free `getAttr(StringAttr)` upstream (§2.6).
- **Builder:** one creation entry point `MLIR_CreateOpFromSpec(ctx, &spec)`
  plus a 4-field `MLIR_Builder` insertion point and `static inline` shape
  helpers. Results are created by the op; operands of successors are flat, as
  upstream. One call and (natively) one allocation per op (§2.7).
- **Traversal:** iterator functions (`GetBlockFirstOp`/`GetOpNext`,
  first/next for blocks, uses and predecessors) as the primary interface; O(1)
  on both backends. Walk and RPO helpers on top (§3.2).
- **Passes:** a function-pointer `MLIR_Pass` and `MLIR_RunPasses` pipeline
  with timing and verification; upstream wraps each pass in an
  `mlir::OperationPass` adapter and runs a real `mlir::PassManager`. No
  pattern-rewrite driver for now (§3.3, §3.4).
- **Open question 6:** API-level LLVM passes run on the LLVM dialect on both
  backends now. If LFortran wants them inside LLVM's pipeline, add a *third*
  link-time implementation over `llvm::Module` for the LLVM-level subset,
  rather than a run-time dual view inside the upstream backend (§3.6).

Decisions needed from the owner (each is discussed where it appears):

| # | Decision | Recommendation |
|---|---|---|
| D1 | Handle representation: pointer-to-incomplete-struct vs `struct { uintptr_t }` wrappers (roadmap A4 text); natively, arena pointers vs 32-bit indices into context-owned vectors (option D) | pointer typedefs (§2.2); natively indices, option D (decided after `bench/ir_storage`, #370; implemented in #354) |
| D2 | Unregistered ops: dynamic op-type ids in the context vs `OP_TYPE_UNREGISTERED` + name on the op (#145) | dynamic ids (§2.5) |
| D3 | Def-use tracking: always on with intrusive operand lists vs keep the `no_def_use_tracking` toggle | always on, if A9 measures it within a budget (proposed: ≤ 5% compile time, ≤ 15% IR memory on the wasm-lift pipeline) (§2.3, N6) |
| D4 | Native block storage: op arrays vs intrusive lists | decide on A9 numbers; the public API (iterators) is the same either way (§6 N7) |
| D5 | Operand model: flat operands with per-successor ranges (upstream's) vs separate successor-operand arrays | flat (§2.7) |
| D6 | Open question 6: LLVM dialect now, `llvm::Module` backend later vs dual view now | LLVM dialect now (§3.6) |
| D7 | Pattern-rewrite driver | not now (§3.4) |
| D8 | Enumerator prefix: keep `OP_TYPE_*` vs rename to `MLIR_OP_*` | keep (§2.1) |
| D9 | The 33 parser-only enumerators (tt, gpu, affine, vector, std, tensor, linalg, index, math, memref) | move to dynamic registration after D2 lands (§1.2) |

## 1. Inventory

### 1.1 Functions

Method: every function declared in `mlir_api.h` (176, extracted by regex from
the header), counted with a script that strips comments and string literals
and counts every reference that is not a definition (calls and
function-pointer uses, e.g. `driver.c:996-1004`). Counts are references in
source, not dynamic calls. Users are grouped by file:

| Group | Files |
|---|---|
| `emit` | `examples/tinyc/emit.c` |
| `drv` | `examples/tinyc/driver.c` |
| `x64` | `mlir_llvm_to_x64.c`, `mlir_llvm_to_x64_emit.inc` |
| `a64` | `mlir_llvm_to_aarch64.c`, `mlir_aarch64_to_macho.c`, `mlir_aarch64_asm.c` |
| `wasm` | `mlir_llvm_to_wasmssa.c`, `mlir_wasmssa_to_wasmstack.c`, `mlir_wasmstack_to_bin.c`, `mlir_translate_to_wasm.c`, `mlir_wasm_to_wat.c` |
| `wlift` | `mlir_wasm_to_wasmstack.c`, `mlir_wasmstack_to_wasmssa.c`, `mlir_wasmssa_to_llvm.c`, `mlir_wasm_to_macho.c`, `mlir_wasm_link.c` |
| `llir` | `mlir_translate_to_llvm_ir.c` |
| `conv` | `mlir_lower_to_llvm.c`, `mlir_lift_cf_to_scf.c` (dialect conversions) |
| `opt` | `mlir_llvm_mem2reg.c`, `mlir_llvm_load_cse.c`, `mlir_llvm_arith_gvn.c`, `mlir_llvm_dce.c`, `mlir_llvm_global_init.c` |
| `shr` | `mlir_llvm_type_layout.h`, `mlir_llvm_op_traits.h`, `mlir_regalloc.c`, `mlir_machine.h`, `mlir_obj.h` (shared backend code) |
| `pp` | `parser.c`, `op_parsers.c`, `mlir_parser.c`, `mlir_classic_printer.c`, `mlir_generic_printer.c` |
| `test` | `tests/cross/driver.c`, `tests/upstream_main.c` |

No other file references an API function (checked with grep over all `.c`,
`.h`, `.inc`, `.cpp`). Totals: 4491 references; by group pp 1078, conv 567,
wasm 558, wlift 488, a64 480, opt 397, llir 214, x64 211, emit 187, shr 153,
drv 97, test 61. The ten most referenced functions are `GetOpOperand` (407),
`GetOpType` (194), `GetOpResult` (189), `AppendBlockOp` (179), `GetValueType`
(175), `GetOpNumOperands` (172), `GetRegionBlock` (169), `GetBlockOp` (152),
`GetBlockNumOps` (139) and `GetOpRegion` (114) — structure walking dominates,
which is why §3.2 matters for upstream overhead.

- **Unused (23):** `GetOpSourceLineStart`, `GetOpUnnumberedLocationDef`,
  `InsertBlockOpBeforeTerminator`, `GetValueResultIndex`, the five
  `SetType*Properties`, `IsTypeTensor`, `IsTypeMemref`, `IsTypePointer`,
  `IsTypeIndex`, `IsTypeUnknown`, `IsTypeOpaque`, `IsTypeFunction`,
  `GetTypeLLVMPointerAddressSpace`, `SetOpRegion`, `GetBlockOpIndex`,
  `GetBlockParentOp`, `BlockIsEntry`, `SetOpOperands`, `GetLocationRefId`.
- **Parser/printer/test only (35):** location map (2), `MLIR_OpTypeToString`,
  `InitApi`, `AppendOpAttribute`, the four `Print*`/`Parse*` entry points,
  `GetOpName_string`, `GetOpTrailingComment`, `GetOpNumResultTypes`,
  `GetBlockIndex`, `GetValueRegisterName`, `GetValueLocation`,
  `CreateTypeUnknown/Tensor/Memref/Pointer/Opaque`, `GetTypeShapedElement`,
  `CreateAttributeDict`, the array/dict attribute getters (4), and the nine
  location constructors and getters other than `CreateLocationUnknown` and
  the unused `GetLocationRefId`.

Columns: **Call sites** is the total followed by the per-group counts;
**Native**/**Upstream** describe the implementation and any per-call cost
(`N*`/`U*` refer to §6/§5); `h:` is the declaration line. The `MLIR_` prefix
is omitted from the first column.

#### Context and lifecycle (7)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `InitApi` (h:525) | No-op init hook | **1** test 1 | empty (c:567) | empty (u:298) | **remove**; replace by `MLIR_CreateContext` |
| `SetArenaAllocator` (h:526) | Swap the IR arena | **19** drv 15 a64 2 pp 1 test 1 | field write (c:572) | field write (u:304) | **keep** for now; arena per module later (§2.3) |
| `GetArenaAllocator` (h:527) | Read the IR arena | **50** a64 17 conv 10 pp 7 x64 4 wasm 4 wlift 3 drv 2 emit 1 llir 1 shr 1 | field read (c:577) | field read (u:305) | **keep** |
| `ResetInternRegistry` (h:531) | Drop process-wide type/op registries before freeing an arena | **4** drv 4 | clears 3 globals (c:1287) | no-op (u:303) | **remove** once registries live in the context and types in `type_arena` (§2.3, #140) |
| `GetLocationMapSize` (h:55) | Parser location-map size | **2** pp 2 | parser (mlir_parser.c) | parser | **move** to `mlir_parser.h` (already redeclared at mlir_parser.h:168) |
| `CollectLocationMap` (h:56) | Parser location-map dump | **2** pp 2 | parser | parser | **move** to `mlir_parser.h` |
| `MLIR_OpTypeToString` (h:519) | Registered name of an op type | **6** pp 6 | c:1908 → 2-level switch (mlir_op_names.c:316) | same table (u:1631) | **rename** `MLIR_GetOpTypeName`; array lookup from `mlir_ops.def` |

#### Op creation and mutation (13)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `CreateOp` (h:545) | Create a detached op (17 params) | **95** pp 39 wlift 17 conv 12 emit 9 wasm 7 a64 5 test 4 opt 2 | c:712 → c:582: ≤6 array copies + IR_Op + aux + 1 UseNode/operand; name compare (N5, N13) | std::string name, OperationState by string, attr snapshot (u:519, U6, U2) | **replace** by `MLIR_CreateOpFromSpec` / builder (§2.7) |
| `CreateOpWithSuccessors` (h:562) | Same with successors (21 params) | **14** conv 6 emit 2 wlift 2 pp 2 a64 1 test 1 | c:582; per-successor arrays (c:650-670) | u:384; segment-size string compares (u:465-504) | **merge** into `MLIR_OpSpec` (flat successor operands) |
| `AppendOpAttribute` (h:578) | Add/replace an attribute after creation | **2** pp 2 | reallocates array per append (c:1767) | setAttr + snapshot update (u:537) | **rename** `MLIR_SetOpAttr(op, key, attr)` |
| `SetOpOperand` (h:957) | Replace operand (flat index incl. successor operands) | **8** opt 5 wlift 1 conv 1 test 1 | slot write + use relink; succ segment scan (c:2231) | op->setOperand (u:1748) | **keep** (flat index is the rule, §2.7) |
| `SetOpOperands` (h:1051) | Replace all operands | **0** | new array + use relink (c:2635) | op->setOperands (u:1948) | **remove** (0 uses) |
| `SetOpSuccessor` (h:964) | Retarget successor | **3** conv 3 | slot write (c:2286) | setSuccessor (u:1758) | **keep** |
| `SetOpSuccessorOperands` (h:973) | Replace one successor operand list | **5** opt 4 conv 1 | realloc + relink (c:2294) | BranchOpInterface MutableOperandRange (u:1767) | **keep** |
| `AppendOpSuccessorOperand` (h:1058) | Append to one successor operand list (phi edge) | **4** conv 4 | realloc whole segment per append (c:2657) | MutableOperandRange::append (u:1965) | **keep**; needed for phi creation |
| `SetOpRegion` (h:921) | Replace a region | **0** | slot write (c:2157) | takeBody + delete (u:1664) | **remove** (0 uses; `TakeOpRegion` + spec covers it) |
| `TakeOpRegion` (h:928) | Detach a region body | **3** conv 3 | swap in empty region (c:2165) | takeBody into new Region (u:1682) | **keep** |
| `EraseOp` (h:916) | Erase an op | **23** conv 9 opt 8 a64 3 drv 2 wlift 1 | unlink uses, O(n) compaction of block array; shell stays valid (c:2121) | op->erase() frees; 2 map erases (u:1649) | **keep**; unify semantics: handle dead after erase (§2.3); add `MLIR_DetachOp` |
| `MoveOpToBlockEnd` (h:946) | Move op to end of block | **7** drv 5 a64 1 conv 1 | O(n) removal (c:2196) | moveBefore (u:1723) | **merge** into `MLIR_InsertOpBefore` (detaches first) |
| `ReplaceAllUsesOfValue` (h:908) | RAUW | **14** conv 14 | O(uses) splice; no-op without def-use tracking (c:2075) | replaceAllUsesWith (u:1639) | **rename** `MLIR_ReplaceAllUses`; must work in every context (§2.3) |

#### Op accessors (23)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `GetOpType` (h:656) | Op enum | **194** pp 65 a64 28 wasm 25 opt 22 x64 18 wlift 18 llir 8 conv 5 shr 4 drv 1 | field read (c:843) | DenseMap by OperationName impl (u:560) | **keep** (hottest accessor) |
| `GetOpName` (h:657) | Op name string | **38** conv 8 wasm 7 a64 6 wlift 6 x64 3 drv 2 opt 2 shr 2 llir 1 pp 1 | stored name or switch (c:914) | getStringRef, "module" special case (u:576) | **keep** for diagnostics/printing only; never for dispatch (§2.5) |
| `GetOpName_string` (h:658) | Duplicate of GetOpName | **6** pp 6 | same body (c:921) | forwards (u:583) | **remove** (#145) |
| `GetOpLocation` (h:659) | Op location | **27** conv 23 pp 4 | field read (c:909) | **returns INVALID** (u:584) | **keep**; fix upstream (`op->getLoc()`) |
| `GetOpTrailingComment` (h:660) | Printer metadata | **1** pp 1 | aux read (c:859) | empty (u:585) | **move** to parser/printer side table |
| `GetOpSourceLineStart` (h:661) | Printer metadata | **0** | aux read (c:1781) | -1 (u:588) | **move** (0 uses outside impls) |
| `GetOpUnnumberedLocationDef` (h:662) | Printer metadata | **0** | aux read (c:1786) | INVALID (u:589) | **move** (0 uses) |
| `GetOpNumOperands` (h:663) | Regular operand count | **172** pp 56 wasm 33 opt 22 a64 19 conv 14 wlift 10 llir 7 shr 7 x64 4 | field (c:887) | BranchOpInterface dyn_cast + loop per call (u:593, U8) | **keep**; define as flat count, add successor ranges |
| `GetOpOperand` (h:664) | Operand i | **407** pp 95 a64 84 wasm 57 x64 42 opt 39 wlift 37 llir 29 conv 17 shr 7 | array read (c:892) | ValueBox map lookup (u:606, U1) | **keep** |
| `GetOpNumResults` (h:665) | Result count | **83** wasm 27 a64 12 conv 9 opt 9 shr 7 llir 6 wlift 5 pp 5 x64 3 | field (c:898) | getNumResults (u:609) | **keep** |
| `GetOpResult` (h:666) | Result i | **189** a64 51 wasm 35 wlift 26 x64 25 opt 15 llir 13 conv 12 pp 7 shr 5 | array read (c:903) | ValueBox map lookup (u:612, U1) | **keep** |
| `GetOpNumResultTypes` (h:667) | Result-type count | **30** pp 30 | separate field (c:928) | = getNumResults (u:615) | **remove**; result types = result value types |
| `GetOpResult_type` (h:668) | Result type i | **40** pp 31 llir 5 opt 3 conv 1 | array read (c:933) | getResult(i).getType (u:618) | **rename** `MLIR_GetOpResultType` (or drop) |
| `GetOpNumAttributes` (h:669) | Attribute count | **55** pp 40 conv 6 llir 4 wasm 3 opt 2 | field (c:864) | snapshot vector, copied on first access (u:646, U2) | **keep** for printers/generic code |
| `GetOpAttribute` (h:670) | Attribute i | **56** pp 41 conv 6 llir 4 wasm 3 opt 2 | array read (c:869) | pointer into snapshot (u:650, U2) | **keep**; returns (key, value) |
| `GetOpAttributeByName` (h:674) | Attribute by name | **80** a64 22 x64 17 wasm 12 wlift 11 drv 9 opt 6 conv 3 | strlen + memcmp scan (c:875, N1) | StringAttr::get + heap NamedAttribute per call (u:654, U3) | **replace** by `MLIR_GetOpAttr(op, key)` (§2.6) |
| `GetOpNumRegions` (h:675) | Region count | **86** wasm 20 pp 18 opt 14 conv 9 wlift 7 a64 6 x64 5 llir 4 shr 2 drv 1 | field (c:848) | getNumRegions (u:727) | **keep** |
| `GetOpRegion` (h:676) | Region i | **114** wasm 29 pp 19 opt 14 drv 11 a64 11 wlift 10 x64 8 llir 6 conv 6 | array read (c:853) | &getRegion(i) (u:730) | **keep** |
| `GetOpNumSuccessors` (h:677) | Successor count | **60** conv 26 opt 12 shr 10 pp 7 wlift 2 x64 1 a64 1 llir 1 | field (c:942) | getNumSuccessors (u:734) | **keep** |
| `GetOpSuccessor` (h:678) | Successor i | **56** conv 27 a64 7 pp 6 shr 5 llir 4 opt 4 x64 3 | array read (c:946) | getSuccessor (u:737) | **keep** |
| `GetOpNumSuccessorOperands` (h:679) | Successor operand count | **40** opt 10 shr 9 conv 8 x64 4 pp 4 a64 3 wlift 2 | array read (c:951) | interface dyn_cast per call (u:740, U8) | **replace** by `MLIR_GetOpSuccessorOperandRange(op, s, &start, &n)` |
| `GetOpSuccessorOperand` (h:680) | Successor operand j | **42** opt 14 shr 9 conv 7 pp 7 x64 1 a64 1 wlift 1 llir 1 test 1 | array read (c:956) | interface + ValueBox (u:747) | **merge** into flat `GetOpOperand(start + j)` |
| `GetOpParentBlock` (h:933) | Containing block | **8** conv 3 opt 3 x64 1 llir 1 | field (c:2177) | getBlock (u:1695) | **keep** |

#### Region (7)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `CreateRegion` (h:686) | New detached region | **53** wlift 15 wasm 12 emit 8 pp 7 a64 4 conv 4 test 3 | arena alloc (c:1762) | new Region (u:311) | **keep** (spec can also create regions) |
| `AppendRegionBlock` (h:687) | Append block | **53** wlift 16 wasm 12 pp 7 emit 6 a64 6 test 4 conv 2 | amortized array push (c:804) | push_back (u:317) | **keep** |
| `GetRegionNumBlocks` (h:688) | Block count | **106** conv 23 pp 19 opt 17 wasm 16 a64 9 x64 7 wlift 7 llir 7 drv 1 | field (c:821) | **O(n)** std::distance (u:760, U7) | **keep** + add `MLIR_IsRegionEmpty` |
| `GetRegionBlock` (h:689) | Block i | **169** opt 27 conv 24 wasm 23 a64 22 shr 19 pp 15 wlift 13 drv 10 x64 8 llir 8 | array read (c:826) | **O(i)** std::advance (u:764, U7) | **replace** in loops by `GetRegionFirstBlock`/`GetBlockNext` (§3.2) |
| `InsertRegionBlockAfter` (h:986) | Insert block after | **2** conv 2 | O(n) find + memmove (c:2353) | ilist insert O(1) (u:1800) | **merge** into `MLIR_InsertBlockBefore(region, block, before)` |
| `InsertRegionBlockBefore` (h:993) | Insert block before | **3** conv 3 | O(n) (c:2399) | O(1) (u:1819) | **rename** `MLIR_InsertBlockBefore`: the one block-insertion primitive (detaches first) |
| `MoveBlockToRegionEnd` (h:951) | Move block | **8** conv 8 | O(n) removal (c:2214) | moveBefore (u:1734) | **merge** into `InsertBlockBefore(region, b, NULL)` |

#### Block (21)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `CreateBlock` (h:695) | New detached block | **63** wlift 17 wasm 13 conv 9 pp 8 emit 6 a64 6 test 4 | arena alloc (c:1757) | new Block (u:314) | **keep** |
| `AppendBlockOp` (h:696) | Append op to block | **179** wlift 77 a64 35 wasm 24 emit 17 conv 13 pp 9 test 4 | amortized push (c:736) | push_back + cache invalidation (u:321) | **merge** into `MLIR_InsertOpBefore(block, op, NULL)` / builder |
| `InsertBlockOpBeforeTerminator` (h:697) | Insert before terminator | **0** | **plain append** (bug, c:753) | checks IsTerminator (u:331) | **remove** (0 uses) |
| `InsertBlockOpAtIndex` (h:700) | Insert at index | **15** conv 12 opt 2 emit 1 | O(n) memmove (c:758) | O(idx) ilist walk (u:345) | **replace** by `MLIR_InsertOpBefore(block, op, before)` |
| `AppendBlockArg` (h:701) | Attach a pre-created block-arg value | **21** pp 8 wlift 4 test 4 wasm 3 emit 1 conv 1 | array push, sets def (c:785) | addArgument + RAUW from holder block (u:360) | **replace** by `MLIR_AddBlockArgument` (returns the value) |
| `GetBlockNumOps` (h:702) | Op count | **139** opt 29 a64 19 shr 16 wasm 15 wlift 14 conv 12 x64 10 pp 10 llir 8 drv 6 | field (c:832) | O(n) unless cached (u:771, U7) | **keep** (rarely needed with iterators) |
| `GetBlockOp` (h:703) | Op i | **152** opt 26 a64 25 wasm 19 wlift 19 shr 16 llir 11 x64 10 conv 10 pp 10 drv 6 | array read (c:837) | 16-slot LRU cache, O(i) on miss (u:782, U7) | **replace** in loops by `GetBlockFirstOp`/`GetOpNext` |
| `GetBlockNumArgs` (h:704) | Arg count | **49** conv 17 pp 11 shr 5 a64 4 wasm 4 x64 3 wlift 2 llir 2 opt 1 | field (c:963) | getNumArguments (u:841) | **keep** |
| `GetBlockArg` (h:705) | Arg i | **58** conv 20 pp 12 a64 6 shr 6 wasm 5 x64 4 llir 3 wlift 1 opt 1 | array read (c:968) | ValueBox (u:844, U1) | **keep** |
| `GetBlockIndex` (h:708) | Position in region | **6** pp 6 | O(n) scan (c:974) | O(n) (u:848) | **remove** from core (printer computes its own numbering) |
| `GetBlockOpIndex` (h:937) | Position of op | **0** | O(n) (c:2182) | O(n) (u:1702) | **remove** (0 uses) |
| `GetBlockParentRegion` (h:941) | Containing region | **12** conv 10 emit 1 x64 1 | field (c:2191) | getParent (u:1716) | **keep** |
| `GetBlockTerminator` (h:998) | Last op | **39** conv 37 opt 2 | array read (c:2442) | back() (u:1835) | **keep** (= `GetBlockLastOp`) |
| `GetBlockParentOp` (h:1002) | Op owning the region | **0** | **O(#ops) scan of g_all_ops**; INVALID if tracking off (c:2448) | getParentOp (u:1842) | **keep**; native: region stores parent op (N10) |
| `BlockIsEntry` (h:1005) | First block of region? | **0** | array read (c:2464) | isEntryBlock (u:1851) | **rename** `MLIR_IsBlockEntry` (0 uses today) |
| `GetBlockNumPredecessors` (h:1011) | Predecessor count | **9** conv 9 | O(blocks in region) scan (c:2500) | O(preds) (u:1857) | **replace** by pred iterator / CFG info (§3.2) |
| `GetBlockPredecessor` (h:1017) | Predecessor i | **7** conv 7 | O(region) per call, O(R*P) to enumerate (c:2504) | O(i) (u:1865) | **replace** by `GetBlockFirstPred`/`GetPredNext` |
| `AddBlockArgument` (h:1022) | Create and append block arg | **13** conv 12 opt 1 | arena alloc + array push (c:2530) | addArgument + ValueBox (u:1881) | **keep**; add `MLIR_InsertBlockArgument(block, idx, ...)` |
| `EraseBlockArguments` (h:1029) | Erase block args | **2** conv 1 opt 1 | shift + renumber (c:2563) | eraseArguments + map erase (u:1895) | **keep** |
| `EraseBlock` (h:980) | Detach block | **4** conv 4 | O(n) compaction (c:2338) | remove from ilist, no free (u:1790) | **rename** `MLIR_DetachBlock`; add real erase |
| `SpliceBlockOps` (h:1063) | Move all ops of a block | **3** conv 3 | memcpy + reparent (c:2686) | splice O(1)+reparent (u:1981) | **keep** |

#### Value (11)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `CreateValueBlockArg` (h:721) | Pre-create a block-arg value (deferred) | **27** pp 14 wlift 4 test 4 wasm 3 emit 1 conv 1 | arena IR_Value + aux (c:1734) | heap ValueBox + holder-block arg (u:864, U1) | **remove**; parser uses `AddBlockArgument` + placeholders (§5 U1) |
| `CreateValueOpResult` (h:722) | Pre-create a result value before its op | **66** emit 44 pp 8 conv 6 wlift 3 wasm 2 opt 2 test 1 | arena IR_Value + aux (c:1745) | heap ValueBox, never freed (u:883, U1) | **remove**; results are created by the op (§2.7) |
| `GetValueKind` (h:725) | Block arg or op result | **15** opt 13 pp 2 | field (c:1882) | isa<BlockArgument> (u:899) | **keep**; prefix enumerators (`MLIR_VALUE_BLOCK_ARG`) |
| `GetValueType` (h:726) | Type | **175** pp 67 wasm 43 conv 20 llir 19 a64 9 opt 8 wlift 7 x64 2 | field (c:1887) | ValueBox deref (u:903) | **keep** |
| `GetValueRegisterName` (h:727) | SSA name for printer | **18** pp 18 | aux read (c:1892) | ValueBox string (u:906) | **move** to printer/parser side table |
| `GetValueResultIndex` (h:728) | Result/arg number | **0** | field (c:1897) | getResultNumber/getArgNumber (u:909) | **keep** (needed by passes; 0 uses today) |
| `GetValueDefiningOp` (h:729) | Defining op | **33** a64 10 opt 9 wasm 4 wlift 3 shr 3 x64 2 llir 1 pp 1 | field (c:1902) | OpResult owner (u:915) | **keep** |
| `GetValueLocation` (h:730) | Location | **2** pp 2 | field (c:985) | getLoc (u:1620) | **keep** |
| `GetValueParentBlock` (h:1036) | Defining block | **1** conv 1 | field(s) (c:2581) | getParentBlock (u:1912) | **keep** |
| `GetValueNumUses` (h:1040) | Use count | **4** test 3 conv 1 | O(1) cached; 0 when tracking off (c:2594) | O(uses) loop (u:1920, U9) | **replace** by `MLIR_ValueHasOneUse`/`IsUnused` + use iterator |
| `GetValueUseOwner` (h:1047) | Use i owner + operand index | **2** conv 1 test 1 | O(i) list walk (c:2601) | O(i) (u:1929) | **replace** by `GetValueFirstUse`/`GetUseNext`/`GetUseOwner`/`GetUseOperandIndex` |

#### Types: creation (16)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `CreateTypeInteger` (h:737) | iN | **75** wlift 43 wasm 10 emit 8 pp 5 test 5 conv 3 a64 1 | linear intern scan (c:1152, N3) | IntegerType::get (u:925) | **keep**; drop ignored `is_signed` |
| `CreateTypeFloat` (h:738) | f16/f32/f64/bf16 | **15** wlift 8 wasm 4 emit 2 pp 1 | linear intern (c:1166) | switch, width 8/80/128 silently f32 (u:929) | **keep** |
| `CreateTypeIndex` (h:739) | index | **4** test 2 emit 1 pp 1 | linear intern (c:1174) | IndexType::get (u:939) | **keep** |
| `CreateTypeUnknown` (h:740) | placeholder type | **1** pp 1 | intern (c:1180) | NoneType (u:942) | **move** to parser |
| `CreateTypeTensor` (h:741) | tensor<...> | **3** pp 3 | copy shape + intern (c:1203) | RankedTensorType (u:945) | **keep** (parser/test dialects) |
| `CreateTypeMemref` (h:742) | memref<...> | **3** pp 3 | copy + intern (c:1211) | MemRefType (u:965) | **keep** |
| `CreateTypePointer` (h:743) | !tt.ptr | **3** pp 3 | intern (c:1219) | OpaqueType "tt" with printed elem (u:982) | **move** to parser (Triton test type) |
| `CreateTypeLLVMPointer` (h:751) | !llvm.ptr | **9** wlift 5 test 2 emit 1 conv 1 | intern (c:1228) | LLVMPointerType (u:1276) | **keep** |
| `CreateTypeLLVMPointerInAddressSpace` (h:752) | !llvm.ptr<N> | **1** emit 1 | intern (c:1232) | LLVMPointerType (u:1281) | **merge** with above (`as` param) |
| `CreateTypeLLVMStructIdentified` (h:753) | named struct | **1** emit 1 | linear name scan (c:1294, N4) | getIdentified (u:1287) | **keep**; add literal (anonymous/packed) structs |
| `SetTypeLLVMStructBody` (h:754) | set struct body once | **2** emit 2 | copy fields (c:1298) | setBody (u:1293) | **keep** |
| `CreateTypeLLVMArray` (h:756) | !llvm.array | **17** emit 16 wlift 1 | intern (c:1312) | LLVMArrayType (u:1308) | **keep** |
| `CreateTypeOpaque` (h:799) | opaque type | **1** pp 1 | **ignores name** (c:1995) | OpaqueType "?" (u:1004) | **move** to parser |
| `CreateTypeFunction` (h:802) | builtin function type | **7** emit 6 pp 1 | allocs arrays before intern lookup (c:1445) | FunctionType (u:1009) | **keep** |
| `CreateTypeLLVMFunction` (h:812) | LLVM function type | **3** conv 2 emit 1 | allocs before lookup (c:1320) | LLVMFunctionType (u:1313) | **keep** |
| `CreateTypeLLVMVoid` (h:818) | LLVM void | **2** emit 1 conv 1 | intern (c:1338) | LLVMVoidType (u:1324) | **keep** |

#### Types: mutation (5)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `SetTypeIntegerProperties` (h:821) | Mutate a type in place | **0** | **mutates an interned type** (c:1575) | no-op (u:1021) | **remove** (0 uses, unsound) |
| `SetTypeFloatProperties` (h:822) | same | **0** | mutates interned type (c:1585) | no-op (u:1022) | **remove** |
| `SetTypeTensorProperties` (h:823) | same | **0** | stores caller pointer (c:2002) | no-op (u:1023) | **remove** |
| `SetTypeMemrefProperties` (h:824) | same | **0** | stores caller pointer (c:2011) | no-op (u:1024) | **remove** |
| `SetTypePointerProperties` (h:825) | same | **0** | mutates (c:2020) | no-op (u:1025) | **remove** |

#### Types: introspection (23)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `IsTypeInteger` (h:828) | kind test | **2** conv 1 opt 1 | field (c:2030) | isa (u:1027) | **merge** into `MLIR_GetTypeKind` (#230) |
| `IsTypeFloat` (h:829) | kind test | **2** opt 2 | field (c:2035) | isa (u:1028) | **merge** into `GetTypeKind` |
| `IsTypeTensor` (h:830) | kind test | **0** | field (c:2040) | isa (u:1029) | **merge** |
| `IsTypeMemref` (h:831) | kind test | **0** | field (c:2045) | isa (u:1030) | **merge** |
| `IsTypePointer` (h:832) | kind test (!tt.ptr) | **0** | field (c:2050) | OpaqueType namespace compare (u:1031) | **merge** |
| `IsTypeIndex` (h:833) | kind test | **0** | field (c:2055) | isa (u:1035) | **merge** |
| `IsTypeUnknown` (h:834) | kind test | **0** | field (c:2060) | isa NoneType (u:1036) | **merge** |
| `IsTypeOpaque` (h:835) | kind test | **0** | field (c:2065) | namespace compare (u:1037) | **merge** |
| `IsTypeFunction` (h:836) | kind test | **0** | builtin or LLVM fn (c:1465) | builtin only (u:1041) — **differs** | **merge** |
| `GetTypeFunctionNumInputs` (h:837) | fn inputs | **9** wasm 3 llir 2 x64 1 a64 1 conv 1 pp 1 | field (c:1470) | dyn_cast x2 (u:1042) | **keep** |
| `GetTypeFunctionInput` (h:838) | fn input i | **6** llir 3 wasm 1 conv 1 pp 1 | array (c:1478) | dyn_cast (u:1048) | **keep** |
| `GetTypeFunctionNumResults` (h:839) | fn results (void = 0) | **6** llir 2 conv 2 wasm 1 pp 1 | field (c:1492) | dyn_cast (u:1060) | **keep** |
| `GetTypeFunctionResult` (h:840) | fn result i | **8** llir 3 conv 2 pp 2 wasm 1 | array (c:1507) | dyn_cast (u:1071) | **keep** |
| `GetTypeFunctionIsVarArg` (h:843) | LLVM varargs flag | **7** wasm 4 x64 1 a64 1 llir 1 | field (c:1523) | dyn_cast (u:1085) | **keep** |
| `GetTypeShapedElement` (h:846) | element of tensor/memref | **1** pp 1 | field (c:1530) | ShapedType (u:1093) | **rename** `MLIR_GetTypeElement` (+ vector, array) |
| `IsTypeLLVMStruct` (h:849) | kind test | **12** opt 5 shr 3 x64 1 a64 1 wasm 1 llir 1 | field (c:1539) | isa (u:1101) | **merge** into `GetTypeKind` |
| `GetTypeLLVMStructNumFields` (h:850) | struct fields | **7** opt 3 shr 3 llir 1 | field (c:1543) | getBody().size (u:1104) | **keep**; add `IsOpaque`/`IsPacked` |
| `GetTypeLLVMStructField` (h:851) | field i | **14** opt 5 shr 4 llir 2 x64 1 a64 1 wasm 1 | array (c:1548) | getBody()[i] (u:1109) | **keep** |
| `IsTypeLLVMArray` (h:852) | kind test | **13** opt 4 llir 3 shr 3 x64 1 a64 1 wasm 1 | field (c:1554) | isa (u:1114) | **merge** |
| `GetTypeLLVMArrayElement` (h:853) | array element | **10** shr 3 llir 2 opt 2 x64 1 a64 1 wasm 1 | field (c:1558) | getElementType (u:1117) | **merge** into `GetTypeElement` |
| `GetTypeLLVMArrayNumElements` (h:854) | array length | **6** llir 2 opt 2 shr 2 | field (c:1563) | getNumElements (u:1122) | **rename** `MLIR_GetTypeNumElements` (array + vector) |
| `GetTypeLLVMPointerAddressSpace` (h:855) | ptr address space | **0** | field (c:1569) | dyn_cast (u:1128) | **keep** (0 uses) |
| `GetTypeString` (h:856) | Format a type | **104** pp 81 llir 9 wasm 3 wlift 3 conv 3 x64 2 shr 2 a64 1 | **format() allocates per call** (c:1001, N12) | DenseMap cache + arena copy per call (u:1133, U12) | **keep for printing only**; backends must use kind/width (§6 N12) |

#### LLVM global helpers (3)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `CreateLLVMGlobalString` (h:775) | llvm.mlir.global with string init | **1** emit 1 | CreateOp(UNREGISTERED, "llvm.mlir.global") + string attrs (c:1363) | OpBuilder GlobalOp (u:1329) | **replace** by spec + typed attrs; keep as inline helper |
| `CreateLLVMGlobal` (h:779) | llvm.mlir.global, int/float/region/extern | **12** emit 7 wlift 5 | same, `init_kind` 0..4 magic ints (c:1407) | OpBuilder GlobalOp (u:1350) | **replace** (`init_kind` → enum or separate helpers) |
| `CreateLLVMGlobalArrayInit` (h:792) | array global from bytes | **2** emit 1 wlift 1 | string `value` attr (c:1387) | region of insertvalue chain (u:1391) — **differs** | **replace**; needs a dense-elements attribute (A6) |

#### Attributes: creation (11)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `CreateAttributeInteger` (h:870) | named IntegerAttr | **34** wlift 10 emit 8 test 5 wasm 4 pp 4 a64 1 conv 1 opt 1 | arena attr + **name copy** (c:1594, N2) | `new NamedAttribute` (u:1202, U4) | **change**: drop `name` → `CreateAttributeInteger(ctx, type, v)`; the key is given where the op uses it (§2.6) |
| `CreateAttributeFloat` (h:871) | named FloatAttr | **6** emit 3 wlift 1 opt 1 pp 1 | same (c:1611) | new NamedAttribute (u:1206) | **change**: drop `name` |
| `CreateAttributeBool` (h:872) | named BoolAttr | **7** wasm 2 wlift 2 pp 2 a64 1 | same (c:1620) | new (u:1210) | **change**: drop `name` |
| `CreateAttributeString` (h:873) | named StringAttr | **23** emit 12 wlift 3 wasm 2 conv 2 pp 2 a64 1 test 1 | name + value copies (c:1603) | new (u:1215) | **change**: drop `name` |
| `CreateAttributeLLVMLinkageInternal` (h:874) | linkage attr | **1** emit 1 | string "#llvm.linkage<internal>" (c:1725) | LinkageAttr (u:1220) | **replace** by `CreateAttributeLLVMLinkage(ctx, MLIR_LINKAGE_*)` |
| `CreateAttributeArray` (h:875) | ArrayAttr | **4** conv 2 pp 1 test 1 | copy elements (c:1642) | ArrayAttr (u:1225) | **keep**; drop `name` |
| `CreateAttributeDenseI32Array` (h:878) | DenseI32ArrayAttr | **2** emit 1 conv 1 | **stored as text** "array<i32: ...>" (c:1685, N16) | DenseI32ArrayAttr (u:1260) | **keep**; real storage + `GetAttributeDenseI32` |
| `CreateAttributeDenseI64Array` (h:882) | DenseI64ArrayAttr | **2** emit 1 conv 1 | stored as text (c:1704) | DenseI64ArrayAttr (u:1268) | **keep**; real storage |
| `CreateAttributeDict` (h:884) | DictionaryAttr | **2** pp 1 test 1 | copy elements (c:1651) | DictionaryAttr (u:1237) | **keep** (keyed entries) |
| `CreateAttributeType` (h:887) | TypeAttr | **13** emit 8 conv 2 wlift 1 opt 1 pp 1 | arena (c:1660) | TypeAttr (u:1249) | **keep**; drop `name` |
| `CreateAttributeSymbolRef` (h:891) | FlatSymbolRefAttr | **8** emit 6 wlift 1 conv 1 | string "@name" via format (c:1673) | FlatSymbolRefAttr (u:1253) | **keep**; native stores interned symbol, no "@" |

#### Attributes: introspection (13)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `GetAttributeKind` (h:1107) | kind enum | **59** pp 44 wasm 5 opt 4 llir 3 x64 1 a64 1 conv 1 | switch; INVALID → INTEGER (c:1792) | isa chain, SymbolRef → STRING (u:1458) | **keep**; add SYMBOL, DENSE_*, FASTMATH, LINKAGE kinds |
| `GetAttributeName` (h:1108) | attr name | **72** pp 61 conv 4 llir 3 opt 2 wasm 1 test 1 | field (c:1807) | NamedAttribute name (u:1470) | **replace** by key from `GetOpAttribute` |
| `GetAttributeAsString` (h:1111) | printed form | **25** llir 6 a64 5 wasm 4 opt 4 pp 3 conv 2 x64 1 | string payload or "" (c:1813) — **differs** | prints every call (u:1529, U5) | **move** to printer API; no semantic use |
| `GetAttributeInteger` (h:1112) | int value | **57** pp 21 a64 8 wasm 8 wlift 7 x64 4 opt 4 llir 3 test 2 | union read, no kind check (c:1825) | cast<IntegerAttr> (u:1473) | **keep** |
| `GetAttributeFloat` (h:1113) | float value | **14** pp 4 x64 2 a64 2 wasm 2 llir 2 opt 2 | union read (c:1835) | cast<FloatAttr> (u:1476) | **keep** |
| `GetAttributeType` (h:1116) | type of int/float attr | **3** pp 2 conv 1 | field (c:1840) | dyn_cast (u:1479) | **keep** |
| `GetAttributeBool` (h:1117) | bool value | **17** pp 10 wasm 3 wlift 2 a64 1 llir 1 | union (c:1848) | cast<BoolAttr> (u:1490) | **keep** |
| `GetAttributeString` (h:1118) | string / "@sym" / printed fallback | **71** pp 28 wasm 14 drv 8 x64 5 a64 5 llir 5 conv 4 wlift 2 | union (c:1830) | thread_local caches, prints fallback (u:1493, U5) | **split**: `GetAttributeString` (strings only) + `GetAttributeSymbol` |
| `GetAttributeArraySize` (h:1119) | array size | **3** pp 2 test 1 | field (c:1853) | cast (u:1536) | **keep** |
| `GetAttributeArrayElement` (h:1120) | array elem | **3** pp 2 test 1 | array read (c:1858) | **`new` NamedAttribute per call** (u:1539, U4) | **keep** (value handles fix upstream) |
| `GetAttributeDictSize` (h:1121) | dict size | **5** pp 4 test 1 | field (c:1864) | cast (u:1545) | **keep** |
| `GetAttributeDictElement` (h:1122) | dict entry | **4** pp 3 test 1 | array read (c:1869) | pointer into dict storage (u:1548) | **keep**; return (key, value) |
| `GetAttributeTypeValue` (h:1125) | TypeAttr payload | **31** wasm 9 a64 6 llir 6 x64 4 pp 3 opt 2 conv 1 | field (c:1875) | dyn_cast (u:1485) | **keep** |

#### Locations (11)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `CreateLocationUnknown` (h:1141) | unknown loc (+ original text) | **54** wlift 32 wasm 12 a64 6 pp 2 emit 1 opt 1 | arena alloc per call (c:1956) | UnknownLoc::get (u:1567) | **keep**; native: one shared object, text to parser table |
| `CreateLocationFile` (h:1142) | file:line:col | **1** pp 1 | arena + format text (c:1963) | FileLineColLoc (u:1570) | **keep** |
| `CreateLocationName` (h:1143) | name loc | **1** pp 1 | arena + format (c:1975) | NameLoc (u:1577) | **keep** |
| `CreateLocationRef` (h:1144) | parser `#locN` ref | **1** pp 1 | arena (c:1985) | UnknownLoc, id dropped (u:1582) | **move** to parser |
| `GetLocationKind` (h:1147) | kind | **3** pp 3 | switch (c:1913) | isa chain (u:1586) | **keep** |
| `GetLocationOriginalText` (h:1148) | parser text | **4** pp 4 | field (c:1926) | empty (u:1595) | **move** to parser |
| `GetLocationFileFilename` (h:1149) | filename | **3** pp 3 | field (c:1931) | dyn_cast (u:1598) | **keep** |
| `GetLocationFileLine` (h:1150) | line | **4** pp 4 | field (c:1936) | dyn_cast (u:1603) | **keep** |
| `GetLocationFileColumn` (h:1151) | column | **4** pp 4 | field (c:1941) | dyn_cast (u:1608) | **keep** |
| `GetLocationName` (h:1152) | name | **3** pp 3 | field (c:1946) | dyn_cast (u:1613) | **keep** |
| `GetLocationRefId` (h:1153) | parser ref id | **0** | field (c:1951) | 0 (u:1618) | **move** to parser (0 uses) |

#### Print, parse, pipelines (12)

| Function | Purpose | Call sites | Native | Upstream | Recommendation |
|---|---|---|---|---|---|
| `PrintOperationUpstream` (h:585) | upstream printer | **2** pp 2 | error string stub (c:991) | Operation::print (u:1168) | **move** to `mlir_api_upstream.h` (#151) |
| `PrintOperationClassic` (h:586) | classic printer | **2** pp 2 | mlir_classic_printer.c:2711 | same TU | **keep** in `mlir_printer.h` |
| `PrintOperationGeneric` (h:587) | generic printer | **10** drv 8 pp 2 | mlir_generic_printer.c:395 | same TU | **keep** in `mlir_printer.h` |
| `ParseTextUpstream` (h:591) | upstream parser | **1** pp 1 | stub INVALID (c:996) | parseSourceString (u:1179) | **move** to `mlir_api_upstream.h` |
| `ParseTextClassic` (h:592) | classic parser | **2** pp 2 | mlir_parser.c:1623 | same TU | **keep** in `mlir_parser.h` |
| `LowerToLLVMDialect` (h:627) | in-tree lowering to llvm dialect | **1** drv 1 | mlir_lower_to_llvm.c:820 | same TU | **move** to `mlir_passes.h`, expose as `MLIR_Pass` |
| `TranslateModuleToLLVMIR` (h:632) | llvm dialect → .ll text | **1** drv 1 | mlir_translate_to_llvm_ir.c:1438 | same TU | **move** to `mlir_passes.h` |
| `TranslateModuleToWasm` (h:645) | llvm dialect → wasm object | **1** drv 1 | mlir_translate_to_wasm.c:21 | same TU | **move** to `mlir_passes.h` |
| `LowerToLLVMDialectUpstream` (h:648) | upstream PassManager lowering | **1** drv 1 | link error (not defined) | u:2091 | **move** to `mlir_api_upstream.h` |
| `TranslateModuleToLLVMIRUpstream` (h:649) | upstream translate | **1** drv 1 | link error | u:2355 | **move** |
| `TranslateModuleToWasmUpstream` (h:650) | upstream wasm codegen | **1** drv 1 | link error | u:2394 | **move** |
| `LiftCfToScf` (h:1092) | cf → scf structurizer | **1** wasm 1 | mlir_lift_cf_to_scf.c via c:2722 | eraseUnreachableBlocks + same C port (u:2259) | **move** to `mlir_passes.h` as a pass |


### 1.2 The `MLIR_OpType` enum

`h:68-517`: 282 enumerators before `OP_TYPE_COUNT` — `OP_TYPE_UNREGISTERED`
plus 281 registered ops, each with a unique name in
`mlir_op_names.c:19-314` (uniqueness asserted at `mlir_op_names.c:371`).

| Dialect | Ops | Used by the compiler? | Registered upstream (`u:100-112`)? |
|---|---|---|---|
| `aarch64` | 64 | yes (a64) | no — unregistered ops upstream |
| `llvm` (incl. 8 `llvm.intr.*`, 7 `llvm.mlir.*`) | 62 | yes (674 enumerator references outside parser/printers) | yes |
| `wasmstack` | 38 | yes | no |
| `arith` | 36 | yes (emit, conv) | yes |
| `wasmssa` | 31 | yes | no |
| `tt` (Triton) | 15 | parser tests only | no |
| `scf`, `func`, `cf`, `builtin` (module, unrealized cast) | 6, 5, 3, 2 | yes | yes |
| bare `return` | 1 | yes (`mlir_lower_to_llvm.c:730`) | no |
| `memref` 4, `tensor` 3, `std` 2, `math` 2, `linalg` 2, `affine` 2, `vector` 1, `index` 1, `gpu` 1 | 18 | parser tests only | yes, except `std` (no longer exists upstream) |

Observations:

- **Organization.** One flat enum, grouped by dialect with comments, but not
  strictly: `OP_TYPE_ARITH_XORI/SHLI/SHRSI` sit inside the LLVM block
  (`h:244-246`). Five enumerators are never referenced outside the name table
  (`ARITH_MAXF`, `ARITH_MINSI`, `MATH_EXP`, `MATH_LOG`, `MEMREF_DEALLOC`).
- **Name table.** `op_type_to_string` is two chained `switch`es
  (`mlir_op_names.c:19`, `:174`, `:316`), split because one switch exceeded
  the wasm→Mach-O backend's 256-deep block stack (comment at
  `mlir_op_names.c:4-9`). The reverse map is a 1024-slot FNV hash built on
  first use (`mlir_op_names.c:325-389`). Tables indexed by op type use the
  literal `OP_TYPE_TABLE_CAPACITY 512` because tinyC rejects an enumerator as
  a global array length (`mlir_op_names.h:16-25`).
- **Facts about ops live elsewhere.** Purity (`mlir_llvm_op_traits.h:16`),
  "has result" (`mlir_wasmssa_to_wasmstack.c:153`), commutativity
  (`mlir_llvm_arith_gvn.c:145`), the dialect of an op
  (`mlir_llvm_to_aarch64.c:3705`, by name prefix) are each re-derived per
  file.

**Proposal: a single `mlir_ops.def`.**

```c
// mlir_ops.def — one line per op; included with different MLIR_OP definitions.
//      enumerator               dialect   name                   traits
MLIR_OP(OP_TYPE_LLVM_FADD,       LLVM,     "llvm.fadd",           MLIR_T_PURE | MLIR_T_ELEMENTWISE | MLIR_T_FASTMATH)
MLIR_OP(OP_TYPE_LLVM_BR,         LLVM,     "llvm.br",             MLIR_T_TERMINATOR | MLIR_T_BRANCH)
MLIR_OP(OP_TYPE_LLVM_LOAD,       LLVM,     "llvm.load",           MLIR_T_MEM_READ)
```

It generates, in one place: the enum (`h`), `static const string
g_op_names[]` (an array, so no switch depth problem and O(1) lookup), `static
const uint8_t g_op_dialect[]`, and `static const uint64_t g_op_traits[]`
(§3.1). Ops of one dialect are contiguous, with `MLIR_DIALECT_<D>_FIRST/LAST`
markers, so "is an llvm op" is a range check. tinyC must preprocess X-macros
(it already handles `DEFINE_VECTOR_FOR_TYPE`, `h:59`); verify that it accepts
re-including a file with a redefined macro when the change is made.

**Adding a dialect:** (1) add a block of `MLIR_OP` lines to `mlir_ops.def`
with traits, and attribute keys to `mlir_attr_keys.def` (§2.6); (2) if the
dialect exists upstream, register it in `UpstreamCtx` (`u:100-112`) and add a
translation test; otherwise it is an unregistered dialect upstream
(`allowUnregisteredDialects`, `u:115`), which already works for
aarch64/wasmssa/wasmstack; (3) add the printer/parser forms if the classic
printer needs a custom syntax; (4) enumerator values are not ABI-stable —
everything is rebuilt together. Dialects used only by tests (D9) should be
registered dynamically by the parser (§2.5) instead of occupying the static
enum.

## 2. Design rules

### 2.1 Naming

| Rule | Today's exceptions to fix |
|---|---|
| Functions are `MLIR_<Verb><Entity>[<Property>]`. Verbs: `Create` (new object, or uniqued type/attribute), `Get`/`Set`, `Is`/`Has` (return `bool`), `Insert`/`Append`/`Move`/`Detach`/`Erase`/`Replace` (structure), `Build` (create and insert through a builder), `Walk`. | `BlockIsEntry` → `IsBlockEntry` |
| Collections: `MLIR_Get<Entity>Num<Things>(h)` and `MLIR_Get<Entity><Thing>(h, i)`; iteration: `MLIR_Get<Entity>First<Thing>(h)` and `MLIR_Get<Thing>Next(x)`. | — |
| CamelCase only; no `_` after the prefix; no doubled prefix. | `GetOpResult_type`, `GetOpName_string`, `MLIR_MLIR_OpTypeToString` |
| Type names may abbreviate (`MLIR_AttrKind`, `MLIR_AttrKey`); function names spell `Attribute`. | — |
| Enumerators carry their enum's prefix: `MLIR_<ENUM>_<NAME>`. The op enum keeps its established `OP_TYPE_` prefix (D8: renaming 282 enumerators and ~2,200 references buys nothing). | `BLOCK_ARG`, `OP_RESULT` (`h:715-718`) → `MLIR_VALUE_BLOCK_ARG`, `MLIR_VALUE_OP_RESULT` |
| Text is `string` (pointer + length), never `const char *` (printf-style format strings excepted). | `GetOpAttributeByName(op, const char *)` (`h:674`) |
| Indices and counts are `size_t` in the API; out-parameters come last and accept `NULL`. | — |
| Backend-only functions live in a backend header with a backend suffix (`mlir_api_upstream.h`, #151); the core header declares only what both backends implement. | five `*Upstream` functions in `h:585-650` |
| Passes and tools built on the API (`LowerToLLVMDialect`, `Translate*`, `LiftCfToScf`) live in `mlir_passes.h`, not in the core header; parser/printer-only functions live in `mlir_parser.h`/`mlir_printer.h`. | `h:51-62`, `h:585-650`, `h:1066-1093` |

### 2.2 Handle types

Today all seven handle kinds are `uintptr_t` (`h:21-27`), so passing a block
where an op is expected compiles silently. Three options:

| Option | Type safety | Cost | Ergonomics |
|---|---|---|---|
| A. `uintptr_t` (today) | none | zero | `==`, `0`, hashing as integers |
| B. `typedef struct { uintptr_t p; } MLIR_OpHandle;` (roadmap A4 text) | compile error on mixing | zero on SysV x86-64, AAPCS64 and clang's wasm32 ABI (single-scalar structs travel in registers); **unknown for tinyC's own ABI**, which matters for the self-hosted compiler's speed (#225) | no `==` in C: needs `MLIR_OpEq(a, b)` or `.p` everywhere; `MLIR_INVALID_HANDLE` becomes a compound literal; every hash map in the passes changes |
| C. `typedef struct MLIR_OpImpl *MLIR_OpHandle;` (pointer to a distinct incomplete struct) | a diagnostic on mixing (`-Wincompatible-pointer-types`; an error by default in GCC ≥ 14; build with `-Werror=incompatible-pointer-types`) | zero everywhere, including tinyC: it is a pointer | `==`, `NULL`, `!h` keep working; hash with `(uintptr_t)h` |

**Recommendation: C.** `#define MLIR_INVALID_HANDLE NULL` keeps every
existing `h == MLIR_INVALID_HANDLE` comparison compiling; the migration is
mostly mechanical (casts where handles are stored in `uintptr_t` maps,
e.g. `mlir_llvm_dce.c:20`, `mlir_llvm_mem2reg.c:21`). Both backends already
store real pointers in every handle (`c:253-279`; upstream `u:177-187`), so no
backend changes. Upstream MLIR's own C API chose struct wrappers (B); we
prefer C because our hot loops compare and hash handles constantly. Add two
handle kinds that §3.2 needs: `MLIR_UseHandle` (an operand slot; upstream
`mlir::OpOperand *`) and `MLIR_PredHandle` (a predecessor edge; upstream
`mlir::BlockOperand *`).

A–C choose how a handle is typed. What a handle *holds* on the native
backend is a separate choice:

| Option | What the native handle holds | Reads | Storage |
|---|---|---|---|
| Pointer (before #354) | the address of an object allocated individually in the IR arena (`arena_new` per op, value, block, region, interleaved with operand arrays) | `GetOpNumOperands(op)`: one load | the arena never moves or reallocates, so addresses stay valid |
| D. Index | a 32-bit index into a context-owned vector per kind (ops, values, blocks, regions); 0 is invalid | `GetOpNumOperands(ctx, op)`: every read takes the context, because resolving an index needs the vector's base | vectors may grow by reallocation (the index survives, a pointer would not); erased ops stay as empty slots; every stored reference is 4 bytes instead of 8; dense ids let passes use arrays instead of hash maps for per-op/per-value data |

D applies to the native backend only; the upstream backend stays pointers
(its handles wrap `mlir::Operation *` etc.). D composes with the typing
options: with B the struct holds the index, with C the native backend stores
the index in the pointer-typed handle and never dereferences it.

History. D is the original plan. Until #69 the top of `mlir_api_impl.c`
carried a TODO to replace the vector of pointers with "a vector of indices,
store all operations in one vector", because "you can't double and copy the
vector allocation" while pointers into it exist ("using indices is probably
the best way"), with per-object arrays of indices still in an arena and easy
serializing as a bonus. Commit a56e8e3 (#69, "Use handles instead of
pointers") removed that TODO and introduced the `uintptr_t` handles, but
implemented them as arena pointers: `resolve_op(h)` was a cast.

Decision. `bench/ir_storage` (#370, not merged) implemented a minimal IR
core once per storage design behind one API and timed construction, an op
walk, a use-list walk and a fold+DCE rewrite at 10k, 100k and 1M ops, built
with gcc, clang and tinyC. The pre-#354 layout was 1.6–2.5× slower than
either compact layout. Between compact layouts, indices into one operand pool
(B-pool) took 0.95–1.06× the time of arena pointers up to 100k ops and won
at 1M (0.66–0.74×, 108 vs 164 MB peak RSS). Growing the vectors by doubling
alone cost 2× in construction; reserving them from a size hint removed that
cost and was the fastest variant at every size. The owner chose D with the
B-pool layout, flat vectors reserved from a hint and grown by doubling.

As implemented (#354):
- **API.** Every getter of an op, region, block or value takes the context as
  its first parameter (`MLIR_GetOpOperand(ctx, op, i)`). The upstream backend
  uses it only for an ownership check under `MLIR_API_CHECKS`. Getters of
  types, attributes and locations are unchanged: types stay uniqued pointers
  and attributes and locations arena pointers until their interning (#349,
  #350) makes them indices too. Most sketches in §3 and §4 omit the context
  for brevity.
- **Layout** (`mlir_api_impl.c`, "IR storage"). Vectors per kind: ops (56 B),
  values (40 B), operands (16 B: `{value, owner, next, prev}`, all indices),
  blocks and regions. An op's regular and successor operands are consecutive
  in the operand vector (D5); the def-use lists are threaded through the
  operand records, so a use is an operand index. An op's results are
  consecutive values, so `GetOpResult(ctx, op, i)` is `first_result + i`
  (`MLIR_CreateOp` requires its result values to be consecutive). Region and
  successor `(block, operand count)` entries share a u32 vector; attribute
  handles, which are still pointers, share a pointer-sized vector. Block op
  and argument lists and region block lists are u32 arrays in the IR arena.
  Erased ops stay as empty slots.
- **Growth.** One flat array per vector. `MLIR_ContextOptions.expected_ops`
  and `MLIR_SetContextExpectedOps` reserve the vectors that grow with the
  op count (ops, values, operands, attribute entries) from a hint (tinyc passes 1.5 ops per token, 6 when compiling to wasm, and
  1.7 ops per byte of wasm for `--from-wasm`). The reservation is capped at
  1 GiB of buddy blocks: corec's heap is a 4 GiB reservation (and wasm32
  memory is capped at 4 GiB), and a reserved block uses it up even where it
  is never touched. A full vector doubles and copies, through the one
  hook `vec_reserve`, which a later change can make reserve address space
  instead. Inside the backend no pointer into a vector is held across a call
  that can grow it; `MLIR_API_CHECKS` builds check that with a per-vector
  generation counter, and tag handles with their kind and a per-context salt
  so a handle of another kind or context is caught.
- **Lifetimes.** Freeing an IR arena still invalidates the IR created in it,
  but no longer returns the op, value and operand records, which live in the
  context until it is destroyed. Measured on the pipelines that free arenas
  per phase or per function, peak RSS still went down (`--from-wasm
  --emit=elf` 2.70 → 2.02 GB, the `--emit=macho --macho-backend=llvm` lift
  2.77 → 2.05 GB, the wasm32 self-host lift under wasmtime 2.91 →
  2.61 GB), so there is no mark/release yet.
- **Results.** Against the pointer storage, with the gcc `-O3 -flto` build of
  tinyc (medians of 7, `--time-stages`): compiling `emit.c` with
  `--emit=x64` takes 0.86× the time (emit 0.68×, mem2reg 0.72×,
  lower-to-llvm 0.40×) and 0.86× the peak RSS; the self-host lift
  (`--from-wasm --emit=elf`, 7.5M ops) 0.74× the time (8.7 → 6.5 s) and
  0.75× the peak RSS. The self-hosted, tinyC-built compiler gets 0.92× and
  0.84×. The IR-heavy stages gain 1.3–2.5×; the rest of the compile time is
  preprocessing, register allocation and encoding, which barely touch IR
  storage. The PR that implemented it has every table.

### 2.3 Ownership and lifetimes

Today:

- `MLIR_Context` is a public struct with an arena and two options
  (`h:31-49`). The native backend keeps **process-wide** registries
  (`g_all_ops` `c:286`, `g_type_handles` `c:509`, `g_struct_*` `c:1244`) that
  point into whichever arena was current, hence `MLIR_ResetInternRegistry`
  (`c:1287`) and the save/restore dance in the driver
  (`driver.c:112-120`, `:392-400`, `:532-533`, `:1102-1103`). The upstream
  backend ignores the context entirely and uses one static `MLIRContext`
  (`u:96-118`) plus static caches (`u:568`, `u:1152`, `u:1499`, `u:1515`).
  #140 is one consequence.
- **Erase differs:** native `EraseOp` keeps the op shell readable
  (`c:2125-2127`, and `mlir_llvm_dce.c:72` relies on
  `GetOpParentBlock(op) == INVALID` to recognise an erased op); upstream
  `op->erase()` frees it (`u:1661`), so the same check is a use-after-free.
- **Def-use is optional** (`no_def_use_tracking`, `h:33-40`). Without it,
  `ReplaceAllUsesOfValue` silently does nothing (`c:2081`),
  `GetValueNumUses` returns 0 (`c:2598`) and `GetBlockParentOp` returns
  INVALID (`c:2454`, because `g_all_ops` is not filled, `c:394`). Passes
  that must work in both modes rebuild use counts by hand
  (`mlir_llvm_dce.c:1-8`, `mlir_llvm_mem2reg.c:8-12`).

Rules:

1. **The context owns everything that is not IR**, and becomes opaque:
   `MLIR_Context *MLIR_CreateContext(const MLIR_ContextOptions *)`,
   `MLIR_DestroyContext`. It holds the op-type table (static + dynamic),
   the attribute-key table, type and attribute uniquing tables, a persistent
   arena for them (today's `type_arena`, made the default), the diagnostic
   handler, the target description (§4), and upstream's `MLIRContext` and
   caches. No process-wide mutable state on either backend.
2. **IR lives in the current IR arena** (`MLIR_SetArenaAllocator` stays):
   ops, blocks, regions, values and their operand storage. Freeing an arena
   invalidates the IR created in it and nothing else, so the wasm-lift
   pipeline's arena swapping keeps working without
   `MLIR_ResetInternRegistry`, which is removed.
3. **Handles are borrowed pointers.** An op handle is valid until the op is
   erased, its arena is freed or the context is destroyed. **After
   `MLIR_EraseOp` the handle is dead on both backends.** Add
   `MLIR_DetachOp(op)` for unlink-and-keep (today's native erase behaviour,
   used for moves).
4. **Returned strings are borrowed**: valid while the object (for names:
   the context) lives. Getters never allocate. Formatting and printing take
   an explicit output arena (`MLIR_PrintType(ctx, t, Arena *out)`); today
   native `GetTypeString` allocates on every call (`c:1016`) and upstream's
   `GetAttributeString` grows thread-local caches forever (`u:1499`,
   `u:1515`).
5. **Input arrays are copied** (as today, `c:615-624`): callers may pass
   stack arrays and compound literals.
6. **Def-use is always available** (D3). Intrusive operand lists inline in
   the op (§6 N6) cost 32 B per operand instead of today's 48 B with tracking
   on — and 8 B with tracking off. If A9 shows the always-on cost is within
   budget (D3), remove `no_def_use_tracking`; otherwise keep it as a context
   option, but functions that need uses `assert` instead of silently
   returning wrong answers.

### 2.4 Errors and diagnostics

Today: accessors bounds-check and return `MLIR_INVALID_HANDLE`/0 on misuse
(`c:839`, `c:894`, …); some mismatches are silently different per backend
(`GetAttributeKind(INVALID)` is `INTEGER` natively, `c:1794`;
`GetAttributeInteger` on a non-integer reads the union natively, `c:1827`,
and hits `llvm::cast` upstream, `u:1474`; `load_cse`/`arith_gvn` call it on
every attribute regardless of kind, `mlir_llvm_arith_gvn.c:197-201`,
`mlir_llvm_load_cse.c:203-208`).
Failures print with `fprintf(stderr, …)` from the backends (18 in
`mlir_llvm_to_aarch64.c`, 12 in x64, 10 in `mlir_llvm_to_wasmssa.c`) and
return `false`. Upstream defines `UNIMPLEMENTED()` (`u:88`) but never uses it;
native defines stubs for upstream-only entry points that return error strings
(`c:991-999`).

Rules:

1. **Contract violations** (wrong handle kind, null where not allowed, index
   out of range, kind mismatch in a typed getter, erasing an op that still
   has uses) are bugs: checked by `assert` when `MLIR_API_CHECKS` is defined
   (debug builds, CI), not checked in release builds. The API does not
   promise defined results for them.
2. **Recoverable failures** (parse error, verifier failure, pass failure,
   unsupported construct in a backend, translation failure) return `bool`
   and report through the context's diagnostic handler:

   ```c
   typedef enum { MLIR_DIAG_ERROR, MLIR_DIAG_WARNING, MLIR_DIAG_NOTE, MLIR_DIAG_REMARK } MLIR_DiagSeverity;
   typedef void (*MLIR_DiagnosticHandler)(void *user, MLIR_DiagSeverity sev,
                                          MLIR_LocationHandle loc, string message);
   void MLIR_SetDiagnosticHandler(MLIR_Context *ctx, MLIR_DiagnosticHandler fn, void *user);
   void MLIR_EmitDiagnostic(MLIR_Context *ctx, MLIR_DiagSeverity sev,
                            MLIR_LocationHandle loc, const char *fmt, ...);
   ```

   The default handler prints `file:line:col: error: message` to stderr.
   Upstream installs an `mlir::ScopedDiagnosticHandler` that forwards
   `mlir::Diagnostic`s to the same handler, so upstream verifier and pass
   errors arrive in the same place.
3. **Not implemented on a backend** is a link error, not a run-time stub
   (follows from §2.1's header split).
4. A trait-driven `bool MLIR_VerifyOp(MLIR_Context *, MLIR_OpHandle)` (§3.1)
   runs between passes in debug pipelines.

### 2.5 Enum-first op identity (#51, #145)

Today a non-empty name *decides* the type at creation (`h:539-544`,
`c:599-607`, `u:397-407`): every creator passes both, e.g.
`emit.c:379`, so native creation does a string compare per op
(`c:602-603`) and upstream builds a `std::string` and resolves the
`OperationName` by string hash per op (`u:397-412`). Registered ops are still
created by name with `OP_TYPE_UNREGISTERED` in nine places (`emit.c:2555`,
`:3299`, `:4244`, `:4321`, `:4324`, `:6199`, `:7554`;
`mlir_wasmssa_to_llvm.c:406`, `:421`) and in the native global helpers
(`c:1373`, `c:1401`, `c:1439`). Name-based dispatch remains in
`driver.c:84-87`, `driver.c:229-231`, `mlir_lower_to_llvm.c:773`,
`mlir_llvm_to_aarch64.c:3705-3708` (dialect by prefix),
`mlir_wasmssa_to_llvm.c:1661-1669`, and CSE/GVN hash op *names* into their
value-number keys (`mlir_llvm_load_cse.c:199`, `mlir_llvm_arith_gvn.c:193`).

Rules:

1. **An op's identity is its `MLIR_OpType`.** Names exist for printing,
   parsing and diagnostics only. No API function takes an op name except
   name lookup and registration.
2. **Creation takes only the type** (`MLIR_OpSpec.type`, §2.7).
3. **Unknown names get dynamic types (D2).**

   ```c
   // Returns the registered or previously registered type for `name`, or
   // registers a new one (an id >= OP_TYPE_COUNT, stable for the context).
   MLIR_OpType MLIR_RegisterOpType(MLIR_Context *ctx, string name);
   // OP_TYPE_UNREGISTERED if `name` is unknown; never registers.
   MLIR_OpType MLIR_LookupOpType(MLIR_Context *ctx, string name);
   string      MLIR_GetOpTypeName(MLIR_Context *ctx, MLIR_OpType type);
   ```

   Ops store only the integer. `MLIR_GetOpName(op)` becomes
   `MLIR_GetOpTypeName(ctx, MLIR_GetOpType(ctx, op))`; `GetOpName_string` is
   removed (#145). Dynamic types have no traits (conservative: not pure, may
   read and write memory), so passes treat them like an unknown call.
   `OP_TYPE_UNREGISTERED` survives only as the lookup-miss value. This
   satisfies #51 (no op stores a name for a registered type) and the
   preference stated in #145 (integers internally, a user-visible
   registration step), without the alternative of storing a name on each
   unregistered op (#145's other option), which keeps two code paths and a
   per-op string.
4. **Upstream mapping.** The context builds `mlir::OperationName
   opName[n_types]` once (registered ops: `OperationName(name, mctx)`;
   dynamic ones on registration). `CreateOp` uses
   `OperationState(loc, opName[type])` — no string, no hash.
   `GetOpType` keeps the `OperationName`-impl → enum `DenseMap` (`u:560-575`)
   but per context.

**As implemented (step 5, #347).** The rules above hold; where the code
differs from the sketches in §1.2 and here:

- **The static names are one string.** tinyC accepts no pointer in a global
  initializer, so `static const string g_op_names[]` would not self-host.
  `mlir_ops.def` instead generates `static const char g_static_op_names[]`,
  the names in enum order separated by NULs (tinyC now concatenates adjacent
  literals in a global initializer). At creation each context walks it once
  into one `string names[]` (static types first, then the dynamic ones) and
  builds the name → type hash (`mlir_op_names.c`). There is no switch and no
  lazily built global table: op names and types are per context. (The
  wasmstack lifter still has two diagnostic-only globals,
  `g_current_ctx`/`g_current_op_type`, for its stack-underflow message.) As
  a consequence `MLIR_GetOpTypeName` takes the context for every type, and
  `MLIR_GetOpName(op)` is `MLIR_GetOpTypeName(ctx, MLIR_GetOpType(ctx, op))`.
- **`mlir_ops.def` has two columns**, `MLIR_OP(enumerator, name)`; step 6
  adds the traits. A dialect's ops sit between `MLIR_DIALECT_BEGIN(D)` and
  `MLIR_DIALECT_END(D)`, which produce `MLIR_DIALECT_<D>_FIRST` and `_LAST`
  enumerators without shifting the values;
  `MLIR_OP_TYPE_IN_DIALECT(t, D)` is one unsigned comparison.
- **Parser-only ops (D9).** They are `MLIR_PARSER_OP(kind, name)` lines.
  The parser and both printers register these names in their context
  (`parser_op_types_init`, `mlir_printer.h`) and dispatch on an *op kind*:
  the `MLIR_OpType` of a static op, a `PARSER_OP_*` value (from
  `OP_TYPE_COUNT` up) for a parser-only op, and `OP_TYPE_UNREGISTERED` for
  any other dynamic type. One switch handles all three, and the kind of a
  dynamic type is a subtraction because the parser registers its names
  consecutively. The two helpers are out of line in `mlir_op_names.c`:
  tinyC emits a header's `static inline` functions into every file that
  includes it. Four names that only the classic printer tests
  (`affine.yield`, `linalg.yield`, `tensor.from_elements`,
  `gpu.terminator`) are `MLIR_PRINTER_OP` lines of the same kind.
- **Upstream.** The context binds `mlir::OperationName` for every static
  type at creation (`builtin.module` for `OP_TYPE_MODULE`, whose C name is
  `module`; `MLIR_OP_ALIAS` makes `builtin.module` a second name of the
  type, so registering it cannot create a dynamic type that would take over
  the module's `OperationName`) and for each dynamic type when it is
  registered. A name already bound to a type is never rebound. `GetOpType`
  registers names it has not seen, such as ops the upstream parser or a
  pass created.

### 2.6 Attribute access without strings (A3)

Today an attribute handle is a *named* attribute: natively an
`IR_Attribute` that stores a copied name string (`c:123`, `c:552-557`),
upstream a heap `mlir::NamedAttribute` (`u:1196-1200`). Lookup is by C string
(`c:875-885`, `u:654-726`). Rules:

1. **Attribute handles are values** — immutable and, where cheap,
   uniqued: `MLIR_CreateAttributeInteger(ctx, type, v)` and the other
   `CreateAttribute*` functions lose their `name` parameter.
   Upstream the handle is the `mlir::Attribute` storage pointer, so creating
   and reading attributes allocates nothing in the API layer.
2. **Names are keys.** `typedef uint32_t MLIR_AttrKey;` Registered keys
   come from `mlir_attr_keys.def` (`MLIR_KEY_VALUE`, `MLIR_KEY_SYM_NAME`,
   `MLIR_KEY_PREDICATE`, `MLIR_KEY_CALLEE`, `MLIR_KEY_GLOBAL_NAME`,
   `MLIR_KEY_ELEM_TYPE`, `MLIR_KEY_RAW_CONSTANT_INDICES`,
   `MLIR_KEY_FASTMATH_FLAGS`, `MLIR_KEY_RD`, `MLIR_KEY_RN`, …: the 18
   distinct literal names in today's `GetOpAttributeByName` calls plus those
   behind the `attr_i`/`at_i` helpers). Other names are interned per context.

   ```c
   typedef struct { MLIR_AttrKey key; MLIR_AttributeHandle value; } MLIR_NamedAttr;

   MLIR_AttrKey MLIR_InternAttrKey(MLIR_Context *ctx, string name);
   string       MLIR_GetAttrKeyName(MLIR_Context *ctx, MLIR_AttrKey key);

   MLIR_AttributeHandle MLIR_GetOpAttr(MLIR_OpHandle op, MLIR_AttrKey key);  // NULL if absent
   bool MLIR_GetOpAttrInt(MLIR_OpHandle op, MLIR_AttrKey key, int64_t *out); // false if absent
   void MLIR_SetOpAttr(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key, MLIR_AttributeHandle v);
   void MLIR_RemoveOpAttr(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key);
   size_t         MLIR_GetOpNumAttributes(MLIR_OpHandle op);
   MLIR_NamedAttr MLIR_GetOpAttribute(MLIR_OpHandle op, size_t i);
   ```

3. **Native:** an op stores `MLIR_NamedAttr[]` (16 B each);
   `GetOpAttr` is a scan comparing `uint32_t`s over typically ≤ 5 entries —
   no `strlen`, no `memcmp`, no name copies. **Upstream:** the context keeps
   `mlir::StringAttr keyAttr[n_keys]`; `GetOpAttr` is
   `op->getAttr(keyAttr[key])` (inherent properties included, as
   `u:664-673` does by hand for `llvm.call` today), returning the
   `Attribute` pointer.
4. **Typed payloads, no text.** Add real storage and getters for dense
   arrays (`MLIR_GetAttributeDenseI32(attr, const int32_t **data, size_t *n)`;
   today native stores `"array<i32: …>"` text, `c:1685-1719`, re-parsed in
   six places, §6 N16), symbols (`MLIR_GetAttributeSymbol`, no `@`), linkage
   (`MLIR_LLVMLinkage` enum), fast-math flags (bitmask), and dense elements
   (splats, §4). `GetAttributeAsString` moves to the printer API; no pass
   should depend on printed attribute text.

### 2.7 Builder API (A5)

Today one op costs, in `emit.c`'s `emit_binop` (`emit.c:517-527`): a
`CreateValueOpResult` call (with a formatted `"%N"` register name,
`emit.c:347-350`), three scratch arrays, and a 17-parameter `CreateOp`
(`h:545-557`; 19 argument words, 13 of them on the stack on SysV x86-64), which copies the
arrays again (`c:625-630`), compares the name (`c:602`), allocates `aux`,
`operand_uses` and one `UseNode` per operand (`c:640-696`); then
`AppendBlockOp`. Upstream adds a heap `ValueBox` per result and a
`std::string` per op (§5).

Proposal: one data-driven entry point, an insertion point, and inline shape
helpers.

```c
typedef struct MLIR_OpSpec {
    MLIR_OpType                 type;
    const MLIR_ValueHandle     *operands;      // regular operands, then the operands of
    uint32_t                    n_operands;    // successor 0, successor 1, ... (flat, D5)
    const MLIR_TypeHandle      *result_types;  // results are created by the op
    uint32_t                    n_results;
    const MLIR_NamedAttr       *attrs;
    uint32_t                    n_attrs;
    uint32_t                    n_regions;     // fresh empty regions...
    const MLIR_RegionHandle    *regions;       // ...or, if non-NULL, these are adopted
    const MLIR_BlockHandle     *successors;
    uint32_t                    n_successors;
    const uint32_t             *successor_operand_counts; // n_successors entries; NULL = all 0
    MLIR_LocationHandle         loc;           // NULL = unknown / builder default
} MLIR_OpSpec;

// Create a detached op.
MLIR_OpHandle MLIR_CreateOpFromSpec(MLIR_Context *ctx, const MLIR_OpSpec *spec);

// Insertion point: before `before`, or at the end of `block` if `before` is NULL.
typedef struct MLIR_Builder {
    MLIR_Context       *ctx;
    MLIR_BlockHandle    block;
    MLIR_OpHandle       before;
    MLIR_LocationHandle loc;     // default location for created ops
} MLIR_Builder;

static inline MLIR_Builder MLIR_BuilderAtEnd(MLIR_Context *c, MLIR_BlockHandle b);
static inline MLIR_Builder MLIR_BuilderBefore(MLIR_Context *c, MLIR_OpHandle op);
static inline MLIR_Builder MLIR_BuilderAfter(MLIR_Context *c, MLIR_OpHandle op);

// Create and insert at the builder's position.
MLIR_OpHandle MLIR_BuildOp(MLIR_Builder *b, const MLIR_OpSpec *spec);

// Structural insertion (replaces AppendBlockOp / InsertBlockOpAtIndex /
// MoveOpToBlockEnd / InsertBlockOpBeforeTerminator). Detaches `op` from its
// current block first, so it also moves ops.
void MLIR_InsertOpBefore(MLIR_Context *ctx, MLIR_BlockHandle block,
                         MLIR_OpHandle op, MLIR_OpHandle before /* NULL = end */);

// Successor operand ranges in the flat operand list.
void MLIR_GetOpSuccessorOperandRange(MLIR_OpHandle op, size_t succ,
                                     size_t *first, size_t *count);
```

Shape helpers are `static inline` in the header, so they cost one call to
`MLIR_BuildOp` and nothing else:

```c
static inline MLIR_ValueHandle MLIR_BuildBinary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y) {
    MLIR_ValueHandle ops[2] = {x, y};
    MLIR_OpSpec s = {.type = t, .operands = ops, .n_operands = 2,
                     .result_types = &rt, .n_results = 1};
    return MLIR_GetOpResult(MLIR_BuildOp(b, &s), 0);
}
static inline MLIR_ValueHandle MLIR_BuildUnary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x);
static inline MLIR_ValueHandle MLIR_BuildFBinary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y,
        MLIR_FastMath fmf);               // adds MLIR_KEY_FASTMATH_FLAGS (§4)
static inline MLIR_ValueHandle MLIR_BuildConstantInt(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, int64_t v);   // t = OP_TYPE_ARITH_CONSTANT or OP_TYPE_LLVM_MLIR_CONSTANT
static inline MLIR_ValueHandle MLIR_BuildConstantFloat(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, double v);    // splat if rt is a vector type
static inline void MLIR_BuildBr(MLIR_Builder *b, MLIR_OpType t, MLIR_BlockHandle dest,
        const MLIR_ValueHandle *args, uint32_t n);
static inline void MLIR_BuildCondBr(MLIR_Builder *b, MLIR_OpType t, MLIR_ValueHandle cond,
        MLIR_BlockHandle t_dest, const MLIR_ValueHandle *t_args, uint32_t n_t,
        MLIR_BlockHandle f_dest, const MLIR_ValueHandle *f_args, uint32_t n_f);
```

Why this adds no overhead (and removes some):

- **Calls.** Today: `CreateValueOpResult` + `CreateOp` + `AppendBlockOp` (3
  out-of-line calls, 20+ argument words). Proposed: 1 call with 2 pointer
  arguments; the inline helper builds the spec in the caller's frame (the
  same stores the caller makes today to fill its scratch arrays,
  `emit.c:522-524`).
- **Allocations (native).** Today: about 14 `arena_alloc` calls for a binary
  op with tracking on (4 in `emit.c` including the formatted name; value and
  value aux; then op, aux, 3 array copies, `operand_uses` and 2 `UseNode`s in
  `CreateOp`), each an out-of-line call (`corec/base/arena.c:74`). Proposed: one allocation holding the op header,
  its operands (with inline use links, N6) and results (§6 N5).
- **Upstream.** `OperationState(loc, opName[type])`, `addOperands` from
  handles that *are* `Value` opaque pointers, `addTypes`, `addAttribute` with
  the cached `StringAttr` key, `Operation::create`, `Block::getOperations()
  .insert(before)`: the same calls `OpBuilder::create` makes in C++, minus
  the typed builder. No `std::string`, no map insertion, no snapshot.
- **What goes away:** `CreateOpWithSuccessors`' `ValueHandle **` and
  `size_t *` per-successor arrays (`h:562-577`); the printer-only parameters
  `unnumbered_loc_def`, `trailing_comment`, `source_line_start` (move to a
  parser side table); the separate result-type and result-value arrays
  (`h:550-551`; a result's type is its value's type).

`MLIR_CreateOp`/`CreateOpWithSuccessors` remain as deprecated wrappers until
every caller (109 sites) is migrated, then are removed.

## 3. Passes and rewriting (A7, A8)

### 3.1 Traits and interfaces

Traits are bits in the `mlir_ops.def` row of each op (§1.2), looked up by
array index:

```c
typedef uint64_t MLIR_OpTraits;
enum {
    MLIR_T_TERMINATOR   = 1u << 0,  MLIR_T_BRANCH      = 1u << 1,   // has successors
    MLIR_T_RETURN_LIKE  = 1u << 2,  MLIR_T_PURE        = 1u << 3,   // no memory effects, speculatable
    MLIR_T_MEM_READ     = 1u << 4,  MLIR_T_MEM_WRITE   = 1u << 5,
    MLIR_T_COMMUTATIVE  = 1u << 6,  MLIR_T_CONSTANT    = 1u << 7,   // constant-like, value in MLIR_KEY_VALUE
    MLIR_T_ELEMENTWISE  = 1u << 8,  MLIR_T_FASTMATH    = 1u << 9,   // carries MLIR_KEY_FASTMATH_FLAGS
    MLIR_T_CALL         = 1u << 10, MLIR_T_FUNCTION    = 1u << 11,  // func.func, llvm.func, wasmssa.func, aarch64.func
    MLIR_T_SYMBOL       = 1u << 12, MLIR_T_ISOLATED    = 1u << 13,
    MLIR_T_NO_TERMINATOR= 1u << 14, MLIR_T_SINGLE_BLOCK= 1u << 15,
    MLIR_T_ZERO_RESULT  = 1u << 16, MLIR_T_ONE_RESULT  = 1u << 17,
};
MLIR_OpTraits MLIR_GetOpTypeTraits(MLIR_OpType t);   // 0 for dynamic types
static inline bool MLIR_OpHasTrait(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_OpTraits t) {
    return (MLIR_GetOpTypeTraits(MLIR_GetOpType(ctx, op)) & t) == t;
}
```

This is #169's design (static table, O(1)), with the table generated instead
of hand-maintained. It replaces `mlir_llvm_op_traits.h:16`,
`lc_is_pure` (`mlir_llvm_load_cse.c:143`), `ag_is_commutative`
(`mlir_llvm_arith_gvn.c:145`) and `ssa_op_has_result`
(`mlir_wasmssa_to_wasmstack.c:153`).

**Interfaces are plain functions**, not vtables: generic accessors with a
`switch` on op type inside the API, for the few questions whose answer is not
a bit — `MLIR_GetCallee(op)` (symbol of `llvm.call`/`func.call`/…),
`MLIR_GetFunctionType(op)`, `MLIR_GetOpFastMath(op)`,
`MLIR_GetOpSuccessorOperandRange` (§2.7). Upstream they map to
`CallOpInterface`, `FunctionOpInterface`, `FastmathFlagsInterface`,
`BranchOpInterface` where the op is registered, or to the attribute key
otherwise. The trait table is ours on *both* backends (op identity is our
enum); a test cross-checks `MLIR_T_TERMINATOR`/`MLIR_T_PURE` against
upstream's `hasTrait<IsTerminator>()`/`isPure()` for registered ops.

### 3.2 Traversal and walk orders

| Need | API | Native | Upstream |
|---|---|---|---|
| ops of a block, in order (both directions) | `GetBlockFirstOp`, `GetBlockLastOp`, `GetOpNext`, `GetOpPrev` | array index stored in op, or list links (D4) | `Block::front/back`, `Operation::getNextNode/getPrevNode` |
| blocks of a region | `GetRegionFirstBlock`, `GetBlockNext`, `GetBlockPrev` | same | `Region::front`, `Block::getNextNode` |
| uses of a value | `GetValueFirstUse`, `GetUseNext`, `GetUseOwner`, `GetUseOperandIndex`, `ValueHasOneUse`, `IsValueUnused` | intrusive list (N6) | `Value::use_begin`, `OpOperand::getNextOperandUsingThisValue` |
| predecessors | `GetBlockFirstPred`, `GetPredNext`, `GetPredBlock`, `GetPredSuccessorIndex` | per-block edge list maintained when successors are set (N8) | `BlockOperand` list (`Block::pred_begin`) |
| nested ops, pre- or post-order | `MLIR_Walk(op, MLIR_WALK_PRE/POST, fn, user)`; `fn` returns `ADVANCE`/`SKIP`/`INTERRUPT` | recursive loop | `Operation::walk<WalkOrder>` |
| block orders | `size_t MLIR_GetBlockOrder(ctx, region, MLIR_ORDER_RPO / PO, Arena *out, MLIR_BlockHandle **blocks)` | iterative DFS | `llvm::ReversePostOrderTraversal<mlir::Block *>` |
| dominators, loops | `MLIR_CFGInfo *MLIR_ComputeCFGInfo(ctx, region, Arena *out)` with preds, RPO index, idom, loop depth (issue 10, §7) | Cooper–Harvey–Kennedy on RPO | `DominanceInfo`, `CFGLoopInfo` |

All are O(1) per step on both backends. The index functions
(`GetRegionBlock`, `GetBlockOp`, `GetBlockPredecessor`, `GetValueUseOwner`)
stay as O(index) conveniences outside loops; upstream drops its 16-slot cache
(`u:216-290`). Mutation rules, as upstream: in a post-order walk the callback
may erase the visited op; in a pre-order walk it may return `SKIP` after
replacing it; iterators over uses must fetch `GetUseNext` before changing the
current use.

### 3.3 Pass and pipeline abstraction

Today passes are C functions called from the driver in a hand-written order
(`driver.c:169`, `:441`, `:535-545`, `:1105-1139`), with ad-hoc
environment switches (`TINYC_NO_DCE`, `mlir_llvm_dce.h:22`) and no timing.

```c
typedef struct MLIR_PassContext {
    MLIR_Context          *ctx;
    const MLIR_TargetInfo *target;   // §4
    Arena                 *scratch;  // reset after each pass invocation
} MLIR_PassContext;

typedef enum { MLIR_ANCHOR_MODULE, MLIR_ANCHOR_FUNCTION } MLIR_PassAnchor;
typedef bool (*MLIR_PassFn)(MLIR_PassContext *pc, MLIR_OpHandle anchor, void *state);

typedef struct MLIR_Pass {
    string          name;            // for timing, --disable=<name>, diagnostics
    MLIR_PassAnchor anchor;          // FUNCTION = every op with MLIR_T_FUNCTION and a body
    MLIR_PassFn     run;
    void           *state;
} MLIR_Pass;

typedef struct MLIR_PipelineOptions {
    bool verify_each;                // MLIR_VerifyOp after every pass
    bool timing;                     // per-pass wall time, reported via diagnostics (feeds C1 --time-stages)
    string print_after;              // pass name or "*"; empty = none
} MLIR_PipelineOptions;

bool MLIR_RunPasses(MLIR_Context *ctx, MLIR_OpHandle module,
                    const MLIR_Pass *passes, size_t n, const MLIR_PipelineOptions *opts);
```

Native: a loop over passes and anchors. Upstream: each `MLIR_Pass` becomes a
`CPassAdapter final : mlir::OperationPass<>` whose `runOnOperation` calls
`run` and calls `signalPassFailure()` on `false`; function-anchored passes
are added under `pm.nest<mlir::LLVM::LLVMFuncOp>()` /
`pm.nest<mlir::func::FuncOp>()`; `timing` maps to `pm.enableTiming()`,
`verify_each` to `pm.enableVerifier()`, `print_after` to
`enableIRPrinting`. `mlir_api_upstream.h` can additionally accept an upstream
pipeline string (`"convert-scf-to-cf"`) as a pass, which is how
`LowerToLLVMDialectUpstream` (`u:2091`) would be expressed. Existing passes
(`mem2reg`, `load_cse`, `arith_gvn`, `dce`, `LowerToLLVMDialect`,
`LiftCfToScf`) become `MLIR_Pass` values in `mlir_passes.h`.

### 3.4 Is a pattern-rewrite driver needed?

Not now (D7). None of the in-tree passes is pattern-shaped: they are
analyses followed by targeted rewrites (mem2reg: Braun et al. with a
remap table, `mlir_llvm_mem2reg.c:1-12`; DCE: worklist, `mlir_llvm_dce.c:61`;
GVN/CSE: hashing). Neither are the LFortran passes: both collect roots first
and then rewrite (`rsqrt:272-313`, `red:610-619`). A greedy driver revisits ops until
nothing changes, which costs a worklist and repeated visits on every run; the
native compile-speed budget does not want that by default. What the passes do need, and what the API should provide instead:
the builder (§2.7), RAUW/erase with defined semantics (§2.3), use and
predecessor iterators (§3.2), a handle→handle hash map utility (today
reimplemented in `mlir_llvm_dce.c:16-56` and `mlir_llvm_mem2reg.c:20-60`), and
a small op worklist with a "queued" bit. If dialect conversion later needs
patterns, the shape is `{MLIR_OpType root; int benefit; bool (*rewrite)(MLIR_Builder *, MLIR_OpHandle)}`,
mapped upstream to an `mlir::RewritePattern` rooted at `opName[root]` and
driven by `applyPatternsGreedily`, natively by one worklist pass; revisit
after issue 15 (§7).

### 3.5 Mapping onto upstream

| API | Upstream |
|---|---|
| `MLIR_Context` | owns `mlir::MLIRContext`, `OperationName opName[]`, `StringAttr keyAttr[]`, the type→enum and name→enum maps |
| op/value/block/region/type/attr/location handles | `Operation *`, `Value::getAsOpaquePointer()`, `Block *`, `Region *`, `Type::getAsOpaquePointer()`, `Attribute::getAsOpaquePointer()`, `Location::getAsOpaquePointer()` |
| use / predecessor handles | `OpOperand *` / `BlockOperand *` |
| `MLIR_CreateOpFromSpec`, `MLIR_BuildOp` | `OperationState` + `Operation::create` + `Block::getOperations().insert` |
| traits | our table; cross-checked against upstream traits in a test |
| `MLIR_Walk` | `Operation::walk<WalkOrder::PreOrder/PostOrder>` |
| `MLIR_RunPasses` | `mlir::PassManager` with `CPassAdapter` passes |
| diagnostics | `ScopedDiagnosticHandler` |
| patterns (later) | `RewritePattern` + `applyPatternsGreedily` |

### 3.6 Open question 6: where LLVM-level passes run on upstream

LFortran runs its passes on optimized `llvm::Module` IR after LLVM's
vectorizer (`rsqrt:46`, `red:55`), using `TargetMachine` subtarget queries
(`red:571`). Options:

| Option | What it means | Pros | Cons |
|---|---|---|---|
| (a) LLVM dialect only | API passes see MLIR's LLVM dialect on both backends | One implementation of every API function per backend; already the model for tinyc | On upstream the pass runs before translation, so before LLVM's optimizer and vectorizer: wrong placement for LFortran's passes |
| (b) dual view inside the upstream backend | the same handles may denote `mlir::Operation *` or `llvm::Instruction *` | runs inside LLVM's pipeline, no translation | every accessor must branch on handle kind (most accessors take no context, so the tag must live in the pointer); ~150 functions implemented twice in one file; constants are not instructions in LLVM IR, phis are not block arguments, intrinsics are calls — each needs an emulation |
| (c) a third implementation `mlir_api_impl_llvmir.cpp` | the LLVM-level subset of the header implemented over `llvm::Function`, selected at link time like native/upstream | no run-time dispatch; LFortran (which uses LLVM, not MLIR) links only this one; tinyc never needs it | a third backend to keep in sync, limited to the LLVM-level subset |
| (d) (a) now, (c) when needed | — | (a) is needed anyway for tinyc and for testing | — |

**Recommendation: (d).** The acceptance tests (A12) run on the LLVM dialect:
natively in tinyc's pipeline, and upstream by importing the optimized `.ll`
that LFortran's tests produce with `mlir::translateLLVMIRToModule`, running
the pass, and translating back — identical input on both backends, at the
cost of a round trip on upstream that only the test pays. The API subset
needed by LLVM-level passes is designed so that (c) is possible without
changing it: block arguments map to `PHINode`s and successor operands to phi
incoming values; constants are reached through value-level matchers
(`MLIR_MatchConstantFloat(value, lane, &d)`, §4) rather than "the defining
op of a constant", so a constant that is not an instruction needs no fake op;
intrinsic identity is an enum or an interned name on both. Build (c) only
when LFortran decides to adopt the API for production passes (owner decision
D6).

## 4. Gap analysis against the LFortran passes (A12)

| # | Capability | Used at | Today | Minimal addition |
|---|---|---|---|---|
| 1 | Iterate blocks and instructions of a function | `rsqrt:274-275`, `red:589-590`, `red:613-614` | Yes, by index; O(i) per step upstream (`u:764-769`, `u:782-840`) | Iterators (§3.2) |
| 2 | Reverse post-order of blocks | `rsqrt:322` | No | `MLIR_GetBlockOrder(..., MLIR_ORDER_RPO, ...)` |
| 3 | Snapshot a block's ops before rewriting it | `rsqrt:324-327` | Yes (user code) | — |
| 4 | Opcode tests: fadd, fsub, fmul, fdiv, fneg, select, bitcast, call | `rsqrt:222`, `red:132-134`, `red:432-516` | Yes (`OP_TYPE_LLVM_F*`, `SELECT`, `BITCAST`, `CALL`) | — |
| 5 | Vector element ops: shufflevector, insertelement, extractelement | `red:104-130`, `red:468-503` | **No ops** | `OP_TYPE_LLVM_SHUFFLEVECTOR`, `INSERTELEMENT`, `EXTRACTELEMENT` (and `EXTRACTVALUE`, also missing) with `MLIR_KEY_MASK` as a dense i32 array |
| 6 | Generic intrinsics: `sqrt`, `pow` | `rsqrt:284-292` | `sqrt` only (`OP_TYPE_LLVM_INTR_SQRT`) | enum entries for the common ones (`pow`, `fma`, `fabs`, …) |
| 7 | Target intrinsics: `x86.avx512.rsqrt14.pd.*`, `x86.avx.hadd.pd.256` | `rsqrt:186-197`, `red:548-560` | **No** | `OP_TYPE_LLVM_CALL_INTRINSIC` (upstream `llvm.call_intrinsic`) with an interned intrinsic-name key; `MLIR_BuildCallIntrinsic(b, name, rt, args, n, fmf)` |
| 8 | Callee of a call (`sqrtf`, `powf`, …) | `rsqrt:285-289` | Through `GetOpAttributeByName(op, "callee")` and string compare | `MLIR_GetCallee(op)` → interned symbol (§3.1) |
| 9 | Scalar float/double and fixed vector types (element type, length) | `rsqrt:112-140`, `red:141-143`, `red:302-305` | **No vector type**; no width getter — backends parse type strings (§6 N12) | `MLIR_TypeKind MLIR_GetTypeKind(t)` (one enum incl. LLVM kinds, answering #230), `MLIR_GetTypeWidth`, `MLIR_CreateTypeVector(ctx, n, elem)`, `MLIR_GetTypeElement`, `MLIR_GetTypeNumElements` |
| 10 | Fast-math flags: test and set | `rsqrt:206-209`, `rsqrt:318`, `red:67-70`, `red:650` | **No** | `MLIR_FastMath` bitmask (`MLIR_FMF_FAST` = all), `MLIR_GetOpFastMath(op)`, a `fmf` argument on `Build*` helpers; stored under `MLIR_KEY_FASTMATH_FLAGS` (upstream `#llvm.fastmath<...>`) |
| 11 | Constants incl. splats; zero; per-lane element; finiteness | `rsqrt:228-260`, `red:386-421` | Integer/float scalar constants via the `value` attribute; **no splat/dense** | `MLIR_MatchConstantFloat(v, lane, &d)`, `MLIR_MatchConstantInt(v, lane, &i)`, `MLIR_IsConstantZero(v, lane)` working on values (constant op natively, `ConstantFP`/`getSplatValue` in option (c)); `MLIR_CreateAttributeSplat(ctx, vec_type, scalar)` |
| 12 | Undef/poison/null of a type | `rsqrt:197`, `rsqrt:363`, `red:215`, `red:662` | Ops exist (`LLVM_MLIR_UNDEF/POISON/ZERO`) | `MLIR_BuildUndef/Poison/Zero(b, type)` helpers |
| 13 | Uses: iterate users, `hasOneUse`, `use_empty` | `rsqrt:304`, `rsqrt:371`, `rsqrt:520` | `GetValueNumUses`/`GetValueUseOwner`: O(i) per use, 0 with tracking off | use iterator + `ValueHasOneUse`/`IsValueUnused` (§3.2); def-use always on (D3) |
| 14 | Phi: incoming values and blocks, set incoming, create phi before a phi, add incoming | `rsqrt:346-365`, `rsqrt:495-559` | Block args + predecessors (O(blocks) per query natively, `c:2475-2528`) + successor operands; append only | pred iterator; `MLIR_GetIncomingValue(pred, arg)`/`MLIR_SetIncomingValue(ctx, pred, arg, v)` over the predecessor's terminator; `MLIR_InsertBlockArgument(ctx, block, idx, type, loc)` that also inserts a slot in every predecessor's successor operands |
| 15 | Insertion before/after an instruction and before a terminator | `rsqrt:351`, `rsqrt:411`, `rsqrt:454`, `rsqrt:545`, `red:649` | `InsertBlockOpAtIndex` (O(n)); `InsertBlockOpBeforeTerminator` is a plain append natively (`c:753-756`) | `MLIR_Builder` (§2.7) |
| 16 | setOperand, RAUW, erase | `rsqrt:485`, `rsqrt:372`, `red:692` | Yes; RAUW needs tracking natively; erase semantics differ (§2.3) | rule 3 of §2.3 |
| 17 | Function is a declaration | `rsqrt:572`, `red:581` | Region has no blocks (`GetRegionNumBlocks`, O(n) upstream) | `MLIR_IsFunctionDeclaration(fn)` |
| 18 | Function attributes: has/add string attribute (`reciprocal-estimates`), read `target-features` | `rsqrt:153`, `rsqrt:573`, `rsqrt:589` | Only generic string lookup; nothing maps to LLVM's `passthrough` natively | `MLIR_GetFunctionPassthrough(fn, key)`, `MLIR_SetFunctionPassthrough(ctx, fn, key, value)` (upstream `LLVMFuncOp::passthrough`, already used at `u:2061-2088`; `mlir_translate_to_llvm_ir.c` must learn to emit it) |
| 19 | Target triple and features (incl. CPU-implied features) | `rsqrt:107-109`, `rsqrt:152-181`, `red:563-572` | **No** | `MLIR_TargetInfo { MLIR_Arch arch; uint64_t features; string triple; }` in the pass context, filled by the host (tinyc's driver; upstream/LLVM-IR from the `TargetMachine` subtarget for the function); `MLIR_TargetIsX86(t)`; `MLIR_TargetHasFeature(t, fn, MLIR_FEATURE_AVX)` also reads the function's `target_features` override |
| 20 | Module of a function (to declare intrinsics) | `rsqrt:191`, `red:645` | `GetBlockParentOp` is O(#ops) natively (`c:2448-2462`) | `MLIR_GetParentOp(op)` O(1) (region stores its op) |

Sketch of `llvm_rsqrt` against the proposed API (float path, x86 check,
phi fix-up; `K_RECIP_EST` is a key interned once):

```c
static bool is_fast(MLIR_OpHandle op) { return MLIR_GetOpFastMath(op) == MLIR_FMF_FAST; }

// Every use of the square root s is a fast `a / s` with a != s.
static bool only_divisor(MLIR_ValueHandle s) {
    for (MLIR_UseHandle u = MLIR_GetValueFirstUse(s); u; u = MLIR_GetUseNext(u)) {
        MLIR_OpHandle d = MLIR_GetUseOwner(u);
        if (MLIR_GetOpType(d) != OP_TYPE_LLVM_FDIV || MLIR_GetUseOperandIndex(u) != 1
                || MLIR_GetOpOperand(d, 0) == s || !is_fast(d))
            return false;
    }
    return true;
}

static void expand_float(Expansion *E, MLIR_OpHandle sqrt) {   // rsqrt:408-419
    MLIR_ValueHandle s = MLIR_GetOpResult(sqrt, 0), x = MLIR_GetOpOperand(sqrt, 0);
    MLIR_TypeHandle  t = MLIR_GetValueType(s);
    MLIR_Builder     b = MLIR_BuilderAfter(E->ctx, sqrt);
    MLIR_ValueHandle e = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FDIV, t,
                             MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, t, 1.0), s, MLIR_FMF_FAST);
    MLIR_ValueHandle xee = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, t, x,
                             MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, t, e, e, MLIR_FMF_FAST), MLIR_FMF_FAST);
    MLIR_ValueHandle z = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, t, e,
                             MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD, t, xee,
                                 MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, t, -3.0),
                                 MLIR_FMF_FAST), MLIR_FMF_FAST);
    scaled_put(&E->newton, s, z, -0.5);
}

// A phi (block argument) takes its incoming values at the end of the
// predecessors: apply the factor there (rsqrt:340-356).
static void fix_incoming(Expansion *E, MLIR_BlockHandle blk) {
    for (size_t a = 0; a < MLIR_GetBlockNumArgs(blk); a++)
        for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(blk); p; p = MLIR_GetPredNext(p)) {
            Scaled sc;
            if (!scaled_get(&E->scaled, MLIR_GetIncomingValue(p, a), &sc)) continue;
            MLIR_Builder b = MLIR_BuilderBefore(E->ctx, MLIR_GetBlockLastOp(MLIR_GetPredBlock(p)));
            MLIR_SetIncomingValue(E->ctx, p, a, materialize(&b, sc));
        }
}

bool expand_reciprocal_sqrt(MLIR_PassContext *pc, MLIR_OpHandle fn, void *state) {
    if (!MLIR_TargetIsX86(pc->target) || MLIR_IsFunctionDeclaration(fn)
            || MLIR_GetFunctionPassthrough(fn, K_RECIP_EST).size) return false;
    ...                                         // square_roots(), then over the RPO:
    size_t n; MLIR_BlockHandle *rpo;
    n = MLIR_GetBlockOrder(pc->ctx, MLIR_GetOpRegion(pc->ctx, fn, 0), MLIR_ORDER_RPO, pc->scratch, &rpo);
    ...
    MLIR_SetFunctionPassthrough(pc->ctx, fn, K_RECIP_EST, str_lit("sqrtf:0,vec-sqrtf:0"));
    return true;
}
```

The port is line-for-line with the C++ (the maps become handle-keyed hash
maps); the only strings left are data that is text in LLVM as well: callee
names (`sqrtf`, `powf`) and the passthrough attribute.

## 5. Upstream overhead

Each point lists where the cost is, what it costs per call, and how to remove
it. Ordered by expected impact (call-site counts from §1.1).

| # | Overhead | Where | Removal plan |
|---|---|---|---|
| U1 | **Value handles through a global map.** Every `GetOpOperand`/`GetOpResult`/`GetBlockArg`/`GetOpSuccessorOperand` does an `unordered_map` lookup and possibly a `new ValueBox` (696 call sites). Boxes are never freed. Block args created before their block go through a hidden holder block and a RAUW. | `u:120-134` (ValueBox, map), `u:167-175` (`boxFor`), `u:606-614`, `u:747-754`, `u:844-846`, `u:864-897`, `u:360-378`, `u:509-514`, `u:1892` | Handle = `Value::getAsOpaquePointer()`; `Value::getFromOpaquePointer` to use it. Requires three API changes: results are created by the op (§2.7, removes `CreateValueOpResult`); block args are created on an existing block (`AddBlockArgument`, removes `CreateValueBlockArg`/`AppendBlockArg`); register names move to a parser/printer side table (only `pp` reads them). The parser's forward references (`mlir_parser.c:931`) use `MLIR_CreatePlaceholderValue(ctx, type)` (a result of a detached unregistered op) plus RAUW, which is what upstream's own parser does. Handle identity is preserved: one `Value` has one opaque pointer. |
| U2 | **Attribute snapshots.** `CreateOp` copies the user attributes into a per-op `std::vector` in a global map; first access to any other op's attributes copies `getAttrs()`; erase must erase map entries. | `u:136-148`, `u:459-463`, `u:507`, `u:537-554`, `u:636-653`, `u:1659-1660` | Attributes are read from the op (`getAttrs()`, inherent properties via `getInherentAttr`); handles are `Attribute` pointers and names keys (§2.6). The snapshot existed to hide default-valued inherent properties (`arith.addi`'s `overflowFlags`) from the printers so native and upstream text compare equal; move that filter into the upstream printer path, the only consumer that cares. |
| U3 | **`GetOpAttributeByName` allocates.** Per call: `strlen`, `StringAttr::get` (uniquer hash under the context lock), and a heap `unique_ptr<NamedAttribute>` pushed onto a per-op vector that grows on every query; plus `dyn_cast<LLVM::CallOp>` and string compares for `callee`, `var_callee_type`, `case_values`. | `u:154-165`, `u:654-726` | `GetOpAttr(op, key)` = `op->getAttr(keyAttr[key])`, returning the pointer: no allocation. `case_values` stays a `DenseIntElementsAttr` and is read through the dense getter (§2.6) instead of being rebuilt as `DenseI32ArrayAttr` per call. |
| U4 | **Attribute creation allocates.** `new mlir::NamedAttribute` per created attribute, never freed; `GetAttributeArrayElement` allocates one per call. | `u:1196-1200`, `u:1539-1544` | Value handles (§2.6). |
| U5 | **Printing to answer queries.** `GetAttributeString` prints non-string attributes and interns the text in `thread_local` maps; symbol refs build a `std::string` key per call; `GetAttributeAsString` prints on every call (and `arith_gvn`/`load_cse` call it for every attribute of every candidate op, `mlir_llvm_arith_gvn.c:201`). | `u:1493-1535` | Typed getters (`GetAttributeSymbol`, linkage enum, dense arrays); `GetAttributeAsString` moves to the printer API; GVN/CSE keys hash `(key, Attribute pointer)` — upstream attributes are uniqued, so pointer equality is value equality. |
| U6 | **`CreateOp` string work.** A `std::string` for the name (heap for names over 15 chars, e.g. `llvm.mlir.constant`); `OperationState(loc, StringRef)` resolves `OperationName` by string; per-attribute `nm == "cf.switch"`/`"llvm.switch"` compares; segment-size synthesis keyed on name compares. | `u:397-412`, `u:439-440`, `u:465-504` | `OperationState(loc, opName[type])` (§2.5); the segment logic switches on `type`; with flat operands and explicit successor counts (§2.7) the `operandSegmentSizes` construction stays but is enum-driven. |
| U7 | **Linear block/op indexing.** `GetRegionNumBlocks` is `std::distance`, `GetRegionBlock(i)` is `std::advance` (so an index loop over blocks is O(n²)); `GetBlockNumOps` is O(n) unless cached; `GetBlockOp` relies on a 16-slot LRU cache that every mutation must invalidate; `GetBlockIndex`, `GetBlockOpIndex`, `InsertBlockOpAtIndex` are O(n). | `u:760-769`, `u:771-840`, `u:216-290`, `u:321-359`, `u:848-858`, `u:1702-1714` | Iterators (§3.2), each one pointer chase; index functions remain but are not used in loops; the cache and all its invalidation calls are deleted. |
| U8 | **Interface lookups on operand counts.** `GetOpNumOperands` does `dyn_cast<BranchOpInterface>` and subtracts every successor's operand count, per call; `GetOpNumSuccessorOperands`/`GetOpSuccessorOperand` rebuild `SuccessorOperands` per call. | `u:593-605`, `u:740-754` | Flat operand model (D5): `GetOpNumOperands` = `getNumOperands()`; `GetOpSuccessorOperandRange` computes (first, count) once per call from `getSuccessorOperands(s)`, only when a pass asks for it. |
| U9 | **Uses and predecessors by index.** `GetValueNumUses` counts the list; `GetValueUseOwner(i)` and `GetBlockPredecessor(i)` walk from the head, so enumerating is quadratic. | `u:1857-1879`, `u:1920-1946` | Use handle = `OpOperand *`, pred handle = `BlockOperand *` (§3.2). |
| U10 | **Type strings for type tests.** `GetTypeString` hashes into a static `DenseMap` and copies into the arena per call (104 call sites; backends use it to classify scalars). | `u:1133-1166` | `GetTypeKind`/`GetTypeWidth` (§4 row 9); `GetTypeString` remains for printers only. |
| U11 | **Global context and static caches.** One static `MLIRContext` regardless of `MLIR_Context`; static `nameCache`, `typeStringCache` and two `thread_local` caches. | `u:96-118`, `u:568-569`, `u:1152-1154`, `u:1499`, `u:1515` | Per-context state (§2.3 rule 1). |
| U12 | **`GetOpType` hash.** A `DenseMap` lookup keyed by the `OperationName` impl on the most called accessor after `GetOpOperand` (194 sites). | `u:560-575` | Keep (one pointer hash); make it per context. If profiling shows it, a 64-entry direct-mapped cache in front. |
| U13 | **Correctness gaps found while auditing** (not overhead): `GetOpLocation` always returns INVALID, so locations are lost in the conversions that copy them (23 uses in `conv`); `IsTypeFunction` ignores `LLVMFunctionType` unlike native; `CreateTypeFloat` maps unknown widths to f32. | `u:584`, `u:1041`, `u:936` | Fix with the corresponding sub-issues. |

After U1–U11 the upstream layer is a cast per handle plus the work the C++ API
itself does; A10c measures it against the same work written with the C++ API.

## 6. Native performance notes

What in the native data structures (`c:19-251`) could block the
compile-speed goal (tcc-class, ≥ 10× faster than `clang -O0`), and what to do.
Sizes are for LP64.

| # | Issue | Where | Proposal |
|---|---|---|---|
| N1 | **String attribute scans.** `GetOpAttributeByName` = `strlen` + `memcmp` over the op's attributes; reached from ~440 call sites, e.g. every encoded AArch64 instruction reads `rd`/`rn`/`rm`/… by string (`attr_i` at `mlir_aarch64_asm.c:471`, 195 call sites). | `c:875-885` | `MLIR_AttrKey` (§2.6): a `uint32_t` compare per attribute. |
| N2 | **Attribute name copies.** Every attribute constructor copies its name into the arena (`arena_dup_str`), 16 B + the bytes per attribute. | `c:552-557`, `c:1594-1666` | Keys: 4 B, no copy. |
| N3 | **Linear type interning.** `intern_type` compares against *every* interned type (`type_eq`) on each `CreateType*` call; the number of types grows with distinct array sizes and function signatures. `CreateTypeFunction`/`CreateTypeLLVMFunction` allocate their arrays before the lookup, so a hit leaks them. | `c:513-537`, `c:1320-1336`, `c:1445-1463` | Hash table keyed by the type's contents in the context; cached handles for i1/i8/i16/i32/i64/f32/f64/ptr in the context; allocate only on a miss. |
| N4 | **Linear identified-struct interning** by name `memcmp`. | `c:1249-1255` | Same hash table, keyed by name. |
| N5 | **Allocations per op.** `IR_Op` is 120 B; creation makes up to 6 separate array copies, an `IR_OpAux` (64 B) whenever tracking is on, an `operand_uses` array, one 32-B `UseNode` per operand; each result is a separate `IR_Value` (40 B) plus `IR_ValueAux` (32 B) created by the caller beforehand. `result_types` duplicates the results' types. | `c:213-235`, `c:582-710`, `c:1745-1754`, `c:425-433` | One allocation per op: header + operands (with inline use links) + results + attributes as trailing storage; results created by the op; drop `result_types`; printer-only fields to a side table. |
| N6 | **Def-use tracking cost.** With tracking: 8 B slot + 8 B `operand_uses` pointer + 32 B `UseNode` = 48 B per operand, plus 64 B aux per op and 32 B aux per value; without it 8 B per operand, but RAUW/use queries silently break (§2.3), so passes reimplement use counting. | `c:141-174`, `c:290-360`, `c:631-696` | Intrusive operand: `{value, next_use, prev_use_slot, owner}` = 32 B inline in the op (as upstream's `OpOperand`); no separate allocation, no aux. Measure (A9) and decide D3. |
| N7 | **Block op arrays.** Erase compacts the array (O(n), so a DCE erasing k ops of a block is O(kn)); `MoveOpToBlockEnd`, `InsertBlockOpAtIndex`, `GetBlockOpIndex` are O(n); `InsertBlockOpBeforeTerminator` just appends (wrong). | `c:2146-2153`, `c:2196-2212`, `c:758-783`, `c:2182-2189`, `c:753-756` | Either intrusive prev/next (O(1) insert/erase, +16 B/op) or arrays with deferred compaction (erase marks, the next iteration compacts). The public API (iterators, `InsertOpBefore`) is the same for both (D4); decide on A9 numbers. |
| N8 | **Predecessors by scanning.** Each `GetBlockNumPredecessors`/`GetBlockPredecessor` call scans every block of the region; enumerating is O(R·P). | `c:2475-2528` | Per-block predecessor edge lists maintained when successors are set (upstream's `BlockOperand` model), or computed once in `MLIR_ComputeCFGInfo`. |
| N9 | **Use enumeration by index**: `GetValueUseOwner(i)` walks i nodes. | `c:2601-2633` | Use iterator. |
| N10 | **Parent op by global scan**: `GetBlockParentOp` scans all ops ever created, and fails when tracking is off. | `c:2448-2462` | Region stores its parent op (8 B per region). |
| N11 | **Global op registry**: every op is appended to `g_all_ops` (with tracking on); its only reader is N10; it is the #140 hazard. | `c:281-288`, `c:372-397` | Delete. |
| N12 | **Type classification by formatting.** Backends call `GetTypeString` and parse `"i32"`/`"f64"`/`"!llvm.ptr"` for operand types; natively each call `format()`s into the arena. | `c:1001-1016`; callers `mlir_llvm_to_x64.c:323`, `:357`; `mlir_llvm_to_aarch64.c:468`, `:594`; `mlir_llvm_to_wasmssa.c:129`, `:150`; `mlir_wasmssa_to_wasmstack.c:355`; `mlir_llvm_type_layout.h:77`, `:117` | `GetTypeKind`/`GetTypeWidth` (field reads). |
| N13 | **Name compare on every creation**: `str_eq(opname, op_type_to_string(type))`, where `op_type_to_string` is two chained switches. | `c:602-603`, `mlir_op_names.c:316-322` | Enum-only creation (§2.5); name array. |
| N14 | **Register names for every value.** `emit.c` formats `"%N"` for each result (`emit.c:347-350`), which also forces an `IR_ValueAux` even with tracking off (`c:428`); the wasm→LLVM lifter creates a fresh unknown location per op (32 sites, `CreateLocationUnknown` allocates each time, `c:1956-1961`); `mlir_wasmstack_to_wasmssa.c:552-556` copies each op's name into a buffer for diagnostics. | as listed | Printer numbers values itself; one shared unknown location per context; diagnostics resolve names lazily. |
| N15 | **`AppendOpAttribute` reallocates** the array per append (O(n²) for repeated appends). | `c:1767-1779` | Attributes in trailing storage with spare capacity, or rebuild once. |
| N16 | **Dense arrays as text.** `DenseI32Array`/`DenseI64Array` are stored formatted and re-parsed by `mlir_llvm_global_init.c:17`, `mlir_llvm_mem2reg.c:669`, `mlir_llvm_to_wasmssa.c:533`, `:1057`, `mlir_translate_to_llvm_ir.c:781`, `mlir_llvm_to_aarch64.c:1232`. | `c:1685-1719` | Store `int32_t[]`/`int64_t[]`; typed getter (§2.6). |
| N17 | **Out-of-line arena calls.** Every `arena_new` is a call to `arena_alloc` (`corec/base/arena.c:74`). | `corec/base/arena.h:124-140` | Inline the bump fast path in the header (corec change); matters less once N5 reduces allocations to one per op. |
| N18 | **Process-wide state** makes the native backend single-threaded and single-context (#140). | `c:286`, `c:509`, `c:1244` | Move into the context (§2.3). |

`mlir_llvm_to_x64.c` encodes bytes directly from the LLVM dialect and only
reads IR; for it, N1, N12 and the traversal costs are what matter. The
wasm-lift pipeline, which builds IR for whole programs, pays N5/N6 most;
`emit.c` pays N3, N5, N13, N14.

As implemented (#354): N5 and N6 are replaced by the index storage of §2.2
(an op is one 56-byte record plus its 16-byte operand records, which carry
the use links; results are consecutive values; no aux record unless the
printer needs one), and N15 by appending in place when the op's attributes
are the last ones in their vector. N7 remains: erasing an op still costs
O(block size), now a scan to the op and a shift of the tail instead of a
filter over the whole list (on the self-host lift that change alone made
mem2reg 1.9× and dce 1.3× faster). N8 and N9 are unchanged.

## 7. Ordered follow-up issues

These replace roadmap items A2–A12. "Replaces" names the roadmap item each
one refines; dependencies refer to numbers in this list. C1 (#329, benchmark
harness) should land before step 1's numbers are taken.

| # | Title | Scope | Depends on | Replaces |
|---|---|---|---|---|
| 1 | API microbenchmarks and IR memory accounting | Create/walk/RAUW/erase micro-benchmarks on both backends; bytes per op/operand/value on the wasm-lift and tinyc pipelines; the baseline for D3/D4 | C1 (#329) | A9 (first half) |
| 2 | Shrink the header: remove dead API, split headers | Remove the 23 unused functions (§1.1) and the unsound `SetType*Properties`; move parser/printer-only functions to `mlir_parser.h`/`mlir_printer.h`, `*Upstream` to `mlir_api_upstream.h` (#151), pipelines to `mlir_passes.h`; naming fixes of §2.1 (`GetOpResult_type`, `GetOpName_string`, `MLIR_MLIR_`, `BLOCK_ARG`/`OP_RESULT`) | — | A1 follow-up, #37, #43, #151 |
| 3 | Typed handles | Pointer-to-incomplete-struct handle typedefs (D1), `MLIR_INVALID_HANDLE` = `NULL`, `-Werror=incompatible-pointer-types` in CI; add `MLIR_UseHandle`, `MLIR_PredHandle` | 2 | A4 |
| 4 | Context ownership and diagnostics | Opaque `MLIR_Context` with create/destroy; move native globals and upstream static caches/`MLIRContext` into it; types/attrs in the persistent arena; remove `ResetInternRegistry`; diagnostic handler; `MLIR_API_CHECKS` asserts; unify erase semantics and add `DetachOp` | 2 | new; #140 |
| 5 | Op identity by enum only | `mlir_ops.def` X-macro (enum, name array, dialect ranges); creation without names; `RegisterOpType`/`LookupOpType`/`GetOpTypeName` with per-context dynamic ids (D2); remove the name-based dispatch listed in §2.5; upstream `OperationName` table | 4 | A2; #51, #145 |
| 6 | Traits from `mlir_ops.def` | Trait bits (§3.1) for all ops; `MLIR_VerifyOp`; replace `mlir_llvm_op_traits.h`, `lc_is_pure`, `ag_is_commutative`, `ssa_op_has_result`; upstream cross-check test | 5 | A7; #169, #277 |
| 7 | Type API: kinds, widths, vectors | `MLIR_TypeKind` covering builtin and LLVM types (answers #230 without a dialect-specific API); `GetTypeWidth`, `GetTypeElement`, `GetTypeNumElements`, vector types; replace type-string parsing in all backends (N12, U10) | 3 | A6 (types); #230 |
| 8 | Attribute keys and value handles | `mlir_attr_keys.def`, interned keys, value-only attribute handles, `GetOpAttr`/`SetOpAttr`; real dense-array storage and typed getters; migrate ~440 string lookups; remove `GetOpAttributeByName` and upstream snapshots (U2–U5) | 3, 5 | A3; part of A10 |
| 9 | Builder API | `MLIR_OpSpec`, `MLIR_Builder`, inline helpers; results created by the op; flat successor operands (D5); `InsertOpBefore`; migrate `emit.c`, lifters and passes; remove `CreateOp`, `CreateOpWithSuccessors`, `CreateValueOpResult` | 5, 8 | A5 |
| 10 | Traversal API | Iterators for ops, blocks, uses, predecessors; `MLIR_Walk`; `GetBlockOrder`; `MLIR_ComputeCFGInfo` (preds, RPO, dominators, loops); migrate index loops in backends and passes | 3, 4 | A8 (first half) |
| 11 | Upstream: zero-copy handles | Remove `ValueBox` and its map (U1: needs 9), the block-op cache (U7: needs 10), interface lookups on operand counts (U8), per-call index walks (U9); fix U13 | 8, 9, 10 | A10 |
| 12 | Native IR layout | Single-allocation ops with trailing operands/results/attributes (N5); intrusive use lists and the D3 decision (N6); block storage per D4 (N7); predecessor lists (N8); region parent pointer (N10, N11); hashed type interning (N3, N4) | 1, 9, 10 | A9 (second half) |
| 13 | Pass and pipeline API | `MLIR_Pass`, `MLIR_RunPasses` with timing/verify/print-after; upstream `CPassAdapter` on `mlir::PassManager`; port `mem2reg`, `load_cse`, `arith_gvn`, `dce`, `LowerToLLVMDialect`, `LiftCfToScf`; driver builds its pipelines from pass lists | 6, 10 | A8 (second half) |
| 14 | LLVM-level completeness | Fast-math flags; vector element ops and `extractvalue`; `call_intrinsic` with interned names and common intrinsic enums; splat/dense constants and value matchers; function passthrough attributes; `MLIR_TargetInfo` (§4) | 6, 7, 8, 9 | A6 (attributes), A12 (prerequisites) |
| 15 | Port `llvm_rsqrt` and `llvm_reductions` as acceptance tests | Both passes on the API; run natively and on upstream via `.ll` import (D6 option a); compare output with LFortran's C++ passes on its test inputs; time on native | 13, 14 | A12 |
| 16 | Upstream overhead benchmark | The step-1 benchmarks against the same work written with the MLIR C++ API; document the remaining overhead | 11 | A10 (measurement) |
| 17 | API reference | Reference generated from header comments plus examples (a pass, a lowering, a backend loop); this document's rules as the preface | 13 | A11 |
| 18 | (Optional, D6) `llvm::Module` implementation | `mlir_api_impl_llvmir.cpp` implementing the LLVM-level subset over `llvm::Function`, for running API passes inside LLVM's pipeline | 15 | open question 6 |

Steps 2–5 change many call sites mechanically and should land quickly and
in order, before the larger behavioural changes (8–13), to keep rebases cheap.
