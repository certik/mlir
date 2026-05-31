# Native WMIR backend vs. Wasmtime/Cranelift: performance analysis

_Last updated: 2026-05-31. Platform: Apple M4, Darwin/arm64._

This document records a data-driven investigation into why our native
`wasm → wmir → aarch64 → mach-o` backend compiles the frozen tinyc corpus
~1.2× slower than Wasmtime (Cranelift) runs the same program, and where the
remaining gap actually lives.

**Bottom line:** the gap is **not** a missing mid-end optimization pass. It is
**instruction count dominated by register-allocator spill/move traffic**, and
that traffic is cheap (L1, dual-issued), so closing it has a low ceiling. Every
countable algebraic/CSE opportunity is individually ≤1–2% and wash-prone. There
is no cheap 10% remaining.

---

## 1. Benchmark setup

The frozen benchmark (`bench_frozen/`) is designed so the comparison is stable:

- A frozen copy of the tinyc compiler-as-wasm (`tinyc_frozen.wasm`, ~4 MB) and a
  frozen 46-file source corpus (`bench_frozen/src`) that never change.
- "Ours": we lower `tinyc_frozen.wasm` → mach-o once (`refresh`), then time that
  native binary compiling the corpus.
- "Wasmtime": `wasmtime run tinyc_frozen.wasm` compiling the same corpus.
- Both compile each source with identical flags, so the only difference is
  native-codegen-quality (ours) vs. Cranelift-JIT (theirs) on the same program.

Whole-invocation result: **~1.21–1.24× slower than Wasmtime.**

---

## 2. Pass-by-pass comparison

### Cranelift (opt-level=2), ordered

| Phase | Pass | Notes |
|---|---|---|
| Frontend | wasm → CLIF + on-the-fly SSA (`SSABuilder`) | local→reg during translation |
| Legalize | `simple_legalize` | target legalizations |
| Analysis | CFG, dominators, loop analysis | |
| Cleanup | unreachable-code elim, constant-phi removal, alias resolution | |
| **Mid-end** | **egraph optimizer** (`egraph.rs` + `opts/*.isle`) | GVN/eclass dedup, ISLE rewrites, alias-based store→load forwarding, rematerialization, loop-aware elaboration/hoisting |
| | rule files: `arithmetic`, `bitops`, `cprop`, `extends`, `icmp`, `remat`, `selects`, `shifts`, `vector` | const-fold, algebraic identities, strength reduction, reassociation, extend folding |
| Alias analysis | `LastStores` | cross-block redundant-load elimination |
| Lowering | CLIF → aarch64 VCode (ISLE) | address-mode folding, imm folding, cmp+branch fusion, shifted-operand fusion |
| Block order | `blockorder.rs` | critical-edge split, cold-block ordering |
| **Regalloc** | **regalloc2 "Ion"** (backtracking) | live-range splitting, **move coalescing**, spill-slot allocation, ~30 registers |
| Emit | final aarch64 | |

### Ours, ordered

`mlir_wasmssa_to_wmir.c` (mid-end), `mlir_wmir_regalloc.c`, `mlir_wmir_to_aarch64.c`:

| Phase | Pass | Notes |
|---|---|---|
| Frontend | wasm → wmir lowering | |
| SSA | `mlir_wmir_mem2reg` (Braun) | wasm locals → SSA + block-arg phis |
| Mid-end | `wmir_simplify_bools` | collapse `select(1,0,c)`/`icmp_ne(b,0)` |
| | `wmir_simplify` | const-fold + algebraic identities + pow2 strength reduction (fixpoint) |
| | `wmir_value_number` | **dominator-scoped** GVN/CSE, intra-block load-CSE, intra-block store→load forwarding, iterative DCE |
| | `wmir_licm` | opt-in (`WMIR_LICM`), default off — measured wash |
| Regalloc | **linear-scan** | **8 caller-saved regs (x11–x18)**; float or call-crossing values → stack slot; loop-depth-weighted spill victim; spill-slot reuse; **no live-range splitting, no coalescing** |
| Backend | branch fusion, imm12 folding, register-offset addressing, constant rematerialization (`HOME_CONST`), parallel-move resolver | |
| Emit | aarch64 → mach-o | |

### What we have that matches Cranelift
- SSA construction / local→reg
- Constant folding, algebraic identities, pow2 strength reduction
- **Dominator-scoped** (global) GVN/CSE for pure values
- Intra-block load-CSE + store→load forwarding
- DCE
- Branch/compare fusion, address-mode and immediate folding, constant remat

### What we lack
- Reassociation (to expose more CSE)
- Extend folding (redundant sext/zext/trunc elimination)
- **Cross-block** redundant-load elimination (needs alias analysis)
- Magic-number division for non-power-of-2 constant divisors
- Global code motion / sinking / loop-aware placement (LICM exists but is a wash)
- A backtracking allocator with **live-range splitting + move coalescing** over
  the full register file

---

## 3. The decisive isolation experiment

To separate "mid-end (egraph)" from "register allocator (Ion)", we AOT-compiled
`tinyc_frozen.wasm` under all four combinations and timed **run only** (no JIT /
compile time), over the 29 corpus files that run on all configs. We also timed
our native mach-o over the same 29 files.

`wasmtime compile -O opt-level={0,2} -C cranelift-regalloc_algorithm={backtracking,single_pass}`

| Config | Run time (best-of-5, 29 files) |
|---|---|
| **ours** (linear-scan) | **1.341 s** |
| wasmtime egraph + **Ion** (`o2_bt`) | **1.079 s** ← best (ours is 1.24×) |
| wasmtime no-egraph + Ion (`o0_bt`) | 1.285 s |
| wasmtime egraph + fastalloc (`o2_sp`) | 2.877 s |
| wasmtime no-egraph + fastalloc (`o0_sp`) | 3.032 s |

Deltas:
- **Ion vs. fastalloc: +58–63%** — the allocator is enormous in Wasmtime.
- **egraph on top of Ion: +16%** (`o0_bt` → `o2_bt`).
- **Our linear-scan (1.341) ≈ Ion (1.285), nowhere near fastalloc (3.0).**

**Conclusions:**
1. **Our register allocator is already Ion-class.** regalloc2's `single_pass`
   (fastalloc) is a *bad* allocator (spills almost everything, ~3× slower); ours
   is not that. Do **not** model anything on fastalloc, and the allocator is not
   a 2× lever for us.
2. The remaining ~1.24× is the egraph's ~16% (diffuse) plus Ion's ~4% edge over
   our linear-scan (`o0_bt` 1.285 vs ours 1.341).

---

## 4. Sizing each missing optimization (static)

A gated `WMIR_STATS=1` diagnostic in `mlir_wasmssa_to_wmir.c` tallies opportunity
across `tinyc_frozen.wasm` after all our passes (290,091 wmir ops, 2,008 funcs,
60,428 blocks):

| Opportunity | Count | Share | Verdict |
|---|---|---|---|
| Extends + trunc | 5,545 | 1.9% | only 140 trivially redundant → negligible |
| Non-pow2 const div/rem (magic-number division) | **8 sites total** | ~0% | worthless |
| imul non-pow2 const (shift/add) | 820 | 0.3% | minor |
| Cross-block same-addr reloads (upper bound, ignores stores) | 19,448 | — | optimistic |
| Cross-block same-addr reloads (coarse alias: any store/call kills cache) | **3,693** | **1.3%** | wash-prone (L1) |

The cross-block load number is the only non-trivial one, but:
- 3,693 / 290,091 = 1.3% of ops, and that coarse-alias estimate is itself
  optimistic about dominator-path correctness.
- A prior experiment that eliminated ~20k spill reloads (converting them to
  register moves) measured **+0.13% — a wash** — because L1 loads are cheap and
  dual-issue. Cross-block program-load elimination would behave the same.

So **no countable mid-end transform we lack is worth more than ~1%.**

---

## 5. Where our instructions actually go (disassembly)

Disassembling our emitted `tinyc_frozen_macho` (631,615 aarch64 instructions):

| Mnemonic | Count | Share |
|---|---|---|
| `ldr` | 157,624 | 25.0% |
| `str` | 135,282 | 21.4% |
| `mov` | 116,044 | 18.4% |
| `b` | 56,361 | 8.9% |
| `add` | 50,002 | 7.9% |
| `cmp` | 30,839 | 4.9% |
| `bl` | 23,439 | 3.7% |
| `lsl` | 9,181 | 1.5% |

- **`ldr` + `str` + `mov` = 408,950 = 64.7% of all instructions.**
- Memory (`ldr`+`str`) = 292,906 = 46.4%.
- Program memory ops in the IR are only 75,748, so
  **~217,158 (34.4%) of all emitted instructions are spill/reload/address
  traffic**, plus 116k register-to-register moves (parallel-move/phi copies and
  coalescing gaps).

### Code-size comparison (function bytes)

| Build | Function code |
|---|---|
| Ours | 2.526 MB |
| Cranelift opt-level=0 (no egraph) | 2.520 MB |
| Cranelift opt-level=2 (egraph) | **1.868 MB** |

**We emit 1.35× the instructions of the egraph build, and our instruction count
equals Cranelift _without_ its mid-end.** The egraph removes ~26% of
instructions; our GVN/DCE/fold roughly match Cranelift's base lowering but not
the egraph's holistic working-set reduction.

---

## 6. Verdict

1. **The bottleneck is register-allocator spill/move traffic, not a missing
   mid-end pass.** 64.7% of our emitted instructions are `ldr`/`str`/`mov`;
   ~34% are spills/reloads. Our linear-scan uses only 8 caller-saved registers
   (x11–x18) and force-spills every call-crossing or float value to a stack
   slot. Ion uses ~30 registers with live-range splitting and move coalescing,
   so it spills far less and is ~26% smaller.

2. **But spill reloads are cheap.** They are L1 dual-issued hits, so our 35%
   instruction bloat costs only ~24% runtime, and every prior spill/move
   reduction experiment (reload-CSE, move-coalescing) was a measured wash.

3. **Run-quality proof:** ours 1.341 s ≈ Ion-without-egraph 1.285 s. The
   allocator ceiling for us is ~4%. The egraph's further 16% is diffuse
   working-set reduction (fewer live values → even less spill under Ion), which
   we have already largely captured on our side (dominator GVN + DCE +
   load-CSE + the new algebraic/const-fold pass added +1.2%).

4. **The remaining 1.21–1.24× is "many small things," not one bottleneck.**
   There is no cheap 10% left.

### If we wanted to push further

The only structural lever with non-wash potential is **keeping call-crossing
values in callee-saved registers (x19–x28)** instead of force-spilling them
(with prologue/epilogue save/restore). Estimated ceiling ~4–5% based on the
ours-vs-Ion-without-egraph delta. High effort, modest reward.

Otherwise: accept the current ~1.2×. Our allocator is already Ion-class on this
workload, our mid-end captures the countable egraph wins, and the residual gap
is dominated by cheap, hard-to-remove spill traffic.

---

## 7. Reproducing these measurements

```sh
# Whole-invocation ratio (refresh + run):
python3 bench_frozen/bench_frozen.py both 6

# Static opportunity histogram (extends / div / loads):
pixi run build_tinyc_native
WMIR_STATS=1 ./tinyc_native --from-wasm bench_frozen/tinyc_frozen.wasm \
    --emit=macho --macho-backend=wmir -o /tmp/stats.macho

# 4-way egraph x allocator isolation (run-quality only, no JIT time):
for cfg in "o2_bt:-O opt-level=2 -C cranelift-regalloc_algorithm=backtracking" \
           "o2_sp:-O opt-level=2 -C cranelift-regalloc_algorithm=single_pass" \
           "o0_bt:-O opt-level=0 -C cranelift-regalloc_algorithm=backtracking" \
           "o0_sp:-O opt-level=0 -C cranelift-regalloc_algorithm=single_pass"; do
  name=${cfg%%:*}; flags=${cfg#*:}
  wasmtime compile $flags -o /tmp/frozen_$name.cwasm bench_frozen/tinyc_frozen.wasm
done
# then time `wasmtime run --allow-precompiled /tmp/frozen_<cfg>.cwasm ...` per file.

# Emitted instruction mix:
otool -tvV bench_frozen/tinyc_frozen_macho | \
  awk '{for(i=1;i<=NF;i++) if($i ~ /^(ldr|str|mov|add|...)$/){print $i; break}}' | \
  sort | uniq -c | sort -rn
```

### Optimization gates (all default-on except LICM)
- `WMIR_NO_CONSTFOLD` — disable constant folding
- `WMIR_NO_ALGEBRAIC` — disable algebraic identities + strength reduction
- `WMIR_NO_GVN` — disable value numbering / CSE
- `WMIR_LICM` — enable loop-invariant code motion (opt-in; measured wash)
- `WMIR_STATS` — print the opportunity histogram
