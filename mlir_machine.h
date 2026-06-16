// =============================================================================
// mlir_machine.h — the format-neutral seam between an ISA backend and a
// binary-container emitter.
// =============================================================================
//
// An ISA backend (today: `mlir_llvm_to_aarch64.c` + the encoder in
// `mlir_aarch64_to_macho.c`; tomorrow: an x86_64 sibling) lowers the `llvm`
// dialect to physical-register machine code and packages the result as a
// `MachineModule`. A container emitter (today: the Mach-O writer in
// `mlir_aarch64_to_macho.c`; tomorrow: ELF and PE) consumes a `MachineModule`
// and produces the final on-disk image.
//
// The point of this header is to name and document that seam so the M ISA
// backends and N container formats compose as M + N translation units instead
// of M * N monoliths. Everything here is deliberately format-neutral:
//
//   * code is raw encoded bytes,
//   * cross-function / data references are carried as *symbolic* relocations
//     (a target name or section + an addend), never as resolved file offsets,
//   * the entry point is identified by index, not by container metadata.
//
// A container emitter is then free to satisfy each relocation in whatever way
// the format prescribes (Mach-O `__stubs`/`__got`/chained-fixups + libSystem,
// ELF `.plt`/`.got` or a static syscall image, PE IAT + kernel32, ...).
//
// IMPORTANT: this header is part of the tinyC self-host source set (it is
// included by `mlir_aarch64_to_macho.c`, which tinyC recompiles). It must stay
// within the tinyC C subset: no designated initializers, no global aggregate
// initializers. Zero a struct with `= {0}` and assign fields individually.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_codegen_buf.h"  // Buf (growable byte buffer for encoded code)

// -----------------------------------------------------------------------------
// Relocations. These are symbolic: they name *what* a site references, not the
// resolved address. The container resolves them once the final layout (section
// VM addresses, import stubs) is known.
// -----------------------------------------------------------------------------

// A PC-relative call (aarch64 `BL`, x86 `call rel32`, ...) to a named symbol.
// Defined symbols resolve to an internal offset; undefined ones become the
// container's import machinery (libSystem stub / PLT / IAT).
typedef struct {
    string   callee;        // referenced symbol name
    uint32_t fn_off;        // byte offset of the call site within the function
} MachineCallReloc;

// A reference to a data section, emitted on aarch64 as an ADRP (page) + ADD
// (low-12) pair. `kind` selects the section the addend is relative to:
// "data_priv" / "globals" / "linmem". The encoding fields (`rd`/`rn`/
// `is_add_lo`) are aarch64-specific; an x86_64 backend would carry a
// RIP-relative analogue here. The container computes the section VM address and
// patches the instruction.
typedef struct {
    string   kind;          // "data_priv" / "globals" / "linmem"
    bool     is_add_lo;     // false = ADRP (page), true = ADD imm12 (low-12)
    uint8_t  rd;
    uint8_t  rn;
    uint32_t fn_off;
    uint32_t addend;        // byte offset added to the resolved section base
} MachineDataReloc;

// Intra-function branch placeholder. EMITTER-INTERNAL: these are resolved to
// PC-relative immediates by the ISA backend's branch patcher *before* the
// container consumes the module; container emitters never read them.
enum BranchKind { BR_B, BR_B_COND, BR_CBZ, BR_CBNZ };
typedef struct {
    int              kind;            // enum BranchKind
    MLIR_BlockHandle target;
    uint32_t         fn_off;          // offset of the branch insn within fn
    uint8_t          cond_or_rt;      // cond for B_COND, rt for CBZ/CBNZ
    bool             sf;              // for CBZ/CBNZ
} MachineBranchReloc;

// Position of a block within a function's code buffer. EMITTER-INTERNAL: filled
// in as blocks are emitted, consumed only by the branch patcher.
typedef struct {
    MLIR_BlockHandle blk;
    uint32_t         fn_off;
} MachineBlockPos;

// One lowered function: a name, an exported flag, the encoded bytes, and the
// symbolic relocations against that code. `br`/`bp` are emitter scratch (see
// above) and are zero by the time a container reads the function.
typedef struct {
    string             name;
    bool               exported;
    Buf                code;
    MachineCallReloc  *relocs;
    size_t             n_relocs, c_relocs;
    MachineDataReloc  *dr;
    size_t             n_dr, c_dr;
    MachineBranchReloc *br;       // emitter-internal
    size_t             n_br, c_br;
    MachineBlockPos   *bp;        // emitter-internal
    size_t             n_bp, c_bp;
    uint32_t           text_off;  // assigned by the container during layout
} MachineFunc;

// A contribution to the program's initialized data image: `bytes` at `offset`.
typedef struct { uint32_t offset; string bytes; } MachineDataInit;

// -----------------------------------------------------------------------------
// The module handed from an ISA backend to a container emitter.
// -----------------------------------------------------------------------------
//
// Ownership: a container emitter consumes the module — it frees each function's
// heap buffers, the `funcs` array, and the `inits` array before returning. The
// `MachineModule` struct itself is caller-owned (typically stack-allocated).
typedef struct {
    MachineFunc     *funcs;
    size_t           n_funcs;
    size_t           entry_idx;     // index of the `_start` entry function

    MachineDataInit *inits;
    size_t           n_inits;

    // Wasm-derived layout hints (0 on the direct llvm path). Describe the
    // linear-memory / globals model the data image encodes.
    uint32_t         n_globals;
    uint64_t         global0_init;
    uint64_t         linmem_size;
} MachineModule;
