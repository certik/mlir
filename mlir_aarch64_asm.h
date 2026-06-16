// aarch64 instruction assembler interface.
//
// The EmittedFunc object model (per-function code bytes + relocation records)
// produced by mlir_aarch64_asm.c and consumed by the Mach-O container
// (mlir_aarch64_to_macho.c). This is the clean boundary between the
// architecture assembler and the binary-format container.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_codegen_buf.h"


// =============================================================================
// Per-function emission. A pass over an aarch64.func produces a byte
// buffer + a list of `bl` call sites that need PC-relative patching
// once all function offsets are known.
// =============================================================================
typedef struct {
    string   callee;        // referenced symbol name
    uint32_t fn_off;        // byte offset within the function's code
} BlReloc;

// ADRP / ADD imm12 relocs. `kind` is one of "data_priv" / "globals" /
// "linmem"; the patcher computes the page-relative ADRP imm21 and the
// low-12 ADD imm12 once segment VM addresses are known.
typedef struct {
    string   kind;          // "data_priv" / "globals" / "linmem"
    bool     is_add_lo;     // false = ADRP, true = ADD imm12
    uint8_t  rd;
    uint8_t  rn;
    uint32_t fn_off;
    uint32_t addend;        // byte offset added to the resolved section base
} DataReloc;

// Branch reloc. Identifies a placeholder branch instruction emitted
// for an aarch64.b / b_cond / cbz / cbnz op so we can resolve it to a
// PC-relative imm once all blocks have known function offsets.
enum BranchKind { BR_B, BR_B_COND, BR_CBZ, BR_CBNZ };
typedef struct {
    int              kind;            // enum BranchKind
    MLIR_BlockHandle target;
    uint32_t         fn_off;          // offset of the branch insn within fn
    uint8_t          cond_or_rt;      // cond for B_COND, rt for CBZ/CBNZ
    bool             sf;              // for CBZ/CBNZ
} BranchReloc;

// Position of a block within the function's code buffer. Filled in as
// blocks are emitted, consumed by the branch patcher.
typedef struct {
    MLIR_BlockHandle blk;
    uint32_t         fn_off;
} BlockPos;

typedef struct {
    string       name;
    bool         exported;
    Buf          code;
    BlReloc     *relocs;
    size_t       n_relocs, c_relocs;
    DataReloc   *dr;
    size_t       n_dr, c_dr;
    BranchReloc *br;
    size_t       n_br, c_br;
    BlockPos    *bp;
    size_t       n_bp, c_bp;
    uint32_t     text_off;   // assigned after layout
} EmittedFunc;

// Emit one aarch64.func to a byte buffer + relocation lists. Returns false on
// an unsupported op (diagnostics to stderr).
bool emit_aarch64_func(MLIR_OpHandle fn, EmittedFunc *out);

// Resolve the function-internal branch placeholders (b/b.cond/cbz/cbnz) once
// every block's offset within the function is known.
bool patch_branches(EmittedFunc *e);

// Instruction encoders the Mach-O container needs for reloc patching (bl call
// sites, ADRP/ADD data references) and libSystem stub generation.
uint32_t arm64_bl(int32_t imm26);
uint32_t arm64_adrp(uint8_t rd, int64_t rel_pages);
uint32_t arm64_add_imm(uint8_t rd, uint8_t rn, uint16_t imm12, bool sf);

// aarch64.func op-attribute readers (shared by the emitter + container).
int64_t attr_i(MLIR_OpHandle op, const char *name);
string  attr_s(MLIR_OpHandle op, const char *name);
