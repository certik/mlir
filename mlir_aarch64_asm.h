// aarch64 instruction assembler interface.
//
// The MachineFunc object model (per-function code bytes + relocation records)
// produced by mlir_aarch64_asm.c and consumed by the Mach-O container
// (mlir_aarch64_to_macho.c). This is the clean boundary between the
// architecture assembler and the binary-format container. The shared,
// format-neutral MachineFunc / MachineCallReloc / MachineDataReloc /
// MachineBranchReloc / MachineBlockPos types live in mlir_machine.h.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_codegen_buf.h"
#include "mlir_machine.h"


// Emit one aarch64.func to a byte buffer + relocation lists. Returns false on
// an unsupported op (diagnostics to stderr).
bool emit_aarch64_func(MLIR_OpHandle fn, MachineFunc *out);

// Resolve the function-internal branch placeholders (b/b.cond/cbz/cbnz) once
// every block's offset within the function is known.
bool patch_branches(MachineFunc *e);

// Instruction encoders the Mach-O container needs for reloc patching (bl call
// sites, ADRP/ADD data references) and libSystem stub generation.
uint32_t arm64_bl(int32_t imm26);
uint32_t arm64_adrp(uint8_t rd, int64_t rel_pages);
uint32_t arm64_add_imm(uint8_t rd, uint8_t rn, uint16_t imm12, bool sf);

// aarch64.func op-attribute readers (shared by the emitter + container).
int64_t attr_i(MLIR_OpHandle op, const char *name);
string  attr_s(MLIR_OpHandle op, const char *name);
