// Static ELF64 x86-64 executable container.
//
// Consumes a target-independent ObjModule (functions + PC-rel32 fixups + a data
// blob) and emits a statically-linked, no-PIE ELF executable that the Linux
// kernel can run directly: no dynamic linker, no PLT/GOT, no libc. External
// "libc" calls are expected to have been resolved to local syscall-thunk
// functions by the code generator, so every reloc target is in-module.
//
// Native-only; not part of the tinyC self-host source set.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_obj.h"

#ifdef __cplusplus
extern "C" {
#endif

// Write `m` to a freshly malloc'd ELF executable image (`*out_data`/`*out_size`,
// caller frees). Returns false on error (diagnostics to stderr).
bool mlir_obj_to_elf(ObjModule *m, uint8_t **out_data, size_t *out_size);

// Write a single flat-buffer translation unit (`.text` + `.data` + relocs +
// defined function symbols) to a freshly malloc'd ELF64 relocatable object
// file (`ET_REL`, `.o`, caller frees).
typedef struct {
    const char *name;
    size_t      name_len;
    uint32_t    off;
    uint32_t    size;
} ObjFuncSym;

bool mlir_obj_to_elf_reloc(const uint8_t *text, size_t text_len,
                           const uint8_t *data, size_t data_len,
                           const ObjFuncSym *funcs, size_t n_funcs,
                           const ObjReloc *relocs, size_t n_relocs,
                           uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif
