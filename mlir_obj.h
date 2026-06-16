// Target-independent object model shared by the native code-generation
// backends (x86_64 today; aarch64/others later) and the binary-format
// containers (ELF today; PE later).
//
// An assembler lowers each function to a flat byte buffer plus a list of
// PC-relative-32 fixups; the container lays the functions + data out at known
// virtual addresses and resolves every fixup. The container therefore needs to
// know nothing about the instruction encoding beyond "patch a little-endian
// 32-bit PC-relative field" — which is all x86-64 `call rel32` and RIP-relative
// `lea`/`mov` displacements require.
//
// Native-only (NOT part of the tinyC self-host source set), so it may use the
// full C the host clang/gcc accepts.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_codegen_buf.h"

#ifdef __cplusplus
extern "C" {
#endif

// One PC-relative 32-bit fixup inside a function's code buffer.
//
//   patched_value = (target_VA + addend) - (own_func_VA + pcrel_from)
//
// For x86-64 the displacement is measured from the end of the instruction, so
// `pcrel_from` is the byte offset of the field plus 4. `to_data` selects the
// target: the module data blob (target_VA = data_VA, addend = byte offset into
// it) or another function (target_VA = that function's VA, by name in `sym`).
typedef struct {
    uint32_t off;          // offset of the 4-byte LE field within the function
    uint32_t pcrel_from;   // within-function offset the displacement is relative to
    bool     to_data;      // true: target is the data blob; false: target is `sym`
    char    *sym;          // callee function name (heap-owned) when !to_data
    int64_t  addend;       // data-blob byte offset when to_data
} ObjReloc;

typedef struct {
    char    *name;         // symbol name (heap-owned)
    bool     exported;
    Buf      code;         // machine-code bytes
    ObjReloc *relocs;
    size_t   n_relocs, c_relocs;
    uint64_t va;           // assigned by the container during layout
} ObjFunc;

typedef struct {
    ObjFunc *funcs;
    size_t   n_funcs, c_funcs;
    uint8_t *data;         // module data blob (globals / string literals)
    size_t   data_len;
    size_t   start_idx;    // index of the entry function (`_start`)
} ObjModule;

// Append a fresh empty function and return it (stable until the module is
// freed; callers must not retain the pointer across another obj_add_func).
ObjFunc *obj_add_func(ObjModule *m, char *name, bool exported);

// Record a PC-relative-32 fixup on the current end of `f`'s relocs list.
void obj_add_reloc(ObjFunc *f, uint32_t off, uint32_t pcrel_from,
                   bool to_data, char *sym, int64_t addend);

void obj_module_free(ObjModule *m);

#ifdef __cplusplus
}
#endif
