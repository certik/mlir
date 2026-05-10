// Constructors / destructors for the wasmssa and wasmstack C-struct
// intermediate representations declared in mlir_wasm_dialect.h.

#include <stdlib.h>
#include <string.h>

#include "mlir_wasm_dialect.h"

// ---- generic growable-array helper -----------------------------------------
// Note: we embed length+capacity into each container struct; helpers below
// just amortized-double when needed.

#define ENSURE(arr, n, c, T) do { \
    if ((n) == (c)) { \
        (c) = (c) ? (c) * 2 : 8; \
        (arr) = (T *)realloc((arr), (c) * sizeof(T)); \
    } \
} while (0)

// ============================================================================
// wasmssa
// ============================================================================
wasmssa_module_t *wasmssa_module_new(void) {
    wasmssa_module_t *m = (wasmssa_module_t *)calloc(1, sizeof(*m));
    return m;
}

static void wasmssa_func_free(wasmssa_func_t *f) {
    free(f->name);
    free(f->param_types);
    free(f->result_types);
    free(f->carrier_vts);
    for (size_t i = 0; i < f->n_ops; i++) {
        free(f->ops[i].operands);
        free(f->ops[i].call_target);
        free(f->ops[i].sig_params);
        free(f->ops[i].sig_results);
    }
    free(f->ops);
}

void wasmssa_module_free(wasmssa_module_t *m) {
    if (!m) return;
    for (size_t i = 0; i < m->n_funcs; i++) wasmssa_func_free(&m->funcs[i]);
    free(m->funcs);
    for (size_t i = 0; i < m->n_globals; i++) {
        free(m->globals[i].name);
        free(m->globals[i].data);
        for (size_t j = 0; j < m->globals[i].n_relocs; j++)
            free(m->globals[i].relocs[j].target);
        free(m->globals[i].relocs);
    }
    free(m->globals);
    free(m);
}

wasmssa_func_t *wasmssa_module_add_func(wasmssa_module_t *m) {
    ENSURE(m->funcs, m->n_funcs, m->c_funcs, wasmssa_func_t);
    wasmssa_func_t *f = &m->funcs[m->n_funcs++];
    memset(f, 0, sizeof(*f));
    return f;
}

wasmssa_op_t *wasmssa_func_add_op(wasmssa_func_t *f) {
    ENSURE(f->ops, f->n_ops, f->c_ops, wasmssa_op_t);
    wasmssa_op_t *o = &f->ops[f->n_ops++];
    memset(o, 0, sizeof(*o));
    return o;
}

// ============================================================================
// wasmstack
// ============================================================================
wasmstack_module_t *wasmstack_module_new(void) {
    wasmstack_module_t *m = (wasmstack_module_t *)calloc(1, sizeof(*m));
    return m;
}

static void wasmstack_func_free(wasmstack_func_t *f) {
    free(f->name);
    free(f->param_types);
    free(f->result_types);
    free(f->local_types);
    for (size_t i = 0; i < f->n_ops; i++) {
        free(f->ops[i].call_target);
        free(f->ops[i].sig_params);
        free(f->ops[i].sig_results);
    }
    free(f->ops);
}

void wasmstack_module_free(wasmstack_module_t *m) {
    if (!m) return;
    for (size_t i = 0; i < m->n_funcs; i++) wasmstack_func_free(&m->funcs[i]);
    free(m->funcs);
    for (size_t i = 0; i < m->n_globals; i++) {
        free(m->globals[i].name);
        free(m->globals[i].data);
        for (size_t j = 0; j < m->globals[i].n_relocs; j++)
            free(m->globals[i].relocs[j].target);
        free(m->globals[i].relocs);
    }
    free(m->globals);
    free(m);
}

wasmstack_func_t *wasmstack_module_add_func(wasmstack_module_t *m) {
    ENSURE(m->funcs, m->n_funcs, m->c_funcs, wasmstack_func_t);
    wasmstack_func_t *f = &m->funcs[m->n_funcs++];
    memset(f, 0, sizeof(*f));
    return f;
}

wasmstack_op_t *wasmstack_func_add_op(wasmstack_func_t *f) {
    ENSURE(f->ops, f->n_ops, f->c_ops, wasmstack_op_t);
    wasmstack_op_t *o = &f->ops[f->n_ops++];
    memset(o, 0, sizeof(*o));
    return o;
}

uint32_t wasmstack_func_add_local(wasmstack_func_t *f, uint8_t vt) {
    ENSURE(f->local_types, f->n_locals, f->c_locals, uint8_t);
    f->local_types[f->n_locals] = vt;
    return (uint32_t)(f->n_params + f->n_locals++);
}

// ============================================================================
// globals
// ============================================================================
wasm_global_t *wasmssa_module_add_global(wasmssa_module_t *m) {
    ENSURE(m->globals, m->n_globals, m->c_globals, wasm_global_t);
    wasm_global_t *g = &m->globals[m->n_globals++];
    memset(g, 0, sizeof(*g));
    return g;
}
wasm_global_t *wasmstack_module_add_global(wasmstack_module_t *m) {
    ENSURE(m->globals, m->n_globals, m->c_globals, wasm_global_t);
    wasm_global_t *g = &m->globals[m->n_globals++];
    memset(g, 0, sizeof(*g));
    return g;
}
void wasm_global_add_reloc(wasm_global_t *g, uint32_t off,
                           const char *target, int32_t addend) {
    ENSURE(g->relocs, g->n_relocs, g->c_relocs, wasm_data_reloc_t);
    wasm_data_reloc_t *r = &g->relocs[g->n_relocs++];
    r->offset = off;
    size_t n = strlen(target);
    r->target = (char *)malloc(n + 1);
    memcpy(r->target, target, n + 1);
    r->addend = addend;
}
