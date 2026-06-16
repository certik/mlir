// Static ELF64 x86-64 executable container + ObjModule helpers.
// See mlir_elf.h / mlir_obj.h.

#include "mlir_elf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// ObjModule helpers (shared by the x64 code generator and this container).
// ---------------------------------------------------------------------------
ObjFunc *obj_add_func(ObjModule *m, char *name, bool exported) {
    if (m->n_funcs == m->c_funcs) {
        m->c_funcs = m->c_funcs ? m->c_funcs * 2 : 8;
        m->funcs = (ObjFunc *)realloc(m->funcs, m->c_funcs * sizeof(ObjFunc));
    }
    ObjFunc *f = &m->funcs[m->n_funcs++];
    memset(f, 0, sizeof(*f));
    f->name = name;
    f->exported = exported;
    return f;
}

void obj_add_reloc(ObjFunc *f, uint32_t off, uint32_t pcrel_from,
                   bool to_data, char *sym, int64_t addend) {
    if (f->n_relocs == f->c_relocs) {
        f->c_relocs = f->c_relocs ? f->c_relocs * 2 : 4;
        f->relocs = (ObjReloc *)realloc(f->relocs, f->c_relocs * sizeof(ObjReloc));
    }
    ObjReloc *r = &f->relocs[f->n_relocs++];
    r->off = off;
    r->pcrel_from = pcrel_from;
    r->to_data = to_data;
    r->sym = sym;
    r->addend = addend;
}

void obj_module_free(ObjModule *m) {
    for (size_t i = 0; i < m->n_funcs; i++) {
        free(m->funcs[i].name);
        free(m->funcs[i].code.data);
        for (size_t k = 0; k < m->funcs[i].n_relocs; k++)
            free(m->funcs[i].relocs[k].sym);
        free(m->funcs[i].relocs);
    }
    free(m->funcs);
    free(m->data);
    m->funcs = NULL; m->n_funcs = m->c_funcs = 0;
    m->data = NULL; m->data_len = 0;
}

// ---------------------------------------------------------------------------
// ELF64 emission.
// ---------------------------------------------------------------------------
#define ELF_BASE_VADDR 0x400000ull
#define EHDR_SIZE      64u
#define PHDR_SIZE      56u
#define ELF_PAGE       0x1000ull

static size_t align_up_sz(size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); }

static int64_t obj_find_func_va(ObjModule *m, const char *sym, uint64_t *out) {
    for (size_t i = 0; i < m->n_funcs; i++) {
        if (strcmp(m->funcs[i].name, sym) == 0) { *out = m->funcs[i].va; return 0; }
    }
    return -1;
}

bool mlir_obj_to_elf(ObjModule *m, uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;
    if (m->n_funcs == 0 || m->start_idx >= m->n_funcs) {
        fprintf(stderr, "elf: empty module or invalid start index\n");
        return false;
    }

    // Layout order: entry (_start) first, then the rest in module order.
    size_t n = m->n_funcs;
    size_t *order = (size_t *)malloc(n * sizeof(size_t));
    size_t oi = 0;
    order[oi++] = m->start_idx;
    for (size_t i = 0; i < n; i++) if (i != m->start_idx) order[oi++] = i;

    uint64_t code_off = EHDR_SIZE + PHDR_SIZE;
    uint64_t cursor = code_off;
    for (size_t k = 0; k < n; k++) {
        ObjFunc *f = &m->funcs[order[k]];
        f->va = ELF_BASE_VADDR + cursor;
        cursor += f->code.len;
    }
    uint64_t code_end = cursor;
    uint64_t data_off = align_up_sz((size_t)code_end, 16);
    uint64_t data_va  = ELF_BASE_VADDR + data_off;
    uint64_t total    = data_off + m->data_len;

    // Resolve every PC-relative-32 fixup against the final VAs.
    for (size_t i = 0; i < n; i++) {
        ObjFunc *f = &m->funcs[i];
        for (size_t r = 0; r < f->n_relocs; r++) {
            ObjReloc *rl = &f->relocs[r];
            uint64_t target_va;
            if (rl->to_data) {
                target_va = data_va + (uint64_t)rl->addend;
            } else {
                if (obj_find_func_va(m, rl->sym, &target_va) != 0) {
                    fprintf(stderr, "elf: unresolved call to '%s'\n", rl->sym);
                    free(order);
                    return false;
                }
            }
            uint64_t site_va = f->va + rl->pcrel_from;
            int32_t val = (int32_t)((int64_t)target_va - (int64_t)site_va);
            uint8_t *p = f->code.data + rl->off;
            p[0] = (uint8_t)(val & 0xff);
            p[1] = (uint8_t)((val >> 8) & 0xff);
            p[2] = (uint8_t)((val >> 16) & 0xff);
            p[3] = (uint8_t)((val >> 24) & 0xff);
        }
    }

    uint64_t entry = m->funcs[m->start_idx].va;

    Buf img = {0};
    // ELF header.
    static const uint8_t ident[16] = {0x7f,'E','L','F',2,1,1,0,0,0,0,0,0,0,0,0};
    buf_append(&img, ident, 16);
    buf_le16(&img, 2);            // e_type = ET_EXEC
    buf_le16(&img, 62);           // e_machine = EM_X86_64
    buf_le32(&img, 1);            // e_version
    buf_le64(&img, entry);        // e_entry
    buf_le64(&img, EHDR_SIZE);    // e_phoff
    buf_le64(&img, 0);            // e_shoff
    buf_le32(&img, 0);            // e_flags
    buf_le16(&img, (uint16_t)EHDR_SIZE);  // e_ehsize
    buf_le16(&img, (uint16_t)PHDR_SIZE);  // e_phentsize
    buf_le16(&img, 1);            // e_phnum
    buf_le16(&img, 0);            // e_shentsize
    buf_le16(&img, 0);            // e_shnum
    buf_le16(&img, 0);            // e_shstrndx

    // One PT_LOAD covering the whole image, mapped R+W+X.
    buf_le32(&img, 1);            // p_type = PT_LOAD
    buf_le32(&img, 7);            // p_flags = R|W|X
    buf_le64(&img, 0);            // p_offset
    buf_le64(&img, ELF_BASE_VADDR); // p_vaddr
    buf_le64(&img, ELF_BASE_VADDR); // p_paddr
    buf_le64(&img, total);        // p_filesz
    buf_le64(&img, total);        // p_memsz
    buf_le64(&img, ELF_PAGE);     // p_align

    // Code, in layout order.
    for (size_t k = 0; k < n; k++) {
        ObjFunc *f = &m->funcs[order[k]];
        buf_append(&img, f->code.data, f->code.len);
    }
    // Pad to data, then the data blob.
    buf_pad_to(&img, (size_t)data_off);
    if (m->data_len) buf_append(&img, m->data, m->data_len);

    free(order);
    *out_data = img.data;
    *out_size = img.len;
    return true;
}
