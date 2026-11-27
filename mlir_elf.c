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
    r->sym_len = 0;
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

static uint32_t elf_str_hash(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

typedef struct {
    const char *name;
    size_t      len;
    uint32_t    hash;
    uint32_t    val;   // index or sym_idx (0 = empty)
} ElfSymEntry;

static uint32_t elf_sym_lookup(const ElfSymEntry *tab, size_t mask,
                               const char *name, size_t len, uint32_t h) {
    size_t k = (size_t)h & mask;
    while (tab[k].name) {
        if (tab[k].hash == h && tab[k].len == len &&
            memcmp(tab[k].name, name, len) == 0)
            return tab[k].val;
        k = (k + 1) & mask;
    }
    return 0;
}

static void elf_sym_insert(ElfSymEntry *tab, size_t mask,
                           const char *name, size_t len, uint32_t h, uint32_t val) {
    size_t k = (size_t)h & mask;
    while (tab[k].name) {
        if (tab[k].hash == h && tab[k].len == len &&
            memcmp(tab[k].name, name, len) == 0)
            return;
        k = (k + 1) & mask;
    }
    tab[k].name = name;
    tab[k].len = len;
    tab[k].hash = h;
    tab[k].val = val;
}

bool mlir_obj_to_elf(ObjModule *m, uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;
    if (m->n_funcs == 0 || m->start_idx >= m->n_funcs) {
        fprintf(stderr, "elf: empty module or invalid start index\n");
        return false;
    }

    // Layout order: entry (_start) first, then the rest in module order.
    size_t n = m->n_funcs;
    size_t order_stk[256];
    size_t *order = (n <= 256) ? order_stk : (size_t *)malloc(n * sizeof(size_t));
    size_t oi = 0;
    order[oi++] = m->start_idx;
    for (size_t i = 0; i < n; i++) if (i != m->start_idx) order[oi++] = i;

    size_t hcap = 16;
    while (hcap < n * 2u + 1u) hcap <<= 1;
    ElfSymEntry htab_stk[256];
    ElfSymEntry *htab;
    if (hcap <= 256) {
        htab = htab_stk;
        memset(htab, 0, hcap * sizeof(ElfSymEntry));
    } else {
        htab = (ElfSymEntry *)calloc(hcap, sizeof(ElfSymEntry));
    }
    size_t hmask = hcap - 1;

    uint64_t code_off = EHDR_SIZE + PHDR_SIZE;
    uint64_t cursor = code_off;
    for (size_t k = 0; k < n; k++) {
        size_t fi = order[k];
        ObjFunc *f = &m->funcs[fi];
        f->va = ELF_BASE_VADDR + cursor;
        cursor += f->code.len;
        size_t nl = strlen(f->name);
        elf_sym_insert(htab, hmask, f->name, nl, elf_str_hash(f->name, nl), (uint32_t)(fi + 1));
    }
    uint64_t code_end = cursor;
    uint64_t data_off = align_up_sz((size_t)code_end, 16);
    uint64_t data_va  = ELF_BASE_VADDR + data_off;
    uint64_t total    = data_off + m->data_len + m->bss_len;

    // Resolve every PC-relative-32 fixup against the final VAs.
    for (size_t i = 0; i < n; i++) {
        ObjFunc *f = &m->funcs[i];
        for (size_t r = 0; r < f->n_relocs; r++) {
            ObjReloc *rl = &f->relocs[r];
            uint64_t target_va;
            if (rl->to_data) {
                target_va = data_va + (uint64_t)rl->addend;
            } else {
                size_t sl = rl->sym_len ? (size_t)rl->sym_len : strlen(rl->sym);
                uint32_t idx1 = elf_sym_lookup(htab, hmask, rl->sym, sl, elf_str_hash(rl->sym, sl));
                if (idx1 == 0) {
                    fprintf(stderr, "elf: unresolved call to '%s'\n", rl->sym);
                    if (htab != htab_stk) free(htab);
                    if (order != order_stk) free(order);
                    return false;
                }
                target_va = m->funcs[idx1 - 1].va;
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
    if (htab != htab_stk) free(htab);

    // Trim trailing zero bytes from `.data` into `.bss` (p_filesz < p_memsz):
    // the kernel zero-fills [p_filesz .. p_memsz), avoiding copying/writing
    // large zero-initialized globals to disk.
    size_t data_filesz = m->data_len;
    while (data_filesz >= 8) {
        uint64_t w;
        memcpy(&w, m->data + data_filesz - 8, 8);
        if (w != 0) break;
        data_filesz -= 8;
    }
    while (data_filesz > 0 && m->data[data_filesz - 1] == 0) data_filesz--;
    uint64_t filesz = data_filesz ? (data_off + data_filesz) : code_end;

    uint64_t entry = m->funcs[m->start_idx].va;

    Buf img = {0};
    buf_grow(&img, (size_t)filesz);
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
    buf_le64(&img, filesz);       // p_filesz
    buf_le64(&img, total);        // p_memsz
    buf_le64(&img, ELF_PAGE);     // p_align

    // Code, in layout order.
    for (size_t k = 0; k < n; k++) {
        ObjFunc *f = &m->funcs[order[k]];
        buf_append(&img, f->code.data, f->code.len);
    }
    // Pad to data, then the initialized prefix of the data blob.
    if (data_filesz) {
        buf_pad_to(&img, (size_t)data_off);
        buf_append(&img, m->data, data_filesz);
    }

    if (order != order_stk) free(order);
    *out_data = img.data;
    *out_size = img.len;
    return true;
}

// ---------------------------------------------------------------------------
// ELF64 relocatable object (.o, ET_REL) writer.
// Sections:
//   [0] SHT_NULL
//   [1] .text      (PROGBITS, ALLOC|EXECINSTR)
//   [2] .data      (PROGBITS, ALLOC|WRITE)
//   [3] .rela.text (RELA, sh_link=4, sh_info=1)
//   [4] .symtab    (SYMTAB, sh_link=5, sh_info=3 [first global sym])
//   [5] .strtab    (STRTAB)
//   [6] .shstrtab  (STRTAB)
// ---------------------------------------------------------------------------
static void elf_write_shdr(Buf *img, uint32_t sh_name, uint32_t sh_type,
                           uint64_t sh_flags, uint64_t sh_offset,
                           uint64_t sh_size, uint32_t sh_link,
                           uint32_t sh_info, uint64_t sh_align,
                           uint64_t sh_entsize) {
    buf_le32(img, sh_name);
    buf_le32(img, sh_type);
    buf_le64(img, sh_flags);
    buf_le64(img, 0);          // sh_addr
    buf_le64(img, sh_offset);
    buf_le64(img, sh_size);
    buf_le32(img, sh_link);
    buf_le32(img, sh_info);
    buf_le64(img, sh_align);
    buf_le64(img, sh_entsize);
}

static void elf_write_sym(Buf *img, uint32_t st_name, uint8_t st_info,
                          uint16_t st_shndx, uint64_t st_value, uint64_t st_size) {
    buf_le32(img, st_name);
    buf_u8(img, st_info);
    buf_u8(img, 0);            // st_other = STV_DEFAULT
    buf_le16(img, st_shndx);
    buf_le64(img, st_value);
    buf_le64(img, st_size);
}

bool mlir_obj_to_elf_reloc(const uint8_t *text, size_t text_len,
                           const uint8_t *data, size_t data_len,
                           const ObjFuncSym *funcs, size_t n_funcs,
                           const ObjReloc *relocs, size_t n_relocs,
                           uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;

    // Build symbol map: [0]=null, [1]=.text sec, [2]=.data sec,
    // [3..3+n_funcs-1]=defined global funcs, [3+n_funcs..]=external undefs.
    size_t max_syms = 3 + n_funcs + n_relocs;
    size_t hcap = 32;
    while (hcap < max_syms * 2u + 1u) hcap <<= 1;
    ElfSymEntry htab_stk[256];
    ElfSymEntry *htab;
    if (hcap <= 256) {
        htab = htab_stk;
        memset(htab, 0, hcap * sizeof(ElfSymEntry));
    } else {
        htab = (ElfSymEntry *)calloc(hcap, sizeof(ElfSymEntry));
    }
    size_t hmask = hcap - 1;

    Buf strtab = {0};
    buf_grow(&strtab, (n_funcs + 16) * 24);
    buf_u8(&strtab, 0); // offset 0 = ""

    size_t scratch_n = n_funcs + 2 * n_relocs + 3;
    uint32_t scratch_stk[1024];
    uint32_t *scratch = (scratch_n <= 1024) ? scratch_stk
        : (uint32_t *)malloc(scratch_n * sizeof(uint32_t));
    uint32_t *def_name_off = scratch;
    uint32_t *ext_name_off = def_name_off + (n_funcs ? n_funcs : 1);
    uint32_t *rel_sym_idx  = ext_name_off + (n_relocs ? n_relocs : 1);

    for (size_t i = 0; i < n_funcs; i++) {
        def_name_off[i] = (uint32_t)strtab.len;
        buf_strn(&strtab, funcs[i].name, funcs[i].name_len);
        uint32_t h = elf_str_hash(funcs[i].name, funcs[i].name_len);
        elf_sym_insert(htab, hmask, funcs[i].name, funcs[i].name_len, h, (uint32_t)(3 + i));
    }

    // Collect external undefined symbols + count emitted relocations.
    size_t n_ext = 0;
    size_t n_emit_relocs = 0;
    uint32_t first_ext = (uint32_t)(3 + n_funcs);
    for (size_t r = 0; r < n_relocs; r++) {
        const ObjReloc *rl = &relocs[r];
        if (rl->to_data) {
            rel_sym_idx[r] = 2u; // .data section symbol
            n_emit_relocs++;
        } else {
            size_t sl = rl->sym_len ? (size_t)rl->sym_len : strlen(rl->sym);
            uint32_t h = elf_str_hash(rl->sym, sl);
            uint32_t si = elf_sym_lookup(htab, hmask, rl->sym, sl, h);
            if (si == 0) {
                si = first_ext + (uint32_t)n_ext;
                ext_name_off[n_ext++] = (uint32_t)strtab.len;
                buf_strn(&strtab, rl->sym, sl);
                elf_sym_insert(htab, hmask, rl->sym, sl, h, si);
            }
            rel_sym_idx[r] = si;
            n_emit_relocs++;
        }
    }

    // Section header string table (.shstrtab).
    static const char shstrtab_bytes[] =
        "\0.text\0.data\0.rela.text\0.symtab\0.strtab\0.shstrtab";
    const size_t shstrtab_len = sizeof(shstrtab_bytes);
    // Offsets into shstrtab_bytes:
    // 0="", 1=".text", 7=".data", 13=".rela.text", 24=".symtab", 32=".strtab", 40=".shstrtab"

    size_t n_syms = 3 + n_funcs + n_ext;
    size_t text_off     = align_up_sz(EHDR_SIZE, 16);
    size_t data_off     = align_up_sz(text_off + text_len, 16);
    size_t rela_off     = align_up_sz(data_off + data_len, 8);
    size_t rela_size    = n_emit_relocs * 24u;
    size_t symtab_off   = align_up_sz(rela_off + rela_size, 8);
    size_t symtab_size  = n_syms * 24u;
    size_t strtab_off   = symtab_off + symtab_size;
    size_t shstrtab_off = strtab_off + strtab.len;
    size_t shdr_off     = align_up_sz(shstrtab_off + shstrtab_len, 8);
    size_t total        = shdr_off + 7u * 64u;

    Buf img = {0};
    buf_grow(&img, total);

    // ELF64 header (ET_REL).
    static const uint8_t ident[16] = {0x7f,'E','L','F',2,1,1,0,0,0,0,0,0,0,0,0};
    buf_append(&img, ident, 16);
    buf_le16(&img, 1);                  // e_type = ET_REL
    buf_le16(&img, 62);                 // e_machine = EM_X86_64
    buf_le32(&img, 1);                  // e_version
    buf_le64(&img, 0);                  // e_entry
    buf_le64(&img, 0);                  // e_phoff
    buf_le64(&img, (uint64_t)shdr_off); // e_shoff
    buf_le32(&img, 0);                  // e_flags
    buf_le16(&img, (uint16_t)EHDR_SIZE);// e_ehsize
    buf_le16(&img, 0);                  // e_phentsize
    buf_le16(&img, 0);                  // e_phnum
    buf_le16(&img, 64);                 // e_shentsize
    buf_le16(&img, 7);                  // e_shnum
    buf_le16(&img, 6);                  // e_shstrndx

    // [1] .text
    buf_pad_to(&img, text_off);
    if (text_len) buf_append(&img, text, text_len);

    // [2] .data
    buf_pad_to(&img, data_off);
    if (data_len) buf_append(&img, data, data_len);

    // [3] .rela.text (Elf64_Rela: r_offset, r_info, r_addend)
    buf_pad_to(&img, rela_off);
    for (size_t r = 0; r < n_relocs; r++) {
        const ObjReloc *rl = &relocs[r];
        uint32_t si = rel_sym_idx[r];
        uint32_t rtype = rl->to_data ? 2u /* R_X86_64_PC32 */ : 4u /* R_X86_64_PLT32 */;
        int64_t disp_bias = (int64_t)rl->pcrel_from - (int64_t)rl->off;
        int64_t addend = (rl->to_data ? rl->addend : 0) - disp_bias;
        buf_le64(&img, (uint64_t)rl->off);
        buf_le64(&img, ((uint64_t)si << 32) | (uint64_t)rtype);
        buf_le64(&img, (uint64_t)addend);
    }

    // [4] .symtab
    buf_pad_to(&img, symtab_off);
    elf_write_sym(&img, 0, 0, 0, 0, 0);                         // [0] null
    elf_write_sym(&img, 0, (0u << 4) | 3u, 1, 0, 0);            // [1] .text (LOCAL SECTION)
    elf_write_sym(&img, 0, (0u << 4) | 3u, 2, 0, 0);            // [2] .data (LOCAL SECTION)
    for (size_t i = 0; i < n_funcs; i++) {
        elf_write_sym(&img, def_name_off[i], (1u << 4) | 2u,    // GLOBAL FUNC
                      1, (uint64_t)funcs[i].off, (uint64_t)funcs[i].size);
    }
    for (size_t i = 0; i < n_ext; i++) {
        elf_write_sym(&img, ext_name_off[i], (1u << 4) | 0u,    // GLOBAL NOTYPE UNDEF
                      0, 0, 0);
    }

    // [5] .strtab
    buf_append(&img, strtab.data, strtab.len);
    // [6] .shstrtab
    buf_append(&img, shstrtab_bytes, shstrtab_len);

    // Section headers (7 entries).
    buf_pad_to(&img, shdr_off);
    elf_write_shdr(&img,  0, 0, 0, 0, 0, 0, 0, 0, 0);                                    // [0] NULL
    elf_write_shdr(&img,  1, 1, 6, text_off, text_len, 0, 0, 16, 0);                     // [1] .text
    elf_write_shdr(&img,  7, 1, 3, data_off, data_len, 0, 0, 16, 0);                     // [2] .data
    elf_write_shdr(&img, 13, 4, 0, rela_off, rela_size, 4, 1, 8, 24);                    // [3] .rela.text
    elf_write_shdr(&img, 24, 2, 0, symtab_off, symtab_size, 5, 3, 8, 24);                // [4] .symtab
    elf_write_shdr(&img, 32, 3, 0, strtab_off, strtab.len, 0, 0, 1, 0);                  // [5] .strtab
    elf_write_shdr(&img, 40, 3, 0, shstrtab_off, shstrtab_len, 0, 0, 1, 0);              // [6] .shstrtab

    free(strtab.data);
    if (scratch != scratch_stk) free(scratch);
    if (htab != htab_stk) free(htab);
    *out_data = img.data;
    *out_size = img.len;
    return true;
}
