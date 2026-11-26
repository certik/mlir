// tinyC preprocessor — a self-contained C preprocessor used to feed the
// existing tinyC lexer/parser. Implements the useful subset of C cpp:
//   #include "..." / #include <...>     (with -I search paths)
//   #define / #undef                    (object-like, function-like, variadic)
//   #ifdef / #ifndef / #if / #elif / #else / #endif
//   #if defined(X)  +  unary/binary/ternary integer constant expressions
//   #error                              (diagnostic, then halt)
//   #pragma once                        (per-file include-guard)
//   #line N ["file"]                    (line/file tracking)
//   stringize (#) and token-paste (##)
//   variadic macros:  __VA_ARGS__  __VA_OPT__(...)
//   predefined:       __FILE__  __LINE__  __TINYC__
//
// Output: a single contiguous string suitable for the existing tinyC
// lexer. `#line N "file"` directives are emitted whenever the source
// location changes so diagnostics stay accurate.
//
// Design: text input -> PP-token vector per file -> macro-expanded
// PP-token stream -> serialized text. The lexer/parser layers are
// untouched aside from the lexer accepting `#line` directives.

#include "tinyc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/format.h>
#include <base/io.h>
#include <base/string.h>

// ----------------------------------------------------------------------
// PP token
// ----------------------------------------------------------------------

typedef enum {
    PP_END = 0,
    PP_NEWLINE,    // '\n' end of logical line (for directive parsing)
    PP_IDENT,
    PP_NUMBER,     // pp-number (digits, letters, dots, signed exponents)
    PP_STRING,     // "..." (raw form, including surrounding quotes)
    PP_CHAR,       // '...' (raw form)
    PP_PUNCT,      // operator / punctuation (text holds the literal bytes)
    PP_HASH,       // # (only at start of directive line, after dispatch)
    PP_HASHHASH,   // ##
    PP_PLACEMARKER // empty token used only during expansion (for ##)
} PPKind;

typedef struct PPHide PPHide;
struct PPHide {
    string  name;
    PPHide *next;
};

typedef struct {
    uint8_t  kind;      // PPKind
    uint8_t  has_space; // whitespace immediately preceded this token (in source)
    uint16_t file_id;   // index into pp->files[]
    int      line;      // source line in `file`
    string   text;      // raw bytes (may be empty for newline / placemarker)
    PPHide  *hide;      // hide-set chain (for non-self-recursion)
} PPTok;

DEFINE_VECTOR_FOR_TYPE(PPTok, VecPPTok)

// ----------------------------------------------------------------------
// Macro table (open-addressing hash table of identifier names)
// ----------------------------------------------------------------------

typedef struct Macro Macro;
struct Macro {
    string   name;
    bool     is_func;
    bool     is_variadic;
    string  *params;     // arena-allocated; n_params entries
    size_t   n_params;
    PPTok   *body;       // arena-allocated; n_body entries
    size_t   n_body;
    bool     expanding;          // currently-expanding flag (no self-recursion)
};

// One slot per identifier name ever defined (or predefined). A slot is never
// removed: `#undef` just clears `macro`, so probing needs no tombstones.
// `special` marks the names expand_tokens() replaces before any macro lookup.
#define PP_SPECIAL_NONE 0
#define PP_SPECIAL_FILE 1
#define PP_SPECIAL_LINE 2

typedef struct {
    string   name;      // name.str == NULL: empty slot
    uint32_t hash;
    int      special;   // PP_SPECIAL_*
    Macro   *macro;     // NULL: not currently defined
} MacroSlot;

// ----------------------------------------------------------------------
// State
// ----------------------------------------------------------------------

typedef struct PathNode {
    string path;          // canonical-ish path (as read; we don't normalize)
    struct PathNode *next;
} PathNode;

typedef struct {
    Arena    *arena;
    MacroSlot *mtab;             // macro_cap slots, a power of two
    size_t    macro_cap;
    size_t    macro_used;        // occupied slots
    uint64_t  macro_filter[64];  // 4096-bit L1 filter on (c0, c_last, len) to skip macro_find_slot
    string   *include_dirs;
    size_t    n_include_dirs;
    PathNode *pragma_once;       // files that emitted `#pragma once`
    // collect_args() gathers argument tokens here (reused, only grows).
    PPTok    *arg_buf;
    size_t    arg_buf_cap;
    // Rewinding stack buffer for macro-expanded token slices (avoids arena churn).
    PPTok    *exp_buf;
    size_t    exp_top;
    size_t    exp_cap;
    // Output text buffer (grown geometrically).
    char     *out;
    size_t    out_size;
    size_t    out_cap;
    // Direct TcTok output (non-NULL in tinyc_preprocess_lex() fused mode).
    VecTcTok *tok_out;
    // Interned file names (indexed by PPTok.file_id).
    string   *files;
    size_t    n_files;
    size_t    files_cap;
    // Current emission location — used to decide when to emit a #line.
    uint16_t  cur_file_id;
    int       cur_line;
    bool      cur_loc_valid;
    int       n_errors;
} PP;

static uint16_t pp_intern_file(PP *pp, string file) {
    for (size_t i = 0; i < pp->n_files; i++) {
        if (pp->files[i].size == file.size &&
            (pp->files[i].str == file.str ||
             memcmp(pp->files[i].str, file.str, file.size) == 0)) {
            return (uint16_t)i;
        }
    }
    if (pp->n_files == pp->files_cap) {
        size_t ncap = pp->files_cap ? pp->files_cap * 2 : 32;
        string *nf = arena_new_array(pp->arena, string, ncap);
        if (pp->n_files) memcpy(nf, pp->files, pp->n_files * sizeof(string));
        pp->files = nf;
        pp->files_cap = ncap;
    }
    uint16_t id = (uint16_t)pp->n_files++;
    pp->files[id] = file;
    return id;
}

// ----------------------------------------------------------------------
// Diagnostics
// ----------------------------------------------------------------------

// Diagnostics go to stderr, so `tinyc x.c > x.ll` never writes them into the
// output.
static void pp_error(PP *pp, string file, int line, string msg) {
    pp->n_errors++;
    string text = format(pp->arena, str_lit("tinyc preprocess error at {}:{}: {}\n"),
                         file, (int64_t)line, msg);
    ciovec_t iov;
    iov.buf = text.str;
    iov.buf_len = text.size;
    write_all(PLATFORM_STDERR_FD, &iov, 1);
}

// ----------------------------------------------------------------------
// Output buffer helpers
// ----------------------------------------------------------------------

static void out_grow(PP *pp, size_t need) {
    if (pp->out_size + need <= pp->out_cap) return;
    size_t new_cap = pp->out_cap ? pp->out_cap * 2 : 4096;
    while (new_cap < pp->out_size + need) new_cap *= 2;
    char *nb = arena_new_array(pp->arena, char, new_cap);
    if (pp->out_size) memcpy(nb, pp->out, pp->out_size);
    pp->out = nb;
    pp->out_cap = new_cap;
}

static void out_append(PP *pp, string s) {
    out_grow(pp, s.size);
    // Tokens are a few bytes: a plain loop beats a memcpy call.
    char *d = pp->out + pp->out_size;
    for (size_t k = 0; k < s.size; k++) d[k] = s.str[k];
    pp->out_size += s.size;
}

static void out_char(PP *pp, char c) {
    out_grow(pp, 1);
    pp->out[pp->out_size++] = c;
}

// ----------------------------------------------------------------------
// String helpers (no libc beyond memcpy / strlen which are fine)
// ----------------------------------------------------------------------

static string s_lit(const char *p) {
    string r;
    r.str = (char *)p;
    r.size = strlen(p);
    return r;
}

static string s_dup_cstr(Arena *arena, const char *p, size_t n) {
    char *b = arena_new_array(arena, char, n + 1);
    memcpy(b, p, n);
    b[n] = '\0';
    string r = { b, n };
    return r;
}

static string s_dup(Arena *arena, string s) {
    return s_dup_cstr(arena, s.str, s.size);
}

static string s_concat3(Arena *arena, string a, string b, string c) {
    char *buf = arena_new_array(arena, char, a.size + b.size + c.size + 1);
    if (a.size) memcpy(buf, a.str, a.size);
    if (b.size) memcpy(buf + a.size, b.str, b.size);
    if (c.size) memcpy(buf + a.size + b.size, c.str, c.size);
    buf[a.size + b.size + c.size] = '\0';
    string r = { buf, a.size + b.size + c.size };
    return r;
}

static string i64_to_str(Arena *arena, int64_t v) {
    char tmp[32];
    int n = 0;
    bool neg = v < 0;
    uint64_t u = neg ? (uint64_t)-v : (uint64_t)v;
    if (u == 0) tmp[n++] = '0';
    while (u) { tmp[n++] = (char)('0' + (u % 10)); u /= 10; }
    if (neg) tmp[n++] = '-';
    char *buf = arena_new_array(arena, char, n + 1);
    for (int i = 0; i < n; i++) buf[i] = tmp[n - 1 - i];
    buf[n] = '\0';
    string r = { buf, (uint64_t)n };
    return r;
}

// ----------------------------------------------------------------------
// PP tokenizer
// ----------------------------------------------------------------------

#define PP_CT_ALPHA 1u
#define PP_CT_DIGIT 2u
#define PP_CT_ALNUM 3u
#define PP_CT_WS    4u

static uint8_t g_pp_ctype[256];
static uint8_t g_pp_punct1[128];
static bool g_pp_ctype_init = false;

static void pp_init_ctype(void) {
    if (g_pp_ctype_init) return;
    for (int c = 'a'; c <= 'z'; c++) g_pp_ctype[c] = PP_CT_ALPHA;
    for (int c = 'A'; c <= 'Z'; c++) g_pp_ctype[c] = PP_CT_ALPHA;
    g_pp_ctype['_'] = PP_CT_ALPHA;
    for (int c = '0'; c <= '9'; c++) g_pp_ctype[c] = PP_CT_DIGIT;
    g_pp_ctype[' '] = PP_CT_WS;
    g_pp_ctype['\t'] = PP_CT_WS;
    g_pp_ctype['\r'] = PP_CT_WS;
    g_pp_punct1['('] = TC_TK_LPAREN;
    g_pp_punct1[')'] = TC_TK_RPAREN;
    g_pp_punct1['{'] = TC_TK_LBRACE;
    g_pp_punct1['}'] = TC_TK_RBRACE;
    g_pp_punct1['['] = TC_TK_LBRACK;
    g_pp_punct1[']'] = TC_TK_RBRACK;
    g_pp_punct1[';'] = TC_TK_SEMI;
    g_pp_punct1[','] = TC_TK_COMMA;
    g_pp_punct1['.'] = TC_TK_DOT;
    g_pp_punct1['+'] = TC_TK_PLUS;
    g_pp_punct1['-'] = TC_TK_MINUS;
    g_pp_punct1['*'] = TC_TK_STAR;
    g_pp_punct1['/'] = TC_TK_SLASH;
    g_pp_punct1['%'] = TC_TK_PERCENT;
    g_pp_punct1['<'] = TC_TK_LT;
    g_pp_punct1['>'] = TC_TK_GT;
    g_pp_punct1['='] = TC_TK_ASSIGN;
    g_pp_punct1['!'] = TC_TK_BANG;
    g_pp_punct1['&'] = TC_TK_AMP;
    g_pp_punct1['|'] = TC_TK_PIPE;
    g_pp_punct1['^'] = TC_TK_CARET;
    g_pp_punct1['~'] = TC_TK_TILDE;
    g_pp_punct1['?'] = TC_TK_QUESTION;
    g_pp_punct1[':'] = TC_TK_COLON;
    g_pp_ctype_init = true;
}

static void pp_vec_grow(Arena *arena, VecPPTok *out) {
    size_t nmax = out->max ? out->max * 2 : 256;
    if (arena) {
        PPTok *nd = arena_new_array(arena, PPTok, nmax);
        if (out->size) memcpy(nd, out->data, out->size * sizeof(PPTok));
        out->data = nd;
    } else {
        out->data = (PPTok *)realloc(out->data, nmax * sizeof(PPTok));
    }
    out->max = nmax;
}

// Tokenize `src` into PP tokens. Comments are dropped. Newlines become
// PP_NEWLINE tokens (one per source newline, even adjacent ones — needed
// to keep #line tracking accurate). `has_space` is set when whitespace
// (excluding newlines) preceded the token. Token text points into `src`,
// which must outlive the tokens (it is never modified).
static void pp_tokenize(Arena *arena, string src, uint16_t file_id, VecPPTok *out) {
    const char *s = src.str;
    size_t sz = src.size;
    int line = 1;
    size_t i = 0;
    uint8_t space_pending = 0;
    while (i < sz) {
        char c = s[i];
        uint8_t ct = g_pp_ctype[(unsigned char)c];
        if (ct & PP_CT_WS) {
            i++;
            while (i < sz && (g_pp_ctype[(unsigned char)s[i]] & PP_CT_WS)) i++;
            space_pending = 1;
            continue;
        }
        if (c == '\n') {
            if (out->size > 0 && out->data[out->size - 1].kind != PP_NEWLINE) {
                if (out->size == out->max) pp_vec_grow(arena, out);
                PPTok *t = &out->data[out->size++];
                t->kind = PP_NEWLINE;
                t->has_space = 0;
                t->file_id = file_id;
                t->line = line;
                t->text.str = NULL;
                t->text.size = 0;
                t->hide = NULL;
            }
            i++;
            line++;
            space_pending = 0;
            continue;
        }
        if (ct & PP_CT_ALPHA) {
            size_t j = i + 1;
            while (j < sz && (g_pp_ctype[(unsigned char)s[j]] & PP_CT_ALNUM)) j++;
            if (out->size == out->max) pp_vec_grow(arena, out);
            PPTok *t = &out->data[out->size++];
            t->kind = PP_IDENT;
            t->has_space = space_pending;
            t->file_id = file_id;
            t->line = line;
            t->text.str = (char *)(s + i);
            t->text.size = j - i;
            t->hide = NULL;
            i = j;
            space_pending = 0;
            continue;
        }
        if (ct & PP_CT_DIGIT) {
            size_t j = i + 1;
            while (j < sz) {
                char d = s[j];
                if ((g_pp_ctype[(unsigned char)d] & PP_CT_ALNUM) || d == '.') { j++; continue; }
                if ((d == '+' || d == '-') &&
                    (s[j - 1] == 'e' || s[j - 1] == 'E' ||
                     s[j - 1] == 'p' || s[j - 1] == 'P')) {
                    j++; continue;
                }
                break;
            }
            if (out->size == out->max) pp_vec_grow(arena, out);
            PPTok *t = &out->data[out->size++];
            t->kind = PP_NUMBER;
            t->has_space = space_pending;
            t->file_id = file_id;
            t->line = line;
            t->text.str = (char *)(s + i);
            t->text.size = j - i;
            t->hide = NULL;
            i = j;
            space_pending = 0;
            continue;
        }
        if (c == '\\') {
            if (i + 1 < sz && s[i + 1] == '\n') {
                i += 2;
                line++;
                continue;
            }
            if (i + 2 < sz && s[i + 1] == '\r' && s[i + 2] == '\n') {
                i += 3;
                line++;
                continue;
            }
        }
        if (c == '/' && i + 1 < sz) {
            if (s[i + 1] == '/') {
                i += 2;
                const char *nl = (const char *)memchr(s + i, '\n', sz - i);
                i = nl ? (size_t)(nl - s) : sz;
                space_pending = 1;
                continue;
            }
            if (s[i + 1] == '*') {
                i += 2;
                while (i + 1 < sz && !(s[i] == '*' && s[i + 1] == '/')) {
                    if (s[i] == '\n') line++;
                    i++;
                }
                if (i + 1 < sz) i += 2;
                else i = sz;
                space_pending = 1;
                continue;
            }
        }
        if (c == '"') {
            int start_line = line;
            size_t j = i + 1;
            while (j < sz && s[j] != '"') {
                if (s[j] == '\\' && j + 1 < sz) j += 2;
                else { if (s[j] == '\n') line++; j++; }
            }
            if (j < sz) j++;
            if (out->size == out->max) pp_vec_grow(arena, out);
            PPTok *t = &out->data[out->size++];
            t->kind = PP_STRING;
            t->has_space = space_pending;
            t->file_id = file_id;
            t->line = start_line;
            t->text.str = (char *)(s + i);
            t->text.size = j - i;
            t->hide = NULL;
            i = j;
            space_pending = 0;
            continue;
        }
        if (c == '\'') {
            size_t j = i + 1;
            while (j < sz && s[j] != '\'') {
                if (s[j] == '\\' && j + 1 < sz) j += 2;
                else j++;
            }
            if (j < sz) j++;
            if (out->size == out->max) pp_vec_grow(arena, out);
            PPTok *t = &out->data[out->size++];
            t->kind = PP_CHAR;
            t->has_space = space_pending;
            t->file_id = file_id;
            t->line = line;
            t->text.str = (char *)(s + i);
            t->text.size = j - i;
            t->hide = NULL;
            i = j;
            space_pending = 0;
            continue;
        }
        if (c == '.' && i + 1 < sz && (g_pp_ctype[(unsigned char)s[i + 1]] & PP_CT_DIGIT)) {
            size_t j = i + 2;
            while (j < sz) {
                char d = s[j];
                if ((g_pp_ctype[(unsigned char)d] & PP_CT_ALNUM) || d == '.') { j++; continue; }
                if ((d == '+' || d == '-') &&
                    (s[j - 1] == 'e' || s[j - 1] == 'E' ||
                     s[j - 1] == 'p' || s[j - 1] == 'P')) {
                    j++; continue;
                }
                break;
            }
            if (out->size == out->max) pp_vec_grow(arena, out);
            PPTok *t = &out->data[out->size++];
            t->kind = PP_NUMBER;
            t->has_space = space_pending;
            t->file_id = file_id;
            t->line = line;
            t->text.str = (char *)(s + i);
            t->text.size = j - i;
            t->hide = NULL;
            i = j;
            space_pending = 0;
            continue;
        }
        if (c == '#') {
            if (out->size == out->max) pp_vec_grow(arena, out);
            PPTok *t = &out->data[out->size++];
            t->has_space = space_pending;
            t->file_id = file_id;
            t->line = line;
            t->hide = NULL;
            if (i + 1 < sz && s[i + 1] == '#') {
                t->kind = PP_HASHHASH;
                t->text.str = (char *)(s + i);
                t->text.size = 2;
                i += 2;
            } else {
                t->kind = PP_HASH;
                t->text.str = (char *)(s + i);
                t->text.size = 1;
                i++;
            }
            space_pending = 0;
            continue;
        }

        // Punctuation (1-, 2-, or 3-char) via single switch on first byte.
        size_t mlen = 1;
        if (i + 1 < sz) {
            char d = s[i + 1];
            switch (c) {
            case '<':
                if (d == '<') mlen = (i + 2 < sz && s[i + 2] == '=') ? 3 : 2;
                else if (d == '=') mlen = 2;
                break;
            case '>':
                if (d == '>') mlen = (i + 2 < sz && s[i + 2] == '=') ? 3 : 2;
                else if (d == '=') mlen = 2;
                break;
            case '.':
                if (d == '.' && i + 2 < sz && s[i + 2] == '.') mlen = 3;
                break;
            case '=': case '!': case '*': case '/': case '%': case '^':
                if (d == '=') mlen = 2;
                break;
            case '+':
                if (d == '+' || d == '=') mlen = 2;
                break;
            case '-':
                if (d == '-' || d == '=' || d == '>') mlen = 2;
                break;
            case '&':
                if (d == '&' || d == '=') mlen = 2;
                break;
            case '|':
                if (d == '|' || d == '=') mlen = 2;
                break;
            default:
                break;
            }
        }
        if (out->size == out->max) pp_vec_grow(arena, out);
        PPTok *t = &out->data[out->size++];
        t->kind = PP_PUNCT;
        t->has_space = space_pending;
        t->file_id = file_id;
        t->line = line;
        t->text.str = (char *)(s + i);
        t->text.size = mlen;
        t->hide = NULL;
        i += mlen;
        space_pending = 0;
    }
    // Final newline guarantees directives at EOF terminate.
    if (out->size + 2 > out->max) pp_vec_grow(arena, out);
    PPTok *eol = &out->data[out->size++];
    eol->kind = PP_NEWLINE;
    eol->has_space = 0;
    eol->file_id = file_id;
    eol->line = line;
    eol->text.str = NULL;
    eol->text.size = 0;
    eol->hide = NULL;
    PPTok *end = &out->data[out->size++];
    end->kind = PP_END;
    end->has_space = 0;
    end->file_id = file_id;
    end->line = line;
    end->text.str = NULL;
    end->text.size = 0;
    end->hide = NULL;
}

// ----------------------------------------------------------------------
// Macro table operations
// ----------------------------------------------------------------------

// FNV-1a.
static uint32_t pp_hash(string s) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < s.size; i++) {
        h ^= (uint32_t)(unsigned char)s.str[i];
        h *= 16777619u;
    }
    return h;
}

static bool pp_str_eq(string a, string b) {
    return a.size == b.size && (a.str == b.str || memcmp(a.str, b.str, a.size) == 0);
}

static inline uint32_t pp_macro_filter_idx(string s) {
    uint32_t c0 = (uint32_t)(unsigned char)s.str[0];
    uint32_t cl = (uint32_t)(unsigned char)s.str[s.size - 1];
    return (c0 * 131u + cl * 31u + (uint32_t)s.size * 7u) & 4095u;
}

static inline bool pp_macro_maybe(const PP *pp, string s) {
    if (s.size == 0) return false;
    uint32_t fi = pp_macro_filter_idx(s);
    return ((pp->macro_filter[fi >> 6] >> (fi & 63u)) & 1u) != 0;
}

static inline void pp_macro_filter_add(PP *pp, string s) {
    if (s.size == 0) return;
    uint32_t fi = pp_macro_filter_idx(s);
    pp->macro_filter[fi >> 6] |= (1ull << (fi & 63u));
}

// The slot holding `name`, or NULL.
static MacroSlot *macro_find_slot(PP *pp, string name) {
    if (pp->macro_cap == 0 || !pp_macro_maybe(pp, name)) return NULL;
    uint32_t h = pp_hash(name);
    size_t mask = pp->macro_cap - 1;
    size_t k = (size_t)h & mask;
    while (pp->mtab[k].name.str) {
        MacroSlot *e = &pp->mtab[k];
        if (e->hash == h && pp_str_eq(e->name, name)) return e;
        k = (k + 1) & mask;
    }
    return NULL;
}

static void macro_table_grow(PP *pp) {
    size_t old_cap = pp->macro_cap;
    MacroSlot *old = pp->mtab;
    size_t cap = old_cap ? old_cap * 2 : 64;
    MacroSlot *tab = arena_new_array(pp->arena, MacroSlot, cap);
    memset(tab, 0, cap * sizeof(MacroSlot));
    for (size_t i = 0; i < old_cap; i++) {
        if (!old[i].name.str) continue;
        size_t k = (size_t)old[i].hash & (cap - 1);
        while (tab[k].name.str) k = (k + 1) & (cap - 1);
        tab[k] = old[i];
    }
    pp->mtab = tab;
    pp->macro_cap = cap;
}

// The slot holding `name`, inserted (undefined) if missing.
static MacroSlot *macro_get_slot(PP *pp, string name) {
    MacroSlot *e = macro_find_slot(pp, name);
    if (e) return e;
    if ((pp->macro_used + 1) * 2 > pp->macro_cap) macro_table_grow(pp);
    uint32_t h = pp_hash(name);
    size_t mask = pp->macro_cap - 1;
    size_t k = (size_t)h & mask;
    while (pp->mtab[k].name.str) k = (k + 1) & mask;
    e = &pp->mtab[k];
    e->name = s_dup(pp->arena, name);
    e->hash = h;
    e->special = PP_SPECIAL_NONE;
    e->macro = NULL;
    pp->macro_used++;
    pp_macro_filter_add(pp, e->name);
    return e;
}

static Macro *macro_lookup(PP *pp, string name) {
    MacroSlot *e = macro_find_slot(pp, name);
    return e ? e->macro : NULL;
}

static void macro_undef(PP *pp, string name) {
    MacroSlot *e = macro_find_slot(pp, name);
    if (e) e->macro = NULL;
}

static Macro *macro_define(PP *pp, string name) {
    MacroSlot *e = macro_get_slot(pp, name);
    Macro *m = arena_new(pp->arena, Macro);
    *m = (Macro){0};
    m->name = e->name;
    e->macro = m;
    return m;
}

// ----------------------------------------------------------------------
// Token-stream helpers
// ----------------------------------------------------------------------

static bool is_eol_or_end(PPTok *t) {
    return t->kind == PP_NEWLINE || t->kind == PP_END;
}

static bool is_punct1(PPTok *t, char c) {
    return t->kind == PP_PUNCT && t->text.size == 1 && t->text.str[0] == c;
}

// (No unused-helper section here — utilities live inline.)

// ----------------------------------------------------------------------
// #line emission
// ----------------------------------------------------------------------

static void emit_line_directive_if_needed(PP *pp, uint16_t file_id, int line) {
    if (pp->tok_out) return;
    if (pp->cur_loc_valid &&
        pp->cur_file_id == file_id && pp->cur_line == line) return;
    string file = pp->files[file_id];
    out_grow(pp, file.size * 2 + 32);
    char *d = pp->out + pp->out_size;
    if (pp->out_size > 0 && d[-1] != '\n') *d++ = '\n';
    memcpy(d, "#line ", 6); d += 6;
    char tmp[16];
    int n = 0;
    int v = line;
    if (v <= 0) tmp[n++] = '0';
    else { while (v) { tmp[n++] = (char)('0' + (v % 10)); v /= 10; } }
    for (int i = n - 1; i >= 0; i--) *d++ = tmp[i];
    *d++ = ' ';
    *d++ = '"';
    for (size_t i = 0; i < file.size; i++) {
        char c = file.str[i];
        if (c == '\\' || c == '"') *d++ = '\\';
        *d++ = c;
    }
    *d++ = '"';
    *d++ = '\n';
    pp->out_size = (size_t)(d - pp->out);
    pp->cur_file_id = file_id;
    pp->cur_line = line;
    pp->cur_loc_valid = true;
}

static TcTokKind pp_punct_kind(string s) {
    if (s.size == 1) {
        unsigned char c = (unsigned char)s.str[0];
        if (c < 128 && g_pp_punct1[c]) return (TcTokKind)g_pp_punct1[c];
        return TC_TK_EOF;
    }
    if (s.size == 2) {
        char c = s.str[0], d = s.str[1];
        switch (c) {
            case '<': return d == '=' ? TC_TK_LE : d == '<' ? TC_TK_SHL : TC_TK_EOF;
            case '>': return d == '=' ? TC_TK_GE : d == '>' ? TC_TK_SHR : TC_TK_EOF;
            case '=': return d == '=' ? TC_TK_EQEQ : TC_TK_EOF;
            case '!': return d == '=' ? TC_TK_NE : TC_TK_EOF;
            case '&': return d == '&' ? TC_TK_AMPAMP : d == '=' ? TC_TK_AMPEQ : TC_TK_EOF;
            case '|': return d == '|' ? TC_TK_PIPEPIPE : d == '=' ? TC_TK_PIPEEQ : TC_TK_EOF;
            case '-': return d == '>' ? TC_TK_ARROW : d == '=' ? TC_TK_MINUSEQ : d == '-' ? TC_TK_MINUSMINUS : TC_TK_EOF;
            case '+': return d == '+' ? TC_TK_PLUSPLUS : d == '=' ? TC_TK_PLUSEQ : TC_TK_EOF;
            case '*': return d == '=' ? TC_TK_STAREQ : TC_TK_EOF;
            case '/': return d == '=' ? TC_TK_SLASHEQ : TC_TK_EOF;
            case '%': return d == '=' ? TC_TK_PERCENTEQ : TC_TK_EOF;
            case '^': return d == '=' ? TC_TK_CARETEQ : TC_TK_EOF;
            default:  return TC_TK_EOF;
        }
    }
    if (s.size == 3) {
        if (s.str[0] == '.' && s.str[1] == '.' && s.str[2] == '.') return TC_TK_ELLIPSIS;
        if (s.str[0] == '<' && s.str[1] == '<' && s.str[2] == '=') return TC_TK_SHLEQ;
        if (s.str[0] == '>' && s.str[1] == '>' && s.str[2] == '=') return TC_TK_SHREQ;
    }
    return TC_TK_EOF;
}

static void pp_emit_tctok(PP *pp, const PPTok *t) {
    VecTcTok *v = pp->tok_out;
    if (v->size == v->max) {
        size_t nmax = v->max ? v->max * 2 : 256;
        TcTok *nd = arena_new_array(pp->arena, TcTok, nmax);
        if (v->size) memcpy(nd, v->data, v->size * sizeof(TcTok));
        v->data = nd;
        v->max = nmax;
    }
    TcTok *dst = &v->data[v->size];
    switch (t->kind) {
    case PP_IDENT:
        dst->kind = tinyc_keyword_or_ident(t->text);
        dst->line = t->line;
        dst->text = t->text;
        v->size++;
        return;
    case PP_PUNCT: {
        TcTokKind k = pp_punct_kind(t->text);
        if (k == TC_TK_EOF) {
            println(str_lit("tinyc lex error at line {}: unexpected '{}'"),
                    (int64_t)t->line, t->text);
            return;
        }
        dst->kind = k;
        dst->line = t->line;
        v->size++;
        return;
    }
    case PP_NUMBER: {
        // Fast path for common small decimal integers (0, 1, 2, ..., 999999999).
        const char *s = t->text.str;
        size_t n = t->text.size;
        if (n == 1 && s[0] >= '0' && s[0] <= '9') {
            dst->kind = TC_TK_INT_LIT;
            dst->line = t->line;
            dst->is_i64 = false;
            dst->is_long_long = false;
            dst->is_unsigned = false;
            dst->is_f64 = false;
            dst->int_value = (int64_t)(s[0] - '0');
            v->size++;
            return;
        }
        if (n >= 2 && n <= 9 && s[0] >= '1' && s[0] <= '9') {
            int64_t val = s[0] - '0';
            bool all_dig = true;
            for (size_t k = 1; k < n; k++) {
                unsigned d = (unsigned)(unsigned char)s[k] - '0';
                if (d > 9u) { all_dig = false; break; }
                val = val * 10 + (int64_t)d;
            }
            if (all_dig) {
                dst->kind = TC_TK_INT_LIT;
                dst->line = t->line;
                dst->is_i64 = false;
                dst->is_long_long = false;
                dst->is_unsigned = false;
                dst->is_f64 = false;
                dst->int_value = val;
                v->size++;
                return;
            }
        }
        *dst = tinyc_lex_number(t->text, t->line);
        v->size++;
        return;
    }
    case PP_STRING:
        *dst = tinyc_lex_string(pp->arena, t->text, t->line);
        v->size++;
        return;
    case PP_CHAR:
        *dst = tinyc_lex_char(t->text, t->line);
        v->size++;
        return;
    default:
        return;
    }
}

// Emit a token to the output stream (after macro expansion). Inserts a
// `#line` directive when the source location changes; otherwise pads with
// newlines so token line numbers stay roughly faithful.
static bool g_need_space_sep = false;
static void emit_pp_token(PP *pp, PPTok *t) {
    if (t->kind == PP_PLACEMARKER || t->kind == PP_NEWLINE || t->kind == PP_END) return;
    if (pp->tok_out) { pp_emit_tctok(pp, t); return; }
    if (!pp->cur_loc_valid || pp->cur_file_id != t->file_id) {
        emit_line_directive_if_needed(pp, t->file_id, t->line);
        g_need_space_sep = false;
    } else if (pp->cur_line != t->line) {
        int diff = t->line - pp->cur_line;
        if (diff > 0 && diff <= 8) {
            out_grow(pp, (size_t)diff + t->text.size + 1);
            char *d = pp->out + pp->out_size;
            for (int k = 0; k < diff; k++) *d++ = '\n';
            size_t ts = t->text.size;
            for (size_t k = 0; k < ts; k++) d[k] = t->text.str[k];
            pp->out_size = (size_t)(d - pp->out) + ts;
            pp->cur_line = t->line;
            g_need_space_sep = true;
            return;
        } else {
            emit_line_directive_if_needed(pp, t->file_id, t->line);
            g_need_space_sep = false;
        }
    }
    size_t ts = t->text.size;
    out_grow(pp, ts + 1);
    char *d = pp->out + pp->out_size;
    if (g_need_space_sep) *d++ = ' ';
    for (size_t k = 0; k < ts; k++) d[k] = t->text.str[k];
    pp->out_size = (size_t)(d - pp->out) + ts;
    g_need_space_sep = true;
}

// ----------------------------------------------------------------------
// Macro expansion
// ----------------------------------------------------------------------

static bool hide_contains(PPHide *h, string name) {
    for (; h; h = h->next) if (str_eq(h->name, name)) return true;
    return false;
}
static PPHide *hide_add(Arena *arena, PPHide *h, string name) {
    PPHide *n = arena_new(arena, PPHide);
    n->name = name;
    n->next = h;
    return n;
}

// Stringize a sequence of tokens into a string-literal PP token.
static PPTok stringize(PP *pp, PPTok *toks, size_t n) {
    // Build inner text: tokens separated by single spaces if has_space.
    size_t cap = 16;
    char *buf = arena_new_array(pp->arena, char, cap);
    size_t sz = 0;
    bool first = true;
    for (size_t i = 0; i < n; i++) {
        if (toks[i].kind == PP_PLACEMARKER) continue;
        if (!first && toks[i].has_space) {
            if (sz + 1 >= cap) { cap *= 2; char *nb = arena_new_array(pp->arena, char, cap); memcpy(nb, buf, sz); buf = nb; }
            buf[sz++] = ' ';
        }
        first = false;
        for (size_t k = 0; k < toks[i].text.size; k++) {
            char c = toks[i].text.str[k];
            size_t need = (c == '\\' || c == '"') ? 2 : 1;
            // Inside string/char literals we must double-escape backslashes
            // — but the raw text already contains them; we just escape
            // any backslash and quote.
            if (sz + need >= cap) {
                while (sz + need >= cap) cap *= 2;
                char *nb = arena_new_array(pp->arena, char, cap);
                memcpy(nb, buf, sz);
                buf = nb;
            }
            if (c == '\\' || c == '"') buf[sz++] = '\\';
            buf[sz++] = c;
        }
    }
    // Wrap in quotes.
    char *out = arena_new_array(pp->arena, char, sz + 3);
    out[0] = '"';
    memcpy(out + 1, buf, sz);
    out[1 + sz] = '"';
    out[2 + sz] = '\0';
    PPTok r = {0};
    r.kind = PP_STRING;
    r.text = (string){ out, sz + 2 };
    return r;
}

// Glue two tokens by lexing the concatenation of their raw text. Returns
// a single token (most of the time). If the joined text re-tokenizes to
// more than one PP token we just concatenate (best effort).
static PPTok paste_tokens(PP *pp, PPTok a, PPTok b) {
    if (a.kind == PP_PLACEMARKER) return b;
    if (b.kind == PP_PLACEMARKER) return a;
    string joined = s_concat3(pp->arena, a.text, b.text, (string){"",0});
    VecPPTok tmp; VecPPTok_reserve(pp->arena, &tmp, 4);
    pp_tokenize(pp->arena, joined, a.file_id, &tmp);
    // Find the first non-newline token.
    PPTok r = {0};
    r.kind = PP_IDENT; // fallback
    r.text = joined;
    r.line = a.line;
    r.file_id = a.file_id;
    for (size_t i = 0; i < tmp.size; i++) {
        if (tmp.data[i].kind == PP_END) break;
        if (tmp.data[i].kind == PP_NEWLINE) continue;
        r = tmp.data[i];
        r.line = a.line;
        r.file_id = a.file_id;
        break;
    }
    return r;
}

// Forward decl.
static void expand_tokens(PP *pp, PPTok *toks, size_t n, VecPPTok *out);

static void arg_buf_push(PP *pp, size_t *n, PPTok *t) {
    if (*n == pp->arg_buf_cap) {
        size_t cap = pp->arg_buf_cap ? pp->arg_buf_cap * 2 : 256;
        PPTok *nb = arena_new_array(pp->arena, PPTok, cap);
        if (*n) memcpy(nb, pp->arg_buf, *n * sizeof(PPTok));
        pp->arg_buf = nb;
        pp->arg_buf_cap = cap;
    }
    pp->arg_buf[*n] = *t;
    *n = *n + 1;
}

static bool arg_needs_expand(PP *pp, const VecPPTok *arg) {
    for (size_t i = 0; i < arg->size; i++) {
        if (arg->data[i].kind == PP_IDENT) {
            MacroSlot *slot = macro_find_slot(pp, arg->data[i].text);
            if (slot && (slot->special != PP_SPECIAL_NONE || slot->macro != NULL))
                return true;
        }
    }
    return false;
}

// Substitute parameters in macro body, handling # and ##. Returns the
// substituted token list (still un-rescanned) in `out`.
static void substitute(PP *pp, Macro *m, VecPPTok *args, VecPPTok *out) {
    PPTok *body = m->body;
    size_t nb = m->n_body;
    // Lazily pre-expand arguments ONLY when actually used in a non-#/## slot,
    // and skip allocation/expansion altogether when the argument has no macros.
    size_t n_args = m->n_params + (m->is_variadic ? 1 : 0);
    VecPPTok exp_stk[16];
    bool done_stk[16];
    VecPPTok *expanded = (n_args <= 16) ? exp_stk : arena_new_array(pp->arena, VecPPTok, n_args);
    bool *exp_done = (n_args <= 16) ? done_stk : arena_new_array(pp->arena, bool, n_args);
    for (size_t k = 0; k < n_args; k++) exp_done[k] = false;

    // Helper: find param index for an identifier, or -1. Variadic pseudo-
    // parameter is `__VA_ARGS__` (returns m->n_params).
    int va_idx = m->is_variadic ? (int)m->n_params : -1;

    for (size_t i = 0; i < nb; i++) {
        PPTok t = body[i];

        // # PARAM: stringize
        if (t.kind == PP_HASH && i + 1 < nb && body[i + 1].kind == PP_IDENT) {
            int p = -1;
            for (size_t k = 0; k < m->n_params; k++)
                if (str_eq(m->params[k], body[i + 1].text)) { p = (int)k; break; }
            if (p < 0 && va_idx >= 0 && str_eq(body[i + 1].text, s_lit("__VA_ARGS__"))) p = va_idx;
            if (p >= 0) {
                PPTok s = stringize(pp, args[p].data, args[p].size);
                s.line = t.line; s.file_id = t.file_id; s.has_space = t.has_space;
                VecPPTok_push_back(pp->arena, out, s);
                i++;
                continue;
            }
        }

        // __VA_OPT__(content): emits content if VA_ARGS non-empty.
        if (t.kind == PP_IDENT && str_eq(t.text, s_lit("__VA_OPT__"))
            && i + 1 < nb && body[i + 1].kind == PP_PUNCT
            && str_eq(body[i + 1].text, s_lit("("))) {
            // Find matching ).
            int depth = 1;
            size_t j = i + 2;
            size_t start = j;
            while (j < nb && depth > 0) {
                if (body[j].kind == PP_PUNCT && str_eq(body[j].text, s_lit("("))) depth++;
                else if (body[j].kind == PP_PUNCT && str_eq(body[j].text, s_lit(")"))) {
                    depth--;
                    if (depth == 0) break;
                }
                j++;
            }
            size_t end = j;
            bool va_nonempty = (va_idx >= 0 && args[va_idx].size > 0);
            if (va_nonempty) {
                // Substitute content recursively.
                VecPPTok inner; VecPPTok_reserve(pp->arena, &inner, 4);
                for (size_t k = start; k < end; k++) {
                    VecPPTok_push_back(pp->arena, &inner, body[k]);
                }
                // Build a shadow Macro with inner as body to re-run substitute.
                Macro shadow = *m;
                shadow.body = inner.data;
                shadow.n_body = inner.size;
                substitute(pp, &shadow, args, out);
            }
            i = end; // skip closing ')'
            continue;
        }

        // ## handling. We do this lazily: when we see `A ## B` we replace
        // the previously-pushed token (or empty placemarker) with the
        // paste of (A_unexp, B_unexp).
        if (t.kind == PP_HASHHASH && out->size > 0 && i + 1 < nb) {
            PPTok left = out->data[out->size - 1];
            // Compute right operand: if it's a parameter, use unexpanded
            // arg list; else the literal token.
            PPTok right_default = body[i + 1];
            i++;
            if (right_default.kind == PP_IDENT) {
                int p = -1;
                for (size_t k = 0; k < m->n_params; k++)
                    if (str_eq(m->params[k], right_default.text)) { p = (int)k; break; }
                bool is_va = (p < 0 && va_idx >= 0 &&
                              str_eq(right_default.text, s_lit("__VA_ARGS__")));
                if (is_va) p = va_idx;
                // GNU extension: `, ## __VA_ARGS__` ("comma swallowing").
                //   * If __VA_ARGS__ is empty, drop the preceding comma.
                //   * If __VA_ARGS__ is non-empty, leave the comma intact
                //     and emit the variadic arguments as-is — do NOT paste
                //     the comma to the first variadic token.
                // This matches the behaviour clang/gcc give to widely-used
                // logging macros of the form `f(fixed, ## __VA_ARGS__)`.
                if (is_va && left.kind == PP_PUNCT && str_eq(left.text, s_lit(","))) {
                    if (args[p].size == 0) {
                        // Drop the comma already pushed into `out`.
                        out->size--;
                    } else {
                        for (size_t k = 0; k < args[p].size; k++) {
                            VecPPTok_push_back(pp->arena, out, args[p].data[k]);
                        }
                    }
                    continue;
                }
                if (p >= 0) {
                    if (args[p].size == 0) {
                        // Right is empty: pasted result is just left.
                        // (Already in `out`.) No-op.
                        continue;
                    }
                    // Paste left with args[p][0]; append rest unmodified.
                    PPTok glued = paste_tokens(pp, left, args[p].data[0]);
                    out->data[out->size - 1] = glued;
                    for (size_t k = 1; k < args[p].size; k++) {
                        VecPPTok_push_back(pp->arena, out, args[p].data[k]);
                    }
                    continue;
                }
            }
            PPTok glued = paste_tokens(pp, left, right_default);
            out->data[out->size - 1] = glued;
            continue;
        }

        // Plain parameter substitution (use pre-expanded form).
        if (t.kind == PP_IDENT) {
            int p = -1;
            for (size_t k = 0; k < m->n_params; k++)
                if (str_eq(m->params[k], t.text)) { p = (int)k; break; }
            if (p < 0 && va_idx >= 0 && str_eq(t.text, s_lit("__VA_ARGS__"))) p = va_idx;
            if (p >= 0) {
                // Look ahead for ## — if the next non-this body token is
                // ##, do NOT pre-expand (per C); use unexpanded form.
                bool followed_by_paste = (i + 1 < nb && body[i + 1].kind == PP_HASHHASH);
                VecPPTok *src;
                if (followed_by_paste) {
                    src = &args[p];
                } else {
                    if (!exp_done[p]) {
                        exp_done[p] = true;
                        if (arg_needs_expand(pp, &args[p])) {
                            VecPPTok_reserve(pp->arena, &expanded[p], args[p].size + 4);
                            expand_tokens(pp, args[p].data, args[p].size, &expanded[p]);
                        } else {
                            expanded[p] = args[p];
                        }
                    }
                    src = &expanded[p];
                }
                if (src->size == 0 && followed_by_paste) {
                    PPTok pm = {0};
                    pm.kind = PP_PLACEMARKER;
                    pm.line = t.line; pm.file_id = t.file_id;
                    VecPPTok_push_back(pp->arena, out, pm);
                } else {
                    for (size_t k = 0; k < src->size; k++) {
                        PPTok x = src->data[k];
                        if (k == 0) x.has_space = t.has_space;
                        VecPPTok_push_back(pp->arena, out, x);
                    }
                }
                continue;
            }
        }

        VecPPTok_push_back(pp->arena, out, t);
    }
}

typedef struct {
    const PPTok *toks;
    size_t pos;
    size_t end;
    size_t saved_exp_top;
    bool   in_exp_buf;
} PPSlice;

static PPTok *pp_exp_alloc(PP *pp, size_t n) {
    if (pp->exp_top + n > pp->exp_cap) {
        size_t ncap = pp->exp_cap ? pp->exp_cap * 2 : 512;
        while (ncap < pp->exp_top + n) ncap *= 2;
        PPTok *nb = arena_new_array(pp->arena, PPTok, ncap);
        if (pp->exp_top) memcpy(nb, pp->exp_buf, pp->exp_top * sizeof(PPTok));
        pp->exp_buf = nb;
        pp->exp_cap = ncap;
    }
    PPTok *p = pp->exp_buf + pp->exp_top;
    pp->exp_top += n;
    return p;
}

static bool pp_slice_peek_lparen(PP *pp, const PPSlice *stk, size_t sp) {
    for (size_t s = sp; s-- > 0; ) {
        const PPTok *base = stk[s].in_exp_buf ? pp->exp_buf : stk[s].toks;
        for (size_t p = stk[s].pos; p < stk[s].end; p++) {
            uint8_t k = base[p].kind;
            if (k == PP_NEWLINE) continue;
            if (k == PP_END) return false;
            return k == PP_PUNCT && base[p].text.size == 1 &&
                   base[p].text.str[0] == '(';
        }
    }
    return false;
}

static bool pp_slice_next_tok(PP *pp, PPSlice *stk, size_t *sp, PPTok *out) {
    while (*sp > 0) {
        PPSlice *top = &stk[*sp - 1];
        if (top->pos < top->end) {
            const PPTok *base = top->in_exp_buf ? pp->exp_buf : top->toks;
            *out = base[top->pos++];
            return true;
        }
        if (top->in_exp_buf) pp->exp_top = top->saved_exp_top;
        (*sp)--;
    }
    return false;
}

static bool collect_args_slices(PP *pp, PPSlice *stk, size_t *sp,
                                Macro *m, VecPPTok *args, size_t *saved_arg_exp) {
    PPTok t;
    while (pp_slice_next_tok(pp, stk, sp, &t)) {
        if (t.kind == PP_NEWLINE) continue;
        break;
    }
    int depth = 0;
    size_t arg_idx = 0;
    size_t expected = m->n_params + (m->is_variadic ? 1 : 0);
    size_t arg_end[72];
    size_t nbuf = 0;
    while (pp_slice_next_tok(pp, stk, sp, &t)) {
        if (t.kind == PP_END) {
            pp_error(pp, pp->files[t.file_id], t.line, s_lit("unterminated macro arguments"));
            return false;
        }
        if (t.kind == PP_NEWLINE) continue;
        if (is_punct1(&t, '(')) depth++;
        else if (is_punct1(&t, ')')) {
            if (depth == 0) break;
            depth--;
        } else if (is_punct1(&t, ',') && depth == 0) {
            if (m->is_variadic && arg_idx >= m->n_params) {
                arg_buf_push(pp, &nbuf, &t);
                continue;
            }
            arg_end[arg_idx] = nbuf;
            arg_idx++;
            if (arg_idx >= expected) {
                pp_error(pp, pp->files[t.file_id], t.line, s_lit("too many arguments to macro"));
                return false;
            }
            continue;
        }
        arg_buf_push(pp, &nbuf, &t);
    }
    arg_end[arg_idx] = nbuf;
    *saved_arg_exp = pp->exp_top;
    PPTok *block = pp_exp_alloc(pp, nbuf ? nbuf : 1);
    if (nbuf) memcpy(block, pp->arg_buf, nbuf * sizeof(PPTok));
    size_t start = 0;
    for (size_t k = 0; k < expected; k++) {
        size_t end = k <= arg_idx ? arg_end[k] : nbuf;
        args[k].data = block + start;
        args[k].size = end - start;
        args[k].max = end - start;
        start = end;
    }
    return true;
}

// Append a fully expanded token to `out`, or emit it straight to the output
// text / TcTok stream when `out` is NULL.
static inline void expand_put(PP *pp, VecPPTok *out, const PPTok *t) {
    if (out) VecPPTok_push_back(pp->arena, out, *t);
    else if (pp->tok_out) pp_emit_tctok(pp, t);
    else emit_pp_token(pp, (PPTok *)t);
}

// Expand tokens in `toks` and append the result to `out` (NULL: emit them).
// When `stop_at_directive` is true (top-level process_tokens), stops at the
// first PP_NEWLINE followed by PP_HASH and returns the index in `toks` where
// processing stopped, eliminating the separate lookahead scan in process_tokens.
static size_t expand_tokens_ex(PP *pp, PPTok *toks, size_t n, VecPPTok *out,
                               bool stop_at_directive) {
    PPSlice stk_inline[64];
    PPSlice *stk = stk_inline;
    size_t stk_cap = 64;
    size_t sp = 1;
    stk[0].toks = toks;
    stk[0].pos = 0;
    stk[0].end = n;
    stk[0].saved_exp_top = pp->exp_top;
    stk[0].in_exp_buf = false;

    while (sp > 0) {
        PPSlice *top = &stk[sp - 1];
        if (top->pos >= top->end) {
            if (top->in_exp_buf) pp->exp_top = top->saved_exp_top;
            sp--;
            continue;
        }
        const PPTok *base = top->in_exp_buf ? pp->exp_buf : top->toks;
        const PPTok *tp = &base[top->pos++];
        uint8_t tkind = tp->kind;
        if (tkind == PP_END) break;
        if (tkind == PP_NEWLINE) {
            if (stop_at_directive && sp == 1 && top->pos < top->end &&
                base[top->pos].kind == PP_HASH) {
                top->pos--; // leave PP_NEWLINE for process_tokens
                break;
            }
            continue;
        }
        if (tkind == PP_IDENT) {
            MacroSlot *slot = macro_find_slot(pp, tp->text);
            if (slot) {
                PPTok t = *tp;
                if (slot->special == PP_SPECIAL_FILE) {
                    PPTok s = {0};
                    string tf = pp->files[t.file_id];
                    size_t fsz = tf.size;
                    char *buf = arena_new_array(pp->arena, char, 2 * fsz + 3);
                    size_t k = 0;
                    buf[k++] = '"';
                    for (size_t j = 0; j < fsz; j++) {
                        char c = tf.str[j];
                        if (c == '\\' || c == '"') buf[k++] = '\\';
                        buf[k++] = c;
                    }
                    buf[k++] = '"';
                    buf[k] = '\0';
                    s.kind = PP_STRING;
                    s.text = (string){buf, k};
                    s.line = t.line; s.file_id = t.file_id; s.has_space = t.has_space;
                    expand_put(pp, out, &s);
                    continue;
                }
                if (slot->special == PP_SPECIAL_LINE) {
                    PPTok s = {0};
                    s.kind = PP_NUMBER;
                    s.text = i64_to_str(pp->arena, (int64_t)t.line);
                    s.line = t.line; s.file_id = t.file_id; s.has_space = t.has_space;
                    expand_put(pp, out, &s);
                    continue;
                }
                Macro *m = slot->macro;
                if (m && !m->expanding && !hide_contains(t.hide, t.text)) {
                    if (!m->is_func) {
                        // Fast path: 1-token non-ident object-like macro (e.g.
                        // numeric constant, string literal, etc.) cannot expand
                        // further — emit directly with zero allocation.
                        if (m->n_body == 1 && m->body[0].kind != PP_IDENT &&
                            m->body[0].kind != PP_HASHHASH) {
                            PPTok one = m->body[0];
                            one.line = t.line;
                            one.file_id = t.file_id;
                            one.has_space = t.has_space;
                            expand_put(pp, out, &one);
                            continue;
                        }
                        if (m->n_body == 0) continue;
                        size_t saved_top = pp->exp_top;
                        PPTok inner_stk[32];
                        VecPPTok inner;
                        if (m->n_body <= 32) {
                            inner.data = inner_stk; inner.size = 0; inner.max = 32;
                        } else {
                            VecPPTok_reserve(pp->arena, &inner, m->n_body + 2);
                        }
                        m->expanding = true;
                        substitute(pp, m, NULL, &inner);
                        m->expanding = false;
                        if (inner.size > 0) {
                            PPHide *shared_hide = hide_add(pp->arena, NULL, m->name);
                            size_t start_idx = pp->exp_top;
                            PPTok *dst = pp_exp_alloc(pp, inner.size);
                            for (size_t k = 0; k < inner.size; k++) {
                                dst[k] = inner.data[k];
                                dst[k].line = t.line;
                                dst[k].file_id = t.file_id;
                                dst[k].hide = dst[k].hide
                                    ? hide_add(pp->arena, dst[k].hide, m->name)
                                    : shared_hide;
                            }
                            dst[0].has_space = t.has_space;
                            if (sp == stk_cap) {
                                size_t ncap = stk_cap * 2;
                                PPSlice *ns = arena_new_array(pp->arena, PPSlice, ncap);
                                memcpy(ns, stk, sp * sizeof(PPSlice));
                                stk = ns; stk_cap = ncap;
                            }
                            stk[sp].toks = NULL;
                            stk[sp].pos = start_idx;
                            stk[sp].end = start_idx + inner.size;
                            stk[sp].saved_exp_top = saved_top;
                            stk[sp].in_exp_buf = true;
                            sp++;
                        }
                        continue;
                    } else if (pp_slice_peek_lparen(pp, stk, sp)) {
                        size_t expected = m->n_params + (m->is_variadic ? 1 : 0);
                        VecPPTok args_stk[16];
                        VecPPTok *args = (expected <= 16) ? args_stk
                            : arena_new_array(pp->arena, VecPPTok, expected);
                        size_t saved_arg_exp = pp->exp_top;
                        if (!collect_args_slices(pp, stk, &sp, m, args, &saved_arg_exp)) {
                            pp->exp_top = saved_arg_exp;
                            continue;
                        }
                        size_t est = m->n_body + 4;
                        for (size_t k = 0; k < expected; k++) est += args[k].size;
                        PPTok inner_stk[64];
                        VecPPTok inner;
                        if (est <= 64) {
                            inner.data = inner_stk; inner.size = 0; inner.max = 64;
                        } else {
                            VecPPTok_reserve(pp->arena, &inner, est);
                        }
                        m->expanding = true;
                        substitute(pp, m, args, &inner);
                        m->expanding = false;
                        // Rewind argument token storage before pushing inner!
                        pp->exp_top = saved_arg_exp;
                        if (inner.size > 0) {
                            PPHide *shared_hide = hide_add(pp->arena, NULL, m->name);
                            size_t start_idx = pp->exp_top;
                            PPTok *dst = pp_exp_alloc(pp, inner.size);
                            for (size_t k = 0; k < inner.size; k++) {
                                dst[k] = inner.data[k];
                                dst[k].line = t.line;
                                dst[k].file_id = t.file_id;
                                dst[k].hide = dst[k].hide
                                    ? hide_add(pp->arena, dst[k].hide, m->name)
                                    : shared_hide;
                            }
                            dst[0].has_space = t.has_space;
                            if (sp == stk_cap) {
                                size_t ncap = stk_cap * 2;
                                PPSlice *ns = arena_new_array(pp->arena, PPSlice, ncap);
                                memcpy(ns, stk, sp * sizeof(PPSlice));
                                stk = ns; stk_cap = ncap;
                            }
                            stk[sp].toks = NULL;
                            stk[sp].pos = start_idx;
                            stk[sp].end = start_idx + inner.size;
                            stk[sp].saved_exp_top = saved_arg_exp;
                            stk[sp].in_exp_buf = true;
                            sp++;
                        }
                        continue;
                    }
                }
            }
        }
        expand_put(pp, out, tp);
    }
    // Rewind any remaining exp_buf slices if we broke out early on PP_END / directive.
    size_t consumed = stk[0].pos;
    while (sp > 0) {
        sp--;
        if (stk[sp].in_exp_buf) pp->exp_top = stk[sp].saved_exp_top;
    }
    return consumed;
}

static void expand_tokens(PP *pp, PPTok *toks, size_t n, VecPPTok *out) {
    (void)expand_tokens_ex(pp, toks, n, out, false);
}

// ----------------------------------------------------------------------
// Constant expression evaluator for #if / #elif
// ----------------------------------------------------------------------

typedef struct {
    PP *pp;
    PPTok *t;
    size_t n;
    size_t i;
    string file;
    int line;
    bool err;
} CEX;

static int64_t cex_expr(CEX *c);
static int64_t cex_ternary(CEX *c);

static PPTok *cex_peek(CEX *c) {
    static PPTok eof = { PP_END, 0, 0, 0, {NULL,0}, NULL };
    return c->i < c->n ? &c->t[c->i] : &eof;
}
static bool cex_match_punct(CEX *c, const char *s) {
    PPTok *t = cex_peek(c);
    if (t->kind == PP_PUNCT && str_eq(t->text, s_lit(s))) { c->i++; return true; }
    return false;
}
static bool cex_match_ident(CEX *c, const char *s) {
    PPTok *t = cex_peek(c);
    if (t->kind == PP_IDENT && str_eq(t->text, s_lit(s))) { c->i++; return true; }
    return false;
}

static int64_t parse_pp_number(string s) {
    // Parse a C integer literal: 0x..., 0..., or decimal. Ignore U/L/LL.
    if (s.size == 0) return 0;
    int64_t v = 0;
    size_t i = 0;
    int base = 10;
    if (s.size >= 2 && s.str[0] == '0' && (s.str[1] == 'x' || s.str[1] == 'X')) {
        base = 16; i = 2;
    } else if (s.size >= 1 && s.str[0] == '0' && s.size > 1) {
        base = 8; i = 1;
    }
    for (; i < s.size; i++) {
        char ch = s.str[i];
        int d;
        if (ch >= '0' && ch <= '9') d = ch - '0';
        else if (ch >= 'a' && ch <= 'f') d = 10 + ch - 'a';
        else if (ch >= 'A' && ch <= 'F') d = 10 + ch - 'A';
        else break;
        if (d >= base) break;
        v = v * base + d;
    }
    return v;
}

static int64_t cex_primary(CEX *c) {
    PPTok *t = cex_peek(c);
    if (t->kind == PP_NUMBER) {
        c->i++;
        return parse_pp_number(t->text);
    }
    if (t->kind == PP_CHAR) {
        // Best-effort: 'x' -> x; '\n' -> 10; etc.
        if (t->text.size >= 3 && t->text.str[0] == '\'') {
            char ch = t->text.str[1];
            if (ch == '\\' && t->text.size >= 4) {
                char e = t->text.str[2];
                switch (e) {
                    case 'n': ch = '\n'; break;
                    case 't': ch = '\t'; break;
                    case 'r': ch = '\r'; break;
                    case '\\': ch = '\\'; break;
                    case '\'': ch = '\''; break;
                    case '"': ch = '"'; break;
                    case '0': ch = '\0'; break;
                    default: ch = e;
                }
            }
            c->i++;
            return (int64_t)(unsigned char)ch;
        }
        c->i++;
        return 0;
    }
    if (t->kind == PP_IDENT) {
        // Bare identifiers in #if are 0 (they would have been macro-
        // expanded earlier). `defined` is handled at a higher level
        // before expansion, so we shouldn't see it here — but tolerate.
        if (str_eq(t->text, s_lit("defined"))) {
            c->i++;
            bool paren = cex_match_punct(c, "(");
            PPTok *id = cex_peek(c);
            int64_t r = 0;
            if (id->kind == PP_IDENT) {
                r = macro_lookup(c->pp, id->text) ? 1 : 0;
                c->i++;
            }
            if (paren) cex_match_punct(c, ")");
            return r;
        }
        c->i++;
        return 0;
    }
    if (cex_match_punct(c, "(")) {
        int64_t v = cex_expr(c);
        if (!cex_match_punct(c, ")")) {
            pp_error(c->pp, c->file, c->line, s_lit("expected ')' in #if expr"));
            c->err = true;
        }
        return v;
    }
    if (cex_match_punct(c, "+")) return cex_primary(c);
    if (cex_match_punct(c, "-")) return -cex_primary(c);
    if (cex_match_punct(c, "!")) return !cex_primary(c);
    if (cex_match_punct(c, "~")) return ~cex_primary(c);
    pp_error(c->pp, c->file, c->line, s_lit("bad token in #if expression"));
    c->err = true;
    c->i++;
    return 0;
}

// Parser for binary ops by precedence level.
typedef struct { const char *op; int prec; } BinPrec;
// Higher prec = tighter binding. 1 = || ... 11 = * / %.
static int prec_of(PPTok *t, bool *out_is) {
    *out_is = (t->kind == PP_PUNCT);
    if (!*out_is) return -1;
    if (str_eq(t->text, s_lit("||"))) return 1;
    if (str_eq(t->text, s_lit("&&"))) return 2;
    if (str_eq(t->text, s_lit("|")))  return 3;
    if (str_eq(t->text, s_lit("^")))  return 4;
    if (str_eq(t->text, s_lit("&")))  return 5;
    if (str_eq(t->text, s_lit("==")) || str_eq(t->text, s_lit("!="))) return 6;
    if (str_eq(t->text, s_lit("<")) || str_eq(t->text, s_lit("<="))
     || str_eq(t->text, s_lit(">")) || str_eq(t->text, s_lit(">="))) return 7;
    if (str_eq(t->text, s_lit("<<")) || str_eq(t->text, s_lit(">>"))) return 8;
    if (str_eq(t->text, s_lit("+"))  || str_eq(t->text, s_lit("-")))  return 9;
    if (str_eq(t->text, s_lit("*"))  || str_eq(t->text, s_lit("/")) || str_eq(t->text, s_lit("%"))) return 11;
    return -1;
}

static int64_t cex_binop(CEX *c, int min_prec) {
    int64_t lhs = cex_primary(c);
    while (1) {
        bool is;
        int p = prec_of(cex_peek(c), &is);
        if (!is || p < min_prec) break;
        // Special-case ?: at lower precedence than ||
        PPTok op = *cex_peek(c);
        c->i++;
        int64_t rhs = cex_binop(c, p + 1);
        if (str_eq(op.text, s_lit("||"))) lhs = (lhs || rhs);
        else if (str_eq(op.text, s_lit("&&"))) lhs = (lhs && rhs);
        else if (str_eq(op.text, s_lit("|"))) lhs = lhs | rhs;
        else if (str_eq(op.text, s_lit("^"))) lhs = lhs ^ rhs;
        else if (str_eq(op.text, s_lit("&"))) lhs = lhs & rhs;
        else if (str_eq(op.text, s_lit("=="))) lhs = (lhs == rhs);
        else if (str_eq(op.text, s_lit("!="))) lhs = (lhs != rhs);
        else if (str_eq(op.text, s_lit("<")))  lhs = (lhs < rhs);
        else if (str_eq(op.text, s_lit("<="))) lhs = (lhs <= rhs);
        else if (str_eq(op.text, s_lit(">")))  lhs = (lhs > rhs);
        else if (str_eq(op.text, s_lit(">="))) lhs = (lhs >= rhs);
        else if (str_eq(op.text, s_lit("<<"))) lhs = lhs << rhs;
        else if (str_eq(op.text, s_lit(">>"))) lhs = lhs >> rhs;
        else if (str_eq(op.text, s_lit("+")))  lhs = lhs + rhs;
        else if (str_eq(op.text, s_lit("-")))  lhs = lhs - rhs;
        else if (str_eq(op.text, s_lit("*")))  lhs = lhs * rhs;
        else if (str_eq(op.text, s_lit("/")))  lhs = rhs ? lhs / rhs : 0;
        else if (str_eq(op.text, s_lit("%")))  lhs = rhs ? lhs % rhs : 0;
    }
    return lhs;
}

static int64_t cex_ternary(CEX *c) {
    int64_t cond = cex_binop(c, 1);
    if (cex_match_punct(c, "?")) {
        int64_t a = cex_ternary(c);
        if (!cex_match_punct(c, ":")) {
            pp_error(c->pp, c->file, c->line, s_lit("expected ':' in ?: in #if"));
            c->err = true;
        }
        int64_t b = cex_ternary(c);
        return cond ? a : b;
    }
    return cond;
}

static int64_t cex_expr(CEX *c) { return cex_ternary(c); }

// Replace `defined X` / `defined(X)` with 1/0 PP_NUMBER tokens BEFORE
// macro expansion (per C standard).
static void resolve_defined(PP *pp, PPTok *toks, size_t n, VecPPTok *out) {
    size_t i = 0;
    while (i < n) {
        PPTok t = toks[i];
        if (t.kind == PP_IDENT && str_eq(t.text, s_lit("defined"))) {
            size_t j = i + 1;
            bool paren = false;
            if (j < n && toks[j].kind == PP_PUNCT && str_eq(toks[j].text, s_lit("("))) {
                paren = true; j++;
            }
            int64_t v = 0;
            if (j < n && toks[j].kind == PP_IDENT) {
                v = macro_lookup(pp, toks[j].text) ? 1 : 0;
                j++;
            }
            if (paren && j < n && toks[j].kind == PP_PUNCT && str_eq(toks[j].text, s_lit(")"))) j++;
            PPTok num = {0};
            num.kind = PP_NUMBER;
            num.text = i64_to_str(pp->arena, v);
            num.line = t.line; num.file_id = t.file_id; num.has_space = t.has_space;
            VecPPTok_push_back(pp->arena, out, num);
            i = j;
            continue;
        }
        VecPPTok_push_back(pp->arena, out, t);
        i++;
    }
}

static int64_t eval_if_expr(PP *pp, PPTok *toks, size_t n, string file, int line) {
    // 1) Replace defined(X) with 1/0.
    VecPPTok phase1; VecPPTok_reserve(pp->arena, &phase1, 4);
    resolve_defined(pp, toks, n, &phase1);
    // 2) Macro-expand the rest.
    VecPPTok phase2; VecPPTok_reserve(pp->arena, &phase2, 4);
    expand_tokens(pp, phase1.data, phase1.size, &phase2);
    // 3) Evaluate.
    CEX c = {0};
    c.pp = pp; c.t = phase2.data; c.n = phase2.size;
    c.file = file; c.line = line;
    int64_t v = cex_expr(&c);
    return v;
}

// ----------------------------------------------------------------------
// File processing
// ----------------------------------------------------------------------

static bool path_join(Arena *arena, string dir, string name, string *out) {
    // Trim trailing separators from dir, ensure exactly one between.
    size_t ds = dir.size;
    while (ds > 0 && (dir.str[ds - 1] == '/' || dir.str[ds - 1] == '\\')) ds--;
    char sep = '/';
    char *buf = arena_new_array(arena, char, ds + 1 + name.size + 1);
    if (ds) memcpy(buf, dir.str, ds);
    size_t k = ds;
    if (name.size == 0) { buf[k] = '\0'; *out = (string){buf, k}; return true; }
    if (ds > 0) buf[k++] = sep;
    memcpy(buf + k, name.str, name.size);
    k += name.size;
    buf[k] = '\0';
    *out = (string){buf, k};
    return true;
}

static string parent_dir(Arena *arena, string path) {
    size_t i = path.size;
    while (i > 0 && path.str[i - 1] != '/' && path.str[i - 1] != '\\') i--;
    if (i == 0) return s_dup_cstr(arena, ".", 1);
    return s_dup_cstr(arena, path.str, i - 1);
}

static bool pragma_once_seen(PP *pp, string path) {
    for (PathNode *n = pp->pragma_once; n; n = n->next)
        if (str_eq(n->path, path)) return true;
    return false;
}
static void pragma_once_add(PP *pp, string path) {
    PathNode *n = arena_new(pp->arena, PathNode);
    n->path = s_dup(pp->arena, path);
    n->next = pp->pragma_once;
    pp->pragma_once = n;
}

// Find the include file and read it (once: the contents are what
// process_file() then tokenizes). Checks pragma_once_seen BEFORE reading
// from disk so repeated includes of `#pragma once` headers do zero I/O.
static bool resolve_include(PP *pp, string from_file, string name, bool angle,
                            string *out, string *out_src, bool *already_seen) {
    *already_seen = false;
    if (!angle) {
        // First: relative to the directory of `from_file`.
        string dir = parent_dir(pp->arena, from_file);
        string cand;
        path_join(pp->arena, dir, name, &cand);
        if (pragma_once_seen(pp, cand)) { *out = cand; *already_seen = true; return true; }
        if (read_file(pp->arena, cand, out_src)) { *out = cand; return true; }
    }
    for (size_t i = 0; i < pp->n_include_dirs; i++) {
        string cand;
        path_join(pp->arena, pp->include_dirs[i], name, &cand);
        if (pragma_once_seen(pp, cand)) { *out = cand; *already_seen = true; return true; }
        if (read_file(pp->arena, cand, out_src)) { *out = cand; return true; }
    }
    return false;
}

// Forward decl.
static void process_tokens(PP *pp, PPTok *toks, size_t n, string file, uint16_t file_id);
static void process_file(PP *pp, string path, string src);

typedef struct {
    bool was_active;     // active before entering this group
    bool branch_taken;   // some branch in this if-chain emitted tokens
    bool in_else;        // we're now in the #else branch
    int  line;
    string file;
} CondFrame;

DEFINE_VECTOR_FOR_TYPE(CondFrame, VecCondFrame)

static void process_tokens(PP *pp, PPTok *toks, size_t n, string file, uint16_t file_id) {
    // Cond stack.
    VecCondFrame conds; VecCondFrame_reserve(pp->arena, &conds, 4);
    bool active = true;
    size_t i = 0;
    while (i < n) {
        PPTok t = toks[i];
        if (t.kind == PP_END) break;
        if (t.kind == PP_NEWLINE) { i++; continue; }
        // Detect a directive: '#' as the first non-newline token of a line.
        bool at_line_start = (i == 0) || toks[i - 1].kind == PP_NEWLINE;
        if (at_line_start && t.kind == PP_HASH) {
            size_t j = i + 1;
            // Find end-of-line
            size_t end = j;
            while (end < n && toks[end].kind != PP_NEWLINE && toks[end].kind != PP_END) end++;
            // Empty directive ('#' alone) — accept.
            if (j == end) { i = end + 1; continue; }
            PPTok name = toks[j];
            if (name.kind != PP_IDENT) {
                if (active) pp_error(pp, file, t.line, s_lit("expected directive name after '#'"));
                i = end + 1;
                continue;
            }
            string dn = name.text;
            j++;

            if (str_eq(dn, s_lit("if"))) {
                CondFrame f = {0};
                f.was_active = active;
                f.branch_taken = false;
                f.line = t.line;
                f.file = file;
                if (active) {
                    int64_t v = eval_if_expr(pp, toks + j, end - j, file, t.line);
                    if (v != 0) { f.branch_taken = true; active = true; }
                    else active = false;
                }
                VecCondFrame_push_back(pp->arena, &conds, f);
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("ifdef")) || str_eq(dn, s_lit("ifndef"))) {
                bool want_def = str_eq(dn, s_lit("ifdef"));
                CondFrame f = {0};
                f.was_active = active;
                f.line = t.line;
                f.file = file;
                if (active) {
                    if (j < end && toks[j].kind == PP_IDENT) {
                        bool is_def = macro_lookup(pp, toks[j].text) != NULL;
                        bool ok = (want_def ? is_def : !is_def);
                        f.branch_taken = ok;
                        active = ok;
                    } else {
                        pp_error(pp, file, t.line, s_lit("expected identifier"));
                        active = false;
                    }
                }
                VecCondFrame_push_back(pp->arena, &conds, f);
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("elif"))) {
                if (conds.size == 0) { pp_error(pp, file, t.line, s_lit("#elif without #if")); i = end + 1; continue; }
                CondFrame *f = &conds.data[conds.size - 1];
                if (f->in_else) { pp_error(pp, file, t.line, s_lit("#elif after #else")); i = end + 1; continue; }
                if (!f->was_active) { active = false; i = end + 1; continue; }
                if (f->branch_taken) { active = false; }
                else {
                    int64_t v = eval_if_expr(pp, toks + j, end - j, file, t.line);
                    if (v != 0) { f->branch_taken = true; active = true; } else active = false;
                }
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("else"))) {
                if (conds.size == 0) { pp_error(pp, file, t.line, s_lit("#else without #if")); i = end + 1; continue; }
                CondFrame *f = &conds.data[conds.size - 1];
                if (f->in_else) { pp_error(pp, file, t.line, s_lit("duplicate #else")); i = end + 1; continue; }
                f->in_else = true;
                if (!f->was_active) { active = false; }
                else active = !f->branch_taken;
                if (active) f->branch_taken = true;
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("endif"))) {
                if (conds.size == 0) { pp_error(pp, file, t.line, s_lit("#endif without #if")); i = end + 1; continue; }
                CondFrame f = conds.data[--conds.size];
                active = f.was_active;
                i = end + 1;
                continue;
            }
            if (!active) { i = end + 1; continue; }

            if (str_eq(dn, s_lit("define"))) {
                if (j >= end || toks[j].kind != PP_IDENT) {
                    pp_error(pp, file, t.line, s_lit("expected macro name"));
                    i = end + 1; continue;
                }
                string mname = toks[j].text;
                Macro *m = macro_define(pp, mname);
                j++;
                // Function-like? '(' must immediately follow with no space.
                if (j < end && toks[j].kind == PP_PUNCT && str_eq(toks[j].text, s_lit("(")) && !toks[j].has_space) {
                    m->is_func = true;
                    j++;
                    string params_buf[64];
                    size_t np = 0;
                    bool first_param = true;
                    while (j < end) {
                        if (toks[j].kind == PP_PUNCT && str_eq(toks[j].text, s_lit(")"))) { j++; break; }
                        if (!first_param) {
                            if (toks[j].kind != PP_PUNCT || !str_eq(toks[j].text, s_lit(","))) {
                                pp_error(pp, file, t.line, s_lit("expected ',' in macro params"));
                                break;
                            }
                            j++;
                        }
                        if (j < end && toks[j].kind == PP_PUNCT && str_eq(toks[j].text, s_lit("..."))) {
                            m->is_variadic = true;
                            j++;
                            // Optional ')'.
                            if (j < end && toks[j].kind == PP_PUNCT && str_eq(toks[j].text, s_lit(")"))) j++;
                            break;
                        }
                        if (j >= end || toks[j].kind != PP_IDENT) {
                            pp_error(pp, file, t.line, s_lit("expected parameter name"));
                            break;
                        }
                        if (np < 64) params_buf[np++] = toks[j].text;
                        j++;
                        first_param = false;
                    }
                    m->n_params = np;
                    m->params = arena_new_array(pp->arena, string, np ? np : 1);
                    for (size_t k = 0; k < np; k++) m->params[k] = params_buf[k];
                }
                size_t bn = end - j;
                m->body = arena_new_array(pp->arena, PPTok, bn ? bn : 1);
                for (size_t k = 0; k < bn; k++) m->body[k] = toks[j + k];
                m->n_body = bn;
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("undef"))) {
                if (j < end && toks[j].kind == PP_IDENT) macro_undef(pp, toks[j].text);
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("error"))) {
                // Build message from raw token text on the line.
                size_t cap = 64;
                char *buf = arena_new_array(pp->arena, char, cap);
                size_t sz = 0;
                for (size_t k = j; k < end; k++) {
                    if (k > j) {
                        if (sz + 1 >= cap) { cap *= 2; char *nb = arena_new_array(pp->arena, char, cap); memcpy(nb, buf, sz); buf = nb; }
                        buf[sz++] = ' ';
                    }
                    string txt = toks[k].text;
                    if (sz + txt.size + 1 >= cap) { while (sz + txt.size + 1 >= cap) cap *= 2; char *nb = arena_new_array(pp->arena, char, cap); memcpy(nb, buf, sz); buf = nb; }
                    memcpy(buf + sz, txt.str, txt.size);
                    sz += txt.size;
                }
                buf[sz] = '\0';
                pp_error(pp, file, t.line, (string){buf, sz});
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("pragma"))) {
                if (j < end && toks[j].kind == PP_IDENT && str_eq(toks[j].text, s_lit("once"))) {
                    pragma_once_add(pp, file);
                }
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("line"))) {
                if (!pp->tok_out) {
                    emit_line_directive_if_needed(pp, file_id, t.line);
                    if (pp->out_size > 0 && pp->out[pp->out_size - 1] != '\n') out_char(pp, '\n');
                    out_append(pp, s_lit("#line"));
                    for (size_t k = j; k < end; k++) {
                        out_char(pp, ' ');
                        out_append(pp, toks[k].text);
                    }
                    out_char(pp, '\n');
                    pp->cur_loc_valid = false;
                }
                i = end + 1;
                continue;
            }
            if (str_eq(dn, s_lit("include"))) {
                // Build the header name. Two forms: "..." or <...>.
                bool angle = false;
                string name = {NULL, 0};
                if (j < end && toks[j].kind == PP_STRING) {
                    string s = toks[j].text;
                    if (s.size >= 2) name = (string){s.str + 1, s.size - 2};
                } else if (j < end && toks[j].kind == PP_PUNCT && str_eq(toks[j].text, s_lit("<"))) {
                    angle = true;
                    size_t k = j + 1;
                    size_t start_off = 0;
                    // Compute byte offsets relative to the first token's start.
                    // Reconstruct from token text concatenation.
                    size_t cap = 32;
                    char *buf = arena_new_array(pp->arena, char, cap);
                    size_t sz = 0;
                    while (k < end && !(toks[k].kind == PP_PUNCT && str_eq(toks[k].text, s_lit(">")))) {
                        string tx = toks[k].text;
                        if (sz + tx.size + 1 >= cap) { while (sz + tx.size + 1 >= cap) cap *= 2; char *nb = arena_new_array(pp->arena, char, cap); memcpy(nb, buf, sz); buf = nb; }
                        memcpy(buf + sz, tx.str, tx.size);
                        sz += tx.size;
                        k++;
                    }
                    buf[sz] = '\0';
                    name = (string){buf, sz};
                    (void)start_off;
                } else {
                    // Try macro expansion of the rest of the line.
                    VecPPTok ex; VecPPTok_reserve(pp->arena, &ex, 4);
                    expand_tokens(pp, toks + j, end - j, &ex);
                    if (ex.size > 0 && ex.data[0].kind == PP_STRING) {
                        string s = ex.data[0].text;
                        if (s.size >= 2) name = (string){s.str + 1, s.size - 2};
                    } else {
                        pp_error(pp, file, t.line, s_lit("malformed #include"));
                        i = end + 1; continue;
                    }
                }
                string resolved;
                string inc_src;
                bool already_seen = false;
                if (!resolve_include(pp, file, name, angle, &resolved, &inc_src, &already_seen)) {
                    pp_error(pp, file, t.line,
                             format(pp->arena, str_lit("cannot find include file {}{}{}"),
                                    angle ? str_lit("<") : str_lit("\""), name,
                                    angle ? str_lit(">") : str_lit("\"")));
                    i = end + 1; continue;
                }
                if (already_seen) { i = end + 1; continue; }
                process_file(pp, resolved, inc_src);
                // Restore #line tracking back to enclosing file.
                emit_line_directive_if_needed(pp, file_id, toks[end].line);
                i = end + 1;
                continue;
            }
            // Unknown directive. Tolerate: skip silently.
            i = end + 1;
            continue;
        }

        if (!active) {
            // Skip until next newline.
            while (i < n && toks[i].kind != PP_NEWLINE && toks[i].kind != PP_END) i++;
            if (i < n) i++;
            continue;
        }

        // Non-directive line: expand directly until the next directive line
        // (PP_NEWLINE followed by PP_HASH) or EOF in a single pass.
        size_t consumed = expand_tokens_ex(pp, toks + i, n - i, NULL, true);
        i += consumed;
        // Don't consume the newline; loop will see it next iteration.
    }
    if (conds.size > 0) {
        pp_error(pp, file, conds.data[conds.size - 1].line, s_lit("unterminated #if"));
    }
}

static PPTok *g_pp_file_bufs[16];
static size_t g_pp_file_caps[16];
static int    g_pp_file_depth = 0;

// `src` is the file's contents as read_file() returns them.
static void process_file(PP *pp, string path, string src) {
    string canonical = s_dup(pp->arena, path);
    uint16_t file_id = pp_intern_file(pp, canonical);
    if (src.size > 0 && src.str[src.size - 1] == '\0') src.size -= 1;
    // Reuse a per-depth heap buffer so included headers stay in warm pages
    // instead of malloc/free + minor page faults on every #include.
    size_t init_cap = src.size / 5 + 256;
    VecPPTok toks;
    int depth = g_pp_file_depth++;
    if (depth < 16) {
        if (g_pp_file_caps[depth] < init_cap) {
            free(g_pp_file_bufs[depth]);
            g_pp_file_bufs[depth] = (PPTok *)malloc(init_cap * sizeof(PPTok));
            g_pp_file_caps[depth] = init_cap;
        }
        toks.data = g_pp_file_bufs[depth];
        toks.size = 0;
        toks.max = g_pp_file_caps[depth];
    } else {
        toks.data = (PPTok *)malloc(init_cap * sizeof(PPTok));
        toks.size = 0;
        toks.max = init_cap;
    }
    pp_tokenize(NULL, src, file_id, &toks);
    if (depth < 16) {
        g_pp_file_bufs[depth] = toks.data;
        g_pp_file_caps[depth] = toks.max;
    }
    process_tokens(pp, toks.data, toks.size, canonical, file_id);
    g_pp_file_depth--;
    if (depth >= 16) free(toks.data);
}

// ----------------------------------------------------------------------
// Public entry
// ----------------------------------------------------------------------

static int g_last_preprocess_errors = 0;

int tinyc_last_preprocess_errors(void) { return g_last_preprocess_errors; }

static void pp_init_state(PP *pp, Arena *arena,
                          string *include_dirs, size_t n_include_dirs,
                          string *defines, size_t n_defines) {
    pp_init_ctype();
    *pp = (PP){0};
    pp->arena = arena;
    pp->include_dirs = include_dirs;
    pp->n_include_dirs = n_include_dirs;

    macro_get_slot(pp, s_lit("__FILE__"))->special = PP_SPECIAL_FILE;
    macro_get_slot(pp, s_lit("__LINE__"))->special = PP_SPECIAL_LINE;

    Macro *tcm = macro_define(pp, s_lit("__TINYC__"));
    PPTok one = {0};
    one.kind = PP_NUMBER;
    one.text = s_lit("1");
    tcm->body = arena_new_array(arena, PPTok, 1);
    tcm->body[0] = one;
    tcm->n_body = 1;

    uint16_t cmdline_fid = pp_intern_file(pp, s_lit("<command-line>"));
    for (size_t k = 0; k < n_defines; k++) {
        string d = defines[k];
        size_t eq = 0;
        while (eq < d.size && d.str[eq] != '=') eq++;
        string name = (string){d.str, eq};
        Macro *m = macro_define(pp, name);
        if (eq >= d.size) {
            PPTok body = {0};
            body.kind = PP_NUMBER;
            body.text = s_lit("1");
            m->body = arena_new_array(arena, PPTok, 1);
            m->body[0] = body;
            m->n_body = 1;
        } else {
            string body_src = (string){d.str + eq + 1, d.size - eq - 1};
            VecPPTok body_toks; VecPPTok_reserve(arena, &body_toks, 4);
            pp_tokenize(arena, body_src, cmdline_fid, &body_toks);
            size_t bn = body_toks.size;
            while (bn > 0 && (body_toks.data[bn - 1].kind == PP_NEWLINE
                              || body_toks.data[bn - 1].kind == PP_END)) bn--;
            if (bn > 0) {
                m->body = arena_new_array(arena, PPTok, bn);
                for (size_t bi = 0; bi < bn; bi++) m->body[bi] = body_toks.data[bi];
                m->n_body = bn;
            }
        }
    }
}

string tinyc_preprocess(Arena *arena, string path,
                        string *include_dirs, size_t n_include_dirs,
                        string *defines, size_t n_defines) {
    PP pp;
    pp_init_state(&pp, arena, include_dirs, n_include_dirs, defines, n_defines);
    string main_src = read_file_ok(arena, path);
    out_grow(&pp, main_src.size * 2 + 8192);
    process_file(&pp, path, main_src);
    out_char(&pp, '\n');
    g_last_preprocess_errors = pp.n_errors;
    return (string){pp.out, pp.out_size};
}

VecTcTok tinyc_preprocess_lex(Arena *arena, string path,
                              string *include_dirs, size_t n_include_dirs,
                              string *defines, size_t n_defines) {
    PP pp;
    pp_init_state(&pp, arena, include_dirs, n_include_dirs, defines, n_defines);
    string main_src = read_file_ok(arena, path);
    VecTcTok toks;
    VecTcTok_reserve(arena, &toks, main_src.size / 3 + 4096);
    pp.tok_out = &toks;
    process_file(&pp, path, main_src);
    TcTok eof = {0};
    eof.kind = TC_TK_EOF;
    eof.line = toks.size > 0 ? toks.data[toks.size - 1].line : 1;
    VecTcTok_push_back(arena, &toks, eof);
    g_last_preprocess_errors = pp.n_errors;
    return toks;
}
