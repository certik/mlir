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
#include <string.h>

#include <base/arena.h>
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
    PPKind  kind;
    string  text;       // raw bytes (may be empty for newline / placemarker)
    int     line;       // source line in `file`
    string  file;       // source file path
    bool    has_space;  // whitespace immediately preceded this token (in source)
    PPHide *hide;       // hide-set chain (for non-self-recursion)
} PPTok;

DEFINE_VECTOR_FOR_TYPE(PPTok, VecPPTok)

// ----------------------------------------------------------------------
// Macro table (linked list)
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
    bool     is_predefined_file; // __FILE__: dynamic
    bool     is_predefined_line; // __LINE__: dynamic
    bool     expanding;          // currently-expanding flag (no self-recursion)
    Macro   *next;
};

// ----------------------------------------------------------------------
// State
// ----------------------------------------------------------------------

typedef struct PathNode {
    string path;          // canonical-ish path (as read; we don't normalize)
    struct PathNode *next;
} PathNode;

typedef struct {
    Arena    *arena;
    Macro    *macros;
    string   *include_dirs;
    size_t    n_include_dirs;
    PathNode *pragma_once;       // files that emitted `#pragma once`
    // Output text buffer (grown geometrically).
    char     *out;
    size_t    out_size;
    size_t    out_cap;
    // Current emission location — used to decide when to emit a #line.
    string    cur_file;
    int       cur_line;
    bool      cur_loc_valid;
    bool      had_error;
} PP;

// ----------------------------------------------------------------------
// Diagnostics
// ----------------------------------------------------------------------

static void pp_error(PP *pp, string file, int line, string msg) {
    pp->had_error = true;
    println(str_lit("tinyc preprocess error at {}:{}: {}"),
            file, (int64_t)line, msg);
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
    if (s.size) memcpy(pp->out + pp->out_size, s.str, s.size);
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

static bool pp_is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static bool pp_is_digit(char c) { return c >= '0' && c <= '9'; }
static bool pp_is_alnum(char c) { return pp_is_alpha(c) || pp_is_digit(c); }

// Tokenize `src` into PP tokens. Comments are dropped. Newlines become
// PP_NEWLINE tokens (one per source newline, even adjacent ones — needed
// to keep #line tracking accurate). `has_space` is set when whitespace
// (excluding newlines) preceded the token.
static void pp_tokenize(Arena *arena, string src, string file, VecPPTok *out) {
    int line = 1;
    size_t i = 0;
    bool sol = true;       // at start of a logical line (only ws so far)
    bool space_pending = false;
    while (i < src.size) {
        char c = src.str[i];
        if (c == ' ' || c == '\t' || c == '\r') {
            i++;
            space_pending = true;
            continue;
        }
        if (c == '\\' && i + 1 < src.size && src.str[i + 1] == '\n') {
            // line continuation
            i += 2;
            line++;
            continue;
        }
        if (c == '\n') {
            PPTok t = {0};
            t.kind = PP_NEWLINE;
            t.line = line;
            t.file = file;
            VecPPTok_push_back(arena, out, t);
            i++;
            line++;
            sol = true;
            space_pending = false;
            continue;
        }
        if (c == '/' && i + 1 < src.size && src.str[i + 1] == '/') {
            while (i < src.size && src.str[i] != '\n') i++;
            space_pending = true;
            continue;
        }
        if (c == '/' && i + 1 < src.size && src.str[i + 1] == '*') {
            i += 2;
            while (i + 1 < src.size &&
                   !(src.str[i] == '*' && src.str[i + 1] == '/')) {
                if (src.str[i] == '\n') line++;
                i++;
            }
            if (i + 1 < src.size) i += 2;
            else i = src.size;
            space_pending = true;
            continue;
        }

        PPTok t = {0};
        t.line = line;
        t.file = file;
        t.has_space = space_pending;

        if (c == '"') {
            size_t j = i + 1;
            while (j < src.size && src.str[j] != '"') {
                if (src.str[j] == '\\' && j + 1 < src.size) j += 2;
                else { if (src.str[j] == '\n') line++; j++; }
            }
            if (j < src.size) j++;
            t.kind = PP_STRING;
            t.text = s_dup_cstr(arena, src.str + i, j - i);
            VecPPTok_push_back(arena, out, t);
            i = j;
            sol = false;
            space_pending = false;
            continue;
        }
        if (c == '\'') {
            size_t j = i + 1;
            while (j < src.size && src.str[j] != '\'') {
                if (src.str[j] == '\\' && j + 1 < src.size) j += 2;
                else j++;
            }
            if (j < src.size) j++;
            t.kind = PP_CHAR;
            t.text = s_dup_cstr(arena, src.str + i, j - i);
            VecPPTok_push_back(arena, out, t);
            i = j;
            sol = false;
            space_pending = false;
            continue;
        }
        if (pp_is_digit(c) ||
            (c == '.' && i + 1 < src.size && pp_is_digit(src.str[i + 1]))) {
            // pp-number: digit ( digit | letter | '.' | 'e+'|'e-'|'E+'|'E-' )*
            size_t j = i + 1;
            while (j < src.size) {
                char d = src.str[j];
                if (pp_is_alnum(d) || d == '.') { j++; continue; }
                if ((d == '+' || d == '-') && j > 0 &&
                    (src.str[j - 1] == 'e' || src.str[j - 1] == 'E' ||
                     src.str[j - 1] == 'p' || src.str[j - 1] == 'P')) {
                    j++; continue;
                }
                break;
            }
            t.kind = PP_NUMBER;
            t.text = s_dup_cstr(arena, src.str + i, j - i);
            VecPPTok_push_back(arena, out, t);
            i = j;
            sol = false;
            space_pending = false;
            continue;
        }
        if (pp_is_alpha(c)) {
            size_t j = i + 1;
            while (j < src.size && pp_is_alnum(src.str[j])) j++;
            t.kind = PP_IDENT;
            t.text = s_dup_cstr(arena, src.str + i, j - i);
            VecPPTok_push_back(arena, out, t);
            i = j;
            sol = false;
            space_pending = false;
            continue;
        }
        if (c == '#') {
            if (i + 1 < src.size && src.str[i + 1] == '#') {
                t.kind = PP_HASHHASH;
                t.text = s_lit("##");
                VecPPTok_push_back(arena, out, t);
                i += 2;
                sol = false;
                space_pending = false;
                continue;
            }
            t.kind = PP_HASH;
            t.text = s_lit("#");
            VecPPTok_push_back(arena, out, t);
            i++;
            sol = false;
            space_pending = false;
            continue;
        }

        // Multi-char punctuation we recognize. Order matters (longest match).
        static const char *MULTI3[] = { "<<=", ">>=", "...", NULL };
        static const char *MULTI2[] = {
            "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
            "++", "--", "->", NULL
        };
        const char *match = NULL;
        size_t mlen = 1;
        if (i + 2 < src.size) {
            for (int k = 0; MULTI3[k]; k++) {
                if (src.str[i] == MULTI3[k][0] &&
                    src.str[i + 1] == MULTI3[k][1] &&
                    src.str[i + 2] == MULTI3[k][2]) {
                    match = MULTI3[k]; mlen = 3; break;
                }
            }
        }
        if (!match && i + 1 < src.size) {
            for (int k = 0; MULTI2[k]; k++) {
                if (src.str[i] == MULTI2[k][0] &&
                    src.str[i + 1] == MULTI2[k][1]) {
                    match = MULTI2[k]; mlen = 2; break;
                }
            }
        }
        t.kind = PP_PUNCT;
        if (match) t.text = s_dup_cstr(arena, src.str + i, mlen);
        else       t.text = s_dup_cstr(arena, src.str + i, 1);
        VecPPTok_push_back(arena, out, t);
        i += mlen;
        sol = false;
        space_pending = false;
        (void)sol;
    }
    // Final newline guarantees directives at EOF terminate.
    PPTok eol = {0};
    eol.kind = PP_NEWLINE;
    eol.line = line;
    eol.file = file;
    VecPPTok_push_back(arena, out, eol);
    PPTok end = {0};
    end.kind = PP_END;
    end.line = line;
    end.file = file;
    VecPPTok_push_back(arena, out, end);
}

// ----------------------------------------------------------------------
// Macro table operations
// ----------------------------------------------------------------------

static Macro *macro_lookup(PP *pp, string name) {
    for (Macro *m = pp->macros; m; m = m->next) {
        if (str_eq(m->name, name)) return m;
    }
    return NULL;
}

static void macro_undef(PP *pp, string name) {
    Macro **p = &pp->macros;
    while (*p) {
        if (str_eq((*p)->name, name)) { *p = (*p)->next; return; }
        p = &(*p)->next;
    }
}

static Macro *macro_define(PP *pp, string name) {
    macro_undef(pp, name);
    Macro *m = arena_new(pp->arena, Macro);
    *m = (Macro){0};
    m->name = s_dup(pp->arena, name);
    m->next = pp->macros;
    pp->macros = m;
    return m;
}

// ----------------------------------------------------------------------
// Token-stream helpers
// ----------------------------------------------------------------------

static bool is_eol_or_end(PPTok *t) {
    return t->kind == PP_NEWLINE || t->kind == PP_END;
}

// (No unused-helper section here — utilities live inline.)

// ----------------------------------------------------------------------
// #line emission
// ----------------------------------------------------------------------

static void emit_line_directive_if_needed(PP *pp, string file, int line) {
    if (pp->cur_loc_valid &&
        str_eq(pp->cur_file, file) && pp->cur_line == line) return;
    if (pp->out_size > 0 && pp->out[pp->out_size - 1] != '\n')
        out_char(pp, '\n');
    out_append(pp, s_lit("#line "));
    out_append(pp, i64_to_str(pp->arena, line));
    out_append(pp, s_lit(" \""));
    // Quote-escape backslashes and quotes.
    for (size_t i = 0; i < file.size; i++) {
        char c = file.str[i];
        if (c == '\\' || c == '"') out_char(pp, '\\');
        out_char(pp, c);
    }
    out_append(pp, s_lit("\"\n"));
    pp->cur_file = file;
    pp->cur_line = line;
    pp->cur_loc_valid = true;
}

// Emit a token to the output stream (after macro expansion). Inserts a
// `#line` directive when the source location changes; otherwise pads with
// newlines so token line numbers stay roughly faithful.
static bool g_need_space_sep = false;
static void emit_pp_token(PP *pp, PPTok *t) {
    if (t->kind == PP_PLACEMARKER) return;
    if (t->kind == PP_NEWLINE || t->kind == PP_END) return;
    if (!pp->cur_loc_valid || !str_eq(pp->cur_file, t->file)) {
        emit_line_directive_if_needed(pp, t->file, t->line);
        g_need_space_sep = false;
    } else if (pp->cur_line != t->line) {
        if (t->line > pp->cur_line && t->line - pp->cur_line <= 8) {
            for (int k = 0; k < t->line - pp->cur_line; k++) out_char(pp, '\n');
            pp->cur_line = t->line;
            g_need_space_sep = false;
        } else {
            emit_line_directive_if_needed(pp, t->file, t->line);
            g_need_space_sep = false;
        }
    }
    if (g_need_space_sep) out_char(pp, ' ');
    out_append(pp, t->text);
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
    pp_tokenize(pp->arena, joined, a.file, &tmp);
    // Find the first non-newline token.
    PPTok r = {0};
    r.kind = PP_IDENT; // fallback
    r.text = joined;
    r.line = a.line;
    r.file = a.file;
    for (size_t i = 0; i < tmp.size; i++) {
        if (tmp.data[i].kind == PP_END) break;
        if (tmp.data[i].kind == PP_NEWLINE) continue;
        r = tmp.data[i];
        r.line = a.line;
        r.file = a.file;
        break;
    }
    return r;
}

// Forward decl.
static void expand_tokens(PP *pp, PPTok *toks, size_t n, VecPPTok *out);

// Find arguments for a function-like macro at `*pi` (which points just
// after the macro name). The opening '(' may be preceded by whitespace
// but not by a newline that we've already emitted — for simplicity we
// allow any number of newlines too. Sets `*pi` past the closing ')'.
// Returns true if the call form was consumed; otherwise false (and *pi
// is unchanged).
static bool collect_args(PP *pp, PPTok *toks, size_t n, size_t *pi,
                         Macro *m, VecPPTok *args /* array length n_params(+1 for var) */ ) {
    size_t i = *pi;
    while (i < n && (toks[i].kind == PP_NEWLINE)) i++;
    if (i >= n || toks[i].kind != PP_PUNCT || !str_eq(toks[i].text, s_lit("(")))
        return false;
    i++; // consume '('
    // Initialize argument vectors (already reserved by caller).
    int depth = 0;
    size_t arg_idx = 0;
    size_t expected = m->n_params + (m->is_variadic ? 1 : 0);
    bool any_token_in_arg = false;
    while (i < n) {
        PPTok t = toks[i];
        if (t.kind == PP_END) {
            pp_error(pp, t.file, t.line, s_lit("unterminated macro arguments"));
            return false;
        }
        if (t.kind == PP_NEWLINE) { i++; continue; }
        if (t.kind == PP_PUNCT && str_eq(t.text, s_lit("("))) depth++;
        else if (t.kind == PP_PUNCT && str_eq(t.text, s_lit(")"))) {
            if (depth == 0) { i++; break; }
            depth--;
        } else if (t.kind == PP_PUNCT && str_eq(t.text, s_lit(",")) && depth == 0) {
            // Variadic: stop splitting once we've reached the variadic slot.
            if (m->is_variadic && arg_idx >= m->n_params) {
                VecPPTok_push_back(pp->arena, &args[arg_idx], t);
                i++;
                any_token_in_arg = true;
                continue;
            }
            arg_idx++;
            if (arg_idx >= expected) {
                pp_error(pp, t.file, t.line, s_lit("too many arguments to macro"));
                return false;
            }
            i++;
            any_token_in_arg = false;
            continue;
        }
        VecPPTok_push_back(pp->arena, &args[arg_idx], t);
        any_token_in_arg = true;
        i++;
    }
    // If we collected nothing for the last slot AND total slots == 1 AND
    // expected == 0 (zero-param macro), that's fine — caller saw `()`.
    if (expected == 0) {
        // Accept M() with no args.
        *pi = i;
        return true;
    }
    // Variadic with no extra args: __VA_ARGS__ becomes empty.
    if (m->is_variadic && arg_idx < expected - 1) {
        // Last slot stays empty (already so).
        arg_idx = expected - 1;
    }
    if (!any_token_in_arg && arg_idx == 0 && m->n_params == 1 && !m->is_variadic) {
        // M() called on M(x) — treat as one empty arg.
    }
    *pi = i;
    return true;
}

// Substitute parameters in macro body, handling # and ##. Returns the
// substituted token list (still un-rescanned) in `out`.
static void substitute(PP *pp, Macro *m, VecPPTok *args, VecPPTok *out) {
    PPTok *body = m->body;
    size_t nb = m->n_body;
    // Pre-expand each argument once for use in non-stringize, non-paste
    // positions. Per C: stringize and paste use the un-expanded form.
    size_t n_args = m->n_params + (m->is_variadic ? 1 : 0);
    VecPPTok *expanded = arena_new_array(pp->arena, VecPPTok, n_args ? n_args : 1);
    for (size_t k = 0; k < n_args; k++) {
        VecPPTok_reserve(pp->arena, &expanded[k], 4);
        expand_tokens(pp, args[k].data, args[k].size, &expanded[k]);
    }

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
                s.line = t.line; s.file = t.file; s.has_space = t.has_space;
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
                if (p < 0 && va_idx >= 0 && str_eq(right_default.text, s_lit("__VA_ARGS__"))) p = va_idx;
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
                VecPPTok *src = followed_by_paste ? &args[p] : &expanded[p];
                if (src->size == 0 && followed_by_paste) {
                    PPTok pm = {0};
                    pm.kind = PP_PLACEMARKER;
                    pm.line = t.line; pm.file = t.file;
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

// Expand tokens in `toks` and append the result to `out`.
// Replace work->data[pos .. pos+del) with the `ins_n` tokens at `ins`.
// Grows the vector as needed; preserves all other entries.
static void vec_pptok_splice(Arena *a, VecPPTok *v, size_t pos, size_t del,
                             const PPTok *ins, size_t ins_n) {
    size_t old_size = v->size;
    size_t tail_n = old_size - (pos + del);
    if (ins_n > del) {
        size_t need = ins_n - del;
        // Use push_back with dummy values to grow capacity.
        PPTok zero = {0};
        for (size_t k = 0; k < need; k++) VecPPTok_push_back(a, v, zero);
        // Move tail right (in reverse, since regions overlap).
        for (size_t k = tail_n; k > 0; k--) {
            v->data[pos + ins_n + k - 1] = v->data[pos + del + k - 1];
        }
    } else if (ins_n < del) {
        size_t shrink = del - ins_n;
        // Move tail left.
        for (size_t k = 0; k < tail_n; k++) {
            v->data[pos + ins_n + k] = v->data[pos + del + k];
        }
        v->size -= shrink;
    }
    for (size_t k = 0; k < ins_n; k++) v->data[pos + k] = ins[k];
}

static void expand_tokens(PP *pp, PPTok *toks, size_t n, VecPPTok *out) {
    // Copy input into a working buffer so substitutions can be spliced
    // back in for proper rescanning across the original input boundary.
    // This is required by C: after a macro is replaced, the result is
    // reinserted into the source stream so that nested function-like
    // macros can pick up arguments that follow in the original input.
    VecPPTok work; VecPPTok_reserve(pp->arena, &work, n + 4);
    for (size_t k = 0; k < n; k++) VecPPTok_push_back(pp->arena, &work, toks[k]);
    size_t i = 0;
    while (i < work.size) {
        PPTok t = work.data[i];
        if (t.kind == PP_END) break;
        if (t.kind == PP_NEWLINE) { i++; continue; }
        if (t.kind == PP_IDENT) {
            // __FILE__ / __LINE__: expand to dynamic literal.
            if (str_eq(t.text, s_lit("__FILE__"))) {
                PPTok s = {0};
                // File name as string literal.
                size_t fsz = t.file.size;
                char *buf = arena_new_array(pp->arena, char, fsz + 4);
                size_t k = 0;
                buf[k++] = '"';
                for (size_t j = 0; j < fsz; j++) {
                    char c = t.file.str[j];
                    if (c == '\\' || c == '"') { buf[k++] = '\\'; }
                    buf[k++] = c;
                }
                buf[k++] = '"';
                buf[k] = '\0';
                s.kind = PP_STRING;
                s.text = (string){buf, k};
                s.line = t.line; s.file = t.file; s.has_space = t.has_space;
                VecPPTok_push_back(pp->arena, out, s);
                i++;
                continue;
            }
            if (str_eq(t.text, s_lit("__LINE__"))) {
                PPTok s = {0};
                s.kind = PP_NUMBER;
                s.text = i64_to_str(pp->arena, (int64_t)t.line);
                s.line = t.line; s.file = t.file; s.has_space = t.has_space;
                VecPPTok_push_back(pp->arena, out, s);
                i++;
                continue;
            }
            if (hide_contains(t.hide, t.text)) {
                VecPPTok_push_back(pp->arena, out, t);
                i++;
                continue;
            }
            Macro *m = macro_lookup(pp, t.text);
            if (m && !m->expanding) {
                if (!m->is_func) {
                    // Object-like: substitute, then splice the result back
                    // into the work buffer at position `i` so rescanning
                    // happens against the rest of the original input too.
                    m->expanding = true;
                    VecPPTok inner; VecPPTok_reserve(pp->arena, &inner, 4);
                    substitute(pp, m, NULL, &inner);
                    m->expanding = false;
                    for (size_t k = 0; k < inner.size; k++) {
                        inner.data[k].line = t.line;
                        inner.data[k].file = t.file;
                        inner.data[k].hide = hide_add(pp->arena, inner.data[k].hide, m->name);
                    }
                    if (inner.size > 0) inner.data[0].has_space = t.has_space;
                    vec_pptok_splice(pp->arena, &work, i, 1, inner.data, inner.size);
                    continue;
                } else {
                    // Function-like: peek for '(' across the work buffer.
                    size_t j = i + 1;
                    while (j < work.size && work.data[j].kind == PP_NEWLINE) j++;
                    if (j >= work.size || work.data[j].kind != PP_PUNCT
                            || !str_eq(work.data[j].text, s_lit("("))) {
                        VecPPTok_push_back(pp->arena, out, t);
                        i++;
                        continue;
                    }
                    size_t expected = m->n_params + (m->is_variadic ? 1 : 0);
                    VecPPTok *args = arena_new_array(pp->arena, VecPPTok,
                                                     expected ? expected : 1);
                    for (size_t k = 0; k < expected; k++)
                        VecPPTok_reserve(pp->arena, &args[k], 2);
                    size_t cur = j;
                    if (!collect_args(pp, work.data, work.size, &cur, m, args)) {
                        i++;
                        continue;
                    }
                    m->expanding = true;
                    VecPPTok inner; VecPPTok_reserve(pp->arena, &inner, 4);
                    substitute(pp, m, args, &inner);
                    m->expanding = false;
                    for (size_t k = 0; k < inner.size; k++) {
                        inner.data[k].line = t.line;
                        inner.data[k].file = t.file;
                        inner.data[k].hide = hide_add(pp->arena, inner.data[k].hide, m->name);
                    }
                    if (inner.size > 0) inner.data[0].has_space = t.has_space;
                    // Splice inner tokens into `work` in place of the call
                    // (identifier + arg list); rescan from position `i`.
                    vec_pptok_splice(pp->arena, &work, i, cur - i,
                                     inner.data, inner.size);
                    continue;
                }
            }
        }
        VecPPTok_push_back(pp->arena, out, t);
        i++;
    }
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
    static PPTok eof = { PP_END, {NULL,0}, 0, {NULL,0}, false, NULL };
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
            num.line = t.line; num.file = t.file; num.has_space = t.has_space;
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

static bool file_exists(Arena *arena, string path) {
    string txt;
    Arena *probe = arena_create(4 * 1024);
    bool ok = read_file(probe, path, &txt);
    arena_destroy(probe);
    (void)arena;
    return ok;
}

static bool resolve_include(PP *pp, string from_file, string name, bool angle, string *out) {
    if (!angle) {
        // First: relative to the directory of `from_file`.
        string dir = parent_dir(pp->arena, from_file);
        string cand;
        path_join(pp->arena, dir, name, &cand);
        if (file_exists(pp->arena, cand)) { *out = cand; return true; }
    }
    for (size_t i = 0; i < pp->n_include_dirs; i++) {
        string cand;
        path_join(pp->arena, pp->include_dirs[i], name, &cand);
        if (file_exists(pp->arena, cand)) { *out = cand; return true; }
    }
    return false;
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

// Forward decl.
static void process_tokens(PP *pp, PPTok *toks, size_t n, string file);
static void process_file(PP *pp, string path);

typedef struct {
    bool was_active;     // active before entering this group
    bool branch_taken;   // some branch in this if-chain emitted tokens
    bool in_else;        // we're now in the #else branch
    int  line;
    string file;
} CondFrame;

DEFINE_VECTOR_FOR_TYPE(CondFrame, VecCondFrame)

static void process_tokens(PP *pp, PPTok *toks, size_t n, string file) {
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
                // Update line/file metadata for subsequent tokens. We
                // implement this by directly emitting a #line directive
                // and adjusting cur_line; but the simplest path is to
                // forward the directive through unchanged so the lexer
                // honors it.
                emit_line_directive_if_needed(pp, file, t.line);
                if (pp->out_size > 0 && pp->out[pp->out_size - 1] != '\n') out_char(pp, '\n');
                out_append(pp, s_lit("#line"));
                for (size_t k = j; k < end; k++) {
                    out_char(pp, ' ');
                    out_append(pp, toks[k].text);
                }
                out_char(pp, '\n');
                pp->cur_loc_valid = false;
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
                if (!resolve_include(pp, file, name, angle, &resolved)) {
                    pp_error(pp, file, t.line, s_lit("cannot find include file"));
                    i = end + 1; continue;
                }
                if (pragma_once_seen(pp, resolved)) { i = end + 1; continue; }
                process_file(pp, resolved);
                // Restore #line tracking back to enclosing file.
                emit_line_directive_if_needed(pp, file, toks[end].line);
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

        // Non-directive line: collect through end-of-line, expand macros,
        // emit results. To support macro invocations whose argument list
        // spans newlines (e.g. println( ..., \n ..., ...)), we keep
        // extending `end` past plain newlines until we hit a directive
        // line (a '#' at line start) or EOF. Tokens stay tagged with
        // their original source line via #line directives in the output.
        size_t end = i;
        while (end < n && toks[end].kind != PP_END) {
            if (toks[end].kind == PP_NEWLINE) {
                size_t k = end + 1;
                while (k < n && toks[k].kind == PP_NEWLINE) k++;
                if (k < n && toks[k].kind == PP_HASH) break;
                end = k;
                continue;
            }
            end++;
        }
        VecPPTok ex; VecPPTok_reserve(pp->arena, &ex, 4);
        expand_tokens(pp, toks + i, end - i, &ex);
        for (size_t k = 0; k < ex.size; k++) emit_pp_token(pp, &ex.data[k]);
        i = end;
        // Don't consume the newline; loop will see it next iteration.
    }
    if (conds.size > 0) {
        pp_error(pp, file, conds.data[conds.size - 1].line, s_lit("unterminated #if"));
    }
}

static void process_file(PP *pp, string path) {
    string canonical = s_dup(pp->arena, path);
    string src = read_file_ok(pp->arena, canonical);
    if (src.size > 0 && src.str[src.size - 1] == '\0') src.size -= 1;
    VecPPTok toks; VecPPTok_reserve(pp->arena, &toks, 256);
    pp_tokenize(pp->arena, src, canonical, &toks);
    process_tokens(pp, toks.data, toks.size, canonical);
}

// ----------------------------------------------------------------------
// Public entry
// ----------------------------------------------------------------------

string tinyc_preprocess(Arena *arena, string path,
                        string *include_dirs, size_t n_include_dirs,
                        string *defines, size_t n_defines) {
    PP pp = {0};
    pp.arena = arena;
    pp.include_dirs = include_dirs;
    pp.n_include_dirs = n_include_dirs;

    // Predefined macros: __TINYC__ = 1.
    Macro *tcm = macro_define(&pp, s_lit("__TINYC__"));
    PPTok one = {0};
    one.kind = PP_NUMBER;
    one.text = s_lit("1");
    tcm->body = arena_new_array(arena, PPTok, 1);
    tcm->body[0] = one;
    tcm->n_body = 1;
    // __FILE__ and __LINE__ are intercepted at expansion time.

    // Command-line -D defines. Each entry is either "NAME" or "NAME=BODY".
    // The body is tokenized via the same pp_tokenize() the source uses.
    for (size_t k = 0; k < n_defines; k++) {
        string d = defines[k];
        size_t eq = 0;
        while (eq < d.size && d.str[eq] != '=') eq++;
        string name = (string){d.str, eq};
        Macro *m = macro_define(&pp, name);
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
            pp_tokenize(arena, body_src, s_lit("<command-line>"), &body_toks);
            // Strip trailing PP_NEWLINE / PP_END that pp_tokenize appends.
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

    process_file(&pp, path);
    out_char(&pp, '\n');
    return (string){pp.out, pp.out_size};
}
