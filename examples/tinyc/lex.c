// tinyC lexer — hand-rolled, character-by-character. No re2c, no libc.

#include "tinyc.h"

// Convert a decimal mantissa (< 2^63) and base-10 exponent to IEEE-754 double.
// The Dekker two-product in the net_exp in [-22, -1] branch relies on
// non-contracted IEEE-754 rounding: if the host compiler fuses `f - qp` into
// `fma(-q, p, f)` under `-O3 -march=native`, `qp_err` is subtracted twice (#383).
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
static double lex_decimal_to_double(uint64_t mantissa, int net_exp) {
    // mantissa < 2^63: its top 52 bits and its low 11 bits each convert to
    // double exactly, so `f + lo` is the mantissa without any rounding. (Only
    // int -> double conversions of non-negative values are used: tinyc lowers
    // double -> int64 through i32, and this file has to self-host.)
    double f = (double)(int64_t)(mantissa >> 11) * 2048.0;
    double lo = (double)(int64_t)(mantissa & 2047u);
    if (net_exp > 0) {
        // Multiply by 10^net_exp using exact integer powers up to 10^22
        // (which is exactly representable in f64), chunking larger exponents.
        f += lo;
        while (net_exp > 0) {
            int chunk = net_exp > 22 ? 22 : net_exp;
            double p = 1.0;
            for (int k = 0; k < chunk; k++) p *= 10.0;
            f *= p;
            net_exp -= chunk;
        }
    } else if (net_exp < 0 && net_exp >= -22) {
        // mantissa / 10^n with 10^n exact. Carry the low mantissa bits `lo`
        // and the exact division remainder (Dekker two-product) into one
        // correction term so 17+ digit literals round to the nearest ULP.
        double p = 1.0;
        for (int k = 0; k < -net_exp; k++) p *= 10.0;
        double q = f / p;
        double qp = q * p;
        double t = 134217729.0 * q;   // 2^27 + 1
        double qh = t - (t - q);
        double ql = q - qh;
        t = 134217729.0 * p;
        double ph = t - (t - p);
        double pl = p - ph;
        double qp_err = ((qh * ph - qp) + qh * pl + ql * ph) + ql * pl;
        double rem = ((f - qp) - qp_err) + lo;
        f = q + rem / p;
    } else if (net_exp == 0) {
        f += lo;
    } else if (net_exp < 0) {
        f += lo;
        int n = -net_exp;
        while (n > 0) {
            int chunk = n > 22 ? 22 : n;
            double p = 1.0;
            for (int k = 0; k < chunk; k++) p *= 10.0;
            f /= p;
            n -= chunk;
        }
    }
    return f;
}

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <base/io.h>
#include <base/string.h>

#define LEX_CT_ALPHA 1u
#define LEX_CT_DIGIT 2u
#define LEX_CT_ALNUM 3u
#define LEX_CT_WS    4u

static uint8_t g_lex_ctype[256];
static bool g_lex_ctype_init = false;

static void lex_init_ctype(void) {
    if (g_lex_ctype_init) return;
    for (int c = 'a'; c <= 'z'; c++) g_lex_ctype[c] = LEX_CT_ALPHA;
    for (int c = 'A'; c <= 'Z'; c++) g_lex_ctype[c] = LEX_CT_ALPHA;
    g_lex_ctype['_'] = LEX_CT_ALPHA;
    for (int c = '0'; c <= '9'; c++) g_lex_ctype[c] = LEX_CT_DIGIT;
    g_lex_ctype[' '] = LEX_CT_WS;
    g_lex_ctype['\t'] = LEX_CT_WS;
    g_lex_ctype['\r'] = LEX_CT_WS;
    g_lex_ctype_init = true;
}

static bool is_alpha(char c) { return (g_lex_ctype[(unsigned char)c] & LEX_CT_ALPHA) != 0; }
static bool is_digit(char c) { return (g_lex_ctype[(unsigned char)c] & LEX_CT_DIGIT) != 0; }
static bool is_alnum(char c) { return (g_lex_ctype[(unsigned char)c] & LEX_CT_ALNUM) != 0; }

TcTokKind tinyc_keyword_or_ident(string s) {
    if (s.size < 2 || s.size > 17) return TC_TK_IDENT;
    const char *p = s.str;
    if ((unsigned)((unsigned char)p[0] - 'a') > 25u && p[0] != '_') return TC_TK_IDENT;
    switch (s.size) {
    case 2:
        if (p[0] == 'i' && p[1] == 'f') return TC_TK_KW_IF;
        if (p[0] == 'd' && p[1] == 'o') return TC_TK_KW_DO;
        return TC_TK_IDENT;
    case 3:
        if (p[0] == 'i' && p[1] == 'n' && p[2] == 't') return TC_TK_KW_INT;
        if (p[0] == 'f' && p[1] == 'o' && p[2] == 'r') return TC_TK_KW_FOR;
        return TC_TK_IDENT;
    case 4:
        switch (p[0]) {
        case 'e': if (memcmp(p, "else", 4) == 0) return TC_TK_KW_ELSE;
                  if (memcmp(p, "enum", 4) == 0) return TC_TK_KW_ENUM; break;
        case 'c': if (memcmp(p, "char", 4) == 0) return TC_TK_KW_CHAR;
                  if (memcmp(p, "case", 4) == 0) return TC_TK_KW_CASE; break;
        case 'v': if (memcmp(p, "void", 4) == 0) return TC_TK_KW_VOID; break;
        case 'l': if (memcmp(p, "long", 4) == 0) return TC_TK_KW_LONG; break;
        case 'b': if (memcmp(p, "bool", 4) == 0) return TC_TK_KW_BOOL; break;
        case 'g': if (memcmp(p, "goto", 4) == 0) return TC_TK_KW_GOTO; break;
        case 'n': if (memcmp(p, "null", 4) == 0) return TC_TK_KW_NULL; break;
        }
        return TC_TK_IDENT;
    case 5:
        switch (p[0]) {
        case 'c': if (memcmp(p, "const", 5) == 0) return TC_TK_KW_CONST; break;
        case 'w': if (memcmp(p, "while", 5) == 0) return TC_TK_KW_WHILE; break;
        case 'b': if (memcmp(p, "break", 5) == 0) return TC_TK_KW_BREAK; break;
        case 's': if (memcmp(p, "short", 5) == 0) return TC_TK_KW_SHORT; break;
        case 'f': if (memcmp(p, "float", 5) == 0) return TC_TK_KW_FLOAT; break;
        case 'u': if (memcmp(p, "union", 5) == 0) return TC_TK_KW_UNION; break;
        case '_': if (memcmp(p, "_Bool", 5) == 0) return TC_TK_KW_BOOL; break;
        }
        return TC_TK_IDENT;
    case 6:
        switch (p[0]) {
        case 'r': if (memcmp(p, "return", 6) == 0) return TC_TK_KW_RETURN; break;
        case 's': if (memcmp(p, "struct", 6) == 0) return TC_TK_KW_STRUCT;
                  if (memcmp(p, "static", 6) == 0) return TC_TK_KW_STATIC;
                  if (memcmp(p, "sizeof", 6) == 0) return TC_TK_KW_SIZEOF;
                  if (memcmp(p, "switch", 6) == 0) return TC_TK_KW_SWITCH;
                  if (memcmp(p, "signed", 6) == 0) return TC_TK_KW_SIGNED; break;
        case 'd': if (memcmp(p, "double", 6) == 0) return TC_TK_KW_DOUBLE; break;
        case 'e': if (memcmp(p, "extern", 6) == 0) return TC_TK_KW_EXTERN; break;
        case 'i': if (memcmp(p, "inline", 6) == 0) return TC_TK_KW_INLINE; break;
        }
        return TC_TK_IDENT;
    case 7:
        if (p[0] == 't' && memcmp(p, "typedef", 7) == 0) return TC_TK_KW_TYPEDEF;
        if (p[0] == 'd' && memcmp(p, "default", 7) == 0) return TC_TK_KW_DEFAULT;
        if (p[0] == 'v' && memcmp(p, "va_list", 7) == 0) return TC_TK_KW_VA_LIST;
        if (p[0] == '_' && memcmp(p, "__cdecl", 7) == 0) return TC_TK_KW_CALLCONV;
        return TC_TK_IDENT;
    case 8:
        if (p[0] == 'c' && memcmp(p, "continue", 8) == 0) return TC_TK_KW_CONTINUE;
        if (p[0] == 'u' && memcmp(p, "unsigned", 8) == 0) return TC_TK_KW_UNSIGNED;
        if (p[0] == 'v' && memcmp(p, "volatile", 8) == 0) return TC_TK_KW_VOLATILE;
        if (p[0] == 'r' && memcmp(p, "restrict", 8) == 0) return TC_TK_KW_RESTRICT;
        if (p[0] == 'r' && memcmp(p, "register", 8) == 0) return TC_TK_KW_REGISTER;
        if (p[0] == '_' && memcmp(p, "_Alignof", 8) == 0) return TC_TK_KW_ALIGNOF;
        if (p[0] == '_' && memcmp(p, "_Generic", 8) == 0) return TC_TK_KW_GENERIC;
        return TC_TK_IDENT;
    case 9:
        if (p[0] == '_' && memcmp(p, "__stdcall", 9) == 0) return TC_TK_KW_CALLCONV;
        return TC_TK_IDENT;
    case 10:
        if (p[0] == '_' && memcmp(p, "__restrict", 10) == 0) return TC_TK_KW_RESTRICT;
        if (p[0] == '_' && memcmp(p, "__volatile", 10) == 0) return TC_TK_KW_VOLATILE;
        if (p[0] == '_' && memcmp(p, "__declspec", 10) == 0) return TC_TK_KW_DECLSPEC;
        if (p[0] == '_' && memcmp(p, "__fastcall", 10) == 0) return TC_TK_KW_CALLCONV;
        return TC_TK_IDENT;
    case 12:
        if (p[0] == '_' && memcmp(p, "__restrict__", 12) == 0) return TC_TK_KW_RESTRICT;
        if (p[0] == '_' && memcmp(p, "__volatile__", 12) == 0) return TC_TK_KW_VOLATILE;
        return TC_TK_IDENT;
    case 13:
        if (p[0] == '_' && memcmp(p, "__attribute__", 13) == 0) return TC_TK_KW_ATTRIBUTE;
        return TC_TK_IDENT;
    case 17:
        if (p[0] == '_' && memcmp(p, "__builtin_va_list", 17) == 0) return TC_TK_KW_VA_LIST;
        return TC_TK_IDENT;
    default:
        return TC_TK_IDENT;
    }
}

// Give an integer-literal token its C type (C11 6.4.4.1): the first type
// that can represent the value out of int, long, long long for a decimal
// literal; a hex/octal literal may also take the unsigned variants, a
// `u`-suffixed one only those, and an `l`/`ll` suffix skips the narrower
// types. tinyc models the type as a width (`is_i64`: at least `long`;
// `is_long_long`: 64-bit on every target) and a signedness. Signedness is
// decided for a 64-bit `long`; the parser adjusts a single-L literal that
// stays 32-bit on wasm32.
static void int_lit_type(TcTok *t, int n_l, bool has_u, bool decimal) {
    uint64_t v = (uint64_t)t->d.int_value;
    bool fits_i32 = v <= 0x7FFFFFFFull;
    bool fits_u32 = v <= 0xFFFFFFFFull;
    bool fits_i64 = v <= 0x7FFFFFFFFFFFFFFFull;
    // Can a 32-bit type (int / unsigned int, or a 32-bit long) hold it?
    bool fits32 = (has_u || !decimal) ? fits_u32 : fits_i32;
    bool is_i64 = n_l >= 1 || !fits32;
    bool is_ll  = n_l >= 2 || !fits32;
    bool is_u   = is_i64 ? (has_u || !fits_i64) : (has_u || !fits_i32);
    t->flags = (uint8_t)((is_i64 ? TC_TOK_F_I64 : 0u) |
                         (is_ll  ? TC_TOK_F_LONG_LONG : 0u) |
                         (is_u   ? TC_TOK_F_UNSIGNED : 0u));
}

static string *g_last_lex_files = NULL;
static size_t  g_last_lex_nfiles = 0;

void tinyc_set_last_lex_files(string *files, size_t n) {
    g_last_lex_files = files;
    g_last_lex_nfiles = n;
}

string *tinyc_last_lex_files(size_t *out_n) {
    if (out_n) *out_n = g_last_lex_nfiles;
    return g_last_lex_files;
}

static uint16_t lex_intern_file(Arena *arena, string **files, size_t *n, size_t *cap, string f) {
    for (size_t i = 0; i < *n; i++) {
        if ((*files)[i].size == f.size &&
            ((*files)[i].str == f.str || memcmp((*files)[i].str, f.str, f.size) == 0))
            return (uint16_t)(i + 1);
    }
    if (*n == *cap) {
        size_t ncap = *cap ? *cap * 2 : 16;
        string *nf = arena_new_array(arena, string, ncap);
        if (*n) memcpy(nf, *files, *n * sizeof(string));
        *files = nf;
        *cap = ncap;
    }
    (*files)[*n] = f;
    *n = *n + 1;
    return (uint16_t)(*n);
}

VecTcTok tinyc_lex(Arena *arena, string src) {
    lex_init_ctype();
    VecTcTok toks; VecTcTok_reserve(arena, &toks, src.size / 4 + 256);
    string *lex_files = NULL;
    size_t  n_lex_files = 0;
    size_t  lex_files_cap = 0;
    uint16_t cur_file_id = 0;
    size_t  file_stamp_pos = 0;
    const char *s = src.str;
    size_t sz = src.size;
    size_t i = 0;
    int line = 1;
    while (i < sz) {
        char c = s[i];
        uint8_t ct = g_lex_ctype[(unsigned char)c];
        if (ct & LEX_CT_WS) {
            i++;
            while (i < sz && (g_lex_ctype[(unsigned char)s[i]] & LEX_CT_WS)) i++;
            continue;
        }
        if (c == '\n') { line++; i++; continue; }
        if (ct & LEX_CT_ALPHA) {
            size_t j = i + 1;
            while (j < sz && (g_lex_ctype[(unsigned char)s[j]] & LEX_CT_ALNUM)) j++;
            string name = (string){.str = (char *)(s + i), .size = j - i};
            TcTok t = {0};
            t.kind = (uint8_t)tinyc_keyword_or_ident(name);
            t.text_str = name.str;
            t.d.size = name.size;
            t.line = line;
            VecTcTok_push_back(arena, &toks, t);
            i = j; continue;
        }
        if (c == '/' && i + 1 < src.size && src.str[i + 1] == '/') {
            while (i < src.size && src.str[i] != '\n') i++;
            continue;
        }
        if (c == '/' && i + 1 < src.size && src.str[i + 1] == '*') {
            i += 2;
            while (i + 1 < src.size && !(src.str[i] == '*' && src.str[i + 1] == '/')) {
                if (src.str[i] == '\n') line++;
                i++;
            }
            if (i + 1 < src.size) i += 2;
            else i = src.size;
            continue;
        }
        if (c == '#') {
            // #line N "file" — emitted by the preprocessor. Update line
            // tracking for subsequent diagnostics; we ignore the file name
            // since we report only line numbers.
            size_t j = i + 1;
            while (j < src.size && (src.str[j] == ' ' || src.str[j] == '\t')) j++;
            // Optional "line" keyword (not required by our PP, but accepted).
            if (j + 4 <= src.size && src.str[j] == 'l' && src.str[j + 1] == 'i'
                && src.str[j + 2] == 'n' && src.str[j + 3] == 'e'
                && (j + 4 == src.size || !is_alnum(src.str[j + 4]))) {
                j += 4;
                while (j < src.size && (src.str[j] == ' ' || src.str[j] == '\t')) j++;
            }
            if (j < src.size && is_digit(src.str[j])) {
                int new_line = 0;
                while (j < src.size && is_digit(src.str[j])) {
                    new_line = new_line * 10 + (src.str[j] - '0');
                    j++;
                }
                while (j < src.size && (src.str[j] == ' ' || src.str[j] == '\t')) j++;
                uint16_t new_fid = cur_file_id;
                if (j < src.size && src.str[j] == '"') {
                    j++;
                    size_t fstart = j;
                    while (j < src.size && src.str[j] != '"' && src.str[j] != '\n') {
                        if (src.str[j] == '\\' && j + 1 < src.size) j += 2;
                        else j++;
                    }
                    size_t raw_len = j - fstart;
                    char *fbuf = arena_new_array(arena, char, raw_len + 1);
                    size_t flen = 0;
                    for (size_t fk = fstart; fk < fstart + raw_len; fk++) {
                        if (src.str[fk] == '\\' && fk + 1 < fstart + raw_len) fk++;
                        fbuf[flen++] = src.str[fk];
                    }
                    fbuf[flen] = '\0';
                    new_fid = lex_intern_file(arena, &lex_files, &n_lex_files, &lex_files_cap,
                                              (string){.str = fbuf, .size = flen});
                }
                while (j < src.size && src.str[j] != '\n') j++;
                if (j < src.size) j++; // consume newline
                if (new_fid != cur_file_id) {
                    if (cur_file_id != 0) {
                        for (size_t k = file_stamp_pos; k < toks.size; k++)
                            toks.data[k].file_id = cur_file_id;
                    }
                    file_stamp_pos = toks.size;
                    cur_file_id = new_fid;
                }
                line = new_line;
                i = j;
                continue;
            }
            // Otherwise: stray '#'. Fall through to error.
        }
        if (c == '"') {
            // String literal. Scan raw length first so we allocate only the
            // string's size (not the entire remaining file).
            size_t end = i + 1;
            while (end < src.size && src.str[end] != '"') {
                if (src.str[end] == '\\' && end + 1 < src.size) end += 2;
                else end++;
            }
            size_t j = i + 1;
            char *buf = arena_new_array(arena, char, (end - j) + 1);
            size_t n = 0;
            while (j < src.size && src.str[j] != '"') {
                char ch = src.str[j++];
                if (ch == '\\' && j < src.size) {
                    char esc = src.str[j++];
                    switch (esc) {
                        case 'n': ch = '\n'; break;
                        case 't': ch = '\t'; break;
                        case 'r': ch = '\r'; break;
                        case 'v': ch = '\v'; break;
                        case 'f': ch = '\f'; break;
                        case 'a': ch = '\a'; break;
                        case 'b': ch = '\b'; break;
                        case '\\': ch = '\\'; break;
                        case '"': ch = '"'; break;
                        case '\'': ch = '\''; break;
                        case '?': ch = '?'; break;
                        case '0': case '1': case '2': case '3':
                        case '4': case '5': case '6': case '7': {
                            int v = esc - '0';
                            for (int oc = 0; oc < 2 && j < src.size &&
                                 src.str[j] >= '0' && src.str[j] <= '7'; oc++) {
                                v = v * 8 + (src.str[j++] - '0');
                            }
                            ch = (char)(v & 0xff);
                            break;
                        }
                        case 'x': {
                            // Hex escape: \xHH... — consume hex digits until
                            // a non-hex byte. Result is the low 8 bits.
                            int v = 0;
                            int nh = 0;
                            while (j < src.size) {
                                char h = src.str[j];
                                int dv;
                                if (h >= '0' && h <= '9') dv = h - '0';
                                else if (h >= 'a' && h <= 'f') dv = 10 + (h - 'a');
                                else if (h >= 'A' && h <= 'F') dv = 10 + (h - 'A');
                                else break;
                                v = v * 16 + dv;
                                j++; nh++;
                            }
                            if (nh == 0) {
                                println(str_lit("tinyc lex error at line {}: \\x with no hex digits"),
                                        (int64_t)line);
                            }
                            ch = (char)(v & 0xff);
                            break;
                        }
                        default:
                            println(str_lit("tinyc lex error at line {}: unknown escape '\\{}'"),
                                    (int64_t)line, str_substr(src, j - 1, 1));
                            ch = esc;
                    }
                } else if (ch == '\n') {
                    line++;
                }
                buf[n++] = ch;
            }
            if (j >= src.size) {
                println(str_lit("tinyc lex error at line {}: unterminated string literal"),
                        (int64_t)line);
            } else {
                j++; // consume closing "
            }
            buf[n++] = '\0';   // NUL terminator (counted in str.size)
            string text = (string){.str = buf, .size = n};
            TcTok t = {0};
            t.kind = TC_TK_STRING_LIT;
            t.text_str = text.str;
            t.d.size = text.size;
            t.line = line;
            VecTcTok_push_back(arena, &toks, t);
            i = j; continue;
        }
        if (c == '\'') {
            // Char literal — produce a TC_TK_INT_LIT carrying the byte value.
            size_t j = i + 1;
            int64_t v = 0;
            if (j < src.size && src.str[j] == '\\' && j + 1 < src.size) {
                j++;
                char esc = src.str[j++];
                switch (esc) {
                    case 'n': v = '\n'; break;
                    case 't': v = '\t'; break;
                    case 'r': v = '\r'; break;
                    case 'v': v = '\v'; break;
                    case 'f': v = '\f'; break;
                    case 'a': v = '\a'; break;
                    case 'b': v = '\b'; break;
                    case '\\': v = '\\'; break;
                    case '\'': v = '\''; break;
                    case '"': v = '"'; break;
                    case '?': v = '?'; break;
                    case '0': case '1': case '2': case '3':
                    case '4': case '5': case '6': case '7': {
                        int ov = esc - '0';
                        for (int oc = 0; oc < 2 && j < src.size &&
                             src.str[j] >= '0' && src.str[j] <= '7'; oc++) {
                            ov = ov * 8 + (src.str[j++] - '0');
                        }
                        v = (int64_t)(ov & 0xff);
                        break;
                    }
                    case 'x': {
                        // Hex char escape: '\xHH...' — accept as many hex
                        // digits as appear; result is the low 8 bits.
                        int hv = 0;
                        int nh = 0;
                        while (j < src.size) {
                            char h = src.str[j];
                            int dv;
                            if (h >= '0' && h <= '9') dv = h - '0';
                            else if (h >= 'a' && h <= 'f') dv = 10 + (h - 'a');
                            else if (h >= 'A' && h <= 'F') dv = 10 + (h - 'A');
                            else break;
                            hv = hv * 16 + dv;
                            j++; nh++;
                        }
                        if (nh == 0) {
                            println(str_lit("tinyc lex error at line {}: \\x with no hex digits"),
                                    (int64_t)line);
                        }
                        v = (int64_t)(hv & 0xff);
                        break;
                    }
                    default:
                        println(str_lit("tinyc lex error at line {}: unknown char escape"),
                                (int64_t)line);
                        v = esc;
                }
            } else if (j < src.size) {
                v = (unsigned char)src.str[j++];
            }
            if (j < src.size && src.str[j] == '\'') j++;
            else println(str_lit("tinyc lex error at line {}: unterminated char literal"),
                         (int64_t)line);
            // A char literal is an `int` holding the value of the byte
            // converted to (signed) char: '\xff' is -1.
            if (v > 127) v -= 256;
            TcTok t = {0}; t.kind = TC_TK_INT_LIT; t.d.int_value = v; t.line = line;
            VecTcTok_push_back(arena, &toks, t);
            i = j; continue;
        }
        if (is_digit(c)) {
            size_t j = i;
            int64_t v = 0;
            int n_l = 0;
            bool has_u = false;
            // Hex literal: 0x... / 0X...
            if (c == '0' && j + 1 < src.size &&
                (src.str[j + 1] == 'x' || src.str[j + 1] == 'X')) {
                j += 2;
                while (j < src.size) {
                    char h = src.str[j];
                    int dv;
                    if (h >= '0' && h <= '9') dv = h - '0';
                    else if (h >= 'a' && h <= 'f') dv = 10 + (h - 'a');
                    else if (h >= 'A' && h <= 'F') dv = 10 + (h - 'A');
                    else break;
                    v = v * 16 + dv;
                    j++;
                }
                TcTok t = {0}; t.kind = TC_TK_INT_LIT; t.d.int_value = v; t.line = line;
                // Optional integer-literal suffix: any mix of 'l'/'L' /
                // 'u'/'U' in any order (see below for the decimal path).
                for (int k = 0; k < 3 && j < src.size; k++) {
                    char c2 = src.str[j];
                    if (c2 == 'l' || c2 == 'L') { n_l++; j++; }
                    else if (c2 == 'u' || c2 == 'U') { has_u = true; j++; }
                    else break;
                }
                int_lit_type(&t, n_l, has_u, /*decimal=*/false);
                VecTcTok_push_back(arena, &toks, t);
                i = j; continue;
            }
            while (j < src.size && is_digit(src.str[j])) {
                v = v * 10 + (src.str[j] - '0');
                j++;
            }
            // Float literal: <digits>.<digits>[(e|E)[+-]?<digits>] OR
            // <digits>(e|E)[+-]?<digits>
            bool has_dot = (j < src.size && src.str[j] == '.' &&
                            j + 1 < src.size && is_digit(src.str[j + 1]));
            bool has_exp_only = (!has_dot && j < src.size &&
                                 (src.str[j] == 'e' || src.str[j] == 'E'));
            if (has_dot || has_exp_only) {
                // Build the literal from an integer mantissa and a single
                // decimal exponent so we incur as little rounding as possible.
                // The naive `f += digit * 0.1; scale *= 0.1` form accumulates
                // ulp-level error per digit and miscompiles literals like
                // 0.75 to 0x1.8000000000001p-1 (see tests/float_literal_0_75).
                //
                // Digits are accumulated into the mantissa for as long as it
                // stays below 2^63 (18 digits always fit, a 19th usually), so
                // it can neither wrap nor turn negative; further digits cannot
                // change a correctly rounded double in any but near-halfway
                // cases, so they only adjust the exponent. This covers both
                // long fractions such as 3.14159265358979323846 and integer
                // parts that do not fit in 64 bits such as
                // 18446744073709551616.0.
                const uint64_t mant_limit = 922337203685477579ULL; // (2^63-10)/10
                uint64_t mantissa = 0;
                int dropped_int = 0;  // integer-part digits not in `mantissa`
                int dec_places = 0;   // fraction digits kept in `mantissa`
                for (size_t k = i; k < j; k++) {
                    if (mantissa <= mant_limit) {
                        mantissa = mantissa * 10u + (uint64_t)(src.str[k] - '0');
                    } else {
                        dropped_int++;
                    }
                }
                if (has_dot) {
                    j++; // consume '.'
                    while (j < src.size && is_digit(src.str[j])) {
                        if (mantissa <= mant_limit) {
                            mantissa = mantissa * 10u + (uint64_t)(src.str[j] - '0');
                            dec_places++;
                        }
                        j++;
                    }
                }
                int exp_sign = 1;
                int exp = 0;
                if (j < src.size && (src.str[j] == 'e' || src.str[j] == 'E')) {
                    j++;
                    if (j < src.size && (src.str[j] == '+' || src.str[j] == '-')) {
                        if (src.str[j] == '-') exp_sign = -1;
                        j++;
                    }
                    while (j < src.size && is_digit(src.str[j])) {
                        exp = exp * 10 + (src.str[j] - '0');
                        j++;
                    }
                }
                int net_exp = (exp_sign < 0 ? -exp : exp) - dec_places + dropped_int;
                double f = lex_decimal_to_double(mantissa, net_exp);
                // Optional 'f'/'F' suffix marks the literal as TY_F32; otherwise
                // (per C99) it has TY_F64 (double).
                bool is_f64 = true;
                if (j < src.size && (src.str[j] == 'f' || src.str[j] == 'F')) { j++; is_f64 = false; }
                TcTok t = {0}; t.kind = TC_TK_FLOAT_LIT; t.d.float_value = f;
                t.flags = is_f64 ? TC_TOK_F_F64 : 0u; t.line = line;
                VecTcTok_push_back(arena, &toks, t);
                i = j; continue;
            }
            // Octal integer literal (C semantics): a digit run that begins
            // with '0', has more than one digit, is not a float (handled
            // above) and is not hex (0x, handled above) is base-8. The loop
            // above accumulated it in base-10, so re-scan [i, j) in base-8.
            // A lone '0' stays 0; a run containing '8'/'9' is left as-is
            // (decimal) rather than rejected, matching the lexer's lenient
            // style. `i` still points at the first digit; `j` is one past
            // the last (suffix parsing has not run yet).
            bool decimal = true;
            if (src.str[i] == '0' && j - i > 1) {
                int64_t ov = 0;
                bool octal_ok = true;
                for (size_t k = i; k < j; k++) {
                    char d = src.str[k];
                    if (d < '0' || d > '7') { octal_ok = false; break; }
                    ov = ov * 8 + (d - '0');
                }
                if (octal_ok) { v = ov; decimal = false; }
            }
            TcTok t = {0}; t.kind = TC_TK_INT_LIT; t.d.int_value = v; t.line = line;
            // Optional integer-literal suffix: any combination of 'l'/'L'
            // and 'u'/'U' in either order. So 'L', 'LL', 'UL', 'LU', 'ULL',
            // 'LLU', 'U' are all accepted. A single L denotes `long`
            // (target-dependent: 32-bit on wasm32, 64-bit on native); LL
            // always denotes `long long` (64-bit). The parser consults
            // `target_wasm32` to decide the size for a single-L literal.
            for (int k = 0; k < 3 && j < src.size; k++) {
                char c2 = src.str[j];
                if (c2 == 'l' || c2 == 'L') { n_l++; j++; }
                else if (c2 == 'u' || c2 == 'U') { has_u = true; j++; }
                else break;
            }
            int_lit_type(&t, n_l, has_u, decimal);
            VecTcTok_push_back(arena, &toks, t);
            i = j; continue;
        }
        // punctuation / multi-char via single switch on first byte
        TcTokKind k = TC_TK_EOF;
        size_t consumed = 1;
        char d = (i + 1 < sz) ? s[i + 1] : '\0';
        switch (c) {
            case '(': k = TC_TK_LPAREN; break;
            case ')': k = TC_TK_RPAREN; break;
            case '{': k = TC_TK_LBRACE; break;
            case '}': k = TC_TK_RBRACE; break;
            case '[': k = TC_TK_LBRACK; break;
            case ']': k = TC_TK_RBRACK; break;
            case ';': k = TC_TK_SEMI; break;
            case ',': k = TC_TK_COMMA; break;
            case '~': k = TC_TK_TILDE; break;
            case '?': k = TC_TK_QUESTION; break;
            case ':': k = TC_TK_COLON; break;
            case '.':
                if (d == '.' && i + 2 < sz && s[i + 2] == '.') { k = TC_TK_ELLIPSIS; consumed = 3; }
                else k = TC_TK_DOT;
                break;
            case '<':
                if (d == '<') {
                    if (i + 2 < sz && s[i + 2] == '=') { k = TC_TK_SHLEQ; consumed = 3; }
                    else { k = TC_TK_SHL; consumed = 2; }
                } else if (d == '=') { k = TC_TK_LE; consumed = 2; }
                else k = TC_TK_LT;
                break;
            case '>':
                if (d == '>') {
                    if (i + 2 < sz && s[i + 2] == '=') { k = TC_TK_SHREQ; consumed = 3; }
                    else { k = TC_TK_SHR; consumed = 2; }
                } else if (d == '=') { k = TC_TK_GE; consumed = 2; }
                else k = TC_TK_GT;
                break;
            case '=':
                if (d == '=') { k = TC_TK_EQEQ; consumed = 2; }
                else k = TC_TK_ASSIGN;
                break;
            case '!':
                if (d == '=') { k = TC_TK_NE; consumed = 2; }
                else k = TC_TK_BANG;
                break;
            case '&':
                if (d == '&') { k = TC_TK_AMPAMP; consumed = 2; }
                else if (d == '=') { k = TC_TK_AMPEQ; consumed = 2; }
                else k = TC_TK_AMP;
                break;
            case '|':
                if (d == '|') { k = TC_TK_PIPEPIPE; consumed = 2; }
                else if (d == '=') { k = TC_TK_PIPEEQ; consumed = 2; }
                else k = TC_TK_PIPE;
                break;
            case '-':
                if (d == '>') { k = TC_TK_ARROW; consumed = 2; }
                else if (d == '=') { k = TC_TK_MINUSEQ; consumed = 2; }
                else if (d == '-') { k = TC_TK_MINUSMINUS; consumed = 2; }
                else k = TC_TK_MINUS;
                break;
            case '+':
                if (d == '+') { k = TC_TK_PLUSPLUS; consumed = 2; }
                else if (d == '=') { k = TC_TK_PLUSEQ; consumed = 2; }
                else k = TC_TK_PLUS;
                break;
            case '*':
                if (d == '=') { k = TC_TK_STAREQ; consumed = 2; }
                else k = TC_TK_STAR;
                break;
            case '/':
                if (d == '=') { k = TC_TK_SLASHEQ; consumed = 2; }
                else k = TC_TK_SLASH;
                break;
            case '%':
                if (d == '=') { k = TC_TK_PERCENTEQ; consumed = 2; }
                else k = TC_TK_PERCENT;
                break;
            case '^':
                if (d == '=') { k = TC_TK_CARETEQ; consumed = 2; }
                else k = TC_TK_CARET;
                break;
            default: {
                println(str_lit("tinyc lex error at line {}: unexpected '{}'"),
                        (int64_t)line, str_substr(src, i, 1));
                TcTok eof = (TcTok){.kind = TC_TK_EOF, .line = line};
                VecTcTok_push_back(arena, &toks, eof);
                return toks;
            }
        }
        TcTok t = (TcTok){.kind = k, .line = line};
        VecTcTok_push_back(arena, &toks, t);
        i += consumed;
    }
    TcTok eof = (TcTok){.kind = TC_TK_EOF, .line = line};
    VecTcTok_push_back(arena, &toks, eof);
    if (cur_file_id != 0) {
        for (size_t k = file_stamp_pos; k < toks.size; k++)
            toks.data[k].file_id = cur_file_id;
    }
    tinyc_set_last_lex_files(lex_files, n_lex_files);
    return toks;
}

TcTok tinyc_lex_string(Arena *arena, string raw, int line) {
    const char *s = raw.str;
    size_t sz = raw.size;
    size_t j = (sz > 0 && s[0] == '"') ? 1 : 0;
    size_t end = (sz > j && s[sz - 1] == '"') ? sz - 1 : sz;
    char *buf = arena_new_array(arena, char, (end - j) + 1);
    size_t len = end - j;
    if (len == 0 || memchr(s + j, '\\', len) == NULL) {
        if (len) memcpy(buf, s + j, len);
        buf[len] = '\0';
        TcTok t = {0}; t.kind = TC_TK_STRING_LIT; t.text_str = buf; t.d.size = len + 1; t.line = line;
        return t;
    }
    size_t n = 0;
    while (j < end) {
        char ch = s[j++];
        if (ch == '\\' && j < end) {
            char esc = s[j++];
            switch (esc) {
                case 'n': ch = '\n'; break;
                case 't': ch = '\t'; break;
                case 'r': ch = '\r'; break;
                case 'v': ch = '\v'; break;
                case 'f': ch = '\f'; break;
                case 'a': ch = '\a'; break;
                case 'b': ch = '\b'; break;
                case '\\': ch = '\\'; break;
                case '"': ch = '"'; break;
                case '\'': ch = '\''; break;
                case '?': ch = '?'; break;
                case '0': case '1': case '2': case '3':
                case '4': case '5': case '6': case '7': {
                    int v = esc - '0';
                    for (int oc = 0; oc < 2 && j < end &&
                         s[j] >= '0' && s[j] <= '7'; oc++) {
                        v = v * 8 + (s[j++] - '0');
                    }
                    ch = (char)(v & 0xff);
                    break;
                }
                case 'x': {
                    int v = 0, nh = 0;
                    while (j < end) {
                        char h = s[j];
                        int dv;
                        if (h >= '0' && h <= '9') dv = h - '0';
                        else if (h >= 'a' && h <= 'f') dv = 10 + (h - 'a');
                        else if (h >= 'A' && h <= 'F') dv = 10 + (h - 'A');
                        else break;
                        v = v * 16 + dv;
                        j++; nh++;
                    }
                    if (nh == 0) {
                        println(str_lit("tinyc lex error at line {}: \\x with no hex digits"),
                                (int64_t)line);
                    }
                    ch = (char)(v & 0xff);
                    break;
                }
                default:
                    println(str_lit("tinyc lex error at line {}: unknown escape '\\{}'"),
                            (int64_t)line, str_substr(raw, j - 1, 1));
                    ch = esc;
            }
        }
        buf[n++] = ch;
    }
    buf[n++] = '\0';
    string text = (string){.str = buf, .size = n};
    TcTok t = {0}; t.kind = TC_TK_STRING_LIT; t.text_str = text.str; t.d.size = text.size; t.line = line;
    return t;
}

TcTok tinyc_lex_char(string raw, int line) {
    const char *s = raw.str;
    size_t sz = raw.size;
    size_t j = (sz > 0 && s[0] == '\'') ? 1 : 0;
    size_t end = (sz > j && s[sz - 1] == '\'') ? sz - 1 : sz;
    int64_t v = 0;
    if (j < end && s[j] == '\\' && j + 1 < end) {
        j++;
        char esc = s[j++];
        switch (esc) {
            case 'n': v = '\n'; break;
            case 't': v = '\t'; break;
            case 'r': v = '\r'; break;
            case 'v': v = '\v'; break;
            case 'f': v = '\f'; break;
            case 'a': v = '\a'; break;
            case 'b': v = '\b'; break;
            case '\\': v = '\\'; break;
            case '\'': v = '\''; break;
            case '"': v = '"'; break;
            case '?': v = '?'; break;
            case '0': case '1': case '2': case '3':
            case '4': case '5': case '6': case '7': {
                int ov = esc - '0';
                for (int oc = 0; oc < 2 && j < end &&
                     s[j] >= '0' && s[j] <= '7'; oc++) {
                    ov = ov * 8 + (s[j++] - '0');
                }
                v = (int64_t)(ov & 0xff);
                break;
            }
            case 'x': {
                int hv = 0, nh = 0;
                while (j < end) {
                    char h = s[j];
                    int dv;
                    if (h >= '0' && h <= '9') dv = h - '0';
                    else if (h >= 'a' && h <= 'f') dv = 10 + (h - 'a');
                    else if (h >= 'A' && h <= 'F') dv = 10 + (h - 'A');
                    else break;
                    hv = hv * 16 + dv;
                    j++; nh++;
                }
                if (nh == 0) {
                    println(str_lit("tinyc lex error at line {}: \\x with no hex digits"),
                            (int64_t)line);
                }
                v = (int64_t)(hv & 0xff);
                break;
            }
            default:
                println(str_lit("tinyc lex error at line {}: unknown char escape"),
                        (int64_t)line);
                v = esc;
        }
    } else if (j < end) {
        v = (unsigned char)s[j++];
    }
    if (v > 127) v -= 256;
    TcTok t = {0}; t.kind = TC_TK_INT_LIT; t.d.int_value = v; t.line = line;
    return t;
}

TcTok tinyc_lex_number(string text, int line) {
    const char *s = text.str;
    size_t sz = text.size;
    size_t j = 0;
    int64_t v = 0;
    int n_l = 0;
    bool has_u = false;
    if (sz >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        j = 2;
        while (j < sz) {
            char h = s[j];
            int dv;
            if (h >= '0' && h <= '9') dv = h - '0';
            else if (h >= 'a' && h <= 'f') dv = 10 + (h - 'a');
            else if (h >= 'A' && h <= 'F') dv = 10 + (h - 'A');
            else break;
            v = v * 16 + dv;
            j++;
        }
        TcTok t = {0}; t.kind = TC_TK_INT_LIT; t.d.int_value = v; t.line = line;
        for (int k = 0; k < 3 && j < sz; k++) {
            char c2 = s[j];
            if (c2 == 'l' || c2 == 'L') { n_l++; j++; }
            else if (c2 == 'u' || c2 == 'U') { has_u = true; j++; }
            else break;
        }
        int_lit_type(&t, n_l, has_u, false);
        return t;
    }
    while (j < sz && s[j] >= '0' && s[j] <= '9') {
        v = v * 10 + (s[j] - '0');
        j++;
    }
    bool has_dot = (j < sz && s[j] == '.' && j + 1 < sz && s[j + 1] >= '0' && s[j + 1] <= '9');
    bool has_exp_only = (!has_dot && j < sz && (s[j] == 'e' || s[j] == 'E'));
    if (has_dot || has_exp_only) {
        const uint64_t mant_limit = 922337203685477579ULL;
        uint64_t mantissa = 0;
        int dropped_int = 0;
        int dec_places = 0;
        for (size_t k = 0; k < j; k++) {
            if (mantissa <= mant_limit) mantissa = mantissa * 10u + (uint64_t)(s[k] - '0');
            else dropped_int++;
        }
        if (has_dot) {
            j++;
            while (j < sz && s[j] >= '0' && s[j] <= '9') {
                if (mantissa <= mant_limit) {
                    mantissa = mantissa * 10u + (uint64_t)(s[j] - '0');
                    dec_places++;
                }
                j++;
            }
        }
        int exp_sign = 1, exp = 0;
        if (j < sz && (s[j] == 'e' || s[j] == 'E')) {
            j++;
            if (j < sz && (s[j] == '+' || s[j] == '-')) {
                if (s[j] == '-') exp_sign = -1;
                j++;
            }
            while (j < sz && s[j] >= '0' && s[j] <= '9') {
                exp = exp * 10 + (s[j] - '0');
                j++;
            }
        }
        int net_exp = (exp_sign < 0 ? -exp : exp) - dec_places + dropped_int;
        double f = lex_decimal_to_double(mantissa, net_exp);
        bool is_f64 = true;
        if (j < sz && (s[j] == 'f' || s[j] == 'F')) { j++; is_f64 = false; }
        TcTok t = {0}; t.kind = TC_TK_FLOAT_LIT; t.d.float_value = f;
        t.flags = is_f64 ? TC_TOK_F_F64 : 0u; t.line = line;
        return t;
    }
    bool decimal = true;
    if (sz > 1 && s[0] == '0' && j > 1) {
        int64_t ov = 0;
        bool octal_ok = true;
        for (size_t k = 0; k < j; k++) {
            char d = s[k];
            if (d < '0' || d > '7') { octal_ok = false; break; }
            ov = ov * 8 + (d - '0');
        }
        if (octal_ok) { v = ov; decimal = false; }
    }
    TcTok t = {0}; t.kind = TC_TK_INT_LIT; t.d.int_value = v; t.line = line;
    for (int k = 0; k < 3 && j < sz; k++) {
        char c2 = s[j];
        if (c2 == 'l' || c2 == 'L') { n_l++; j++; }
        else if (c2 == 'u' || c2 == 'U') { has_u = true; j++; }
        else break;
    }
    int_lit_type(&t, n_l, has_u, decimal);
    return t;
}
