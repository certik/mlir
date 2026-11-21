// tinyC lexer — hand-rolled, character-by-character. No re2c, no libc.

#include "tinyc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/io.h>
#include <base/string.h>

static bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static bool is_digit(char c) { return c >= '0' && c <= '9'; }
static bool is_alnum(char c) { return is_alpha(c) || is_digit(c); }

static TcTokKind keyword_or_ident(string s) {
    if (str_eq(s, str_lit("int")))      return TC_TK_KW_INT;
    if (str_eq(s, str_lit("float")))    return TC_TK_KW_FLOAT;
    if (str_eq(s, str_lit("double")))   return TC_TK_KW_DOUBLE;
    if (str_eq(s, str_lit("return")))   return TC_TK_KW_RETURN;
    if (str_eq(s, str_lit("if")))       return TC_TK_KW_IF;
    if (str_eq(s, str_lit("else")))     return TC_TK_KW_ELSE;
    if (str_eq(s, str_lit("while")))    return TC_TK_KW_WHILE;
    if (str_eq(s, str_lit("do")))       return TC_TK_KW_DO;
    if (str_eq(s, str_lit("for")))      return TC_TK_KW_FOR;
    if (str_eq(s, str_lit("break")))    return TC_TK_KW_BREAK;
    if (str_eq(s, str_lit("continue"))) return TC_TK_KW_CONTINUE;
    if (str_eq(s, str_lit("struct")))   return TC_TK_KW_STRUCT;
    if (str_eq(s, str_lit("union")))    return TC_TK_KW_UNION;
    if (str_eq(s, str_lit("null")))     return TC_TK_KW_NULL;
    if (str_eq(s, str_lit("sizeof")))   return TC_TK_KW_SIZEOF;
    if (str_eq(s, str_lit("char")))     return TC_TK_KW_CHAR;
    if (str_eq(s, str_lit("typedef")))  return TC_TK_KW_TYPEDEF;
    if (str_eq(s, str_lit("extern")))   return TC_TK_KW_EXTERN;
    if (str_eq(s, str_lit("switch")))   return TC_TK_KW_SWITCH;
    if (str_eq(s, str_lit("case")))     return TC_TK_KW_CASE;
    if (str_eq(s, str_lit("default")))  return TC_TK_KW_DEFAULT;
    if (str_eq(s, str_lit("enum")))     return TC_TK_KW_ENUM;
    if (str_eq(s, str_lit("const")))    return TC_TK_KW_CONST;
    if (str_eq(s, str_lit("void")))     return TC_TK_KW_VOID;
    if (str_eq(s, str_lit("static")))   return TC_TK_KW_STATIC;
    if (str_eq(s, str_lit("inline")))   return TC_TK_KW_INLINE;
    if (str_eq(s, str_lit("long")))     return TC_TK_KW_LONG;
    if (str_eq(s, str_lit("signed")))   return TC_TK_KW_SIGNED;
    if (str_eq(s, str_lit("unsigned"))) return TC_TK_KW_UNSIGNED;
    if (str_eq(s, str_lit("short")))    return TC_TK_KW_SHORT;
    if (str_eq(s, str_lit("_Bool")))    return TC_TK_KW_BOOL;
    if (str_eq(s, str_lit("bool")))     return TC_TK_KW_BOOL;
    if (str_eq(s, str_lit("_Generic")))    return TC_TK_KW_GENERIC;
    if (str_eq(s, str_lit("va_list"))) return TC_TK_KW_VA_LIST;
    if (str_eq(s, str_lit("__builtin_va_list"))) return TC_TK_KW_VA_LIST;
    if (str_eq(s, str_lit("goto")))     return TC_TK_KW_GOTO;
    if (str_eq(s, str_lit("__attribute__"))) return TC_TK_KW_ATTRIBUTE;
    return TC_TK_IDENT;
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
    uint64_t v = (uint64_t)t->int_value;
    bool fits_i32 = v <= 0x7FFFFFFFull;
    bool fits_u32 = v <= 0xFFFFFFFFull;
    bool fits_i64 = v <= 0x7FFFFFFFFFFFFFFFull;
    // Can a 32-bit type (int / unsigned int, or a 32-bit long) hold it?
    bool fits32 = (has_u || !decimal) ? fits_u32 : fits_i32;
    t->is_i64 = n_l >= 1 || !fits32;
    t->is_long_long = n_l >= 2 || !fits32;
    if (t->is_i64) t->is_unsigned = has_u || !fits_i64;
    else           t->is_unsigned = has_u || !fits_i32;
}

VecTcTok tinyc_lex(Arena *arena, string src) {
    VecTcTok toks; VecTcTok_reserve(arena, &toks, 64);
    size_t i = 0;
    int line = 1;
    while (i < src.size) {
        char c = src.str[i];
        if (c == ' ' || c == '\t' || c == '\r') { i++; continue; }
        if (c == '\n') { line++; i++; continue; }
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
                // Skip rest of line (optional "file" string and any
                // trailing junk).
                while (j < src.size && src.str[j] != '\n') j++;
                if (j < src.size) j++; // consume newline
                line = new_line;
                i = j;
                continue;
            }
            // Otherwise: stray '#'. Fall through to error.
        }
        if (c == '"') {
            // String literal. Decode escapes into an arena buffer; append a
            // trailing NUL so callers can pass the bytes to printf/fputs.
            size_t j = i + 1;
            char *buf = arena_new_array(arena, char, src.size - i + 1);
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
                        case '0': ch = '\0'; break;
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
            TcTok t = (TcTok){.kind = TC_TK_STRING_LIT, .text = text, .line = line};
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
                    case '0': v = '\0'; break;
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
            TcTok t = (TcTok){.kind = TC_TK_INT_LIT, .int_value = v, .line = line};
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
                TcTok t = (TcTok){.kind = TC_TK_INT_LIT, .int_value = v, .line = line};
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
                // mantissa < 2^63: its top 52 bits and its low 11 bits each
                // convert to double exactly, so `f + lo` is the mantissa
                // without any rounding. (Only int -> double conversions of
                // non-negative values are used: tinyc lowers double -> int64
                // through i32, and this file has to self-host.)
                double f = (double)(int64_t)(mantissa >> 11) * 2048.0;
                double lo = (double)(int64_t)(mantissa & 2047u);
                if (net_exp > 0) {
                    // Multiply by 10^net_exp using exact integer powers
                    // up to 10^22 (which is exactly representable in f64),
                    // chunking larger exponents.
                    f += lo;
                    while (net_exp > 0) {
                        int chunk = net_exp > 22 ? 22 : net_exp;
                        double p = 1.0;
                        for (int k = 0; k < chunk; k++) p *= 10.0;
                        f *= p;
                        net_exp -= chunk;
                    }
                } else if (net_exp < 0 && net_exp >= -22) {
                    // mantissa / 10^n with 10^n exact. A mantissa above 2^53
                    // is not exactly representable, so a plain
                    // `(double)mantissa / p` rounds twice and is off by one
                    // ulp for about one literal in ten with 17+ digits (M_PI
                    // among them). Carry the low mantissa bits `lo` and the
                    // exact division remainder (Dekker two-product; every
                    // partial product below is exact, so FMA contraction
                    // cannot change the result) into one correction term.
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
                // Optional 'f'/'F' suffix marks the literal as TY_F32; otherwise
                // (per C99) it has TY_F64 (double).
                bool is_f64 = true;
                if (j < src.size && (src.str[j] == 'f' || src.str[j] == 'F')) { j++; is_f64 = false; }
                TcTok t = (TcTok){.kind = TC_TK_FLOAT_LIT, .float_value = f, .is_f64 = is_f64, .line = line};
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
            TcTok t = (TcTok){.kind = TC_TK_INT_LIT, .int_value = v, .line = line};
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
        if (is_alpha(c)) {
            size_t j = i;
            while (j < src.size && is_alnum(src.str[j])) j++;
            string name = str_substr(src, i, j - i);
            TcTok t = (TcTok){.kind = keyword_or_ident(name), .text = name, .line = line};
            VecTcTok_push_back(arena, &toks, t);
            i = j; continue;
        }
        // punctuation / multi-char
        TcTokKind k = TC_TK_EOF;
        size_t consumed = 1;
        // 3-char tokens first.
        if (i + 2 < src.size) {
            char d = src.str[i + 1], e2 = src.str[i + 2];
            if      (c == '<' && d == '<' && e2 == '=') { k = TC_TK_SHLEQ; consumed = 3; }
            else if (c == '>' && d == '>' && e2 == '=') { k = TC_TK_SHREQ; consumed = 3; }
            else if (c == '.' && d == '.' && e2 == '.') { k = TC_TK_ELLIPSIS; consumed = 3; }
        }
        if (k == TC_TK_EOF && i + 1 < src.size) {
            char d = src.str[i + 1];
            if      (c == '<' && d == '=') { k = TC_TK_LE; consumed = 2; }
            else if (c == '>' && d == '=') { k = TC_TK_GE; consumed = 2; }
            else if (c == '=' && d == '=') { k = TC_TK_EQEQ; consumed = 2; }
            else if (c == '!' && d == '=') { k = TC_TK_NE; consumed = 2; }
            else if (c == '&' && d == '&') { k = TC_TK_AMPAMP; consumed = 2; }
            else if (c == '|' && d == '|') { k = TC_TK_PIPEPIPE; consumed = 2; }
            else if (c == '-' && d == '>') { k = TC_TK_ARROW; consumed = 2; }
            else if (c == '+' && d == '=') { k = TC_TK_PLUSEQ; consumed = 2; }
            else if (c == '-' && d == '=') { k = TC_TK_MINUSEQ; consumed = 2; }
            else if (c == '*' && d == '=') { k = TC_TK_STAREQ; consumed = 2; }
            else if (c == '/' && d == '=') { k = TC_TK_SLASHEQ; consumed = 2; }
            else if (c == '%' && d == '=') { k = TC_TK_PERCENTEQ; consumed = 2; }
            else if (c == '&' && d == '=') { k = TC_TK_AMPEQ; consumed = 2; }
            else if (c == '|' && d == '=') { k = TC_TK_PIPEEQ; consumed = 2; }
            else if (c == '^' && d == '=') { k = TC_TK_CARETEQ; consumed = 2; }
            else if (c == '<' && d == '<') { k = TC_TK_SHL; consumed = 2; }
            else if (c == '>' && d == '>') { k = TC_TK_SHR; consumed = 2; }
            else if (c == '+' && d == '+') { k = TC_TK_PLUSPLUS; consumed = 2; }
            else if (c == '-' && d == '-') { k = TC_TK_MINUSMINUS; consumed = 2; }
        }
        if (k == TC_TK_EOF) {
            switch (c) {
                case '(': k = TC_TK_LPAREN; break;
                case ')': k = TC_TK_RPAREN; break;
                case '{': k = TC_TK_LBRACE; break;
                case '}': k = TC_TK_RBRACE; break;
                case '[': k = TC_TK_LBRACK; break;
                case ']': k = TC_TK_RBRACK; break;
                case ';': k = TC_TK_SEMI; break;
                case ',': k = TC_TK_COMMA; break;
                case '.': k = TC_TK_DOT; break;
                case '+': k = TC_TK_PLUS; break;
                case '-': k = TC_TK_MINUS; break;
                case '*': k = TC_TK_STAR; break;
                case '/': k = TC_TK_SLASH; break;
                case '%': k = TC_TK_PERCENT; break;
                case '<': k = TC_TK_LT; break;
                case '>': k = TC_TK_GT; break;
                case '=': k = TC_TK_ASSIGN; break;
                case '!': k = TC_TK_BANG; break;
                case '&': k = TC_TK_AMP; break;
                case '|': k = TC_TK_PIPE; break;
                case '^': k = TC_TK_CARET; break;
                case '~': k = TC_TK_TILDE; break;
                case '?': k = TC_TK_QUESTION; break;
                case ':': k = TC_TK_COLON; break;
                default: {
                    println(str_lit("tinyc lex error at line {}: unexpected '{}'"),
                            (int64_t)line, str_substr(src, i, 1));
                    TcTok eof = (TcTok){.kind = TC_TK_EOF, .line = line};
                    VecTcTok_push_back(arena, &toks, eof);
                    return toks;
                }
            }
        }
        TcTok t = (TcTok){.kind = k, .line = line};
        VecTcTok_push_back(arena, &toks, t);
        i += consumed;
    }
    TcTok eof = (TcTok){.kind = TC_TK_EOF, .line = line};
    VecTcTok_push_back(arena, &toks, eof);
    return toks;
}
