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
    if (str_eq(s, str_lit("double")))   return TC_TK_KW_FLOAT;
    if (str_eq(s, str_lit("return")))   return TC_TK_KW_RETURN;
    if (str_eq(s, str_lit("if")))       return TC_TK_KW_IF;
    if (str_eq(s, str_lit("else")))     return TC_TK_KW_ELSE;
    if (str_eq(s, str_lit("while")))    return TC_TK_KW_WHILE;
    if (str_eq(s, str_lit("do")))       return TC_TK_KW_DO;
    if (str_eq(s, str_lit("for")))      return TC_TK_KW_FOR;
    if (str_eq(s, str_lit("break")))    return TC_TK_KW_BREAK;
    if (str_eq(s, str_lit("continue"))) return TC_TK_KW_CONTINUE;
    if (str_eq(s, str_lit("print")))    return TC_TK_KW_PRINT;
    if (str_eq(s, str_lit("struct")))   return TC_TK_KW_STRUCT;
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
    return TC_TK_IDENT;
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
                        case '\\': ch = '\\'; break;
                        case '"': ch = '"'; break;
                        case '\'': ch = '\''; break;
                        case '0': ch = '\0'; break;
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
                    case '\\': v = '\\'; break;
                    case '\'': v = '\''; break;
                    case '"': v = '"'; break;
                    case '0': v = '\0'; break;
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
            TcTok t = (TcTok){.kind = TC_TK_INT_LIT, .int_value = v, .line = line};
            VecTcTok_push_back(arena, &toks, t);
            i = j; continue;
        }
        if (is_digit(c)) {
            size_t j = i;
            int64_t v = 0;
            while (j < src.size && is_digit(src.str[j])) {
                v = v * 10 + (src.str[j] - '0');
                j++;
            }
            // Float literal: <digits>.<digits>[ (e|E)[+-]?<digits> ]
            if (j < src.size && src.str[j] == '.' &&
                j + 1 < src.size && is_digit(src.str[j + 1])) {
                double f = (double)v;
                j++; // consume '.'
                double scale = 0.1;
                while (j < src.size && is_digit(src.str[j])) {
                    f += (double)(src.str[j] - '0') * scale;
                    scale *= 0.1;
                    j++;
                }
                if (j < src.size && (src.str[j] == 'e' || src.str[j] == 'E')) {
                    j++;
                    int sign = 1;
                    if (j < src.size && (src.str[j] == '+' || src.str[j] == '-')) {
                        if (src.str[j] == '-') sign = -1;
                        j++;
                    }
                    int exp = 0;
                    while (j < src.size && is_digit(src.str[j])) {
                        exp = exp * 10 + (src.str[j] - '0');
                        j++;
                    }
                    double m = 1.0;
                    for (int k = 0; k < exp; k++) m *= 10.0;
                    if (sign < 0) f /= m; else f *= m;
                }
                // Optional 'f'/'F' suffix
                if (j < src.size && (src.str[j] == 'f' || src.str[j] == 'F')) j++;
                TcTok t = (TcTok){.kind = TC_TK_FLOAT_LIT, .float_value = f, .line = line};
                VecTcTok_push_back(arena, &toks, t);
                i = j; continue;
            }
            TcTok t = (TcTok){.kind = TC_TK_INT_LIT, .int_value = v, .line = line};
            // Optional integer-literal suffix: 'l', 'L', 'll', 'LL' marks
            // the literal as TY_I64. We do not distinguish signedness.
            if (j < src.size && (src.str[j] == 'l' || src.str[j] == 'L')) {
                t.is_i64 = true;
                j++;
                if (j < src.size && (src.str[j] == 'l' || src.str[j] == 'L')) j++;
            }
            // Also accept a trailing 'u'/'U' (silently — we don't track signedness).
            if (j < src.size && (src.str[j] == 'u' || src.str[j] == 'U')) j++;
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
