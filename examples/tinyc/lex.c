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
    if (str_eq(s, str_lit("int")))    return TC_TK_KW_INT;
    if (str_eq(s, str_lit("return"))) return TC_TK_KW_RETURN;
    if (str_eq(s, str_lit("if")))     return TC_TK_KW_IF;
    if (str_eq(s, str_lit("else")))   return TC_TK_KW_ELSE;
    if (str_eq(s, str_lit("while")))  return TC_TK_KW_WHILE;
    if (str_eq(s, str_lit("print")))  return TC_TK_KW_PRINT;
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
        if (is_digit(c)) {
            size_t j = i;
            int64_t v = 0;
            while (j < src.size && is_digit(src.str[j])) {
                v = v * 10 + (src.str[j] - '0');
                j++;
            }
            TcTok t = (TcTok){.kind = TC_TK_INT_LIT, .int_value = v, .line = line};
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
        if (i + 1 < src.size) {
            char d = src.str[i + 1];
            if      (c == '<' && d == '=') { k = TC_TK_LE; consumed = 2; }
            else if (c == '>' && d == '=') { k = TC_TK_GE; consumed = 2; }
            else if (c == '=' && d == '=') { k = TC_TK_EQEQ; consumed = 2; }
            else if (c == '!' && d == '=') { k = TC_TK_NE; consumed = 2; }
            else if (c == '&' && d == '&') { k = TC_TK_AMPAMP; consumed = 2; }
            else if (c == '|' && d == '|') { k = TC_TK_PIPEPIPE; consumed = 2; }
        }
        if (k == TC_TK_EOF) {
            switch (c) {
                case '(': k = TC_TK_LPAREN; break;
                case ')': k = TC_TK_RPAREN; break;
                case '{': k = TC_TK_LBRACE; break;
                case '}': k = TC_TK_RBRACE; break;
                case ';': k = TC_TK_SEMI; break;
                case ',': k = TC_TK_COMMA; break;
                case '+': k = TC_TK_PLUS; break;
                case '-': k = TC_TK_MINUS; break;
                case '*': k = TC_TK_STAR; break;
                case '/': k = TC_TK_SLASH; break;
                case '%': k = TC_TK_PERCENT; break;
                case '<': k = TC_TK_LT; break;
                case '>': k = TC_TK_GT; break;
                case '=': k = TC_TK_ASSIGN; break;
                case '!': k = TC_TK_BANG; break;
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
