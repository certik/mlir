#pragma once

#include <string>


enum class TokenType {
    TK_EOF,
    TK_NEWLINE,
    TK_ERROR,
    TK_LPAREN,
    TK_RPAREN,
    TK_LBRACKET,
    TK_RBRACKET,
    TK_LBRACE,
    TK_RBRACE,
    TK_PLUS,
    TK_MINUS,
    TK_EQUAL,
    TK_COLON,
    TK_SEMICOLON,
    TK_SLASH,
    TK_PERCENT,
    TK_COMMA,
    TK_STAR,
    TK_VBAR,
    TK_COMMENT,
    TK_ARROW,
    TK_REAL,
    TK_STRING,
    KW_ABSTRACT,
    KW_ALL,
    KW_WRITE,
};

// Set the string to tokenize. The caller must ensure `str` will stay valid
// as long as `lex` is being called.
void tokenizer_set_string(const std::string &str, unsigned char *&cur);

// Get next token. Token type, first and last string index is returned
void tokenizer_get_next_token(
        const unsigned char *string_start,
        unsigned char *&cur,
        TokenType &token_type,
        uint64_t &first,
        uint64_t &last);
