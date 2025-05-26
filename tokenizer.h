#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TK_EOF,
    TK_NEWLINE,
    TK_WHITESPACE,
    TK_ERROR,
    TK_NAME,
    TK_LPAREN,
    TK_RPAREN,
    TK_LBRACKET,
    TK_RBRACKET,
    TK_LBRACE,
    TK_RBRACE,
    TK_LANGLE,
    TK_RANGLE,
    TK_EXCLAMATION,
    TK_DOLLAR,
    TK_HASH,
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
    TK_AT,
    TK_CARET,
    TK_DOT,
    TK_COMMENT,
    TK_ARROW,
    TK_REAL,
    TK_INTEGER,
    TK_STRING,
    TK_REGISTER,
    TK_TYPE_DIM,
    TK_FUNCTION_NAME,
    KW_ABSTRACT,
    KW_ALL,
    KW_WRITE,
} TokenType;

// Get the token in the `string` at the `position`. The function
// returns the `token_type` and the `position` one character after the token.
void tokenizer_get_next_token(const unsigned char *string,
        uint64_t *position, TokenType *token_type);

#ifdef __cplusplus
}
#endif
