#pragma once


struct TokenizerState
{
    unsigned char *cur;
    unsigned char *tok;
    unsigned char *string_start;
}

// Set the string to tokenize. The caller must ensure `str` will stay valid
// as long as `lex` is being called.
void tokenizer_set_string(TokenizerState &state, const std::string &str);

// Get next token. Token type, first and last string index is returned
void lex(TokenizerState &state, TokenType &token_type,
        uint64_t &first, uint64_t &last);
