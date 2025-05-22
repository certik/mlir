// Compile with:
// re2c -b tokenizer.re -o tokenizer.cpp

#include <limits>

#include "tokenizer.h"

void token_loc(
        const unsigned char *string_start,
        const unsigned char *tok,
        const unsigned char *cur,
        uint64_t &first,
        uint64_t &last)
{
    first = tok-string_start;
    last = cur-string_start-1;
}

void tokenizer_set_string(const std::string &str, unsigned char *&cur)
{
    // The input string must be NULL terminated, otherwise the tokenizer will
    // not detect the end of string. After C++11, the std::string is guaranteed
    // to end with \0, but we check this here just in case.
    //ASSERT(str[str.size()] == '\0');
    cur = (unsigned char *)(&str[0]);
}

#define RET(x) \
    token_loc(string_start, tok, cur, first, last); \
    token_type=TokenType::x; \
    return;

void tokenizer_get_next_token(
        const unsigned char *string_start,
        unsigned char *&cur,
        TokenType &token_type,
        uint64_t &first,
        uint64_t &last)
{
    while (true) {
        unsigned char *tok = cur;

        /*
        Re2c has excellent documentation at:

        https://re2c.org/manual/manual_c.html

        The first paragraph there explains the basics:

        * If multiple rules match, the longest match takes precedence
        * If multiple rules match the same string, the earlier rule takes
          precedence
        * Default rule `*` should always be defined, it has the lowest priority
          regardless of its place and matches any code unit
        * We use the "Sentinel character" method for end of input:
            * The end of the input text is denoted with a null character \x00
            * Thus the null character cannot be part of the input otherwise
            * There is one rule to match \x00 to end the parser
            * No other rule is allowed to match \x00, otherwise the re2c block
              would parse past the end of the string and segfaults
            * A special case of the previous point are negated character
              ranges, such as [^"\x00], where one must include \x00 in it to
              ensure this rule does not match \x00 (all other rules simply do
              not mention \x00)
            * See the "Handling the end of input" section in the re2c
              documentation for more info

        The re2c block interacts with the rest of the code via just one pointer
        variable `cur`. On entering the re2c block, the `cur` variable must
        point to the first character of the token to be tokenized by the block.
        The re2c block below then executes on its own until a rule is matched:
        the action in {} is then executed. In that action `cur` points to the
        first character of the next token.

        Before the re2c block we save the current `cur` into `tok`, so that we
        can use `tok` and `cur` in the action in {} to extract the token that
        corresponds to the rule that got matched:

        * `tok` points to the first character of the token
        * `cur-1` points to the last character of the token
        * `cur` points to the first character of the next token
        * `cur-tok` is the length of the token

        In the action, we do one of:

        * call `continue` which executes another cycle in the for loop (which
          will parse the next token); we use this to skip a token
        * call `return` which returns from this function; we return a token

        In both cases, `cur` points to first character of the next
        token, which becomes `tok` at the next iteration of the loop (either
        right away after `continue` or after the `lex` function is called again
        after `return`).

        See the manual for more details.
        */


        // These two variables are needed by the re2c block below internally,
        // initialization is not needed. One can think of them as local
        // variables of the re2c block.
        unsigned char *mar; //, *ctxmar;
        /*!re2c
            re2c:define:YYCURSOR = cur;
            re2c:define:YYMARKER = mar;
            re2c:define:YYCTXMARKER = ctxmar;
            re2c:yyfill:enable = 0;
            re2c:define:YYCTYPE = "unsigned char";

            end = "\x00";
            whitespace = [ \t\v\r]+;
            newline = "\n";
            digit = [0-9];
            char =  [a-zA-Z_];
            name = char (char | digit)*;
            significand = (digit+"."digit*) | ("."digit+);
            exp = [edED][-+]? digit+;
            integer = digit+;
            real = (significand exp?) | (digit+ exp);
            string = '"' [^"\x00]* '"';
            comment = "//" [^\n\x00]*;

            * { RET(TK_ERROR) }
            end { RET(TK_EOF); }
            whitespace { continue; }
            newline { RET(TK_NEWLINE) }

            // Keywords
            'abstract' { RET(KW_ABSTRACT) }
            'all' { RET(KW_ALL) }
            'write' { RET(KW_WRITE) }

            // Single character symbols
            "(" { RET(TK_LPAREN) }
            ")" { RET(TK_RPAREN) }
            "[" { RET(TK_LBRACKET) }
            "]" { RET(TK_RBRACKET) }
            "{" { RET(TK_LBRACE) }
            "}" { RET(TK_RBRACE) }
            "+" { RET(TK_PLUS) }
            "-" { RET(TK_MINUS) }
            "=" { RET(TK_EQUAL) }
            ":" { RET(TK_COLON) }
            ";" { RET(TK_SEMICOLON) }
            "/" { RET(TK_SLASH) }
            "%" { RET(TK_PERCENT) }
            "," { RET(TK_COMMA) }
            "*" { RET(TK_STAR) }
            "|" { RET(TK_VBAR) }

            // Multiple character symbols
            "//" { RET(TK_COMMENT) }
            "->" { RET(TK_ARROW) }

            name { RET(TK_NAME) }
            integer { RET(TK_INTEGER) }
            real { RET(TK_REAL) }
            string { RET(TK_STRING) }
        */
    }
}
