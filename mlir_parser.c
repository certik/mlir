#include <stdbool.h>

#include "mlir_parser.h"



bool parser_accept(Parser *parser, TokenType s) {
    if (parser->sym == s) {
        tokenizer_get_next_token(parser->input, &parser->cur, &parser->sym);
        return true;
    } else {
        return false;
    }
}

void parser_error(

void parser_expect(Parser *parser, TokenType s) {
    if (parser_accept(parser, s)) {
        return;
    } else {
        parser_error("expect: unexpected symbol", parser->first, parser->last);
    }
}


Operation* parse_module(Parser *parser) {
    expect_token(TokenType::TOKEN_IDENTIFIER, "module");
    expect_token(TokenType::TOKEN_PUNCTUATION, "{");
    symbol_table.push_scope();
    std::vector<Operation*> ops;
    while (current_token.value != "}") {
        ops.push_back(parse_operation());
    }
    expect_token(TokenType::TOKEN_PUNCTUATION, "}");
    symbol_table.pop_scope();
    auto block = new Block{{}, ops};
    auto region = new Region{{block}};
    return new Operation{"builtin.module", {}, {}, {}, {region}, {}};
}
