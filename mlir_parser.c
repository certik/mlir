#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "mlir_parser.h"

string str_from_cstr_view(char *cstr) {
    return (string){.str=cstr, .size=strlen(cstr)-1};
}

string str_from_cstr_len_view(char *cstr, uint64_t size) {
    return (string){.str=cstr, .size=size};
}

char *str_to_cstr_copy(Arena *arena, string str) {
    char *cstr = arena_alloc_array(arena, char, str.size+1);
    memcpy(cstr, str.str, str.size);
    cstr[str.size] = '\0';
    return cstr;
}

bool str_eq(string a, string b) {
    if (a.size == b.size) {
        return (memcmp(a.str, b.str, a.size) == 0);
    } else {
        return false;
    }
}


bool parser_accept(Parser *parser, TokenType s) {
    if (parser->sym == s) {
        tokenizer_get_next_token(parser->input, &parser->cur, &parser->sym);
        return true;
    } else {
        return false;
    }
}

void parser_error(Arena *arena, string msg, uint64_t first, uint64_t last) {
    printf("Syntax error (%llu:%llu): %s\n", first, last,
            str_to_cstr_copy(arena, msg));
    exit(1);
}

void parser_expect(Parser *parser, TokenType s) {
    if (parser_accept(parser, s)) {
        return;
    } else {
        parser_error(parser->arena,
                str_lit("expect: unexpected symbol"),
                parser->first, parser->last);
    }
}

string parser_token_str(Parser *parser) {
    char *t = (char*) &parser->input[parser->first];
    return str_from_cstr_len_view(t, parser->last - parser->first+1);
}

void parser_expect_name(Parser *parser, string name) {
    if (parser->sym == TK_NAME && str_eq(parser_token_str(parser), name)) {
        parser_accept(parser, TK_NAME);
        return;
    } else {
        parser_error(parser->arena,
                str_lit("expect: unexpected symbol"),
                parser->first, parser->last);
    }
}


Operation* parse_module(Parser *parser) {
    parser_expect_name(parser, str_lit("module"));
    parser_expect(parser, TK_LBRACE);
    Operation *op = arena_alloc(parser->arena, Operation);
    return op;
    /*
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
    */
}
