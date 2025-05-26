#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

string str_substr(string str, uint64_t min, uint64_t max) {
    return (string){.str=str.str+min, .size=max-min};
}


void parser_error(Arena *arena, string msg, uint64_t first, uint64_t last) {
    printf("Syntax error (%llu:%llu): %s\n", first, last,
            str_to_cstr_copy(arena, msg));
    exit(1);
}

bool parser_peek(Parser *parser, TokenType s) {
    return parser->sym == s;
}

bool parser_accept(Parser *parser, TokenType s) {
    if (parser_peek(parser, s)) {
        tokenizer_get_next_token(parser->input, &parser->cur, &parser->sym);
        return true;
    } else {
        return false;
    }
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
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), name)){
        parser_accept(parser, TK_NAME);
        return;
    } else {
        parser_error(parser->arena,
                str_lit("expect: unexpected symbol"),
                parser->first, parser->last);
    }
}

Operation* parse_operation(Parser *parser);

Operation* parse_module(Parser *parser) {
    parser_expect_name(parser, str_lit("module"));
    parser_expect(parser, TK_LBRACE);
    while (!parser_peek(parser, TK_RBRACE)) {
        parse_operation(parser);
    }
    parser_expect(parser, TK_RBRACE);
    Operation *op = arena_alloc(parser->arena, Operation);
    op->opcode = str_lit("module");
    return op;
}

Operation* parse_func_func(Parser *parser) {
    parser_expect_name(parser, str_lit("func.func"));
    //string func_name = parser_token_str(parser);
    parser_expect(parser, TK_FUNCTION_NAME);
    while (!parser_peek(parser, TK_LBRACE)) {
        tokenizer_get_next_token(parser->input, &parser->cur, &parser->sym);
    }
    parser_expect(parser, TK_LBRACE);
    while (!parser_peek(parser, TK_RBRACE)) {
        parse_operation(parser);
    }
    parser_expect(parser, TK_RBRACE);
    Operation *op = arena_alloc(parser->arena, Operation);
    op->opcode = str_lit("func.func");
    return op;
}

Operation* parse_operation(Parser *parser) {
    if (parser_peek(parser, TK_NAME_DOT_NAME)) {
        string op_name = parser_token_str(parser);
        if (str_eq(op_name, str_lit("func.func"))) {
            return parse_func_func(parser);
        } else if (str_eq(op_name, str_lit("scf.for"))) {
            parser_error(parser->arena,
                    str_lit("unsupported operation: scf.for"),
                    parser->first, parser->last);
//            return parse_scf_for(parser);
        } else {
            parser_error(parser->arena,
                    str_lit("unsupported operation"),
                    parser->first, parser->last);
        }
    } else if (parser_peek(parser, TK_REGISTER)) {
        parser_error(parser->arena,
                str_lit("unsupported operation reg"),
                parser->first, parser->last);
    } else {
        parser_error(parser->arena,
                str_lit("expected a name or a register"),
                parser->first, parser->last);
    }
    return NULL;
}
