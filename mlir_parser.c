#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <base/string.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"

void get_newlines(Arena *arena, const string s, vector_int64_t *newlines) {
    for (int64_t pos=0; pos < s.size; pos++) {
        if (s.str[pos] == '\n') vector_int64_t_push_back(arena, newlines, pos);
    }
}

void parser_error(Parser *parser, string msg, uint64_t first, uint64_t last) {
    vector_int64_t newlines;
    vector_int64_t_reserve(parser->arena, &newlines, 16);
    get_newlines(parser->arena, str_from_cstr_view((char*)parser->input), &newlines);
    println(parser->arena, str_lit("Syntax error ({}:{}): {}"), first, last, msg);
    exit(1);
}

void parser_next_token(Parser *parser) {
    parser->first = parser->cur;
    tokenizer_get_next_token(parser->input, &parser->cur, &parser->sym);
    parser->last = parser->cur-1;
    while (parser->sym == TK_WHITESPACE) {
        parser->first = parser->cur;
        tokenizer_get_next_token(parser->input, &parser->cur, &parser->sym);
        parser->last = parser->cur-1;
    }
}

void parser_init(Arena *arena, Parser *parser, string text) {
    string text_null = str_concat(arena, text, str_lit("\0"));
    parser->arena = arena;
    parser->input = (unsigned char*) text_null.str;
    parser->cur = 0;
    parser_next_token(parser);
}

bool parser_peek(Parser *parser, TokenType s) {
    return parser->sym == s;
}

bool parser_accept(Parser *parser, TokenType s) {
    if (parser_peek(parser, s)) {
        parser_next_token(parser);
        return true;
    } else {
        return false;
    }
}


void parser_expect(Parser *parser, TokenType s) {
    if (parser_accept(parser, s)) {
        return;
    } else {
        parser_error(parser,
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
        parser_error(parser,
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
        parser_next_token(parser);
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
            parser_error(parser,
                    str_lit("unsupported operation: scf.for"),
                    parser->first, parser->last);
//            return parse_scf_for(parser);
        } else {
            parser_error(parser,
                    str_lit("unsupported operation"),
                    parser->first, parser->last);
        }
    } else if (parser_peek(parser, TK_REGISTER)) {
        parser_error(parser,
                str_lit("unsupported operation reg"),
                parser->first, parser->last);
    } else {
        parser_error(parser,
                str_lit("expected a name or a register"),
                parser->first, parser->last);
    }
    return NULL;
}
