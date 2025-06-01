#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"

string tokentype_to_string(TokenType tt) {
    switch (tt) {
#define X(token) case token: return str_lit(#token);
        LIST_OF_TOKENS
#undef X
        default: abort();
    }
}


void get_newlines(Arena *arena, const string s, vector_i64 *newlines) {
    for (int64_t pos=0; pos < s.size; pos++) {
        if (s.str[pos] == '\n') vector_i64_push_back(arena, newlines, pos);
    }
    // Append end of file if not already present (doesn't end with \n)
    if (!(s.size > 0 && s.str[s.size-1] == '\n')) {
        vector_i64_push_back(arena, newlines, s.size);
    }
    assert(newlines->size > 0);
    assert(newlines->data[newlines->size-1] >= s.size-1);
}

void linear_to_line_column(const vector_i64 newlines, uint64_t first,
        int64_t *start_of_line,
        int64_t *end_of_line,
        int64_t *line_first,
        int64_t *column_first) {
    // Find the line index (number of newlines before 'first')
    int64_t line_idx = 0;
    while (line_idx < newlines.size && newlines.data[line_idx] < first) {
        line_idx++;
    }
    int64_t line_number = line_idx + 1;
    *start_of_line = (line_idx == 0) ? 0 : newlines.data[line_idx - 1] + 1;
    assert(line_idx < newlines.size);
    *end_of_line = newlines.data[line_idx];
    *line_first = line_number;
    *column_first = first - *start_of_line + 1;
    assert(first >= *start_of_line);
    assert(first <= *end_of_line);
}

void parser_error(Parser *parser, string msg, uint64_t first, uint64_t last) {
    assert(first <= last);
    string s = str_from_cstr_view((char*)parser->input);
    vector_i64 newlines;
    vector_i64_reserve(parser->arena, &newlines, 16);
    get_newlines(parser->arena, s, &newlines);

    int64_t start_of_line, end_of_line, line_first, column_first;
    linear_to_line_column(newlines, first, &start_of_line, &end_of_line,
            &line_first, &column_first);

    bool debug = false;
    if (debug) {
        println(parser->arena, str_lit("{}"), newlines);
        println(parser->arena,
                str_lit("first: {}, last: {}, line_first: {}, column_first: {}, start_of_line: {}, end_of_line: {}"),
                first, last, line_first, column_first, start_of_line, end_of_line);
    }

    // Extract the line as a string
    string line = { .str = s.str + start_of_line, .size = end_of_line - start_of_line };

    // Create the caret string (spaces followed by carets)
    int64_t token_length = last - first + 1; // Assuming 'last' is inclusive
    assert(first >= start_of_line);
    char* caret_buf = arena_alloc_array(parser->arena, char, first - start_of_line + token_length);
    for (int64_t i = 0; i < first - start_of_line; i++) {
        caret_buf[i] = ' ';
    }
    for (int64_t i = 0; i < token_length; i++) {
        caret_buf[first - start_of_line + i] = '^';
    }
    string caret_str = { .str = caret_buf, .size = first - start_of_line + token_length };

    // Print the error message, line, and caret string
    println(parser->arena, str_lit("Syntax error ({}:{}): {}"),
            line_first, column_first, msg);
    println(parser->arena, str_lit("{}"), line);
    println(parser->arena, caret_str);

    exit(1);
}

void parser_next_token(Parser *parser) {
    do {
        parser->first = parser->cur;
        tokenizer_get_next_token(parser->input, &parser->cur, &parser->sym);
        parser->last = parser->cur-1;
    } while (parser->sym == TK_WHITESPACE || parser->sym == TK_COMMENT);
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

void parser_expect(Parser *parser, TokenType s) {
    if (parser_peek(parser, s)) {
        parser_next_token(parser);
    } else {
        parser_error(parser,
            format(parser->arena,
                str_lit("Expected {}, got {}"),
                tokentype_to_string(s),
                tokentype_to_string(parser->sym)
            ), parser->first, parser->last);
    }
}

string parser_token_str(Parser *parser) {
    char *t = (char*) &parser->input[parser->first];
    return str_from_cstr_len_view(t, parser->last - parser->first+1);
}

void parser_expect_opname(Parser *parser, string name) {
    if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
        if (str_eq(parser_token_str(parser), name)) {
            parser_next_token(parser);
            return;
        }
    }
    parser_error(parser,
        format(parser->arena,
            str_lit("Expected TK_NAME or TK_NAME_DOT_NAME '{}', got {}"),
            name,
            tokentype_to_string(parser->sym)
        ), parser->first, parser->last);
}

Operation* parse_operation(Parser *parser);

Block* parse_block(Parser *parser) {
    if (parser_peek(parser, TK_CARET_NAME)) {
        parser_expect(parser, TK_CARET_NAME);
        while (!(parser_peek(parser, TK_NEWLINE))) {
            parser_next_token(parser);
        }
        parser_expect(parser, TK_NEWLINE);
    }
    VecOperation operations;
    VecOperation_reserve(parser->arena, &operations, 16);
    while (! (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_CARET_NAME))) {
        Operation *op = parse_operation(parser);
        VecOperation_push_back(parser->arena, &operations, op);
        parser_expect(parser, TK_NEWLINE);

        // Skip empty lines
        while (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        }
    }

    Block *block = arena_alloc(parser->arena, Block);
    block->operations = operations.data;
    block->n_operations = operations.size;

    return block;
}

// Parses a region from { to } inclusive
Region* parse_region(Parser *parser) {
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);
    vector_i64 blocks;
    vector_i64_reserve(parser->arena, &blocks, 8);
    while (!parser_peek(parser, TK_RBRACE)) {
        Block *block = parse_block(parser);
        vector_i64_push_back(parser->arena, &blocks, (int64_t)(block));
    }
    parser_expect(parser, TK_RBRACE);

    Region *region = arena_alloc(parser->arena, Region);
    region->blocks = (Block **)blocks.data;
    region->n_blocks = blocks.size;

    return region;
}

Operation* parse_module(Parser *parser) {
    Operation *op = parse_operation(parser);
    if (!str_eq(op->opname, str_lit("module"))) {
        parser_error(parser, str_lit("The top level operation should be a module"), 0, 0);
    }
    return op;
}

// parse loc()
void parse_loc(Parser *parser) {
    parser_expect(parser, TK_NAME);
    parser_expect(parser, TK_LPAREN);
    while (!(parser_peek(parser, TK_RPAREN))) {
        parser_next_token(parser);
    }
    parser_expect(parser, TK_RPAREN);
}

void parse_tt_func(Parser *parser, Operation *op) {
    //parser_expect(parser, TK_FUNCTION_NAME);
    while (!parser_peek(parser, TK_LBRACE_END)) {
        parser_next_token(parser);
    }
    Region *region = parse_region(parser);

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            parse_loc(parser);
        }
    }

    Region **regions = arena_alloc(parser->arena, Region*);
    regions[0] = region;
    op->regions = regions;
    op->n_regions = 1;
}

void parse_scf_if(Parser *parser, Operation *op) {
    while (!parser_peek(parser, TK_LBRACE_END)) {
        parser_next_token(parser);
    }
    int n_regions = 1;
    Region *region1 = parse_region(parser);
    Region *region2 = NULL;

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("else"))) {
            parser_expect(parser, TK_NAME);
            region2 = parse_region(parser);
            n_regions++;
        }
    }

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            parse_loc(parser);
        }
    }

    Region **regions = arena_alloc_array(parser->arena, Region*, n_regions);
    regions[0] = region1;
    if (region2) {
        regions[1] = region2;
    }
    op->regions = regions;
    op->n_regions = n_regions;
}

void parse_scf_for(Parser *parser, Operation *op) {
    while (!parser_peek(parser, TK_LBRACE_END)) {
        parser_next_token(parser);
    }
    Region *region1 = parse_region(parser);

    while (!parser_peek(parser, TK_NEWLINE)) {
        parser_next_token(parser);
    }

    Region **regions = arena_alloc(parser->arena, Region*);
    regions[0] = region1;
    op->regions = regions;
    op->n_regions = 1;
}

void parse_scf_while(Parser *parser, Operation *op) {
    while (!parser_peek(parser, TK_LBRACE_END)) {
        parser_next_token(parser);
    }
    int n_regions = 1;
    Region *region1 = parse_region(parser);
    Region *region2 = NULL;

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("do"))) {
            parser_expect(parser, TK_NAME);
            region2 = parse_region(parser);
            n_regions++;
        }
    }

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            parse_loc(parser);
        }
    }

    Region **regions = arena_alloc_array(parser->arena, Region*, n_regions);
    regions[0] = region1;
    if (region2) {
        regions[1] = region2;
    }
    op->regions = regions;
    op->n_regions = n_regions;
}

Operation* parse_operation(Parser *parser) {
    Operation *op = arena_alloc(parser->arena, Operation);
    op->regions = NULL;
    op->n_regions = 0;
    op->n_result_types = 0;
    op->opname = str_lit("");

    // Skip empty lines and attributes
    while (
        parser_peek(parser, TK_NEWLINE) ||
        parser_peek(parser, TK_HASH_NAME)
            ) {
        if (parser_peek(parser, TK_HASH_NAME)) {
            while (!parser_peek(parser, TK_NEWLINE)) {
                parser_next_token(parser);
            }
            parser_expect(parser, TK_NEWLINE);
        } else if (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        } else {
            // Shouldn't happen
            abort();
        }
    }

    // Parse return registers if any
    if (parser_peek(parser, TK_REGISTER)) {
        //string reg = parser_token_str(parser);
        op->n_result_types = 1;
        parser_expect(parser, TK_REGISTER);
        if (parser_peek(parser, TK_COLON)) {
            parser_expect(parser, TK_COLON);
            parser_expect(parser, TK_INTEGER);
        }
        parser_expect(parser, TK_EQUAL);
    }

    // Parse operation name
    if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
        op->opname = parser_token_str(parser);
        parser_next_token(parser);
    } else if (parser_peek(parser, TK_STRING)) {
        op->opname = parser_token_str(parser);
        op->opname = str_substr(op->opname, 1, op->opname.size-2);
        parser_expect(parser, TK_STRING);
    } else {
        parser_error(parser,
            format(parser->arena,
                str_lit("Expected operation name (TK_NAME, TK_NAME_DOT_NAME or TK_STRING), got {}"),
                tokentype_to_string(parser->sym)
            ), parser->first, parser->last);
    }

    // First we handle specific opnames with special parsing rules
    if (str_eq(op->opname, str_lit("tt.func"))) {
        parse_tt_func(parser, op);
    } else if (str_eq(op->opname, str_lit("scf.if"))) {
        parse_scf_if(parser, op);
    } else if (str_eq(op->opname, str_lit("scf.for"))) {
        parse_scf_for(parser, op);
    } else if (str_eq(op->opname, str_lit("scf.while"))) {
        parse_scf_while(parser, op);
    } else {
        // Then we parse a general opname


    // Parse details
    /*
        parser_expect(parser, TK_LPAREN);
        if (parser_peek(parser, TK_REGISTER)) {
            //string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
        }
        parser_expect(parser, TK_RPAREN);
        if (parser_peek(parser, TK_LBRACE)) {
            parser_expect(parser, TK_LBRACE);
            parser_expect(parser, TK_NAME);
            parser_expect(parser, TK_EQUAL);
            parser_expect(parser, TK_INTEGER);
            parser_expect(parser, TK_RBRACE);
        }
        parser_expect(parser, TK_COLON);
        parser_expect(parser, TK_LPAREN);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME);
        }
        parser_expect(parser, TK_RPAREN);
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME);
        } else {
            parser_expect(parser, TK_LPAREN);
            parser_expect(parser, TK_RPAREN);
        }
    */

        // Parse regions (if any), for now we assume 0 or 1 regions
        while (!(
                parser_peek(parser, TK_LBRACE_END)
                || parser_peek(parser, TK_NEWLINE)
                || parser_peek(parser, TK_LPAREN_BRACE)
                )) {
            parser_next_token(parser);
        }
        bool lparen_brace = false;
        if (parser_peek(parser, TK_LPAREN_BRACE)) {
            lparen_brace = true;
            parser_next_token(parser);
        }
        if (parser_peek(parser, TK_LBRACE_END)) {
            Region *region = parse_region(parser);

            // TODO: for now we assume one region
            Region **regions = arena_alloc(parser->arena, Region*);
            regions[0] = region;
            op->regions = regions;
            op->n_regions = 1;
        }
        if (lparen_brace) {
            parser_expect(parser, TK_RPAREN);
            while (!parser_peek(parser, TK_NEWLINE)) {
                parser_next_token(parser);
            }
        }
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);
            while (!parser_peek(parser, TK_NEWLINE)) {
                parser_next_token(parser);
            }
        }
        if (parser_peek(parser, TK_NAME)) {
            if (str_eq(parser_token_str(parser), str_lit("loc"))) {
                parse_loc(parser);
            }
        }
    }

    return op;
}
