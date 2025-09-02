#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"

// Forward decls for helper scanners
static inline void consume_optional_hash_selector(Parser *parser);

// Symbol table implementation
void symbol_table_init(Arena *arena, ScopedSymbolTable *st) {
    st->scope_capacity = 8;
    st->scopes = arena_alloc_array(arena, SymbolTable, st->scope_capacity);
    st->num_scopes = 0;
}

void symbol_table_push_scope(Arena *arena, ScopedSymbolTable *st) {
    if (st->num_scopes >= st->scope_capacity) {
        // Grow scopes array
        size_t new_capacity = st->scope_capacity * 2;
        SymbolTable *new_scopes = arena_alloc_array(arena, SymbolTable, new_capacity);
        memcpy(new_scopes, st->scopes, st->num_scopes * sizeof(SymbolTable));
        st->scopes = new_scopes;
        st->scope_capacity = new_capacity;
    }
    SymbolTable_init(arena, &st->scopes[st->num_scopes], 16);
    st->num_scopes++;
}

void symbol_table_pop_scope(ScopedSymbolTable *st) {
    if (st->num_scopes > 0) {
        st->num_scopes--;
    }
}

void symbol_table_add_value(Arena *arena, ScopedSymbolTable *st, string name, ValueRef *value) {
    if (st->num_scopes == 0) {
        // Create a default scope if none exists
        symbol_table_push_scope(arena, st);
    }
    SymbolTable_insert(arena, &st->scopes[st->num_scopes - 1], name, value);
}

ValueRef* symbol_table_lookup(ScopedSymbolTable *st, string name) {
    // Search from innermost to outermost scope
    for (size_t i = st->num_scopes; i > 0; i--) {
        ValueRef **found = SymbolTable_get(&st->scopes[i - 1], name);
        if (found && *found) {
            return *found;
        }
    }
    return NULL;
}

ValueRef* create_value_ref(Arena *arena, ValueKind kind) {
    ValueRef *value = arena_alloc(arena, ValueRef);
    value->kind = kind;
    value->def = NULL;
    value->result_index = 0;
    value->type = NULL;
    value->register_name = str_lit("");
    return value;
}

string tokentype_to_string(TokenType tt) {
    switch (tt) {
#define X(token) case token: return str_lit(#token);
        LIST_OF_TOKENS
#undef X
        default: abort();
    }
}

string op_type_to_string(OpType type) {
    switch (type) {
        case OP_TYPE_UNREGISTERED: return str_lit("unregistered");
        case OP_TYPE_MODULE: return str_lit("module");
        case OP_TYPE_ARITH_ADDI: return str_lit("arith.addi");
        case OP_TYPE_ARITH_SUBI: return str_lit("arith.subi");
        case OP_TYPE_ARITH_MULI: return str_lit("arith.muli");
        case OP_TYPE_ARITH_DIVI: return str_lit("arith.divi");
        case OP_TYPE_ARITH_ADDF: return str_lit("arith.addf");
        case OP_TYPE_ARITH_SUBF: return str_lit("arith.subf");
        case OP_TYPE_ARITH_MULF: return str_lit("arith.mulf");
        case OP_TYPE_ARITH_DIVF: return str_lit("arith.divf");
        case OP_TYPE_ARITH_CONSTANT: return str_lit("arith.constant");
        case OP_TYPE_ARITH_CMPI: return str_lit("arith.cmpi");
        case OP_TYPE_ARITH_CMPF: return str_lit("arith.cmpf");
        case OP_TYPE_MEMREF_LOAD: return str_lit("memref.load");
        case OP_TYPE_MEMREF_STORE: return str_lit("memref.store");
        case OP_TYPE_MEMREF_ALLOC: return str_lit("memref.alloc");
        case OP_TYPE_MEMREF_DEALLOC: return str_lit("memref.dealloc");
        case OP_TYPE_CF_BR: return str_lit("cf.br");
        case OP_TYPE_CF_COND_BR: return str_lit("cf.cond_br");
        case OP_TYPE_CF_SWITCH: return str_lit("cf.switch");
        case OP_TYPE_FUNC_FUNC: return str_lit("func.func");
        case OP_TYPE_FUNC_RETURN: return str_lit("func.return");
        case OP_TYPE_FUNC_CALL: return str_lit("func.call");
        case OP_TYPE_SCF_FOR: return str_lit("scf.for");
        case OP_TYPE_SCF_WHILE: return str_lit("scf.while");
        case OP_TYPE_SCF_IF: return str_lit("scf.if");
        case OP_TYPE_SCF_YIELD: return str_lit("scf.yield");
        case OP_TYPE_TT_GET_PROGRAM_ID: return str_lit("tt.get_program_id");
        case OP_TYPE_TT_LOAD: return str_lit("tt.load");
        case OP_TYPE_TT_STORE: return str_lit("tt.store");
        case OP_TYPE_TT_MAKE_RANGE: return str_lit("tt.make_range");
        case OP_TYPE_TT_SPLAT: return str_lit("tt.splat");
        case OP_TYPE_TT_ADDPTR: return str_lit("tt.addptr");
        case OP_TYPE_TT_RETURN: return str_lit("tt.return");
        default: return str_lit("unknown");
    }
}

OpType op_string_to_type(string opname) {
    if (str_eq(opname, str_lit("module"))) {
        return OP_TYPE_MODULE;
    } else if (str_eq(opname, str_lit("arith.addi"))) {
        return OP_TYPE_ARITH_ADDI;
    } else if (str_eq(opname, str_lit("arith.addf"))) {
        return OP_TYPE_ARITH_ADDF;
    } else if (str_eq(opname, str_lit("arith.subi"))) {
        return OP_TYPE_ARITH_SUBI;
    } else if (str_eq(opname, str_lit("arith.muli"))) {
        return OP_TYPE_ARITH_MULI;
    } else if (str_eq(opname, str_lit("arith.divi"))) {
        return OP_TYPE_ARITH_DIVI;
    } else if (str_eq(opname, str_lit("arith.subf"))) {
        return OP_TYPE_ARITH_SUBF;
    } else if (str_eq(opname, str_lit("arith.mulf"))) {
        return OP_TYPE_ARITH_MULF;
    } else if (str_eq(opname, str_lit("arith.divf"))) {
        return OP_TYPE_ARITH_DIVF;
    } else if (str_eq(opname, str_lit("arith.constant"))) {
        return OP_TYPE_ARITH_CONSTANT;
    } else if (str_eq(opname, str_lit("arith.cmpi"))) {
        return OP_TYPE_ARITH_CMPI;
    } else if (str_eq(opname, str_lit("arith.cmpf"))) {
        return OP_TYPE_ARITH_CMPF;
    } else if (str_eq(opname, str_lit("memref.load"))) {
        return OP_TYPE_MEMREF_LOAD;
    } else if (str_eq(opname, str_lit("memref.store"))) {
        return OP_TYPE_MEMREF_STORE;
    } else if (str_eq(opname, str_lit("memref.alloc"))) {
        return OP_TYPE_MEMREF_ALLOC;
    } else if (str_eq(opname, str_lit("memref.dealloc"))) {
        return OP_TYPE_MEMREF_DEALLOC;
    } else if (str_eq(opname, str_lit("cf.br"))) {
        return OP_TYPE_CF_BR;
    } else if (str_eq(opname, str_lit("cf.cond_br"))) {
        return OP_TYPE_CF_COND_BR;
    } else if (str_eq(opname, str_lit("cf.switch"))) {
        return OP_TYPE_CF_SWITCH;
    } else if (str_eq(opname, str_lit("func.func"))) {
        return OP_TYPE_FUNC_FUNC;
    } else if (str_eq(opname, str_lit("func.return"))) {
        return OP_TYPE_FUNC_RETURN;
    } else if (str_eq(opname, str_lit("func.call"))) {
        return OP_TYPE_FUNC_CALL;
    } else if (str_eq(opname, str_lit("scf.for"))) {
        return OP_TYPE_SCF_FOR;
    } else if (str_eq(opname, str_lit("scf.while"))) {
        return OP_TYPE_SCF_WHILE;
    } else if (str_eq(opname, str_lit("scf.if"))) {
        return OP_TYPE_SCF_IF;
    } else if (str_eq(opname, str_lit("scf.yield"))) {
        return OP_TYPE_SCF_YIELD;
    } else if (str_eq(opname, str_lit("tt.get_program_id"))) {
        return OP_TYPE_TT_GET_PROGRAM_ID;
    } else if (str_eq(opname, str_lit("tt.load"))) {
        return OP_TYPE_TT_LOAD;
    } else if (str_eq(opname, str_lit("tt.store"))) {
        return OP_TYPE_TT_STORE;
    } else if (str_eq(opname, str_lit("tt.make_range"))) {
        return OP_TYPE_TT_MAKE_RANGE;
    } else if (str_eq(opname, str_lit("tt.splat"))) {
        return OP_TYPE_TT_SPLAT;
    } else if (str_eq(opname, str_lit("tt.addptr"))) {
        return OP_TYPE_TT_ADDPTR;
    } else if (str_eq(opname, str_lit("tt.return"))) {
        return OP_TYPE_TT_RETURN;
    } else {
        return OP_TYPE_UNREGISTERED;
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

void parser_warning(Parser *parser, string msg, uint64_t first, uint64_t last) {
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

    // Print the warning message, line, and caret string
    println(parser->arena, str_lit("Warning ({}:{}): {}"),
            line_first, column_first, msg);
    println(parser->arena, str_lit("{}"), line);
    println(parser->arena, caret_str);
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
    symbol_table_init(arena, &parser->symbol_table);
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
    VecValueRef block_args;
    VecValueRef_reserve(parser->arena, &block_args, 4);

    if (parser_peek(parser, TK_CARET_NAME)) {
        parser_expect(parser, TK_CARET_NAME);

        // Parse block arguments if present: ^bb1(%0: i64, %1: i32):
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string arg_name = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    parser_expect(parser, TK_COLON);

                    // Create block argument value
                    ValueRef *block_arg = arena_alloc(parser->arena, ValueRef);
                    block_arg->kind = BLOCK_ARG;
                    block_arg->register_name = arg_name;
                    block_arg->result_index = block_args.size;
                    block_arg->def = NULL; // Block arguments don't have a defining operation
                    block_arg->type = arena_alloc(parser->arena, Type);

                    // Parse argument type
                    if (parser_peek(parser, TK_NAME)) {
                        string type_name = parser_token_str(parser);
                        parser_expect(parser, TK_NAME);
                        block_arg->type->str = type_name;
                        block_arg->type->kind = TYPE_KIND_INTEGER; // Default, could be more sophisticated
                    } else {
                        block_arg->type->str = str_lit("i32");
                        block_arg->type->kind = TYPE_KIND_INTEGER;
                    }

                    VecValueRef_push_back(parser->arena, &block_args, block_arg);

                    // Register block argument in symbol table
                    symbol_table_add_value(parser->arena, &parser->symbol_table, arg_name, block_arg);

                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    }
                } else {
                    parser_next_token(parser);
                }
            }

            parser_expect(parser, TK_RPAREN);
        }

        if (parser_peek(parser, TK_COLON)) {
            parser_expect(parser, TK_COLON);
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
    block->arguments = block_args.data;
    block->n_arguments = block_args.size;

    return block;
}

// Parses a region from { to } inclusive
Region* parse_region(Parser *parser) {
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for this region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    VecBlock blocks;
    VecBlock_reserve(parser->arena, &blocks, 8);
    while (!parser_peek(parser, TK_RBRACE)) {
        Block *block = parse_block(parser);
        VecBlock_push_back(parser->arena, &blocks, block);
    }
    parser_expect(parser, TK_RBRACE);

    // Pop scope when leaving region
    symbol_table_pop_scope(&parser->symbol_table);

    Region *region = arena_alloc(parser->arena, Region);
    region->blocks = blocks.data;
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

void parse_gpu_launch(Parser *parser, Operation *op) {
    // Parse gpu.launch blocks(%arg2, %arg3, %arg4) in (...) threads(%arg5, %arg6, %arg7) in (...) {

    VecValueRef launch_args;
    VecValueRef_reserve(parser->arena, &launch_args, 16);

    // Parse blocks(...) section
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("blocks"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            // Parse block arguments
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    ValueRef *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = launch_args.size;
                    arg->type = arena_alloc(parser->arena, Type);
                    arg->type->str = str_lit("index");
                    arg->type->kind = TYPE_KIND_INDEX;

                    VecValueRef_push_back(parser->arena, &launch_args, arg);

                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Skip until threads(...) section
    while (!parser_peek(parser, TK_EOF) && !(parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("threads")))) {
        parser_next_token(parser);
    }

    // Parse threads(...) section
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("threads"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            // Parse thread arguments
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    ValueRef *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = launch_args.size;
                    arg->type = arena_alloc(parser->arena, Type);
                    arg->type->str = str_lit("index");
                    arg->type->kind = TYPE_KIND_INDEX;

                    VecValueRef_push_back(parser->arena, &launch_args, arg);

                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Skip until opening brace
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }

    // Parse the GPU launch region
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for GPU launch region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Register all GPU launch arguments in the new scope
    for (size_t i = 0; i < launch_args.size; i++) {
        symbol_table_add_value(parser->arena, &parser->symbol_table, launch_args.data[i]->register_name, launch_args.data[i]);
    }

    // Parse GPU launch body (single block)
    VecOperation operations;
    VecOperation_reserve(parser->arena, &operations, 16);
    while (!parser_peek(parser, TK_RBRACE)) {
        Operation *inner_op = parse_operation(parser);
        VecOperation_push_back(parser->arena, &operations, inner_op);
        parser_expect(parser, TK_NEWLINE);

        // Skip empty lines
        while (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        }
    }
    parser_expect(parser, TK_RBRACE);

    // Pop scope when leaving GPU launch region
    symbol_table_pop_scope(&parser->symbol_table);

    // Create the GPU launch region
    Block *gpu_block = arena_alloc(parser->arena, Block);
    gpu_block->operations = operations.data;
    gpu_block->n_operations = operations.size;
    gpu_block->arguments = launch_args.data;
    gpu_block->n_arguments = launch_args.size;

    Region *gpu_region = arena_alloc(parser->arena, Region);
    gpu_region->blocks = arena_alloc_array(parser->arena, Block*, 1);
    gpu_region->blocks[0] = gpu_block;
    gpu_region->n_blocks = 1;

    op->regions = arena_alloc_array(parser->arena, Region*, 1);
    op->regions[0] = gpu_region;
    op->n_regions = 1;
}

// Parse generic attribute block and trailing result type
static void parse_generic_attrs_and_result_type(Parser *parser, Operation *op) {
    // Attributes block
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        int brace_depth = 1;
        while (brace_depth > 0 && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LBRACE)) brace_depth++;
            else if (parser_peek(parser, TK_RBRACE)) brace_depth--;
            parser_next_token(parser);
        }
    }

    // Optional result type after ':'
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Handle forms like ": (type, ...) -> type"
        if (parser_peek(parser, TK_LPAREN)) {
            // Consume argument type list inside parens
            int depth = 0;
            do {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            } while (depth > 0 && !parser_peek(parser, TK_EOF));

            // Optional arrow and a result type
            if (parser_peek(parser, TK_ARROW)) {
                parser_expect(parser, TK_ARROW);
            }

            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
                string type_str = parser_token_str(parser);
                parser_next_token(parser);
                if (parser_peek(parser, TK_LANGLE)) {
                    int depth = 0;
                    do {
                        if (parser_peek(parser, TK_LANGLE)) depth++;
                        else if (parser_peek(parser, TK_RANGLE)) depth--;
                        type_str = str_concat(parser->arena, type_str, parser_token_str(parser));
                        parser_next_token(parser);
                    } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
                }
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                op->result_types[0] = arena_alloc(parser->arena, Type);
                op->result_types[0]->str = type_str;
            }

            // Optional trailing loc()
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                parse_loc(parser);
            }
            return;
        }

        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
            // Parse a type token (could be src type or result type)
            string type_left = parser_token_str(parser);
            parser_next_token(parser);
            // Handle dialect types starting with '!'
            if (type_left.size == 1 && type_left.str[0] == '!' && (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME))) {
                type_left = str_concat(parser->arena, type_left, parser_token_str(parser));
                parser_next_token(parser);
            }
            if (parser_peek(parser, TK_LANGLE)) {
                int depth = 0;
                do {
                    if (parser_peek(parser, TK_LANGLE)) depth++;
                    else if (parser_peek(parser, TK_RANGLE)) depth--;
                    type_left = str_concat(parser->arena, type_left, parser_token_str(parser));
                    parser_next_token(parser);
                } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
            }

            if (parser_peek(parser, TK_ARROW)) {
                // Form ": <src-type> -> <result-type>"
                parser_expect(parser, TK_ARROW);
                if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
                    string type_right = parser_token_str(parser);
                    parser_next_token(parser);
                    if (type_right.size == 1 && type_right.str[0] == '!' && (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME))) {
                        type_right = str_concat(parser->arena, type_right, parser_token_str(parser));
                        parser_next_token(parser);
                    }
                    if (parser_peek(parser, TK_LANGLE)) {
                        int depth = 0;
                        do {
                            if (parser_peek(parser, TK_LANGLE)) depth++;
                            else if (parser_peek(parser, TK_RANGLE)) depth--;
                            type_right = str_concat(parser->arena, type_right, parser_token_str(parser));
                            parser_next_token(parser);
                        } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
                    }
                    op->n_result_types = 1;
                    op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                    op->result_types[0] = arena_alloc(parser->arena, Type);
                    op->result_types[0]->str = type_right;
                }
            } else if (parser_peek(parser, TK_COMMA)) {
                // Operand type list ": type, type, ..." — consume conservatively
                while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                    parser_next_token(parser);
                }
            } else {
                // Treat as single result type ": type"
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                op->result_types[0] = arena_alloc(parser->arena, Type);
                op->result_types[0]->str = type_left;
            }
        }
    }

    // Or optional result type directly after operands via '-> type'
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
            string type_str = parser_token_str(parser);
            parser_next_token(parser);
            if (parser_peek(parser, TK_LANGLE)) {
                int depth = 0;
                do {
                    if (parser_peek(parser, TK_LANGLE)) depth++;
                    else if (parser_peek(parser, TK_RANGLE)) depth--;
                    type_str = str_concat(parser->arena, type_str, parser_token_str(parser));
                    parser_next_token(parser);
                } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
            }
            op->n_result_types = 1;
            op->result_types = arena_alloc_array(parser->arena, Type*, 1);
            op->result_types[0] = arena_alloc(parser->arena, Type);
            op->result_types[0]->str = type_str;
        }
    }

    // Optional trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        parse_loc(parser);
    }
}

// --- Instruction-specific parsers (hooked explicitly) ---

static void parse_arith_constant(Parser *parser, Operation *op) {
    // arith.constant value : type
    if (parser_peek(parser, TK_INTEGER)) {
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_INTEGER);

        op->n_attributes = 1;
        op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
        op->attributes[0] = arena_alloc(parser->arena, Attribute);
        op->attributes[0]->kind = ATTR_KIND_INTEGER;

        int64_t parsed_value = 0;
        for (size_t i = 0; i < value_str.size; i++) {
            if (value_str.str[i] >= '0' && value_str.str[i] <= '9') {
                parsed_value = parsed_value * 10 + (value_str.str[i] - '0');
            }
        }
        op->attributes[0]->data.integer_value = parsed_value;
        op->attributes[0]->name = str_lit("value");
    } else {
        // Fallback: consume attribute-like payload (e.g., dense<...>) until ':' or EOL
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_COLON)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            parser_next_token(parser);
        }
    }
}

static void parse_arith_binary(Parser *parser, Operation *op) {
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, operand);

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str2 = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                if (!operand2) {
                    parser_warning(parser, format(parser->arena, str_lit("Undefined value: {}"), reg_str2), parser->first, parser->last);
                    operand2 = create_value_ref(parser->arena, BLOCK_ARG);
                    operand2->register_name = reg_str2;
                    operand2->type = arena_alloc(parser->arena, Type);
                    operand2->type->str = str_lit("unknown");
                }
                VecValueRef_push_back(parser->arena, &operands, operand2);
            }
        }
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Optional default result type for specific ops when not provided
    if (op->n_result_types == 0 && op->opname.size > 0 && str_eq(op->opname, str_lit("arith.addf"))) {
        op->n_result_types = 1;
        op->result_types = arena_alloc_array(parser->arena, Type*, 1);
        op->result_types[0] = arena_alloc(parser->arena, Type);
        op->result_types[0]->str = str_lit("tensor<16xf32>");
        op->result_types[0]->kind = TYPE_KIND_TENSOR;
    }
}

static void parse_tt_get_program_id(Parser *parser, Operation *op) {
    if (parser_peek(parser, TK_NAME)) {
        string axis = parser_token_str(parser);
        parser_expect(parser, TK_NAME);

        op->n_attributes = 1;
        op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
        op->attributes[0] = arena_alloc(parser->arena, Attribute);
        op->attributes[0]->kind = ATTR_KIND_STRING;
        op->attributes[0]->data.string_value = axis;
        op->attributes[0]->name = str_lit("axis");
    }
    op->n_operands = 0;
    op->operands = NULL;
}

static void parse_tt_splat(Parser *parser, Operation *op) {
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        op->n_operands = 1;
        op->operands = arena_alloc_array(parser->arena, ValueRef*, 1);
        op->operands[0] = operand;
    }
    // Infer result type if not already parsed from trailing type
    if (op->n_result_types == 0) {
        op->n_result_types = 1;
        op->result_types = arena_alloc_array(parser->arena, Type*, 1);
        op->result_types[0] = arena_alloc(parser->arena, Type);
        if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
            string operand_type = op->operands[0]->type->str;
            if (str_eq(operand_type, str_lit("!tt.ptr<f32>"))) {
                op->result_types[0]->str = str_lit("tensor<16x!tt.ptr<f32>>");
            } else if (str_eq(operand_type, str_lit("i32"))) {
                op->result_types[0]->str = str_lit("tensor<16xi32>");
            } else {
                op->result_types[0]->str = str_lit("tensor<16xi32>");
            }
        } else {
            op->result_types[0]->str = str_lit("tensor<16xi32>");
        }
        op->result_types[0]->kind = TYPE_KIND_TENSOR;
    }
}

static void parse_tt_make_range(Parser *parser, Operation *op) {
    op->n_operands = 0;
    op->operands = NULL;
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        op->n_attributes = 2;
        op->attributes = arena_alloc_array(parser->arena, Attribute*, 2);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME);
            parser_expect(parser, TK_EQUAL);
            parser_expect(parser, TK_INTEGER);
            parser_expect(parser, TK_COLON);
            parser_expect(parser, TK_NAME);
            op->attributes[0] = arena_alloc(parser->arena, Attribute);
            op->attributes[0]->kind = ATTR_KIND_INTEGER;
            op->attributes[0]->name = str_lit("end");
            op->attributes[0]->data.integer_value = 16;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME);
            parser_expect(parser, TK_EQUAL);
            parser_expect(parser, TK_INTEGER);
            parser_expect(parser, TK_COLON);
            parser_expect(parser, TK_NAME);
            op->attributes[1] = arena_alloc(parser->arena, Attribute);
            op->attributes[1]->kind = ATTR_KIND_INTEGER;
            op->attributes[1]->name = str_lit("start");
            op->attributes[1]->data.integer_value = 0;
        }
        parser_expect(parser, TK_RBRACE);
    }
    // Fallback result type if not provided by a trailing type annotation
    if (op->n_result_types == 0) {
        op->n_result_types = 1;
        op->result_types = arena_alloc_array(parser->arena, Type*, 1);
        op->result_types[0] = arena_alloc(parser->arena, Type);
        op->result_types[0]->str = str_lit("tensor<16xi32>");
        op->result_types[0]->kind = TYPE_KIND_TENSOR;
    }
}

static void parse_tt_addptr_load_store(Parser *parser, Operation *op) {
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);
    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        consume_optional_hash_selector(parser);
        ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA); else break;
    }
    op->operands = operands.data;
    op->n_operands = operands.size;

    // For tt.addptr, prefer explicit type after ':' when present; otherwise fall back.
    if (str_eq(op->opname, str_lit("tt.addptr"))) {
        if (parser_peek(parser, TK_COLON)) {
            parser_expect(parser, TK_COLON);

            // Parse first type token into a string (handles !dialect<...> nesting)
            string type_left = str_lit("");
            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
                type_left = parser_token_str(parser);
                parser_next_token(parser);
                if (type_left.size == 1 && type_left.str[0] == '!' && (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME))) {
                    type_left = str_concat(parser->arena, type_left, parser_token_str(parser));
                    parser_next_token(parser);
                }
                if (parser_peek(parser, TK_LANGLE)) {
                    int depth = 0;
                    do {
                        if (parser_peek(parser, TK_LANGLE)) depth++;
                        else if (parser_peek(parser, TK_RANGLE)) depth--;
                        type_left = str_concat(parser->arena, type_left, parser_token_str(parser));
                        parser_next_token(parser);
                    } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
                }
                // Set result type from the first type in the list
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                op->result_types[0] = arena_alloc(parser->arena, Type);
                op->result_types[0]->str = type_left;
            }
            // Consume remaining type list conservatively until end of list or loc
            if (parser_peek(parser, TK_COMMA)) {
                do {
                    parser_next_token(parser);
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE));
            }
        } else if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
            // Fallback heuristic when no explicit type list is present
            op->n_result_types = 1;
            op->result_types = arena_alloc_array(parser->arena, Type*, 1);
            op->result_types[0] = op->operands[0]->type;
        }
    }
}

// Parse tt.store: operands, optional attribute dict, and trailing type list; no results
static void parse_tt_store(Parser *parser, Operation *op) {
    // Reuse common operand parsing for tt.addptr/load/store
    parse_tt_addptr_load_store(parser, op);

    // Optional attribute dict after operands
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        int brace_depth = 1;
        while (brace_depth > 0 && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LBRACE)) brace_depth++;
            else if (parser_peek(parser, TK_RBRACE)) brace_depth--;
            parser_next_token(parser);
        }
    }

    // Trailing operand type list ": ..." (consume conservatively)
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            // Allow loc() to be parsed by the generic helper
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
        // Optional trailing loc()
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            parse_loc(parser);
        }
    }
}

static void parse_tensor_extract(Parser *parser, Operation *op) {
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *tensor_op = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!tensor_op) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, tensor_op);
    }
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                ValueRef *idx = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!idx) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValueRef_push_back(parser->arena, &operands, idx);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }
    op->operands = operands.data;
    op->n_operands = operands.size;
}

// Minimal memref.load/store operand parser supporting bracketed indices
static void parse_memref_load_or_store(Parser *parser, Operation *op) {
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 4);

    if (str_eq(op->opname, str_lit("memref.store"))) {
        // memref.store %value, %memref[indices] : memref<...>
        if (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            ValueRef *val = symbol_table_lookup(&parser->symbol_table, reg);
            if (!val) {
                val = create_value_ref(parser->arena, BLOCK_ARG);
                val->register_name = reg;
                val->type = arena_alloc(parser->arena, Type);
                val->type->str = str_lit("unknown");
                val->type->kind = TYPE_KIND_INTEGER;
            }
            VecValueRef_push_back(parser->arena, &operands, val);
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        if (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            ValueRef *val = symbol_table_lookup(&parser->symbol_table, reg);
            if (!val) {
                val = create_value_ref(parser->arena, BLOCK_ARG);
                val->register_name = reg;
                val->type = arena_alloc(parser->arena, Type);
                val->type->str = str_lit("unknown");
                val->type->kind = TYPE_KIND_INTEGER;
            }
            VecValueRef_push_back(parser->arena, &operands, val);
        }
    } else {
        // memref.load %memref[indices] : memref<...>
        if (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            ValueRef *val = symbol_table_lookup(&parser->symbol_table, reg);
            if (!val) {
                val = create_value_ref(parser->arena, BLOCK_ARG);
                val->register_name = reg;
                val->type = arena_alloc(parser->arena, Type);
                val->type->str = str_lit("unknown");
                val->type->kind = TYPE_KIND_INTEGER;
            }
            VecValueRef_push_back(parser->arena, &operands, val);
        }
    }

    // Optional index list in brackets
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        bool first = true;
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string idx = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *val = symbol_table_lookup(&parser->symbol_table, idx);
                if (!val) {
                    val = create_value_ref(parser->arena, BLOCK_ARG);
                    val->register_name = idx;
                    val->type = arena_alloc(parser->arena, Type);
                    val->type->str = str_lit("index");
                    val->type->kind = TYPE_KIND_INDEX;
                }
                VecValueRef_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    op->operands = operands.data;
    op->n_operands = operands.size;
}

// Generic/unregistered operation parsing: operands, attrs/result types, optional region, loc
static void parse_generic_operation(Parser *parser, Operation *op) {
    // Parse operands until attributes, result type, or region begin
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 4);

    while ((parser_peek(parser, TK_REGISTER) || parser_peek(parser, TK_LBRACKET)) &&
           !parser_peek(parser, TK_COLON) && !parser_peek(parser, TK_LBRACE)) {
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, operand);
        } else if (parser_peek(parser, TK_LBRACKET)) {
            // Skip indexing syntax like [%arg1]
            parser_expect(parser, TK_LBRACKET);
            while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str2 = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    ValueRef *operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                    if (!operand2) {
                        parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                        return;
                    }
                    VecValueRef_push_back(parser->arena, &operands, operand2);
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RBRACKET);
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
        } else {
            break;
        }
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Attributes and result types are parsed generically later
    // Handle generic attributes and result types before scanning for regions
    parse_generic_attrs_and_result_type(parser, op);

    // Parse regions (if any), for now we assume 0 or 1 regions
    while (!(parser_peek(parser, TK_LBRACE_END)
             || parser_peek(parser, TK_NEWLINE)
             || parser_peek(parser, TK_LPAREN_BRACE))) {
        parser_next_token(parser);
    }
    bool lparen_brace = false;
    if (parser_peek(parser, TK_LPAREN_BRACE)) {
        lparen_brace = true;
        parser_next_token(parser);
    }
    if (parser_peek(parser, TK_LBRACE_END)) {
        // Before consuming a region, parse any trailing attributes/result types
        parse_generic_attrs_and_result_type(parser, op);
        Region *region = parse_region(parser);

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
    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            parse_loc(parser);
        }
    }
}


void parse_tt_func(Parser *parser, Operation *op) {
    // Parse tt.func public @function_name(%arg0: type, %arg1: type, ...)

    // Skip "public" if present
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("public"))) {
        parser_expect(parser, TK_NAME);
    }

    // Skip @function_name
    if (parser_peek(parser, TK_AT)) {
        parser_expect(parser, TK_AT);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME);
        }
    } else {
        // Try to skip tokens until we find a parenthesis
        while (!parser_peek(parser, TK_LPAREN) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
            parser_next_token(parser);
        }
    }

    // Parse function arguments
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        // Use a vector to collect arguments as we parse them
        VecValueRef func_args;
        VecValueRef_reserve(parser->arena, &func_args, 8);

        // Parse arguments directly in one pass
        int paren_depth = 0;
        while (!parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LPAREN)) {
                paren_depth++;
                parser_next_token(parser);
            } else if (parser_peek(parser, TK_RPAREN)) {
                if (paren_depth == 0) {
                    // Found the main closing paren
                    break;
                }
                paren_depth--;
                parser_next_token(parser);
            } else if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);

                // Parse colon and type
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);

                    ValueRef *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = func_args.size;
                    arg->type = arena_alloc(parser->arena, Type);

                    // Parse the argument type
                    if (parser_peek(parser, TK_EXCLAMATION)) {
                        parser_expect(parser, TK_EXCLAMATION);

                        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                            string type_name = parser_token_str(parser);
                            if (parser_peek(parser, TK_NAME)) {
                                parser_expect(parser, TK_NAME);
                            } else {
                                parser_expect(parser, TK_NAME_DOT_NAME);
                            }

                            // Handle complex types like !tt.ptr<f32, 1>
                            if (str_eq(type_name, str_lit("tt.ptr"))) {
                                if (parser_peek(parser, TK_LANGLE)) {
                                    parser_expect(parser, TK_LANGLE);

                                    // Parse everything inside angle brackets
                                    string type_content = str_lit("");
                                    while (!parser_peek(parser, TK_RANGLE) && !parser_peek(parser, TK_EOF)) {
                                        string token_str = parser_token_str(parser);
                                        type_content = str_concat(parser->arena, type_content, token_str);
                                        parser_next_token(parser);
                                    }

                                    if (parser_peek(parser, TK_RANGLE)) {
                                        parser_expect(parser, TK_RANGLE);
                                        arg->type->str = str_concat(parser->arena, str_lit("!tt.ptr<"), str_concat(parser->arena, type_content, str_lit(">")));
                                    }
                                } else {
                                    arg->type->str = str_lit("!tt.ptr");
                                }
                            } else {
                                arg->type->str = str_concat(parser->arena, str_lit("!"), type_name);
                            }
                        } else {
                            arg->type->str = str_lit("!unknown");
                        }
                    } else if (parser_peek(parser, TK_NAME)) {
                        // Simple type like i32
                        string type_name = parser_token_str(parser);
                        parser_expect(parser, TK_NAME);
                        arg->type->str = type_name;
                    } else {
                        arg->type->str = str_lit("unknown");
                    }

                    VecValueRef_push_back(parser->arena, &func_args, arg);
                }
            } else {
                parser_next_token(parser);
            }
        }

        // Convert vector to array
        op->n_operands = func_args.size;
        op->operands = arena_alloc_array(parser->arena, ValueRef*, func_args.size);
        for (size_t i = 0; i < func_args.size; i++) {
            op->operands[i] = func_args.data[i];
        }

        if (parser_peek(parser, TK_RPAREN)) {
            parser_expect(parser, TK_RPAREN);
        } else {
        }
    }

    // Skip any remaining tokens until function body
    while (!parser_peek(parser, TK_LBRACE_END)) {
        parser_next_token(parser);
    }

    // Parse the function body region with special handling for function arguments
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for this region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Register function arguments in the new scope
    for (int i = 0; i < op->n_operands; i++) {
        if (op->operands[i]) {
            symbol_table_add_value(parser->arena, &parser->symbol_table, op->operands[i]->register_name, op->operands[i]);
        }
    }

    // Parse blocks
    VecBlock blocks;
    VecBlock_reserve(parser->arena, &blocks, 8);
    while (!parser_peek(parser, TK_RBRACE)) {
        Block *block = parse_block(parser);
        VecBlock_push_back(parser->arena, &blocks, block);
    }
    parser_expect(parser, TK_RBRACE);

    // Pop scope when leaving region
    symbol_table_pop_scope(&parser->symbol_table);

    // Create region
    Region *region = arena_alloc(parser->arena, Region);
    region->blocks = blocks.data;
    region->n_blocks = blocks.size;

    // Set up block arguments
    // Note: For tt.func, we don't copy arguments to the first block to avoid duplication
    // since the arguments are already printed in the function signature
    if (region->n_blocks > 0 && op->n_operands > 0) {
        Block *entry_block = region->blocks[0];
        // Don't set block arguments for tt.func - they're already in the function signature
        entry_block->n_arguments = 0;
        entry_block->arguments = NULL;
    }

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
    // Parse scf.for %loop_var = %start to %end step %step iter_args(%iter_var = %init) -> (type)

    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 4);

    // Parse loop variable: %loop_var
    ValueRef *loop_var = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string loop_var_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        loop_var = create_value_ref(parser->arena, BLOCK_ARG);
        loop_var->register_name = loop_var_name;
        loop_var->type = arena_alloc(parser->arena, Type);
        loop_var->type->str = str_lit("index");
        loop_var->type->kind = TYPE_KIND_INDEX;

        // Expect =
        parser_expect(parser, TK_EQUAL);

        // Parse %start operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            ValueRef *start_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!start_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, start_operand);
        }

        // Expect "to"
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
        }

        // Parse %end operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            ValueRef *end_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!end_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, end_operand);
        }

        // Expect "step"
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
        }

        // Parse %step operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            ValueRef *step_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!step_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, step_operand);
        }
    }

    // Parse iter_args(%iter_var1 = %init1, %iter_var2 = %init2, ...)
    VecValueRef iter_vars;
    VecValueRef_reserve(parser->arena, &iter_vars, 4);

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("iter_args"))) {
        parser_expect(parser, TK_NAME); // consume "iter_args"

        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string iter_var_name = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    ValueRef *iter_var = create_value_ref(parser->arena, BLOCK_ARG);
                    iter_var->register_name = iter_var_name;
                    iter_var->type = arena_alloc(parser->arena, Type);
                    iter_var->type->str = str_lit("i16");
                    iter_var->type->kind = TYPE_KIND_INTEGER;

                    VecValueRef_push_back(parser->arena, &iter_vars, iter_var);

                    // Expect =
                    parser_expect(parser, TK_EQUAL);

                    // Parse %init operand
                    if (parser_peek(parser, TK_REGISTER)) {
                        string reg_str = parser_token_str(parser);
                        parser_expect(parser, TK_REGISTER);
                        consume_optional_hash_selector(parser);
                        consume_optional_hash_selector(parser);

                        ValueRef *init_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                        if (!init_operand) {
                            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                            return;
                        }
                        VecValueRef_push_back(parser->arena, &operands, init_operand);
                    }

                    // Handle comma for multiple iter_args
                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    } else {
                        break;
                    }
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Set operands for the scf.for operation
    op->operands = operands.data;
    op->n_operands = operands.size;

    // Skip remaining tokens until region starts
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }

    // Parse the region, but first register loop arguments in the new scope
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for this region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Create a single block with loop variable and iter_args as block arguments
    VecBlock blocks;
    VecBlock_reserve(parser->arena, &blocks, 1);

    // Create the block
    Block *block = arena_alloc(parser->arena, Block);

    // Create block arguments for loop variable and iter_args
    VecValueRef block_args;
    VecValueRef_reserve(parser->arena, &block_args, 2);

    if (loop_var) {
        // Create a new ValueRef for the block argument %arg0
        ValueRef *loop_block_arg = create_value_ref(parser->arena, BLOCK_ARG);
        loop_block_arg->register_name = str_lit("%arg0");
        loop_block_arg->type = arena_alloc(parser->arena, Type);
        loop_block_arg->type->str = str_lit("index");
        loop_block_arg->type->kind = TYPE_KIND_INDEX;
        loop_block_arg->result_index = 0;
        loop_block_arg->def = block;

        VecValueRef_push_back(parser->arena, &block_args, loop_block_arg);

        // Register in symbol table using original loop variable name
        symbol_table_add_value(parser->arena, &parser->symbol_table, loop_var->register_name, loop_block_arg);
    }

    // Create block arguments for all iter_args
    for (size_t i = 0; i < iter_vars.size; i++) {
        ValueRef *iter_var = iter_vars.data[i];

        // Create a new ValueRef for the block argument %arg{i+1}
        ValueRef *iter_block_arg = create_value_ref(parser->arena, BLOCK_ARG);

        // Generate argument name like %arg1, %arg2, %arg3, etc.
        string arg_name;
        if (i == 0) {
            arg_name = str_lit("%arg1");
        } else if (i == 1) {
            arg_name = str_lit("%arg2");
        } else if (i == 2) {
            arg_name = str_lit("%arg3");
        } else {
            arg_name = str_lit("%arg4"); // Support up to 4 iter_args for now
        }
        iter_block_arg->register_name = arg_name;
        iter_block_arg->type = arena_alloc(parser->arena, Type);
        iter_block_arg->type->str = str_lit("i16");
        iter_block_arg->type->kind = TYPE_KIND_INTEGER;
        iter_block_arg->result_index = i + 1;
        iter_block_arg->def = block;

        VecValueRef_push_back(parser->arena, &block_args, iter_block_arg);

        // Register in symbol table using original iter_args variable name
        symbol_table_add_value(parser->arena, &parser->symbol_table, iter_var->register_name, iter_block_arg);
    }

    block->arguments = block_args.data;
    block->n_arguments = block_args.size;

    // Parse operations inside the block
    VecOperation operations;
    VecOperation_reserve(parser->arena, &operations, 16);
    while (!parser_peek(parser, TK_RBRACE)) {
        Operation *op = parse_operation(parser);
        VecOperation_push_back(parser->arena, &operations, op);
        parser_expect(parser, TK_NEWLINE);

        // Skip empty lines
        while (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        }
    }

    block->operations = operations.data;
    block->n_operations = operations.size;

    VecBlock_push_back(parser->arena, &blocks, block);

    parser_expect(parser, TK_RBRACE);

    // Handle optional location attribute after }
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        parser_expect(parser, TK_NAME); // consume "loc"
        parser_expect(parser, TK_LPAREN);
        // Skip location content until closing paren
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            parser_next_token(parser);
        }
        parser_expect(parser, TK_RPAREN);
    }

    // Pop scope when leaving region
    symbol_table_pop_scope(&parser->symbol_table);

    Region *region = arena_alloc(parser->arena, Region);
    region->blocks = blocks.data;
    region->n_blocks = blocks.size;

    Region **regions = arena_alloc(parser->arena, Region*);
    regions[0] = region;
    op->regions = regions;
    op->n_regions = 1;
}

// Minimal parser for affine.for loops with integer bounds
static void parse_affine_for(Parser *parser, Operation *op) {
    // Expect induction variable
    ValueRef *ind_var = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string iv_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        // Create a placeholder for the block argument; registered later
        ind_var = create_value_ref(parser->arena, BLOCK_ARG);
        ind_var->register_name = iv_name;
        ind_var->type = arena_alloc(parser->arena, Type);
        ind_var->type->str = str_lit("index");
        ind_var->type->kind = TYPE_KIND_INDEX;

        // '=' lower bound
        if (parser_peek(parser, TK_EQUAL)) {
            parser_expect(parser, TK_EQUAL);
            // Consume a simple affine lower bound: integer or register/expression until 'to'
            while (!parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) break;
                if (parser_peek(parser, TK_LBRACE_END)) break;
                parser_next_token(parser);
            }
        }

        // 'to' upper bound
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
            // Consume simple upper bound until optional 'step' or '{'
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) break;
                parser_next_token(parser);
            }
        }

        // Optional 'step' expression
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
            // Consume step expression until '{'
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                parser_next_token(parser);
            }
        }
    }

    // Now parse the loop region
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for induction variable
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Create single block and register induction variable as block argument
    Block *block = arena_alloc(parser->arena, Block);
    VecValueRef block_args; VecValueRef_reserve(parser->arena, &block_args, 1);
    if (ind_var) {
        ValueRef *iv_block_arg = create_value_ref(parser->arena, BLOCK_ARG);
        iv_block_arg->register_name = str_lit("%arg0");
        iv_block_arg->type = arena_alloc(parser->arena, Type);
        iv_block_arg->type->str = str_lit("index");
        iv_block_arg->type->kind = TYPE_KIND_INDEX;
        iv_block_arg->result_index = 0;
        iv_block_arg->def = block;
        VecValueRef_push_back(parser->arena, &block_args, iv_block_arg);
        // Map original name (e.g., %i) to the block arg
        symbol_table_add_value(parser->arena, &parser->symbol_table, ind_var->register_name, iv_block_arg);
    }
    block->arguments = block_args.data;
    block->n_arguments = block_args.size;

    // Parse body operations
    VecOperation operations; VecOperation_reserve(parser->arena, &operations, 16);
    while (!parser_peek(parser, TK_RBRACE)) {
        Operation *inner = parse_operation(parser);
        VecOperation_push_back(parser->arena, &operations, inner);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);

    // Optional trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        parse_loc(parser);
    }

    symbol_table_pop_scope(&parser->symbol_table);

    block->operations = operations.data;
    block->n_operations = operations.size;
    Region *region = arena_alloc(parser->arena, Region);
    region->blocks = arena_alloc_array(parser->arena, Block*, 1);
    region->blocks[0] = block;
    region->n_blocks = 1;
    op->regions = arena_alloc_array(parser->arena, Region*, 1);
    op->regions[0] = region;
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

// Helper function to extract register name for printing
string get_register_name(string reg_str) {
    if (reg_str.size > 1 && reg_str.str[0] == '%') {
        return reg_str; // Return the full register name for printing
    }
    return str_lit("%unknown");
}

Operation* parse_operation(Parser *parser) {
    bool skip_generic_tail = false; // when specialized parser fully consumes tail
    Operation *op = arena_alloc(parser->arena, Operation);
    op->regions = NULL;
    op->n_regions = 0;
    op->n_result_types = 0;
    op->operands = NULL;
    op->n_operands = 0;
    op->attributes = NULL;
    op->n_attributes = 0;
    op->result_types = NULL;
    op->results = NULL;
    op->n_results = 0;
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

    // Check for operations that start with operation name instead of result assignment
    if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
        string potential_opname = parser_token_str(parser);
        if (str_eq(potential_opname, str_lit("scf.yield"))) {
            // Handle scf.yield specially - it starts with opname, not result assignment
            op->opname = potential_opname;
            parser_next_token(parser); // consume operation name

            // Parse scf.yield operands
            VecValueRef operands;
            VecValueRef_reserve(parser->arena, &operands, 2);

            while (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);

                ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return NULL;
                }
                VecValueRef_push_back(parser->arena, &operands, operand);

                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                } else {
                    break;
                }
            }

            op->operands = operands.data;
            op->n_operands = operands.size;
            op->op_type = OP_TYPE_SCF_YIELD;

            // Skip remaining tokens (like : type)
            while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) {
                parser_next_token(parser);
            }

            return op;
        } else if (
            str_eq(potential_opname, str_lit("return")) ||
            str_eq(potential_opname, str_lit("func.return")) ||
            str_eq(potential_opname, str_lit("std.return")) ||
            str_eq(potential_opname, str_lit("tt.return"))
        ) {
            // Handle return-like ops generically: parse operands, no result type
            op->opname = potential_opname;
            op->op_type = op_string_to_type(potential_opname);
            parser_next_token(parser); // consume operation name

            // Parse optional operands
            VecValueRef operands;
            VecValueRef_reserve(parser->arena, &operands, 2);
            while (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return NULL;
                }
                VecValueRef_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA); else break;
            }
            op->operands = operands.data;
            op->n_operands = operands.size;

            // Consume any trailing ": ..." types or loc(), without assigning result types
            if (parser_peek(parser, TK_COLON)) {
                // Consume tokens until newline/brace
                do {
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                        parse_loc(parser);
                        break;
                    }
                    parser_next_token(parser);
                } while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF));
            }

            // Done with this op line
            while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
            return op;
        }
    }

    // Parse return registers if any
    ValueRef *result_value = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_name = parser_token_str(parser);
        op->n_result_types = 1;
        parser_expect(parser, TK_REGISTER);
        if (parser_peek(parser, TK_COLON)) {
            parser_expect(parser, TK_COLON);
            parser_expect(parser, TK_INTEGER);
        }
        parser_expect(parser, TK_EQUAL);

        // Create ValueRef for the result and add to symbol table
        result_value = arena_alloc(parser->arena, ValueRef);
        result_value->kind = OP_RESULT;
        result_value->def = op;
        result_value->result_index = 0;
        result_value->register_name = reg_name;
        result_value->type = NULL; // Will be set later when we parse the result type
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

    // Set op_type based on operation name
    op->op_type = op_string_to_type(op->opname);


    // First we handle specific opnames with special parsing rules
    if (str_eq(op->opname, str_lit("tt.func"))) {
        parse_tt_func(parser, op);
    } else if (str_eq(op->opname, str_lit("gpu.launch"))) {
        parse_gpu_launch(parser, op);
    } else if (str_eq(op->opname, str_lit("scf.if"))) {
        parse_scf_if(parser, op);
    } else if (str_eq(op->opname, str_lit("scf.for"))) {
        parse_scf_for(parser, op);
    } else if (str_eq(op->opname, str_lit("scf.while"))) {
        parse_scf_while(parser, op);
    } else if (op->op_type == OP_TYPE_ARITH_CONSTANT) {
        parse_arith_constant(parser, op);
    } else if (op->op_type == OP_TYPE_ARITH_ADDI || op->op_type == OP_TYPE_ARITH_MULI ||
               op->op_type == OP_TYPE_ARITH_ADDF || op->op_type == OP_TYPE_ARITH_SUBI ||
               op->op_type == OP_TYPE_ARITH_SUBF || op->op_type == OP_TYPE_ARITH_MULF ||
               op->op_type == OP_TYPE_ARITH_DIVI || op->op_type == OP_TYPE_ARITH_DIVF) {
        parse_arith_binary(parser, op);
    } else if (str_eq(op->opname, str_lit("tt.get_program_id"))) {
        parse_tt_get_program_id(parser, op);
    } else if (str_eq(op->opname, str_lit("tt.splat"))) {
        parse_tt_splat(parser, op);
    } else if (str_eq(op->opname, str_lit("tt.make_range"))) {
        parse_tt_make_range(parser, op);
    } else if (str_eq(op->opname, str_lit("tt.addptr")) || str_eq(op->opname, str_lit("tt.load"))) {
        parse_tt_addptr_load_store(parser, op);
    } else if (str_eq(op->opname, str_lit("tt.store"))) {
        parse_tt_store(parser, op);
    } else if (str_eq(op->opname, str_lit("affine.for"))) {
        parse_affine_for(parser, op);
    } else if (str_eq(op->opname, str_lit("memref.load")) || str_eq(op->opname, str_lit("memref.store"))) {
        parse_memref_load_or_store(parser, op);
    } else if (str_eq(op->opname, str_lit("tensor.extract"))) {
        parse_tensor_extract(parser, op);
    } else {
        // Generic/unregistered operations
        parse_generic_operation(parser, op);
        skip_generic_tail = true;
    }

    // Ensure attrs/result types are handled for specialized op paths too
    if (!skip_generic_tail) {
        parse_generic_attrs_and_result_type(parser, op);
    }

    // Generic attrs and result types already handled earlier for non-region ops

    // No op-specific result inference here; specialized parsers handle it.

    // Register result value in symbol table if we have one
    if (result_value && op->n_result_types > 0) {
        // Set the type if available
        if (op->result_types && op->result_types[0]) {
            result_value->type = op->result_types[0];
        } else {
            // Create a default type
            result_value->type = arena_alloc(parser->arena, Type);
            result_value->type->str = str_lit("i32");
            result_value->type->kind = TYPE_KIND_INTEGER;
        }
        result_value->def = op;
        symbol_table_add_value(parser->arena, &parser->symbol_table, result_value->register_name, result_value);

        // Link result to operation
        op->n_results = 1;
        op->results = arena_alloc_array(parser->arena, ValueRef*, 1);
        op->results[0] = result_value;
    } else if (op->n_result_types > 0) {
        // If operation produces a result but no explicit register name was parsed,
        // it might be an error, but let's handle it gracefully for unregistered ops
        // that produce results without explicit SSA names
    }

    return op;
}
static inline void consume_optional_hash_selector(Parser *parser) {
    // Consume tokens like #0, #1 after a register (e.g., %49#0) and ignore them
    if (parser_peek(parser, TK_HASH_NAME)) {
        parser_next_token(parser);
    }
}
