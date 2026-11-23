#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
#include <base/strbuf.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"
#include "mlir_api.h"
#include "op_parsers.h"

// Forward decls for helper scanners (moved to op_parsers.h)

// Parse a single MLIR type token into a string.
// Handles:
// - bare types like `i32`, `tensor`
// - dialect types starting with '!' like `!tt.ptr`
// - optional nested angle bracket payloads like `<...>` with proper depth
// On success, consumes the tokens that form the type and writes the
// concatenated textual form to `*out`. Returns true if a type was parsed.
bool parse_type_string(Parser *parser, string *out);

// Symbol table implementation
void symbol_table_init(Arena *arena, ScopedSymbolTable *st) {
    st->scope_capacity = 8;
    st->scopes = arena_new_array(arena, SymbolTable, st->scope_capacity);
    st->num_scopes = 0;
}

void symbol_table_push_scope(Arena *arena, ScopedSymbolTable *st) {
    if (st->num_scopes >= st->scope_capacity) {
        // Grow scopes array
        size_t new_capacity = st->scope_capacity * 2;
        SymbolTable *new_scopes = arena_new_array(arena, SymbolTable, new_capacity);
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

void symbol_table_add_value(Arena *arena, ScopedSymbolTable *st, string name, MLIR_ValueHandle value) {
    if (st->num_scopes == 0) {
        // Create a default scope if none exists
        symbol_table_push_scope(arena, st);
    }
    SymbolTable_insert(arena, &st->scopes[st->num_scopes - 1], name, value);
}

MLIR_ValueHandle symbol_table_lookup(ScopedSymbolTable *st, string name) {
    // Search from innermost to outermost scope
    for (size_t i = st->num_scopes; i > 0; i--) {
        MLIR_ValueHandle *found = SymbolTable_get(&st->scopes[i - 1], name);
        if (found && *found) {
            return *found;
        }
    }
    return MLIR_INVALID_HANDLE;
}

string tokentype_to_string(TokenType tt) {
    switch (tt) {
#define X(token) case token: return str_lit(#token);
        LIST_OF_TOKENS
#undef X
        default: abort();
    }
}

static bool is_space_char(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static string string_trim_whitespace(string s) {
    size_t start = 0;
    while (start < s.size && is_space_char(s.str[start])) start++;
    size_t end = s.size;
    while (end > start && is_space_char(s.str[end - 1])) end--;
    return str_substr(s, start, end - start);
}

static uint32_t parse_uint32_from_string(string s) {
    uint32_t value = 0;
    for (size_t i = 0; i < s.size; i++) {
        char c = s.str[i];
        if (c >= '0' && c <= '9') {
            value = value * 10u + (uint32_t)(c - '0');
        }
    }
    return value;
}

// tensor<...>, memref<...> or (is_vector) vector<...>.
static MLIR_TypeHandle parse_tensor_like_type(MLIR_Context *ctx, string content, bool is_tensor,
                                              bool is_vector) {
    int last_x_pos = -1;
    int bracket_depth = 0;
    for (int i = (int)content.size - 1; i >= 0; i--) {
        char c = content.str[i];
        if (c == '>') bracket_depth++;
        else if (c == '<') bracket_depth--;
        else if (c == 'x' && bracket_depth == 0 && i > 0 &&
                 ((content.str[i - 1] >= '0' && content.str[i - 1] <= '9') ||
                  content.str[i - 1] == '?')) {
            // A dimension separator follows a size or `?` (the `x` of an
            // element type such as `index` does not).
            last_x_pos = i;
            break;
        }
    }

    if (last_x_pos >= 0) {
        string elem_part = string_trim_whitespace(str_substr(content, (size_t)last_x_pos + 1, content.size - (size_t)last_x_pos - 1));
        MLIR_TypeHandle element_type = mlir_type_create_from_string(ctx, elem_part);

        string shape_str = str_substr(content, 0, (size_t)last_x_pos);
        uint32_t rank = 1;
        for (size_t i = 0; i < shape_str.size; i++) {
            if (shape_str.str[i] == 'x') rank++;
        }

        if (rank > 0) {
            int64_t *dims = arena_new_array(MLIR_GetArenaAllocator(ctx), int64_t, rank);
            size_t pos = 0;
            uint32_t dim_idx = 0;
            while (pos <= shape_str.size && dim_idx < rank) {
                size_t next = pos;
                while (next < shape_str.size && shape_str.str[next] != 'x') next++;
                string tok = string_trim_whitespace(str_substr(shape_str, pos, next - pos));
                if (tok.size == 1 && tok.str[0] == '?') {
                    dims[dim_idx++] = -1;
                } else {
                    int64_t val = 0;
                    for (size_t j = 0; j < tok.size; j++) {
                        char ch = tok.str[j];
                        if (ch >= '0' && ch <= '9') {
                            val = val * 10 + (int64_t)(ch - '0');
                        }
                    }
                    dims[dim_idx++] = val;
                }
                pos = next + 1;
            }

            if (is_vector) {
                return MLIR_CreateTypeVector(ctx, dims, rank, element_type);
            }
            if (is_tensor) {
                return MLIR_CreateTypeTensor(ctx, dims, rank, element_type);
            }
            return MLIR_CreateTypeMemref(ctx, dims, rank, element_type);
        }
    }

    MLIR_TypeHandle elem_fallback = mlir_type_create_from_string(ctx, string_trim_whitespace(content));
    if (is_vector) {
        // A vector has at least one dimension.
        return MLIR_CreateTypeOpaque(ctx, content);
    }
    if (is_tensor) {
        return MLIR_CreateTypeTensor(ctx, MLIR_INVALID_HANDLE, 0, elem_fallback);
    }
    return MLIR_CreateTypeMemref(ctx, MLIR_INVALID_HANDLE, 0, elem_fallback);
}

MLIR_TypeHandle mlir_type_create_from_string(MLIR_Context *ctx, string type_str) {
    string type = string_trim_whitespace(type_str);
    if (type.size == 0) {
        return MLIR_CreateTypeInteger(ctx, 32, true);
    }

    if (str_eq(type, str_lit("?"))) {
        return MLIR_CreateTypeUnknown(ctx);
    }

    if (str_eq(type, str_lit("unknown")) || str_eq(type, str_lit("!unknown"))) {
        return MLIR_CreateTypeOpaque(ctx, type);
    }

    if (str_eq(type, str_lit("index"))) {
        return MLIR_CreateTypeIndex(ctx);
    }

    if (type.size >= 2 && type.str[0] == 'i') {
        bool all_digits = true;
        for (size_t i = 1; i < type.size; i++) {
            char c = type.str[i];
            if (c < '0' || c > '9') {
                all_digits = false;
                break;
            }
        }

        if (all_digits) {
            uint32_t width = 0;
            for (size_t i = 1; i < type.size; i++) {
                width = width * 10u + (uint32_t)(type.str[i] - '0');
            }
            return MLIR_CreateTypeInteger(ctx, width, true);
        }
        return MLIR_CreateTypeInteger(ctx, 32, true);
    }

    bool is_float_prefix = false;
    bool is_bfloat = false;
    if (type.size >= 1 && type.str[0] == 'f') {
        is_float_prefix = true;
    } else if (type.size >= 2 && type.str[0] == 'b' && type.str[1] == 'f') {
        is_float_prefix = true;
        is_bfloat = true;
    }

    if (is_float_prefix) {
        size_t width_start = is_bfloat ? 2 : 1;
        uint32_t width = 32;
        for (size_t i = width_start; i < type.size; i++) {
            char c = type.str[i];
            if (c >= '0' && c <= '9') {
                if (i == width_start) width = 0;
                width = width * 10u + (uint32_t)(c - '0');
            }
        }
        return MLIR_CreateTypeFloat(ctx, width, is_bfloat);
    }

    if (type.size >= 7 && str_eq(str_substr(type, 0, 7), str_lit("!tt.ptr"))) {
        if (type.size > 8 && type.str[7] == '<' && type.str[type.size - 1] == '>') {
            string content = str_substr(type, 8, type.size - 9);
            size_t comma_pos = content.size;
            for (size_t i = 0; i < content.size; i++) {
                if (content.str[i] == ',') {
                    comma_pos = i;
                    break;
                }
            }

            if (comma_pos < content.size) {
                string elem_part = string_trim_whitespace(str_substr(content, 0, comma_pos));
                string addr_part = string_trim_whitespace(str_substr(content, comma_pos + 1, content.size - comma_pos - 1));
                MLIR_TypeHandle elem_type = mlir_type_create_from_string(ctx, elem_part);
                uint32_t addr_space = parse_uint32_from_string(addr_part);
                return MLIR_CreateTypePointer(ctx, elem_type, true, addr_space);
            }

            MLIR_TypeHandle elem_type = mlir_type_create_from_string(ctx, string_trim_whitespace(content));
            return MLIR_CreateTypePointer(ctx, elem_type, false, 1);
        }

        MLIR_TypeHandle fallback_elem = mlir_type_create_from_string(ctx, str_lit("f32"));
        return MLIR_CreateTypePointer(ctx, fallback_elem, false, 1);
    }

    if (type.size >= 6 && str_eq(str_substr(type, 0, 6), str_lit("tensor"))) {
        if (type.size > 7 && type.str[6] == '<' && type.str[type.size - 1] == '>') {
            string content = str_substr(type, 7, type.size - 8);
            return parse_tensor_like_type(ctx, content, true, false);
        }

        MLIR_TypeHandle default_elem = mlir_type_create_from_string(ctx, str_lit("f32"));
        return MLIR_CreateTypeTensor(ctx, MLIR_INVALID_HANDLE, 0, default_elem);
    }

    if (type.size > 8 && str_eq(str_substr(type, 0, 7), str_lit("vector<")) &&
        type.str[type.size - 1] == '>') {
        string content = str_substr(type, 7, type.size - 8);
        return parse_tensor_like_type(ctx, content, false, true);
    }

    if (type.size >= 6 && str_eq(str_substr(type, 0, 6), str_lit("memref"))) {
        if (type.size > 7 && type.str[6] == '<' && type.str[type.size - 1] == '>') {
            string content = str_substr(type, 7, type.size - 8);
            return parse_tensor_like_type(ctx, content, false, false);
        }
        return MLIR_CreateTypeMemref(ctx, MLIR_INVALID_HANDLE, 0, MLIR_INVALID_HANDLE);
    }

    return MLIR_CreateTypeInteger(ctx, 32, true);
}

MLIR_TypeHandle parse_type_from_string(MLIR_Context *ctx, string type_str) {
    return mlir_type_create_from_string(ctx, type_str);
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
        println(str_lit("{}"), newlines);
        println(str_lit("first: {}, last: {}, line_first: {}, column_first: {}, start_of_line: {}, end_of_line: {}"),
                first, last, line_first, column_first, start_of_line, end_of_line);
    }

    // Extract the line as a string
    string line = { .str = s.str + start_of_line, .size = end_of_line - start_of_line };

    // Create the caret string (spaces followed by carets)
    int64_t token_length = last - first + 1; // Assuming 'last' is inclusive
    assert(first >= start_of_line);
    char* caret_buf = arena_new_array(parser->arena, char, first - start_of_line + token_length);
    for (int64_t i = 0; i < first - start_of_line; i++) {
        caret_buf[i] = ' ';
    }
    for (int64_t i = 0; i < token_length; i++) {
        caret_buf[first - start_of_line + i] = '^';
    }
    string caret_str = { .str = caret_buf, .size = first - start_of_line + token_length };

    // Print the error message, line, and caret string
    println(str_lit("Syntax error ({}:{}): {}"),
            line_first, column_first, msg);
    println(str_lit("{}"), line);
    println(caret_str);

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
        println(str_lit("{}"), newlines);
        println(str_lit("first: {}, last: {}, line_first: {}, column_first: {}, start_of_line: {}, end_of_line: {}"),
                first, last, line_first, column_first, start_of_line, end_of_line);
    }

    // Extract the line as a string
    string line = { .str = s.str + start_of_line, .size = end_of_line - start_of_line };

    // Create the caret string (spaces followed by carets)
    int64_t token_length = last - first + 1; // Assuming 'last' is inclusive
    assert(first >= start_of_line);
    char* caret_buf = arena_new_array(parser->arena, char, first - start_of_line + token_length);
    for (int64_t i = 0; i < first - start_of_line; i++) {
        caret_buf[i] = ' ';
    }
    for (int64_t i = 0; i < token_length; i++) {
        caret_buf[first - start_of_line + i] = '^';
    }
    string caret_str = { .str = caret_buf, .size = first - start_of_line + token_length };

    // Print the warning message, line, and caret string
    println(str_lit("Warning ({}:{}): {}"),
            line_first, column_first, msg);
    println(str_lit("{}"), line);
    println(caret_str);
}

void parser_next_token(Parser *parser) {
    do {
        parser->first = parser->cur;
        tokenizer_get_next_token(parser->input, &parser->cur, &parser->sym);
        parser->last = parser->cur-1;
    } while (parser->sym == TK_WHITESPACE || parser->sym == TK_COMMENT);
}

void parser_init(MLIR_Context *ctx, Parser *parser, string text) {
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    string text_null = str_concat(arena, text, str_lit("\0"));
    parser->ctx = ctx;
    parser->arena = arena;
    parser_op_types_init(ctx, &parser->op_types);
    parser->input = (unsigned char*) text_null.str;
    parser->cur = 0;
    symbol_table_init(arena, &parser->symbol_table);
    LocationMap_init(arena, &parser->location_map, 16);
    parser->next_loc_id = 0;
    parser->unnumbered_loc_def = MLIR_INVALID_HANDLE;
    parser->capture_trailing_comments = false;
    parser->block_label_stack = NULL;
    parser->block_label_depth = 0;
    parser->block_label_capacity = 0;
    parser_next_token(parser);
}

void parser_push_region_blocks(Parser *parser) {
    if (parser->block_label_depth >= parser->block_label_capacity) {
        size_t new_cap = parser->block_label_capacity ? parser->block_label_capacity * 2 : 4;
        BlockLabelMap *new_stack = arena_new_array(parser->arena, BlockLabelMap, new_cap);
        if (parser->block_label_stack) {
            memcpy(new_stack, parser->block_label_stack,
                   parser->block_label_depth * sizeof(BlockLabelMap));
        }
        parser->block_label_stack = new_stack;
        parser->block_label_capacity = new_cap;
    }
    BlockLabelMap_init(parser->arena, &parser->block_label_stack[parser->block_label_depth], 8);
    parser->block_label_depth++;
}

void parser_pop_region_blocks(Parser *parser) {
    if (parser->block_label_depth > 0) parser->block_label_depth--;
}

MLIR_BlockHandle parser_get_or_create_block(Parser *parser, string label) {
    if (parser->block_label_depth == 0) {
        return MLIR_CreateBlock(parser->ctx);
    }
    BlockLabelMap *m = &parser->block_label_stack[parser->block_label_depth - 1];
    MLIR_BlockHandle *existing = BlockLabelMap_get(m, label);
    if (existing) {
        return *existing;
    }
    MLIR_BlockHandle nb = MLIR_CreateBlock(parser->ctx);
    BlockLabelMap_insert(parser->arena, m, label, nb);
    return nb;
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

// Parse a single MLIR type token into a string representation.
// Consumes:
// - A leading TK_NAME/TK_NAME_DOT_NAME or TK_EXCLAMATION (dialect)
// - If leading '!' is present, the following name token
// - An optional angle-bracket payload with proper nesting until matching '>'
bool parse_type_string(Parser *parser, string *out) {
    if (!(parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION))) {
        return false;
    }

    strbuf ty = strbuf_make();
    strbuf_append(parser->arena, &ty, parser_token_str(parser));
    parser_next_token(parser);

    // Dialect types starting with '!'
    if (ty.size == 1 && ty.str[0] == '!' && (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME))) {
        strbuf_append(parser->arena, &ty, parser_token_str(parser));
        parser_next_token(parser);
    }

    // Optional angle-bracketed content (supports nesting)
    if (parser_peek(parser, TK_LANGLE)) {
        int depth = 0;
        do {
            if (parser_peek(parser, TK_LANGLE)) depth++;
            else if (parser_peek(parser, TK_RANGLE)) depth--;
            strbuf_append(parser->arena, &ty, parser_token_str(parser));
            parser_next_token(parser);
        } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
    }

    *out = strbuf_to_string(ty);
    return true;
}




MLIR_OpHandle parse_operation(Parser *parser);

MLIR_BlockHandle parse_block(Parser *parser) {
    VecValue block_args;
    VecValue_reserve(parser->arena, &block_args, 4);

    string block_label = str_lit("");
    if (parser_peek(parser, TK_CARET_NAME)) {
        block_label = parser_token_str(parser);
        parser_expect(parser, TK_CARET_NAME);

        // Parse block arguments if present: ^bb1(%0: i64, %1: i32):
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string arg_name = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    parser_expect(parser, TK_COLON);

                    // Parse argument type
                    string type_name = str_lit("");
                    MLIR_TypeHandle arg_type = MLIR_INVALID_HANDLE;
                    if (parse_type_string(parser, &type_name)) {
                        arg_type = mlir_type_create_from_string(parser->ctx, type_name);
                    } else {
                        arg_type = mlir_type_create_from_string(parser->ctx, str_lit("i32"));
                    }
                    // Create block argument value with parsed type
                    MLIR_ValueHandle block_arg = MLIR_CreateValueBlockArg(parser->ctx, arg_name, (uint32_t)block_args.size, arg_type, MLIR_INVALID_HANDLE);

                    VecValue_push_back(parser->arena, &block_args, block_arg);

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
    // Create the block and bind its arguments BEFORE parsing operations,
    // so operand references to block args resolve to real values during
    // op construction (required by upstream MLIR's strict Value model).
    // Use forward-reference-friendly creation so cf.br targets resolve to
    // the same MLIR_BlockHandle whether the label appears before or after
    // the branch op.
    MLIR_BlockHandle block;
    if (block_label.size > 0) {
        block = parser_get_or_create_block(parser, block_label);
    } else {
        block = MLIR_CreateBlock(parser->ctx);
    }
    for (size_t i = 0; i < block_args.size; i++) {
        MLIR_AppendBlockArg(parser->ctx, block, block_args.data[i]);
    }

    VecOp operations;
    VecOp_reserve(parser->arena, &operations, 16);
    while (! (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_CARET_NAME))) {
        MLIR_OpHandle op = parse_operation(parser);

        VecOp_push_back(parser->arena, &operations, op);
        parser_expect(parser, TK_NEWLINE);

        // Skip empty lines
        while (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        }
    }

    for (size_t i = 0; i < operations.size; i++) {
        MLIR_AppendBlockOp(parser->ctx, block, operations.data[i]);
    }

    return block;
}

// Parses a region from { to } inclusive
MLIR_RegionHandle parse_region(Parser *parser) {
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for this region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);
    parser_push_region_blocks(parser);

    VecBlock blocks;
    VecBlock_reserve(parser->arena, &blocks, 8);
    while (!parser_peek(parser, TK_RBRACE)) {
        MLIR_BlockHandle block = parse_block(parser);
        VecBlock_push_back(parser->arena, &blocks, block);
    }
    parser_expect(parser, TK_RBRACE);

    // Pop scope when leaving region
    symbol_table_pop_scope(&parser->symbol_table);
    parser_pop_region_blocks(parser);

    MLIR_RegionHandle region = MLIR_CreateRegion(parser->ctx);
    for (size_t i = 0; i < blocks.size; i++) {
        MLIR_AppendRegionBlock(parser->ctx, region, blocks.data[i]);
    }

    return region;
}

MLIR_OpHandle parse_module(Parser *parser) {
    // Capture any top-of-file #loc definitions before the module
    MLIR_LocationHandle loc0_def = MLIR_INVALID_HANDLE;
    while (parser_peek(parser, TK_HASH_NAME)) {
        string hash_name = parser_token_str(parser);
        parser_next_token(parser); // consume '#name'
        if (parser_peek(parser, TK_EQUAL)) {
            parser_next_token(parser); // consume '='
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                MLIR_LocationHandle loc_def = parse_loc(parser);
                if (loc_def != MLIR_INVALID_HANDLE) {
                    LocationMap_insert(parser->arena, &parser->location_map, hash_name, loc_def);
                    if (hash_name.size == 4 && strncmp(hash_name.str, "#loc", 4) == 0) {
                        loc0_def = loc_def;
                    }
                }
            } else {
                // Not a loc() definition; skip until end of line
                while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
            }
        } else {
            // No '=', skip rest of the line
            while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
        }
        if (parser_peek(parser, TK_NEWLINE)) parser_next_token(parser);
        while (parser_peek(parser, TK_NEWLINE) || parser_peek(parser, TK_WHITESPACE)) parser_next_token(parser);
    }

    // Use the top-of-file #loc definition if available
    if (loc0_def != MLIR_INVALID_HANDLE) {
        parser->unnumbered_loc_def = loc0_def;
    }

    MLIR_OpHandle op = parse_operation(parser);
    if (MLIR_GetOpType(parser->ctx, op) != OP_TYPE_MODULE) {
        parser_error(parser, str_lit("The top level operation should be a module"), 0, 0);
    }

    // Skip whitespace and newlines after the module
    while (parser_peek(parser, TK_NEWLINE) || parser_peek(parser, TK_WHITESPACE)) {
        parser_next_token(parser);
    }

    // Parse location definitions that appear after the module
    // Format: #locN = loc(...)
    while (parser_peek(parser, TK_HASH_NAME)) {
        string hash_name = parser_token_str(parser);
        parser_next_token(parser);
        if (parser_peek(parser, TK_EQUAL)) {
            parser_next_token(parser); // consume '='
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                MLIR_LocationHandle loc_def = parse_loc(parser);
                if (loc_def != MLIR_INVALID_HANDLE) {
                    LocationMap_insert(parser->arena, &parser->location_map, hash_name, loc_def);
                }
            } else {
                // Skip other definitions
                while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
            }
        }
        if (parser_peek(parser, TK_NEWLINE)) parser_next_token(parser);
        while (parser_peek(parser, TK_NEWLINE) || parser_peek(parser, TK_WHITESPACE)) parser_next_token(parser);
    }

    return op;
}

MLIR_OpHandle mlir_parse_module(MLIR_Context *ctx, const char *input, size_t input_len, MLIR_LocationMap **out_location_map) {
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    Parser *parser = arena_new(arena, Parser);
    string input_string = {
        .str = (char*)input,
        .size = input_len
    };
    parser_init(ctx, parser, input_string);
    MLIR_OpHandle module = parse_module(parser);
    if (out_location_map) {
        MLIR_LocationMap *map_wrapper = arena_new(arena, MLIR_LocationMap);
        map_wrapper->impl = &parser->location_map;
        *out_location_map = map_wrapper;
    }
    return module;
}

const char *mlir_tokentype_to_string(int token_type) {
    string s = tokentype_to_string((TokenType)token_type);
    return s.str;
}

size_t MLIR_GetLocationMapSize(const MLIR_LocationMap *location_map) {
    if (!location_map) return 0;
    const LocationMap *lm = (const LocationMap*)location_map->impl;
    if (!lm) return 0;
    return lm->size;
}

size_t MLIR_CollectLocationMap(const MLIR_LocationMap *location_map, string *out_keys, MLIR_LocationHandle *out_locs, size_t max) {
    if (!location_map) return 0;
    const LocationMap *lm = (const LocationMap*)location_map->impl;
    if (!lm) return 0;
    size_t written = 0;
    for (size_t i = 0; i < lm->num_buckets && written < max; i++) {
        if (!lm->buckets[i].occupied) continue;
        out_keys[written] = lm->buckets[i].key;
        out_locs[written] = lm->buckets[i].value;
        written++;
    }
    return written;
}

// parse loc()
MLIR_LocationHandle parse_loc(Parser *parser) {
    parser_expect(parser, TK_NAME); // 'loc'
    parser_expect(parser, TK_LPAREN);

    MLIR_LocationHandle loc = MLIR_INVALID_HANDLE;

    // Check what kind of location this is
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("callsite"))) {
        // Capture loc(callsite(...)) verbatim as original_text
        strbuf text = strbuf_make();
        strbuf_append(parser->arena, &text, str_lit("loc("));
        // Accumulate tokens until we hit the matching ')' of loc(
        int depth = 0;
        while (!parser_peek(parser, TK_EOF)) {
            string tok = parser_token_str(parser);
            if (parser_peek(parser, TK_LPAREN)) depth++;
            else if (parser_peek(parser, TK_RPAREN)) {
                if (depth == 0) break; else depth--;
            }
            // Insert space before identifiers or #loc if needed
            if (tok.size > 0) {
                char c0 = tok.str[0];
                char last = text.size > 0 ? text.str[text.size - 1] : 0;
                bool need_space = false;
                if ((c0 == '#' || (c0 >= 'A' && c0 <= 'z')) && last != '(' && last != ' ' && last != ',') {
                    need_space = true;
                }
                if (need_space) strbuf_append(parser->arena, &text, str_lit(" "));
            }
            strbuf_append(parser->arena, &text, tok);
            parser_next_token(parser);
        }
        parser_expect(parser, TK_RPAREN);
        strbuf_append(parser->arena, &text, str_lit(")"));
        loc = MLIR_CreateLocationUnknown(parser->ctx, strbuf_to_string(text));
        return loc;
    } else if (parser_peek(parser, TK_STRING)) {
        // loc("filename":line:col) or loc("name")
        string filename = parser_token_str(parser);
        // The lexer includes the surrounding double quotes; strip them so
        // the stored filename is the raw content. The printer will re-quote
        // when needed.
        if (filename.size >= 2 && filename.str[0] == '"' && filename.str[filename.size - 1] == '"') {
            filename.str += 1;
            filename.size -= 2;
        }
        parser_next_token(parser);

        if (parser_peek(parser, TK_COLON)) {
            // File location: loc("filename":line:col)
            parser_next_token(parser); // consume ':'

            int line = 0;
            int column = 0;

            if (parser_peek(parser, TK_INTEGER)) {
                string line_str = parser_token_str(parser);
                line = atoi(line_str.str);
                parser_next_token(parser);

                if (parser_peek(parser, TK_COLON)) {
                    parser_next_token(parser); // consume ':'
                    if (parser_peek(parser, TK_INTEGER)) {
                        string col_str = parser_token_str(parser);
                        column = atoi(col_str.str);
                        parser_next_token(parser);
                    }
                }
            }

            loc = MLIR_CreateLocationFile(parser->ctx, filename, line, column);
        } else {
            // Named location: loc("name")
            loc = MLIR_CreateLocationName(parser->ctx, filename);
        }
    } else if (parser_peek(parser, TK_HASH_NAME)) {
        // Reference location: loc(#locN)
        string hash_name = parser_token_str(parser);
        parser_next_token(parser); // consume '#locN'
        int ref_id = 0;
        if (hash_name.size > 4 && strncmp(hash_name.str, "#loc", 4) == 0) {
            ref_id = atoi(hash_name.str + 4);
        }
        loc = MLIR_CreateLocationRef(parser->ctx, ref_id);
    } else {
        // Unknown location format, just consume tokens until ')'
        while (!(parser_peek(parser, TK_RPAREN))) {
            parser_next_token(parser);
        }
        loc = MLIR_CreateLocationUnknown(parser->ctx, str_lit("loc(unknown)"));
    }

    parser_expect(parser, TK_RPAREN);
    return loc;
}

// Helper function to consume optional hash selector
void consume_optional_hash_selector(Parser *parser) {
    // Consume tokens like #0, #1 after a register (e.g., %49#0) and ignore them
    if (parser_peek(parser, TK_HASH_NAME)) {
        parser_next_token(parser);
    }
}





const char *string_data_or_null(string s) {
    return s.size > 0 ? s.str : NULL;
}

bool parse_register_operand(Parser *parser, VecValue *operands, bool allow_hash_selector) {
    if (!parser_peek(parser, TK_REGISTER)) return false;
    string reg_str = parser_token_str(parser);
    parser_expect(parser, TK_REGISTER);
    if (allow_hash_selector) consume_optional_hash_selector(parser);
    MLIR_ValueHandle operand = symbol_table_lookup(&parser->symbol_table, reg_str);
    if (operand == MLIR_INVALID_HANDLE) {
        parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
        return false;
    }
    VecValue_push_back(parser->arena, operands, operand);
    return true;
}

MLIR_OpHandle parser_create_op_with_successors(
    Parser *parser, const OperationParserParams *params,
    MLIR_OpType type,
    const MLIR_NamedAttr *attributes, size_t n_attributes,
    const MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *out_results, size_t n_results,
    const MLIR_ValueHandle *operands, size_t n_operands,
    const MLIR_RegionHandle *regions, size_t n_regions,
    const MLIR_BlockHandle *successors, size_t n_successors,
    MLIR_ValueHandle **successor_operands,
    const size_t *n_successor_operands,
    MLIR_LocationHandle location) {
    MLIR_Context *ctx = parser->ctx;
    uint32_t nr = (uint32_t)(n_result_types > n_results ? n_result_types : n_results);
    const MLIR_TypeHandle *rtypes = result_types;
    if (nr > n_result_types) {
        MLIR_TypeHandle *padded = arena_new_array(parser->arena, MLIR_TypeHandle, nr);
        for (uint32_t i = 0; i < nr; i++) {
            padded[i] = (i < n_result_types && result_types) ? result_types[i] : MLIR_INVALID_HANDLE;
        }
        rtypes = padded;
    }

    size_t n_succ_ops = 0;
    if (n_successor_operands) {
        for (size_t s = 0; s < n_successors; s++) n_succ_ops += n_successor_operands[s];
    }
    const MLIR_ValueHandle *flat_ops = operands;
    uint32_t total_ops = (uint32_t)(n_operands + n_succ_ops);
    uint32_t *succ_counts = NULL;
    if (n_successors > 0) {
        succ_counts = arena_new_array(parser->arena, uint32_t, n_successors);
        for (size_t s = 0; s < n_successors; s++) {
            succ_counts[s] = n_successor_operands ? (uint32_t)n_successor_operands[s] : 0;
        }
        if (n_succ_ops > 0) {
            MLIR_ValueHandle *buf = arena_new_array(parser->arena, MLIR_ValueHandle, total_ops);
            size_t idx = 0;
            for (size_t i = 0; i < n_operands; i++) buf[idx++] = operands[i];
            for (size_t s = 0; s < n_successors; s++) {
                size_t k = n_successor_operands ? n_successor_operands[s] : 0;
                for (size_t j = 0; j < k; j++) buf[idx++] = successor_operands[s][j];
            }
            flat_ops = buf;
        }
    }

    MLIR_OpSpec spec = {0};
    spec.type = type;
    spec.operands = flat_ops;
    spec.n_operands = total_ops;
    spec.result_types = rtypes;
    spec.n_results = nr;
    spec.attrs = attributes;
    spec.n_attrs = (uint32_t)n_attributes;
    spec.regions = regions;
    spec.n_regions = (uint32_t)n_regions;
    spec.successors = successors;
    spec.n_successors = (uint32_t)n_successors;
    spec.successor_operand_counts = succ_counts;
    spec.loc = location;
    MLIR_OpHandle op = MLIR_CreateOpFromSpec(ctx, &spec);

    if (params && params->trailing_comment.size > 0) {
        MLIR_SetOpTrailingComment(ctx, op, params->trailing_comment);
    }
    for (uint32_t i = 0; i < nr; i++) {
        MLIR_ValueHandle rv = MLIR_GetOpResult(ctx, op, i);
        if (params && params->lhs_result_names && i < params->n_lhs_results &&
            params->lhs_result_names[i].size > 0) {
            MLIR_SetValueRegisterName(ctx, rv, params->lhs_result_names[i]);
        }
        if (out_results && i < n_results) out_results[i] = rv;
    }
    return op;
}

MLIR_OpHandle parser_create_op(
    Parser *parser, const OperationParserParams *params,
    MLIR_OpType type,
    const MLIR_NamedAttr *attributes, size_t n_attributes,
    const MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *out_results, size_t n_results,
    const MLIR_ValueHandle *operands, size_t n_operands,
    const MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_LocationHandle location) {
    return parser_create_op_with_successors(
        parser, params, type,
        attributes, n_attributes,
        result_types, n_result_types,
        out_results, n_results,
        operands, n_operands,
        regions, n_regions,
        NULL, 0, NULL, NULL,
        location);
}

MLIR_ValueHandle *finalize_results(const OperationParserParams *params,
                                     MLIR_OpHandle op,
                                     MLIR_TypeHandle *result_types,
                                     size_t n_result_types,
                                     size_t *out_n_results) {
    (void)result_types;
    MLIR_ValueHandle *results = MLIR_INVALID_HANDLE;
    size_t n_results = 0;
    if (op != MLIR_INVALID_HANDLE) {
        size_t op_nr = MLIR_GetOpNumResults(params->ctx, op);
        n_results = (n_result_types > 0 && n_result_types <= op_nr) ? n_result_types : op_nr;
        if (n_results > 0) {
            results = arena_new_array(params->arena, MLIR_ValueHandle, n_results);
            for (size_t i = 0; i < n_results; i++) {
                MLIR_ValueHandle rv = MLIR_GetOpResult(params->ctx, op, i);
                if (params->lhs_result_names && i < params->n_lhs_results &&
                    params->lhs_result_names[i].size > 0) {
                    MLIR_SetValueRegisterName(params->ctx, rv, params->lhs_result_names[i]);
                }
                results[i] = rv;
            }
        }
    }
    if (out_n_results) *out_n_results = n_results;
    return results;
}

MLIR_NamedAttr create_string_attr(Parser *parser, string name, string value) {
    return MLIR_Named(MLIR_InternAttrKey(parser->ctx, name), MLIR_CreateAttributeString(parser->ctx, value));
}

MLIR_NamedAttr create_integer_attr(Parser *parser, string name, int64_t value, MLIR_TypeHandle type) {
    return MLIR_Named(MLIR_InternAttrKey(parser->ctx, name), MLIR_CreateAttributeInteger(parser->ctx, value, type));
}

MLIR_NamedAttr create_float_attr(Parser *parser, string name, double value, MLIR_TypeHandle type) {
    return MLIR_Named(MLIR_InternAttrKey(parser->ctx, name), MLIR_CreateAttributeFloat(parser->ctx, value, type));
}

MLIR_NamedAttr create_bool_attr(Parser *parser, string name, bool value) {
    return MLIR_Named(MLIR_InternAttrKey(parser->ctx, name), MLIR_CreateAttributeBool(parser->ctx, value));
}

// The text of an attribute value up to the `,` or `}` that ends it, outside
// `<...>`. An unmatched `<` is an error (it would otherwise read to the end
// of the input).
static string parse_attr_payload(Parser *parser) {
    string payload = str_lit("");
    int angle = 0;
    uint64_t first = parser->first;
    while (!parser_peek(parser, TK_EOF) &&
           !(angle == 0 && (parser_peek(parser, TK_COMMA) || parser_peek(parser, TK_RBRACE)))) {
        string tok = parser_token_str(parser);
        if (parser_peek(parser, TK_LANGLE)) angle++;
        else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
        payload = payload.size ? str_concat(parser->arena, payload, tok) : tok;
        parser_next_token(parser);
    }
    if (angle > 0)
        parser_error(parser, str_lit("Unmatched '<' in an attribute value"), first, first);
    return payload;
}

// An attribute the parser keeps as its text, except a dense integer array
// (`array<i32: 1, 2>`, `array<i64>`), which becomes a typed one so that
// passes can read it (MLIR_GetAttributeDenseElement).
MLIR_NamedAttr create_payload_attr(Parser *parser, string name, string payload) {
    string p32 = str_lit("array<i32");
    string p64 = str_lit("array<i64");
    bool is32 = payload.size >= p32.size && memcmp(payload.str, p32.str, p32.size) == 0;
    bool is64 = payload.size >= p64.size && memcmp(payload.str, p64.str, p64.size) == 0;
    if ((!is32 && !is64) || payload.str[payload.size - 1] != '>')
        return create_string_attr(parser, name, payload);
    size_t cap = 1;
    for (size_t i = 0; i < payload.size; i++) if (payload.str[i] == ',') cap++;
    int64_t *vals = arena_new_array(parser->arena, int64_t, cap);
    size_t n = 0;
    size_t i = p32.size;
    while (i < payload.size && (payload.str[i] == ' ' || payload.str[i] == ':')) i++;
    while (i < payload.size && payload.str[i] != '>') {
        bool neg = false;
        if (payload.str[i] == '-') { neg = true; i++; }
        if (i >= payload.size || payload.str[i] < '0' || payload.str[i] > '9')
            return create_string_attr(parser, name, payload);
        // Accumulate the magnitude unsigned; a value out of the element
        // type's range keeps the attribute as text, exactly as written.
        uint64_t mag = 0;
        uint64_t limit = is64 ? (neg ? (uint64_t)1 << 63 : ((uint64_t)1 << 63) - 1)
                              : (neg ? (uint64_t)1 << 31 : ((uint64_t)1 << 31) - 1);
        while (i < payload.size && payload.str[i] >= '0' && payload.str[i] <= '9') {
            uint64_t d = (uint64_t)(payload.str[i++] - '0');
            if (mag > (limit - d) / 10) return create_string_attr(parser, name, payload);
            mag = mag * 10 + d;
        }
        vals[n++] = neg ? (int64_t)(0 - mag) : (int64_t)mag;
        while (i < payload.size && (payload.str[i] == ' ' || payload.str[i] == ',')) i++;
    }
    MLIR_AttributeHandle a;
    if (is64) {
        a = MLIR_CreateAttributeDenseI64Array(parser->ctx, vals, n);
    } else {
        int32_t *v32 = arena_new_array(parser->arena, int32_t, cap);
        for (size_t k = 0; k < n; k++) v32[k] = (int32_t)vals[k];
        a = MLIR_CreateAttributeDenseI32Array(parser->ctx, v32, n);
    }
    return MLIR_Named(MLIR_InternAttrKey(parser->ctx, name), a);
}

void operation_append_attribute(Parser *parser, MLIR_OpHandle op, MLIR_NamedAttr attr) {
    if (attr.value == MLIR_INVALID_HANDLE) return;
    MLIR_SetOpAttr(parser->ctx, op, attr.key, attr.value);
}

MLIR_ValueHandle lookup_or_create_value(Parser *parser, string reg, string default_type) {
    MLIR_ValueHandle val = symbol_table_lookup(&parser->symbol_table, reg);
    if (val == MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle ty = MLIR_INVALID_HANDLE;
        if (default_type.size > 0) {
            ty = mlir_type_create_from_string(parser->ctx, default_type);
        }
        val = MLIR_CreateValueBlockArg(parser->ctx, reg, 0, ty, MLIR_INVALID_HANDLE);
    }
    return val;
}

void append_attr(Parser *parser, MLIR_NamedAttr **attrs, size_t *n, size_t *cap, MLIR_NamedAttr attr) {
    if (attr.value == MLIR_INVALID_HANDLE) return;
    size_t new_cap = (*cap == 0) ? 4 : *cap;
    if (*attrs == MLIR_INVALID_HANDLE) {
        *attrs = arena_new_array(parser->arena, MLIR_NamedAttr, new_cap);
        *cap = new_cap;
    } else if (*n >= *cap) {
        new_cap = (*cap) * 2;
        MLIR_NamedAttr *new_attrs = arena_new_array(parser->arena, MLIR_NamedAttr, new_cap);
        for (size_t i = 0; i < *n; i++) new_attrs[i] = (*attrs)[i];
        *attrs = new_attrs;
        *cap = new_cap;
    }
    (*attrs)[(*n)++] = attr;
}

void attr_list_init_from_op(Parser *parser, MLIR_OpHandle op, MLIR_NamedAttr **attrs, size_t *n, size_t *cap) {
    size_t count = MLIR_GetOpNumAttributes(parser->ctx, op);
    if (count > 0) {
        *cap = count + 4;
        *attrs = arena_new_array(parser->arena, MLIR_NamedAttr, *cap);
        for (size_t i = 0; i < count; i++)
            (*attrs)[i] = MLIR_Named(MLIR_GetOpAttributeKey(parser->ctx, op, i),
                                     MLIR_GetOpAttribute(parser->ctx, op, i));
        *n = count;
    }
}

// Include the extracted functions

// Helper function to parse attributes from <{...}> blocks
void parse_angle_brace_attributes(Parser *parser, MLIR_NamedAttr **attributes, size_t *n_attributes, size_t *attributes_capacity) {
    if (!parser_peek(parser, TK_LANGLE)) return;

    // Lookahead for '<{' sequence
    uint64_t save_first = parser->first, save_last = parser->last, save_cur = parser->cur; TokenType save_sym = parser->sym;
    parser_expect(parser, TK_LANGLE);
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        if (!*attributes) {
            *attributes_capacity = 4;
            *attributes = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
        }
        while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                string attr_name = parser_token_str(parser);
                parser_next_token(parser);
                if (parser_peek(parser, TK_EQUAL)) {
                    parser_expect(parser, TK_EQUAL);
                    if (*n_attributes >= *attributes_capacity) {
                        *attributes_capacity *= 2;
                        MLIR_NamedAttr *new_attrs = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
                        for (size_t i = 0; i < *n_attributes; i++) new_attrs[i] = (*attributes)[i];
                        *attributes = new_attrs;
                    }
                    string payload = parse_attr_payload(parser);
                    (*attributes)[(*n_attributes)++] = create_payload_attr(parser, attr_name, payload);
                }
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        if (parser_peek(parser, TK_RBRACE)) parser_expect(parser, TK_RBRACE);
        if (parser_peek(parser, TK_RANGLE)) parser_expect(parser, TK_RANGLE);
    } else {
        // Not actually an attribute block; rewind
        parser->first = save_first;
        parser->last = save_last;
        parser->cur = save_cur;
        parser->sym = save_sym;
    }
}

// Helper function to parse attributes from {...} blocks
void parse_brace_attributes(Parser *parser, MLIR_NamedAttr **attributes, size_t *n_attributes, size_t *attributes_capacity) {
    if (!parser_peek(parser, TK_LBRACE)) return;

    parser_expect(parser, TK_LBRACE);

    if (!*attributes) {
        *attributes_capacity = 4;
        *attributes = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
    }

    while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
            string attr_name = parser_token_str(parser);
            parser_next_token(parser);

            if (parser_peek(parser, TK_EQUAL)) {
                parser_expect(parser, TK_EQUAL);

                // Grow array if needed
                if (*n_attributes >= *attributes_capacity) {
                    *attributes_capacity *= 2;
                    MLIR_NamedAttr *new_attrs = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
                    for (size_t i = 0; i < *n_attributes; i++) new_attrs[i] = (*attributes)[i];
                    *attributes = new_attrs;
                }

                // Parse attribute value
                string payload = parse_attr_payload(parser);

                (*attributes)[(*n_attributes)++] = create_payload_attr(parser, attr_name, payload);
            }

            if (parser_peek(parser, TK_COMMA)) {
                parser_expect(parser, TK_COMMA);
            }
        } else {
            parser_next_token(parser);
        }
    }

    if (parser_peek(parser, TK_RBRACE)) {
        parser_expect(parser, TK_RBRACE);
    }
}

// Helper function to parse result types from : and -> syntax
void parse_result_types(Parser *parser, MLIR_TypeHandle **result_types, size_t *n_result_types,
                              MLIR_NamedAttr **attributes, size_t *n_attributes, size_t *attributes_capacity,
                              MLIR_OpType op_type, MLIR_OpHandle op_for_attributes) {
    // Parse result type after ':'
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Handle forms like ": (type, ...) -> type"
        if (parser_peek(parser, TK_LPAREN)) {
            // Parse and capture the first type inside parentheses
            parser_expect(parser, TK_LPAREN);
            string src_sig = str_lit("");
            (void)parse_type_string(parser, &src_sig);
            // Consume until matching ')'
            int depth = 1;
            while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            }

            // Optional arrow and a result type
            if (parser_peek(parser, TK_ARROW)) {
                parser_expect(parser, TK_ARROW);
            }

            string type_str = str_lit("");
            if (parse_type_string(parser, &type_str)) {
                MLIR_TypeHandle type = mlir_type_create_from_string(parser->ctx, type_str);
                if (result_types) {
                    *result_types = arena_new_array(parser->arena, MLIR_TypeHandle, 1);
                    (*result_types)[0] = type;
                    *n_result_types = 1;
                }
            }

            // Record that the signature used parenthesized operand types
            if (attributes && n_attributes && attributes_capacity) {
                // Building arrays for later use
                if (!*attributes) {
                    *attributes_capacity = 4;
                    *attributes = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
                }
                if (*n_attributes >= *attributes_capacity) {
                    *attributes_capacity *= 2;
                    MLIR_NamedAttr *new_attrs = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
                    for (size_t i = 0; i < *n_attributes; i++) new_attrs[i] = (*attributes)[i];
                    *attributes = new_attrs;
                }
                (*attributes)[(*n_attributes)++] = create_bool_attr(parser, str_lit("_sig_parens"), true);
                if (*n_attributes >= *attributes_capacity) {
                    *attributes_capacity *= 2;
                    MLIR_NamedAttr *new_attrs = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
                    for (size_t i = 0; i < *n_attributes; i++) new_attrs[i] = (*attributes)[i];
                    *attributes = new_attrs;
                }
                (*attributes)[(*n_attributes)++] = create_string_attr(parser, str_lit("_sig_src"), src_sig);
            } else if (op_for_attributes) {
                // Directly append to existing operation
                operation_append_attribute(parser, op_for_attributes, create_bool_attr(parser, str_lit("_sig_parens"), true));
                operation_append_attribute(parser, op_for_attributes, create_string_attr(parser, str_lit("_sig_src"), src_sig));
            }
        } else if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
            // Parse a type token (could be src type or result type)
            string type_left = str_lit("");
            parse_type_string(parser, &type_left);

            // Handle special form: "type * type -> result"
            if (parser_peek(parser, TK_STAR)) {
                parser_expect(parser, TK_STAR);
                string type_right_mul = str_lit("");
                parse_type_string(parser, &type_right_mul);
                // Expect arrow and a result type
                if (parser_peek(parser, TK_ARROW)) parser_expect(parser, TK_ARROW);
                string type_res = str_lit("");
                if (parse_type_string(parser, &type_res)) {
                    MLIR_TypeHandle type = mlir_type_create_from_string(parser->ctx, type_res);
                    if (result_types) {
                        *result_types = arena_new_array(parser->arena, MLIR_TypeHandle, 1);
                        (*result_types)[0] = type;
                        *n_result_types = 1;
                    }
                }
            } else if (parser_peek(parser, TK_ARROW)) {
                // Form ": <src-type> -> <result-type>"
                parser_expect(parser, TK_ARROW);
                string type_right = str_lit("");
                if (parse_type_string(parser, &type_right)) {
                    MLIR_TypeHandle type = mlir_type_create_from_string(parser->ctx, type_right);
                    if (result_types) {
                        *result_types = arena_new_array(parser->arena, MLIR_TypeHandle, 1);
                        (*result_types)[0] = type;
                        *n_result_types = 1;
                    }
                }
                // Record source signature string for classic printing
                if (attributes && n_attributes && attributes_capacity) {
                    // Building arrays for later use
                    if (!*attributes) {
                        *attributes_capacity = 4;
                        *attributes = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
                    }
                    if (*n_attributes >= *attributes_capacity) {
                        *attributes_capacity *= 2;
                        MLIR_NamedAttr *new_attrs = arena_new_array(parser->arena, MLIR_NamedAttr, *attributes_capacity);
                        for (size_t i = 0; i < *n_attributes; i++) new_attrs[i] = (*attributes)[i];
                        *attributes = new_attrs;
                    }
                    (*attributes)[(*n_attributes)++] = create_string_attr(parser, str_lit("_sig_src"), type_left);
                } else if (op_for_attributes) {
                    // Directly append to existing operation
                    operation_append_attribute(parser, op_for_attributes, create_string_attr(parser, str_lit("_sig_src"), type_left));
                }
            } else if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
                // Casts like ": src_ty to dst_ty"; set result to dst_ty
                parser_expect(parser, TK_NAME);
                string type_dst = str_lit("");
                if (parse_type_string(parser, &type_dst)) {
                    MLIR_TypeHandle type = mlir_type_create_from_string(parser->ctx, type_dst);
                    if (result_types) {
                        *result_types = arena_new_array(parser->arena, MLIR_TypeHandle, 1);
                        (*result_types)[0] = type;
                        *n_result_types = 1;
                    }
                }
            } else if (parser_peek(parser, TK_COMMA)) {
                // Operand type list ": type, type, ..." — consume conservatively
                while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                    parser_next_token(parser);
                }
            } else {
                // Treat as single result type ": type" for most ops,
                // but do NOT override for compare ops like arith.cmpi where
                // the colon type is the operand type, not the result.
                if (op_type != OP_TYPE_ARITH_CMPI) {
                    MLIR_TypeHandle type = mlir_type_create_from_string(parser->ctx, type_left);
                    if (result_types) {
                        *result_types = arena_new_array(parser->arena, MLIR_TypeHandle, 1);
                        (*result_types)[0] = type;
                        *n_result_types = 1;
                    }
                }
            }
        }
    }

    // Or optional result type directly after operands via '-> type'
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        string type_str = str_lit("");
        if (parse_type_string(parser, &type_str)) {
            MLIR_TypeHandle type = mlir_type_create_from_string(parser->ctx, type_str);
            if (result_types) {
                *result_types = arena_new_array(parser->arena, MLIR_TypeHandle, 1);
                (*result_types)[0] = type;
                *n_result_types = 1;
            }
        }
    }
}

// Helper function to parse location from loc(...) syntax
MLIR_LocationHandle parse_optional_location(Parser *parser) {
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        return parse_loc(parser);
    }
    return MLIR_INVALID_HANDLE;
}

void parse_generic_attrs_and_result_type(Parser *parser,
                                          MLIR_NamedAttr **attributes,
                                          size_t *n_attributes,
                                          size_t *attributes_capacity,
                                          MLIR_TypeHandle **result_types,
                                          size_t *n_result_types,
                                          MLIR_OpType op_type) {
    // Parse attributes from both <{...}> and {...} blocks
    // These functions will append to existing arrays if provided
    parse_angle_brace_attributes(parser, attributes, n_attributes, attributes_capacity);
    parse_brace_attributes(parser, attributes, n_attributes, attributes_capacity);

    // Parse result types using output parameters
    parse_result_types(parser, result_types, n_result_types, attributes, n_attributes, attributes_capacity, op_type, MLIR_INVALID_HANDLE);
}


MLIR_OpHandle parse_operation(Parser *parser) {

    int64_t recorded_source_line = -1;
    MLIR_ValueHandle *lhs_results = MLIR_INVALID_HANDLE;
    string *lhs_result_names = NULL;
    size_t n_lhs_results = 0;
    size_t n_new_results_from_parser = 0;

    // Skip empty lines and attributes
    while (
        parser_peek(parser, TK_NEWLINE) ||
        parser_peek(parser, TK_HASH_NAME)
            ) {
        if (parser_peek(parser, TK_HASH_NAME)) {
            // Capture location definitions like "#loc = loc(...)" instead of discarding
            string hash_name = parser_token_str(parser);
            parser_next_token(parser); // consume '#loc...'
            if (parser_peek(parser, TK_EQUAL)) {
                parser_next_token(parser); // consume '='
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                    MLIR_LocationHandle loc_def = parse_loc(parser);
                    if (loc_def) {
                        LocationMap_insert(parser->arena, &parser->location_map, hash_name, loc_def);
                        if (hash_name.size == 4 && strncmp(hash_name.str, "#loc", 4) == 0) {
                            parser->unnumbered_loc_def = loc_def;
                        }
                    }
                }
            }
            // Consume rest of the line
            while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
            if (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
        } else if (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        } else {
            // Shouldn't happen
            abort();
        }
    }

    // Record source line start for this operation
    {
        int64_t pos = (int64_t)parser->first;
        while (pos > 0) {
            unsigned char c = parser->input[pos - 1];
            if (c == '\n' || c == '\r') break;
            pos--;
        }
        recorded_source_line = pos;
    }

    // Parse return registers if any
    MLIR_ValueHandle result_value = MLIR_INVALID_HANDLE;
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        size_t result_count = 1;
        if (parser_peek(parser, TK_COLON)) {
            parser_expect(parser, TK_COLON);
            string count_str = parser_token_str(parser);
            parser_expect(parser, TK_INTEGER);
            // Parse the integer to get the count
            result_count = 0;
            for (size_t i = 0; i < count_str.size; i++) {
                result_count = result_count * 10 + (count_str.str[i] - '0');
            }
        }
        parser_expect(parser, TK_EQUAL);

        // Reserve result_count slots and record the LHS register name(s);
        // the real value handles are populated when the op is created.
        lhs_results = arena_new_array(parser->arena, MLIR_ValueHandle, result_count);
        lhs_result_names = arena_new_array(parser->arena, string, result_count);
        for (size_t i = 0; i < result_count; i++) {
            lhs_results[i] = MLIR_INVALID_HANDLE;
            lhs_result_names[i] = (i == 0) ? reg_name : (string){0};
        }
        n_lhs_results = result_count;
        result_value = MLIR_INVALID_HANDLE;
    }

    // Parse operation name
    string opname = str_lit("");
    if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
        strbuf name_buf = strbuf_make();
        strbuf_append(parser->arena, &name_buf, parser_token_str(parser));
        parser_next_token(parser);
        // Accumulate dotted segments into full opname, e.g., tt.reduce.return
        while (parser_peek(parser, TK_DOT)) {
            parser_expect(parser, TK_DOT);
            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                strbuf_append(parser->arena, &name_buf, str_lit("."));
                strbuf_append(parser->arena, &name_buf, parser_token_str(parser));
                parser_next_token(parser);
            } else {
                break;
            }
        }
        opname = strbuf_to_string(name_buf);
    } else if (parser_peek(parser, TK_STRING)) {
        opname = parser_token_str(parser);
        opname = str_substr(opname, 1, opname.size-2);
        parser_expect(parser, TK_STRING);
    } else {
        parser_error(parser,
            format(parser->arena,
                str_lit("Expected operation name (TK_NAME, TK_NAME_DOT_NAME or TK_STRING), got {}"),
                tokentype_to_string(parser->sym)
            ), parser->first, parser->last);
    }


    // The op's type: its static type, or a dynamic type of the context for
    // other names. The classic syntax's `call` is the op `func.call`, and a
    // builtin op may omit its dialect (builtin is the default dialect).
    MLIR_OpType op_type;
    if (str_eq(opname, str_lit("call"))) op_type = OP_TYPE_FUNC_CALL;
    else if (str_eq(opname, str_lit("unrealized_conversion_cast")))
        op_type = OP_TYPE_UNREALIZED_CONVERSION_CAST;
    else op_type = MLIR_RegisterOpType(parser->ctx, opname);
    int op_kind = parser_op_kind(&parser->op_types, op_type);

    // Capture trailing comment from the current line
    string trailing_comment = str_lit("");
    {
        int64_t line_start = recorded_source_line;
        // Find end of this line
        int64_t line_end = line_start;
        while (parser->input[line_end] != '\0' && parser->input[line_end] != '\n' && parser->input[line_end] != '\r') {
            line_end++;
        }
        if (line_end > line_start) {
            // Search for // within this line
            int64_t comment_pos = -1;
            for (int64_t i = line_start; i + 1 < line_end; i++) {
                if (parser->input[i] == '/' && parser->input[i + 1] == '/') { comment_pos = i; break; }
            }
            if (comment_pos >= 0) {
                int64_t begin = comment_pos;
                // include preceding spaces to keep formatting
                while (begin > line_start && parser->input[begin - 1] == ' ') begin--;
                int64_t len = line_end - begin;
                if (len > 0) {
                    trailing_comment = str_from_cstr_len_view((char*)parser->input + begin, len);
                }
            }
        }
    }

    OperationParserParams params = {
        .ctx = parser->ctx,
        .arena = parser->arena,
        .op_type = op_type,
        .op_kind = op_kind,
        .lhs_results = lhs_results,
        .lhs_result_names = lhs_result_names,
        .n_lhs_results = n_lhs_results,
        .unnumbered_loc_def = parser->unnumbered_loc_def,
        .source_line_start = recorded_source_line,
        .trailing_comment = trailing_comment
    };
    OperationParserResult parsed = {0};

    // Debug: Print all operation names


    // First we handle specific ops with special parsing rules
    switch (op_kind) {
        case PARSER_OP_TT_FUNC:
            parsed = parse_tt_func_op(parser, &params);
            break;
        case PARSER_OP_GPU_LAUNCH:
            parsed = parse_gpu_launch_op(parser, &params);
            break;
        case OP_TYPE_SCF_IF:
            parsed = parse_scf_if_op(parser, &params);
            break;
        case OP_TYPE_SCF_FOR:
            parsed = parse_scf_for_op(parser, &params);
            break;
        case OP_TYPE_SCF_WHILE:
            parsed = parse_scf_while_op(parser, &params);
            break;
        case OP_TYPE_ARITH_CONSTANT:
            parsed = parse_arith_constant_op(parser, &params);
            break;
        case OP_TYPE_ARITH_CMPI:
            parsed = parse_arith_cmpi_op(parser, &params);
            break;
        case OP_TYPE_ARITH_ADDI:
        case OP_TYPE_ARITH_MULI:
        case OP_TYPE_ARITH_ADDF:
        case OP_TYPE_ARITH_SUBI:
        case OP_TYPE_ARITH_SUBF:
        case OP_TYPE_ARITH_MULF:
        case OP_TYPE_ARITH_DIVI:
        case OP_TYPE_ARITH_DIVF:
            parsed = parse_arith_binary_op(parser, &params);
            break;
        case OP_TYPE_ARITH_SELECT:
            parsed = parse_arith_select_op(parser, &params);
            break;
        case PARSER_OP_TT_GET_PROGRAM_ID:
            parsed = parse_tt_get_program_id_op(parser, &params);
            break;
        case PARSER_OP_TT_SPLAT:
            parsed = parse_tt_splat_op(parser, &params);
            break;
        case PARSER_OP_TT_MAKE_RANGE:
            parsed = parse_tt_make_range_op(parser, &params);
            break;
        case PARSER_OP_TT_ADDPTR:
            parsed = parse_tt_addptr_op(parser, &params);
            break;
        case PARSER_OP_TT_LOAD:
            parsed = parse_tt_load_op(parser, &params);
            break;
        case PARSER_OP_TT_STORE:
            parsed = parse_tt_store_op(parser, &params);
            break;
        case PARSER_OP_TT_CALL:
            parsed = parse_tt_call_op(parser, &params);
            break;
        case OP_TYPE_FUNC_FUNC:
            parsed = parse_func_func_op(parser, &params);
            break;
        case OP_TYPE_FUNC_CALL:
            parsed = parse_func_call_op(parser, &params);
            break;
        case PARSER_OP_AFFINE_FOR:
            parsed = parse_affine_for_op(parser, &params);
            break;
        case PARSER_OP_MEMREF_LOAD:
            parsed = parse_memref_load_op(parser, &params);
            break;
        case PARSER_OP_MEMREF_STORE:
            parsed = parse_memref_store_op(parser, &params);
            break;
        case PARSER_OP_VECTOR_PRINT:
            parsed = parse_vector_print_op(parser, &params);
            break;
        case PARSER_OP_STD_CONSTANT:
            parsed = parse_std_constant_op(parser, &params);
            break;
        case PARSER_OP_TT_REDUCE:
            parsed = parse_tt_reduce_op(parser, &params);
            break;
        case PARSER_OP_TT_REDUCE_RETURN:
        case PARSER_OP_TT_RETURN:
        case PARSER_OP_STD_RETURN:
        case OP_TYPE_FUNC_RETURN:
        case OP_TYPE_RETURN:
            parsed = parse_return_op(parser, &params);
            break;
        case PARSER_OP_TENSOR_EXTRACT:
            parsed = parse_tensor_extract_op(parser, &params);
            break;
        case OP_TYPE_CF_BR:
            parsed = parse_cf_br_op(parser, &params);
            break;
        case OP_TYPE_CF_COND_BR:
            parsed = parse_cf_cond_br_op(parser, &params);
            break;
        case PARSER_OP_LINALG_FILL:
        case PARSER_OP_LINALG_COPY:
            parsed = parse_linalg_fill_op(parser, &params);
            break;
        case PARSER_OP_AFFINE_LOAD:
            parsed = parse_affine_load_op(parser, &params);
            break;
        case PARSER_OP_INDEX_CONSTANT:
            parsed = parse_index_constant_op(parser, &params);
            break;
        case PARSER_OP_TENSOR_SPLAT:
            parsed = parse_tensor_splat_op(parser, &params);
            break;
        case PARSER_OP_TENSOR_COLLAPSE_SHAPE:
            parsed = parse_tensor_collapse_shape_op(parser, &params);
            break;
        case OP_TYPE_SCF_YIELD:
            parsed = parse_scf_yield_op(parser, &params);
            break;
        default:
            // Generic/unregistered operations
            parsed = parse_generic_op(parser, &params);
            break;
    }

    assert(parsed.operation != MLIR_INVALID_HANDLE);
    MLIR_OpHandle op = parsed.operation;
    n_new_results_from_parser = parsed.n_results;

    // Handle return value(s) for all operations
    assert(parsed.operation != MLIR_INVALID_HANDLE);

    bool has_lhs = n_lhs_results > 0;
    if (has_lhs) {
        if (parsed.results && n_new_results_from_parser > 0) {
            result_value = parsed.results[0];
        } else if (lhs_results && lhs_results[0] != MLIR_INVALID_HANDLE) {
            result_value = lhs_results[0];
        } else if (MLIR_GetOpNumResults(parser->ctx, op) > 0) {
            result_value = MLIR_GetOpResult(parser->ctx, op, 0);
        }
    }

    if (has_lhs && n_new_results_from_parser > 0) {
        // Parser returned results - for named results, add to symbol table
        for (size_t i = 0; i < parsed.n_results; i++) {
            if (parsed.results[i] != MLIR_INVALID_HANDLE) {
                string reg_name = MLIR_GetValueRegisterName(parser->ctx, parsed.results[i]);
                if (reg_name.size > 0) {
                    symbol_table_add_value(parser->arena, &parser->symbol_table, reg_name, parsed.results[i]);
                }
            }
        }
    } else if (!has_lhs && n_new_results_from_parser > 0) {
        // Operation produces results but no SSA name was provided - this is invalid MLIR
        parser_error(parser, str_lit("Operation produces results but no SSA name provided on left-hand side"), parser->first, parser->last);
    } else if (has_lhs) {
        if (MLIR_GetOpNumResultTypes(parser->ctx, op) > 0 && result_value != MLIR_INVALID_HANDLE) {
            symbol_table_add_value(parser->arena, &parser->symbol_table, MLIR_GetValueRegisterName(parser->ctx, result_value), result_value);
        } else {
            parser_error(parser, str_lit("Result Value parsed on LHS but no Type present on RHS"), parser->first, parser->last);
        }
    }

    // Only capture comments for operations that definitely should have them
    // This conservative approach avoids the comment duplication issue
    bool should_capture = false;

    if (should_capture && MLIR_GetOpTrailingComment(parser->ctx, op).size == 0) {
        // Only capture comments if we can find "//" in the rest of the current line
        // after some whitespace (to ensure it belongs to this operation)
        string text = str_from_cstr_view((char*)parser->input);
        int64_t scan_start = (int64_t)parser->last + 1;
        if (scan_start < (int64_t)text.size) {
            int64_t line_end = scan_start - 1; // Default: no text to scan
            // Find the end of the current line (stop at first newline)
            for (int64_t i = scan_start; i < (int64_t)text.size; i++) {
                char ch = text.str[i];
                if (ch == '\n' || ch == '\r') {
                    line_end = i - 1;
                    break;
                }
                line_end = i; // Keep extending until we hit newline
            }


            // Look for "//" in the remaining part of this line
            bool found_comment = false;
            int64_t comment_start = -1;
            for (int64_t i = scan_start; i <= line_end - 1; i++) {
                if (text.str[i] == '/' && i + 1 <= line_end && text.str[i+1] == '/') {
                    // Found comment - include preceding whitespace
                    comment_start = i;
                    while (comment_start > scan_start && text.str[comment_start - 1] == ' ') {
                        comment_start--;
                    }
                    found_comment = true;
                    break;
                }
            }

            if (found_comment && comment_start >= 0) {
                int64_t len = line_end - comment_start + 1;
                if (len > 0) {
                }
            }
        }
    }

    return op;
}

MLIR_OpHandle MLIR_ParseTextClassic(MLIR_Context *ctx, string text) {
    MLIR_LocationMap *locmap = NULL;
    return mlir_parse_module(ctx, (const char*)text.str, text.size, &locmap);
}
