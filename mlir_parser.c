#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"
#include "op_parsers.h"

// Forward decls for helper scanners (moved to op_parsers.h)

// Parse a single MLIR type token into a string.
// Handles:
// - bare types like `i32`, `tensor`
// - dialect types starting with '!' like `!tt.ptr`
// - optional nested angle bracket payloads like `<...>` with proper depth
// On success, consumes the tokens that form the type and writes the
// concatenated textual form to `*out`. Returns true if a type was parsed.
static bool parse_type_string(Parser *parser, string *out);

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
    value->location = NULL;
    value->has_divisibility = false;
    value->divisibility_value = 0;
    value->divisibility_type = NULL;
    value->has_max_divisibility = false;
    value->max_divisibility_value = 0;
    value->max_divisibility_type = NULL;
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
        case OP_TYPE_ARITH_SELECT: return str_lit("arith.select");
        case OP_TYPE_ARITH_BITCAST: return str_lit("arith.bitcast");
        case OP_TYPE_ARITH_SITOFP: return str_lit("arith.sitofp");
        case OP_TYPE_ARITH_EXTSI: return str_lit("arith.extsi");
        case OP_TYPE_ARITH_TRUNCI: return str_lit("arith.trunci");
        case OP_TYPE_ARITH_EXTF: return str_lit("arith.extf");
        case OP_TYPE_ARITH_TRUNCF: return str_lit("arith.truncf");
        case OP_TYPE_ARITH_EXTUI: return str_lit("arith.extui");
        case OP_TYPE_ARITH_MAXF: return str_lit("arith.maxf");
        case OP_TYPE_ARITH_DIVSI: return str_lit("arith.divsi");
        case OP_TYPE_ARITH_REMSI: return str_lit("arith.remsi");
        case OP_TYPE_ARITH_ORI: return str_lit("arith.ori");
        case OP_TYPE_ARITH_MINSI: return str_lit("arith.minsi");
        case OP_TYPE_ARITH_ANDI: return str_lit("arith.andi");
        case OP_TYPE_MATH_EXP: return str_lit("math.exp");
        case OP_TYPE_MATH_LOG: return str_lit("math.log");
        case OP_TYPE_TT_FUNC: return str_lit("tt.func");
        case OP_TYPE_TT_CALL: return str_lit("tt.call");
        case OP_TYPE_TT_REDUCE: return str_lit("tt.reduce");
        case OP_TYPE_TT_BROADCAST: return str_lit("tt.broadcast");
        case OP_TYPE_TT_EXPAND_DIMS: return str_lit("tt.expand_dims");
        case OP_TYPE_TT_DOT: return str_lit("tt.dot");
        case OP_TYPE_TT_PURE_EXTERN_ELEMENTWISE: return str_lit("tt.pure_extern_elementwise");
        case OP_TYPE_GPU_LAUNCH: return str_lit("gpu.launch");
        case OP_TYPE_AFFINE_FOR: return str_lit("affine.for");
        case OP_TYPE_AFFINE_LOAD: return str_lit("affine.load");
        case OP_TYPE_VECTOR_PRINT: return str_lit("vector.print");
        case OP_TYPE_STD_CONSTANT: return str_lit("std.constant");
        case OP_TYPE_STD_RETURN: return str_lit("std.return");
        case OP_TYPE_TENSOR_EXTRACT: return str_lit("tensor.extract");
        case OP_TYPE_TENSOR_SPLAT: return str_lit("tensor.splat");
        case OP_TYPE_TENSOR_COLLAPSE_SHAPE: return str_lit("tensor.collapse_shape");
        case OP_TYPE_LINALG_FILL: return str_lit("linalg.fill");
        case OP_TYPE_INDEX_CONSTANT: return str_lit("index.constant");
        case OP_TYPE_LLVM_MLIR_UNDEF: return str_lit("llvm.mlir.undef");
        case OP_TYPE_RETURN: return str_lit("return");
        case OP_TYPE_TT_REDUCE_RETURN: return str_lit("tt.reduce.return");
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
    } else if (str_eq(opname, str_lit("arith.bitcast"))) {
        return OP_TYPE_ARITH_BITCAST;
    } else if (str_eq(opname, str_lit("arith.sitofp"))) {
        return OP_TYPE_ARITH_SITOFP;
    } else if (str_eq(opname, str_lit("arith.extsi"))) {
        return OP_TYPE_ARITH_EXTSI;
    } else if (str_eq(opname, str_lit("arith.trunci"))) {
        return OP_TYPE_ARITH_TRUNCI;
    } else if (str_eq(opname, str_lit("arith.extf"))) {
        return OP_TYPE_ARITH_EXTF;
    } else if (str_eq(opname, str_lit("arith.truncf"))) {
        return OP_TYPE_ARITH_TRUNCF;
    } else if (str_eq(opname, str_lit("arith.extui"))) {
        return OP_TYPE_ARITH_EXTUI;
    } else if (str_eq(opname, str_lit("arith.maxf"))) {
        return OP_TYPE_ARITH_MAXF;
    } else if (str_eq(opname, str_lit("arith.divsi"))) {
        return OP_TYPE_ARITH_DIVSI;
    } else if (str_eq(opname, str_lit("arith.remsi"))) {
        return OP_TYPE_ARITH_REMSI;
    } else if (str_eq(opname, str_lit("arith.ori"))) {
        return OP_TYPE_ARITH_ORI;
    } else if (str_eq(opname, str_lit("arith.minsi"))) {
        return OP_TYPE_ARITH_MINSI;
    } else if (str_eq(opname, str_lit("arith.andi"))) {
        return OP_TYPE_ARITH_ANDI;
    } else if (str_eq(opname, str_lit("math.exp"))) {
        return OP_TYPE_MATH_EXP;
    } else if (str_eq(opname, str_lit("math.log"))) {
        return OP_TYPE_MATH_LOG;
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
    } else if (str_eq(opname, str_lit("call"))) {
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
    } else if (str_eq(opname, str_lit("arith.select"))) {
        return OP_TYPE_ARITH_SELECT;
    } else if (str_eq(opname, str_lit("tt.func"))) {
        return OP_TYPE_TT_FUNC;
    } else if (str_eq(opname, str_lit("tt.call"))) {
        return OP_TYPE_TT_CALL;
    } else if (str_eq(opname, str_lit("tt.reduce"))) {
        return OP_TYPE_TT_REDUCE;
    } else if (str_eq(opname, str_lit("tt.broadcast"))) {
        return OP_TYPE_TT_BROADCAST;
    } else if (str_eq(opname, str_lit("tt.expand_dims"))) {
        return OP_TYPE_TT_EXPAND_DIMS;
    } else if (str_eq(opname, str_lit("tt.dot"))) {
        return OP_TYPE_TT_DOT;
    } else if (str_eq(opname, str_lit("tt.pure_extern_elementwise"))) {
        return OP_TYPE_TT_PURE_EXTERN_ELEMENTWISE;
    } else if (str_eq(opname, str_lit("gpu.launch"))) {
        return OP_TYPE_GPU_LAUNCH;
    } else if (str_eq(opname, str_lit("affine.for"))) {
        return OP_TYPE_AFFINE_FOR;
    } else if (str_eq(opname, str_lit("affine.load"))) {
        return OP_TYPE_AFFINE_LOAD;
    } else if (str_eq(opname, str_lit("vector.print"))) {
        return OP_TYPE_VECTOR_PRINT;
    } else if (str_eq(opname, str_lit("std.constant"))) {
        return OP_TYPE_STD_CONSTANT;
    } else if (str_eq(opname, str_lit("std.return"))) {
        return OP_TYPE_STD_RETURN;
    } else if (str_eq(opname, str_lit("tensor.extract"))) {
        return OP_TYPE_TENSOR_EXTRACT;
    } else if (str_eq(opname, str_lit("tensor.splat"))) {
        return OP_TYPE_TENSOR_SPLAT;
    } else if (str_eq(opname, str_lit("tensor.collapse_shape"))) {
        return OP_TYPE_TENSOR_COLLAPSE_SHAPE;
    } else if (str_eq(opname, str_lit("linalg.fill"))) {
        return OP_TYPE_LINALG_FILL;
    } else if (str_eq(opname, str_lit("index.constant"))) {
        return OP_TYPE_INDEX_CONSTANT;
    } else if (str_eq(opname, str_lit("llvm.mlir.undef"))) {
        return OP_TYPE_LLVM_MLIR_UNDEF;
    } else if (str_eq(opname, str_lit("return"))) {
        return OP_TYPE_RETURN;
    } else if (str_eq(opname, str_lit("tt.reduce.return"))) {
        return OP_TYPE_TT_REDUCE_RETURN;
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
    LocationMap_init(arena, &parser->location_map, 16);
    parser->next_loc_id = 0;
    parser->unnumbered_loc_def = NULL;
    parser->capture_trailing_comments = false;
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

// Parse a single MLIR type token into a string representation.
// Consumes:
// - A leading TK_NAME/TK_NAME_DOT_NAME or TK_EXCLAMATION (dialect)
// - If leading '!' is present, the following name token
// - An optional angle-bracket payload with proper nesting until matching '>'
static bool parse_type_string(Parser *parser, string *out) {
    if (!(parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION))) {
        return false;
    }

    string ty = parser_token_str(parser);
    parser_next_token(parser);

    // Dialect types starting with '!'
    if (ty.size == 1 && ty.str[0] == '!' && (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME))) {
        ty = str_concat(parser->arena, ty, parser_token_str(parser));
        parser_next_token(parser);
    }

    // Optional angle-bracketed content (supports nesting)
    if (parser_peek(parser, TK_LANGLE)) {
        int depth = 0;
        do {
            if (parser_peek(parser, TK_LANGLE)) depth++;
            else if (parser_peek(parser, TK_RANGLE)) depth--;
            ty = str_concat(parser->arena, ty, parser_token_str(parser));
            parser_next_token(parser);
        } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
    }

    *out = ty;
    return true;
}

Type* parse_type_from_string(Arena *arena, string type_str) {
    Type *type = arena_alloc(arena, Type);

    // Unknown type (printed as '?')
    if (str_eq(type_str, str_lit("?"))) {
        type->kind = TYPE_KIND_UNKNOWN;
    }
    // Opaque/unspecified type (printed as 'unknown')
    else if (str_eq(type_str, str_lit("unknown")) || str_eq(type_str, str_lit("!unknown"))) {
        type->kind = TYPE_KIND_OPAQUE;
    }
    // Parse index type first (before integer check below)
    else if (str_eq(type_str, str_lit("index"))) {
        type->kind = TYPE_KIND_INDEX;
    }
    // Parse integer types like "i32", "i64", etc.
    else if (type_str.size >= 2 && type_str.str[0] == 'i') {
        // Require that all remaining characters are digits
        bool all_digits = true;
        for (size_t i = 1; i < type_str.size; i++) {
            char c = type_str.str[i];
            if (c < '0' || c > '9') { all_digits = false; break; }
        }
        if (!all_digits) {
            // Fallback to default unknown integer width
            type->kind = TYPE_KIND_INTEGER;
            type->data.integer.is_signed = true;
            type->data.integer.width = 32;
            return type;
        }
        type->kind = TYPE_KIND_INTEGER;
        type->data.integer.is_signed = true;

        // Extract width
        string width_str = str_substr(type_str, 1, type_str.size - 1);
        if (str_eq(width_str, str_lit("1"))) type->data.integer.width = 1;
        else if (str_eq(width_str, str_lit("8"))) type->data.integer.width = 8;
        else if (str_eq(width_str, str_lit("16"))) type->data.integer.width = 16;
        else if (str_eq(width_str, str_lit("32"))) type->data.integer.width = 32;
        else if (str_eq(width_str, str_lit("64"))) type->data.integer.width = 64;
        else type->data.integer.width = 32; // Default
    }
    // Parse floating point types like "f32", "f64", etc.
    else if (type_str.size >= 2 && type_str.str[0] == 'f') {
        type->kind = TYPE_KIND_FLOAT;
        type->data.floating.is_bfloat = false;
        // Extract width
        string width_str = str_substr(type_str, 1, type_str.size - 1);
        if (str_eq(width_str, str_lit("16"))) type->data.floating.width = 16;
        else if (str_eq(width_str, str_lit("32"))) type->data.floating.width = 32;
        else if (str_eq(width_str, str_lit("64"))) type->data.floating.width = 64;
        else type->data.floating.width = 32; // Default
    }
    // bfloat16
    else if (str_eq(type_str, str_lit("bf16"))) {
        type->kind = TYPE_KIND_FLOAT;
        type->data.floating.width = 16;
        type->data.floating.is_bfloat = true;
    }
    // Parse pointer types like "!tt.ptr<f32, 1>"
    else if (type_str.size >= 7 && str_eq(str_substr(type_str, 0, 7), str_lit("!tt.ptr"))) {
        type->kind = TYPE_KIND_POINTER;
        type->data.pointer.address_space = 1; // Default
        type->data.pointer.has_address_space = false;
        type->data.pointer.element_type = arena_alloc(arena, Type);

        // Find the content inside < >
        size_t start = 8; // After "!tt.ptr<"
        size_t end = type_str.size - 1; // Before ">"
        if (start < end && type_str.str[7] == '<' && type_str.str[end] == '>') {
            string content = str_substr(type_str, start, end - start);
            // Find comma to separate element type from address space
            size_t comma_pos = content.size;
            for (size_t i = 0; i < content.size; i++) {
                if (content.str[i] == ',') {
                    comma_pos = i;
                    break;
                }
            }

            if (comma_pos < content.size) {
                // Parse element type on the left of comma
                string elem_type_str = str_substr(content, 0, comma_pos);
                // Trim whitespace
                while (elem_type_str.size > 0 && elem_type_str.str[0] == ' ') {
                    elem_type_str = str_substr(elem_type_str, 1, elem_type_str.size - 1);
                }
                while (elem_type_str.size > 0 && elem_type_str.str[elem_type_str.size - 1] == ' ') {
                    elem_type_str = str_substr(elem_type_str, 0, elem_type_str.size - 1);
                }
                type->data.pointer.element_type = parse_type_from_string(arena, elem_type_str);

                // Parse address space from right of comma (trim spaces)
                string addr_str = str_substr(content, comma_pos + 1, content.size - comma_pos - 1);
                while (addr_str.size > 0 && addr_str.str[0] == ' ') {
                    addr_str = str_substr(addr_str, 1, addr_str.size - 1);
                }
                while (addr_str.size > 0 && addr_str.str[addr_str.size - 1] == ' ') {
                    addr_str = str_substr(addr_str, 0, addr_str.size - 1);
                }
                uint32_t as = 0;
                for (size_t i = 0; i < addr_str.size; i++) {
                    if (addr_str.str[i] >= '0' && addr_str.str[i] <= '9') {
                        as = as * 10 + (uint32_t)(addr_str.str[i] - '0');
                    }
                }
                type->data.pointer.address_space = as;
                type->data.pointer.has_address_space = true;
            } else {
                // No comma, assume f32 and address space 1
                type->data.pointer.element_type = parse_type_from_string(arena, content);
                type->data.pointer.address_space = 1;
                type->data.pointer.has_address_space = false;
            }
        } else {
            // Malformed, default to f32 pointer
            type->data.pointer.element_type = parse_type_from_string(arena, str_lit("f32"));
            type->data.pointer.address_space = 1;
            type->data.pointer.has_address_space = false;
        }
    }
    // Parse tensor types like "tensor<4xi32>" or "tensor<4x!tt.ptr<f32,1>>"
    else if (type_str.size >= 6 && str_eq(str_substr(type_str, 0, 6), str_lit("tensor"))) {
        type->kind = TYPE_KIND_TENSOR;

        // Find the content inside < >
        size_t start = 7; // After "tensor<"
        size_t end = type_str.size - 1; // Before ">"
        if (start < end && type_str.str[6] == '<' && type_str.str[end] == '>') {
            string content = str_substr(type_str, start, end - start);

            // Parse dimensions and element type (e.g., "4xi32" or "4x!tt.ptr<f32,1>")
            // Find the last 'x' to separate dimensions from element type
            int last_x_pos = -1;
            int bracket_depth = 0;
            for (int i = content.size - 1; i >= 0; i--) {
                if (content.str[i] == '>') bracket_depth++;
                else if (content.str[i] == '<') bracket_depth--;
                else if (content.str[i] == 'x' && bracket_depth == 0) {
                    last_x_pos = i;
                    break;
                }
            }

            if (last_x_pos > 0) {
                string elem_type_str = str_substr(content, last_x_pos + 1, content.size - last_x_pos - 1);
                type->data.shaped.element_type = parse_type_from_string(arena, elem_type_str);

                // Parse shape dims from tokens separated by 'x'
                string shape_str = str_substr(content, 0, last_x_pos);
                uint32_t dims = 1;
                for (size_t i = 0; i < shape_str.size; i++) if (shape_str.str[i] == 'x') dims++;
                type->data.shaped.rank = dims;
                type->data.shaped.shape = arena_alloc_array(arena, int64_t, dims);
                size_t pos = 0; uint32_t dim_idx = 0;
                while (pos <= shape_str.size && dim_idx < dims) {
                    size_t next = pos;
                    while (next < shape_str.size && shape_str.str[next] != 'x') next++;
                    string tok = str_substr(shape_str, pos, next - pos);
                    if (tok.size == 1 && tok.str[0] == '?') {
                        type->data.shaped.shape[dim_idx++] = -1;
                    } else {
                        int64_t val = 0;
                        for (size_t j = 0; j < tok.size; j++) {
                            if (tok.str[j] >= '0' && tok.str[j] <= '9') {
                                val = val * 10 + (tok.str[j] - '0');
                            }
                        }
                        type->data.shaped.shape[dim_idx++] = val;
                    }
                    pos = next + 1;
                }
            } else {
                // No 'x' found, assume it's just the element type
                type->data.shaped.element_type = parse_type_from_string(arena, content);
                type->data.shaped.rank = 0;
                type->data.shaped.shape = NULL;
            }
        } else {
            // Malformed tensor, default to f32
            type->data.shaped.element_type = parse_type_from_string(arena, str_lit("f32"));
            type->data.shaped.rank = 0;
            type->data.shaped.shape = NULL;
        }
    }
    // Parse memref types like "memref<2x3xf32>"
    else if (type_str.size >= 6 && str_eq(str_substr(type_str, 0, 6), str_lit("memref"))) {
        type->kind = TYPE_KIND_MEMREF;
        type->data.shaped.element_type = NULL;
        type->data.shaped.shape = NULL;
        type->data.shaped.rank = 0;
        // Extract inside of angle brackets
        size_t start = 7; // after "memref<"
        size_t end = type_str.size - 1; // before '>'
        if (start < end && type_str.str[6] == '<' && type_str.str[end] == '>') {
            string content = str_substr(type_str, start, end - start);
            // Find last 'x' not inside nested '<>' to split shape and element type
            int last_x_pos = -1;
            int bracket_depth = 0;
            for (int i = (int)content.size - 1; i >= 0; i--) {
                char c = content.str[i];
                if (c == '>') bracket_depth++;
                else if (c == '<') bracket_depth--;
                else if (c == 'x' && bracket_depth == 0) { last_x_pos = i; break; }
            }
            if (last_x_pos > 0) {
                string elem_type_str = str_substr(content, last_x_pos + 1, content.size - last_x_pos - 1);
                type->data.shaped.element_type = parse_type_from_string(arena, elem_type_str);
                string shape_str = str_substr(content, 0, last_x_pos);
                // Count dims
                uint32_t dims = 1;
                for (size_t i = 0; i < shape_str.size; i++) if (shape_str.str[i] == 'x') dims++;
                type->data.shaped.rank = dims;
                type->data.shaped.shape = arena_alloc_array(arena, int64_t, dims);
                // Parse each dimension token separated by 'x'
                size_t pos = 0, dim_idx = 0;
                while (pos <= shape_str.size && dim_idx < dims) {
                    size_t next = pos;
                    while (next < shape_str.size && shape_str.str[next] != 'x') next++;
                    string tok = str_substr(shape_str, pos, next - pos);
                    // Parse integer dim or '?' for dynamic
                    if (tok.size == 1 && tok.str[0] == '?') {
                        type->data.shaped.shape[dim_idx++] = -1;
                    } else {
                        int64_t val = 0;
                        for (size_t j = 0; j < tok.size; j++) {
                            if (tok.str[j] >= '0' && tok.str[j] <= '9') {
                                val = val * 10 + (tok.str[j] - '0');
                            }
                        }
                        type->data.shaped.shape[dim_idx++] = val;
                    }
                    pos = next + 1; // skip 'x'
                }
            } else {
                // No shape dims, treat content as element type
                type->data.shaped.element_type = parse_type_from_string(arena, content);
                type->data.shaped.rank = 0;
                type->data.shaped.shape = NULL;
            }
        }
    }
    // Default to integer if unrecognized
    else {
        type->kind = TYPE_KIND_INTEGER;
        type->data.integer.width = 32;
        type->data.integer.is_signed = true;
    }
    return type;
}

string type_to_string(Arena *arena, Type *type) {
    if (!type) {
        return str_lit("null");
    }

    // Debug output to help track crashes
    // printf("type_to_string: kind=%d\n", type->kind);

    switch (type->kind) {
        case TYPE_KIND_UNKNOWN:
            return str_lit("?");
        case TYPE_KIND_OPAQUE:
            return str_lit("unknown");
        case TYPE_KIND_INTEGER:
            if (type->data.integer.is_signed) {
                return format(arena, str_lit("i{}"), (int64_t)type->data.integer.width);
            } else {
                return format(arena, str_lit("ui{}"), (int64_t)type->data.integer.width);
            }
        case TYPE_KIND_FLOAT:
            if (type->data.floating.is_bfloat && type->data.floating.width == 16) {
                return str_lit("bf16");
            }
            return format(arena, str_lit("f{}"), (int64_t)type->data.floating.width);
        case TYPE_KIND_TENSOR:
            if (type->data.shaped.element_type) {
                string elem_str = type_to_string(arena, type->data.shaped.element_type);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    // Build shape string like "4x" or "4x2x"
                    string shape_str = str_lit("");
                    for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                        int64_t dim = type->data.shaped.shape[i];
                        if (dim < 0) {
                            shape_str = str_concat(arena, shape_str, str_lit("?x"));
                        } else {
                            shape_str = str_concat(arena, shape_str, format(arena, str_lit("{}x"), dim));
                        }
                    }
                    return format(arena, str_lit("tensor<{}{}>"), shape_str, elem_str);
                } else {
                    return format(arena, str_lit("tensor<{}>"), elem_str);
                }
            }
            return str_lit("tensor<?>");
        case TYPE_KIND_MEMREF:
            if (type->data.shaped.element_type) {
                string elem_str = type_to_string(arena, type->data.shaped.element_type);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    string shape_str = str_lit("");
                    for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                        int64_t dim = type->data.shaped.shape[i];
                        if (dim < 0) {
                            shape_str = str_concat(arena, shape_str, str_lit("?x"));
                        } else {
                            shape_str = str_concat(arena, shape_str, format(arena, str_lit("{}x"), dim));
                        }
                    }
                    return format(arena, str_lit("memref<{}{}>"), shape_str, elem_str);
                } else {
                    return format(arena, str_lit("memref<{}>"), elem_str);
                }
            }
            return str_lit("memref<?>");
        case TYPE_KIND_POINTER:
            if (type->data.pointer.element_type) {
                string elem_str = type_to_string(arena, type->data.pointer.element_type);
                // Only show address space if it's explicitly set and not the default (0)
                if (type->data.pointer.has_address_space && type->data.pointer.address_space != 0) {
                    return format(arena, str_lit("!tt.ptr<{}, {}>"), elem_str, (int64_t)type->data.pointer.address_space);
                } else {
                    return format(arena, str_lit("!tt.ptr<{}>"), elem_str);
                }
            }
            return str_lit("!tt.ptr<?>");
        case TYPE_KIND_INDEX:
            return str_lit("index");
        case TYPE_KIND_FUNCTION:
            return str_lit("function");
        default:
            return str_lit("unknown");
    }
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
                    string type_name = str_lit("");
                    if (parse_type_string(parser, &type_name)) {
                        block_arg->type = parse_type_from_string(parser->arena, type_name);
                    } else {
                        block_arg->type = parse_type_from_string(parser->arena, str_lit("i32"));
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
        // Capture trailing inline comment based on the original line where the op started
        // This avoids mis-associating comments if tokenization peeks into the next line.
        if (op && op->source_line_start >= 0) {
            int64_t line_start = op->source_line_start;
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
                        op->trailing_comment = str_from_cstr_len_view((char*)parser->input + begin, len);
                    }
                }
            }
        }
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
    // Capture any top-of-file #loc definitions before the module
    Location *loc0_def = NULL;
    while (parser_peek(parser, TK_HASH_NAME)) {
        string hash_name = parser_token_str(parser);
        parser_next_token(parser); // consume '#name'
        if (parser_peek(parser, TK_EQUAL)) {
            parser_next_token(parser); // consume '='
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                Location *loc_def = parse_loc(parser);
                if (loc_def) {
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

    Operation *op = parse_operation(parser);
    if (op->op_type != OP_TYPE_MODULE) {
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
                Location *loc_def = parse_loc(parser);
                if (loc_def) {
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
    
    // Attach unnumbered '#loc' definition captured during initial scan or in parse_operation
    if (!loc0_def) loc0_def = parser->unnumbered_loc_def;
    op->unnumbered_loc_def = loc0_def;
    return op;
}

// parse loc()
Location* parse_loc(Parser *parser) {
    Arena *arena = parser->arena;
    Location *loc = arena_alloc(arena, Location);
    loc->kind = LOC_KIND_UNKNOWN;
    loc->original_text = str_lit("");
    
    
    parser_expect(parser, TK_NAME); // 'loc'
    parser_expect(parser, TK_LPAREN);
    
    // Check what kind of location this is
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("callsite"))) {
        // Capture loc(callsite(...)) verbatim as original_text
        string text = str_lit("loc(");
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
                if (need_space) text = str_concat(parser->arena, text, str_lit(" "));
            }
            text = str_concat(parser->arena, text, tok);
            parser_next_token(parser);
        }
        parser_expect(parser, TK_RPAREN);
        text = str_concat(parser->arena, text, str_lit(")"));
        loc->kind = LOC_KIND_UNKNOWN;
        loc->original_text = text;
        return loc;
    } else if (parser_peek(parser, TK_STRING)) {
        // loc("filename":line:col) or loc("name")
        string filename = parser_token_str(parser);
        parser_next_token(parser);
        
        if (parser_peek(parser, TK_COLON)) {
            // File location: loc("filename":line:col)
            parser_next_token(parser); // consume ':'
            
            if (parser_peek(parser, TK_INTEGER)) {
                loc->kind = LOC_KIND_FILE;
                loc->data.file.filename = filename;
                
                // Parse line number
                string line_str = parser_token_str(parser);
                loc->data.file.line = atoi(line_str.str);
                parser_next_token(parser);
                
                if (parser_peek(parser, TK_COLON)) {
                    parser_next_token(parser); // consume ':'
                    if (parser_peek(parser, TK_INTEGER)) {
                        // Parse column number
                        string col_str = parser_token_str(parser);
                        loc->data.file.column = atoi(col_str.str);
                        parser_next_token(parser);
                    }
                }
            }
        } else {
            // Named location: loc("name")
            loc->kind = LOC_KIND_NAME;
            loc->data.name.name = filename;
        }
    } else if (parser_peek(parser, TK_HASH_NAME)) {
        // Reference location: loc(#locN)
        string hash_name = parser_token_str(parser);
        parser_next_token(parser); // consume '#locN'
        loc->kind = LOC_KIND_REF;
        // Extract number from "#locN" format
        if (hash_name.size > 4 && strncmp(hash_name.str, "#loc", 4) == 0) {
            loc->data.ref.ref_id = atoi(hash_name.str + 4);
        } else {
            loc->data.ref.ref_id = 0;
        }
    } else {
        // Unknown location format, just consume tokens until ')'
        while (!(parser_peek(parser, TK_RPAREN))) {
            parser_next_token(parser);
        }
    }
    
    parser_expect(parser, TK_RPAREN);
    
    // Capture original text for printing (simple reconstruction)
    if (loc->kind == LOC_KIND_FILE) {
        loc->original_text = format(parser->arena, str_lit("loc({}:{}:{})"),
            loc->data.file.filename, (int64_t)loc->data.file.line, (int64_t)loc->data.file.column);
    } else if (loc->kind == LOC_KIND_NAME) {
        loc->original_text = format(parser->arena, str_lit("loc(\"{}\")"), loc->data.name.name);
    } else if (loc->kind == LOC_KIND_REF) {
        loc->original_text = format(parser->arena, str_lit("loc(#loc{})"), (int64_t)loc->data.ref.ref_id);
    } else {
        loc->original_text = str_lit("loc(unknown)");
    }
    
    return loc;
}


Operation* parse_operation(Parser *parser) {
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
    op->unnumbered_loc_def = NULL;
    op->location = NULL;
    op->trailing_comment = str_lit("");
    op->source_line_start = -1;

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
                    Location *loc_def = parse_loc(parser);
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
        op->source_line_start = pos;
    }

    // Parse return registers if any
    ValueRef *result_value = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_name = parser_token_str(parser);
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
    string opname = str_lit("");
    if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
        opname = parser_token_str(parser);
        parser_next_token(parser);
        // Accumulate dotted segments into full opname, e.g., tt.reduce.return
        while (parser_peek(parser, TK_DOT)) {
            parser_expect(parser, TK_DOT);
            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                opname = str_concat(parser->arena, opname, str_lit("."));
                opname = str_concat(parser->arena, opname, parser_token_str(parser));
                parser_next_token(parser);
            } else {
                break;
            }
        }
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


    // Set op_type based on operation name
    op->op_type = op_string_to_type(opname);
    if (op->op_type == OP_TYPE_UNREGISTERED) {
        op->opname = opname;
    }


    // First we handle specific opnames with special parsing rules
    switch (op->op_type) {
        case OP_TYPE_TT_FUNC:
            parse_tt_func(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_GPU_LAUNCH:
            parse_gpu_launch(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_SCF_IF:
            parse_scf_if(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_SCF_FOR:
            parse_scf_for(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_SCF_WHILE:
            parse_scf_while(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_ARITH_CONSTANT:
            parse_arith_constant(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_ARITH_CMPI:
            parse_arith_cmpi(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_ARITH_ADDI:
        case OP_TYPE_ARITH_MULI:
        case OP_TYPE_ARITH_ADDF:
        case OP_TYPE_ARITH_SUBI:
        case OP_TYPE_ARITH_SUBF:
        case OP_TYPE_ARITH_MULF:
        case OP_TYPE_ARITH_DIVI:
        case OP_TYPE_ARITH_DIVF:
            parse_arith_binary(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_ARITH_SELECT:
            parse_arith_select(parser, op);
            break;
        case OP_TYPE_TT_GET_PROGRAM_ID:
            parse_tt_get_program_id(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_TT_SPLAT:
            parse_tt_splat(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_TT_MAKE_RANGE:
            parse_tt_make_range(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_TT_ADDPTR:
        case OP_TYPE_TT_LOAD:
            parse_tt_addptr_load_store(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_TT_STORE:
            parse_tt_store(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_TT_CALL:
            parse_tt_call(parser, op);
            break;
        case OP_TYPE_FUNC_FUNC:
            parse_func_func(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_FUNC_CALL:
            parse_func_call(parser, op);
            break;
        case OP_TYPE_AFFINE_FOR:
            parse_affine_for(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_MEMREF_LOAD:
            parse_memref_load_or_store(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_MEMREF_STORE:
            parse_memref_store(parser, op);
            break;
        case OP_TYPE_VECTOR_PRINT:
            parse_vector_print(parser, op);
            break;
        case OP_TYPE_STD_CONSTANT:
            parse_std_constant(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_TT_REDUCE:
            parse_tt_reduce(parser, op);
            break;
        case OP_TYPE_TT_REDUCE_RETURN:
        case OP_TYPE_TT_RETURN:
        case OP_TYPE_STD_RETURN:
        case OP_TYPE_FUNC_RETURN:
        case OP_TYPE_RETURN:
            parse_return_operation(parser, op);
            break;
        case OP_TYPE_TENSOR_EXTRACT:
            parse_tensor_extract(parser, op);
            parse_generic_attrs_and_result_type(parser, op);
            break;
        case OP_TYPE_CF_BR:
            parse_cf_br(parser, op);
            break;
        case OP_TYPE_CF_COND_BR:
            parse_cf_cond_br(parser, op);
            break;
        case OP_TYPE_LINALG_FILL:
            parse_linalg_fill(parser, op);
            break;
        case OP_TYPE_AFFINE_LOAD:
            parse_affine_load(parser, op);
            break;
        case OP_TYPE_INDEX_CONSTANT:
            parse_index_constant(parser, op);
            break;
        case OP_TYPE_TENSOR_SPLAT:
            parse_tensor_splat(parser, op);
            break;
        case OP_TYPE_TENSOR_COLLAPSE_SHAPE:
            parse_tensor_collapse_shape(parser, op);
            break;
        case OP_TYPE_SCF_YIELD:
            parse_scf_yield(parser, op);
            break;
        default:
            // Generic/unregistered operations
            parse_generic_operation(parser, op);
            break;
    }

    // Handle return value(s) for all operations
    if (result_value) {
        if (op->n_result_types > 0) {
            // If op->n_result_types > 0, then op->result_types must be set:
            assert(op->result_types != NULL);
            assert(op->result_types[0] != NULL);
            // Set the type
            result_value->type = op->result_types[0];
            result_value->def = op;
            symbol_table_add_value(parser->arena, &parser->symbol_table, result_value->register_name, result_value);

            // Link result to operation
            op->n_results = 1;
            op->results = arena_alloc_array(parser->arena, ValueRef*, 1);
            op->results[0] = result_value;
        } else {
            parser_error(parser, str_lit("Result Value parsed on LHS but no Type present on RHS"), parser->first, parser->last);
        }
    } else {
        if (op->n_result_types > 0) {
            parser_error(parser, str_lit("Result Type parsed on RHS but no result Value on LHS"), parser->first, parser->last);
        }
    }

    // Only capture comments for operations that definitely should have them
    // This conservative approach avoids the comment duplication issue
    bool should_capture = false;
    
    if (should_capture && op->trailing_comment.size == 0) {
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
                    op->trailing_comment = str_from_cstr_len_view(text.str + comment_start, len);
                }
            }
        }
    }

    return op;
}
