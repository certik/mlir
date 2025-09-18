#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
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

void symbol_table_add_value(Arena *arena, ScopedSymbolTable *st, string name, MlirValue *value) {
    if (st->num_scopes == 0) {
        // Create a default scope if none exists
        symbol_table_push_scope(arena, st);
    }
    SymbolTable_insert(arena, &st->scopes[st->num_scopes - 1], name, value);
}

MlirValue* symbol_table_lookup(ScopedSymbolTable *st, string name) {
    // Search from innermost to outermost scope
    for (size_t i = st->num_scopes; i > 0; i--) {
        MlirValue **found = SymbolTable_get(&st->scopes[i - 1], name);
        if (found && *found) {
            return *found;
        }
    }
    return NULL;
}

MlirValue* create_value_ref(Arena *arena, ValueKind kind) {
    MlirValue *value = mlir_value_create(arena, kind);
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




MlirOperation* parse_operation(Parser *parser);

MlirBlock* parse_block(Parser *parser) {
    VecValue block_args;
    VecValue_reserve(parser->arena, &block_args, 4);

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
                    MlirValue *block_arg = mlir_value_create(parser->arena, BLOCK_ARG);
                    mlir_value_set_register_name(block_arg, arg_name.str, arg_name.size);
                    mlir_value_set_result_index(block_arg, (uint32_t)block_args.size);
                    mlir_value_set_def(block_arg, NULL);
                    // Parse argument type
                    string type_name = str_lit("");
                    MlirType *arg_type = NULL;
                    if (parse_type_string(parser, &type_name)) {
                        arg_type = mlir_type_create_from_string(parser->arena, type_name);
                    } else {
                        arg_type = mlir_type_create_from_string(parser->arena, str_lit("i32"));
                    }
                    mlir_value_set_type(block_arg, arg_type);

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
    VecOperation operations;
    VecOperation_reserve(parser->arena, &operations, 16);
    while (! (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_CARET_NAME))) {
        MlirOperation *op = parse_operation(parser);
        // Capture trailing inline comment based on the original line where the op started
        // This avoids mis-associating comments if tokenization peeks into the next line.
        if (op && mlir_operation_get_source_line_start(op) >= 0) {
            int64_t line_start = mlir_operation_get_source_line_start(op);
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
                        string comment = str_from_cstr_len_view((char*)parser->input + begin, len);
                        mlir_operation_set_trailing_comment(op, comment.str, comment.size);
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

    MlirBlock *block = mlir_block_create(parser->arena);
    for (size_t i = 0; i < block_args.size; i++) {
        mlir_block_add_argument(parser->arena, block, block_args.data[i]);
    }
    for (size_t i = 0; i < operations.size; i++) {
        mlir_block_add_operation(parser->arena, block, operations.data[i]);
    }

    return block;
}

// Parses a region from { to } inclusive
MlirRegion* parse_region(Parser *parser) {
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for this region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    VecBlock blocks;
    VecBlock_reserve(parser->arena, &blocks, 8);
    while (!parser_peek(parser, TK_RBRACE)) {
        MlirBlock *block = parse_block(parser);
        VecBlock_push_back(parser->arena, &blocks, block);
    }
    parser_expect(parser, TK_RBRACE);

    // Pop scope when leaving region
    symbol_table_pop_scope(&parser->symbol_table);

    MlirRegion *region = mlir_region_create(parser->arena);
    for (size_t i = 0; i < blocks.size; i++) {
        mlir_region_add_block(parser->arena, region, blocks.data[i]);
    }

    return region;
}

MlirOperation* parse_module(Parser *parser) {
    // Capture any top-of-file #loc definitions before the module
    MlirLocation *loc0_def = NULL;
    while (parser_peek(parser, TK_HASH_NAME)) {
        string hash_name = parser_token_str(parser);
        parser_next_token(parser); // consume '#name'
        if (parser_peek(parser, TK_EQUAL)) {
            parser_next_token(parser); // consume '='
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                MlirLocation *loc_def = parse_loc(parser);
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

    MlirOperation *op = parse_operation(parser);
    if (mlir_operation_get_type(op) != OP_TYPE_MODULE) {
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
                MlirLocation *loc_def = parse_loc(parser);
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
    mlir_operation_set_unnumbered_loc_def(op, loc0_def);
    return op;
}

MlirOperation *mlir_parse_module(Arena *arena, const char *input, size_t input_len, MlirLocationMap **out_location_map) {
    Parser *parser = arena_alloc(arena, Parser);
    string input_string = {
        .str = (char*)input,
        .size = input_len
    };
    parser_init(arena, parser, input_string);
    MlirOperation *module = parse_module(parser);
    if (out_location_map) {
        MlirLocationMap *map_wrapper = arena_alloc(arena, MlirLocationMap);
        map_wrapper->impl = &parser->location_map;
        *out_location_map = map_wrapper;
    }
    return module;
}

const char *mlir_tokentype_to_string(int token_type) {
    string s = tokentype_to_string((TokenType)token_type);
    return s.str;
}

size_t mlir_location_map_size(const MlirLocationMap *location_map) {
    if (!location_map) return 0;
    const LocationMap *lm = (const LocationMap*)location_map->impl;
    if (!lm) return 0;
    return lm->size;
}

size_t mlir_location_map_collect(const MlirLocationMap *location_map, string *out_keys, MlirLocation **out_locs, size_t max) {
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
MlirLocation* parse_loc(Parser *parser) {
    Arena *arena = parser->arena;
    MlirLocation *loc = mlir_location_create(arena);
    mlir_location_set_kind(loc, MLIR_LOC_UNKNOWN);
    mlir_location_set_original_text(loc, str_lit(""));


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
        mlir_location_set_kind(loc, MLIR_LOC_UNKNOWN);
        mlir_location_set_original_text(loc, text);
        return loc;
    } else if (parser_peek(parser, TK_STRING)) {
        // loc("filename":line:col) or loc("name")
        string filename = parser_token_str(parser);
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

                mlir_location_set_file_data(loc, filename, line, column);
            }
        } else {
            // Named location: loc("name")
            mlir_location_set_name_data(loc, filename);
        }
    } else if (parser_peek(parser, TK_HASH_NAME)) {
        // Reference location: loc(#locN)
        string hash_name = parser_token_str(parser);
        parser_next_token(parser); // consume '#locN'
        int ref_id = 0;
        if (hash_name.size > 4 && strncmp(hash_name.str, "#loc", 4) == 0) {
            ref_id = atoi(hash_name.str + 4);
        }
        mlir_location_set_ref_id(loc, ref_id);
    } else {
        // Unknown location format, just consume tokens until ')'
        while (!(parser_peek(parser, TK_RPAREN))) {
            parser_next_token(parser);
        }
    }

    parser_expect(parser, TK_RPAREN);

    // Capture original text for printing (simple reconstruction)
    MlirLocationKind stored_kind = mlir_location_get_kind(loc);
    if (stored_kind == MLIR_LOC_FILE) {
        mlir_location_set_original_text(loc,
            format(parser->arena, str_lit("loc({}:{}:{})"),
                   mlir_location_get_file_filename(loc),
                   (int64_t)mlir_location_get_file_line(loc),
                   (int64_t)mlir_location_get_file_column(loc)));
    } else if (stored_kind == MLIR_LOC_NAME) {
        mlir_location_set_original_text(loc,
            format(parser->arena, str_lit("loc(\"{}\")"), mlir_location_get_name(loc)));
    } else if (stored_kind == MLIR_LOC_REF) {
        mlir_location_set_original_text(loc,
            format(parser->arena, str_lit("loc(#loc{})"), (int64_t)mlir_location_get_ref_id(loc)));
    } else {
        mlir_location_set_original_text(loc, str_lit("loc(unknown)"));
    }

    return loc;
}


MlirOperation* parse_operation(Parser *parser) {
    MlirOperation *op = mlir_op_create(parser->arena, OP_TYPE_UNREGISTERED);
    mlir_operation_set_trailing_comment(op, "", 0);
    mlir_operation_set_source_line_start(op, -1);
    mlir_operation_set_location(op, NULL);
    mlir_operation_set_unnumbered_loc_def(op, NULL);

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
                    MlirLocation *loc_def = parse_loc(parser);
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
        mlir_operation_set_source_line_start(op, pos);
    }

    // Parse return registers if any
    MlirValue *result_value = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        if (parser_peek(parser, TK_COLON)) {
            parser_expect(parser, TK_COLON);
            parser_expect(parser, TK_INTEGER);
        }
        parser_expect(parser, TK_EQUAL);

        // Create MlirValue for the result and add to symbol table
        result_value = mlir_value_create(parser->arena, MLIR_VALUE_OP_RESULT);
        mlir_value_set_def(result_value, op);
        mlir_value_set_result_index(result_value, 0);
        mlir_value_set_register_name(result_value, reg_name.str, reg_name.size);
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
    OpType op_type = op_string_to_type(opname);
    mlir_operation_set_type(op, op_type);
    if (op_type == OP_TYPE_UNREGISTERED) {
        mlir_operation_set_name(op, opname.str, opname.size);
    }
    
    // Debug: Print all operation names


    // First we handle specific opnames with special parsing rules
    switch (op_type) {
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
        if (mlir_operation_num_result_types(op) > 0) {
            MlirType *res_type = mlir_operation_get_result_type(op, 0);
            assert(res_type != NULL);
            mlir_value_set_type(result_value, res_type);
            mlir_value_set_def(result_value, op);
            symbol_table_add_value(parser->arena, &parser->symbol_table, mlir_value_get_register_name(result_value), result_value);

            // Link result to operation
            MlirValue **results = arena_alloc_array(parser->arena, MlirValue*, 1);
            results[0] = result_value;
            mlir_operation_set_results(op, results, 1);
        } else {
            parser_error(parser, str_lit("Result Value parsed on LHS but no Type present on RHS"), parser->first, parser->last);
        }
    } else {
        if (mlir_operation_num_result_types(op) > 0) {
            parser_error(parser, str_lit("Result Type parsed on RHS but no result Value on LHS"), parser->first, parser->last);
        }
    }

    // Only capture comments for operations that definitely should have them
    // This conservative approach avoids the comment duplication issue
    bool should_capture = false;

    if (should_capture && mlir_operation_get_trailing_comment(op).size == 0) {
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
                    string comment = str_from_cstr_len_view(text.str + comment_start, len);
                    mlir_operation_set_trailing_comment(op, comment.str, comment.size);
                }
            }
        }
    }

    return op;
}
