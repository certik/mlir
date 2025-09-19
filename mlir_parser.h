#pragma once

#include <base/arena.h>
#include <base/string.h>
#include <base/hashtable.h>
#include <base/vector.h>

#include "tokenizer.h"
#include "mlir_api.h"


#ifdef __cplusplus
extern "C" {
#endif



// Hash function for strings
static inline size_t string_hash(string str) {
    size_t hash = 5381;
    for (size_t i = 0; i < str.size; i++) {
        hash = ((hash << 5) + hash) + str.str[i];
    }
    return hash;
}

// Equality function for strings
static inline bool string_equal(string a, string b) {
    return str_eq(a, b);
}

// Value kind used for SSA values
typedef enum ValueKind {
    BLOCK_ARG,
    OP_RESULT
} ValueKind;

// Define hashtable for string -> ValueRef* mapping
#define SymbolTable_HASH string_hash
#define SymbolTable_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, MlirValue*, SymbolTable)

// Scoped symbol table for SSA values
typedef struct ScopedSymbolTable {
    SymbolTable *scopes;
    size_t num_scopes;
    size_t scope_capacity;
} ScopedSymbolTable;

// Location map for named location references
#define LocationMap_HASH string_hash
#define LocationMap_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, MlirLocation*, LocationMap)


typedef struct {
    Arena *arena;
    unsigned char *input;
    TokenType sym;
    uint64_t cur;
    uint64_t first, last;
    ScopedSymbolTable symbol_table;
    LocationMap location_map;  // For #locN -> Location mapping
    int next_loc_id;          // Counter for generating #locN IDs
    MlirLocation *unnumbered_loc_def; // Optional: definition of unnumbered '#loc' at file start
    // Parsing mode flag to enable robust trailing comment capture in special contexts
    bool capture_trailing_comments;
} Parser;



// Symbol table functions
void symbol_table_init(Arena *arena, ScopedSymbolTable *st);
void symbol_table_push_scope(Arena *arena, ScopedSymbolTable *st);
void symbol_table_pop_scope(ScopedSymbolTable *st);
void symbol_table_add_value(Arena *arena, ScopedSymbolTable *st, string name, MlirValue *value);
MlirValue* symbol_table_lookup(ScopedSymbolTable *st, string name);

// Helper function to create properly initialized ValueRef
MlirValue* create_value_ref(Arena *arena, ValueKind kind);

// Forward declarations for core IR nodes (opaque here)
// Specialized parsing functions
void parse_gpu_launch(Parser *parser, MlirOperation *op);

string tokentype_to_string(TokenType tt);
void parser_init(Arena *arena, Parser *parser, string text);
void parser_next_token(Parser *parser);
bool parser_peek(Parser *parser, TokenType s);
void parser_expect(Parser *parser, TokenType s);
string parser_token_str(Parser *parser);
void parser_error(Parser *parser, string msg, uint64_t first, uint64_t last);
void parser_warning(Parser *parser, string msg, uint64_t first, uint64_t last);
MlirOperation* parse_module(Parser *parser);
MlirLocation* parse_loc(Parser *parser);
MlirOperation* parse_operation(Parser *parser);
MlirRegion* parse_region(Parser *parser);
MlirBlock* parse_block(Parser *parser);

string op_type_to_string(OpType type);
OpType op_string_to_type(string name);

// TODO: use Type by value
MlirType* parse_type_from_string(Arena *arena, string type_str);
string type_to_string(Arena *arena, MlirType *type);

// Public parser facade
MlirOperation *mlir_parse_module(Arena *arena, const char *input, size_t input_len, MlirLocationMap **out_location_map);
const char *mlir_tokentype_to_string(int token_type);
size_t mlir_location_map_size(const MlirLocationMap *location_map);
size_t mlir_location_map_collect(const MlirLocationMap *location_map, string *out_keys, MlirLocation **out_locs, size_t max);

#ifdef __cplusplus
}
#endif
