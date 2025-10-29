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


typedef struct OperationParserParams {
    Arena  *arena;
    MLIR_OpType  op_type;
    string  opname; /* only non-empty for unregistered ops */

    MLIR_Value **lhs_results;
    size_t     n_lhs_results;

    MLIR_Location *unnumbered_loc_def;
    int64_t       source_line_start;
    string        trailing_comment;
} OperationParserParams;

typedef struct OperationParserResult {
    MLIR_Op *operation;
    MLIR_Value    **results;
    size_t         n_results;
} OperationParserResult;

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

// Define hashtable for string -> ValueRef* mapping
#define SymbolTable_HASH string_hash
#define SymbolTable_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, MLIR_Value*, SymbolTable)

// Scoped symbol table for SSA values
typedef struct ScopedSymbolTable {
    SymbolTable *scopes;
    size_t num_scopes;
    size_t scope_capacity;
} ScopedSymbolTable;

// Location map for named location references
#define LocationMap_HASH string_hash
#define LocationMap_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, MLIR_Location*, LocationMap)


typedef struct {
    Arena *arena;
    unsigned char *input;
    TokenType sym;
    uint64_t cur;
    uint64_t first, last;
    ScopedSymbolTable symbol_table;
    LocationMap location_map;  // For #locN -> Location mapping
    int next_loc_id;          // Counter for generating #locN IDs
    MLIR_Location *unnumbered_loc_def; // Optional: definition of unnumbered '#loc' at file start
    // Parsing mode flag to enable robust trailing comment capture in special contexts
    bool capture_trailing_comments;
} Parser;



// Symbol table functions
void symbol_table_init(Arena *arena, ScopedSymbolTable *st);
void symbol_table_push_scope(Arena *arena, ScopedSymbolTable *st);
void symbol_table_pop_scope(ScopedSymbolTable *st);
void symbol_table_add_value(Arena *arena, ScopedSymbolTable *st, string name, MLIR_Value *value);
MLIR_Value* symbol_table_lookup(ScopedSymbolTable *st, string name);

// Forward declarations for core IR nodes (opaque here)
// Specialized parsing functions
void parse_gpu_launch(Parser *parser, MLIR_Op *op);

string tokentype_to_string(TokenType tt);
void parser_init(Arena *arena, Parser *parser, string text);
void parser_next_token(Parser *parser);
bool parser_peek(Parser *parser, TokenType s);
void parser_expect(Parser *parser, TokenType s);
string parser_token_str(Parser *parser);
void parser_error(Parser *parser, string msg, uint64_t first, uint64_t last);
void parser_warning(Parser *parser, string msg, uint64_t first, uint64_t last);
MLIR_Op* parse_module(Parser *parser);
MLIR_Location* parse_loc(Parser *parser);
MLIR_Op* parse_operation(Parser *parser);
MLIR_Region* parse_region(Parser *parser);
MLIR_Block* parse_block(Parser *parser);

bool parse_type_string(Parser *parser, string *out);
void consume_optional_hash_selector(Parser *parser);


const char *string_data_or_null(string s);

bool parse_register_operand(Parser *parser, VecValue *operands, bool allow_hash_selector);
MLIR_Value **finalize_results(const OperationParserParams *params,
                                    MLIR_Op *op,
                                    MLIR_Type **result_types,
                                    size_t n_result_types,
                                    size_t *out_n_results);
MLIR_Attribute *create_string_attr(Parser *parser, string name, string value);
MLIR_Attribute *create_integer_attr(Parser *parser, string name, int64_t value);
MLIR_Attribute *create_float_attr(Parser *parser, string name, double value);
MLIR_Attribute *create_bool_attr(Parser *parser, string name, bool value);
void operation_append_attribute(Parser *parser, MLIR_Op *op, MLIR_Attribute *attr);
MLIR_Value *lookup_or_create_value(Parser *parser, string reg, string default_type);
void append_attr(Parser *parser, MLIR_Attribute ***attrs, size_t *n, size_t *cap, MLIR_Attribute *attr);
void attr_list_init_from_op(Parser *parser, MLIR_Op *op, MLIR_Attribute ***attrs, size_t *n, size_t *cap);
void parse_angle_brace_attributes(Parser *parser, MLIR_Attribute ***attributes, size_t *n_attributes, size_t *attributes_capacity);
void parse_brace_attributes(Parser *parser, MLIR_Attribute ***attributes, size_t *n_attributes, size_t *attributes_capacity);
void parse_result_types(Parser *parser, MLIR_Type ***result_types, size_t *n_result_types,
                              MLIR_Attribute ***attributes, size_t *n_attributes, size_t *attributes_capacity,
                              MLIR_OpType op_type, MLIR_Op *op_for_attributes);
MLIR_Location *parse_optional_location(Parser *parser);
void parse_generic_attrs_and_result_type(Parser *parser,
                                          MLIR_Attribute ***attributes,
                                          size_t *n_attributes,
                                          size_t *attributes_capacity,
                                          MLIR_Type ***result_types,
                                          size_t *n_result_types,
                                          MLIR_OpType op_type);

void consume_optional_hash_selector(Parser *parser);

string op_type_to_string(MLIR_OpType type);
MLIR_OpType op_string_to_type(string name);

// TODO: use Type by value
MLIR_Type* parse_type_from_string(Arena *arena, string type_str);
string type_to_string(Arena *arena, MLIR_Type *type);

// Public parser facade
MLIR_Op *mlir_parse_module(Arena *arena, const char *input, size_t input_len, MLIR_LocationMap **out_location_map);
const char *mlir_tokentype_to_string(int token_type);
size_t MLIR_LocationMapSize(const MLIR_LocationMap *location_map);
size_t MLIR_LocationMapCollect(const MLIR_LocationMap *location_map, string *out_keys, MLIR_Location **out_locs, size_t max);

MLIR_Type *mlir_type_create_from_string(Arena *arena, string type_str);

#ifdef __cplusplus
}
#endif
