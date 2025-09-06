#pragma once

#include <base/arena.h>
#include <base/string.h>
#include <base/hashtable.h>
#include <base/vector.h>

#include "tokenizer.h"


#ifdef __cplusplus
extern "C" {
#endif

/*
TODO:
* Currently we have a vector of pointers
* Instead, have a vector of indices, store all operations in one vector
* An idea: one can inline the structs at the end of
the parent struct, but that will make it hard to add more later, and each struct will have a different size, so not possible to uniformly store them in a vector by value
* Best is to have one vector for all operations
  and then use either indices or pointers.
* An issue with using pointers is that you can't double and copy the vector allocation. Thus using indices is probably the best way to do that.
* One can use a pool allocator to reuse space when operations are removed. One can move things around.
* We still need an arena to allocate a vector of indices for each Operation/Region/Block
* Indices also allow easy serializing/deserializing
*/


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

// Forward declare ValueRef for hashtable
typedef struct ValueRef ValueRef;

// Define hashtable for string -> ValueRef* mapping
#define SymbolTable_HASH string_hash
#define SymbolTable_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, ValueRef*, SymbolTable)

// Scoped symbol table for SSA values
typedef struct ScopedSymbolTable {
    SymbolTable *scopes;
    size_t num_scopes;
    size_t scope_capacity;
} ScopedSymbolTable;

// Forward declare Location for hashtable
typedef struct Location Location;

// Location map for named location references
#define LocationMap_HASH string_hash
#define LocationMap_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, Location*, LocationMap)

typedef struct {
    Arena *arena;
    unsigned char *input;
    TokenType sym;
    uint64_t cur;
    uint64_t first, last;
    ScopedSymbolTable symbol_table;
    LocationMap location_map;  // For #locN -> Location mapping
    int next_loc_id;          // Counter for generating #locN IDs
    Location *unnumbered_loc_def; // Optional: definition of unnumbered '#loc' at file start
} Parser;


typedef enum {
    // Core ops
    OP_TYPE_UNREGISTERED = 0,  // For dynamic/unregistered operations
    OP_TYPE_MODULE,            // Module operation

    // Arithmetic dialect
    OP_TYPE_ARITH_ADDI,
    OP_TYPE_ARITH_SUBI,
    OP_TYPE_ARITH_MULI,
    OP_TYPE_ARITH_DIVI,
    OP_TYPE_ARITH_ADDF,
    OP_TYPE_ARITH_SUBF,
    OP_TYPE_ARITH_MULF,
    OP_TYPE_ARITH_DIVF,
    OP_TYPE_ARITH_CONSTANT,
    OP_TYPE_ARITH_CMPI,
    OP_TYPE_ARITH_CMPF,
    OP_TYPE_ARITH_SELECT,

    // Memory dialect
    OP_TYPE_MEMREF_LOAD,
    OP_TYPE_MEMREF_STORE,
    OP_TYPE_MEMREF_ALLOC,
    OP_TYPE_MEMREF_DEALLOC,

    // Control flow
    OP_TYPE_CF_BR,
    OP_TYPE_CF_COND_BR,
    OP_TYPE_CF_SWITCH,

    // Function dialect
    OP_TYPE_FUNC_FUNC,
    OP_TYPE_FUNC_RETURN,
    OP_TYPE_FUNC_CALL,

    // SCF dialect
    OP_TYPE_SCF_FOR,
    OP_TYPE_SCF_WHILE,
    OP_TYPE_SCF_IF,
    OP_TYPE_SCF_YIELD,

    // Triton dialect
    OP_TYPE_TT_GET_PROGRAM_ID,
    OP_TYPE_TT_LOAD,
    OP_TYPE_TT_STORE,
    OP_TYPE_TT_MAKE_RANGE,
    OP_TYPE_TT_SPLAT,
    OP_TYPE_TT_ADDPTR,
    OP_TYPE_TT_RETURN,
    OP_TYPE_TT_FUNC,
    OP_TYPE_TT_CALL,
    OP_TYPE_TT_REDUCE,

    // GPU dialect
    OP_TYPE_GPU_LAUNCH,

    // Affine dialect
    OP_TYPE_AFFINE_FOR,
    OP_TYPE_AFFINE_LOAD,

    // Vector dialect
    OP_TYPE_VECTOR_PRINT,

    // Standard dialect
    OP_TYPE_STD_CONSTANT,
    OP_TYPE_STD_RETURN,

    // Tensor dialect
    OP_TYPE_TENSOR_EXTRACT,
    OP_TYPE_TENSOR_SPLAT,
    OP_TYPE_TENSOR_COLLAPSE_SHAPE,

    // Linalg dialect
    OP_TYPE_LINALG_FILL,

    // Index dialect
    OP_TYPE_INDEX_CONSTANT,

    // Return operations
    OP_TYPE_RETURN,
    OP_TYPE_TT_REDUCE_RETURN,

    OP_TYPE_COUNT  // Total number of operation types
} OpType;

typedef struct Region Region;

// Type kinds for MLIR type system
typedef enum {
    TYPE_KIND_UNKNOWN,
    TYPE_KIND_OPAQUE,
    TYPE_KIND_INTEGER,
    TYPE_KIND_FLOAT,
    TYPE_KIND_MEMREF,
    TYPE_KIND_TENSOR,
    TYPE_KIND_FUNCTION,
    TYPE_KIND_INDEX,
    TYPE_KIND_POINTER
} TypeKind;

// MLIR Type representation
typedef struct Type {
    TypeKind kind;
    union {
        struct {
            uint32_t width;     // Bit width for integers
            bool is_signed;
        } integer;
        struct {
            uint32_t width;     // 16, 32, 64, etc.
            bool is_bfloat;     // true for bf16
        } floating;
        struct {
            struct Type *element_type;
            int64_t *shape;     // NULL-terminated or use rank
            uint32_t rank;
        } shaped;  // For memref and tensor
        struct {
            struct Type *element_type;
            uint32_t address_space;  // For !tt.ptr<type, address_space>
            bool has_address_space;
        } pointer;
    } data;
} Type;

// Attribute representation
typedef struct Attribute {
    enum {
        ATTR_KIND_INTEGER,
        ATTR_KIND_FLOAT,
        ATTR_KIND_STRING,
        ATTR_KIND_BOOL,
        ATTR_KIND_ARRAY,
        ATTR_KIND_DICT
    } kind;
    union {
        int64_t integer_value;
        double float_value;
        string string_value;
        bool bool_value;
        struct {
            struct Attribute **elements;
            size_t count;
        } array;
    } data;
    string name;
} Attribute;

// Named attribute for dictionaries
typedef struct NamedAttribute {
    const string *name;
    Attribute *value;
} NamedAttribute;

// Location information for MLIR constructs
typedef enum {
    LOC_KIND_UNKNOWN,
    LOC_KIND_FILE,      // loc("file.py":line:col)
    LOC_KIND_NAME,      // loc("name")
    LOC_KIND_CALLSITE,  // loc(callsite(...))
    LOC_KIND_FUSED,     // loc(fused[...])
    LOC_KIND_REF        // loc(#locN) - reference to named location
} LocationKind;

typedef struct Location {
    LocationKind kind;
    union {
        struct {
            string filename;
            int line;
            int column;
        } file;
        struct {
            string name;
        } name;
        struct {
            int ref_id;  // For #locN references
        } ref;
    } data;
    
    // For storing the original location string for printing
    string original_text;
} Location;


typedef enum ValueKind {
    BLOCK_ARG,
    OP_RESULT
} ValueKind;

struct ValueRef {
    ValueKind kind;
    // TODO: Use an index
    void* def; // Block* or Operation* that produced it
    uint32_t result_index;   // Which result of the operation
    // TODO: use Type by value
    Type *type;              // Type of this value

    // For parsed register names like %0, %c16_i32. These names are not unique
    // in an MLIR module. Two different Values in different regions can have the
    // same name. If this is used for printing, then extra care must be taken
    // that the printed Value name is unique.
    string register_name;

    // Optional per-argument metadata for classic printing
    Location *location;           // e.g., arg loc("file":line:col)
    bool has_divisibility;        // tt.divisibility attribute present
    int64_t divisibility_value;   // value for tt.divisibility
    Type *divisibility_type;      // type for tt.divisibility value (e.g., i32)
    
    bool has_max_divisibility;    // tt.max_divisibility attribute present
    int64_t max_divisibility_value; // value for tt.max_divisibility
    Type *max_divisibility_type;  // type for tt.max_divisibility value (e.g., i32)

    // Maybe later:
    //Operation **users;
    //uint64_t n_users;
};

// Note: we use ** instead of *, because Value has a pointer to Operation or
// Block, so we can't easily move them later. When parsing we do not know how
// many items we will need, so we use pointers, which we can grow by copying as
// needed.

typedef struct Operation {
    OpType op_type; // Enum for registered ops
    // Use indices here
    ValueRef **operands;
    uint64_t n_operands;
    // TODO: use Type by value
    Type **result_types;
    uint64_t n_result_types;
    Attribute **attributes;
    uint64_t n_attributes;
    Region **regions;
    uint64_t n_regions;
    string opname; // Only used for unregistered ops

    // Result values produced by this operation
    ValueRef **results;
    uint64_t n_results;
    
    // Location information
    Location *location;
    // Optional: definition for unnumbered '#loc' header captured pre-module
    Location *unnumbered_loc_def;

    // Optional trailing comment captured from source line (e.g., " // note")
    string trailing_comment;
} Operation;
DEFINE_VECTOR_FOR_TYPE(Operation*, VecOperation)
DEFINE_VECTOR_FOR_TYPE(ValueRef*, VecValueRef)

typedef struct Block {
    Operation **operations;
    uint64_t n_operations;
    ValueRef **arguments;
    uint64_t n_arguments;
} Block;
DEFINE_VECTOR_FOR_TYPE(Block*, VecBlock)

struct Region {
    Block **blocks;
    uint64_t n_blocks;
};

// Symbol table functions
void symbol_table_init(Arena *arena, ScopedSymbolTable *st);
void symbol_table_push_scope(Arena *arena, ScopedSymbolTable *st);
void symbol_table_pop_scope(ScopedSymbolTable *st);
void symbol_table_add_value(Arena *arena, ScopedSymbolTable *st, string name, ValueRef *value);
ValueRef* symbol_table_lookup(ScopedSymbolTable *st, string name);

// Helper function to create properly initialized ValueRef
ValueRef* create_value_ref(Arena *arena, ValueKind kind);

// Specialized parsing functions
void parse_gpu_launch(Parser *parser, Operation *op);

string tokentype_to_string(TokenType tt);
void parser_init(Arena *arena, Parser *parser, string text);
void parser_next_token(Parser *parser);
bool parser_peek(Parser *parser, TokenType s);
void parser_expect(Parser *parser, TokenType s);
string parser_token_str(Parser *parser);
void parser_error(Parser *parser, string msg, uint64_t first, uint64_t last);
void parser_warning(Parser *parser, string msg, uint64_t first, uint64_t last);
Operation* parse_module(Parser *parser);
Location* parse_loc(Parser *parser);
Operation* parse_operation(Parser *parser);
Region* parse_region(Parser *parser);
Block* parse_block(Parser *parser);

string op_type_to_string(OpType type);
OpType op_string_to_type(string name);

// TODO: use Type by value
Type* parse_type_from_string(Arena *arena, string type_str);
string type_to_string(Arena *arena, Type *type);

#ifdef __cplusplus
}
#endif
