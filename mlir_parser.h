#pragma once

#include <base/arena.h>
#include <base/string.h>

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


typedef struct {
    Arena *arena;
    unsigned char *input;
    TokenType sym;
    uint64_t cur;
    uint64_t first, last;
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

    OP_TYPE_COUNT  // Total number of operation types
} OpType;

typedef struct Region Region;

// Type kinds for MLIR type system
typedef enum {
    TYPE_KIND_INTEGER,
    TYPE_KIND_FLOAT,
    TYPE_KIND_MEMREF,
    TYPE_KIND_TENSOR,
    TYPE_KIND_FUNCTION,
    TYPE_KIND_INDEX
} TypeKind;

// MLIR Type representation
typedef struct Type {
    TypeKind kind;
    string str; // TODO: remove later
    union {
        struct {
            uint32_t width;     // Bit width for integers
            bool is_signed;
        } integer;
        struct {
            uint32_t width;     // 16, 32, 64, etc.
        } floating;
        struct {
            struct Type *element_type;
            int64_t *shape;     // NULL-terminated or use rank
            uint32_t rank;
        } shaped;  // For memref and tensor
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
        const char *string_value;
        bool bool_value;
        struct {
            struct Attribute **elements;
            size_t count;
        } array;
    } data;
    const char *name;
} Attribute;

// Named attribute for dictionaries
typedef struct NamedAttribute {
    const char *name;
    Attribute *value;
} NamedAttribute;


typedef enum ValueKind {
    BLOCK_ARG,
    OP_RESULT
} ValueKind;

typedef struct ValueRef {
    ValueKind kind;
    // use an index
    void* def; // Block* or Operation* that produced it
    uint32_t result_index;   // Which result of the operation
    Type *type;              // Type of this value

    // Maybe later:
    //Operation **users;
    //uint64_t n_users;
} ValueRef;

// Note: we use ** instead of *, because Value has a pointer to Operation or
// Block, so we can't easily move them later. When parsing we do not know how
// many items we will need, so we use pointers, which we can grow by copying as
// needed.

typedef struct Operation {
    OpType op_type; // Enum for registered ops
    // Use indices here
    ValueRef **operands;
    uint64_t n_operands;
    Type **result_types;
    uint64_t n_result_types;
    Attribute **attributes;
    uint64_t n_attributes;
    Region **regions;
    uint64_t n_regions;
    string opname; // Only used for unregistered ops
} Operation;
DEFINE_VECTOR_FOR_TYPE(Operation*, VecOperation)

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

string tokentype_to_string(TokenType tt);
void parser_init(Arena *arena, Parser *parser, string text);
Operation* parse_module(Parser *parser);

#ifdef __cplusplus
}
#endif
