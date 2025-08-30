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
*/


typedef struct {
    Arena *arena;
    unsigned char *input;
    TokenType sym;
    uint64_t cur;
    uint64_t first, last;
} Parser;


typedef struct Region Region;

typedef struct Type {
    string str; // For now we keep the type as a string
} Type;

typedef enum ValueKind {
    BLOCK_ARG,
    OP_RESULT
} ValueKind;

typedef struct ValueRef {
    ValueKind kind;
    void* def; // Block* or Operation*
    uint64_t index;
} ValueRef;

// Note: we use ** instead of *, because Value has a pointer to Operation or
// Block, so we can't easily move them later. When parsing we do not know how
// many items we will need, so we use pointers, which we can grow by copying as
// needed.

typedef struct Operation {
    string opname;
    Type **result_types;
    uint64_t n_result_types;
    ValueRef **operands;
    uint64_t n_operands;
    Region **regions;
    uint64_t n_regions;
} Operation;
DEFINE_VECTOR_FOR_TYPE(Operation*, VecOperation)

typedef struct Block {
    Operation **operations;
    uint64_t n_operations;
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
