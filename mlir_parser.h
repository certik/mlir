#pragma once

#include <base/arena.h>
#include <base/string.h>

#include "tokenizer.h"


#ifdef __cplusplus
extern "C" {
#endif


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

// TODO: make all ** just *, since the type is known

typedef struct Operation {
    string opname;
    Type **result_types;
    uint64_t n_result_types;
    ValueRef **operands;
    uint64_t n_operands;
    Region **regions;
    uint64_t n_regions;
} Operation;
typedef Operation* OperationPtr;
DEFINE_VECTOR_FOR_TYPE(OperationPtr, VecOperation)

typedef struct Block {
    Operation **operations;
    uint64_t n_operations;
} Block;

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
