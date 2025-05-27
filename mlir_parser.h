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

typedef struct Operation Operation;
struct Operation {
    string opcode;
    Region **regions;
    uint64_t n_regions;
};

typedef struct Block Block;
struct Block {
    Operation **operations;
    uint64_t n_operations;
};

struct Region {
    Block **blocks;
    uint64_t n_blocks;
};


#ifdef __cplusplus
}
#endif
