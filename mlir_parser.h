#pragma once

#include "tokenizer.h"
#include "arena.h"

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

typedef struct {
    char *str;
    uint64_t size;
} string;

#define str_lit(S)  (string){.str=(char*)(S), .size=sizeof(S)-1}

typedef struct {
    string opcode;
} Operation;


#ifdef __cplusplus
}
#endif
