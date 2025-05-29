#pragma once

#include <base/arena.h>

#ifdef WITH_BASE_ASSERT
static int vec_called_const = 0xdeadbeef;
#endif

#define VecType int64_t

typedef struct {
    VecType* p;
    size_t n, max;
#ifdef WITH_BASE_ASSERT
    int reserve_called;
#endif
} vector_VecType;

void vector_VecType_reserve(Arena *arena, vector_VecType vec, size_t max); 
void vector_VecType_push_back(Arena *arena, vector_VecType vec, VecType x); 

#undef VecType
