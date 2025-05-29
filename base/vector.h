#pragma once

#include <base/arena.h>

#ifdef WITH_BASE_ASSERT
static int vec_called_const = 0xdeadbeef;
#endif

#define _IMPL_CONCAT2(a, b) a##b
#define CONCAT2(a, b) _IMPL_CONCAT2(a, b)

#define _IMPL_CONCAT3(a, b, c) a##b##c
#define CONCAT3(a, b, c) _IMPL_CONCAT3(a, b, c)

#define T int64_t

#define VECTOR_NAME CONCAT2(vector_, T)
#define VECTOR_FUNC(suffix) CONCAT3(vector_, T, _##suffix)

typedef struct VECTOR_NAME {
    T* data;
    size_t size, max;
#ifdef WITH_BASE_ASSERT
    int reserve_called;
#endif
} VECTOR_NAME;

void VECTOR_FUNC(reserve)(Arena *arena, VECTOR_NAME *vec, size_t max);
void VECTOR_FUNC(push_back)(Arena *arena, VECTOR_NAME *vec, T x);

#undef T
#undef VECTOR_NAME
#undef VECTOR_FUNC
#undef CONCAT2
#undef _IMPL_CONCAT2
#undef CONCAT3
#undef _IMPL_CONCAT3
