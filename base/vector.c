#include <assert.h>
#include <string.h>

#include <base/vector.h>

#define _IMPL_CONCAT2(a, b) a##b
#define CONCAT2(a, b) _IMPL_CONCAT2(a, b)

#define _IMPL_CONCAT3(a, b, c) a##b##c
#define CONCAT3(a, b, c) _IMPL_CONCAT3(a, b, c)

#define T int64_t

#define VECTOR_NAME CONCAT2(vector_, T)
#define VECTOR_FUNC(suffix) CONCAT3(vector_, T, _##suffix)

void VECTOR_FUNC(reserve)(Arena *arena, VECTOR_NAME *vec, size_t max) {
    vec->n = 0;
    if (max == 0) max++;
    assert(max > 0);
    vec->max = max;
    vec->p = arena_alloc_array(arena, T, max);
#ifdef WITH_LFORTRAN_ASSERT
    vec->reserve_called = vec_called_const;
#endif
}

void VECTOR_FUNC(push_back)(Arena *arena, VECTOR_NAME *vec, T x) {
#ifdef WITH_BASE_ASSERT
    assert(vec->reserve_called == vec_called_const);
#endif
    if (vec->n == vec->max) {
        size_t max2 = 2*vec->max;
        T* p2 = arena_alloc_array(arena, T, max2);
        memcpy(p2, vec->p, sizeof(T) * vec->max);
        vec->p = p2;
        vec->max = max2;
    }
    vec->p[vec->n] = x;
    vec->n++;
}



#undef T
#undef VECTOR_NAME
#undef VECTOR_FUNC
#undef CONCAT2
#undef _IMPL_CONCAT2
#undef CONCAT3
#undef _IMPL_CONCAT3
