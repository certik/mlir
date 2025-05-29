#include <base/vector.h>

#define VecType int64_t

void vector_VecType_reserve(Arena *arena, vector_VecType *vec, size_t max) {
    vec->n = 0;
    if (max == 0) max++;
    assert(max > 0)
    vec->max = max;
    vec->p = arena_alloc_array(arena, VecType, max);
#ifdef WITH_LFORTRAN_ASSERT
    vec->reserve_called = vec_called_const;
#endif
}

void vector_VecType_push_back(Arena *arena, vector_VecType *vec, VecType x) {
    assert(vec->reserve_called == vec_called_const);
    if (vec->n == vec->max) {
        size_t max2 = 2*vec->max;
        VecType* p2 = arena_alloc_array(arena, VecType, max2);
        std::memcpy(p2, vec->p, sizeof(VecType) * vec->max);
        vec->p = p2;
        vec->max = max2;
    }
    vec->p[vec->n] = x;
    vec->n++;
}


#undef VecType
