#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "stdlib.h"

typedef struct {
    char* start;    // Start of the memory block
    char* current;  // Next available position
    char* end;      // End of the memory block
} Arena;


#define arena_alloc(arena, type, n) \
    (type*)arena_alloc_((arena), (n)*sizeof(type))

Arena* arena_create(size_t size);
void* arena_alloc_(Arena* arena, size_t size);
void arena_free(Arena* arena); 


#ifdef __cplusplus
}
#endif
