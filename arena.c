#include <stdlib.h>
#include <stdint.h>

#include "arena.h"

Arena* arena_create(size_t size) {
    // TODO: merge the two allocations
    Arena* arena = malloc(sizeof(Arena));
    if (arena == NULL) return NULL;
    arena->start = malloc(size);
    if (arena->start == NULL) {
        free(arena);
        exit(1);
        return NULL;
    }
    arena->current = arena->start;
    arena->end = arena->start + size;
    return arena;
}

void* arena_alloc_(Arena* arena, size_t size) {
    if (size == 0) return NULL;
    const size_t align = 16;
    uintptr_t current_addr = (uintptr_t)arena->current;
    uintptr_t aligned_addr = (current_addr + align - 1) & ~(align - 1);
    char* aligned = (char*)aligned_addr;
    if (aligned + size > arena->end) {
        // Not enough space
        exit(2);
        return NULL;
    }
    arena->current = aligned + size;
    return aligned;
}

void arena_free(Arena* arena) {
    if (arena != NULL) {
        free(arena->start);
        free(arena);
    }
}
