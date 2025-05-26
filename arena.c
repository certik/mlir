#include <stdlib.h>
#include <stdint.h>

#include "arena.h"

// Include sanitizer headers if enabled
#if defined(__has_feature)
#  if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#    include <sanitizer/asan_interface.h>
#  endif
#  if __has_feature(memory_sanitizer)
#    include <sanitizer/msan_interface.h>
#  endif
#endif


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

    // Mark entire arena as addressable (ASan) and initialized (MSan)
    #if defined(__has_feature)
    #  if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
    __asan_poison_memory_region(arena->start, size);
    #  endif
    #  if __has_feature(memory_sanitizer)
    __msan_poison(arena->start, size);
    #  endif
    #endif

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

    // Mark padding as poisoned (ASan) and uninitialized (MSan)
    #if defined(__has_feature)
    #  if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
    if (aligned > arena->current) {
        __asan_poison_memory_region(arena->current, aligned - arena->current);
    }
    __asan_unpoison_memory_region(aligned, size);
    #  endif
    #  if __has_feature(memory_sanitizer)
    if (aligned > arena->current) {
        __msan_poison(arena->current, aligned - arena->current);
    }
    __msan_unpoison(aligned, size);
    #  endif
    #endif


    arena->current = aligned + size;
    return aligned;
}

void arena_free(Arena* arena) {
    if (arena != NULL) {
        // Poison the entire arena before freeing (ASan)
        #if defined(__has_feature)
        #  if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
        __asan_poison_memory_region(arena->start, arena->end - arena->start);
        #  endif
        #endif

        free(arena->start);
        free(arena);
    }
}
