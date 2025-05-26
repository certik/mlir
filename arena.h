#pragma once

#define allocate(arena, type, n) \
    ((type*)arena_alloc((arena), (n)*sizeof((type))))

Arena* arena_create(size_t size);
void* arena_alloc(Arena* arena, size_t size);
void arena_free(Arena* arena); 
