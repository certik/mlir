#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Custom string and Arena types
typedef struct {
    char *str;
    uint64_t size;
} string;

#define str_lit(S)  (string){.str=(char*)(S), .size=sizeof(S)-1}

typedef struct {
    char* start;
    char* current;
    char* end;
} Arena;

#define arena_alloc(arena, type) \
    arena_alloc_array((arena), type, 1)
#define arena_alloc_array(arena, type, n) \
    (type*)arena_alloc_((arena), (n)*sizeof(type))

void* arena_alloc_(Arena* arena, size_t size) {
    if (arena->current + size > arena->end) {
        fprintf(stderr, "Arena out of memory\n");
        exit(1);
    }
    void* ptr = arena->current;
    arena->current += size;
    return ptr;
}

Arena* arena_create(size_t size) {
    Arena* arena = malloc(sizeof(Arena));
    arena->start = malloc(size);
    arena->current = arena->start;
    arena->end = arena->start + size;
    return arena;
}

void arena_free(Arena* arena) {
    free(arena->start);
    free(arena);
}

// String utility functions
string str_from_cstr_view(char *cstr) {
    return (string){.str = cstr, .size = strlen(cstr)};
}

string int_to_string(Arena *arena, int value) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%d", value);
    char *str = arena_alloc_array(arena, char, len + 1);
    memcpy(str, buf, len + 1);
    return (string){.str = str, .size = len};
}

string double_to_string(Arena *arena, double value) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%f", value);
    char *str = arena_alloc_array(arena, char, len + 1);
    memcpy(str, buf, len + 1);
    return (string){.str = str, .size = len};
}

string char_to_string(Arena *arena, char c) {
    char *buf = arena_alloc_array(arena, char, 1);
    *buf = c;
    return (string){.str = buf, .size = 1};
}

string str_concat(Arena *arena, string a, string b) {
    char *str = arena_alloc_array(arena, char, a.size + b.size + 1);
    memcpy(str, a.str, a.size);
    memcpy(str + a.size, b.str, b.size);
    str[a.size + b.size] = '\0';
    return (string){.str = str, .size = a.size + b.size};
}

// Arg structure for type safety
typedef enum {
    ARG_INT,
    ARG_DOUBLE,
    ARG_STRING,
    ARG_CHAR
} ArgType;

typedef struct {
    ArgType type;
    uint64_t value;
} Arg;

// Macro to wrap arguments with type detection
#define A(x) _Generic((x), \
    char*:  (Arg){.type = ARG_STRING, .value = (uint64_t)(x)}, \
    double: (Arg){.type = ARG_DOUBLE, .value = (uint64_t)(x)}, \
    char:   (Arg){.type = ARG_CHAR,   .value = (uint64_t)(x)}, \
    int:    (Arg){.type = ARG_INT,    .value = (uint64_t)(x)}  \
    )

// Core formatting function
string format(Arena *arena, string fmt, Arg args[], size_t arg_count) {
    if (arg_count > 8) {
        fprintf(stderr, "Error: format supports up to 8 arguments\n");
        exit(1);
    }

    string result = {.str = arena_alloc_array(arena, char, 1), .size = 0};
    result.str[0] = '\0';
    const char *p = fmt.str;
    const char *end = fmt.str + fmt.size;
    size_t i = 0;

    while (p < end) {
        const char *open_brace = memchr(p, '{', end - p);
        if (open_brace == NULL) {
            string remaining = {.str = (char*)p, .size = end - p};
            result = str_concat(arena, result, remaining);
            break;
        }
        if (open_brace > p) {
            string part = {.str = (char*)p, .size = open_brace - p};
            result = str_concat(arena, result, part);
        }
        p = open_brace + 1;
        if (p >= end) {
            string lit = {.str = (char*)open_brace, .size = 1};
            result = str_concat(arena, result, lit);
            break;
        }
        if (*p == '{') {
            string brace = {.str = (char*)open_brace, .size = 1};
            result = str_concat(arena, result, brace);
            p++;
            continue;
        }
        if (*p != '}') {
            string malformed = {.str = (char*)open_brace, .size = p - open_brace + 1};
            result = str_concat(arena, result, malformed);
            p++;
            continue;
        }
        if (i >= arg_count) {
            string error = str_from_cstr_view("Error: missing argument");
            result = str_concat(arena, result, error);
            p++;
            continue;
        }
        Arg arg = args[i++];
        string s;
        switch (arg.type) {
            case ARG_INT:
                s = int_to_string(arena, (int)arg.value);
                break;
            case ARG_DOUBLE:
                s = double_to_string(arena, (double)arg.value);
                break;
            case ARG_STRING:
                s = str_from_cstr_view((char*)arg.value);
                break;
            case ARG_CHAR:
                s = char_to_string(arena, (char)arg.value);
                break;
            default:
                s = str_from_cstr_view("Unknown type");
        }
        result = str_concat(arena, result, s);
        p++;
    }
    if (i < arg_count) {
        return str_from_cstr_view("Error: excess arguments");
    }
    return result;
}

// Format macro with explicit A() wrapping
#define format(arena, fmt, ...) \
    format(arena, fmt, (Arg[]){ __VA_ARGS__ }, sizeof((Arg[]){ __VA_ARGS__ })/sizeof(Arg))

// Main function with examples
int main() {
    Arena* arena = arena_create(1024);

    // Example 2: One argument
    string fmt2 = str_lit("Hello, {}!");
    string result2 = format(arena, fmt2, A("world"));
    printf("Result 2: %.*s\n", (int)result2.size, result2.str);

    fmt2 = str_lit("Hello, {}, {}!");
    result2 = format(arena, fmt2, A("world"), A("xx"));
    printf("Result 2: %.*s\n", (int)result2.size, result2.str);

    fmt2 = str_lit("Hello, {}, {}, {}!");
    result2 = format(arena, fmt2, A("world"), A("xx"), A(42));
    printf("Result 2: %.*s\n", (int)result2.size, result2.str);

    fmt2 = str_lit("Hello, {}, {}, {}, {}, {}!");
    result2 = format(arena, fmt2, A("world"), A('c'), A("xx"), A(42), A(4.34));
    printf("Result 2: %.*s\n", (int)result2.size, result2.str);


    arena_free(arena);
    return 0;
}
