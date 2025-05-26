#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

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

string double_to_string(Arena *arena, double value, int precision) {
    char buf[32];
    int len;
    printf("XX %f\n", value);
    if (precision >= 0) {
        len = snprintf(buf, sizeof(buf), "%.*f", precision, value);
    } else {
        len = snprintf(buf, sizeof(buf), "%f", value);
    }
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

// FormatSpec structure
typedef struct {
    char alignment;  // '<', '>', '^', or '\0'
    int width;       // -1 if not specified
    int precision;   // -1 if not specified
} FormatSpec;

// Parse format specifier
FormatSpec parse_format_spec(string spec) {
    FormatSpec fs = {.alignment = '\0', .width = -1, .precision = -1};
    const char *p = spec.str;
    const char *end = spec.str + spec.size;
    if (p < end) {
        if (*p == '<' || *p == '>' || *p == '^') {
            fs.alignment = *p++;
        }
    }
    if (p < end && isdigit(*p)) {
        fs.width = 0;
        while (p < end && isdigit(*p)) {
            fs.width = fs.width * 10 + (*p++ - '0');
        }
    }
    if (p < end && *p == '.') {
        p++;
        if (p < end && isdigit(*p)) {
            fs.precision = 0;
            while (p < end && isdigit(*p)) {
                fs.precision = fs.precision * 10 + (*p++ - '0');
            }
        }
    }
    return fs;
}

// Format argument based on type and spec
string format_arg(Arena *arena, Arg arg, FormatSpec spec) {
    string s;
    switch (arg.type) {
        case ARG_INT:
            s = int_to_string(arena, (int)arg.value);
            break;
        case ARG_DOUBLE:
            s = double_to_string(arena, (double)arg.value, spec.precision);
            break;
        case ARG_STRING:
            s = str_from_cstr_view((char*)arg.value);
            if (spec.precision >= 0 && spec.precision < s.size) {
                s.size = spec.precision;
            }
            break;
        case ARG_CHAR:
            s = char_to_string(arena, (char)arg.value);
            break;
        default:
            s = str_from_cstr_view("Unknown type");
    }
    if (spec.alignment == '\0') {
        if (arg.type == ARG_INT || arg.type == ARG_DOUBLE) {
            spec.alignment = '>';
        } else {
            spec.alignment = '<';
        }
    }
    if (spec.width > 0 && s.size < spec.width) {
        size_t pad_size = spec.width - s.size;
        char pad_char = ' ';
        string padding = {.str = arena_alloc_array(arena, char, pad_size), .size = pad_size};
        memset(padding.str, pad_char, pad_size);
        if (spec.alignment == '<') {
            s = str_concat(arena, s, padding);
        } else if (spec.alignment == '^') {
            size_t left_pad = pad_size / 2;
            size_t right_pad = pad_size - left_pad;
            string left = {.str = padding.str, .size = left_pad};
            string right = {.str = padding.str + left_pad, .size = right_pad};
            s = str_concat(arena, left, s);
            s = str_concat(arena, s, right);
        } else {  // '>' or default
            s = str_concat(arena, padding, s);
        }
    }
    return s;
}

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
        const char *close_brace = memchr(p, '}', end - p);
        if (close_brace == NULL) {
            string error = str_from_cstr_view("Error: missing closing brace");
            result = str_concat(arena, result, error);
            break;
        }
        const char *colon = memchr(p, ':', close_brace - p);
        FormatSpec spec;
        if (colon) {
            string spec_str = {.str = (char*)colon + 1, .size = close_brace - (colon + 1)};
            spec = parse_format_spec(spec_str);
        } else {
            if (p != close_brace) {
                string error = str_from_cstr_view("Error: invalid format specifier");
                result = str_concat(arena, result, error);
                p = close_brace + 1;
                continue;
            }
            spec = (FormatSpec){.alignment = '\0', .width = -1, .precision = -1};
        }
        if (i >= arg_count) {
            string error = str_from_cstr_view("Error: missing argument");
            result = str_concat(arena, result, error);
            p = close_brace + 1;
            continue;
        }
        Arg arg = args[i++];
        string s = format_arg(arena, arg, spec);
        result = str_concat(arena, result, s);
        p = close_brace + 1;
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
    double pi = 3.1415926535;

    // Original examples
    string fmt = str_lit("Hello!");
    string result = format(arena, fmt);
    printf("Original 1: %.*s\n", (int)result.size, result.str);

    fmt = str_lit("Hello, {}!");
    result = format(arena, fmt, A("world"));
    printf("Original 2: %.*s\n", (int)result.size, result.str);

    fmt = str_lit("Hello, {:>20}, {}!");
    result = format(arena, fmt, A("world"), A("xx"));
    printf("Original 3: %.*s\n", (int)result.size, result.str);

    // New examples for format specifications
    fmt = str_lit("{:10.3f}");
    result = format(arena, fmt, A(pi));
    printf("Width and Precision: %.*s\n", (int)result.size, result.str);  // "   3.14000"

    fmt = str_lit("{:.5f}");
    result = format(arena, fmt, A(pi));
    printf("Precision only: %.*s\n", (int)result.size, result.str);     // "3.14000"

    fmt = str_lit("{:6}");
    result = format(arena, fmt, A(42));
    printf("Width for int: %.*s\n", (int)result.size, result.str);      // "    42"

    fmt = str_lit("{:6}");
    result = format(arena, fmt, A('x'));
    printf("Width for char: %.*s\n", (int)result.size, result.str);     // "x     "

    fmt = str_lit("{:<20}");
    result = format(arena, fmt, A("left"));
    printf("Left aligned: %.*s\n", (int)result.size, result.str);       // "left                "

    fmt = str_lit("{:>20}");
    result = format(arena, fmt, A("right"));
    printf("Right aligned: %.*s\n", (int)result.size, result.str);      // "               right"

    fmt = str_lit("{:^20}");
    result = format(arena, fmt, A("centered"));
    printf("Centered: %.*s\n", (int)result.size, result.str);           // "     centered      "

    arena_free(arena);
    return 0;
}
