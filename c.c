// clang c.c && ./a.out
// cl /std:c11 c.c
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <stdarg.h>

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
    arena_new_array((arena), type, 1)
#define arena_new_array(arena, type, n) \
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

void arena_destroy(Arena* arena) {
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
    char *str = arena_new_array(arena, char, len + 1);
    memcpy(str, buf, len + 1);
    return (string){.str = str, .size = len};
}

string double_to_string(Arena *arena, double value, int precision) {
    char buf[32];
    int len;
    if (precision >= 0) {
        len = snprintf(buf, sizeof(buf), "%.*f", precision, value);
    } else {
        len = snprintf(buf, sizeof(buf), "%f", value);
    }
    char *str = arena_new_array(arena, char, len + 1);
    memcpy(str, buf, len + 1);
    return (string){.str = str, .size = len};
}

string char_to_string(Arena *arena, char c) {
    char *buf = arena_new_array(arena, char, 1);
    *buf = c;
    return (string){.str = buf, .size = 1};
}

string str_concat(Arena *arena, string a, string b) {
    char *str = arena_new_array(arena, char, a.size + b.size + 1);
    memcpy(str, a.str, a.size);
    memcpy(str + a.size, b.str, b.size);
    str[a.size + b.size] = '\0';
    return (string){.str = str, .size = a.size + b.size};
}

// ArgType enum
typedef enum {
    ARG_INT,
    ARG_DOUBLE,
    ARG_STRING,
    ARG_STRING2,
    ARG_CHAR
} ArgType;

// Macro to wrap arguments with type detection
#define A(x) _Generic((x), \
    char*:  ARG_STRING, \
    string: ARG_STRING2, \
    double: ARG_DOUBLE, \
    char:   ARG_CHAR, \
    int:    ARG_INT  \
    ), (x)

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

// Core formatting function with variadic arguments
string format(Arena *arena, string fmt, size_t arg_count, ...) {
    va_list ap;
    va_start(ap, arg_count);
    string result = {.str = arena_new_array(arena, char, 1), .size = 0};
    result.str[0] = '\0';
    const char *p = fmt.str;
    const char *end = fmt.str + fmt.size;
    size_t arg_index = 0;
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
        if (arg_index >= arg_count) {
            string error = str_from_cstr_view("Error: missing argument");
            result = str_concat(arena, result, error);
            p = close_brace + 1;
            continue;
        }
        ArgType type = (ArgType)va_arg(ap, int);
        string s;
        switch (type) {
            case ARG_INT: {
                int value = va_arg(ap, int);
                s = int_to_string(arena, value);
                break;
            }
            case ARG_DOUBLE: {
                double value = va_arg(ap, double);
                s = double_to_string(arena, value, spec.precision);
                break;
            }
            case ARG_STRING: {
                char* value = va_arg(ap, char*);
                s = str_from_cstr_view(value);
                if (spec.precision >= 0 && spec.precision < s.size) {
                    s.size = spec.precision;
                }
                break;
            }
            case ARG_STRING2: {
                string value = va_arg(ap, string);
                s = value;
                if (spec.precision >= 0 && spec.precision < s.size) {
                    s.size = spec.precision;
                }
                break;
            }
            case ARG_CHAR: {
                char value = (char)va_arg(ap, int);
                s = char_to_string(arena, value);
                break;
            }
            default:
                s = str_from_cstr_view("Unknown type");
        }
        arg_index++;
        // Apply width and alignment
        if (spec.alignment == '\0') {
            if (type == ARG_INT || type == ARG_DOUBLE) {
                spec.alignment = '>';
            } else {
                spec.alignment = '<';
            }
        }
        if (spec.width > 0 && s.size < spec.width) {
            size_t pad_size = spec.width - s.size;
            char pad_char = ' ';
            string padding = {.str = arena_new_array(arena, char, pad_size), .size = pad_size};
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
        result = str_concat(arena, result, s);
        p = close_brace + 1;
    }
    //if (arg_index < arg_count) {
    //    return str_from_cstr_view("Error: excess arguments");
    //}
    va_end(ap);
    return result;
}


#define GET_ARG_COUNT(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define COUNT_ARGS(...) GET_ARG_COUNT(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define APPLY_A0()
#define APPLY_A1(a) A(a)
#define APPLY_A2(a, b) A(a), A(b)
#define APPLY_A3(a, b, c) A(a), A(b), A(c)
#define APPLY_A4(a, b, c, d) A(a), A(b), A(c), A(d)
#define APPLY_A5(a, b, c, d, e) A(a), A(b), A(c), A(d), A(e)
#define APPLY_A6(a, b, c, d, e, f) A(a), A(b), A(c), A(d), A(e), A(f)
#define APPLY_A7(a, b, c, d, e, f, g) A(a), A(b), A(c), A(d), A(e), A(f), A(g)
#define APPLY_A8(a, b, c, d, e, f, g, h) A(a), A(b), A(c), A(d), A(e), A(f), A(g), A(h)

#define APPLY_A_FOR_COUNT_0 APPLY_A0
#define APPLY_A_FOR_COUNT_1 APPLY_A1
#define APPLY_A_FOR_COUNT_2 APPLY_A2
#define APPLY_A_FOR_COUNT_3 APPLY_A3
#define APPLY_A_FOR_COUNT_4 APPLY_A4
#define APPLY_A_FOR_COUNT_5 APPLY_A5
#define APPLY_A_FOR_COUNT_6 APPLY_A6
#define APPLY_A_FOR_COUNT_7 APPLY_A7
#define APPLY_A_FOR_COUNT_8 APPLY_A8

#define CONCAT_AFTER_EXPAND(prefix, count) prefix ## count
#define APPLY_WITH_COUNT(count, ...) CONCAT_AFTER_EXPAND(APPLY_A_FOR_COUNT_, count)(__VA_ARGS__)

#define format(arena, fmt, ...) \
    format(arena, fmt, COUNT_ARGS(__VA_ARGS__), APPLY_WITH_COUNT(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__))

// Main function with examples
int main() {
    Arena* arena = arena_create(1024);
    double pi = 3.1415926535;

    // Example with no arguments
    string fmt = str_lit("Hello, {}!");
    string result = format(arena, fmt, 5);
    printf("No args: %.*s\n", (int)result.size, result.str);

    // Example with one argument
    fmt = str_lit("Hello, {}!");
    result = format(arena, fmt, "world");
    printf("One arg: %.*s\n", (int)result.size, result.str);

    // Example with formatted double
    fmt = str_lit("Value: {:10.5f}");
    result = format(arena, fmt, pi);
    printf("Formatted double: %.*s\n", (int)result.size, result.str);

    // Example with formatted char
    fmt = str_lit("Char: {:^5}");
    result = format(arena, fmt, 'x');
    printf("Formatted char: %.*s\n", (int)result.size, result.str);

    // Example with multiple arguments
    fmt = str_lit("Hello, {}, {}, {}, {}!");
    result = format(arena, fmt, "world", 35.5, 3, fmt);
    printf("Multiple args: %.*s\n", (int)result.size, result.str);

    arena_destroy(arena);
    return 0;
}
