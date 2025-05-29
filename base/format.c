#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <stdarg.h>

#include <base/format.h>


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
string format_explicit_varg(Arena *arena, string fmt, size_t arg_count,
        va_list ap) {
    string result = {arena_alloc_array(arena, char, 1), 0};
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
            case ARG_UINT64: {
                uint64_t value = va_arg(ap, uint64_t);
                s = int_to_string(arena, value);
                break;
            }
            case ARG_INT64: {
                int64_t value = va_arg(ap, int64_t);
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
        result = str_concat(arena, result, s);
        p = close_brace + 1;
    }
    //if (arg_index < arg_count) {
    //    return str_from_cstr_view("Error: excess arguments");
    //}
    return result;
}

string format_explicit(Arena *arena, string fmt, size_t arg_count, ...) {
    va_list ap;
    va_start(ap, arg_count);
    return format_explicit_varg(arena, fmt, arg_count, ap);
    va_end(ap);
}
