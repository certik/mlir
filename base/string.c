#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <base/string.h>

string str_from_cstr_view(char *cstr) {
    string result = {cstr, strlen(cstr)};
    return result;
}

string str_from_cstr_len_view(char *cstr, uint64_t size) {
    string result = {cstr, size};
    return result;
}

char *str_to_cstr_copy(Arena *arena, string str) {
    char *cstr = arena_alloc_array(arena, char, str.size+1);
    memcpy(cstr, str.str, str.size);
    cstr[str.size] = '\0';
    return cstr;
}

bool str_eq(string a, string b) {
    if (a.size == b.size) {
        return (memcmp(a.str, b.str, a.size) == 0);
    } else {
        return false;
    }
}

string str_substr(string str, uint64_t min, uint64_t size) {
    string result = {str.str+min, size};
    return result;
}

string int_to_string(Arena *arena, int value) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%d", value);
    char *str = arena_alloc_array(arena, char, len + 1);
    memcpy(str, buf, len + 1);
    string result = {str, len};
    return result;
}

string double_to_string(Arena *arena, double value, int precision) {
    char buf[32];
    int len;
    if (precision >= 0) {
        len = snprintf(buf, sizeof(buf), "%.*f", precision, value);
    } else {
        len = snprintf(buf, sizeof(buf), "%f", value);
    }
    char *str = arena_alloc_array(arena, char, len + 1);
    memcpy(str, buf, len + 1);
    string result = {str, len};
    return result;
}

string char_to_string(Arena *arena, char c) {
    char *buf = arena_alloc_array(arena, char, 1);
    *buf = c;
    string result = {buf, 1};
    return result;
}

string str_concat(Arena *arena, string a, string b) {
    char *str = arena_alloc_array(arena, char, a.size + b.size + 1);
    memcpy(str, a.str, a.size);
    memcpy(str + a.size, b.str, b.size);
    str[a.size + b.size] = '\0';
    string result = {str, a.size + b.size};
    return result;
}

uint32_t str_hash(string str) {
    // FNV-1a hash
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < str.size; i++) {
        hash ^= (uint32_t)(unsigned char)str.str[i];
        hash *= 16777619u;
    }
    return hash;
}
