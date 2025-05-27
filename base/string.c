#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <base/string.h>

string str_from_cstr_view(char *cstr) {
    return (string){.str=cstr, .size=strlen(cstr)-1};
}

string str_from_cstr_len_view(char *cstr, uint64_t size) {
    return (string){.str=cstr, .size=size};
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
    return (string){.str=str.str+min, .size=size};
}
