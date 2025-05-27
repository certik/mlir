#pragma once

#include "tokenizer.h"
#include "arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Non-null-terminated string, represented by a pointer and a length.

// The string is typically owned by the Arena, so `string` can be seen as a
// "view" of the string.

typedef struct {
    char *str;
    uint64_t size;
} string;

#define str_lit(S)  (string){.str=(char*)(S), .size=sizeof(S)-1}

string str_from_cstr_view(char *cstr); 
string str_from_cstr_len_view(char *cstr, uint64_t size);
char *str_to_cstr_copy(Arena *arena, string str);
bool str_eq(string a, string b);
string str_substr(string str, uint64_t min, uint64_t max);

string int_to_string(Arena *arena, int value);
string double_to_string(Arena *arena, double value, int precision);
string char_to_string(Arena *arena, char c);
string str_concat(Arena *arena, string a, string b);



#ifdef __cplusplus
}
#endif
