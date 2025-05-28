// clang c.c && ./a.out
// cl /std:c11 c.c
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <stdarg.h>
#include <assert.h>

#include <base/format.h>
#include <base/io.h>

void test_io() {
    Arena* arena = arena_create(1024);

    string text;
    bool ok = read_file(arena, str_lit("does not exist"), &text);
    assert(!ok);

    arena_free(arena);
}

void test_format() {
    Arena* arena = arena_create(1024);
    double pi = 3.1415926535;

    // Example with no arguments
    string fmt = str_lit("Hello!");
    string result = format(arena, fmt);
    assert(str_eq(result, str_lit("Hello!")));
    printf("No args: %.*s\n", (int)result.size, result.str);

    // Example with one argument
    fmt = str_lit("Hello, {}!");
    result = format(arena, fmt, str_lit("world"));
    assert(str_eq(result, str_lit("Hello, world!")));
    printf("One arg: %.*s\n", (int)result.size, result.str);

    fmt = str_lit("Hello, {}!");
    result = format(arena, fmt, 5);
    assert(str_eq(result, str_lit("Hello, 5!")));
    printf("No args: %.*s\n", (int)result.size, result.str);

    // Example with formatted double
    fmt = str_lit("Value: {:10.5f}");
    result = format(arena, fmt, pi);
    assert(str_eq(result, str_lit("Value:    3.14159")));
    printf("Formatted double: %.*s\n", (int)result.size, result.str);

    // Example with formatted char
    fmt = str_lit("Char: |{:^5}|");
    result = format(arena, fmt, 'x');
    assert(str_eq(result, str_lit("Char: | 120 |")));
    printf("Formatted char: %.*s\n", (int)result.size, result.str);

    // Example with multiple arguments
    fmt = str_lit("Hello, {}, {}, {}, {}!");
    result = format(arena, fmt, "world", 35.5, str_lit("XX"), 3);
    printf("Multiple args: %.*s\n", (int)result.size, result.str);

    arena_free(arena);
}

int main() {
    test_format();
    test_io();
    return 0;
}
