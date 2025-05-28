// clang c.c && ./a.out
// cl /std:c11 c.c
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <stdarg.h>

#include <base/format.h>


// Main function with examples
int main() {
    Arena* arena = arena_create(1024);
    double pi = 3.1415926535;

    // Example with no arguments
    string fmt = str_lit("Hello!");
    string result = format(arena, fmt);
    printf("No args: %.*s\n", (int)result.size, result.str);

    // Example with one argument
    fmt = str_lit("Hello, {}!");
    result = format(arena, fmt, "world");
    printf("One arg: %.*s\n", (int)result.size, result.str);

    fmt = str_lit("Hello, {}!");
    result = format(arena, fmt, 5);
    printf("No args: %.*s\n", (int)result.size, result.str);

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

    arena_free(arena);
    return 0;
}
