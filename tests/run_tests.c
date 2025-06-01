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
#include <base/hashtable.h>

void test_io() {
    Arena* arena = arena_create(1024*20);
    println(arena, str_lit("test_io()"));

    string text;
    bool ok = read_file(arena, str_lit("does not exist"), &text);
    assert(!ok);

    text.size = 0;
    assert(text.size == 0);
    ok = read_file(arena, str_lit("README.md"), &text);
    assert(ok);
    assert(text.size > 10);
    println(arena, text);

    println(arena, str_lit("Hello from io."));

    println(arena, str_lit("Success()"));
    arena_free(arena);
}

void test_format() {
    Arena* arena = arena_create(1024*10);
    double pi = 3.1415926535;

    // Example with no arguments
    string fmt = str_lit("Hello!");
    string result = format(arena, fmt);
    assert(str_eq(result, str_lit("Hello!")));
    println(arena, str_lit("No args: {}"), result);

    // Example with one argument
    fmt = str_lit("Hello, {}!");
    result = format(arena, fmt, str_lit("world"));
    assert(str_eq(result, str_lit("Hello, world!")));
    println(arena, str_lit("One arg: {}"), result);

    fmt = str_lit("Hello, {}!");
    result = format(arena, fmt, 5);
    assert(str_eq(result, str_lit("Hello, 5!")));
    println(arena, str_lit("One arg: {}"), result);

    // Example with formatted double
    fmt = str_lit("Value: {:10.5f}");
    result = format(arena, fmt, pi);
    assert(str_eq(result, str_lit("Value:    3.14159")));
    println(arena, str_lit("Formatted double: {}"), result);

    // Example with formatted char
    fmt = str_lit("Char: |{:^5}|");
    result = format(arena, fmt, 'x');
    assert(str_eq(result, str_lit("Char: | 120 |")));
    println(arena, str_lit("Formatted char: {}"), result);

    // Example with multiple arguments
    fmt = str_lit("Hello, {}, {}, {}, {}!");
    result = format(arena, fmt, "world", 35.5, str_lit("XX"), 3);
    println(arena, str_lit("Multiple args: {}"), result);

    arena_free(arena);
}

#define MapIntString_HASH(key) ((size_t)(key))
#define MapIntString_EQUAL(key1, key2) ((key1) == (key2))

DEFINE_HASHTABLE_FOR_TYPES(int, string, MapIntString)

void test_hashtable1() {
    Arena* arena = arena_create(1024*10);

    MapIntString ht;
    MapIntString_init(arena, &ht, 16);
    MapIntString_insert(arena, &ht, 42, str_lit("forty-two"));
    string *value = MapIntString_get(&ht, 42);
    assert(value);
    println(arena, str_lit("Value for key 42: {}"), *value);

    arena_free(arena);
}

#define MapStringInt_HASH(key) (str_hash(key))
#define MapStringInt_EQUAL(key1, key2) (str_eq((key1), (key2)))

DEFINE_HASHTABLE_FOR_TYPES(string, int, MapStringInt)

void test_hashtable2() {
    Arena* arena = arena_create(1024*10);

    MapStringInt ht;
    MapStringInt_init(arena, &ht, 16);
    MapStringInt_insert(arena, &ht, str_lit("forty-two"), 42);
    int *value = MapStringInt_get(&ht, str_lit("forty-two"));
    assert(value);
    println(arena, str_lit("Value for key \"forty-two\": {}"), *value);

    arena_free(arena);
}

int main() {
    test_format();
    test_io();
    test_hashtable1();
    test_hashtable2();
    return 0;
}
