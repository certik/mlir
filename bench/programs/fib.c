// Runtime kernel: recursive calls (call/return, prologue/epilogue cost).
#include <stdio.h>

static int fib(int n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    int r = fib(36);
    printf("fib(36) = %d\n", r);
    return 0;
}
