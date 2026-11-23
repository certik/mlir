// Runtime kernel: byte array loads/stores in tight nested loops.
#include <stdio.h>

#define N 4000000
static char composite[N + 1];

static int count_primes(void) {
    for (int i = 0; i <= N; i++) composite[i] = 0;
    int count = 0;
    for (int i = 2; i <= N; i++) {
        if (composite[i]) continue;
        count++;
        for (long j = (long)i * i; j <= N; j += i) composite[j] = 1;
    }
    return count;
}

int main(void) {
    int c = 0;
    for (int rep = 0; rep < 10; rep++) c = count_primes();
    printf("primes below %d: %d\n", N, c);
    return 0;
}
