// Runtime kernel: recursive quicksort of an int array (compares, swaps, calls).
#include <stdio.h>

#define N 1000000
static int v[N];
static unsigned int seed = 12345u;

static unsigned int rnd(void) {
    seed = seed * 1103515245u + 12345u;
    return seed >> 1;
}

static void sort(int *a, int lo, int hi) {
    while (lo < hi) {
        int p = a[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (a[i] < p) i++;
            while (a[j] > p) j--;
            if (i <= j) {
                int t = a[i]; a[i] = a[j]; a[j] = t;
                i++; j--;
            }
        }
        if (j - lo < hi - i) { sort(a, lo, j); lo = i; }
        else                 { sort(a, i, hi); hi = j; }
    }
}

int main(void) {
    unsigned long long check = 0;
    for (int rep = 0; rep < 3; rep++) {
        for (int i = 0; i < N; i++) v[i] = (int)(rnd() % 1000000u);
        sort(v, 0, N - 1);
        for (int i = 1; i < N; i++)
            if (v[i - 1] > v[i]) { printf("not sorted\n"); return 1; }
        check += (unsigned long long)v[N / 2] + (unsigned long long)v[N - 1];
    }
    printf("check = %llu\n", check);
    return 0;
}
