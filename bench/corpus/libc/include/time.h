/* time.h shim: no clock. time()/clock() return 0 (deterministic runs). */
#pragma once
#include <stddef.h>
typedef long time_t;
typedef long clock_t;
#define CLOCKS_PER_SEC 1000000L
time_t time(time_t *t);
clock_t clock(void);
