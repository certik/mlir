/* setjmp.h — uses tinyc's __builtin_setjmp/__builtin_longjmp on x86_64 and
 * aarch64 (buf holds up to 21 8-byte slots on aarch64, 8 on x86_64); falls
 * back to a non-throwing stub on targets without setjmp support (wasm). */
#pragma once
typedef long jmp_buf[22];
#if defined(__TINYC__) && (defined(__x86_64__) || defined(__aarch64__))
#define setjmp(env) __builtin_setjmp(env)
#define _setjmp(env) __builtin_setjmp(env)
#define longjmp(env, val) __builtin_longjmp(env, val)
#define _longjmp(env, val) __builtin_longjmp(env, val)
#else
#define setjmp(env) 0
#define _setjmp(env) 0
void longjmp(jmp_buf env, int val);
#define _longjmp longjmp
#endif
