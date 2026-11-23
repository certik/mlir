/* setjmp.h shim. tinyc has no way to implement setjmp/longjmp (no inline
 * asm, no __builtin_setjmp, no separately assembled objects), so setjmp
 * always returns 0 and longjmp aborts the program. Code that only uses
 * setjmp/longjmp for error recovery runs fine as long as no error occurs. */
#pragma once
typedef long jmp_buf[8];
#define setjmp(env) 0
#define _setjmp(env) 0
void longjmp(jmp_buf env, int val);
#define _longjmp longjmp
