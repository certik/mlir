/* tinyc_rt.c -- corec + corec-stdlib runtime for a whole-program unity build.
 *
 * #include this file LAST from a program's unity TU, after the program's own
 * sources were included with `#define main bench_main` in effect. It pulls in
 * the corec base sources, corec-stdlib's libc and corec's Linux platform layer,
 * and supplies the entry point:
 *
 *   tinyc --emit=elf: the backend synthesises _start, which calls
 *     platform_init(argc, argv, envp) and then main(). The argc/argv it passes
 *     to main are not reliable (the registers are clobbered by the
 *     platform_init call), so main() below re-reads them from the platform.
 *     If BENCH_INIT_GLOBALS is defined, main() calls it first (used by the
 *     #317 workaround that moves global pointer-array initializers to code).
 *   gcc/clang (-nostdlib -nostdinc -static): _start below does the same job.
 *
 * Compile with: -I corec-stdlib/stdlib -I corec
 */
#define PLATFORM_SKIP_ENTRY 1
#define COREC_STDLIB_PROVIDES_MEM 1 /* memcpy/memset come from string_impl.c */
#include "base/assert.c"
#include "base/exit.c"
#include "base/mem.c"
#include "base/string.c"
#include "base/numconv.c"
/* format.c defines its own static isdigit, which clashes with <ctype.h>
 * when the program included it earlier in the unity TU. */
#define isdigit corec_format_isdigit
#include "base/format.c"
#undef isdigit
#include "base/strbuf.c"
#include "base/io.c"
#include "base/buddy.c"
#include "base/arena.c"
#include "base/scratch.c"
#include "string_impl.c"
#include "stdlib.c"
#include "stdio.c"
#include "printf.c"
#include "platform/platform_linux.c"

int bench_main(int argc, char **argv);

int main(void) {
#ifdef BENCH_INIT_GLOBALS
    BENCH_INIT_GLOBALS();
#endif
    size_t argc = 0, buf_size = 0;
    platform_args_sizes_get(&argc, &buf_size);
    char **argv = (char **)malloc((argc + 1) * sizeof(char *));
    char *buf = (char *)malloc(buf_size + 1);
    platform_args_get(argv, buf);
    argv[argc] = NULL;
    return bench_main((int)argc, argv);
}

#ifndef __TINYC__
__attribute__((naked)) void _start(void) {
    __asm__ volatile(
        "xor %rbp, %rbp\n"
        "mov (%rsp), %rdi\n"
        "lea 8(%rsp), %rsi\n"
        "lea 8(%rsi,%rdi,8), %rdx\n"
        "andq $-16, %rsp\n"
        "call bench_start_c\n"
        "mov %eax, %edi\n"
        "mov $60, %eax\n"
        "syscall\n"
        "hlt\n");
}
int bench_start_c(int argc, char **argv, char **envp) {
    platform_init(argc, argv, envp);
    return main();
}
#endif
