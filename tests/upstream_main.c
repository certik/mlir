// Hosted entry shim used by upstream-LLVM builds where libc provides
// _start / __libc_start_main. Compiled together with platform_*.c built with
// -DPLATFORM_SKIP_ENTRY, so corec's bare-metal _start is not defined.

extern int app_main(void);
extern void platform_init(int argc, char **argv);

int main(int argc, char **argv) {
    platform_init(argc, argv);
    return app_main();
}
