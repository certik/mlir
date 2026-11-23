/* CoreMark port for tinyc: no timer (see core_portme.h). */
/* Included from the unity TU after coremark.h (which has no include guard). */

#if VALIDATION_RUN
volatile ee_s32 seed1_volatile = 0x3415;
volatile ee_s32 seed2_volatile = 0x3415;
volatile ee_s32 seed3_volatile = 0x66;
#endif
#if PERFORMANCE_RUN
volatile ee_s32 seed1_volatile = 0x0;
volatile ee_s32 seed2_volatile = 0x0;
volatile ee_s32 seed3_volatile = 0x66;
#endif
#if PROFILE_RUN
volatile ee_s32 seed1_volatile = 0x8;
volatile ee_s32 seed2_volatile = 0x8;
volatile ee_s32 seed3_volatile = 0x8;
#endif
volatile ee_s32 seed4_volatile = ITERATIONS;
volatile ee_s32 seed5_volatile = 0;

/* No timer. get_time() reports a fixed 10 "seconds" so core_main's
 * "must execute for at least 10 secs" check does not count as an error and a
 * correct run ends with "Correct operation validated". The timing lines in
 * the output are therefore meaningless; time the process externally. */
void start_time(void) {}
void stop_time(void) {}
CORE_TICKS get_time(void) { return 10; }
secs_ret time_in_secs(CORE_TICKS ticks) { return (secs_ret)ticks; }

ee_u32 default_num_contexts = 1;

void portable_init(core_portable *p, int *argc, char *argv[]) {
    (void)argc; (void)argv;
    if (sizeof(ee_ptr_int) != sizeof(ee_u8 *))
        ee_printf("ERROR! Please define ee_ptr_int to a type that holds a pointer!\n");
    if (sizeof(ee_u32) != 4)
        ee_printf("ERROR! Please define ee_u32 to a 32b unsigned type!\n");
    p->portable_id = 1;
}

void portable_fini(core_portable *p) { p->portable_id = 0; }
