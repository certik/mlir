/* Minimal OS layer for SQLITE_OS_OTHER=1: a VFS that cannot open files, so
 * only ":memory:" databases work (with SQLITE_TEMP_STORE=3 temp data stays in
 * memory too). Randomness is deterministic and the clock is fixed, so runs are
 * reproducible. */
static int bvfs_open(sqlite3_vfs *v, sqlite3_filename name, sqlite3_file *f, int flags, int *out) {
  (void)v; (void)name; (void)f; (void)flags; (void)out;
  return SQLITE_CANTOPEN;
}
static int bvfs_delete(sqlite3_vfs *v, const char *name, int sync) { (void)v; (void)name; (void)sync; return SQLITE_OK; }
static int bvfs_access(sqlite3_vfs *v, const char *name, int flags, int *res) {
  (void)v; (void)name; (void)flags; *res = 0; return SQLITE_OK;
}
static int bvfs_fullpathname(sqlite3_vfs *v, const char *name, int n, char *out) {
  (void)v;
  sqlite3_snprintf(n, out, "%s", name);
  return SQLITE_OK;
}
static int bvfs_randomness(sqlite3_vfs *v, int n, char *out) {
  static unsigned int s = 12345;
  (void)v;
  for (int i = 0; i < n; i++) { s = s * 1103515245u + 12345u; out[i] = (char)(s >> 16); }
  return n;
}
static int bvfs_sleep(sqlite3_vfs *v, int us) { (void)v; return us; }
static int bvfs_currenttime(sqlite3_vfs *v, double *t) { (void)v; *t = 2460000.5; return SQLITE_OK; }
static int bvfs_getlasterror(sqlite3_vfs *v, int n, char *s) { (void)v; (void)n; (void)s; return 0; }
static int bvfs_currenttimeint64(sqlite3_vfs *v, sqlite3_int64 *t) {
  (void)v; *t = (sqlite3_int64)2460000 * 86400000 + 43200000; return SQLITE_OK;
}

static sqlite3_vfs bench_vfs;

int sqlite3_os_init(void) {
  bench_vfs.iVersion = 2;
  bench_vfs.szOsFile = (int)sizeof(sqlite3_file);
  bench_vfs.mxPathname = 512;
  bench_vfs.zName = "bench";
  bench_vfs.xOpen = bvfs_open;
  bench_vfs.xDelete = bvfs_delete;
  bench_vfs.xAccess = bvfs_access;
  bench_vfs.xFullPathname = bvfs_fullpathname;
  bench_vfs.xRandomness = bvfs_randomness;
  bench_vfs.xSleep = bvfs_sleep;
  bench_vfs.xCurrentTime = bvfs_currenttime;
  bench_vfs.xGetLastError = bvfs_getlasterror;
  bench_vfs.xCurrentTimeInt64 = bvfs_currenttimeint64;
  return sqlite3_vfs_register(&bench_vfs, 1);
}
int sqlite3_os_end(void) { return SQLITE_OK; }
