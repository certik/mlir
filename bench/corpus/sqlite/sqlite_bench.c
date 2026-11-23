/* In-memory SQL workload for benchmarking. usage: sqlite_bench [rows] */
static int print_row(void *ctx, int n, char **vals, char **names) {
  (void)ctx; (void)names;
  for (int i = 0; i < n; i++) printf(i ? "|%s" : "%s", vals[i] ? vals[i] : "NULL");
  printf("\n");
  return 0;
}

static void exec_or_die(sqlite3 *db, const char *sql, int show) {
  char *err = 0;
  if (sqlite3_exec(db, sql, show ? print_row : 0, 0, &err) != SQLITE_OK) {
    printf("SQL error: %s\n  in: %s\n", err ? err : "?", sql);
    exit(1);
  }
}

int main(int argc, char **argv) {
  int rows = argc > 1 ? atoi(argv[1]) : 20000;
  sqlite3 *db;
  if (sqlite3_open(":memory:", &db) != SQLITE_OK) { printf("open failed\n"); return 1; }
  exec_or_die(db, "CREATE TABLE t(a INTEGER PRIMARY KEY, b INTEGER, c TEXT, d REAL)", 0);
  exec_or_die(db, "BEGIN", 0);
  sqlite3_stmt *ins;
  if (sqlite3_prepare_v2(db, "INSERT INTO t(b, c, d) VALUES(?1, ?2, ?3)", -1, &ins, 0) != SQLITE_OK) {
    printf("prepare failed: %s\n", sqlite3_errmsg(db)); return 1;
  }
  char buf[64];
  for (int i = 0; i < rows; i++) {
    int b = (int)((i * 7919L) % 10007);
    sqlite3_snprintf(sizeof(buf), buf, "row-%d-%x", i % 977, b);
    sqlite3_bind_int(ins, 1, b);
    sqlite3_bind_text(ins, 2, buf, -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(ins, 3, b / 8.0);
    if (sqlite3_step(ins) != SQLITE_DONE) { printf("insert failed: %s\n", sqlite3_errmsg(db)); return 1; }
    sqlite3_reset(ins);
  }
  sqlite3_finalize(ins);
  exec_or_die(db, "COMMIT", 0);
  exec_or_die(db, "CREATE INDEX tb ON t(b)", 0);
  exec_or_die(db, "SELECT count(*), sum(b), min(c), max(c), printf('%.3f', avg(d)) FROM t", 1);
  exec_or_die(db, "SELECT b % 7 AS k, count(*), sum(length(c)) FROM t GROUP BY k ORDER BY k", 1);
  exec_or_die(db, "SELECT count(*) FROM t AS x JOIN t AS y ON y.a = x.b + 1 WHERE x.b < 5000", 1);
  exec_or_die(db, "SELECT c, count(*) AS n FROM t GROUP BY c ORDER BY n DESC, c LIMIT 3", 1);
  exec_or_die(db, "UPDATE t SET d = d * 2 WHERE b % 3 = 0", 0);
  exec_or_die(db, "DELETE FROM t WHERE b % 5 = 0", 0);
  exec_or_die(db, "SELECT count(*), total(d), upper(substr(group_concat(c), 1, 20)) FROM t WHERE b BETWEEN 100 AND 2000", 1);
  exec_or_die(db, "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 1000) SELECT sum(n * n) FROM r", 1);
  sqlite3_close(db);
  return 0;
}
