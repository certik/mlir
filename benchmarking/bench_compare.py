import os
import stat
import statistics
import subprocess
import time

REPEATS = int(os.environ.get("REPEATS", "3"))
BENCH_STAT = os.environ.get("BENCH_STAT", "avg")
cc = os.environ["CC"]
src = os.environ["SRC"]
tc = os.environ["TC"]
clang_bin = os.environ["CLANG_BIN"]
clang_bin_o0 = os.environ["CLANG_BIN_O0"]
clang_bin_o3 = os.environ["CLANG_BIN_O3"]
our_bin_bl = os.environ["OUR_BIN_BL"]
our_bin_to = os.environ["OUR_BIN_TO"]
bench_lower = os.environ["BENCH_LOWER"]
tinyc_opt = os.environ["TINYC_OPT"]
clang_inc = os.environ["CLANG_INC"].split()
work = os.environ["WORK"]
n = os.environ.get("BENCH_N", "?")
O3_FLAGS = os.environ.get(
    "CLANG_O3_FLAGS", "-O3 -DNDEBUG -flto -march=native").split()

COL_CLANG = "clang"
COL_O0 = "clang -O0"
COL_O3 = "clang -O3 -march=native"
COL_BL = "small compiler"
COL_TO = "full compiler"

PRE = os.path.join(work, "workload.i")
PRE_O0 = os.path.join(work, "workload_o0.i")
PRE_O3 = os.path.join(work, "workload_o3.i")
OBJ = os.path.join(work, "workload.o")
OBJ_O0 = os.path.join(work, "workload_o0.o")
OBJ_O3 = os.path.join(work, "workload_o3.o")


def ms(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def aggregate_ms(fn):
    times = [ms(fn) for _ in range(REPEATS)]
    if BENCH_STAT == "avg":
        return sum(times) / len(times)
    return statistics.median(times)


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def aggregate_exec(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        p = subprocess.run([path], capture_output=True)
        if p.returncode < 0:
            return None
        times.append((time.perf_counter() - t0) * 1000.0)
    if BENCH_STAT == "avg":
        return sum(times) / len(times)
    return statistics.median(times)


def dash(v):
    return "—" if v is None else f"{v:10.3f}"


def cell(row, col):
    return dash(row.get(col))


def clang_path(flags, stage):
    if stage == "pre":
        return [cc, "-E"] + clang_inc + [src, "-o", flags["pre"]]
    if stage == "compile":
        return [cc, "-c"] + flags["compile"] + [flags["pre"], "-o", flags["obj"]]
    if stage == "link":
        return [cc] + flags["link"] + [flags["obj"], "-o", flags["bin"]]
    if stage == "total":
        return [cc] + flags["total"] + clang_inc + [src, "-o", flags["bin"]]
    raise ValueError(stage)


VARIANTS = {
    COL_CLANG: {
        "pre": PRE, "obj": OBJ, "bin": clang_bin,
        "compile": [], "link": [], "total": [],
    },
    COL_O0: {
        "pre": PRE_O0, "obj": OBJ_O0, "bin": clang_bin_o0,
        "compile": ["-O0"], "link": ["-O0"], "total": ["-O0"],
    },
    COL_O3: {
        "pre": PRE_O3, "obj": OBJ_O3, "bin": clang_bin_o3,
        "compile": O3_FLAGS,
        "link": ["-O3", "-flto", "-march=native"],
        "total": O3_FLAGS,
    },
}

clang_times = {}
for name, flags in VARIANTS.items():
    clang_times[name] = {
        "preprocess (-E)": aggregate_ms(lambda f=flags: run(clang_path(f, "pre"))),
        "compile (-c)": aggregate_ms(lambda f=flags: run(clang_path(f, "compile"))),
        "link": aggregate_ms(lambda f=flags: run(clang_path(f, "link"))),
        "end-to-end compile": aggregate_ms(lambda f=flags: run(clang_path(f, "total"))),
        "runtime": aggregate_exec(flags["bin"]),
    }

exec_env = {**os.environ, "BENCH_TC": tc}

if not bench_lower or not os.path.isfile(bench_lower):
    raise SystemExit(f"bench_lower binary missing: {bench_lower!r}")

mlir_bl_compile = aggregate_ms(lambda: run(
    [bench_lower, "--emit-exec-native", our_bin_bl], env=exec_env))
mlir_bl_runtime = aggregate_exec(our_bin_bl)

if not tinyc_opt or not os.path.isfile(tinyc_opt):
    raise SystemExit(f"tinyc_native_opt binary missing: {tinyc_opt!r}")

mlir_to_compile = aggregate_ms(lambda: run(
    [tinyc_opt, "--emit=elf", "-o", our_bin_to, tc], env=exec_env))
mlir_to_runtime = aggregate_exec(our_bin_to)

cols = [COL_CLANG, COL_O0, COL_O3, COL_BL, COL_TO]

runtime_label = f"runtime ({BENCH_STAT} of {REPEATS})"
step_names = ["preprocess (-E)", "compile (-c)", "link", "end-to-end compile",
              runtime_label]

rows = {}
for step in step_names:
    key = "runtime" if step.startswith("runtime") else step
    row = {}
    for c in (COL_CLANG, COL_O0, COL_O3):
        row[c] = clang_times[c][key]
    row[COL_BL] = mlir_bl_compile if key == "end-to-end compile" else (
        mlir_bl_runtime if key == "runtime" else None)
    row[COL_TO] = mlir_to_compile if key == "end-to-end compile" else (
        mlir_to_runtime if key == "runtime" else None)
    rows[step] = row

width = max(len(c) for c in cols)
print(f"input: bench_lower.tc + workload.c (N={n})")
print(f"compilers: {bench_lower}, {tinyc_opt}")
print(f"({BENCH_STAT} of {REPEATS} runs per measurement)")
print()
header = f"{'metric':<24}" + "".join(f"{c:>{width + 2}}" for c in cols)
print(header)
print("-" * len(header))
for name, row in rows.items():
    print(f"{name:<24}" + "".join(f"{cell(row, c):>{width + 2}}" for c in cols))

print()
print(f"=== summary (end-to-end compile + runtime, ms; {BENCH_STAT} of {REPEATS}) ===")
print(f"{'':<24}" + "".join(f"{c:>{width + 2}}" for c in cols))
print(f"{'compile time':<24}" + "".join(
    f"{dash(rows['end-to-end compile'].get(c)):>{width + 2}}" for c in cols))
print(f"{'runtime':<24}" + "".join(
    f"{dash(rows[runtime_label].get(c)):>{width + 2}}" for c in cols))
