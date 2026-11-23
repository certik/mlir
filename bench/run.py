#!/usr/bin/env python3
"""tinyc benchmark harness (certik/mlir#329).

Measures, on Linux/x86_64:

  a  Self-compile time. Every source of the tinyc self-host build
     (examples/tinyc/selfhost_tinyc_elf.sh, with platform_linux.c in place of
     platform_wasm.c) is compiled file by file, one process per file:
       tinyc    `tinyc --emit=x64`: the full native x86-64 pipeline (preprocess,
                parse, emit, mem2reg, lower, isel/regalloc/encode) per
                translation unit. tinyc has no object writer yet, so it writes
                the raw machine code instead of a .o (see --emit=x64 in
                driver.c); everything else a `-c` compile does is included.
       gcc -O0 -c, clang -O0 -c, tcc -c   (tcc only if found).
     All compilers parse the same headers (corec-stdlib): gcc/clang/tcc get
     -nostdinc plus bench/include (a one-line <sys/stat.h> for driver.c);
     tinyc has no system include path and skips that header. The tinyc binary
     itself is built with gcc -O2 -DNDEBUG. Also timed: the complete path that turns the same
     sources into a runnable tinyc executable (tinyc: the self-host pipeline
     through wasm; gcc/clang: -O0 compile + link).
  b  Per-stage breakdown of workload (a) from `tinyc --time-stages`, summed
     over the files, plus the per-stage breakdown of the self-host lift
     (`--from-wasm --emit=elf`).
  c  Runtime of generated code: small kernels in bench/programs (and the
     corpus programs that build), compiled by tinyc --emit=elf, gcc -O0 and
     clang -O0 against the same corec-stdlib libc; and tinyc itself as a
     workload (tinyc built by gcc -O0, clang -O0 and by tinyc, each running
     workload a).
  d  Corpus: pinned CoreMark / Lua / SQLite sources downloaded into the cache
     dir (sha256-checked); reports which ones tinyc builds and runs.

Every measurement is `--warmup` untimed runs, then `--runs` timed runs; the
tables report the median and the spread (min..max). Peak RSS is the largest
ru_maxrss of any process in the workload (os.wait4).

Usage:
    python bench/run.py                       # everything, 5 runs
    python bench/run.py --sections a,b --runs 3
    python bench/run.py --markdown bench/results/new.md \
        --notes bench/results/baseline-linux-x86_64.notes.md

Tools: gcc from PATH (or --gcc), clang from PATH or the pixi `linux` env
(or --clang), tcc from PATH (or --tcc). Builds go to bench/_work (or --work),
downloads to $TINYC_BENCH_CACHE (default ~/.cache/tinyc-bench).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"

DEFS = ["-DTINYC_ENABLE_ELF=1", "-DPLATFORM_SKIP_ENTRY=1"]
INC = ["-I", "corec", "-I", "corec-stdlib/stdlib", "-I", "."]
CC_INC = ["-nostdinc", "-fno-builtin", "-I", "bench/include"] + INC

# corec + corec-stdlib sources linked into every runtime-benchmark program
# (same set and order as examples/tinyc/run_tinyc_tests.py's unity TU).
COREC_BASE = ["assert", "exit", "mem", "string", "numconv", "format",
              "strbuf", "io", "buddy", "arena", "scratch"]
STDLIB = ["string_impl", "stdlib", "stdio", "printf"]

# ---------------------------------------------------------------------------
# Corpus (section d). Pinned archives; shims live in bench/corpus/<name>/.
# ---------------------------------------------------------------------------
CORPUS: list[dict] = []
try:
    sys.path.insert(0, str(BENCH))
    sys.dont_write_bytecode = True  # no __pycache__ in the source tree
    from corpus import CORPUS  # noqa: E402  (bench/corpus/__init__.py)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
@dataclass
class Run:
    wall: float
    maxrss_kb: int
    stderr: list[str] = field(default_factory=list)


class CmdFailed(RuntimeError):
    pass


def run_cmd(cmd: list[str], cwd: Path, capture_stderr: bool = False,
            stdout_path: Path | None = None) -> tuple[int, str]:
    """Run one process; return (maxrss_kb, stderr [+ stdout]). Raises on failure."""
    err_path = WORK / "_stderr.txt"
    out = open(stdout_path, "wb") if stdout_path else None
    with open(err_path, "wb") as ef:
        # Without stdout_path, stdout goes to the same file as stderr (tinyc
        # prints its diagnostics on stdout).
        p = subprocess.Popen(cmd, cwd=cwd, stdout=out or ef, stderr=ef)
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
    if stdout_path:
        out.close()
    err = err_path.read_text(errors="replace")
    if p.returncode != 0:
        raise CmdFailed(f"exit {p.returncode}: {' '.join(map(str, cmd))}\n{err[:2000]}")
    return ru.ru_maxrss, err if capture_stderr else ""


def run_workload(cmds: list[list[str]], cwd: Path, capture_stderr=False) -> Run:
    t0 = time.perf_counter()
    rss = 0
    errs = []
    for c in cmds:
        r, e = run_cmd(c, cwd, capture_stderr)
        rss = max(rss, r)
        errs.append(e)
    return Run(time.perf_counter() - t0, rss, errs)


@dataclass
class Stat:
    runs: list[Run]

    @property
    def walls(self):
        return [r.wall for r in self.runs]

    @property
    def median(self):
        return statistics.median(self.walls)

    @property
    def lo(self):
        return min(self.walls)

    @property
    def hi(self):
        return max(self.walls)

    @property
    def rss_mb(self):
        return max(r.maxrss_kb for r in self.runs) / 1024.0

    def spread(self) -> str:
        return f"{self.lo:.3f}..{self.hi:.3f}"


def measure(cmds, cwd=ROOT, capture_stderr=False) -> Stat:
    for _ in range(ARGS.warmup):
        run_workload(cmds, cwd)
    return Stat([run_workload(cmds, cwd, capture_stderr) for _ in range(ARGS.runs)])


# ---------------------------------------------------------------------------
# Environment / tools
# ---------------------------------------------------------------------------
def sh(cmd, cwd=ROOT) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              shell=isinstance(cmd, str)).stdout.strip()
    except OSError:
        return ""


def first_line(s: str) -> str:
    return s.splitlines()[0] if s else ""


def find_tool(explicit: str | None, name: str, pixi_envs=()) -> str | None:
    if explicit:
        return explicit
    p = shutil.which(name)
    if p:
        return p
    for env in pixi_envs:
        cand = ROOT / ".pixi" / "envs" / env / "bin" / name
        if cand.exists():
            return str(cand)
    return None


def loadavg() -> str:
    try:
        return "%.2f %.2f %.2f" % os.getloadavg()
    except OSError:
        return "?"


def machine_info() -> dict:
    info = {}
    lscpu = sh(["lscpu"])
    for key in ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket",
                "Socket(s)", "CPU max MHz"):
        m = re.search(rf"^{re.escape(key)}:\s*(.*)$", lscpu, re.M)
        if m:
            info[key] = m.group(1).strip()
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.exists():
        info["governor"] = gov.read_text().strip()
    try:
        mem = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text())
        info["memory"] = f"{int(mem.group(1)) / 1024 / 1024:.0f} GiB"
    except (OSError, AttributeError):
        pass
    osr = Path("/etc/os-release")
    if osr.exists():
        m = re.search(r'^PRETTY_NAME="?(.*?)"?$', osr.read_text(), re.M)
        if m:
            info["os"] = m.group(1)
    info["kernel"] = platform.release()
    return info


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def selfhost_sources(target: str = "x64") -> list[str]:
    """The tinyc self-host source list, parsed from selfhost_tinyc_elf.sh."""
    text = (ROOT / "examples/tinyc/selfhost_tinyc_elf.sh").read_text()
    arrays = {m.group(1): m.group(2).split()
              for m in re.finditer(r"^(\w+_C_FILES)=\(\n(.*?)\n\)", text, re.S | re.M)}
    order = ["COREC_C_FILES", "COREC_STDLIB_C_FILES", "TINYC_C_FILES",
             "NATIVE_C_FILES", "PLATFORM_C_FILES"]
    srcs = [f for k in order for f in arrays[k]]
    if target == "x64":
        srcs = ["corec/platform/platform_linux.c" if s.endswith("platform_wasm.c") else s
                for s in srcs]
    return srcs


def native_build_sources() -> list[str]:
    """Sources of a hosted (libc-linked) tinyc binary: build_tinyc_native.sh."""
    return [s for s in selfhost_sources("x64")
            if not s.startswith("corec-stdlib/")] + ["tests/upstream_main.c"]


def count_lines(files) -> int:
    return sum(len((ROOT / f).read_bytes().splitlines()) for f in files)


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------
def build_tinyc(cc: str, flags: list[str], out: Path, objdir: Path):
    """Build a hosted tinyc binary like build_tinyc_native.sh, with `cc flags`."""
    objdir.mkdir(parents=True, exist_ok=True)
    objs = []
    for src in native_build_sources():
        o = objdir / (src.replace("/", "_") + ".o")
        extra = ["-DPLATFORM_SKIP_ENTRY"] if src.endswith("platform_linux.c") else []
        run_cmd([cc, *flags, "-c", "-I", "corec", "-I", ".", *extra, "-o", str(o), src], ROOT)
        objs.append(str(o))
    run_cmd([cc, *flags, "-o", str(out), *objs, "-lm"], ROOT)


def build_stage1(tinyc: Path, out: Path):
    """tinyc compiled by tinyc: the self-host ELF pipeline (via wasm)."""
    stage = WORK / "stage1_elf"
    run_cmd(["bash", "examples/tinyc/selfhost_tinyc_elf.sh", str(tinyc), str(out), str(stage)], ROOT)


def unity_source(prog: Path, for_tinyc: bool, defines: list[str] = ()) -> Path:
    """One TU: the program + corec + corec-stdlib + platform_linux.c.

    tinyc --emit=elf synthesises its own crt0 (platform_init, then main), so
    its TU skips the platform's _start. gcc/clang link with -nostdlib and use
    platform_linux.c's _start -> app_main, so main is renamed to app_main."""
    rel = os.path.relpath
    d = WORK / "unity"
    d.mkdir(parents=True, exist_ok=True)
    u = d / f"{prog.stem}.{'tinyc' if for_tinyc else 'cc'}.c"
    lines = ["/* Generated by bench/run.py. */",
             "#define COREC_STDLIB_PROVIDES_MEM 1"]
    if for_tinyc:
        lines.append("#define PLATFORM_SKIP_ENTRY 1")
    lines += [f"#define {x}" for x in defines]
    lines += ["#include <stddef.h>", "#include <stdint.h>", "#include <stdarg.h>",
              "#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>"]
    if not for_tinyc:
        lines.append("#define main app_main")
    lines.append(f'#include "{rel(prog, d)}"')
    if not for_tinyc:
        lines.append("#undef main")
    lines += [f'#include "{rel(ROOT / "corec/base" / (b + ".c"), d)}"' for b in COREC_BASE]
    lines += [f'#include "{rel(ROOT / "corec-stdlib/stdlib" / (s + ".c"), d)}"' for s in STDLIB]
    lines.append(f'#include "{rel(ROOT / "corec/platform/platform_linux.c", d)}"')
    u.write_text("\n".join(lines) + "\n")
    return u


PROG_INC = ["-I", "corec-stdlib/stdlib", "-I", "corec"]


def program_build_cmd(compiler: str, unity: Path, exe: Path, extra_inc=()) -> list[str]:
    inc = PROG_INC + [a for d in extra_inc for a in ("-I", str(d))]
    if compiler == "tinyc":
        return [str(TOOLS["tinyc"]), "--emit=elf", *inc, "-o", str(exe), str(unity)]
    return [TOOLS[compiler], "-O0", "-static", "-nostdlib", "-nostdinc", "-fno-builtin",
            *inc, "-o", str(exe), str(unity)]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
REPORT: list[str] = []


def emit(line: str = ""):
    print(line, flush=True)
    REPORT.append(line)


def table(headers: list[str], rows: list[list[str]], align: str | None = None):
    align = align or ("l" + "r" * (len(headers) - 1))
    emit("| " + " | ".join(headers) + " |")
    emit("|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|")
    for r in rows:
        emit("| " + " | ".join(r) + " |")
    emit()


def fmt_s(x: float) -> str:
    return f"{x:.3f}"


# ---------------------------------------------------------------------------
# Section a: self-compile
# ---------------------------------------------------------------------------
def compile_cmds(compiler: str, srcs: list[str], extra: list[str] = ()) -> list[list[str]]:
    outdir = WORK / "out" / compiler
    outdir.mkdir(parents=True, exist_ok=True)
    cmds = []
    for s in srcs:
        o = str(outdir / (s.replace("/", "_") + ".o"))
        if compiler == "tinyc":
            cmds.append([str(TOOLS["tinyc"]), "--emit=x64", *extra, *DEFS, *INC, "-o", o, s])
        elif compiler == "tinyc-wasm":
            # Exactly the per-file step of selfhost_tinyc_elf.sh.
            cmds.append([str(TOOLS["tinyc"]), "--emit=wasm", "--lowering=native", *extra,
                         "-DTINYC_ENABLE_ELF=1", *INC, "-o", o, s])
        elif compiler == "tcc":
            cmds.append([TOOLS["tcc"], "-c", *DEFS, *CC_INC, "-o", o, s])
        else:
            cmds.append([TOOLS[compiler], "-O0", "-c", *DEFS, *CC_INC, "-o", o, s])
    return cmds


SELF = {}
RES: dict = {}  # numbers the summary at the top of the report is built from


def section_a():
    srcs = selfhost_sources("x64")
    nlines = count_lines(srcs)
    emit("## (a) Self-compile: the tinyc self-host sources, one process per file")
    emit()
    emit(f"{len(srcs)} files, {nlines} lines (`selfhost_tinyc_elf.sh` list, "
         "`platform_linux.c` for `platform_wasm.c`). Flags:")
    emit()
    emit(f"- tinyc: `--emit=x64 {' '.join(DEFS)} {' '.join(INC)}` (tinyc has no "
         "system include path, so `-nostdinc` and the `bench/include` stand-in "
         "for `<sys/stat.h>` are not needed: `driver.c` skips that header under "
         "`__TINYC__`)")
    emit(f"- gcc / clang: `-O0 -c {' '.join(DEFS)} {' '.join(CC_INC)}`; tcc: the "
         "same without `-O0`")
    emit(f"- the `--emit=wasm` row: `--emit=wasm --lowering=native -DTINYC_ENABLE_ELF=1 "
         f"{' '.join(INC)}` over the unmodified list (with `platform_wasm.c`), "
         "exactly the per-file step of `selfhost_tinyc_elf.sh`")
    emit()
    emit(f"Median of {ARGS.runs} runs after {ARGS.warmup} warmup; load average "
         f"before: {loadavg()}.")
    emit()
    rows = []
    compilers = [("tinyc", "tinyc --emit=x64 (gcc -O2 build)")]
    for c, label in (("gcc", "gcc -O0 -c"), ("clang", "clang -O0 -c"), ("tcc", "tcc -c")):
        if TOOLS.get(c):
            compilers.append((c, label))
    compilers.append(("tinyc-wasm", "tinyc --emit=wasm (self-host step)"))
    for c, label in compilers:
        files = selfhost_sources("wasm") if c == "tinyc-wasm" else srcs
        flines = count_lines(files)
        st = measure(compile_cmds(c, files))
        SELF[c] = st
        rows.append([label, fmt_s(st.median), st.spread(),
                     f"{len(files) / st.median:.0f}", f"{flines / st.median / 1000:.0f}",
                     f"{st.rss_mb:.0f}"])
    base = SELF["tinyc"].median
    for r, (c, _) in zip(rows, compilers):
        r.append(f"{SELF[c].median / base:.2f}")
    table(["compiler", "total s (median)", "min..max s", "files/s", "kLOC/s",
           "peak RSS MB", "time / tinyc"], rows)
    for c in ("clang", "tcc"):
        if not TOOLS.get(c):
            emit(f"**{c} not measured: {MISSING_HINT[c]}**")
            emit()

    # The whole path to a runnable tinyc executable from the same sources.
    emit("### (a2) Build a runnable tinyc executable from these sources")
    emit()
    emit("tinyc: the self-host pipeline (`selfhost_tinyc_elf.sh`: every file to a "
         "wasm object, `tinyc --link`, then `--from-wasm --emit=elf` lifts the "
         "linked module back to x86-64). gcc/clang: `-O0 -c` per file + link "
         "against glibc (the hosted `build_tinyc_native.sh` file list).")
    emit()
    rows = []
    st = measure([["bash", "examples/tinyc/selfhost_tinyc_elf.sh", str(TOOLS["tinyc"]),
                   str(WORK / "a2_tinyc_elf"), str(WORK / "a2_stage")]])
    SELF["a2-tinyc"] = st
    RES["files"], RES["lines"] = len(srcs), nlines
    rows.append(["tinyc self-host pipeline", fmt_s(st.median), st.spread(), f"{st.rss_mb:.0f}"])
    for c in ("gcc", "clang"):
        if not TOOLS.get(c):
            continue
        objdir = WORK / f"a2_{c}"
        objdir.mkdir(parents=True, exist_ok=True)
        cmds, objs = [], []
        for s in native_build_sources():
            o = str(objdir / (s.replace("/", "_") + ".o"))
            extra = ["-DPLATFORM_SKIP_ENTRY"] if s.endswith("platform_linux.c") else []
            cmds.append([TOOLS[c], "-O0", "-c", "-I", "corec", "-I", ".", *extra, "-o", o, s])
            objs.append(o)
        cmds.append([TOOLS[c], "-O0", "-o", str(objdir / "tinyc"), *objs, "-lm"])
        st = measure(cmds)
        SELF[f"a2-{c}"] = st
        rows.append([f"{c} -O0 compile + link", fmt_s(st.median), st.spread(), f"{st.rss_mb:.0f}"])
    table(["build", "total s (median)", "min..max s", "peak RSS MB"], rows)


# ---------------------------------------------------------------------------
# Section b: per-stage breakdown
# ---------------------------------------------------------------------------
STAGE_RE = re.compile(r"^\s+(\S+)\s+([0-9]+\.[0-9]+)\s+[0-9.]+%$")


def parse_stages(stderr: str) -> dict[str, float]:
    out = {}
    for line in stderr.splitlines():
        m = STAGE_RE.match(line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def stage_table(per_run: list[dict[str, float]], extra_note: str = ""):
    names = []
    for d in per_run:
        for k in d:
            if k not in names and k != "total":
                names.append(k)
    total = statistics.median(d.get("total", 0.0) for d in per_run)
    rows = []
    for k in names:
        med = statistics.median(d.get(k, 0.0) for d in per_run)
        rows.append([k, f"{med:.1f}", f"{100 * med / total:.1f}%" if total else "-"])
    rows.append(["**total**", f"**{total:.1f}**", "100%"])
    table(["stage", "ms (median)", "share"], rows)
    if extra_note:
        emit(extra_note)
        emit()
    meds = {k: statistics.median(d.get(k, 0.0) for d in per_run) for k in names}
    return meds, total


def section_b():
    srcs = selfhost_sources("x64")
    emit("## (b) Per-stage breakdown (`tinyc --time-stages`)")
    emit()
    emit("### (b1) Workload (a), summed over all files")
    emit()
    st = measure(compile_cmds("tinyc", srcs, ["--time-stages"]), capture_stderr=True)
    per_run = []
    per_file = {}
    for r in st.runs:
        tot = {}
        for src, err in zip(srcs, r.stderr):
            d = parse_stages(err)
            for k, v in d.items():
                tot[k] = tot.get(k, 0.0) + v
            per_file.setdefault(src, []).append(d)
        per_run.append(tot)
    note = ""
    if "tinyc" in SELF:
        ovh = (st.median - SELF["tinyc"].median) / SELF["tinyc"].median * 100
        note = (f"Workload wall time with `--time-stages`: {st.median:.3f} s vs "
                f"{SELF['tinyc'].median:.3f} s without ({ovh:+.1f}%, within noise "
                "means the timers cost nothing measurable). `total` is the sum of "
                "the in-process totals; the rest of the wall time is process "
                "start-up/exit.")
    RES['b1'] = stage_table(per_run, note)

    emit("Slowest files (median ms; top 3 stages):")
    emit()
    rows = []
    meds = {}
    for src, ds in per_file.items():
        keys = {k for d in ds for k in d}
        meds[src] = {k: statistics.median(d.get(k, 0.0) for d in ds) for k in keys}
    RES['b1_files'] = meds
    for src in sorted(meds, key=lambda s: -meds[s].get("total", 0))[:10]:
        m = meds[src]
        top = sorted(((v, k) for k, v in m.items() if k not in ("total", "other")), reverse=True)[:3]
        rows.append([f"`{src}`", f"{m.get('total', 0):.1f}",
                     ", ".join(f"{k} {v:.1f}" for v, k in top)])
    table(["file", "total ms", "top stages (ms)"], rows, "lrl")

    emit("### (b2) Self-host lift: `tinyc --from-wasm linked.wasm --emit=elf`")
    emit()
    emit("The last step of the self-host pipeline in (a2): the linked wasm module "
         "of all sources is lifted back to the llvm dialect, optimized and "
         "compiled to one x86-64 ELF.")
    emit()
    stage = WORK / "b2_stage"
    run_cmd(["bash", "examples/tinyc/selfhost_tinyc_elf.sh", str(TOOLS["tinyc"]),
             str(WORK / "b2_tinyc_elf"), str(stage)], ROOT)
    cmd = [str(TOOLS["tinyc"]), "--from-wasm", str(stage / "linked.wasm"), "--emit=elf",
           "--host-platform", "corec/platform/platform_linux.c",
           "--wasi-adapter", "corec/wasm/wasi_adapter.c", "-I", "corec", "-I", ".",
           "-o", str(WORK / "b2_lift_elf"), "--time-stages"]
    st = measure([cmd], capture_stderr=True)
    RES['b2_rss'] = st.rss_mb
    RES['b2'] = stage_table([parse_stages(r.stderr[0]) for r in st.runs],
                f"Peak RSS {st.rss_mb:.0f} MB.")


# ---------------------------------------------------------------------------
# Section c: runtime of generated code
# ---------------------------------------------------------------------------
def build_program(compiler: str, prog: Path, defines=(), extra_inc=()) -> Path:
    exe = WORK / "progs" / f"{prog.stem}.{compiler}"
    exe.parent.mkdir(parents=True, exist_ok=True)
    u = unity_source(prog, compiler == "tinyc", list(defines))
    run_cmd(program_build_cmd(compiler, u, exe, extra_inc), ROOT)
    return exe


def run_program_table(progs: list[tuple[str, Path, list, list, list[str]]]):
    """progs: (name, source, defines, extra include dirs, argv)."""
    comps = ["tinyc"] + [c for c in ("gcc", "clang") if TOOLS.get(c)]
    rows = []
    for name, src, defines, extra_inc, argv in progs:
        outs, stats = {}, {}
        for c in comps:
            try:
                exe = build_program(c, src, defines, extra_inc)
            except CmdFailed as e:
                outs[c] = None
                emit(f"- {name}: {c} build failed: `{str(e).splitlines()[0]}`")
                continue
            outp = WORK / "progs" / f"{name}.{c}.out"
            run_cmd([str(exe), *argv], WORK, stdout_path=outp)
            outs[c] = outp.read_bytes()
            stats[c] = measure([[str(exe), *argv]], cwd=WORK)
        ref = outs.get("gcc") or outs.get("clang")
        same = all(o == ref for o in outs.values() if o is not None) if ref else True
        row = [name]
        for c in comps:
            row.append(fmt_s(stats[c].median) if c in stats else "fail")
        base = stats.get("clang") or stats.get("gcc")
        row.append(f"{stats['tinyc'].median / base.median:.2f}" if "tinyc" in stats and base else "-")
        row.append("yes" if same else "**NO**")
        rows.append(row)
        if "tinyc" in stats and base:
            RES.setdefault("c_ratio", {})[name] = stats["tinyc"].median / base.median
    table(["program"] + [f"{c}{'' if c == 'tinyc' else ' -O0'} s" for c in comps]
          + ["tinyc / clang", "same output"], rows)


def section_c():
    emit("## (c) Runtime of the generated code")
    emit()
    emit("Kernels from `bench/programs`, each one TU together with corec + "
         "corec-stdlib + `platform_linux.c` (static, no libc), so every compiler "
         "compiles the same libc code too. tinyc: `--emit=elf`; gcc/clang: "
         f"`-O0 -static -nostdlib`. Median wall s of {ARGS.runs} runs.")
    emit()
    progs = [(p.stem, p, [], [], []) for p in sorted((BENCH / "programs").glob("*.c"))]
    run_program_table(progs)

    emit("### (c2) tinyc itself as the workload")
    emit()
    emit("The same compiler, built three ways, runs workload (a) (`--emit=x64` "
         "over every self-host source). gcc/clang builds are hosted (glibc); the "
         "tinyc build is `selfhost_tinyc_elf.sh` (tinyc compiles every file to "
         "wasm, links, lifts to x86-64), so it also carries the wasm32 "
         "linear-memory model and corec-stdlib in place of glibc.")
    emit()
    srcs = selfhost_sources("x64")
    builds = []
    for c in ("gcc", "clang"):
        if TOOLS.get(c):
            exe = WORK / f"tinyc_{c}_O0"
            build_tinyc(TOOLS[c], ["-O0"], exe, WORK / f"obj_{c}_O0")
            builds.append((f"tinyc built by {c} -O0", exe))
    exe = WORK / "tinyc_stage1_elf"
    build_stage1(TOOLS["tinyc"], exe)
    builds.append(("tinyc built by tinyc (self-host ELF)", exe))
    builds.append(("reference: tinyc built by gcc -O2", TOOLS["tinyc"]))
    rows = []
    stats = {}
    for label, exe in builds:
        cmds = [[str(exe), *c[1:]] for c in compile_cmds("tinyc", srcs)]
        st = measure(cmds)
        stats[label] = st
        RES.setdefault("c2", {})[label] = st.median
        rows.append([label, fmt_s(st.median), st.spread(), f"{st.rss_mb:.0f}"])
    refs = [(c, next((s for l, s in stats.items() if f"{c} -O0" in l), None))
            for c in ("gcc", "clang")]
    refs = [(c, s) for c, s in refs if s]
    for r, (label, _) in zip(rows, builds):
        for _, ref in refs:
            r.append(f"{stats[label].median / ref.median:.2f}")
    table(["compiler binary", "workload s (median)", "min..max s", "peak RSS MB"]
          + [f"time / {c} -O0 build" for c, _ in refs], rows)


# ---------------------------------------------------------------------------
# Section d: corpus
# ---------------------------------------------------------------------------
def cache_dir() -> Path:
    return Path(os.environ.get("TINYC_BENCH_CACHE", Path.home() / ".cache" / "tinyc-bench"))


def fetch(entry: dict) -> Path:
    """Download + verify + extract one corpus archive; return its source dir."""
    cache = cache_dir()
    dl = cache / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    archive = dl / entry["file"]
    if not archive.exists():
        print(f"  downloading {entry['url']}", flush=True)
        tmp = archive.with_suffix(archive.suffix + ".part")
        with urllib.request.urlopen(entry["url"], timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"{archive}: sha256 {digest} != pinned {entry['sha256']}")
    dest = cache / "src" / entry["name"]
    if not (dest / ".extracted").exists():
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        if entry["file"].endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        else:
            with tarfile.open(archive) as t:
                t.extractall(dest, filter="data")
        (dest / ".extracted").write_text(digest)
    return dest / entry.get("subdir", "")


def corpus_build_cmd(e: dict, src: Path, compiler: str, exe: Path) -> list[str]:
    u = e["unity"](src, WORK / "unity", compiler == "tinyc")
    inc = [a for d in e["inc"](src) for a in ("-I", str(d))] + \
        ["-I", str(ROOT / "corec-stdlib/stdlib"), "-I", str(ROOT / "corec")]
    defs = [f"-D{d}" for d in e["defines"]]
    if compiler == "tinyc":
        return [str(TOOLS["tinyc"]), "--emit=elf", *e["tinyc_flags"], *inc, *defs,
                "-o", str(exe), str(u)]
    return [TOOLS[compiler], "-O0", "-static", "-nostdlib", "-nostdinc", "-fno-builtin",
            "-w", *inc, *defs, "-o", str(exe), str(u), "-lgcc"]


def section_d():
    emit("## (d) Corpus")
    emit()
    emit("Pinned sources, downloaded to `$TINYC_BENCH_CACHE` and sha256-checked; the "
         "porting layer is in `bench/corpus/`. Each program is one TU with corec + "
         "corec-stdlib (no glibc); gcc/clang build the same TU with "
         "`-O0 -static -nostdlib -nostdinc`.")
    emit()
    (WORK / "unity").mkdir(parents=True, exist_ok=True)
    (WORK / "progs").mkdir(parents=True, exist_ok=True)
    comps = ["tinyc"] + [c for c in ("gcc", "clang") if TOOLS.get(c)]
    status_rows, compile_rows, run_rows = [], [], []
    for e in CORPUS:
        name = e["name"]
        try:
            src = fetch(e)
        except Exception as ex:  # noqa: BLE001
            status_rows.append([name, e["version"], "download failed", str(ex)[:100]])
            continue
        builds, outs, ctimes, notes = {}, {}, {}, []
        for c in comps:
            exe = WORK / "progs" / f"{name}.{c}"
            cmd = corpus_build_cmd(e, src, c, exe)
            try:
                st = measure([cmd]) if c == "tinyc" or "tinyc" in builds else None
                if st is None:
                    run_cmd(cmd, ROOT)
                else:
                    ctimes[c] = st
                builds[c] = exe
            except CmdFailed as ex:
                first = next((l for l in str(ex).splitlines()[1:] if l.strip()), "")
                notes.append(f"{c} build failed: `{first[:110]}`")
                continue
            outp = WORK / "progs" / f"{name}.{c}.out"
            try:
                run_cmd([str(exe), *e["argv"](src)], WORK, stdout_path=outp)
                outs[c] = outp.read_bytes()
            except CmdFailed as ex:
                notes.append(f"{c} binary failed: {str(ex).splitlines()[0][:80]}")
        ref = outs.get("gcc") or outs.get("clang")
        ok_t = "tinyc" in outs and ref is not None and outs["tinyc"] == ref and \
            e["check"].encode() in outs["tinyc"]
        if ok_t:
            status = "works"
        elif "tinyc" in builds:
            status = "**wrong output**"
        else:
            status = "blocked"
        RES.setdefault("d_status", {})[name] = status
        status_rows.append([name, e["version"], status,
                            "; ".join(notes + ([e["blockers"]] if e.get("blockers") else []))])
        if ok_t:
            compile_rows.append([name] + [fmt_s(ctimes[c].median) if c in ctimes else "-"
                                          for c in comps])
            stats = {c: measure([[str(builds[c]), *e["argv"](src)]], cwd=WORK)
                     for c in comps if c in outs}
            base = stats.get("clang") or stats.get("gcc")
            run_rows.append([name] + [fmt_s(stats[c].median) if c in stats else "-" for c in comps]
                            + [f"{stats['tinyc'].median / base.median:.2f}"])
            RES.setdefault("d_ratio", {})[name] = stats["tinyc"].median / base.median
    table(["program", "version", "tinyc", "notes"], status_rows, "llll")
    if compile_rows:
        emit("Compile time of the whole-program TU (median s):")
        emit()
        table(["program"] + [f"{c}{'' if c == 'tinyc' else ' -O0'}" for c in comps], compile_rows)
        emit("Runtime (median s; outputs identical across compilers):")
        emit()
        table(["program"] + [f"{c}{'' if c == 'tinyc' else ' -O0'}" for c in comps]
              + ["tinyc / clang"], run_rows)


MISSING_HINT = {
    "clang": "clang was not found on PATH or in the pixi `linux`/`wasm` envs; run "
             "`pixi install -e linux` or pass `--clang PATH`",
    "tcc": "tcc was not found on PATH (it is not on conda-forge); pass `--tcc PATH`",
}


def summary_lines() -> list[str]:
    """The `Summary` and `Hotspots` sections, computed from RES / SELF."""
    out = ["## Summary", ""]
    rows = []
    t = SELF.get("tinyc")
    if t:
        cells = [f"tinyc {t.median:.2f} s"] + [f"{c} -O0 {SELF[c].median:.2f} s"
                                               for c in ("clang", "gcc") if c in SELF]
        if "tcc" in SELF:
            cells.append(f"tcc {SELF['tcc'].median:.2f} s")
        gap = "-"
        if "clang" in SELF:
            r = SELF["clang"].median / t.median
            gap = (f"tinyc is {r:.2f}x faster than clang -O0; 10x needs "
                   f"{10 / r:.1f}x more")
        rows.append(["Self-compile (a), per file: >= 10x faster than clang -O0",
                     ", ".join(cells), gap])
        if "tcc" in SELF:
            r = t.median / SELF["tcc"].median
            rows.append(["Comparable to tcc", f"tinyc / tcc = {r:.2f}", "-"])
        else:
            rows.append(["Comparable to tcc", "not measured (no tcc)", "-"])
    if "a2-tinyc" in SELF:
        ref = SELF.get("a2-clang") or SELF.get("a2-gcc")
        rows.append(["Build a runnable tinyc (a2)",
                     f"self-host pipeline {SELF['a2-tinyc'].median:.1f} s, clang/gcc -O0 "
                     f"compile + link {ref.median:.2f} s" if ref else "-",
                     f"{SELF['a2-tinyc'].median / ref.median:.1f}x slower" if ref else "-"])
    cr = RES.get("c_ratio", {})
    if cr:
        lo, hi = min(cr.values()), max(cr.values())
        cell = f"kernels {lo:.2f}x..{hi:.2f}x of clang -O0"
        for n, r in RES.get("d_ratio", {}).items():
            cell += f"; {n} {r:.2f}x"
        c2 = RES.get("c2", {})
        g = next((v for k, v in c2.items() if "gcc -O0" in k), None)
        st1 = next((v for k, v in c2.items() if "self-host" in k), None)
        if g and st1:
            cell += f"; tinyc-built tinyc {st1 / g:.2f}x of the gcc -O0 build"
        rows.append(["Generated code as fast as clang -O0 (c)", cell, "-"])
    if rows:
        out += ["| goal (#328) | now | gap |", "|:---|:---|:---|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        out.append("")
    if "b1" in RES or "b2" in RES:
        out += ["## Hotspots", ""]
    if "b1" in RES:
        meds, total = RES["b1"]
        top = sorted(((v, k) for k, v in meds.items() if k != "other"), reverse=True)[:6]
        out.append(f"- per-file compile (b1, {total:.0f} ms over all files): " +
                   ", ".join(f"{k} {100 * v / total:.0f}%" for v, k in top))
        files = RES.get("b1_files", {})
        big = sorted(files, key=lambda f: -files[f].get("x64-regalloc", 0))[:2]
        if big:
            out.append("- largest x64-regalloc share: " + ", ".join(
                f"`{f}` {files[f]['x64-regalloc']:.0f} ms of {files[f].get('total', 0):.0f} ms"
                for f in big))
    if "b2" in RES:
        meds, total = RES["b2"]
        top = sorted(((v, k) for k, v in meds.items() if k != "other"), reverse=True)[:6]
        out.append(f"- self-host lift (b2, {total / 1000:.1f} s, peak RSS "
                   f"{RES.get('b2_rss', 0) / 1024:.1f} GB): " +
                   ", ".join(f"{k} {v / 1000:.2f} s" for v, k in top))
    if "b1" in RES or "b2" in RES:
        out.append("")
    if ARGS.notes:
        out += [Path(ARGS.notes).read_text().rstrip(), ""]
    return out


# ---------------------------------------------------------------------------
def main():
    global ARGS, WORK, TOOLS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--sections", default="a,b,c,d")
    ap.add_argument("--work", default=str(BENCH / "_work"))
    ap.add_argument("--markdown", help="also write the report to this file")
    ap.add_argument("--tinyc", help="use this tinyc binary instead of building one")
    ap.add_argument("--gcc")
    ap.add_argument("--clang")
    ap.add_argument("--tcc")
    ap.add_argument("--note", default="", help="free-form note for the report header")
    ap.add_argument("--notes", help="markdown file inserted after the computed summary "
                    "(analysis that the script cannot compute)")
    ARGS = ap.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        sys.exit("bench/run.py: Linux/x86_64 only (tinyc's x86-64 ELF backend)")
    WORK = Path(ARGS.work).resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    TOOLS = {
        "gcc": find_tool(ARGS.gcc, "gcc", ("upstream",)),
        "clang": find_tool(ARGS.clang, "clang", ("linux", "wasm")),
        "tcc": find_tool(ARGS.tcc, "tcc"),
    }
    for c in ("clang", "tcc"):
        if not TOOLS[c]:
            print(f"warning: {MISSING_HINT[c]}; its column is skipped", file=sys.stderr)
    if not TOOLS["gcc"]:
        sys.exit("bench/run.py: gcc not found (run under `pixi run -e upstream`)")
    if ARGS.tinyc:
        TOOLS["tinyc"] = Path(ARGS.tinyc).resolve()
    else:
        print("building tinyc (gcc -O2 -DNDEBUG) ...", flush=True)
        TOOLS["tinyc"] = WORK / "tinyc_O2"
        build_tinyc(TOOLS["gcc"], ["-O2", "-DNDEBUG"], TOOLS["tinyc"], WORK / "obj_gcc_O2")

    sha = sh(["git", "rev-parse", "HEAD"])
    dirty = " (dirty)" if sh(["git", "status", "--porcelain", "--untracked-files=no"]) else ""
    info = machine_info()
    emit("# tinyc benchmark: Linux x86-64")
    emit()
    emit("Generated by `bench/run.py` (`pixi run -e upstream bench`); regenerate "
         "with `--markdown FILE` and compare.")
    emit()
    emit(f"- date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit(f"- git: `{sha}`{dirty}, corec `{sh(['git', 'rev-parse', 'HEAD'], ROOT / 'corec')[:12]}`")
    emit(f"- CPU: {info.get('Model name', '?')}; {info.get('CPU(s)', '?')} logical CPUs "
         f"({info.get('Core(s) per socket', '?')} cores x {info.get('Thread(s) per core', '?')} "
         f"threads x {info.get('Socket(s)', '?')} socket), max {info.get('CPU max MHz', '?')} MHz, "
         f"governor {info.get('governor', '?')}; RAM {info.get('memory', '?')}")
    emit(f"- OS: {info.get('os', '?')}, kernel {info.get('kernel', '?')}")
    tinyc_desc = ("built from this tree by gcc -O2 -DNDEBUG" if not ARGS.tinyc
                  else f"`{ARGS.tinyc}` (given with --tinyc)")
    emit(f"- tinyc: {tinyc_desc}")
    for c in ("gcc", "clang", "tcc"):
        v = first_line(sh([TOOLS[c], "--version"] if c != "tcc" else [TOOLS[c], "-v"])) \
            if TOOLS.get(c) else "not found"
        emit(f"- {c}: {v}" if TOOLS.get(c) else f"- {c}: not measured: {MISSING_HINT[c]}")
    emit(f"- method: {ARGS.warmup} warmup + {ARGS.runs} timed runs per measurement, "
         "median reported, spread = min..max; all runs sequential (one process at a time)")
    emit(f"- load average at start (1/5/15 min): {loadavg()}")
    if ARGS.note:
        emit(f"- note: {ARGS.note}")
    emit()
    summary_at = len(REPORT)

    sections = set(ARGS.sections.split(","))
    for name, fn in (("a", section_a), ("b", section_b), ("c", section_c), ("d", section_d)):
        if name in sections:
            fn()
    emit(f"Load average at end (1/5/15 min): {loadavg()}.")
    REPORT[summary_at:summary_at] = summary_lines()
    if ARGS.markdown:
        Path(ARGS.markdown).write_text("\n".join(REPORT) + "\n")
        print(f"wrote {ARGS.markdown}")


ARGS = None
WORK = BENCH / "_work"
TOOLS: dict = {}

if __name__ == "__main__":
    main()
