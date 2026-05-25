#!/usr/bin/env python3
"""
CI verification: assert that the `native` and `upstream_api` tinyc
configurations produce *byte-identical* wasm output on the full tinyc
self-host source set.

Both configurations use the same native lowering pipeline
(mlir_lift_cf_to_scf + mlir_lower_to_llvm + mlir_llvm_to_wasmssa +
mlir_wasmssa_to_wasmstack + mlir_wasmstack_to_bin). The only thing that
varies is the MLIR_* API implementation backing tinyc_*_opt:

  - native:        mlir_api_impl.c       (corec arena, no libc)
  - upstream_api:  mlir_api_impl_upstream.cpp (upstream LLVM/MLIR C++)

Any divergence in IR construction, lookup or traversal order between
the two impls shows up here as a differing `.wasm.o` byte. We compile
every source twice, link both sets, and `cmp` every produced file.
Exits nonzero on any difference.

Source list, include flags, and per-config command lines are imported
from `bench_tinyc_compile.py` so the two scripts stay in sync.

Usage:
    pixi run verify_tinyc_native_upstream_identical
or:
    python examples/tinyc/verify_native_upstream_identical.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Reuse the bench script's source list / Config helpers.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from bench_tinyc_compile import (  # noqa: E402
    ALL_SOURCES,
    Config,
    ROOT,
    discover_configs,
)


VERIFY_DIR = ROOT / "verify_tinyc_out"


def compile_one(cfg: Config, src: Path, obj: Path) -> None:
    if obj.exists():
        obj.unlink()
    proc = subprocess.run(
        cfg.cmd_compile(src, obj),
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=cfg.env(),
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(
            f"error: {cfg.name} failed to compile {src.relative_to(ROOT)}:\n"
            f"  cmd: {' '.join(cfg.cmd_compile(src, obj))}\n"
            f"  returncode: {proc.returncode}\n"
            f"  stdout:\n{(proc.stdout or '').rstrip()}\n"
            f"  stderr:\n{tail}"
        )


def link_all(cfg: Config, objs: list[Path], out: Path,
             extras: list[Path] | None = None) -> None:
    if out.exists():
        out.unlink()
    proc = subprocess.run(
        cfg.cmd_link(objs, out, extras),
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=cfg.env(),
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(
            f"error: {cfg.name} failed to link:\n"
            f"  cmd: {' '.join(cfg.cmd_link(objs, out, extras))}\n"
            f"  stderr:\n{tail}"
        )


def cmp_bytes(a: Path, b: Path, label: str) -> bool:
    ab = a.read_bytes()
    bb = b.read_bytes()
    if ab == bb:
        return True
    n = min(len(ab), len(bb))
    first_diff = next((i for i in range(n) if ab[i] != bb[i]), n)
    print(
        f"  DIFFER  {label}: sizes {len(ab)} vs {len(bb)}, "
        f"first diff at byte {first_diff}"
    )
    return False


def main() -> int:
    configs = {c.name: c for c in discover_configs()}
    if "native" not in configs or "upstream_api" not in configs:
        print("error: required configs (native, upstream_api) not available",
              file=sys.stderr)
        return 2
    cfg_n = configs["native"]
    cfg_u = configs["upstream_api"]

    if VERIFY_DIR.exists():
        shutil.rmtree(VERIFY_DIR)
    obj_dir_n = VERIFY_DIR / "native"
    obj_dir_u = VERIFY_DIR / "upstream_api"
    obj_dir_n.mkdir(parents=True)
    obj_dir_u.mkdir(parents=True)

    n_files = len(ALL_SOURCES)
    print(f"verifying {n_files} .wasm.o files (native vs upstream_api)...")

    objs_n: list[Path] = []
    objs_u: list[Path] = []
    diffs = 0
    for src_rel in ALL_SOURCES:
        src = ROOT / src_rel
        flat = src_rel.replace("/", "_").replace("\\", "_") + ".wasm.o"
        obj_n = obj_dir_n / flat
        obj_u = obj_dir_u / flat
        compile_one(cfg_n, src, obj_n)
        compile_one(cfg_u, src, obj_u)
        objs_n.append(obj_n)
        objs_u.append(obj_u)
        if not cmp_bytes(obj_n, obj_u, flat):
            diffs += 1

    # Link both and cmp the final wasm.
    vararg = ROOT / "tinyc_wasm_vararg.wasm.o"
    extras = [vararg] if vararg.exists() else None
    out_n = obj_dir_n / "tinyc_bench.wasm"
    out_u = obj_dir_u / "tinyc_bench.wasm"
    link_all(cfg_n, objs_n, out_n, extras)
    link_all(cfg_u, objs_u, out_u, extras)
    linked_ok = cmp_bytes(out_n, out_u, "tinyc_bench.wasm")

    print()
    if diffs == 0 and linked_ok:
        print(f"OK: all {n_files} .wasm.o files + linked tinyc_bench.wasm "
              f"are byte-identical between native and upstream_api")
        return 0
    print(f"FAIL: {diffs} of {n_files} .wasm.o files differ; "
          f"linked wasm differs: {not linked_ok}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
