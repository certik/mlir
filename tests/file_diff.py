#!/usr/bin/env python
"""Cross-platform binary file compare; exits non-zero on mismatch.

Used by Windows pixi tasks where coreutils `diff` is unavailable.
"""
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} EXPECTED ACTUAL", file=sys.stderr)
        return 2
    with open(sys.argv[1], "rb") as f:
        a = f.read()
    with open(sys.argv[2], "rb") as f:
        b = f.read()
    if a == b:
        return 0
    print(f"FAIL: {sys.argv[1]} != {sys.argv[2]}", file=sys.stderr)
    print(f"  expected {len(a)} bytes, got {len(b)} bytes", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
