#!/usr/bin/env python3
"""Diff two local GGUFs by header: tensor names, shapes and ggml types.

The local twin of gguf_remote_diff.py, for a CI job that has already downloaded
both files to run them.

Usage:
    python gguf_local_diff.py A.gguf B.gguf [--label-a V2 --label-b V3]
"""

from __future__ import annotations

import argparse
import struct
import sys
from typing import Any, BinaryIO

# ggml_type -> name, from ggml.h. Only the ones a GGUF is likely to carry.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}

# GGUF metadata value types.
(
    UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY,
    UINT64, INT64, FLOAT64,
) = range(13)

_SCALARS = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}


class RangeReader:
    """A plain local file, with the same read() the remote variant exposes."""

    def __init__(self, path: str) -> None:
        self._fh = open(path, "rb")
        self.bytes_fetched = 0

    def read(self, n: int) -> bytes:
        data = self._fh.read(n)
        if len(data) < n:
            raise EOFError("short read")
        self.bytes_fetched += n
        return data


def _u64(f: RangeReader) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _u32(f: RangeReader) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _string(f: RangeReader) -> str:
    return f.read(_u64(f)).decode("utf-8", errors = "replace")


def _value(f: RangeReader, vtype: int) -> Any:
    if vtype in _SCALARS:
        fmt, size = _SCALARS[vtype]
        return struct.unpack(fmt, f.read(size))[0]
    if vtype == STRING:
        return _string(f)
    if vtype == ARRAY:
        inner = _u32(f)
        count = _u64(f)
        # Token lists run to hundreds of thousands of entries and are not what
        # this tool compares; record the shape and skip the contents.
        if inner == STRING:
            if count > 64:
                for _ in range(count):
                    f.read(_u64(f))
                return f"<{count} strings>"
            return [_string(f) for _ in range(count)]
        fmt, size = _SCALARS[inner]
        if count > 64:
            f.read(size * count)
            return f"<{count} x {fmt}>"
        return [struct.unpack(fmt, f.read(size))[0] for _ in range(count)]
    raise ValueError(f"unknown metadata value type {vtype}")


def read_header(path: str) -> dict:
    f = RangeReader(path)
    magic = f.read(4)
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF: {magic!r}")
    version = _u32(f)
    n_tensors = _u64(f)
    n_kv = _u64(f)
    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key = _string(f)
        metadata[key] = _value(f, _u32(f))
    tensors = []
    for _ in range(n_tensors):
        name = _string(f)
        dims = _u32(f)
        shape = [_u64(f) for _ in range(dims)]
        ttype = _u32(f)
        _offset = _u64(f)
        tensors.append((name, tuple(shape), GGML_TYPES.get(ttype, f"type{ttype}")))
    return {
        "version": version,
        "metadata": metadata,
        "tensors": tensors,
        "bytes_fetched": f.bytes_fetched,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_a")
    parser.add_argument("path_b")
    parser.add_argument("--label-a", default = "A")
    parser.add_argument("--label-b", default = "B")
    args = parser.parse_args()

    a = read_header(args.path_a)
    b = read_header(args.path_b)
    for label, side in ((args.label_a, a), (args.label_b, b)):
        print(
            f"{label}: gguf v{side['version']}, {len(side['tensors'])} tensors, "
            f"{len(side['metadata'])} kv, header {side['bytes_fetched'] / 1e6:.1f} MB"
        )

    print("\n== metadata differences ==")
    keys = sorted(set(a["metadata"]) | set(b["metadata"]))
    diffs = 0
    for key in keys:
        va = a["metadata"].get(key, "<absent>")
        vb = b["metadata"].get(key, "<absent>")
        if va != vb:
            diffs += 1
            print(f"  {key}\n    {args.label_a}: {str(va)[:200]}\n    {args.label_b}: {str(vb)[:200]}")
    if not diffs:
        print("  (none)")

    ta = {name: (shape, ttype) for name, shape, ttype in a["tensors"]}
    tb = {name: (shape, ttype) for name, shape, ttype in b["tensors"]}

    print("\n== tensors only in one side ==")
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))
    for name in only_a:
        print(f"  {args.label_a} only: {name} {ta[name]}")
    for name in only_b:
        print(f"  {args.label_b} only: {name} {tb[name]}")
    if not only_a and not only_b:
        print("  (none)")

    print("\n== tensors whose type or shape changed ==")
    changed = 0
    type_moves: dict[tuple[str, str], int] = {}
    for name in sorted(set(ta) & set(tb)):
        if ta[name] != tb[name]:
            changed += 1
            type_moves[(ta[name][1], tb[name][1])] = (
                type_moves.get((ta[name][1], tb[name][1]), 0) + 1
            )
            if changed <= 60:
                print(f"  {name}: {ta[name][1]} {list(ta[name][0])} -> {tb[name][1]} {list(tb[name][0])}")
    if changed > 60:
        print(f"  ... and {changed - 60} more")
    if not changed:
        print("  (none)")

    print("\n== type moves, by count ==")
    for (src, dst), count in sorted(type_moves.items(), key = lambda kv: -kv[1]):
        print(f"  {src:>8} -> {dst:<8} {count}")

    def histogram(tensors):
        out: dict[str, int] = {}
        for _name, _shape, ttype in tensors:
            out[ttype] = out.get(ttype, 0) + 1
        return out

    print("\n== type histogram ==")
    ha, hb = histogram(a["tensors"]), histogram(b["tensors"])
    for ttype in sorted(set(ha) | set(hb)):
        print(f"  {ttype:>8}  {args.label_a}={ha.get(ttype, 0):<5} {args.label_b}={hb.get(ttype, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
