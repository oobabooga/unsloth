"""Patch the installed torch so every mem_get_info / HIP-context call logs a stack."""
import io
import re
import sys

target = sys.argv[1]  # path to torch/cuda/memory.py
src = io.open(target, encoding="utf-8").read()

MARK = "# UNSLOTH_PROBE_TRACE"
if MARK in src:
    print("already patched")
    raise SystemExit(0)

pat = re.compile(r"(\ndef mem_get_info\([^)]*\)[^:]*:\n)")
m = pat.search(src)
if not m:
    print("mem_get_info def not found")
    raise SystemExit(1)

inject = (
    m.group(1)
    + "    " + MARK + "\n"
    + "    import traceback as _tb, sys as _sys\n"
    + "    _sys.stderr.write('@@@MEM_GET_INFO@@@\\n' + ''.join(_tb.format_stack()) + '@@@END@@@\\n')\n"
    + "    _sys.stderr.flush()\n"
)
src = src[: m.start(1)] + inject + src[m.end(1):]
io.open(target, "w", encoding="utf-8").write(src)
print("patched", target)
