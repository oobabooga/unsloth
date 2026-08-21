"""Patch the installed studio so every _run_amd_smi and mem_get_info call logs a stack."""
import io
import re
import sys

MARK = "# UNSLOTH_PROBE_TRACE"


def inject(path, pattern, indent, tag):
    src = io.open(path, encoding="utf-8").read()
    if MARK in src:
        print("already patched", path)
        return True
    m = re.search(pattern, src)
    if not m:
        print("NOT FOUND in", path, "pattern", pattern)
        return False
    pad = " " * indent
    inject_text = (
        m.group(0)
        + pad + MARK + "\n"
        + pad + "import traceback as _tb, sys as _sys\n"
        + pad + "_sys.stderr.write('@@@" + tag + "@@@\\n' + ''.join(_tb.format_stack()) + '@@@END@@@\\n')\n"
        + pad + "_sys.stderr.flush()\n"
    )
    src = src[: m.start(0)] + inject_text + src[m.end(0):]
    io.open(path, "w", encoding="utf-8").write(src)
    print("patched", path)
    return True


amd_py, torch_mem_py = sys.argv[1], sys.argv[2]
ok = inject(amd_py, r"\ndef _run_amd_smi\([^)]*\)[^:]*:\n(?:    \"\"\"(?:.|\n)*?\"\"\"\n)?", 4, "AMD_SMI")
ok = inject(torch_mem_py, r"\ndef mem_get_info\([^)]*\)[^:]*:\n", 4, "MEM_GET_INFO") and ok
raise SystemExit(0 if ok else 1)
