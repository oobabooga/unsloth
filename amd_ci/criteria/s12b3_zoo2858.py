#!/usr/bin/env python3
"""Criteria: unsloth-zoo PR 2858 on gfx1151 (ROCm). A refused hipHostRegister (same range twice) or hipHostUnregister
(range no longer registered) leaves its error pending; base's next kernel launch raises it, head consumes it.

Defect (base): each of the three GPU tests (`next_kernel`) FAILS, and the failure is the HIP already-registered
(registration tests) / not-registered (unregistration test) error raised inside `_launch`, i.e. after the refusal
was handled as expected. Fixed (head, merge): all three pass. Only the GPU tests are judged; the 9 CPU tests
(they reference the new helper) are reported, not judged.
Gates: torch is a HIP build with a GPU, device arch gfx1151, each state loaded block_swap*.py from its own tree,
base lacks / head has consume_cuda_error, the test file is the head's byte for byte.
"""

from __future__ import annotations

import re

TITLE = "unsloth-zoo PR 2858: refused hipHostRegister / hipHostUnregister error consumed before the next launch, gfx1151"
MODE = "differential"
# The change also has CUDA (cudaGetLastError) and Windows (amdhip64 / cudart64 DLL glob) branches: not reached here.
NEEDS = ["rocm", "gpu", "nvidia", "windows"]
GPU_TESTS = ["test_the_next_kernel_after_a_refused_arena_registration_runs",
             "test_the_next_kernel_after_a_refused_disk_buffer_registration_runs",
             "test_the_next_kernel_after_a_refused_unregistration_runs"]
TEST_SHA = "d218630ee73868cb96ed18c8ad1deb3b2629fff5c1f603e1734faa890bd12d45"
ALREADY = re.compile(r"HostMemoryAlreadyRegistered|already (mapped|registered)", re.I)
NOTREG = re.compile(r"HostMemoryNotRegistered|not correspond to a registered|not registered", re.I)
EXPECT = {GPU_TESTS[0]: ALREADY, GPU_TESTS[1]: ALREADY, GPU_TESTS[2]: NOTREG}


def _case(o, t):
    return (((o or {}).get("gpu") or {}).get(t) or {}).get("cases", {}).get(t) or {}


def _under(path, ck):
    return bool(path) and bool(ck) and str(path).replace("\\", "/").startswith(str(ck).replace("\\", "/") + "/")


def gates(obs):
    out = []
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if o is None:
            continue
        env = o.get("env") or {}
        out.append((f"{name}: venv built", not (o.get("venv") or {}).get("error"),
                    str((o.get("venv") or {}).get("error") or "ok")))
        out.append((f"{name}: torch is a HIP build with a GPU", bool(env.get("hip")) and env.get("cuda_available") is True,
                    f"torch {env.get('torch')} hip {env.get('hip')} available {env.get('cuda_available')}"))
        out.append((f"{name}: device is gfx1151", "gfx1151" in str(env.get("arch")), f"{env.get('device')} {env.get('arch')}"))
        imp = o.get("imported") or {}
        ck = o.get("checkout")
        ok = _under(imp.get("block_swap"), ck) and _under(imp.get("block_swap_disk"), ck) and imp.get("disk_tier_is_loaded_module") is True
        out.append((f"{name}: test loaded block_swap*.py from its own tree", ok,
                    f"{imp.get('block_swap')} | {imp.get('block_swap_disk')} | {imp.get('error', '')[:300] if imp.get('error') else ''}"))
        want = name != "base"
        out.append((f"{name}: consume_cuda_error {'present' if want else 'absent'}",
                    imp.get("disk_has_consume_cuda_error") is want and imp.get("block_swap_has__consume_cuda_error") is want,
                    f"disk {imp.get('disk_has_consume_cuda_error')} block_swap {imp.get('block_swap_has__consume_cuda_error')}"))
        tf = o.get("test_file") or {}
        out.append((f"{name}: test file is the head's (sha256)", tf.get("used_sha256") == TEST_SHA
                    and (not tf.get("present_in_tree") or tf.get("tree_sha256") == TEST_SHA),
                    f"used {str(tf.get('used_sha256'))[:12]} in-tree {tf.get('present_in_tree')}"))
        for t in GPU_TESTS:
            c = _case(o, t)
            out.append((f"{name}: {t} ran (not skipped / collected)", c.get("result") in ("passed", "failed", "error"),
                        str(c.get("result")) + " " + str(c.get("msg", ""))[:200]))
    return out


def _base_fail(o, t):
    c = _case(o, t)
    msg = c.get("msg") or ""
    return c.get("result") in ("failed", "error") and bool(EXPECT[t].search(msg)) and "_launch" in msg


def base_shows_defect(base):
    bad = [t for t in GPU_TESTS if not _base_fail(base, t)]
    return (not bad, "every GPU test raised the HIP error at the launch" if not bad else f"not shown by: {bad}")


def head_is_fixed(head):
    bad = [t for t in GPU_TESTS if _case(head, t).get("result") != "passed"]
    return (not bad, "all three GPU tests pass" if not bad else f"not passing: {bad}")


def state_is_fixed(name, obs):
    return head_is_fixed(obs)


def _first_err(msg):
    for ln in (msg or "").splitlines():
        if "Error" in ln and ("HIP" in ln or "hip" in ln or "CUDA" in ln or "registered" in ln or "mapped" in ln):
            return ln.strip()[:300]
    return (msg or "").splitlines()[0][:300] if msg else ""


def table(obs):
    rows = ["| state | test | result | error at launch |", "|---|---|---|---|"]
    vers = []
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        env = o.get("env") or {}
        vers.append(f"{name}: torch {env.get('torch')} hip {env.get('hip')} triton {env.get('triton')} "
                    f"device {env.get('device')} arch {env.get('arch')} commit {str(o.get('commit'))[:10]} "
                    f"block_swap {((o.get('imported') or {}).get('block_swap'))}")
        for t in GPU_TESTS:
            c = _case(o, t)
            rows.append(f"| {name} | {t} | {c.get('result')} | {_first_err(c.get('msg')) if c.get('result') != 'passed' else ''} |")
        cpu = ((o.get("cpu") or {}).get("cases") or {})
        p = sorted(k for k, v in cpu.items() if v["result"] == "passed")
        f = sorted(k for k, v in cpu.items() if v["result"] != "passed")
        rows.append(f"| {name} | CPU tests (not judged) | {len(p)} passed, {len(f)} not passed | {', '.join(f)[:600]} |")
    raw = (obs.get("base") or {}).get("raw_runtime")
    out = "\n".join(rows) + "\n\n" + "\n".join("- " + v for v in vers)
    if raw:
        out += f"\n\nRaw runtime (base, tree independent): {raw}"
    return out
