#!/usr/bin/env python3
"""Criteria: unsloth-zoo PR 2858 on gfx1151 (ROCm). A refused hipHostRegister (same range twice) or hipHostUnregister
(range no longer registered) leaves its error pending; base's next kernel launch raises it, head consumes it.

Defect (base): each of the three judged GPU tests ([zero_bytes] arena + disk, refused_unregistration) FAILS, and the
failure is the HIP invalid-argument (zero-byte registration) / not-registered (unregistration) error raised inside `_launch`, i.e. after the refusal
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
FUNCS = ["test_the_next_kernel_after_a_refused_arena_registration_runs",
         "test_the_next_kernel_after_a_refused_disk_buffer_registration_runs",
         "test_the_next_kernel_after_a_refused_unregistration_runs"]
# Judged: the three refusals HIP really gives. The [already_registered] cases (HIP accepts a second registration)
# must be SKIPPED in every state.
GPU_TESTS = [FUNCS[0] + "[zero_bytes]", FUNCS[1] + "[zero_bytes]", FUNCS[2]]
SKIPPED = [FUNCS[0] + "[already_registered]", FUNCS[1] + "[already_registered]"]
TEST_SHA = "f0d7574e9d4fd9fb3ddca2b898e87ab9ad82afc250fabe7b26d74e220c241da3"
INVALID = re.compile(r"invalid argument|hipErrorInvalidValue|invalid value", re.I)
NOTREG = re.compile(r"HostMemoryNotRegistered|not correspond to a registered|not registered", re.I)
EXPECT = {GPU_TESTS[0]: INVALID, GPU_TESTS[1]: INVALID, GPU_TESTS[2]: NOTREG}


def _case(o, t):
    for r in ((o or {}).get("gpu") or {}).values():
        c = (r.get("cases") or {}).get(t)
        if c:
            return c
    return {}


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
        for t in SKIPPED:
            c = _case(o, t)
            out.append((f"{name}: {t} skipped (HIP)", c.get("result") == "skipped", str(c.get("result")) + " " + str(c.get("msg", ""))[:150]))
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
        for t in GPU_TESTS + SKIPPED:
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
    sup = (obs.get("base") or {}).get("supp") or {}
    if sup.get("raw"):
        out += "\n\nSupplementary (not judged), raw hipHostRegister refusals then a launch (base process):\n"
        out += "\n".join(f"- {k}: pre rc {v.get('pre_register_rc')} refused rc {v.get('refused_rc')} launch {v.get('launch')}"
                         + ("" if v.get("done") else f" CRASH rc {v.get('rc')}") for k, v in sup["raw"].items())
    trows = ["| state | refusal / target | register rcs | construct | next launch |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        for k, v in (((obs.get(name) or {}).get("supp") or {}).get("tree") or {}).items():
            trows.append(f"| {name} | {k} | {v.get('register_rcs')} | {v.get('construct')} | "
                         f"{v.get('launch') if v.get('done') else 'CRASH rc ' + str(v.get('rc'))} |")
    if len(trows) > 2:
        out += "\n\nSupplementary (not judged), this tree's _Registered / _HostBuf with the registration redirected to that refusal:\n\n" + "\n".join(trows)
    return out
