#!/usr/bin/env python3
"""Criteria: an experiment on the HOST, not a judgement on a diff.

The question is whether torch can be forced below Python to see two devices on this
one-GPU runner. Base and head are the same machine and must agree; the answer is in the
table, and the layered rows say which layer stopped it if it did.

Regression mode with nothing to regress, scored only on the two states disagreeing --
which would mean the reading is unstable and none of it can be trusted.
"""

from __future__ import annotations

TITLE = "Forcing torch to enumerate two devices via HIP interposition"
MODE = "regression"

NEEDS = ["rocm", "gpu", "integrated_gpu", "discrete_gpu", "multi_gpu", "multi_gpu_amd",
         "nvidia", "windows", "xpu", "mlx"]


def _ref(obs: dict) -> dict:
    return obs.get("head") or obs.get("base") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    out: list[tuple[str, bool, str]] = []

    built = {n: v.get("built") for n, v in states.items()}
    out.append(("the shim compiled", all(bool(v) for v in built.values()),
                ", ".join(f"{k}={v}" for k, v in built.items())))

    clean = {n: (v.get("torch_clean") or {}).get("device_count") for n, v in states.items()}
    out.append(("unshimmed torch sees exactly one device (control)",
                all(v == 1 for v in clean.values()),
                ", ".join(f"{k}={v}" for k, v in clean.items())))
    return out


def table(obs: dict) -> str:
    ref = _ref(obs)
    rows: list[str] = []

    rows += ["### Host", "", "| item | reading |", "|---|---|"]
    for label, key in (("ROCm version", "rocm_version"), ("libamdhip64", "hip_lib"),
                       ("amdsmi python", "amdsmi_python"), ("compiler", "cc")):
        rec = ref.get(key) or {}
        body = (rec.get("out") or rec.get("err") or rec.get("error") or "").strip()
        rows.append(f"| {label} | `{body[:200]}` |")
    b = (ref.get("build") or {})
    rows.append(f"| shim build | rc={b.get('rc')} {str(b.get('out') or '')[:160]} |")

    rows += ["", "### Layer by layer", "",
             "| layer | LD_PRELOAD | compiled hipGetDeviceCount | "
             "torch.device_count | C++ count |", "|---|---|---|---|---|"]
    layers = [
        ("no shim (control)", "cprobe_clean", "torch_clean", "no"),
        ("shim, LD_PRELOAD only", "cprobe_shimmed", "torch_shimmed", "yes"),
        ("shim + dlsym hook", "cprobe_shimmed", "torch_shimmed_dlsym", "yes"),
        ("shim + dlsym hook, no expandable segments", "cprobe_shimmed",
         "torch_shimmed_dlsym_noexp", "yes"),
        ("amdsmi blocked, no shim", "cprobe_clean", "torch_no_amdsmi_only", "no"),
    ]
    for label, ckey, tkey, pre in layers:
        c = ref.get(ckey) or {}
        t = ref.get(tkey) or {}
        rows.append(
            f"| {label} | {pre} | {c.get('out', c.get('error', '-'))} | "
            f"{t.get('device_count', t.get('error', '-'))} | "
            f"{t.get('cpp_device_count', '-')} |")

    rows += ["", "| ctypes, shimmed (NOT an interposition test) | "
             f"{(ref.get('ctypes_shimmed') or {}).get('count')} |", "|---|---|"]

    rows += ["", "### Can a tensor actually live on the phantom device", "",
             "| layer | archs | device 0 matmul | device 1 matmul |", "|---|---|---|---|"]
    for label, _c, tkey, _p in layers:
        t = ref.get(tkey) or {}
        rows.append(f"| {label} | `{t.get('device_archs')}` | "
                    f"{str(t.get('device_0_matmul'))[:80]} | "
                    f"{str(t.get('device_1_matmul'))[:80]} |")

    stderrs = []
    for label, _c, tkey, _p in layers:
        t = ref.get(tkey) or {}
        if t.get("no_json"):
            stderrs.append(f"**{label}** exited {t.get('_rc')} without reporting:\n\n"
                           f"```\n{str(t.get('stderr'))[-900:]}\n```")
    if stderrs:
        rows += ["", "### Where it died", ""] + stderrs

    rows += [
        "",
        "The last row is the one that matters, and it is the only question Python "
        "patching cannot answer. A phantom ordinal that enumerates but cannot hold a "
        "tensor is still a selection input; a phantom ordinal that runs a real matmul "
        "is a device as far as every layer above it can tell.",
        "",
        "Read `no shim` first. If unshimmed torch does not report exactly 1, the host "
        "changed and nothing below it means anything.",
        "",
        "`amdsmi blocked, no shim` is the control that separates the two mechanisms. On "
        "ROCm, `torch.cuda.device_count()` prefers `_device_count_amdsmi()` and only "
        "falls back to the HIP count when it fails, so blocking amdsmi changes WHICH "
        "path counts. Without this row, a count of 2 in the row above could be the "
        "fallback path disagreeing rather than the shim working. On this runner amdsmi "
        "is not installed (`has_pynvml` False), so torch already uses the HIP count and "
        "this row is expected to be inert.",
        "",
        "The ctypes row is recorded only as a warning to the next reader. ctypes "
        "resolves through a `dlopen` handle and is therefore NOT interposed by "
        "LD_PRELOAD, so it cannot test the shim. The first run of this probe used it as "
        "the interposition check, read 1, and looked like a clean negative -- while the "
        "shim was in fact working and torch was aborting downstream.",
    ]
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = (base.get("torch_clean") or {}).get("device_count")
    h = (head.get("torch_clean") or {}).get("device_count")
    if b != h:
        return True, (f"the states disagree on the unshimmed device count ({b} vs {h}), "
                      f"so the reading is unstable")
    best = head.get("torch_shimmed_dlsym_noexp") or head.get("torch_shimmed_dlsym") \
        or head.get("torch_shimmed") or {}
    forced = best.get("device_count")
    mm = str(best.get("device_1_matmul"))
    return False, (f"host experiment, not a test of the change: unshimmed torch sees "
                   f"{h}, shimmed it sees {forced}, and a matmul on the phantom device "
                   f"came back {mm[:60]!r}")
