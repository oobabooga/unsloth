#!/usr/bin/env python3
"""Criteria: does the ROCm bundle survive GPU embedding once rocBLAS's kernel packs ship with it?

Pairs with probes/llama_rocm_kpack_probe.py (unslothai/unsloth#13663, unslothai/llama.cpp#277).

Defect: the released bundle's llama-server aborts once a request reaches hipblasSgemm, because
rocBLAS cannot load its own kernels (`hipErrorInvalidKernelFile`). Base shows it when EVERY
bundle configuration's GPU server dies on a request the CPU server answers. Fixed when every
configuration answers every request on the GPU, stays up, and its 64 vectors match the CPU
server's to cosine >= 0.999.

The gates make the comparison non-vacuous: the GPU server must really have offloaded to a ROCm
device (no torch on this job, so the harness cannot say so), the CPU reference must complete,
and the packaging step must be absent at base and applied at head.
"""

from __future__ import annotations

TITLE = "ROCm llama.cpp bundle: GPU embedding with rocBLAS kernel packs"
MODE = "differential"
# torch is not installed on this job, so rocm / gpu / integrated_gpu read UNDETERMINED in the
# gap report; the `rocm device` gate below is what proves the GPU ran.
NEEDS = ["rocm", "gpu", "integrated_gpu", "discrete_gpu", "linux", "windows", "multi_gpu"]
MIN_COS = 0.999
LONG = "long_64x500tok"


def _states(obs: dict):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _configs(state: dict) -> dict:
    return state.get("configs") or {}


def _all_ok(run: dict) -> bool:
    cases = run.get("cases") or []
    return (len(cases) == 4 and all(c.get("status") == 200 and c.get("server_alive_after")
                                    and c.get("finite") for c in cases))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    out = []
    probed = all(v.get("_probe_rc") == 0 and _configs(v) for v in states.values())
    out.append(("every state probed every configuration", probed,
                ", ".join(f"{n}: rc={v.get('_probe_rc')} configs={list(_configs(v))}" for n, v in states.items())))
    dev = [(n, c, cfg["gpu"].get("rocm_devices", 0), cfg["gpu"].get("rocm_model_buffer"))
           for n, v in states.items() for c, cfg in _configs(v).items()]
    out.append(("rocm device: every GPU server found a ROCm device and put the model on it",
                bool(dev) and all(d >= 1 and buf for _, _, d, buf in dev),
                ", ".join(f"{n}/{c}: devices={d} buffer={buf}" for n, c, d, buf in dev)))
    cpu = [(n, c, _all_ok(cfg["cpu"])) for n, v in states.items() for c, cfg in _configs(v).items()]
    out.append(("CPU reference answered every request", bool(cpu) and all(ok for *_, ok in cpu),
                ", ".join(f"{n}/{c}={ok}" for n, c, ok in cpu)))
    base, others = states.get("base") or {}, [v for n, v in states.items() if n != "base"]
    out.append(("packaging step absent at base, present elsewhere",
                base.get("packaging_step_present") is False and all(v.get("packaging_step_present") for v in others),
                ", ".join(f"{n}={v.get('packaging_step_present')}" for n, v in states.items())))
    return out


def table(obs: dict) -> str:
    rows = ["| state | config | kpack refs old/new | packs shipped | GPU: first failure | GPU exit | 64 x 500 tok | min cos vs CPU |",
            "|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        for c, cfg in _configs(v).items():
            g = cfg["gpu"]
            fail = next((f"{x['case']} -> {x.get('status')}" for x in g.get("cases", [])
                         if x.get("status") != 200 or not x.get("server_alive_after")), "none")
            if g.get("kernel_load_error"):
                fail += " (hipErrorInvalidKernelFile)"
            long = next((x for x in g.get("cases", []) if x["case"] == LONG and x.get("status") == 200), None)
            secs = "%s s" % long["seconds"] if long else "-"
            cos = cfg.get("gpu_vs_cpu_min_cos")
            cos_s = "%.5f" % cos if cos is not None else "-"
            refs = cfg.get("kpack_refs") or {}
            rows.append(f"| {n} | {c} | {refs.get('old')}/{refs.get('new')} | {len(cfg.get('shipped_kpacks') or [])} | "
                        f"{fail} | {g.get('exit_code')} | {secs} | {cos_s} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    cfgs = _configs(base)
    # The GPU server must have DIED under a request the CPU server answered; a refusal or a slow
    # answer is a different defect.
    return bool(cfgs) and all(any(x.get("server_alive_after") is False for x in c["gpu"].get("cases", []))
                              and _all_ok(c["cpu"]) for c in cfgs.values())


def head_is_fixed(head: dict) -> bool:
    cfgs = _configs(head)
    return bool(cfgs) and all(_all_ok(c["gpu"]) and (c.get("gpu_vs_cpu_min_cos") or 0) >= MIN_COS
                              for c in cfgs.values())
