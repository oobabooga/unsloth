#!/usr/bin/env python3
"""Criteria: zoo main fused linear losses in fp16 on gfx1151 (probes/fl_fp16_probe.py). Here "base" means the
plain-torch reference (HF CE with fp32-upcast logits, TRL selective_log_softmax / generalized_jsd_loss in fp16) and
"head" the fused ops; all arms ran in the head invocation on zoo main.

Gates (INCONCLUSIVE unless all hold): every arm exited 0; each arm imported unsloth_zoo from the head checkout; GPU is
gfx1151 (Radeon 8060S); the fused arms engaged the fused ops (CALLS ce / logprobs / jsd > 0), and the Triton bodies
(CALLS triton > 0) in the accuracy arm; the training arm ran SFT and GRPO.
head_is_worse (REGRESSION) if any of:
  - accuracy: a fused case errors or is non-finite, or its error vs the fp64 oracle exceeds the reference's by more
    than 25% + 1e-3 (loss abs, grad_hidden rel-max, grad_weight rel-max)
  - perf (per workload): fused median ms above reference median by more than max(5%, reference A/A spread), or fused
    peak memory above the reference's
  - training: not ok, a non-finite loss, or the SFT / GRPO steps never called a fused op
"""

from __future__ import annotations

import math
import os
import statistics

TITLE = "Fused linear losses (CE / log-probs / GKD JSD) in fp16 on gfx1151 vs plain-torch HF / TRL math"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
OPS = ("ce", "logprobs", "jsd")


def _arms(obs):
    return (obs.get("head") or {}).get("arms") or {}


def _res(obs, name):
    return (_arms(obs).get(name) or {}).get("result") or {}


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _sum_calls(d, key):
    return sum((c.get("fused") or {}).get("calls", {}).get(key, 0) or 0 for c in d.get("cases") or [])


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    h, arms = obs.get("head") or {}, _arms(obs)
    co = h.get("checkout") or ""
    out = [("all 6 arms present", len(arms) == 6, ", ".join(arms))]
    for name, a in arms.items():
        r = a.get("result") or {}
        out.append((f"{name}: exited 0 with a result", a.get("rc") == 0 and bool(r),
                    f"rc={a.get('rc')} {'' if a.get('rc') == 0 and r else str(a.get('tail'))[-400:]}"))
        if r.get("zoo"):
            out.append((f"{name}: zoo from the head checkout", _norm(r["zoo"]).startswith(_norm(co)), f"{r['zoo']} vs {co}"))
        if r.get("gpu"):
            out.append((f"{name}: AMD gfx1151", any(s in r["gpu"] for s in ("8060S", "gfx1151", "Radeon")), r["gpu"]))
    acc = _res(obs, "acc")
    for op in OPS:
        out.append((f"acc: fused {op} engaged", _sum_calls(acc, op) > 0, str(_sum_calls(acc, op))))
    out.append(("acc: Triton bodies engaged", _sum_calls(acc, "triton") > 0, str(_sum_calls(acc, "triton"))))
    t = _res(obs, "train")
    out.append(("train: SFT and GRPO ran", bool(t.get("sft_losses")) and bool(t.get("grpo_losses")) or not t.get("ok"),
                f"ok={t.get('ok')} sft={t.get('sft_losses')} grpo={t.get('grpo_losses')}"))
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def table(obs: dict) -> str:
    acc = _res(obs, "acc")
    lines = [f"GPU `{acc.get('gpu')}` torch `{acc.get('torch')}` hip `{acc.get('hip')}`", "",
             "| V | op | head | fused loss / grad_h / grad_w err | reference loss / grad_h / grad_w err | fused calls |",
             "|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.2e}"  # noqa: E731
    for c in acc.get("cases") or []:
        fu, rf = c.get("fused") or {}, c.get("ref") or {}
        lines.append(f"| {c['v']} | {c['op']} | {'trainable' if c['trainable_head'] else 'frozen'} | "
                     f"{fu.get('error') or ' / '.join(f(fu.get(k)) for k in ('loss_abs', 'grad_h_rel', 'grad_w_rel'))} | "
                     f"{rf.get('error') or ' / '.join(f(rf.get(k)) for k in ('loss_abs', 'grad_h_rel', 'grad_w_rel'))} | "
                     f"{fu.get('calls')} |")
    lines += ["", f"Perf {_res(obs, 'perf_fused_1').get('shape')} fp16, frozen head, fwd+bwd:", "",
              "| op | fused ms (1 / 2) | ref ms (1 / 2) | fused peak MiB | ref peak MiB |", "|---|---|---|---|---|"]
    for op in OPS:
        fr = [(_res(obs, f"perf_fused_{i}").get(op) or {}) for i in (1, 2)]
        rr = [(_res(obs, f"perf_ref_{i}").get(op) or {}) for i in (1, 2)]
        lines.append(f"| {op} | {[x.get('ms') or x.get('error') for x in fr]} | {[x.get('ms') or x.get('error') for x in rr]} | "
                     f"{[x.get('peak_mib') for x in fr]} | {[x.get('peak_mib') for x in rr]} |")
    t = _res(obs, "train")
    lines += ["", f"Train (tiny Qwen3 fp16 LoRA, {t.get('trl')} / {t.get('transformers')} / unsloth {t.get('unsloth')}): "
              f"ok={t.get('ok')} SFT losses {t.get('sft_losses')} calls {t.get('sft_calls')}; GRPO losses "
              f"{t.get('grpo_losses')} calls {t.get('grpo_calls')} {t.get('error') or ''}"]
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    obs = {"head": head, "base": base}
    why = []
    for c in _res(obs, "acc").get("cases") or []:
        fu, rf = c.get("fused") or {}, c.get("ref") or {}
        tag = f"V={c['v']} {c['op']} {'trainable' if c['trainable_head'] else 'frozen'}"
        if fu.get("error") or not fu.get("finite"):
            why.append(f"{tag}: fused {fu.get('error') or 'non-finite'}")
            continue
        for k in ("loss_abs", "grad_h_rel", "grad_w_rel"):
            if fu.get(k) is None or rf.get(k) is None:
                continue
            if fu[k] > rf[k] * 1.25 + 1e-3:
                why.append(f"{tag} {k}: fused {fu[k]:.2e} vs reference {rf[k]:.2e}")
    for op in OPS:
        fr = [(_res(obs, f"perf_fused_{i}").get(op) or {}).get("ms") for i in (1, 2)]
        rr = [(_res(obs, f"perf_ref_{i}").get(op) or {}).get("ms") for i in (1, 2)]
        if None in fr or None in rr:
            why.append(f"perf {op}: missing fused {fr} ref {rr}")
            continue
        tol = max(0.05, abs(rr[0] - rr[1]) / min(rr))
        if _med(fr) > _med(rr) * (1 + tol):
            why.append(f"perf {op}: fused {fr} ms vs reference {rr} (> {tol:.1%})")
        fp = _med([(_res(obs, f"perf_fused_{i}").get(op) or {}).get("peak_mib") for i in (1, 2)])
        rp = _med([(_res(obs, f"perf_ref_{i}").get(op) or {}).get("peak_mib") for i in (1, 2)])
        if fp is not None and rp is not None and fp > rp:
            why.append(f"perf {op}: fused peak {fp} MiB vs reference {rp}")
    t = _res(obs, "train")
    ls = (t.get("sft_losses") or []) + (t.get("grpo_losses") or [])
    if not t.get("ok") or not ls or any(not math.isfinite(x) for x in ls):
        why.append(f"train: ok={t.get('ok')} losses {ls} {t.get('error') or ''}")
    if not sum((t.get("sft_calls") or {}).get(k, 0) for k in ("ce", "logprobs")):
        why.append(f"train: SFT never called a fused op ({t.get('sft_calls')})")
    if not sum((t.get("grpo_calls") or {}).get(k, 0) for k in ("ce", "logprobs")):
        why.append(f"train: GRPO never called a fused op ({t.get('grpo_calls')})")
    return bool(why), "; ".join(why) or "fused within tolerance of the reference on accuracy, time, memory and training"
