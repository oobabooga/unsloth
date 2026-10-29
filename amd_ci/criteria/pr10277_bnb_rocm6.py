#!/usr/bin/env python3
"""Criteria (PR 10277): on a ROCm 6.0-6.3 host, does the installed env run 4-bit QLoRA?

Issue 10273: on ROCm 6.1 the installer picks a rocm6.1 PyTorch, the generic
bitsandbytes wheel has no ROCm library older than rocm64, bnb falls forward to it,
and training dies with SIGSEGV. The PR floors the automatic generic pick at rocm6.4.

  base_shows_defect  in EVERY tested ROCm image, the base install chose a torch
                     older than HIP 6.4 AND its bnb 4-bit forward/backward failed
                     (crash, error, or non-finite / wrong output)
  head_is_fixed      in EVERY tested image, the head install chose HIP >= 6.4, the
                     bnb 4-bit step is correct, and a 3-step Unsloth QLoRA run
                     finishes with finite losses

Gates (non-vacuity) run first: every image must really be ROCm 6.x userspace, the
installer must have finished in both states, and plain torch must compute on the
GPU in both, otherwise a bnb failure could just be a broken harness.
"""

from __future__ import annotations

TITLE = "PR 10277: installer torch choice vs bitsandbytes on ROCm 6.x userspace (gfx1100 declared)"
MODE = "differential"
# discrete_gpu: the issue's RX 7900 XTX is discrete; this runner's gfx1151 is integrated.
NEEDS = ["rocm", "gpu", "docker", "discrete_gpu"]


def _hip(rec: dict) -> tuple[int, int] | None:
    h = ((rec.get("torch_info") or {}).get("hip") or "")
    try:
        a, b = h.split(".")[:2]
        return int(a), int(b)
    except ValueError:
        return None


def _bnb_ok(rec: dict) -> bool:
    b = rec.get("bnb_4bit") or {}
    return (b.get("rc") == 0 and b.get("loss_finite") is True and b.get("grad_finite") is True
            and (b.get("fwd_rel_err") is not None and b["fwd_rel_err"] < 0.05)
            and (b.get("dequant_rel_err") is not None and b["dequant_rel_err"] < 0.2))


def _train_ok(rec: dict) -> bool:
    u = rec.get("unsloth_qlora") or {}
    losses = u.get("losses") or []
    return u.get("rc") == 0 and len(losses) == 3 and all(l == l and abs(l) < 1e4 for l in losses)


def _images(state: dict) -> dict:
    return {k: v for k, v in (state.get("images") or {}).items()}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    tags = sorted({t for v in states.values() for t in _images(v)})
    out.append(("every state probed every image", bool(tags) and all(
        set(_images(v)) == set(tags) for v in states.values()), ", ".join(tags) or "none"))
    for n, v in states.items():
        for t, r in _images(v).items():
            ver = r.get("container_rocm_version") or ""
            out.append((f"{n} {t}: container is ROCm 6.0-6.3 userspace",
                        ver.startswith(("6.0", "6.1", "6.2", "6.3")), ver or "unknown"))
            out.append((f"{n} {t}: installer finished", r.get("install_rc") == 0 and bool(r.get("venv")),
                        f"rc={r.get('install_rc')} venv={bool(r.get('venv'))}"))
            ti = r.get("torch_info") or {}
            out.append((f"{n} {t}: plain torch computes on the GPU",
                        ti.get("rc") == 0 and ti.get("plain_matmul_ok") is True,
                        f"torch={ti.get('torch')} arch={ti.get('arch')} rc={ti.get('rc')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | image | ROCm | torch | HIP | device arch (build archs has it?) | torch.compile on CPU | bnb lib on import | bnb 4-bit fwd/bwd | Unsloth QLoRA 3 steps |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in obs.items():
        if n.startswith("_"):
            continue
        for t, r in sorted(_images(v).items()):
            ti, b, u = r.get("torch_info") or {}, r.get("bnb_4bit") or {}, r.get("unsloth_qlora") or {}
            libs = ", ".join(x.rsplit("/", 1)[-1] for x in b.get("loaded_bnb_libs") or []) or "-"
            bcell = ("ok (fwd err %.3g)" % b["fwd_rel_err"]) if _bnb_ok(r) else (
                f"FAIL rc={b.get('rc')}" + (f" SIG{b['signal']}" if b.get("signal") else ""))
            ucell = ("ok " + ", ".join(f"{l:.3f}" for l in u.get("losses", []))) if _train_ok(r) else (
                f"FAIL rc={u.get('rc')}" + (f" SIG{u['signal']}" if u.get("signal") else ""))
            arch = ti.get("arch")
            archcell = f"{arch} ({'yes' if arch and arch.split(':')[0] in (ti.get('arch_list') or []) else 'NO'})"
            dy = (r.get("dynamo") or {})
            dcell = "ok" if dy.get("dynamo_ok") else f"FAIL rc={dy.get('rc')}"
            bi = ", ".join((r.get("bnb_import") or {}).get("loaded_bnb_libs") or []) or "-"
            rows.append(f"| {n} | {t} | {r.get('container_rocm_version')} | {ti.get('torch')} | "
                        f"{ti.get('hip')} | {archcell} | {dcell} | {bi} | {bcell} | {ucell} |")
    rows += ["", "Declared arch gfx1100 via UNSLOTH_ROCM_GFX_ARCH, run through HSA_OVERRIDE_GFX_VERSION=11.0.0 "
             "on gfx1151 silicon. Containers share the runner's host kernel driver, which is new: an old "
             "amdgpu/KFD driver paired with a rocm6.4 torch is NOT exercised by this run."]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    imgs = _images(base)
    return bool(imgs) and all(
        (_hip(r) is not None and _hip(r) < (6, 4)) and not _bnb_ok(r) for r in imgs.values())


def head_is_fixed(head: dict) -> bool:
    imgs = _images(head)
    return bool(imgs) and all(
        (_hip(r) is not None and _hip(r) >= (6, 4)) and _bnb_ok(r) and _train_ok(r) for r in imgs.values())
