"""Shared judgement for probes/core_g4_sdpa_probe.py (imported by core_g4_sdpa_win.py / core_g4_sdpa_lnx.py)."""

from __future__ import annotations

NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "linux", "nvidia", "discrete_gpu"]
# Forward: sdpa must match eager (bf16 kernels differ only by rounding).
TOP1_MIN = 0.98
NLL_GAP_MAX = 0.1
# Training: per step |loss_sdpa - loss_eager|; step 1 is a pure forward (LoRA B starts at 0).
STEP1_GAP_MAX = 0.05
MEAN_GAP_MAX = 0.1
FINAL_GAP_MAX = 0.2


def task(o: dict, name: str) -> dict:
    return ((o or {}).get("tasks") or {}).get(name) or {}


def fwd(o: dict) -> dict:
    return task(o, "fwd").get("fwd") or {}


def losses(o: dict, name: str) -> list:
    return (task(o, name).get("train") or {}).get("losses") or []


def fwd_wrong(f: dict) -> tuple[bool, str]:
    if "top1_agree" not in f:
        return False, "not measured"
    bad = (not f.get("finite") or f["top1_agree"] < TOP1_MIN
           or (f.get("nll_sdpa", 0) - f.get("nll_eager", 0)) > NLL_GAP_MAX)
    return bad, (f"top1 {f['top1_agree']}, nll sdpa {f.get('nll_sdpa')} vs eager {f.get('nll_eager')}, "
                 f"KL mean {f.get('kl_eager_sdpa_mean')}")


def gaps(o: dict) -> dict | None:
    s, e = losses(o, "train_sdpa"), losses(o, "train_eager")
    if not s or not e or len(s) != len(e):
        return None
    d = [abs(a - b) if a == a and b == b else float("inf") for a, b in zip(s, e)]
    return {"step1": d[0], "mean": sum(d) / len(d), "final": d[-1], "max": max(d)}


def train_wrong(o: dict) -> tuple[bool, str]:
    g = gaps(o)
    if g is None:
        return False, "not measured"
    bad = g["step1"] > STEP1_GAP_MAX or g["mean"] > MEAN_GAP_MAX or g["final"] > FINAL_GAP_MAX
    s, e = losses(o, "train_sdpa"), losses(o, "train_eager")
    return bad, (f"|sdpa-eager| step1 {g['step1']:.4f} mean {g['mean']:.4f} final {g['final']:.4f}; "
                 f"sdpa {s[0]}->{s[-1]}, eager {e[0]}->{e[-1]}")


def wide_ops(o: dict) -> dict:
    """Attention ops recorded at head dims above 256 in the profiled sdpa forward."""
    out = {}
    for k, n in (fwd(o).get("sdpa_ops") or {}).items():
        op, _, head = k.partition("@")
        try:
            if int(head) > 256:
                out[op] = out.get(op, 0) + n
        except ValueError:
            continue
    return out


def common_gates(obs: dict, states=("base", "head")) -> list:
    out = []
    for name in states:
        o = obs.get(name) or {}
        out.append((f"{name}: imported its own unsloth tree", bool(o.get("unsloth_from_checkout")),
                    f"{o.get('unsloth_file')}"))
        f = fwd(o)
        out.append((f"{name}: sdpa vs eager forward compared", "top1_agree" in f,
                    str(task(o, "fwd").get("error") or f"{o.get('torch')} {o.get('arch')} tf {o.get('transformers')}")[:300]))
        out.append((f"{name}: sdpa forward reached a head > 256 attention op", bool(wide_ops(o)),
                    str(fwd(o).get("sdpa_ops"))[:300]))
        for t in ("train_sdpa", "train_eager"):
            tr = task(o, t)
            ls = (tr.get("train") or {}).get("losses") or []
            impl = (tr.get("train") or {}).get("attn_impl")
            want = t.split("_")[1]
            ok = len(ls) == 20 and bool(impl) and all(i == want for i in impl if i)
            out.append((f"{name}: {t} ran 20 steps on {want}", ok,
                        str(tr.get("error") or f"{len(ls)} steps, attn {impl}, loader "
                            f"{(tr.get('train') or {}).get('loader')}, retry {tr.get('_retried_with_compile_disable')}")[:300]))
    return out


def table(obs: dict) -> str:
    rows = ["| state | guard | fwd top1 | fwd nll sdpa / eager | KL mean | wide-head ops | train sdpa (1 -> 20) | "
            "train eager (1 -> 20) | train default (attn) | gaps step1 / mean / final |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name)
        if not o:
            continue
        f = fwd(o)
        s, e, d = losses(o, "train_sdpa"), losses(o, "train_eager"), losses(o, "train_default")
        di = (task(o, "train_default").get("train") or {}).get("attn_impl")
        g = gaps(o) or {}
        rows.append(
            f"| {name} | {o.get('guard_installed')} | {f.get('top1_agree')} | {f.get('nll_sdpa')} / {f.get('nll_eager')} "
            f"| {f.get('kl_eager_sdpa_mean')} | {wide_ops(o)} | {s[:1]} -> {s[-1:]} | {e[:1]} -> {e[-1:]} "
            f"| {d[:1]} -> {d[-1:]} ({di}) | {g.get('step1', float('nan')):.4f} / {g.get('mean', float('nan')):.4f} / "
            f"{g.get('final', float('nan')):.4f} |")
    full = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for t in ("train_sdpa", "train_eager", "train_default"):
            ls = losses(o, t)
            if ls:
                full.append(f"- {name} {t}: {ls}")
    return "\n".join(rows) + "\n\nPer-step losses:\n" + "\n".join(full)
