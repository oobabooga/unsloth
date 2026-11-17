"""Criteria: zoo ROCm gfx11 float32 gated delta (Qwen3.5 linear attention) leaves fla's faulting kernels.

base must show the defect: the fp32 op and the fp32 tiny Qwen3.5-MoE forward die (illegal access / abort / hang).
head is fixed when, on the same runner and job:
  - op_fp32 finishes fwd + bwd, every output and gradient within FP32_TOL of an fp64 recurrence,
    and it really took the torch path (torch_path_calls > 0, no fla chunk kernel launched);
  - model_fp32 finishes fwd + bwd and matches the never-injected pure-torch model (logits and every
    linear-attention gradient within FP32_TOL);
  - sft_fp32, sft_bf16 train 5 steps with finite losses; gen_fp32, gen_bf16 generate;
  - every tests/test_multi_lora_experts.py test passes;
  - bfloat16 is untouched: torch_path_calls == 0 in every bf16 cell, the bf16 op output and gradients are
    bit-identical to the base (same fla kernels, same configs), and the bf16 model logits digest too.
"""

from __future__ import annotations

MODE = "differential"
FP32_TOL = 1e-4
_CTX: dict = {}


def _cells(o):
    return o.get("cells") or {}


def _c(o, name):
    return _cells(o).get(name) or {}


def _done(c):
    return c.get("stage") == "done" and not c.get("crashed")


def _dead(c):
    return bool(c) and (c.get("crashed") or c.get("rc") == "timeout")


def _fp32_err_ok(c):
    err = c.get("err_vs_fp64") or {}
    return len(err) == 6 and all(v.get("finite") and v.get("max_abs", 1.0) <= FP32_TOL for v in err.values())


def _state_names(obs):
    return [k for k in obs if not k.startswith("_")]


def gates(obs):
    out = []
    for st in _state_names(obs):
        o = obs[st]
        files = {c.get("zoo_file") for c in _cells(o).values() if c.get("zoo_file")}
        ck = o.get("checkout") or ""
        out.append((f"{st} cells imported the state's unsloth_zoo", bool(files) and all(f and f.startswith(ck) for f in files),
                    ", ".join(sorted(str(f) for f in files))[:300]))
        hips = {str((c.get("versions") or {}).get("hip")) for c in _cells(o).values() if c.get("versions")}
        archs = {str((c.get("device") or {}).get("gcnArchName")) for c in _cells(o).values() if c.get("device")}
        out.append((f"{st} ran on HIP gfx1151", bool(hips) and "None" not in hips and any("gfx1151" in a for a in archs),
                    f"hip {sorted(hips)} arch {sorted(archs)}"))
        p = o.get("pytest") or {}
        out.append((f"{st} pytest collected {(p.get('collected') or ['?'])[0].split('::')[0]}", len(p.get("collected") or []) >= 5,
                    f"{len(p.get('collected') or [])} tests"))
    base = obs.get("base") or {}
    ctl = [n for n in ("op_bf16", "model_bf16") if _done(_c(base, n))]
    out.append(("base bfloat16 controls finished (the card and the fla kernels work at the base)", len(ctl) == 2, str(ctl)))
    head = obs.get("head") or {}
    for n in ("op_bf16", "model_bf16"):
        b, h = _c(base, n), _c(head, n)
        if n == "op_bf16":
            same = bool(b.get("o_digest")) and b.get("o_digest") == h.get("o_digest") and b.get("grad_digests") == h.get("grad_digests")
            ev = f"o {b.get('o_digest')} / {h.get('o_digest')}"
        else:
            same = bool(b.get("logits_digest")) and b.get("logits_digest") == h.get("logits_digest")
            ev = f"logits {b.get('logits_digest')} / {h.get('logits_digest')}"
        _CTX[f"{n}_identical"] = (same, ev)
    return out


def base_shows_defect(base):
    return _dead(_c(base, "op_fp32")) and _dead(_c(base, "model_fp32"))


def head_is_fixed(head):
    op = _c(head, "op_fp32")
    op_ok = (_done(op) and _fp32_err_ok(op) and (op.get("torch_path_calls") or 0) > 0
             and not any("gated_delta" in k for k in op.get("fla_kernels_launched") or []))
    cmp = head.get("model_fp32_vs_torchpath") or {}
    model_ok = (_done(_c(head, "model_fp32")) and _done(_c(head, "model_fp32_torchpath"))
                and (_c(head, "model_fp32").get("torch_path_calls") or 0) > 0
                and cmp.get("n_grads", 0) > 0 and cmp.get("logits_max_abs", 1.0) <= FP32_TOL
                and (cmp.get("grads_max_abs") if cmp.get("grads_max_abs") is not None else 1.0) <= FP32_TOL)
    train_ok = all(_done(_c(head, n)) and _c(head, n).get("losses") and
                   all(x == x and abs(x) != float("inf") for x in _c(head, n)["losses"]) for n in ("sft_fp32", "sft_bf16"))
    gen_ok = all(_done(_c(head, n)) for n in ("gen_fp32", "gen_bf16"))
    bf16_untouched = all((_c(head, n).get("torch_path_calls") or 0) == 0 for n in ("op_bf16", "model_bf16", "sft_bf16", "gen_bf16"))
    identical = all(_CTX.get(f"{n}_identical", (False, ""))[0] for n in ("op_bf16", "model_bf16"))
    tests = (head.get("pytest") or {}).get("tests") or {}
    tests_ok = bool(tests) and all(t.get("outcome") == "passed" for t in tests.values())
    _CTX["head_parts"] = {"op_fp32": op_ok, "model_fp32": model_ok, "sft": train_ok, "gen": gen_ok,
                          "bf16_no_torch_path": bf16_untouched, "bf16_bit_identical": identical, "multi_lora_tests": tests_ok}
    return op_ok and model_ok and train_ok and gen_ok and bf16_untouched and identical and tests_ok


def _fmt_err(c):
    err = c.get("err_vs_fp64") or {}
    return ", ".join(f"{k} {v.get('max_abs'):.2e}" for k, v in err.items()) if err else "-"


def table(obs):
    if "head" in obs:
        head_is_fixed(obs["head"])
    rows = ["| state | cell | rc | done | illegal access | torch-path calls | steps / losses | notes |", "|---|---|---|---|---|---|---|---|"]
    for st in _state_names(obs):
        for name, c in _cells(obs[st]).items():
            note = ""
            if name.startswith("op_"):
                note = _fmt_err(c)
            elif name.startswith("model_"):
                note = f"logits {c.get('logits_digest')}"
            elif name.startswith("gen_"):
                note = str(c.get("generated"))[:80]
            elif c.get("last_launch") and not _done(c):
                note = f"last launch {c['last_launch'].get('k')}"
            if not _done(c) and c.get("last_launch"):
                note += f" last launch {c['last_launch'].get('k')} {c['last_launch'].get('cfg')}"
            steps = c.get("losses") or c.get("steps")
            rows.append(f"| {st} | {name} | {c.get('rc')} | {_done(c)} | {c.get('illegal_access')} | {c.get('torch_path_calls')} | "
                        f"{str(steps)[:90]} | {note[:160]} |")
    rows.append("")
    rows.append("| state | model_fp32 vs pure-torch model | ")
    rows.append("|---|---|")
    for st in _state_names(obs):
        rows.append(f"| {st} | {obs[st].get('model_fp32_vs_torchpath')} |")
    rows.append("")
    rows.append("| state | test | outcome | rc | illegal access |")
    rows.append("|---|---|---|---|---|")
    for st in _state_names(obs):
        for tid, t in ((obs[st].get("pytest") or {}).get("tests") or {}).items():
            rows.append(f"| {st} | {tid.split('::', 1)[1]} | {t.get('outcome')} | {t.get('rc')} | {t.get('illegal_access')} |")
    rows.append("")
    for n in ("op_bf16", "model_bf16"):
        same, ev = _CTX.get(f"{n}_identical", (None, ""))
        rows.append(f"- {n} base vs head bit-identical: {same} ({ev})")
    if "head_parts" in _CTX:
        rows.append(f"- head parts: {_CTX['head_parts']}")
    return "\n".join(rows)
