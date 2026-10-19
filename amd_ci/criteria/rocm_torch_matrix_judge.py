"""Judge rocm_torch_matrix_driver.py observations. The control arm is the build main installs
today: if it fails a core cell, nothing about the candidates can be read (VOID, exit 1).
A candidate failing a cell the control passed is a REGRESSION; one passing all of the
control's cells is OK. Writes VERDICT.md and verdict.json beside observations.json."""

import json
import os
import sys

CORE = ["info", "matmul", "sdpa_default", "matmul_after_sdpa", "train_16bit"]
CELLS = CORE + ["sdpa_flash_attention", "sdpa_efficient_attention", "grouped_mm", "compile", "train_4bit"]

d = sys.argv[1]
with open(os.path.join(d, "observations.json"), encoding = "utf-8") as f:
    obs = json.load(f)
arms = obs["arms"]


def cells(a):
    return (a.get("obs") or {}).get("cells", {})


def ok(a, c):
    return bool(cells(a).get(c, {}).get("ok"))


def losses(a):
    return [s["loss"] for s in cells(a).get("train_16bit", {}).get("steps", [])]


ctrl = next((a for a in arms if a.get("control")), None)
lines = [f"# ROCm torch matrix on {obs['os']} gfx1151", ""]
verdicts = {}
if ctrl is None or not all(ok(ctrl, c) for c in CORE):
    overall = "VOID"
    why = "control arm missing" if ctrl is None else "control failed: " + ", ".join(c for c in CORE if not ok(ctrl, c))
else:
    overall = "DECIDED"
    why = ""
    cl = losses(ctrl)
    for a in arms:
        if a is ctrl:
            verdicts[a["arm"]] = "CONTROL"
            continue
        if not a.get("obs"):
            verdicts[a["arm"]] = "INSTALL_FAIL"
            continue
        lost = [c for c in CELLS if ok(ctrl, c) and not ok(a, c)]
        gained = [c for c in CELLS if ok(a, c) and not ok(ctrl, c)]
        al = losses(a)
        note = ""
        if cl and len(al) == len(cl):
            dl = max(abs(x - y) / max(abs(y), 1e-6) for x, y in zip(al, cl))
            note = f"max per-step loss rel diff vs control {dl:.3f}"
        verdicts[a["arm"]] = ("REGRESSION: " + ", ".join(lost) if lost else "OK") + (
            f" (gained: {', '.join(gained)})" if gained else "") + (f"; {note}" if note else "")

lines.append(f"**{overall}** {why}".rstrip())
lines += ["", "| arm | torch | hip | arch in wheel | " + " | ".join(CELLS) + " | verdict |",
          "|" + "---|" * (len(CELLS) + 5)]
for a in arms:
    i = (a.get("obs") or {}).get("info", {})
    arch = (i.get("gcnArchName") or "").split(":")[0]
    row = [a["arm"], i.get("torch", "-"), str(i.get("hip", "-")),
           ("yes" if arch in (i.get("arch_list") or []) else "NO") if i else "-"]
    for c in CELLS:
        x = cells(a).get(c)
        row.append("-" if x is None else ("ok" if x["ok"] else "FAIL"))
    row.append(verdicts.get(a["arm"], "-"))
    lines.append("| " + " | ".join(row) + " |")
lines += ["", "## Losses (16-bit LoRA, 10 steps) and failures", ""]
for a in arms:
    c = cells(a)
    t = c.get("train_16bit", {})
    lines.append(f"- **{a['arm']}**: losses {[round(x, 4) for x in losses(a)]}, "
                 f"grad_norm {[round(s['grad_norm'], 3) for s in t.get('steps', []) if s.get('grad_norm') is not None]}, "
                 f"step ms {[s['time_ms'] for s in t.get('steps', [])]}, gen {t.get('generate')!r}")
    for k, v in c.items():
        if not v.get("ok") and not k.endswith("_detail"):
            lines.append(f"  - {k}: {v.get('error', '')[:400]}")
    for k in ("train_16bit_detail", "train_4bit_detail"):
        if k in c:
            lines.append(f"  - {k}: {str(c[k].get('error'))[:400]}")
    if not a.get("obs"):
        lines.append("  - install failed: " + (a.get("install_tail") or "")[-800:].replace("\n", " | "))
lines += ["", "## Not tested here",
          "- Only gfx1151 (Strix Halo iGPU). gfx1103 / gfx1152 / gfx1153 / RDNA 2-4 dGPUs / CDNA not reachable.",
          "- No multi-GPU, no Studio installer path (venv-level install), no timing claims (shared pool)."]
with open(os.path.join(d, "VERDICT.md"), "w", encoding = "utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(d, "verdict.json"), "w", encoding = "utf-8") as f:
    json.dump({"verdict": overall, "why": why, "arms": verdicts}, f, indent = 1)
print("\n".join(lines))
sys.exit(0 if overall == "DECIDED" else 1)
