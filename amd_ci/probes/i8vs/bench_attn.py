#!/usr/bin/env python3
"""Studio INT8 attention vs SageAttention (every build that loads) vs torch SDPA, one batched run per machine.

Fair by construction:
  * kernels: every arm in the same process, on the same inputs ([B, S, H, D] strided views of one fused QKV buffer,
    Studio's probe inputs plus an H3-like sink set), interleaved rounds in alternating order, the same thermal /
    co-tenant gate before every cell (clocks and temperature recorded at both cell edges), the same fp32 reference,
    one card per fresh process (kbench.py --device i), every arm reported OK / NOT_RUN (with the reason) / ERROR in
    every cell, build provenance recorded (hub metadata, pip / source version + commit + nvcc)
  * multi-GPU (>= 2 cards): mgpu_check.py, the per-card accuracy gate the CFG replica / layer split rely on
  * models (optional): each family's arms D (Studio default) / S (Sage alone) / U (INT8 forced) as 3 interleaved fresh
    processes each, plus G (plain "sage" request), N (bf16 baseline) and the K / K2 bf16-swap noise floor (run.py +
    arm.py), LPIPS against D1 and N, kernel names from a profiled 2-step render

Outputs OUT/bench_attn.json (everything) and OUT/bench_attn.md (verdict grids + full tables).

  python bench_attn.py --root R --out OUT [--devices all|0,1] [--no-kernels] [--models klein,q21,wan5b]
                       [--model-gpu-ids 0,1 --model-env '{"UNSLOTH_DIFFUSION_CFG_PARALLEL": "0"}' --model-tag split]
R is the setup.sh root (R/src/unsloth = Studio tree, R/studio_py.txt = its python, R/ext = the Sage builds).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def sh(cmd, log: Path, env = None, timeout = None) -> int:
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, env = env, bufsize = 1)
        t0 = time.time()
        for line in p.stdout:
            fh.write(line)
            if line.startswith(("KB_ROW", "KB_VERDICT", "KB_BUILDS", "KB_DONE", "MGPU", "Traceback", "===", "SCORE", "FAMB64", "PROFILE", "LOADED")):
                print(line.rstrip()[:1500], flush = True)
            if timeout and time.time() - t0 > timeout:
                p.kill()
                break
        return p.wait()


def capture(fn, *args) -> str:
    buf = io.StringIO()
    old = sys.argv
    with contextlib.redirect_stdout(buf):
        try:
            sys.argv = ["x", *args]
            fn()
        except SystemExit:
            pass
        finally:
            sys.argv = old
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--devices", default = "all")
    ap.add_argument("--no-kernels", action = "store_true")
    ap.add_argument("--kb-args", default = "", help = "extra kbench.py args, e.g. '--tokens 1024,4096'")
    ap.add_argument("--mgpu", default = "auto", choices = ("auto", "on", "off"))
    ap.add_argument("--models", default = "")
    ap.add_argument("--model-gpu-ids", default = "")
    ap.add_argument("--model-env", default = "{}")
    ap.add_argument("--model-tag", default = "")
    ap.add_argument("--model-variants", default = "",
                    help = 'JSON list of {"tag", "gpu_ids", "env"}: every family once per variant (e.g. CFG replica and '
                           'layer split), all in one plan')
    ap.add_argument("--model-gen", default = "{}", help = "JSON overriding every family's generation params")
    ap.add_argument("--prompts", type = int, default = 2)
    ap.add_argument("--model-arms", default = "", help = "keep only these arm letters, e.g. D,U,N,K (default: all)")
    ap.add_argument("--backend", default = "", help = "Studio backend dir (default ROOT/src/unsloth/studio/backend)")
    ap.add_argument("--python", default = "", help = "Studio venv python (default ROOT/studio_py.txt, else this one)")
    a = ap.parse_args()
    R, OUT = Path(a.root), Path(a.out)
    OUT.mkdir(parents = True, exist_ok = True)
    (OUT / "_logs").mkdir(exist_ok = True)
    spy = a.python or ((R / "studio_py.txt").read_text().strip() if (R / "studio_py.txt").exists() else sys.executable)
    backend = Path(a.backend) if a.backend else R / "src" / "unsloth" / "studio" / "backend"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "UNSLOTH_STUDIO_HOME": str(R / "kb_home"), "HF_HOME": str(R / "hf")})
    probe = subprocess.run([spy, "-c", "import torch,json;print(json.dumps({'n':torch.cuda.device_count(),"
                            "'names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
                            "'torch':torch.__version__,'cuda':torch.version.cuda,'hip':getattr(torch.version,'hip',None)}))"],
                           capture_output = True, text = True, env = env)
    machine = json.loads(probe.stdout.strip().splitlines()[-1])
    try:
        machine["studio_commit"] = subprocess.run(["git", "-C", str(backend), "rev-parse", "HEAD"],
                                                  capture_output = True, text = True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    print("BENCH_MACHINE", json.dumps(machine), flush = True)
    res = {"machine": machine, "kernels": {}, "mgpu": None, "models": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not machine["n"]:
        res["error"] = f"NOT_RUN: torch sees no GPU (torch {machine['torch']}, cuda {machine['cuda']}, hip {machine['hip']})"
        (OUT / "bench_attn.json").write_text(json.dumps(res, indent = 1, default = str))
        (OUT / "bench_attn.md").write_text(f"# NOT_RUN\n\n{res['error']}\n")
        print("BENCH_DONE", res["error"], flush = True)
        return 3
    devs = list(range(machine["n"])) if a.devices == "all" else [int(x) for x in a.devices.split(",")]

    def save():
        (OUT / "bench_attn.json").write_text(json.dumps(res, indent = 1, default = str))

    # 1. kernel sweep, one card per fresh process, cards in turn (never two measurements at once)
    if not a.no_kernels:
        for d in devs:
            kj = OUT / f"kb_dev{d}.json"
            rc = sh([spy, "-u", str(HERE / "kbench.py"), "--studio", str(backend), "--ext", str(R / "ext"),
                     "--out", str(kj), "--device", str(d), *a.kb_args.split()], OUT / "_logs" / f"kbench_dev{d}.log",
                    env = env)
            res["kernels"][str(d)] = json.loads(kj.read_text()) if kj.exists() else {"error": f"kbench rc={rc}"}
            save()
    # 2. multi-GPU gate check
    if a.mgpu == "on" or (a.mgpu == "auto" and machine["n"] >= 2):
        mj = OUT / "mgpu.json"
        sh([spy, "-u", str(HERE / "mgpu_check.py"), "--studio", str(backend), "--out", str(mj)],
           OUT / "_logs" / "mgpu.log", env = env)
        res["mgpu"] = json.loads(mj.read_text()) if mj.exists() else {"error": "mgpu_check produced nothing"}
        save()
    # 3. models through Studio (run.py drives arm.py, one fresh process per arm, interleaved)
    if a.models:
        import mk_plans

        kb0 = next(iter(res["kernels"].values()), {}) if res["kernels"] else {}
        builds = (kb0.get("info") or {}).get("builds") or {}
        sage = "S" if (builds.get("sage_hub") or {}).get("ok") else ("SP" if (builds.get("sage2_pip") or {}).get("ok") else None)
        if not builds:  # no kernel pass on this run: ask the card directly
            sage = "S" if machine.get("hip") is None else None
        fams = []
        variants = json.loads(a.model_variants) if a.model_variants else [
            {"tag": a.model_tag, "gpu_ids": [int(x) for x in a.model_gpu_ids.split(",")] if a.model_gpu_ids else None,
             "env": json.loads(a.model_env)}]
        for var in variants:
            for name in a.models.split(","):
                f = mk_plans.fam(name, sage = sage, prompts = a.prompts)
                gen = json.loads(a.model_gen)
                if gen:
                    f["gen"] = {**f.get("gen", {}), **gen}
                for arm in f["arms"]:
                    if var.get("gpu_ids"):
                        arm["load"] = {**arm.get("load", {}), "gpu_ids": var["gpu_ids"]}
                    if var.get("load"):
                        arm["load"] = {**arm.get("load", {}), **var["load"]}
                    if var.get("env"):
                        arm["env"] = {**arm.get("env", {}), **var["env"]}
                if a.model_arms:
                    keep = set(a.model_arms.split(","))
                    f["arms"] = [x for x in f["arms"] if x["tag"].rstrip("0123456789") in keep]
                    tags = {x["tag"] for x in f["arms"]}
                    f["pairs"] = [pr for pr in f["pairs"] if set(pr.split(":")) <= tags]
                if var.get("tag"):
                    f["name"] = f"{name}{var['tag']}"
                fams.append(f)
        plan = {"run_id": "bench", "families": fams, "backend": str(backend), "python": spy}
        pj = OUT / "models_plan.json"
        pj.write_text(json.dumps(plan, indent = 1))
        mout = OUT / "models"
        rc = sh([spy, "-u", str(HERE / "run.py"), "--plan", str(pj), "--root", str(R), "--bench", str(HERE),
                 "--out", str(mout)], OUT / "_logs" / "models.log", env = {**env, "HF_TOKEN": os.environ.get("HF_TOKEN", "")}
                if os.environ.get("HF_TOKEN") else env)
        res["models"] = {"rc": rc, "sage_letter": sage, "records": {}, "scores": {}}
        for p in mout.glob("*/record.json"):
            r = json.loads(p.read_text())
            r.pop("traceback", None)
            res["models"]["records"][r["tag"]] = r
        for p in mout.glob("scores_*.json"):
            res["models"]["scores"][p.stem[7:]] = json.loads(p.read_text())
        save()
    # 3b. Studio's persisted verdicts (per card, with the ordinal each entry's gpu_uuid maps to here)
    pdj = OUT / "persisted_verdicts.json"
    sh([spy, "-u", str(HERE / "persist_dump.py"), "--out", str(pdj), str(R / "kb_home"), str(R / "studio_home")],
       OUT / "_logs" / "persist_dump.log", env = env)
    res["persisted"] = json.loads(pdj.read_text()) if pdj.exists() else {"error": "persist_dump produced nothing"}
    save()
    # 4. markdown
    import e2e_table
    import kb_summary
    import kb_table

    md = [f"# INT8 vs SageAttention: {', '.join(machine['names'])}\n",
          f"torch {machine['torch']} (cuda {machine['cuda']}, hip {machine['hip']}), Studio {machine.get('studio_commit')}\n"]
    kargs = [f"dev{d}={OUT / f'kb_dev{d}.json'}" for d in devs if (OUT / f"kb_dev{d}.json").exists()]
    if kargs:
        md.append("## Kernel verdict grids\n")
        md.append(capture(kb_summary.main, *kargs))
        md.append("## Kernel tables\n")
        md.append(capture(kb_table.main, *kargs))
        md.append("## NOT_RUN / ERROR arms (per device, first reason)\n")
        for d in devs:
            kb = res["kernels"].get(str(d)) or {}
            seen = {}
            for r in kb.get("rows", []):
                for arm, st in (r.get("status") or {}).items():
                    if st != "OK":
                        seen.setdefault(arm, st)
            md.append(f"- dev{d}: " + ("; ".join(f"`{k}` {v}" for k, v in seen.items()) or "none") + "\n")
        gates = [(d, r.get("gate", {})) for d in devs for r in (res["kernels"].get(str(d)) or {}).get("rows", [])]
        failed = [(d, g) for d, g in gates if g and not g.get("passed")]
        md.append(f"\nThermal / co-tenant gate: {len(gates) - len(failed)} of {len(gates)} cells passed"
                  + (f"; failed: {failed[:5]}" if failed else "") + "\n")
    if res["mgpu"]:
        m = res["mgpu"]
        md.append("## Multi-GPU INT8 gate (per-card accuracy check before the kernel runs on a card)\n")
        md.append(f"ok={m.get('ok')} both_cards_engaged={m.get('both_cards_engaged')} {m.get('why', '')}\n")
        md.append("| shape | card | before own verdict int8/native | verdict ok / impl | after int8/native | out device | cos | rel-L1 | pass |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in m.get("checks", []):
            b, f_ = r["before_verdict"], r["after_verdict"]
            md.append(f"| {r['shape']} | {r['card']} | {b['int8']}/{b['native']}{' (cached)' if b['verdict_already_cached'] else ''} | "
                      f"{r['verdict']['ok']} / {r['verdict']['impl']} | {f_['int8']}/{f_['native']} | {r['out_device']} | "
                      f"{r['acc']['cos']} | {r['acc']['rel_l1']} | {all(r['pass'].values())} |")
        md.append("")
    pers = res.get("persisted") or {}
    if pers.get("files"):
        md.append("## Persisted INT8 verdicts (Studio probe cache, per card)\n")
        md.append("| file | card ordinal | gpu | capability | decisions | ok / impl per key |")
        md.append("|---|---|---|---|---|---|")
        for f in pers["files"]:
            for e in f.get("entries", []):
                ks = "; ".join(f"{k}: {(v or {}).get('ok')}/{(v or {}).get('impl')}" if isinstance(v, dict) else f"{k}: {v}"
                               for k, v in list(e["decisions"].items())[:8])
                md.append(f"| {Path(f['path']).parent.parent.name}/{Path(f['path']).name} | {e['ordinal']} | {e['gpu_name']} | "
                          f"{e['capability']} | {e['n_decisions']} | {ks} |")
        md.append("")
    if a.models and (OUT / "models").exists():
        md.append("## Models end to end\n")
        md.append(capture(e2e_table.main, f"{machine['names'][0]}={OUT / 'models'}"))
    (OUT / "bench_attn.md").write_text("\n".join(md))
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    print("BENCH_DONE", OUT, flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
