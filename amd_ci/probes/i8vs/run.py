#!/usr/bin/env python3
"""VM driver for one plan: pytests, the attention micro bench, then per family its arms (fresh process each, in
plan order), scoring, cache cleanup. Prints a base64 tarball of records, scores, logs and thumbnails at the end.

  python run.py --plan plan.json --root R --bench B --out OUT
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

I8_ENV = "UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"


def summarize(recp: Path) -> str:
    try:
        r = json.loads(recp.read_text())
    except Exception as exc:  # noqa: BLE001
        return f"no record ({exc})"
    st = r.get("status") or {}
    rows = [f"{x['i']}:{x['wall_s']:.2f}s d={x.get('denoise_s')} i8={x.get('int8_calls')} alloc={x.get('alloc_gib')} "
            f"smi={x.get('smi_gib')} rss={x.get('rss_gib')} rc={x.get('route_calls')}" for x in r.get("renders", [])]
    tf = r.get("tf_rows") or []
    rel = [lf["rel_l2"] for row in tf for lf in row["leaves"] if "rel_l2" in lf]
    head = (f"verdict={r.get('verdict')} load={r.get('load_s')} attn={st.get('attention_backend')} "
            f"quant={st.get('transformer_quant')} offload={st.get('offload_policy')} params={r.get('params')} "
            f"dispatch={json.dumps(r.get('dispatch_status_load'), default = str)[:600]} shapes={r.get('shapes')} "
            f"tf_probed={r.get('tf_probed')} tf_rel_l2_max={max(rel) if rel else None}")
    if r.get("error"):
        head += " ERROR " + r["error"][:800]
    return head + "\n    " + "\n    ".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required = True)
    ap.add_argument("--root", required = True)
    ap.add_argument("--bench", required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    R, B, OUT = Path(a.root), Path(a.bench), Path(a.out)
    OUT.mkdir(parents = True, exist_ok = True)
    (OUT / "_logs").mkdir(exist_ok = True)
    spy = plan.get("python") or (R / "studio_py.txt").read_text().strip()
    base_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", I8_ENV)}
    base_env.update({"HF_HOME": str(R / "hf"), "UNSLOTH_STUDIO_HOME": str(R / "studio_home"), "PYTHONUNBUFFERED": "1",
                     "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    results = {}
    for t in plan.get("pytests", []):
        root = R / "src" / t["tree"] / "studio" / "backend"
        tenv = {k: v for k, v in base_env.items() if k not in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")}
        tenv["PYTHONPATH"] = str(root)
        t0 = time.time()
        r = subprocess.run([spy, "-m", "pytest", "-q", "-o", "addopts=", "-m", t.get("marks", "gpu or not gpu"),
                            "-p", "no:cacheprovider", "-rs", "-v"] + t["tests"], cwd = str(root), capture_output = True,
                           text = True, env = tenv, timeout = 3600)
        name = t.get("name", t["tree"])
        (OUT / "_logs" / f"pytest_{name}.log").write_text(r.stdout + r.stderr)
        lines = (r.stdout + r.stderr).splitlines()
        keep = [l for l in lines if "PASSED" not in l] + [l for l in lines if "PASSED" in l and "dispatch" in l]
        print(f"=== PYTEST {name} rc={r.returncode} {round(time.time() - t0)}s\n" + "\n".join(keep)[-8000:], flush = True)
        results[f"pytest_{name}"] = r.returncode
    if plan.get("micro"):
        root = R / "src" / plan["micro"] / "studio" / "backend"
        t0 = time.time()
        r = subprocess.run([spy, "-u", str(B / "micro.py"), "--root", str(root), "--out", str(OUT / "micro.json")],
                           capture_output = True, text = True, env = base_env, timeout = 3600, cwd = str(R))
        (OUT / "_logs" / "micro.log").write_text(r.stdout + r.stderr)
        keep = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("ROW", "VERDICT", "MICRO", "Traceback"))
                or "Error" in l]
        print(f"=== MICRO rc={r.returncode} {round(time.time() - t0)}s\n" + "\n".join(keep)[-12000:], flush = True)
        results["micro"] = r.returncode
    for fam in plan.get("families", []):
        fname = fam["name"]
        ref_dir = R / "tf" / fname
        for arm in fam["arms"]:
            tag = f"{fname}_{arm['tag']}"
            if arm.get("only_if_same_backend"):
                # the bf16-swap floor's fallback: run only when the first swap kept the same backend as the native arm
                x, y = arm["only_if_same_backend"]
                try:
                    bx = (json.loads((OUT / f"{fname}_{x}" / "record.json").read_text()).get("status") or {}).get("attention_backend")
                    by = (json.loads((OUT / f"{fname}_{y}" / "record.json").read_text()).get("status") or {}).get("attention_backend")
                except Exception:  # noqa: BLE001
                    bx, by = None, None
                if bx != by:
                    print(f"=== {tag} skipped: {x} ran {bx}, {y} ran {by}", flush = True)
                    continue
            if arm.get("only_if_unrouted"):
                src = OUT / f"{fname}_{arm['only_if_unrouted']}" / "record.json"
                try:
                    routed = sum(x.get("int8_calls") or 0 for x in json.loads(src.read_text()).get("renders", []))
                except Exception:  # noqa: BLE001
                    routed = 0
                if routed:
                    print(f"=== {tag} skipped: {arm['only_if_unrouted']} ran {routed} INT8 calls", flush = True)
                    continue
            env = dict(base_env)
            env.update({k: str(v).replace("$ROOT", str(R)) for k, v in (arm.get("env") or {}).items()})
            env["TORCHINDUCTOR_CACHE_DIR"] = str(R / "ind" / tag)
            shutil.rmtree(R / "ind" / tag, ignore_errors = True)
            root = Path(plan["backend"]) if plan.get("backend") else R / "src" / (arm.get("tree") or "unsloth") / "studio" / "backend"
            load = {**(fam.get("load") or {}), **(arm.get("load") or {})}
            tf = arm.get("tf", "none")
            tf_ref = arm.get("tf_ref") or "native"
            cmd = [spy, "-u", str(B / "arm.py"), "--root", str(root), "--kind", fam["kind"], "--model", fam["model"],
                   "--tag", tag, "--out", str(OUT), "--load", json.dumps(load), "--gen", json.dumps(fam.get("gen") or {}),
                   "--prompts", str(arm.get("prompts", fam.get("prompts", 8))), "--warmup", str(arm.get("warmup", 1)),
                   "--tf", tf, "--tf-ref", str(ref_dir / tf_ref), "--tf-prompts", str(fam.get("tf_prompts", 8)),
                   "--profile-steps", str(arm.get("profile_steps", 0))]
            t = time.time()
            logp = OUT / "_logs" / f"{tag}.log"
            try:
                clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used",
                                      "--format=csv,noheader"], capture_output = True, text = True).stdout.strip()
            except Exception:  # noqa: BLE001 - ROCm / Windows: no nvidia-smi
                clk = "n/a"
            print(f"=== {tag} start {time.strftime('%H:%M:%S')} idle_gpu [{clk}] tree={root.parts[-3]} "
                  f"env={json.dumps(arm.get('env') or {})} load={json.dumps(load)} tf={tf}", flush = True)
            with open(logp, "w") as lf:
                p = subprocess.Popen(cmd, stdout = lf, stderr = subprocess.STDOUT, env = env, cwd = str(R))
                try:
                    rc = p.wait(timeout = arm.get("timeout", fam.get("arm_timeout", 5400)))
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                    rc = "TIMEOUT"
            results[tag] = rc
            print(f"=== {tag} rc={rc} {round(time.time() - t)}s :: {summarize(OUT / tag / 'record.json')}", flush = True)
            text = logp.read_text(errors = "replace").splitlines()
            keep = [l for l in text if "int8_attention" in l or "diffusion.attention" in l or "Traceback" in l]
            print("\n".join(("  | " + l[:500]) for l in keep[:30]), flush = True)
            if rc != 0:
                print("\n".join(("  ! " + l[:400]) for l in text[-50:]), flush = True)
        sc = subprocess.run([spy, str(B / "score.py"), "--out", str(OUT), "--family", fname, "--ref", fam.get("score_ref", "A1"),
                             "--pairs", ",".join(fam.get("pairs", [])), "--kind", fam["kind"]],
                            capture_output = True, text = True, env = base_env)
        print(f"=== SCORE {fname} rc={sc.returncode}\n{sc.stdout[-6000:]}\n{sc.stderr[-1500:]}", flush = True)
        # this family's records + scores now, in short lines, so a VM lost later still leaves them in the stream
        try:
            fb = io.BytesIO()
            with tarfile.open(fileobj = fb, mode = "w:gz") as ftf:
                for pat in (f"{fname}_*/record.json", f"scores_{fname}.json", f"{fname}_*/thumb_*.jpg"):
                    for p in glob.glob(str(OUT / pat)):
                        ftf.add(p, arcname = os.path.relpath(p, OUT))
            fs = base64.b64encode(fb.getvalue()).decode()
            print(f"FAMB64_BEGIN {fname} {len(fs)}", flush = True)
            for i in range(0, len(fs), 400):
                print(f"FAMB64 {fname} {fs[i:i + 400]}", flush = True)
            print(f"FAMB64_END {fname}", flush = True)
        except Exception as exc:  # noqa: BLE001
            print("FAMB64_FAILED", fname, repr(exc)[:300], flush = True)
        if fam.get("cleanup", True):
            shutil.rmtree(R / "hf" / "hub", ignore_errors = True)
            shutil.rmtree(ref_dir, ignore_errors = True)
            shutil.rmtree(R / "ind", ignore_errors = True)
            for media in glob.glob(str(OUT / f"{fname}_*" / "r*.npz")):
                os.remove(media)
            print("CLEANED", fname, shutil.disk_usage(str(R)), flush = True)
    print("ARMS_DONE", json.dumps(results), flush = True)
    buf = io.BytesIO()
    with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
        for pat in ("*/record.json", "micro.json", "scores_*.json", "*/thumb_*.jpg"):
            for p in glob.glob(str(OUT / pat)):
                tf.add(p, arcname = os.path.relpath(p, OUT))
        for p in glob.glob(str(OUT / "_logs" / "*.log")):
            lines = Path(p).read_text(errors = "replace").splitlines()
            if len(lines) > 4000:
                keys = ("int8", "attention", "Traceback", "Error", "RENDER", "LOADED", "PARAMS", "TF ", "quant", "offload")
                lines = lines[:400] + ["...[trimmed]..."] + [l for l in lines[400:-1500] if any(k in l for k in keys)][:3000] \
                    + ["...[tail]..."] + lines[-1500:]
            data_l = "\n".join(l[:2000] for l in lines).encode()
            ti = tarfile.TarInfo(os.path.relpath(p, OUT))
            ti.size = len(data_l)
            tf.addfile(ti, io.BytesIO(data_l))
    data = buf.getvalue()
    print("ARTIFACT_BYTES", len(data), flush = True)
    s = base64.b64encode(data).decode()
    print("ARTIFACT_B64_BEGIN")
    for i in range(0, len(s), 76 * 2000):
        print(s[i:i + 76 * 2000])
    print("ARTIFACT_B64_END", flush = True)


if __name__ == "__main__":
    sys.exit(main())
