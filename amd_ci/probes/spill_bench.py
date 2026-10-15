#!/usr/bin/env python3
"""llama.cpp #268 vs Strata (strata-q36) on Qwen3.6-35B-A3B, experts spilled at a fixed GPU expert budget.

Observes only: writes one JSON row per run to --out (rows.jsonl) plus each run's stdout / stderr.
Both engines get the SAME prompt token ids (llama-tokenize of the same text): llama.cpp takes the text, Strata the
ids, and each row records the prompt token count the engine reported.  Arms are interleaved per rep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

L_PP = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
L_TG = re.compile(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs.*?([\d.]+) tokens per second")
S_PP = re.compile(r"prompt (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\), (\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)")
KEEP = re.compile(r"(moe_cache|MoE cache|init_device_lookup|materialize|load_mode =|model buffer size|large batches|"
                  r"hit rate|strata-q36|failed|error|graph reallocation|compute buffer size|KV buffer)", re.I)


def free_m() -> str:
    return subprocess.run(["free", "-m"], capture_output=True, text=True).stdout


def tokenize(tok_exe: str, model: str, path: Path) -> list[int]:
    p = subprocess.run([tok_exe, "-m", model, "-f", str(path), "--ids", "--log-disable"], capture_output=True, text=True)
    m = re.search(r"\[([\d,\s]*)\]", p.stdout)
    if not m:
        raise SystemExit(f"tokenize failed: {p.stdout[-500:]} {p.stderr[-1500:]}")
    return [int(x) for x in m.group(1).replace(",", " ").split()]


def make_prompts(tok_exe, model, corpus: str, out: Path, targets: dict[str, int]) -> dict:
    info = {}
    for name, n in targets.items():
        lo, hi = 0, len(corpus)
        cut = min(len(corpus), n * 4)
        ids = []
        for _ in range(12):   # bisect the character cut to land on n tokens (+-8)
            f = out / f"prompt_{name}.txt"
            f.write_text(corpus[:cut], encoding="utf-8")
            ids = tokenize(tok_exe, model, f)
            if abs(len(ids) - n) <= 8:
                break
            if len(ids) > n:
                hi = cut
            else:
                lo = cut
            cut = (lo + hi) // 2 if hi > lo + 1 else cut
        (out / f"prompt_{name}.ids").write_text(" ".join(map(str, ids)), encoding="utf-8")
        info[name] = {"chars": cut, "tokens": len(ids), "file": str(out / f"prompt_{name}.txt")}
        print(f"prompt {name}: {len(ids)} tokens, {cut} chars", flush=True)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-bin", required=True)
    ap.add_argument("--strata", default="")
    ap.add_argument("--llama-bin-head", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache-gib", type=float, default=16.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--arms", default="L_spill,L_spill_b,L_spill_ub512,L_full,S_spill,S_spill_b,S_full")
    ap.add_argument("--prompts", default="p2k:2048,p8k:8192")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs_dir = a.out / "runs"
    runs_dir.mkdir(exist_ok=True)
    lb = Path(a.llama_bin)
    targets = {k: int(v) for k, v in (x.split(":") for x in a.prompts.split(","))}
    corpus = Path(a.corpus).read_text(encoding="utf-8", errors="replace")
    pinfo = make_prompts(str(lb / "llama-tokenize"), a.model, corpus, a.out, targets)
    (a.out / "prompts.json").write_text(json.dumps(pinfo, indent=2), encoding="utf-8")
    mib = int(a.cache_gib * 1024)
    common_l = [str(lb / "llama-completion"), "-m", a.model, "-ngl", "99", "-fa", "on", "-c", "10240", "--temp", "0",
                "--seed", "42", "-lv", "4", "-no-cnv", "--no-display-prompt", "--load-mode", "none", "-n", str(a.n)]
    arms = {
        "L_spill": common_l + ["-cmoe", "--moe-cache-mib", str(mib), "-ub", "2048", "-b", "2048"],
        "L_spill_b": common_l + ["-cmoe", "--moe-cache-mib", str(mib), "-ub", "2048", "-b", "2048"],
        "L_spill_ub512": common_l + ["-cmoe", "--moe-cache-mib", str(mib)],
        "L_full": common_l,
        "S_spill": [a.strata, "--model", a.model, "--max-context", "10240", "--mtp", "0", "--expert-vram", str(a.cache_gib),
                    "--n", str(a.n)],
        "S_spill_b": [a.strata, "--model", a.model, "--max-context", "10240", "--mtp", "0", "--expert-vram",
                      str(a.cache_gib), "--n", str(a.n)],
        "S_full": [a.strata, "--model", a.model, "--max-context", "10240", "--mtp", "0", "--n", str(a.n)],
    }
    envs = {}
    if a.llama_bin_head:
        hb = str(Path(a.llama_bin_head) / "llama-completion")
        sp = ["-cmoe", "--moe-cache-mib", str(mib), "-ub", "2048", "-b", "2048"]
        arms["B_spill"] = common_l + sp
        arms["B_spill_b"] = common_l + sp
        arms["H_spill"] = [hb] + common_l[1:] + sp
        arms["H_spill_k"] = [hb] + common_l[1:] + sp
        envs["H_spill_k"] = {"GGML_CUDA_COPY_BATCH": "0"}  # kill switch: the old unbatched path
        arms["B_ub512"] = common_l + ["-cmoe", "--moe-cache-mib", str(mib)]
        arms["H_ub512"] = [hb] + common_l[1:] + ["-cmoe", "--moe-cache-mib", str(mib)]
    want = [x for x in a.arms.split(",") if x]
    rows = a.out / "rows.jsonl"
    for rep in range(a.reps):
        order = want if rep % 2 == 0 else list(reversed(want))
        for pname in targets:
            for arm in order:
                cmd = list(arms[arm])
                if not arm.startswith("S_"):
                    cmd += ["-f", pinfo[pname]["file"]]
                else:
                    cmd += ["--tokens", (a.out / f"prompt_{pname}.ids").read_text(encoding="utf-8")]
                before = free_m()
                t0 = time.time()
                try:
                    p = subprocess.run(cmd, capture_output=True, timeout=a.timeout, env={**os.environ, **envs.get(arm, {})})
                    rc, so, se = p.returncode, p.stdout, p.stderr
                except subprocess.TimeoutExpired as e:
                    rc, so, se = "timeout", e.stdout or b"", e.stderr or b""
                wall = time.time() - t0
                out_t, err_t = so.decode("utf-8", "replace"), se.decode("utf-8", "replace")
                stem = f"{arm}_{pname}_r{rep}"
                (runs_dir / f"{stem}.stdout.txt").write_text(out_t, encoding="utf-8")
                (runs_dir / f"{stem}.stderr.log").write_text(err_t, encoding="utf-8")
                row = {"env": envs.get(arm, {}), "arm": arm, "prompt": pname, "rep": rep, "rc": rc, "wall_s": round(wall, 2),
                       "cmd": [c if len(c) < 300 else c[:60] + "...(ids)" for c in cmd],
                       "md5": hashlib.md5(so).hexdigest(), "free_before": before, "free_after": free_m()}
                if not arm.startswith("S_"):
                    pp, tg = list(L_PP.finditer(err_t)), list(L_TG.finditer(err_t))
                    if pp:
                        row.update(prompt_tokens=int(pp[-1].group(2)), pp_tps=float(pp[-1].group(3)))
                    if tg:
                        row.update(tg_runs=int(tg[-1].group(2)), tg_tps=float(tg[-1].group(3)))
                else:
                    m = S_PP.search(err_t)
                    if m:
                        row.update(prompt_tokens=int(m.group(1)), pp_tps=float(m.group(3)), tg_runs=int(m.group(4)),
                                   tg_tps=float(m.group(6)))
                row["lines"] = [ln for ln in err_t.splitlines() if KEEP.search(ln)][-80:]
                with open(rows, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{stem}: rc={rc} pp={row.get('pp_tps')} ({row.get('prompt_tokens')} tok) tg={row.get('tg_tps')} "
                      f"md5={row['md5'][:8]} wall={wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
