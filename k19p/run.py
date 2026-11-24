#!/usr/bin/env python3
# run.py: the gfx1151 checks of one OS after the trees are built (linux.sh / win.ps1).
#   ctest per tree (+ i3 with LLAMA_RS_REPLAY=1), test-backend-ops test -b ROCm0 (device vs CPU) per tree,
#   delta-net replay probes (test-gdn-replay, test-gdn-snap-rollback, test-rs-snap) at LLAMA_RS_REPLAY 0 / 1 / auto,
#   llama-server smoke on a tiny Qwen3 MoE GGUF (ROCm0 and CPU) per tree.
# usage: run.py --out DIR --tree NAME=BUILD_DIR[:BIN_DIR] ... [--ctest-replay i3] [--tbo i3,d6] [--replay i3,rp] [--smoke i3,d6]
# Every result is one JSON line in DIR/rows.jsonl; logs in DIR/logs.
import argparse, hashlib, json, os, re, subprocess, sys, time, urllib.request

WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--tree", action="append", default=[])
ap.add_argument("--ctest", default="")
ap.add_argument("--ctest-replay", default="")
ap.add_argument("--tbo", default="")
ap.add_argument("--replay", default="")
ap.add_argument("--smoke", default="")
ap.add_argument("--models", default="", help="dir of generated test models (default: <first replay tree build>/tests/test-models)")
ap.add_argument("--tinymoe", default="")
ap.add_argument("--ctest-j", default="8")
a = ap.parse_args()
OUT = a.out; LOG = os.path.join(OUT, "logs"); os.makedirs(LOG, exist_ok=True)
T = {}
for t in a.tree:
    n, v = t.split("=", 1)
    parts = v.split("|")
    T[n] = (os.path.abspath(parts[0]), os.path.abspath(parts[1] if len(parts) > 1 else os.path.join(parts[0], "bin")))
lst = lambda s: [x for x in s.split(",") if x and (x in T or print("SKIP (not built):", x))]
t_start = time.time()


def row(**kw):
    kw["t"] = round(time.time() - t_start)
    with open(os.path.join(OUT, "rows.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")
    print("ROW", json.dumps(kw)[:600], flush=True)


def run(cmd, log, env=None, cwd=None, timeout=3600):
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + " ".join(cmd) + "  env " + json.dumps(env or {}) + "\n")
        f.flush()
        try:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=e, cwd=cwd, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        except OSError as ex:
            f.write("OSError %s\n" % ex)
            rc = "oserror"
    return rc, round(time.time() - t0, 1), open(log, encoding="utf-8", errors="replace").read()


def ctest(name, env, tag):
    b, _ = T[name]
    log = os.path.join(LOG, "ctest_%s.log" % tag)
    cmd = ["ctest", "-C", "Release", "-j", a.ctest_j, "--timeout", "900", "--output-on-failure"]
    rc, s, txt = run(cmd, log, env=env, cwd=b, timeout=6000)
    failed_first = re.findall(r"^\s*\d+ - (\S+) \(([^)]+)\)", txt.split("The following tests FAILED:")[-1], re.M) if "The following tests FAILED:" in txt else []
    summ = re.findall(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", txt)
    rer = []
    if failed_first:
        log2 = os.path.join(LOG, "ctest_%s_rerun.log" % tag)
        rc2, s2, txt2 = run(["ctest", "-C", "Release", "--rerun-failed", "-j", "1", "--timeout", "2400", "--output-on-failure"], log2, env=env, cwd=b, timeout=9000)
        rer = re.findall(r"^\s*\d+ - (\S+) \(([^)]+)\)", txt2.split("The following tests FAILED:")[-1], re.M) if "The following tests FAILED:" in txt2 else []
    row(kind="ctest", tree=name, tag=tag, env=env, rc=rc, wall_s=s, summary=summ[-1] if summ else None,
        failed_first=failed_first, failed_after_rerun=rer if failed_first else [],
        tests_total=int(summ[-1][2]) if summ else None)


def tbo(name):
    _, d = T[name]
    log = os.path.join(LOG, "tbo_%s.log" % name)
    rc, s, txt = run([os.path.join(d, "test-backend-ops" + EXE), "test", "-b", "ROCm0"], log, cwd=d, timeout=7200)
    passed = re.findall(r"(\d+)/(\d+) tests passed", txt)
    fails = sorted(set(m.strip() for m in re.findall(r"^\s*(\S.*?): .*\bFAIL\b", txt, re.M)))
    row(kind="tbo", tree=name, rc=rc, wall_s=s, passed=passed[-1] if passed else None, n_fail=len(fails), fails=fails[:200],
        backends=re.findall(r"Backend \d+/\d+: \S+", txt), status=re.findall(r"\d+/\d+ backends passed", txt))


def rs_line(txt):
    return sorted(set(re.findall(r"delta-net rollback: [^\n]*", txt)))[:4]


def replay(name, models):
    _, d = T[name]
    exe = lambda x: os.path.join(d, x + EXE)
    modes = ["auto"] if name == "rp" else ["0", "1"]
    if name == "rp":
        modes = ["auto", "spill", "1"]
    log = os.path.join(LOG, "gdn_replay_%s.log" % name)
    if os.path.exists(exe("test-gdn-replay")):
        rc, s, txt = run([exe("test-gdn-replay")], log, cwd=d, timeout=1800)
        row(kind="replay", tree=name, test="test-gdn-replay", rc=rc, wall_s=s, tail=txt[-800:])
    for mode in modes:
        env = {} if mode == "auto" else {"LLAMA_RS_REPLAY": mode}
        for m, args in [("qwen35-dense", ["200", "2", "2"]), ("qwen35moe-moe", ["200", "2", "2"]), ("qwen4exp-moe", ["200", "2", "2"]),
                        ("qwen35-dense", ["600", "3", "4"])]:
            g = os.path.join(models, m + ".gguf")
            tag = "%s_%s_%s_%s" % (name, mode, m, args[0])
            if not os.path.exists(g):
                row(kind="replay", tree=name, test="test-gdn-snap-rollback", mode=mode, model=m, rc="no-model")
                continue
            rc, s, txt = run([exe("test-gdn-snap-rollback"), g] + args, os.path.join(LOG, "snaprb_%s.log" % tag), env=env, cwd=d, timeout=1800)
            row(kind="replay", tree=name, test="test-gdn-snap-rollback", mode=mode, model=m, args=args, rc=rc, wall_s=s,
                rollback=rs_line(txt), diff=re.findall(r"max[^\n]*diff[^\n]*", txt)[-2:], devices=sorted(set(re.findall(r"using device (\S+)|ROCm0 model buffer|offloaded \d+/\d+ layers", txt)))[:4])
        for m in ["qwen4exp-moe", "qwen35-dense", "qwen35moe-moe"]:
            g = os.path.join(models, m + ".gguf")
            if not os.path.exists(g):
                continue
            tag = "%s_%s_%s" % (name, mode, m)
            rc, s, txt = run([exe("test-rs-snap"), g], os.path.join(LOG, "rssnap_%s.log" % tag), env=env, cwd=d, timeout=1800)
            row(kind="replay", tree=name, test="test-rs-snap", mode=mode, model=m, rc=rc, wall_s=s, rollback=rs_line(txt), tail=txt[-400:])
        if os.path.exists(exe("test-llama-context")):
            tag = "%s_%s" % (name, mode)
            e2 = dict(env); e2["GGML_SCHED_DEBUG_REALLOC"] = "1"
            rc, s, txt = run([exe("test-llama-context"), "--models", models + os.sep], os.path.join(LOG, "llctx_%s.log" % tag), env=e2, cwd=d, timeout=3600)
            row(kind="replay", tree=name, test="test-llama-context", mode=mode, rc=rc, wall_s=s, rollback=rs_line(txt),
                fails=re.findall(r"[^\n]*(?:FAIL|abort|unexpected graph realloc)[^\n]*", txt)[:10], tail=txt[-600:])


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8] if os.path.exists(p) else None


def smoke(name, gguf, port=8097):
    _, d = T[name]
    srv = os.path.join(d, "llama-server" + EXE)
    for ngl in ["99", "0"]:
        tag = "%s_ngl%s" % (name, ngl)
        log = os.path.join(LOG, "server_%s.log" % tag)
        lf = open(log, "w", encoding="utf-8", errors="replace")
        p = subprocess.Popen([srv, "-m", gguf, "-ngl", ngl, "-c", "1024", "-t", "4", "--port", str(port), "--host", "127.0.0.1", "-np", "1"],
                             stdout=lf, stderr=subprocess.STDOUT, cwd=d)
        ok, txt, rc = False, None, None
        try:
            for _ in range(240):
                try:
                    urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2).read()
                    ok = True
                    break
                except Exception:
                    if p.poll() is not None:
                        break
                    time.sleep(1)
            if ok:
                outs = []
                for prompt in ["Once upon a time", "The capital of France is"]:
                    body = json.dumps({"prompt": prompt, "n_predict": 48, "temperature": 0, "seed": 1, "cache_prompt": False}).encode()
                    r = urllib.request.Request("http://127.0.0.1:%d/completion" % port, data=body, headers={"Content-Type": "application/json"})
                    j = json.loads(urllib.request.urlopen(r, timeout=300).read().decode("utf-8"))
                    outs.append(j.get("content", ""))
                txt = "\n---\n".join(outs)
                rc = 0
        except Exception as ex:
            rc = "err %s" % ex
        finally:
            p.kill()
            p.wait()
            lf.close()
        lt = open(log, encoding="utf-8", errors="replace").read()
        if txt is not None:
            open(os.path.join(LOG, "smoke_%s.txt" % tag), "w", encoding="utf-8").write(txt)
        row(kind="smoke", tree=name, ngl=ngl, ok=ok and rc == 0, rc=rc, md5=hashlib.md5(txt.encode()).hexdigest()[:8] if txt is not None else None,
            n_chars=len(txt) if txt else 0, text=(txt or "")[:200], gguf_md5=md5(gguf),
            device=re.findall(r"(?:ROCm0|CPU)[^\n]*model buffer size[^\n]*", lt)[:3], offload=re.findall(r"offloaded \d+/\d+ layers[^\n]*", lt)[:1])


for n in lst(a.ctest):
    ctest(n, {}, n)
for n in lst(a.ctest_replay):
    ctest(n, {"LLAMA_RS_REPLAY": "1"}, n + "_rs1")
for n in lst(a.tbo):
    tbo(n)
models = os.path.abspath(a.models) if a.models else ""
a.tinymoe = os.path.abspath(a.tinymoe) if a.tinymoe else ""
for n in lst(a.replay):
    if not models:
        models = os.path.join(T[n][0], "tests", "test-models")
    replay(n, models)
for n in lst(a.smoke):
    smoke(n, a.tinymoe)
row(kind="done")
