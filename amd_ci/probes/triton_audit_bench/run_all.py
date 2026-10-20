"""Driver: every bench module x dtype x repeat, each a fresh process, module order rotated per repeat
(interleaved). Rows appended to --out (jsonl). Prints compact progress lines only.

python run_all.py --out rows.jsonl --reps 3 --dtypes bf16,fp16 [--modules b_a,b_b] [--quick] [--timeout 1200]
"""
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dtypes", default="bf16,fp16")
    ap.add_argument("--modules", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    mods = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(HERE, "b_*.py")))
    if a.modules:
        want = a.modules.split(",")
        mods = [m for m in mods if m in want]
    if a.skip:
        mods = [m for m in mods if m not in a.skip.split(",")]
    dtypes = a.dtypes.split(",")
    print("[run_all] modules=%s dtypes=%s reps=%d" % (mods, dtypes, a.reps), flush=True)
    t_all = time.time()
    with open(a.out, "a") as fo:
        for rep in range(a.reps):
            k = rep % max(len(mods), 1)
            order = mods[k:] + mods[:k]
            if rep % 2:
                order = order[::-1]
            for m in order:
                for dt in (dtypes if rep % 2 == 0 else dtypes[::-1]):
                    cmd = [a.python, "-u", os.path.join(HERE, m + ".py"), "--dtype", dt, "--rep", str(rep)]
                    if a.quick:
                        cmd.append("--quick")
                    t0 = time.time()
                    nrow = nerr = 0
                    status = "ok"
                    try:
                        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                        out, err, rc = p.stdout, p.stderr, p.returncode
                    except subprocess.TimeoutExpired as ex:
                        out = (ex.stdout or b"").decode() if isinstance(ex.stdout, bytes) else (ex.stdout or "")
                        err = "TIMEOUT"
                        rc = -9
                        status = "timeout"
                    for line in out.splitlines():
                        if line.startswith("@@ROW "):
                            try:
                                row = json.loads(line[6:])
                            except Exception:
                                continue
                            row["tag"] = a.tag
                            fo.write(json.dumps(row) + "\n")
                            nrow += 1
                            nerr += row.get("kind") == "error"
                    fo.flush()
                    if rc != 0:
                        status = "rc=%s" % rc
                        fo.write(json.dumps(dict(kind="proc_fail", bench=m, dtype=dt, rep=rep, rc=rc, tag=a.tag,
                                                 stderr=err[-3000:])) + "\n")
                    print("[run_all] rep=%d %s %s %s rows=%d errs=%d %.0fs (total %.0fs)" % (
                        rep, m, dt, status, nrow, nerr, time.time() - t0, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
