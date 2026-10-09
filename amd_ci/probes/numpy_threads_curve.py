"""One fresh process per OPENBLAS_NUM_THREADS value: numpy import cost (threads, committed / reserved memory) and BLAS speed."""
import json, os, subprocess, sys

CHILD = r'''
import json, os, sys, time
import psutil
me = psutil.Process()
def snap():
    m = me.memory_info()
    return {"threads": me.num_threads(), "private_mb": round(getattr(m, "private", 0) / 2**20, 1),
            "vms_mb": round(m.vms / 2**20, 1), "rss_mb": round(m.rss / 2**20, 1)}
def med(f, n=5):
    f()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return round(sorted(ts)[n // 2] * 1e3, 2)
out = {"env": os.environ.get("OPENBLAS_NUM_THREADS"), "before": snap()}
import numpy as np
out["import"] = snap()
g = np.random.default_rng(0)
a = g.standard_normal((2048, 2048)); b = g.standard_normal((1024, 1024)); f32 = a.astype(np.float32)
x = g.standard_normal((20000, 768)); w = g.standard_normal((768, 768))
s = g.standard_normal((800, 800)); s = s @ s.T + 800 * np.eye(800)
out["mm2048_f64_ms"] = med(lambda: a @ a)
out["mm2048_f32_ms"] = med(lambda: f32 @ f32)
out["mm1024_f64_ms"] = med(lambda: b @ b)
out["proj_20000x768_ms"] = med(lambda: x @ w)
out["svd800_ms"] = med(lambda: np.linalg.svd(s), 3)
out["solve800_ms"] = med(lambda: np.linalg.solve(s, s[:, :64]))
out["after"] = snap()
print("CELL " + json.dumps(out), flush=True)
'''

def main():
    py = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    ncpu = os.cpu_count() or 1
    values = [v for v in (1, 2, 4, 6, 8, 12, 16, 24, 32) if v <= ncpu] + [None]
    rows = []
    for rep in range(3):
        for v in values:
            env = {k: x for k, x in os.environ.items() if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
            if v:
                env["OPENBLAS_NUM_THREADS"] = str(v)
            r = subprocess.run([py, "-c", CHILD], env=env, capture_output=True, text=True, timeout=900)
            cell = next((json.loads(l[5:]) for l in r.stdout.splitlines() if l.startswith("CELL ")), None)
            rows.append(dict(cell or {"env": v, "error": (r.stdout + r.stderr)[-800:]}, rep=rep))
            print(json.dumps(rows[-1])[:300], flush=True)
    import psutil
    print("RESULT " + json.dumps({"logical": ncpu, "physical": psutil.cpu_count(logical=False), "rows": rows}))

if __name__ == "__main__":
    main()
