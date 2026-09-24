"""Clone main and the PR head, run each arm in a fresh process with empty MIOpen dbs."""
import json, os, subprocess, sys, time

ROOT = os.environ["WORK_ROOT"]
ARMS = [("pr", "c3ffacd4367c25d935966b16e0c9b6e02e3d070d"), ("main", "df946e0864d2ccf05dfb00193fb79c80d7e48996")]
TIMEOUT = int(os.environ.get("ARM_TIMEOUT", "2400"))
summary = {}
for name, sha in ARMS:
    src = os.path.join(ROOT, "src_" + name)
    os.makedirs(src, exist_ok = True)
    g = lambda *a: subprocess.run(["git", "-C", src, *a], check = True)
    g("init", "-q"); g("remote", "add", "origin", "https://github.com/oobabooga/unsloth")
    g("sparse-checkout", "set", "studio/backend/core")
    g("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha); g("checkout", "-q", "FETCH_HEAD")
    env = dict(os.environ)
    for k in ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"):
        d = os.path.join(ROOT, "miopen_" + name, k.lower()); os.makedirs(d, exist_ok = True); env[k] = d
    log = os.path.join(ROOT, name + ".log"); out = os.path.join(ROOT, name + ".json")
    t = time.perf_counter()
    with open(log, "w") as fh:
        try:
            rc = subprocess.run([sys.executable, "probe.py", src, out], env = env, stdout = fh,
                                stderr = subprocess.STDOUT, timeout = TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    text = open(log, encoding = "utf-8", errors = "replace").read()
    r = json.load(open(out)) if os.path.exists(out) else {}
    r.update(rc = rc, wall_s = round(time.perf_counter() - t, 1),
             miopen_search_lines = text.count("Searching the best solution"),
             miopen_warning_lines = text.count("MIOpen(HIP): Warning"),
             miopen_error_lines = text.count("MIOpen(HIP): Error"))
    summary[name] = r
    print("=====", name, json.dumps(r), flush = True)
    print("\n".join(text.splitlines()[-25:]), flush = True)
print("SUMMARY", json.dumps(summary, indent = 1))
