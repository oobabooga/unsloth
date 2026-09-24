"""Clone main and the PR head, run each arm in a fresh process with empty MIOpen dbs."""
import json, os, subprocess, sys, time

ROOT = os.environ["WORK_ROOT"]
PR, MAIN = "c3ffacd4367c25d935966b16e0c9b6e02e3d070d", "df946e0864d2ccf05dfb00193fb79c80d7e48996"
# (arm, sha, db dir): pr_restart reuses pr's MIOpen dbs, i.e. a second Studio session.
ARMS = [("pr", PR, "pr"), ("pr_restart", PR, "pr"), ("main", MAIN, "main"), ("main_restart", MAIN, "main")]
PATTERNS = ("Searching the best solution", "SearchImpl", "exhaustive", "Compil", "Build", "FindSolution", "Perf Db", "find-db", "FindDb", "Tuning")
TIMEOUT = int(os.environ.get("ARM_TIMEOUT", "2400"))
summary = {}
for name, sha, dbname in ARMS:
    src = os.path.join(ROOT, "src_" + dbname)
    fresh = not os.path.isdir(src)
    os.makedirs(src, exist_ok = True)
    g = lambda *a: subprocess.run(["git", "-C", src, *a], check = True)
    if fresh:
      g("init", "-q"); g("remote", "add", "origin", "https://github.com/oobabooga/unsloth")
      g("sparse-checkout", "set", "studio/backend/core")
      g("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha); g("checkout", "-q", "FETCH_HEAD")
    env = dict(os.environ)
    for k in ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"):
        d = os.path.join(ROOT, "miopen_" + dbname, k.lower()); os.makedirs(d, exist_ok = True); env[k] = d
    env["MIOPEN_LOG_LEVEL"] = "5"
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
             miopen_error_lines = text.count("MIOpen(HIP): Error"),
             patterns = {p: text.count(p) for p in PATTERNS})
    import collections, re
    tmpl = collections.Counter(re.sub(r"[0-9]+", "N", l)[:140] for l in text.splitlines() if "MIOpen" in l)
    r["top_miopen"] = tmpl.most_common(12)
    summary[name] = r
    print("=====", name, json.dumps(r), flush = True)
    print("\n".join(text.splitlines()[-25:]), flush = True)
print("SUMMARY", json.dumps(summary, indent = 1))
