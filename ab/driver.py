"""Cross-platform live A/B for unslothai/unsloth#17206: A = PR merged with main, B = A + ab/fix.patch.

Each scenario gets a fresh PDF-free venv (created by uv, no pip) and its own Studio home."""
import json, os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ["RUNNER_TEMP"]) / "pdf-ab"
MAIN, PR = os.environ["MAIN_SHA"], os.environ["PR_SHA"]
DEPS = "packaging filelock structlog fastapi pydantic pyjwt httpx python-multipart pyyaml huggingface_hub cryptography".split()
SCENARIOS = {
    "S1_cold_export": [["export", "child"]],
    "S2_cold_tool": [["tool", "child", "sandbox"]],
    "S3_cold_web": [["web"]],
    "S4_rag_then_new_process": [["rag"], ["export", "tool", "web", "child"]],
    "S5_downloads_disabled": [["export", "web", "tool"]],
    "S6_parallel_cold_exports": [["parallel", "child"]],
}


def sh(*cmd, cwd = None, env = None):
    print("+", " ".join(map(str, cmd)), flush = True)
    return subprocess.run(list(map(str, cmd)), cwd = cwd, env = env, check = True)


def tree(side):
    root = WORK / side
    if root.exists():
        return root
    if side == "A":
        sh("git", "init", "-q", root)
        # Keep LF so ab/fix.patch applies on Windows too.
        sh("git", "-C", root, "config", "core.autocrlf", "false")
        sh("git", "-C", root, "remote", "add", "origin", "https://github.com/unslothai/unsloth")
        sh("git", "-C", root, "sparse-checkout", "set", "studio/backend")
        sh("git", "-C", root, "fetch", "-q", "--filter=blob:none", "origin", MAIN, PR)
        sh("git", "-C", root, "checkout", "-q", MAIN)
        sh("git", "-C", root, "-c", "user.name=ab", "-c", "user.email=ab@example.invalid", "merge", "-q", "--no-edit", PR)
    else:
        shutil.copytree(tree("A"), root, ignore = shutil.ignore_patterns(".git"))
        sh("git", "apply", "--verbose", HERE / "fix.patch", cwd = root)
    return root


def python_of(venv):
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


results = {}
for side in ("A", "B"):
    backend = tree(side) / "studio" / "backend"
    for name, runs in SCENARIOS.items():
        venv, home = WORK / "venvs" / f"{side}-{name}", WORK / "homes" / f"{side}-{name}"
        home.mkdir(parents = True, exist_ok = True)
        sh("uv", "venv", "-q", "--python", sys.executable, venv)
        sh("uv", "pip", "install", "-q", "--python", python_of(venv), *DEPS)
        env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(home), PROBE_SAMPLE = str(HERE / "sample.pdf"), PYTHONDONTWRITEBYTECODE = "1")
        if name == "S5_downloads_disabled":
            env["UNSLOTH_PDF_AUTO_INSTALL"] = "0"
        lines = []
        for steps in runs:
            proc = subprocess.run([str(python_of(venv)), str(HERE / "probe.py"), *steps], cwd = backend, env = env, capture_output = True, text = True, encoding = "utf-8", errors = "replace")
            lines += [json.loads(line[7:]) for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
            if proc.returncode:
                lines.append({"check": "probe_crashed", "stderr": proc.stderr[-1500:]})
        results[f"{side}/{name}"] = lines
        print(f"=== {side} {name}", flush = True)
        for line in lines:
            print("   ", json.dumps(line)[:700], flush = True)

out = Path(os.environ.get("GITHUB_STEP_SUMMARY", WORK / "summary.md"))
with open(out, "a", encoding = "utf-8") as f:
    f.write(f"## {sys.platform} {sys.version.split()[0]}\n\n```\n")
    for key, lines in results.items():
        f.write(f"{key}\n")
        for line in lines:
            f.write(f"    {json.dumps(line)[:400]}\n")
    f.write("```\n")
