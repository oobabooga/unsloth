"""Live A/B matrix for unslothai/unsloth#11237 against the real CDN / GitHub.

Usage: python live_matrix.py --src BEFORE_DIR --src AFTER_DIR --labels before,after
       --work WORK --pinned OLDER_RELEASE_TAG [--extra-args ...]

Scenarios are interleaved per ref so a release published mid-run hits both sides.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IS_WIN = platform.system() == "Windows"

INJECT = r'''
_inj_orig_vpa = validate_prebuilt_attempts
_inj_state = {"n": 0}
def validate_prebuilt_attempts(*a, **k):
    _inj_state["n"] += 1
    if _inj_state["n"] == 1:
        log("INJECTED: failing the first release's validation")
        raise PrebuiltFallback("injected failure for the first release")
    return _inj_orig_vpa(*a, **k)
'''


def injected_script(src: Path, out: Path) -> Path:
    text = (src / "studio" / "install_llama_prebuilt.py").read_text(encoding = "utf-8")
    marker = '\nif __name__ == "__main__":'
    assert text.count(marker) == 1, "main guard not unique"
    text = text.replace(marker, "\n" + INJECT + marker)
    out.write_text(text, encoding = "utf-8")
    return out


def run(cmd, env_extra, log_path: Path, reqlog: Path, timeout = 3600):
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = str(HERE / "sitecustom") + os.pathsep + env.get("PYTHONPATH", "")
    env["LIVE_REQLOG"] = str(reqlog)
    env["PYTHONUNBUFFERED"] = "1"
    if reqlog.exists():
        reqlog.unlink()
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            cmd, env = env, capture_output = True, text = True, timeout = timeout,
            encoding = "utf-8", errors = "replace",
        )
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = "timeout", str(e.stdout or ""), str(e.stderr or "")
    dt = time.monotonic() - t0
    log_path.write_text(
        f"$ {' '.join(map(str, cmd))}\nenv+={env_extra}\nrc={rc} wall={dt:.2f}s\n"
        f"--- stdout ---\n{out}\n--- stderr ---\n{err}\n",
        encoding = "utf-8",
    )
    reqs = []
    if reqlog.exists():
        reqs = [json.loads(l) for l in reqlog.read_text().splitlines() if l.strip()]
    hosts = {}
    for r in reqs:
        hosts[r["host"]] = hosts.get(r["host"], 0) + 1
    return {"rc": rc, "wall": round(dt, 2), "requests": len(reqs), "hosts": hosts,
            "stdout_tail": out.strip().splitlines()[-3:], "stderr_tail": err.strip().splitlines()[-4:],
            "_out": out, "_err": err}


def marker_summary(d: Path):
    m = d / "UNSLOTH_PREBUILT_INFO.json"
    if not m.exists():
        return None
    j = json.loads(m.read_text(encoding = "utf-8"))
    rf = j.get("runtime_files") or {}
    return {
        "release_tag": j.get("release_tag") or j.get("tag"),
        "llama_tag": j.get("llama_tag"),
        "asset": j.get("asset") or j.get("asset_name") or j.get("selected_asset"),
        "install_kind": j.get("install_kind"),
        "llama_backend": j.get("llama_backend"),
        "runtime_files": len(rf),
        "runtime_files_sha256": sum(1 for e in rf.values() if isinstance(e, dict) and e.get("sha256")),
        "has_host_profile": "host_profile" in j,
        "has_macos_load_probe": "macos_load_probe" in j,
        "keys": sorted(j),
    }


def server_version(d: Path):
    exe = "llama-server.exe" if IS_WIN else "llama-server"
    rt = d / "build" / "bin" / ("Release" if IS_WIN else "")
    cand = rt / exe
    if not cand.exists():
        return {"found": False}
    try:
        p = subprocess.run([str(cand), "--version"], capture_output = True, text = True, timeout = 120,
                           encoding = "utf-8", errors = "replace")
        txt = (p.stdout + p.stderr).strip().splitlines()
        return {"found": True, "rc": p.returncode, "tail": txt[-2:]}
    except Exception as e:
        return {"found": True, "error": str(e)}


def grep(res, *needles):
    blob = res["_out"] + res["_err"]
    return {n: (n in blob) for n in needles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action = "append", required = True)
    ap.add_argument("--labels", required = True)
    ap.add_argument("--work", required = True)
    ap.add_argument("--pinned", required = True)
    ap.add_argument("--extra-args", default = "")
    ap.add_argument("--only", default = "")
    a = ap.parse_args()
    labels = a.labels.split(",")
    srcs = [Path(s).resolve() for s in a.src]
    work = Path(a.work).resolve()
    work.mkdir(parents = True, exist_ok = True)
    extra = a.extra_args.split() if a.extra_args else []
    py = sys.executable
    results = {lab: {} for lab in labels}
    only = set(a.only.split(",")) if a.only else None

    def script(i):
        return [py, str(srcs[i] / "studio" / "install_llama_prebuilt.py")]

    def inj(i):
        p = injected_script(srcs[i], srcs[i] / "studio" / "_live_injected.py")
        # studio/ must still be importable for prebuilt_core / backend.*
        return [py, str(p)]

    def d(i, name):
        return work / labels[i] / name

    def rec(i, name, res, **more):
        res = dict(res)
        res.pop("_out", None)
        res.pop("_err", None)
        res.update(more)
        results[labels[i]][name] = res
        print(f"[{labels[i]}] {name}: rc={res['rc']} wall={res['wall']}s req={res['requests']} "
              f"{json.dumps({k: v for k, v in more.items() if k != 'marker'})[:300]}", flush = True)
        if "marker" in more and more["marker"]:
            mk = dict(more["marker"]); mk.pop("keys", None)
            print(f"    marker={json.dumps(mk)}", flush = True)
        (work / "results.json").write_text(json.dumps(results, indent = 1), encoding = "utf-8")

    def L(i, name):
        p = work / "logs" / labels[i]
        p.mkdir(parents = True, exist_ok = True)
        return p / f"{name}.log", p / f"{name}.req.jsonl"

    def want(name):
        return only is None or name in only

    for i in range(len(srcs)):
        (work / labels[i]).mkdir(parents = True, exist_ok = True)

    # Injected scripts must resolve studio/ from their own src; exec'ing a copy moves
    # __file__, so point _STUDIO_DIR at the real studio dir via PYTHONPATH instead.
    def inj_env(i):
        return {"PYTHONPATH_EXTRA": ""}

    scenarios = []

    def S(name):
        def deco(f):
            scenarios.append((name, f))
            return f
        return deco

    @S("resolve_install_tag")
    def _(i):
        r = run(script(i) + ["--resolve-install-tag", "latest", "--output-format", "json"] + extra, {}, *L(i, "resolve_install_tag"))
        rec(i, "resolve_install_tag", r, out = r["_out"].strip()[-400:])

    @S("resolve_prebuilt")
    def _(i):
        r = run(script(i) + ["--resolve-prebuilt", "latest", "--output-format", "json"] + extra, {}, *L(i, "resolve_prebuilt"))
        rec(i, "resolve_prebuilt", r, out = r["_out"].strip()[-600:])

    @S("resolve_prebuilt_api")
    def _(i):
        r = run(script(i) + ["--resolve-prebuilt", "latest", "--output-format", "json"] + extra,
                {"UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE": "1"}, *L(i, "resolve_prebuilt_api"))
        rec(i, "resolve_prebuilt_api", r, out = r["_out"].strip()[-600:])

    @S("resolve_backends")
    def _(i):
        r = run(script(i) + ["--resolve-backends", "latest", "--output-format", "json"] + extra, {}, *L(i, "resolve_backends"))
        rec(i, "resolve_backends", r, out = r["_out"].strip()[-2500:])

    @S("install_cdn")
    def _(i):
        dd = d(i, "cdn")
        r = run(script(i) + ["--install-dir", str(dd)] + extra, {}, *L(i, "install_cdn"))
        rec(i, "install_cdn", r, marker = marker_summary(dd), server = server_version(dd))

    @S("rerun_cdn")
    def _(i):
        dd = d(i, "cdn")
        r = run(script(i) + ["--install-dir", str(dd)] + extra, {}, *L(i, "rerun_cdn"))
        rec(i, "rerun_cdn", r, marker = marker_summary(dd),
            flags = grep(r, "already matches selected release", "extracting", "Downloading", "downloading"))

    @S("rerun_cdn_full_check")
    def _(i):
        dd = d(i, "cdn")
        r = run(script(i) + ["--install-dir", str(dd)] + extra, {"UNSLOTH_PREBUILT_FULL_CHECK": "1"}, *L(i, "rerun_cdn_full_check"))
        rec(i, "rerun_cdn_full_check", r, marker = marker_summary(dd),
            flags = grep(r, "already matches selected release", "extracting"))

    @S("check_installed")
    def _(i):
        dd = d(i, "cdn")
        r = run(script(i) + ["--check-installed", str(dd)] + extra, {}, *L(i, "check_installed"))
        rec(i, "check_installed", r)

    @S("resolve_backends_installed")
    def _(i):
        dd = d(i, "cdn")
        r = run(script(i) + ["--resolve-backends", "latest", "--install-dir", str(dd), "--output-format", "json"] + extra, {}, *L(i, "resolve_backends_installed"))
        rec(i, "resolve_backends_installed", r, out = r["_out"].strip()[-2500:])
        shutil.rmtree(dd, ignore_errors = True)

    @S("install_api")
    def _(i):
        dd = d(i, "api")
        r = run(script(i) + ["--install-dir", str(dd)] + extra, {"UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE": "1"}, *L(i, "install_api"))
        rec(i, "install_api", r, marker = marker_summary(dd), server = server_version(dd))

    @S("rerun_api")
    def _(i):
        dd = d(i, "api")
        r = run(script(i) + ["--install-dir", str(dd)] + extra, {"UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE": "1"}, *L(i, "rerun_api"))
        rec(i, "rerun_api", r, marker = marker_summary(dd), flags = grep(r, "already matches selected release"))
        shutil.rmtree(dd, ignore_errors = True)

    @S("walkback_api_injected")
    def _(i):
        dd = d(i, "wb_api")
        r = run(inj(i) + ["--install-dir", str(dd)] + extra,
                {"UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE": "1", "LIVE_STUDIO_DIR": str(srcs[i] / "studio")},
                *L(i, "walkback_api_injected"))
        rec(i, "walkback_api_injected", r, marker = marker_summary(dd), server = server_version(dd),
            flags = grep(r, "INJECTED", "trying the next older published prebuilt", "trying an older published prebuilt if one remains"))
        shutil.rmtree(dd, ignore_errors = True)

    @S("walkback_cdn_injected")
    def _(i):
        dd = d(i, "wb_cdn")
        r = run(inj(i) + ["--install-dir", str(dd)] + extra, {"LIVE_STUDIO_DIR": str(srcs[i] / "studio")},
                *L(i, "walkback_cdn_injected"))
        rec(i, "walkback_cdn_injected", r, marker = marker_summary(dd),
            flags = grep(r, "INJECTED", "prebuilt install failed"))
        shutil.rmtree(dd, ignore_errors = True)

    @S("upgrade_from_pinned")
    def _(i):
        dd = d(i, "upg")
        r1 = run(script(i) + ["--install-dir", str(dd), "--published-release-tag", a.pinned] + extra, {}, *L(i, "upgrade_pinned_install"))
        rec(i, "upgrade_pinned_install", r1, marker = marker_summary(dd))
        r2 = run(script(i) + ["--install-dir", str(dd)] + extra, {}, *L(i, "upgrade_to_latest"))
        rec(i, "upgrade_to_latest", r2, marker = marker_summary(dd), server = server_version(dd))
        shutil.rmtree(dd, ignore_errors = True)

    @S("backend_cpu")
    def _(i):
        dd = d(i, "cpu")
        r = run(script(i) + ["--install-dir", str(dd), "--llama-backend", "cpu"] + extra, {}, *L(i, "backend_cpu"))
        rec(i, "backend_cpu", r, marker = marker_summary(dd), server = server_version(dd))
        r2 = run(script(i) + ["--install-dir", str(dd), "--llama-backend", "cpu"] + extra, {}, *L(i, "backend_cpu_rerun"))
        rec(i, "backend_cpu_rerun", r2, flags = grep(r2, "already matches selected release"))
        shutil.rmtree(dd, ignore_errors = True)

    @S("staged_validation")
    def _(i):
        dd = d(i, "staged")
        r = run(script(i) + ["--install-dir", str(dd)] + extra, {"UNSLOTH_LLAMA_STAGED_VALIDATION": "1"}, *L(i, "staged_validation"))
        rec(i, "staged_validation", r, marker = marker_summary(dd), server = server_version(dd),
            flags = grep(r, "stories260K", "validation"))
        shutil.rmtree(dd, ignore_errors = True)

    for name, fn in scenarios:
        if not want(name):
            continue
        for i in range(len(srcs)):
            try:
                fn(i)
            except Exception as e:  # noqa: BLE001
                results[labels[i]][name] = {"harness_error": repr(e)}
                print(f"[{labels[i]}] {name}: HARNESS ERROR {e!r}", flush = True)
    (work / "results.json").write_text(json.dumps(results, indent = 1), encoding = "utf-8")
    print("DONE", flush = True)


if __name__ == "__main__":
    main()
