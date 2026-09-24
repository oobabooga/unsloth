"""Scan shipped install.ps1 builds with the Bitdefender engine in Emsisoft's a2cmd, then
shrink the first detected sample to a 1-minimal set of lines that still trips it."""
import os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

WORK = Path(os.environ["RUNNER_TEMP"]) / "dc005"
HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
EEK_URL = "https://dl.emsisoft.com/EmsisoftEmergencyKit.exe"
DEADLINE = time.time() + float(os.environ.get("BUDGET_SECONDS", "4200"))
MODE = os.environ.get("MODE", "probe")


def log(*a):
    print(*a, flush=True)


def run(cmd, timeout=1800):
    log(">", " ".join(map(str, cmd)))
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")
    out = out.replace("\x00", "")
    log(f"  exit={p.returncode} in {time.time() - t:.1f}s")
    return p.returncode, out


def setup_eek():
    WORK.mkdir(parents=True, exist_ok=True)
    exe = WORK / "EEK.exe"
    if not exe.exists():
        log("downloading EEK")
        urllib.request.urlretrieve(EEK_URL, exe)
    dest = WORK / "EEK"
    code, out = run([str(exe), "-s2", f"-d{dest}"], timeout=900)
    log(out[-2000:])
    hits = list(dest.rglob("a2cmd.exe")) + list(Path("C:/EEK").rglob("a2cmd.exe"))
    hits = [h for h in hits if "bin64" in str(h).lower()] or hits
    if not hits:
        log("a2cmd.exe not found; tree:")
        for p in list(dest.rglob("*"))[:60]:
            log(" ", p)
        sys.exit(1)
    a2 = hits[0]
    log("a2cmd:", a2)
    code, out = run([str(a2), "/?"], timeout=120)
    log(out[:6000])
    code, out = run([str(a2), "/update"], timeout=2400)
    log(out[-4000:])
    return a2


def scan(a2, files, label, verbose=False):
    """files: dict name -> bytes. Returns {name: detection or None}."""
    d = WORK / "scan" / label
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    logf = WORK / "scan" / f"{label}.log"
    code, out = run([str(a2), f"/f={d}", "/pup", f"/l={logf}"], timeout=2400)
    text = out
    if logf.exists():
        raw = logf.read_bytes()
        enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
        text += "\n" + raw.decode(enc, "replace")
    if verbose:
        log(text[-8000:])
    res = {}
    lines = text.splitlines()
    for name in files:
        hit = None
        for line in lines:
            if name.lower() in line.lower() and re.search(r"detect|heur|trojan|\(B\)", line, re.I):
                m = re.search(r"detected:\s*(.+)$", line, re.I)
                hit = (m.group(1) if m else line).strip()
                break
        if hit is None and not (d / name).exists():
            hit = "VANISHED (removed by another scanner?)"
        res[name] = hit
    return res


def probe(a2):
    files = {p.name: p.read_bytes() for p in sorted(SAMPLES.glob("*.ps1"))}
    res = scan(a2, files, "probe", verbose=True)
    log("\n=== PROBE RESULTS ===")
    for name in sorted(res):
        log(f"{name:40s} {len(files[name]):8d}  {res[name] or 'clean'}")
    return files, res


def minimize(a2, name, data):
    nl = b"\r\n" if b"\r\n" in data else b"\n"
    cur = data.split(nl)
    join = lambda ls: nl.join(ls)
    n, rnd = 2, 0
    log(f"\n=== MINIMIZING {name}: {len(cur)} lines ===")
    while time.time() < DEADLINE:
        rnd += 1
        k = len(cur)
        n = min(n, k)
        bounds = [(i * k // n, (i + 1) * k // n) for i in range(n)]
        batch = {}
        for i, (a, b) in enumerate(bounds):
            batch[f"r{rnd:03d}_c{i:04d}.ps1"] = join(cur[:a] + cur[b:])
            if n > 2:
                batch[f"r{rnd:03d}_s{i:04d}.ps1"] = join(cur[a:b])
        res = scan(a2, batch, f"round{rnd:03d}")
        subsets = [i for i in range(n) if res.get(f"r{rnd:03d}_s{i:04d}.ps1")]
        removable = [i for i in range(n) if res.get(f"r{rnd:03d}_c{i:04d}.ps1")]
        log(f"round {rnd}: lines={k} n={n} subset_hits={len(subsets)} removable={len(removable)}")
        if subsets:
            a, b = bounds[subsets[0]]
            cur, n = cur[a:b], 2
            continue
        if removable:
            keep = [l for i, (a, b) in enumerate(bounds) if i not in removable for l in cur[a:b]]
            if len(removable) > 1:
                both = scan(a2, {"all.ps1": join(keep)}, f"round{rnd:03d}_all")
                if not both["all.ps1"]:
                    a, b = bounds[removable[0]]
                    keep = cur[:a] + cur[b:]
            cur = keep
            n = max(n - len(removable), 2)
            continue
        if n >= k:
            break
        n = min(2 * n, k)
    final = join(cur)
    check = scan(a2, {"final.ps1": final}, "final")
    log(f"\n=== MINIMAL ({len(cur)} lines, still detected: {check['final.ps1']}) ===")
    log(final.decode("utf-8", "replace"))
    (WORK / "minimal.ps1").write_bytes(final)


def main():
    a2 = setup_eek()
    files, res = probe(a2)
    if MODE != "probe+min":
        return
    order = ["rel-815-nosig.ps1", "rel-815.ps1"] + sorted(res)
    target = next((n for n in order if res.get(n) and not res[n].startswith("VANISHED")), None)
    if target is None:
        log("nothing detected; skipping minimization")
        return
    minimize(a2, target, files[target])


if __name__ == "__main__":
    main()
