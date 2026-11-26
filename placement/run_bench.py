#!/usr/bin/env python3
"""Placement speed harness: one self-contained script for local B200 and Colab VMs (T4-HM, L4, A100, G4).

Runs llama-bench (pp/tg, -r N) or llama-server + a fixed decode request (server-only knobs: MTP/drafter,
--moe-cache-mib, ctx checkpoints, mmproj placement) for every config of a JSON points file, writing one JSONL
row per config as soon as it finishes (partial results survive a kill). Every row carries the hardware record,
the placement it asked for, bytes per decoded token on GPU / CPU (from the GGUF header), KV bytes, VRAM / RSS
peaks sampled over time, host load and the child's CPU time.

Stdlib only (huggingface_hub used when importable). Usage:
  python run_bench.py --points points.json --out results.jsonl --models-dir DIR [--bin-dir DIR]
                      [--threads N] [--cpus auto|none|0-15] [--keep-models] [--only id1,id2] [--rounds 2]

Points file:
  {"models": {"q9": {"repo": "unsloth/Qwen3.5-9B-MTP-GGUF", "file": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
                     "mmproj": "mmproj-F16.gguf", "drafter": null}},
   "defaults": {"p": 512, "n": 128, "r": 3, "fa": "on", "ctk": "f16", "ctv": "f16"},
   "configs": [
     {"id": "q9_f16_d8k", "model": "q9", "depth": 8192},                         # llama-bench
     {"id": "q9_cpu50", "model": "q9", "cpu_experts": {"frac": 0.5, "which": "all|updown|gate", "where": "first|last"}},
     {"id": "q9_ffn50", "model": "q9", "cpu_ffn": {"frac": 0.5}},               # dense FFN of a layer range on CPU
     {"id": "q9_ngl20", "model": "q9", "ngl": 20},
     {"id": "q9_nkvo", "model": "q9", "nkvo": 1},
     {"id": "q9_ot", "model": "q9", "ot": ["blk\\.(1|2)\\.ffn_.*=CPU"]},
     {"id": "q9_mtp", "model": "q9", "mode": "server", "ctx": 8192, "spec": "draft-mtp", "spec_n_max": 3},
     {"id": "q36_mc", "model": "q36", "mode": "server", "cpu_experts": {"frac": 1.0}, "moe_cache_mib": "auto"},
     {"id": "q9_mm_cpu", "model": "q9", "mode": "server", "mmproj": "cpu"}       # cpu | gpu | none
   ]}
"""
import argparse, json, os, platform, re, shutil, signal, socket, struct, subprocess, sys, tarfile, threading, time
import urllib.request

RELEASE_TAG = "b12676-mix-c32dcc8"
RELEASE_BASE = "https://github.com/unslothai/llama.cpp/releases/download/{tag}/"
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
EXE = ".exe" if IS_WIN else ""


def asset_name(tag, variant):
    """Studio's prebuilt asset for this OS / arch / variant (cuda*, rocm-gfx*, vulkan, cpu; metal = the macOS bundle)."""
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if variant == "metal" or IS_MAC:
        return f"llama-{tag}-bin-macos-{arch}.tar.gz"
    if IS_WIN:
        return f"app-{tag}-windows-{arch}-{variant}.zip"
    return f"app-{tag}-linux-{arch}-{variant}.tar.gz"


def getload():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):  # Windows
        return None
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32, "f8_e4m3": 1.0}
PROMPT = ("You are a careful senior engineer. Explain, step by step, how a hash table with open addressing works, "
          "including insertion, lookup, deletion with tombstones, load factor and resizing. Then write a complete, "
          "well commented Python implementation with a small test at the end.\n\n")


# prompt-dependent quantities (MTP acceptance, MoE hit rate) are calibrated on one set and checked on another
PROMPT_SETS = {
    "calib": [PROMPT,
              "Write a short story about a lighthouse keeper who finds a message in a bottle. Use vivid detail.\n\n",
              "List twenty practical tips for reducing household energy use, each with a one-sentence reason.\n\n"],
    "holdout": ["Solve step by step: a train leaves at 9:40 at 72 km/h, a second at 10:05 at 90 km/h on the same "
                "track. When and where does the second catch the first? Show all arithmetic.\n\n",
                "Translate into French, then explain three grammar choices you made: 'The committee postponed the "
                "vote because several members had not received the report in time.'\n\n",
                "Write a SQL schema for a library system (books, members, loans, holds) and five example queries "
                "with explanations.\n\n"],
    # G4 calibration (calib_card.py): 4 varied prompts (code, prose, list, reasoning), rotated per round
    "calib4": [PROMPT,
               "Write a short story about a lighthouse keeper who finds a message in a bottle. Use vivid detail.\n\n",
               "List twenty practical tips for reducing household energy use, each with a one-sentence reason.\n\n",
               "Solve step by step: a train leaves at 9:40 at 72 km/h, a second at 10:05 at 90 km/h on the same "
               "track. When and where does the second catch the first? Show all arithmetic.\n\n"],
}
# request sampling profiles: greedy, and Studio's T=1 defaults (inkling-2: T 1, top_p 1, top_k off, min_p 0;
# qwen: T 1, top_p 0.95, top_k 20, min_p 0)
SAMPLING = {"greedy": {"temperature": 0.0, "top_k": 1},
            "studio_t1": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0},
            "studio_t1_inkling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0}}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- GGUF header (stdlib)
_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_parse(path):
    """(kv, [(name, dims, n_bytes)]): sizes from gaps between data offsets (no ggml type table needed)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        buf = f.read(min(size, 96 << 20))
    p = 0

    def take(n):
        nonlocal p
        b = buf[p:p + n]; p += n
        return b

    u32 = lambda: struct.unpack("<I", take(4))[0]  # noqa: E731
    u64 = lambda: struct.unpack("<Q", take(8))[0]  # noqa: E731

    def value(t):
        if t == 8:
            return take(u64()).decode("utf-8", "replace")
        if t == 9:
            at = u32(); n = u64()
            if at == 8:
                for _ in range(n):
                    take(u64())
                return None
            raw = take(_SZ[at] * n)
            if n <= 4096:
                return list(struct.unpack("<" + _FMT[at][1] * n, raw))
            return None
        return struct.unpack(_FMT[t], take(_SZ[t]))[0]

    if take(4) != b"GGUF":
        raise ValueError(f"{path}: not a GGUF")
    u32(); nt = u64(); nkv = u64()
    kv = {}
    for _ in range(nkv):
        k = take(u64()).decode(); t = u32(); kv[k] = value(t)
    align = kv.get("general.alignment", 32)
    tensors = []
    for _ in range(nt):
        name = take(u64()).decode(); nd = u32(); dims = struct.unpack(f"<{nd}Q", take(8 * nd)); u32(); off = u64()
        tensors.append([name, dims, off])
    data_start = p + ((-p) % align)
    order = sorted(range(len(tensors)), key=lambda i: tensors[i][2])
    out = [None] * len(tensors)
    for a, i in enumerate(order):
        end = tensors[order[a + 1]][2] if a + 1 < len(order) else size - data_start
        out[i] = (tensors[i][0], tensors[i][1], end - tensors[i][2])
    return kv, out


class _Short(Exception):
    pass


def gguf_header_complete(buf):
    """True when buf holds the whole GGUF header (metadata + tensor infos); a sparse file reads zeros past the
    fetched bytes, so completeness must be checked on the fetched buffer itself."""
    p = 0

    def take(n):
        nonlocal p
        if p + n > len(buf):
            raise _Short()
        p += n
        return buf[p - n:p]

    u32 = lambda: struct.unpack("<I", take(4))[0]  # noqa: E731
    u64 = lambda: struct.unpack("<Q", take(8))[0]  # noqa: E731

    def value(t):
        if t == 8:
            take(u64())
        elif t == 9:
            at = u32(); n = u64()
            if at == 8:
                for _ in range(n):
                    take(u64())
            else:
                take(_SZ[at] * n)
        else:
            take(_SZ[t])
    try:
        if take(4) != b"GGUF":
            raise ValueError("not a GGUF")
        u32(); nt = u64(); nkv = u64()
        for _ in range(nkv):
            take(u64()); value(u32())
        for _ in range(nt):
            take(u64()); nd = u32(); take(8 * nd); u32(); u64()
        return True
    except _Short:
        return False


SHARD_RX = re.compile(r"-(\d{5})-of-(\d{5})\.gguf")


def shard_names(fname):
    """All shard names of a split GGUF (from shard 1's name), else [fname]."""
    m = SHARD_RX.search(fname)
    if not m:
        return [fname]
    n = int(m.group(2))
    return [fname[:m.start()] + f"-{k:05d}-of-{n:05d}.gguf" + fname[m.end():] for k in range(1, n + 1)]


def model_info(path):
    names = shard_names(path)
    if len(names) > 1:  # split GGUF: metadata from shard 1, tensors from every shard
        info = _model_info_one(names[0], extra=[t for nm in names[1:] for t in gguf_parse(nm)[1]])
        info["file_bytes"] = sum(os.path.getsize(nm) for nm in names)
        info["n_shards"] = len(names)
        return info
    return _model_info_one(path)


def _model_info_one(path, extra=()):
    kv, tensors = gguf_parse(path)
    tensors = list(tensors) + list(extra)
    arch = kv.get("general.architecture", "")
    g = lambda k, d=None: kv.get(f"{arch}.{k}", d)  # noqa: E731
    n_layer = int(g("block_count", 0))
    nextn = int(g("nextn_predict_layers", 0) or 0)
    n_expert = int(g("expert_count", 0) or 0)
    n_used = int(g("expert_used_count", 0) or 0)
    n_head = g("attention.head_count", 1)
    n_head_kv = g("attention.head_count_kv", n_head)
    emb = int(g("embedding_length", 0))
    hk = g("attention.key_length"); hv = g("attention.value_length")
    swa = g("attention.sliding_window")
    swa_pat = g("attention.sliding_window_pattern")
    attn_layers = sorted({int(n.split(".")[1]) for n, _, _ in tensors
                          if n.startswith("blk.") and re.search(r"\.attn_(k|kv_a_mqa|k_b)\.weight$", n)})
    kv_layers = []
    for il in attn_layers:
        if il >= n_layer - nextn:
            continue  # MTP / nextn layer: not in the decode path without speculation
        hkv = n_head_kv[il] if isinstance(n_head_kv, list) else n_head_kv
        hh = n_head[il] if isinstance(n_head, list) else n_head
        dk = hk or (emb // max(1, hh)); dv = hv or (emb // max(1, hh))
        is_swa = False
        if swa:
            if isinstance(swa_pat, list) and il < len(swa_pat):
                is_swa = bool(swa_pat[il])
            elif arch == "gpt-oss":
                is_swa = il % 2 == 0
        kv_layers.append({"layer": il, "k_el": int(hkv) * int(dk), "v_el": int(hkv) * int(dv),
                          "swa": int(swa) if is_swa else 0})
    return {"path": path, "arch": arch, "n_layer": n_layer, "nextn": nextn, "n_expert": n_expert,
            "n_used": n_used, "ctx_train": g("context_length"), "file_bytes": os.path.getsize(path),
            "tensors": tensors, "kv_layers": kv_layers, "n_vocab": None}


def kv_bytes(info, ctx, ctk, ctv):
    """Total KV buffer bytes at ctx and the bytes read per decoded token at that depth."""
    tot = 0.0
    for L in info["kv_layers"]:
        cells = min(ctx, L["swa"]) if L["swa"] else ctx
        tot += cells * (L["k_el"] * KV_BYTES[ctk] + L["v_el"] * KV_BYTES[ctv])
    return int(tot)


# ----------------------------------------------------------------------------- placement -> -ot / byte accounting
def layer_regex(layers):
    return "(" + "|".join(str(i) for i in sorted(layers)) + ")"


def build_overrides(cfg, info):
    """-ot patterns (llama.cpp applies the first match) for the abstract placement in cfg."""
    ot = list(cfg.get("ot", []))
    n = info["n_layer"] - info["nextn"]
    ce = cfg.get("cpu_experts")
    if ce:
        k = int(round(ce.get("frac", 1.0) * n))
        layers = range(n - k, n) if ce.get("where", "first") == "last" else range(0, k)
        which = {"all": "(gate|up|down|gate_up)", "updown": "(up|down)", "gate": "(gate)"}[ce.get("which", "all")]
        if k > 0:
            if ce.get("frac", 1.0) >= 1.0 and ce.get("which", "all") == "all":
                ot.append(r"\.ffn_" + which + r"_exps\.=CPU")
            else:
                ot.append(r"blk\." + layer_regex(layers) + r"\.ffn_" + which + r"_exps\.=CPU")
    cf = cfg.get("cpu_ffn")
    if cf:
        k = int(round(cf.get("frac", 0.5) * n))
        layers = range(n - k, n) if cf.get("where", "first") == "last" else range(0, k)
        if k > 0:
            ot.append(r"blk\." + layer_regex(layers) + r"\.ffn_(gate|up|down)\.=CPU")
    return ot


def placement_bytes(info, ot, ngl, spec=None):
    """Static bytes on GPU / CPU and the bytes read per decoded token on each side.

    Expert tensors count n_used / n_expert of their bytes per token, token_embd one row (stays on CPU in llama.cpp),
    nextn / MTP layers only when speculation is on. ngl counts layers from the end (llama.cpp); output on GPU
    when ngl > n_layer (or ngl < 0 = all)."""
    pats = []
    for o in ot:
        pat, _, buft = o.rpartition("=")
        pats.append((re.compile(pat), buft.upper()))
    n_layer = info["n_layer"]
    nextn = info["nextn"]
    ngl_eff = n_layer + 1 if ngl is None or ngl < 0 else ngl
    first_gpu = n_layer - min(ngl_eff, n_layer)
    r = {"w_gpu": 0, "w_cpu": 0, "tok_gpu": 0.0, "tok_cpu": 0.0, "exp_cpu_bytes": 0, "exp_cpu_tok": 0.0}
    frac = (info["n_used"] / info["n_expert"]) if info["n_expert"] else 1.0
    for name, dims, nb in info["tensors"]:
        m = re.match(r"blk\.(\d+)\.", name)
        il = int(m.group(1)) if m else None
        if name.startswith(("token_embd", "per_layer_token_embd")):
            dev = "cpu"  # input embeddings (incl. qwen4exp / gemma engram tables, read lazily) stay on the host
        elif il is None:
            dev = "gpu" if ngl_eff > n_layer else "cpu"
        else:
            dev = "gpu" if il >= first_gpu else "cpu"
        for rx, buft in pats:
            if rx.search(name):
                dev = "cpu" if buft.startswith("CPU") else "gpu"
                break
        r["w_" + dev] += nb
        if name.startswith(("token_embd", "per_layer_token_embd")):
            per_tok = nb / max(1, dims[-1] if dims else 1)
        elif il is not None and il >= n_layer - nextn and not spec:
            per_tok = 0.0
        elif "_exps" in name:
            per_tok = nb * frac
        else:
            per_tok = float(nb)
        r["tok_" + dev] += per_tok
        if "_exps" in name and dev == "cpu":
            r["exp_cpu_bytes"] += nb
            r["exp_cpu_tok"] += per_tok
    return r


# ----------------------------------------------------------------------------- hardware
def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def meminfo():
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
        return out
    except OSError:
        pass
    if IS_MAC:  # MemTotal from hw.memsize, MemAvailable ~ free + inactive + speculative pages
        try:
            out["MemTotal"] = int(sh(["sysctl", "-n", "hw.memsize"]).strip())
            vm = sh(["vm_stat"], 5)
            pg = int(re.search(r"page size of (\d+)", vm).group(1))
            pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative|purgeable):\s+(\d+)", vm)}
            out["MemAvailable"] = pg * sum(pages.values())
        except Exception:
            pass
    elif IS_WIN:
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            out["MemTotal"], out["MemAvailable"] = m.ullTotalPhys, m.ullAvailPhys
        except Exception:
            pass
    return out


def gpu_index():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = cvd.split(",")[0].strip() if cvd else "0"
    return first or "0"


def cpu_model_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if IS_MAC:
        return sh(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    if IS_WIN:
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            return platform.processor()
    return platform.processor()


def amd_gpu_sysfs():
    """{name, card, vram_total, gtt_total} of the first amdgpu card (Linux sysfs), else None."""
    import glob
    for d in sorted(glob.glob("/sys/class/drm/card*/device")):
        try:
            if open(os.path.join(d, "vendor")).read().strip() != "0x1002":
                continue
            r = {"card": d}
            for k in ("mem_info_vram_total", "mem_info_gtt_total", "mem_info_vis_vram_total"):
                try:
                    r[k] = int(open(os.path.join(d, k)).read())
                except OSError:
                    pass
            if "mem_info_vram_total" in r:
                return r
        except OSError:
            continue
    return None


def host_probe(models_dir, mib=1024):
    """VM host record for the speed model: CPU (model, cores, flags of interest), RAM total and single-thread copy
    bandwidth, disk sequential write / cold read (O_DIRECT where the OS has it), PCIe link of the GPU."""
    out = {"cpu_model": cpu_model_name(), "nproc": os.cpu_count(), "os": platform.platform(), "machine": platform.machine()}
    try:
        flags = next((ln.split(":", 1)[1].split() for ln in open("/proc/cpuinfo") if ln.startswith("flags")), [])
        out["cpu_flags"] = sorted(f for f in flags if f in ("avx2", "avx512f", "avx512_bf16", "avx512_vnni", "amx_tile",
                                                            "amx_bf16", "amx_int8", "avx_vnni", "f16c", "fma"))
        out["cpu_mhz"] = next((float(ln.split(":", 1)[1]) for ln in open("/proc/cpuinfo") if ln.startswith("cpu MHz")), None)
        cores = {(ln.split(":", 1)[1].strip()) for ln in open("/proc/cpuinfo") if ln.startswith("core id")}
        socks = {(ln.split(":", 1)[1].strip()) for ln in open("/proc/cpuinfo") if ln.startswith("physical id")}
        out["phys_cores"] = len(cores) * max(1, len(socks)) if cores else None
        out["lscpu"] = sh(["lscpu"], 10)[-1500:]
    except Exception:
        pass
    mi = meminfo()
    out["ram_total"], out["ram_avail"] = mi.get("MemTotal"), mi.get("MemAvailable")
    try:  # single-thread memcpy bandwidth (a lower bound on what one core streams)
        n = 256 << 20
        a = bytearray(n); b = bytearray(n)
        ts = []
        for _ in range(4):
            t = time.perf_counter(); b[:] = a; ts.append(time.perf_counter() - t)
        out["memcpy_1t_GBs"] = round(2 * n / min(ts[1:]) / 1e9, 2)
        del a, b
    except Exception as e:
        out["memcpy_1t_GBs"] = f"err {e}"
    fn = os.path.join(models_dir, ".disk_probe.bin")
    try:  # sequential write then cold read (O_DIRECT on Linux bypasses the page cache)
        chunk = os.urandom(16 << 20)
        t = time.perf_counter()
        with open(fn, "wb") as f:
            for _ in range(mib // 16):
                f.write(chunk)
            f.flush(); os.fsync(f.fileno())
        out["disk_write_MBs"] = round(mib / (time.perf_counter() - t), 1)
        if sys.platform.startswith("linux"):
            r = subprocess.run(["dd", f"if={fn}", "of=/dev/null", "bs=16M", "iflag=direct"], capture_output=True, text=True, timeout=300)
            m = re.search(r"([\d.]+) s, ([\d.]+) ([GMk]B)/s", r.stderr)
            if m:
                out["disk_read_direct_MBs"] = round(float(m.group(2)) * {"GB": 1000, "MB": 1, "kB": 0.001}[m.group(3)], 1)
        out["disk_dev"] = sh(["df", "-h", models_dir], 10).strip().splitlines()[-1:] if not IS_WIN else None
        out["disk_rotational"] = sh("lsblk -d -o NAME,ROTA,MODEL,SIZE 2>/dev/null", 10)[-600:] if sys.platform.startswith("linux") else None
    except Exception as e:
        out["disk_err"] = str(e)[:200]
    finally:
        try:
            os.remove(fn)
        except OSError:
            pass
    q = "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    out["pcie"] = sh(["nvidia-smi", "-i", gpu_index(), f"--query-gpu={q}", "--format=csv,noheader"]).strip() or None
    amd = amd_gpu_sysfs()
    if amd:
        out["amd_sysfs"] = amd
        for k in ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width"):
            try:
                out.setdefault("pcie_amd", {})[k] = open(os.path.join(amd["card"], k)).read().strip()
            except OSError:
                pass
    if IS_MAC:
        out["mac"] = {"model": sh(["sysctl", "-n", "hw.model"]).strip(),
                      "iogpu_wired_limit_mb": sh(["sysctl", "-n", "iogpu.wired_limit_mb"]).strip(),
                      "perf_cores": sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip()}
    return out


def hardware(models_dir):
    q = "name,memory.total,memory.used,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,compute_cap,driver_version,clocks.max.sm,clocks.max.mem"
    idx = gpu_index()
    row = sh(["nvidia-smi", "-i", idx, f"--query-gpu={q}", "--format=csv,noheader,nounits"]).strip().split(", ")
    gpu = dict(zip(q.split(","), row)) if len(row) == len(q.split(",")) else {"raw": row}
    smi = sh(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
    gpu["cuda_driver"] = m.group(1) if m else None
    if not gpu.get("name"):  # AMD / Apple / no GPU: name from sysfs / the OS; real device list from llama.cpp later
        amd = amd_gpu_sysfs()
        if amd:
            gpu["name"] = (sh("cat /sys/class/drm/card*/device/product_name 2>/dev/null", 5).strip()
                           or sh(["rocminfo"], 20).split("Marketing Name:")[-1].splitlines()[0].strip() if amd else "")
            gpu["name"] = gpu["name"] or "AMD GPU"
            gpu["memory.total"] = (amd.get("mem_info_vram_total", 0) + amd.get("mem_info_gtt_total", 0)) >> 20
        elif IS_MAC:
            gpu["name"] = "Apple " + sh(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        elif IS_WIN:
            gpu["name"] = sh("powershell -NoProfile -Command \"(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name\"", 20).strip()
    numa = None
    if gpu.get("pci.bus_id"):
        bus = gpu["pci.bus_id"].lower()[-12:]
        try:
            numa = int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read())
        except Exception:
            pass
    mi = meminfo()
    du = shutil.disk_usage(models_dir)
    return {"host": socket.gethostname(), "gpu_index": idx, "gpu": gpu, "gpu_numa": numa,
            "ram_total": mi.get("MemTotal"), "ram_avail": mi.get("MemAvailable"), "cpu_model": cpu_model_name(),
            "nproc": os.cpu_count(), "numa_nodes": numa_nodes(), "disk_free": du.free, "disk_total": du.total,
            "kernel": platform.release(), "os": sys.platform, "load1": getload(), "colab": os.path.exists("/content")}


def numa_nodes():
    nodes = {}
    base = "/sys/devices/system/node"
    if os.path.isdir(base):
        for d in os.listdir(base):
            if re.fullmatch(r"node\d+", d):
                nodes[int(d[4:])] = parse_cpulist(open(f"{base}/{d}/cpulist").read().strip())
    return nodes


def parse_cpulist(s):
    out = []
    for part in s.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cpu_busy(cpus, dt=1.0):
    """Per-CPU busy fraction over dt seconds from /proc/stat."""
    def snap():
        s = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split(); c = int(f[0][3:]); v = list(map(int, f[1:]))
                s[c] = (sum(v), v[3] + v[4])
        return s
    a = snap(); time.sleep(dt); b = snap()
    return {c: 1.0 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in cpus if c in a and c in b}


def pick_cpus(spec, threads, hw):
    """auto: physical cores on the GPU's NUMA node, the `threads` least busy over 1 s; none: no pinning."""
    if spec in (None, "", "none"):
        return None, None
    if spec != "auto":
        return parse_cpulist(spec), None
    nodes = hw["numa_nodes"]
    if len(nodes) <= 1:
        return None, None  # one node (Colab): let llama.cpp use every core
    def phys_of(cpus):
        out = []
        for c in cpus:
            try:
                sib = parse_cpulist(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip())
            except Exception:
                sib = [c]
            if c == min(sib):
                out.append(c)
        return out
    # prefer the GPU's node, but move to another node when the GPU's node is busy and another is idle:
    # on a shared host idle cores matter more than the cross-socket hop
    allb = cpu_busy(sorted({c for v in nodes.values() for c in v}), 1.0)
    nb = {n: sum(allb.get(c, 1) for c in v) / max(1, len(v)) for n, v in nodes.items()}
    home = hw["gpu_numa"] if hw["gpu_numa"] in nodes else min(nodes)
    best = min(nb, key=nb.get)
    node = best if nb[home] > 0.3 and nb[best] < 0.15 else home
    hw["_cpu_node_choice"] = {"gpu_node": home, "used_node": node, "node_busy": {k: round(v, 2) for k, v in nb.items()}}
    cand = nodes[node]
    phys = []
    for c in cand:
        try:
            sib = parse_cpulist(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip())
        except Exception:
            sib = [c]
        if c == min(sib):
            phys.append(c)
    busy = cpu_busy(phys)
    chosen = sorted(phys, key=lambda c: busy.get(c, 1.0))[:threads]
    return sorted(chosen), {c: round(busy.get(c, 0), 3) for c in chosen}


# ----------------------------------------------------------------------------- sampler
class _Nvml:
    """Device memory used via NVML (ctypes, no nvidia-smi process per sample); None when unavailable."""

    def __init__(self, idx):
        import ctypes
        self.h = None
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() != 0:
                return
            h = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(int(idx)), ctypes.byref(h)) == 0:
                self.h = h

            class Mem(ctypes.Structure):
                _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
            self.mem = Mem()
            self.ct = ctypes
        except Exception:
            self.h = None

    def used_mib(self):
        if self.h is None:
            return None
        if self.lib.nvmlDeviceGetMemoryInfo(self.h, self.ct.byref(self.mem)) != 0:
            return None
        return self.mem.used >> 20

    def mem_mib(self):
        """(total, free, used) MiB of the device, or None."""
        if self.h is None or self.lib.nvmlDeviceGetMemoryInfo(self.h, self.ct.byref(self.mem)) != 0:
            return None
        return self.mem.total >> 20, self.mem.free >> 20, self.mem.used >> 20

    def procs_mib(self):
        """{pid: used MiB} of compute processes on the device (query-compute-apps), or None."""
        if self.h is None:
            return None
        ct = self.ct
        if not hasattr(self, "_pfn"):
            class P4(ct.Structure):
                _fields_ = [("pid", ct.c_uint), ("used", ct.c_ulonglong), ("gi", ct.c_uint), ("ci", ct.c_uint)]

            class P2(ct.Structure):
                _fields_ = [("pid", ct.c_uint), ("used", ct.c_ulonglong)]
            self._pfn = None
            for name, st in (("nvmlDeviceGetComputeRunningProcesses_v3", P4),
                             ("nvmlDeviceGetComputeRunningProcesses_v2", P4),
                             ("nvmlDeviceGetComputeRunningProcesses", P2)):
                try:
                    self._pfn = (getattr(self.lib, name), st); break
                except AttributeError:
                    continue
        if not self._pfn:
            return None
        fn, st = self._pfn
        n = ct.c_uint(64); arr = (st * 64)()
        if fn(self.h, ct.byref(n), arr) != 0:
            return None
        # usedGpuMemory = NVML_VALUE_NOT_AVAILABLE (ULL max) under some containers / WDDM
        return {int(arr[i].pid): (None if arr[i].used >= (1 << 63) else int(arr[i].used >> 20)) for i in range(n.value)}


class _AmdSysfs:
    """amdgpu sysfs memory counters (Linux). Unified-memory APUs (gfx1151) allocate from the VRAM carve-out and GTT
    (system RAM mapped for the GPU): used / total are VRAM + GTT, and both are also recorded apart."""

    def __init__(self):
        self.info = amd_gpu_sysfs()
        self.h = self.info

    def _rd(self, k):
        try:
            return int(open(os.path.join(self.info["card"], k)).read())
        except (OSError, ValueError, TypeError):
            return 0

    def split_mib(self):
        return {k: self._rd(f"mem_info_{k}") >> 20 for k in ("vram_used", "vram_total", "gtt_used", "gtt_total")}

    def used_mib(self):
        if not self.info:
            return None
        return (self._rd("mem_info_vram_used") + self._rd("mem_info_gtt_used")) >> 20

    def mem_mib(self):
        if not self.info:
            return None
        sp = self.split_mib()
        tot = sp["vram_total"] + sp["gtt_total"]; used = sp["vram_used"] + sp["gtt_used"]
        return tot, tot - used, used

    def procs_mib(self):
        return None


class _NoGpuMem:
    h = None

    def used_mib(self):
        return None

    def mem_mib(self):
        return None

    def procs_mib(self):
        return None


_NVML = None


def _nvml():
    global _NVML
    if _NVML is None:
        n = _Nvml(gpu_index() if gpu_index().isdigit() else 0) if not (IS_MAC) else _NoGpuMem()
        if n.h is None:
            a = _AmdSysfs() if sys.platform.startswith("linux") else None
            n = a if (a is not None and a.h) else (n if hasattr(n, "lib") else _NoGpuMem())
        _NVML = n
    return _NVML


def vram_used_mib():
    _nvml()
    v = _NVML.used_mib()
    if v is None:
        out = sh(["nvidia-smi", "-i", gpu_index(), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 5)
        try:
            v = int(out.strip().splitlines()[0])
        except Exception:
            v = None
    return v


def proc_rss(pid):
    """RSS bytes of pid on macOS (ps) / Windows (tasklist), None when unreadable."""
    try:
        if IS_MAC:
            return int(sh(["ps", "-o", "rss=", "-p", str(pid)], 5).strip()) * 1024
        if IS_WIN:
            o = sh(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], 5)
            return int(re.sub(r"[^\d]", "", o.strip().split('","')[-1])) * 1024
    except Exception:
        return None
    return None


class Sampler(threading.Thread):
    """VRAM used on our GPU (NVML), child RSS and host MemAvailable every `period` s."""

    def __init__(self, pid, period=0.5):
        super().__init__(daemon=True)
        self.pid, self.period, self.stop_ev, self.samples = pid, period, threading.Event(), []
        self.vram0 = vram_used_mib()
        self.dev0 = _nvml().mem_mib()  # (total, free, used) MiB just before launch
        self.procs0 = _nvml().procs_mib()
        self.phase = "load"           # server_measure moves it: load -> serve -> fill
        self.proc = []                # (t, phase, own-pid MiB or None, sum of new pids MiB)
        self.load1_max = (getload() or 0)
        self.amd_peak = None
        self.amd0 = _nvml().split_mib() if isinstance(_nvml(), _AmdSysfs) else None

    def run(self):
        t0 = time.time()
        while not self.stop_ev.is_set():
            vram = vram_used_mib()
            rss = rss_file = rss_anon = None
            try:
                for line in open(f"/proc/{self.pid}/status"):
                    if line.startswith("VmRSS"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("RssFile"):
                        rss_file = int(line.split()[1]) * 1024
                    elif line.startswith("RssAnon"):
                        rss_anon = int(line.split()[1]) * 1024
            except Exception:
                pass
            if rss is None and not sys.platform.startswith("linux"):
                rss = proc_rss(self.pid)
            self.samples.append((round(time.time() - t0, 1), vram, rss, meminfo().get("MemAvailable"), rss_file, rss_anon))
            if isinstance(_nvml(), _AmdSysfs):
                sp = _nvml().split_mib()
                self.amd_peak = {k: max(v, (self.amd_peak or {}).get(k, 0)) for k, v in sp.items()}
            self.load1_max = max(self.load1_max, (getload() or 0))
            pm = _nvml().procs_mib()
            if pm is not None:
                # own pid when the PID namespace matches; else the processes that were not there before launch
                new = [v for k, v in pm.items() if k not in (self.procs0 or {}) and v is not None]
                self.proc.append((round(time.time() - t0, 1), self.phase, pm.get(self.pid), sum(new) if new else None))
            self.stop_ev.wait(self.period)

    def summary(self):
        v = [s[1] for s in self.samples if s[1] is not None]
        r = [s[2] for s in self.samples if s[2] is not None]
        a = [s[3] for s in self.samples if s[3] is not None]
        return {"vram_used_mib_peak": max(v) if v else None, "vram_used_mib_before": self.vram0,
                "vram_delta_mib_peak": (max(v) - self.vram0) if v and self.vram0 is not None else None,
                "rss_peak": max(r) if r else None, "memavail_min": min(a) if a else None,
                # mmap'd weights show up as RssFile (page-cache pages mapped and touched), RAM copies as RssAnon
                "rss_file_peak": max([s[4] for s in self.samples if s[4] is not None] or [0]) or None,
                "rss_anon_peak": max([s[5] for s in self.samples if s[5] is not None] or [0]) or None,
                "n_samples": len(self.samples), "load1_max": round(self.load1_max, 1),
                "device_before_mib": dict(zip(("total", "free", "used"), self.dev0)) if self.dev0 else None,
                "procs_before_mib": self.procs0,
                "amd_before_mib": self.amd0, "amd_peak_mib": self.amd_peak,
                **self._proc_summary(),
                "trace": self.samples[:: max(1, len(self.samples) // 60)]}


def _proc_summary_impl(self):
    """Per-process NVML used_memory peaks by phase: after load (load + serve) and after the context fill."""
    if not self.proc:
        return {"proc_used_mib": None}
    own = any(s[2] is not None for s in self.proc)
    col = 2 if own else 3
    def pk(ph):
        v = [s[col] for s in self.proc if s[1] in ph and s[col] is not None]
        return max(v) if v else None
    return {"proc_used_mib": {"source": "own_pid" if own else "new_pids", "peak_load": pk(("load",)),
                              "peak_after_load": pk(("load", "serve")), "peak_fill": pk(("fill",)),
                              "peak_all": pk(("load", "serve", "fill")), "n": len(self.proc),
                              "trace": self.proc[:: max(1, len(self.proc) // 60)]}}


Sampler._proc_summary = _proc_summary_impl


def run_proc(cmd, env, timeout, cpus, log_path, wait_fn=None):
    """Run cmd (optionally pinned) with sampling; returns (rc, stdout, stderr_tail, cpu_s, wall_s, mem)."""
    if cpus:
        cmd = ["taskset", "-c", ",".join(map(str, cpus))] + cmd
    t0 = time.time()
    if IS_WIN:
        return _run_proc_win(cmd, env, timeout, log_path, wait_fn, t0)
    with open(log_path, "w", encoding="utf-8", errors="replace") as errf:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=errf, text=True, start_new_session=True,
                             encoding="utf-8", errors="replace")
        OWNED_PGIDS.add(p.pid)
        smp = Sampler(p.pid, SAMPLE_PERIOD); smp.start()
        out_chunks = []
        reader = threading.Thread(target=lambda: out_chunks.append(p.stdout.read()), daemon=True); reader.start()
        extra = None; err = None
        if wait_fn is not None:
            try:
                extra = wait_fn(p, smp)
            except Exception as e:  # reap the child before reporting
                err = e
            finally:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except Exception:
                    pass
        deadline = t0 + timeout
        status = None; ru = None; reaped = False
        while True:
            try:
                pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            except ChildProcessError:  # already reaped by Popen.poll() (server_measure's early-exit check)
                reaped = True; break
            if pid:
                break
            if time.time() > deadline:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass
                try:
                    pid, status, ru = os.wait4(p.pid, 0)
                except ChildProcessError:
                    pass
                status = -9 << 8
                break
            time.sleep(0.2)
        if reaped:
            p.returncode = p.returncode if p.returncode is not None else -1
        else:
            p.returncode = os.waitstatus_to_exitcode(status) if status >= 0 else -9
        try:
            os.killpg(p.pid, signal.SIGKILL)  # stragglers of our own group only
        except Exception:
            pass
        OWNED_PGIDS.discard(p.pid)
        reader.join(5)
        smp.stop_ev.set(); smp.join(5)
    wall = time.time() - t0
    cpu_s = (ru.ru_utime + ru.ru_stime) if ru else None
    tail = open(log_path, errors="replace").read()[-3000:]
    if err is not None:
        raise RuntimeError(f"{type(err).__name__}: {err} | log: {tail[-800:]}")
    return p.returncode, "".join(c for c in out_chunks if c), tail, cpu_s, wall, smp.summary(), extra


def _run_proc_win(cmd, env, timeout, log_path, wait_fn, t0):
    """Windows: no process groups / wait4; kill the tree with taskkill, CPU time from the process handle."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as errf:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=errf, text=True, encoding="utf-8",
                             errors="replace", creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        smp = Sampler(p.pid, SAMPLE_PERIOD); smp.start()
        out_chunks = []
        reader = threading.Thread(target=lambda: out_chunks.append(p.stdout.read()), daemon=True); reader.start()
        extra = None; err = None
        cpu_s = None

        def cpu_times():
            try:
                import ctypes
                from ctypes import wintypes
                ft = [wintypes.FILETIME() for _ in range(4)]
                h = ctypes.windll.kernel32.OpenProcess(0x1000, False, p.pid)  # QUERY_LIMITED_INFORMATION
                if h and ctypes.windll.kernel32.GetProcessTimes(h, *[ctypes.byref(f) for f in ft]):
                    v = lambda f: ((f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e7  # noqa: E731
                    return v(ft[2]) + v(ft[3])
            except Exception:
                return None
            return None
        if wait_fn is not None:
            try:
                extra = wait_fn(p, smp)
            except Exception as e:
                err = e
            finally:
                cpu_s = cpu_times()
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        try:
            p.wait(timeout=max(1, t0 + timeout - time.time()))
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            p.wait(30); p.returncode = -9
        if cpu_s is None:
            cpu_s = cpu_times()
        reader.join(5)
        smp.stop_ev.set(); smp.join(5)
    wall = time.time() - t0
    tail = open(log_path, encoding="utf-8", errors="replace").read()[-3000:]
    if err is not None:
        raise RuntimeError(f"{type(err).__name__}: {err} | log: {tail[-800:]}")
    return p.returncode, "".join(c for c in out_chunks if c), tail, cpu_s, wall, smp.summary(), extra


# ----------------------------------------------------------------------------- binaries
def bin_env(bin_dir, env=None):
    env = dict(env or os.environ)
    if IS_WIN:
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    elif IS_MAC:
        env["DYLD_LIBRARY_PATH"] = bin_dir + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    else:
        env["LD_LIBRARY_PATH"] = bin_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


DEV_RX = r"\s*(CUDA|ROCm|Vulkan|MTL|Metal|HIP)\d*:"


def list_cuda_devices(bin_dir):
    """Devices the bundle's GPU backend actually loads (a missing libcudart / libcublas leaves CPU only)."""
    try:
        out = subprocess.run([os.path.join(bin_dir, "llama-server" + EXE), "--list-devices"], env=bin_env(bin_dir),
                             capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace")
        txt = out.stdout + out.stderr
    except Exception as e:
        txt = str(e)
    return [ln.strip() for ln in txt.splitlines() if re.match(DEV_RX, ln)], txt[-1500:]


def _extract(archive, d):
    """Unpack the release archive into d keeping relative paths (ROCm bundles carry rocblas/ and hipblaslt/
    library trees next to the binaries); returns the directory holding llama-server."""
    root = os.path.realpath(d)

    def safe(nm):
        t = os.path.realpath(os.path.join(root, nm))
        return t if (t + os.sep).startswith(root + os.sep) and not os.path.isabs(nm) else None
    if archive.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(archive) as zf:
            for m in zf.infolist():
                t = safe(m.filename)
                if t is None or m.is_dir():
                    continue
                os.makedirs(os.path.dirname(t), exist_ok=True)
                with zf.open(m) as src, open(t, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if not (m.isfile() or m.issym() or m.isdir()) or safe(m.name) is None:
                    continue
                if m.issym() and (os.path.isabs(m.linkname) or ".." in m.linkname.split("/")):
                    continue
                tf.extract(m, root)
    for dp, _, fns in os.walk(root):
        if "llama-server" + EXE in fns:
            if not IS_WIN:
                for nm in fns:
                    if nm.startswith("llama-"):
                        os.chmod(os.path.join(dp, nm), 0o755)
            return dp
    raise RuntimeError(f"{archive}: no llama-server{EXE} inside")


def ensure_bins(bin_dir, variant, tag):
    if bin_dir and os.path.exists(os.path.join(bin_dir, "llama-bench" + EXE)):
        return bin_dir
    bin_root = bin_dir or os.path.abspath("llama_bin")
    if IS_MAC:
        order = ["metal"]
    elif variant:
        order = [variant]
    else:
        cc = sh(["nvidia-smi", "-i", gpu_index(), "--query-gpu=compute_cap", "--format=csv,noheader"]).strip()
        sm = int(float(cc or "8.6") * 10)
        drv = re.search(r"CUDA Version:\s*(\d+)", sh(["nvidia-smi"]))
        cuda_major = int(drv.group(1)) if drv else 12
        # the bundle's CUDA runtime major must be installed on the host (they ship no libcudart); cuda13-newer
        # needs sm >= 86, cuda13-portable covers sm 75-120, cuda12-portable sm 70-120
        order = (["cuda13-newer"] if sm >= 86 else []) + ["cuda13-portable", "cuda12-portable"] if cuda_major >= 13 \
            else ["cuda12-portable"]
    for var in order:
        d0 = os.path.join(bin_root, tag + "_" + var)
        d = next((dp for dp, _, fns in os.walk(d0) if "llama-server" + EXE in fns), None) if os.path.isdir(d0) else None
        if d is None:
            os.makedirs(d0, exist_ok=True)
            url = RELEASE_BASE.format(tag=tag) + asset_name(tag, var)
            arc = os.path.join(d0, os.path.basename(url))
            log("downloading prebuilt", url)
            urllib.request.urlretrieve(url, arc)
            d = _extract(arc, d0)
            os.remove(arc)
        devs, txt = list_cuda_devices(d)
        log(f"prebuilt {tag} {var}: GPU devices {devs or 'NONE'}")
        if devs:
            return d
        log("  no GPU backend loaded with", var, "->", txt[-400:].replace("\n", " | "))
    raise RuntimeError(f"no prebuilt variant of {tag} loads a GPU backend on this host (tried {order})")


# ----------------------------------------------------------------------------- downloads
def remote_size(repo, fname):
    try:
        from huggingface_hub import HfApi
        info = HfApi().get_paths_info(repo, [fname])
        for i in info:
            return i.size
    except Exception:
        pass
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{fname}", method="HEAD")
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req) as r:
        return int(r.headers.get("X-Linked-Size") or r.headers.get("Content-Length"))


def fetch(repo, fname, models_dir, hw, budget_note):
    dst = os.path.join(models_dir, repo.replace("/", "__"), fname)
    size = remote_size(repo, fname)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        if OWN_EXISTING:  # a file another step of the same job left (FA check): delete it with this model's block
            own_path(dst)
        return dst, False, size
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    own_path(dst)
    own_path(os.path.join(os.path.dirname(dst), ".cache"))  # hf_hub_download staging (*.incomplete), same fs
    log(f"download {repo}/{fname} ({size / 1e9:.2f} GB) {budget_note}")
    t0 = time.time()
    try:
        from huggingface_hub import hf_hub_download
        for attempt in range(6):  # HF CDN 429 / transient errors: back off, then drop XET for plain HTTP
            try:
                # local_dir = the repo dir: hf_hub_download writes local_dir/<fname>, and fname may carry a subdir
                # (split quants: UD-Q2_K_XL/x-00001-of-00004.gguf). dirname(dst) doubled it (.../UD-Q2_K_XL/UD-Q2_K_XL/),
                # so llama-server found no file: every split-quant arm failed on the 2026-11-26 G4 / A100-80 runs
                hf_hub_download(repo, fname, local_dir=os.path.join(models_dir, repo.replace("/", "__")))
                break
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 30 * (attempt + 1)
                log(f"  download attempt {attempt + 1} failed ({type(e).__name__}: {str(e)[:160]}); retry in {wait} s"
                    + (" without XET" if attempt >= 2 else ""))
                if attempt >= 2:
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                    try:
                        import huggingface_hub.constants as _hc
                        _hc.HF_HUB_DISABLE_XET = True
                    except Exception:
                        pass
                time.sleep(wait)
    except ImportError:
        req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{fname}")
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
        own_path(dst + ".part")
        with urllib.request.urlopen(req) as r, open(dst + ".part", "wb") as f:
            shutil.copyfileobj(r, f, 16 << 20)
        os.replace(dst + ".part", dst)
    log(f"  done in {time.time() - t0:.0f} s")
    if not (os.path.exists(dst) and os.path.getsize(dst) == size):
        raise RuntimeError(f"download landed elsewhere: {dst} missing or wrong size")
    return dst, True, size



# ----------------------------------------------------------------------------- preflight / ownership
OWNED_FILES = []     # files and dirs this run created (deleted on exit unless --keep-models)
OWN_EXISTING = False  # --own-existing
SAMPLE_PERIOD = 0.5  # s; NVML device + per-process memory sampling
FILL_ROUNDS = 0     # >0: the fill request only in the first N rounds of each config
FILL_FRAC = 0.0     # >0: after the decode reps, one request filling this fraction of the slot context
OWNED_PGIDS = set()  # process groups this run started (killed on exit / failure)
BW_GPU_NOMINAL = {"B200": 8.0e12, "H100": 3.35e12, "A100-SXM4-80GB": 2.04e12, "A100": 1.555e12, "L4": 0.3e12,
                  "T4": 0.32e12, "RTX PRO 6000": 1.79e12, "A10": 0.6e12}


def build_shim(out_dir):
    """Compile the cudaMemGetInfo LD_PRELOAD shim from fakemem.c next to this script (any CUDA runtime major)."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakemem.c")
    dst = os.path.join(out_dir, "fakemem.so")
    if os.path.exists(dst):
        return dst
    if not os.path.exists(src):
        return None
    r = subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", dst, src, "-ldl"], capture_output=True, text=True)
    if r.returncode != 0:
        log("shim build failed:", r.stderr[-500:])
        return None
    log("built shim", dst)
    return dst


def cleanup_owned(keep_models):
    for pg in list(OWNED_PGIDS):
        try:
            os.killpg(pg, signal.SIGKILL)
        except Exception:
            pass
        OWNED_PGIDS.discard(pg)
    if keep_models:  # weights stay; header sidecars never do
        for pth in [x for x in OWNED_FILES if x.endswith(".header")]:
            try:
                os.remove(pth)
            except OSError:
                pass
        return
    for pth in reversed(OWNED_FILES):
        try:
            if os.path.isdir(pth):
                shutil.rmtree(pth)
            elif os.path.exists(pth):
                os.remove(pth)
            log("deleted (run-owned)", pth)
        except OSError:
            pass
    OWNED_FILES.clear()


def own_path(pth):
    """Register pth only if it does not exist yet (never delete something this run did not create)."""
    if not os.path.exists(pth) and pth not in OWNED_FILES:
        OWNED_FILES.append(pth)


def cgroup_mem_avail():
    """Bytes the cgroup still allows (memory.max - memory.current), None when unlimited / unknown."""
    for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
        for mx, cur in (("memory.max", "memory.current"), ("memory.limit_in_bytes", "memory.usage_in_bytes")):
            try:
                lim = open(os.path.join(base, mx)).read().strip()
                if lim == "max" or int(lim) > 1 << 60:
                    return None
                return int(lim) - int(open(os.path.join(base, cur)).read().strip())
            except Exception:
                continue
    return None


def vram_free_mib():
    mm = _nvml().mem_mib()
    if mm:
        return mm[1]
    out = sh(["nvidia-smi", "-i", gpu_index(), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], 5)
    try:
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def fetch_header(repo, fname, size, dst_dir):
    """Range-read only the GGUF header into a sparse file of the real size (no weights), for preflight."""
    hdr = os.path.join(dst_dir, os.path.dirname(fname), "." + os.path.basename(fname) + ".header")
    os.makedirs(os.path.dirname(hdr), exist_ok=True)
    if os.path.exists(hdr) and os.path.getsize(hdr) == size:
        return hdr
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    n = 4 << 20
    while True:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
        with urllib.request.urlopen(req, timeout=120) as r:
            buf = r.read(n + 1)
        if len(buf) > n:
            raise RuntimeError("range request ignored")
        if gguf_header_complete(buf):
            with open(hdr + ".tmp", "wb") as f:
                f.write(buf); f.truncate(size)
            os.replace(hdr + ".tmp", hdr)
            return hdr
        if len(buf) < n or n >= (256 << 20):
            raise RuntimeError(f"GGUF header of {fname} larger than {n >> 20} MiB or truncated")
        n *= 4


def estimate_seconds(cfg, by, hw, mode, scale):
    """Conservative runtime for one config (load + warmup + r repetitions), from header bytes."""
    name = hw["gpu"].get("name", "")
    bw = next((v for k, v in BW_GPU_NOMINAL.items() if k in name), 0.3e12) * 0.3
    tg = 1.0 / (by["tok_gpu"] / bw + by["tok_cpu"] / 15e9 + 0.003)
    pp = 1.0 / (1.0 / 1500 + (by["w_cpu"] / 10e9) / 512)  # CPU weights streamed once per 512-token ubatch
    load = by["w_gpu"] / 1.0e9 + by["w_cpu"] / 2.0e9 + 5
    if mode == "server":
        if cfg.get("resume"):
            return scale * (load + 3 * (cfg["resume"].get("prefix_tokens", 8000) / pp) + 30)
        return scale * (load + (cfg["r"] + 2) * (cfg["n"] / tg + 300 / pp))
    depth = cfg.get("depth", 0)
    per = cfg["p"] / pp + cfg["n"] / tg + 2 * depth / pp
    return scale * (load + (cfg["r"] + 1) * per)


# ----------------------------------------------------------------------------- runners
def bench_cmd(bins, path, cfg, ot, threads):
    c = [os.path.join(bins, "llama-bench" + EXE), "-m", path, "-p", str(cfg["p"]), "-n", str(cfg["n"]),
         "-r", str(cfg["r"]), "-fa", cfg["fa"], "-ctk", cfg["ctk"], "-ctv", cfg["ctv"], "-o", "jsonl",
         "-ngl", str(cfg.get("ngl", 999)), "-ub", str(cfg.get("ub", 512)), "-b", str(cfg.get("b", 2048))]
    if cfg.get("depth"):
        c += ["-d", str(cfg["depth"])]
    if threads:
        c += ["-t", str(threads)]
    if cfg.get("nkvo"):
        c += ["-nkvo", "1"]
    if ot:
        c += ["-ot", ";".join(ot)]
    return c + list(cfg.get("extra", []))


def parse_bench(stdout):
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    res = {}
    for r in rows:
        kind = "pp" if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0 else "tg"
        res[kind] = {"avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts"), "samples_ts": r.get("samples_ts"),
                     "n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"), "n_depth": r.get("n_depth"),
                     "model_size": r.get("model_size"), "gpu_info": r.get("gpu_info"), "build": r.get("build_commit")}
    return res


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def http(port, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_cmd(bins, info_paths, cfg, ot, threads, port):
    c = [os.path.join(bins, "llama-server" + EXE), "-m", info_paths["model"], "--host", "127.0.0.1", "--port", str(port),
         "-fa", cfg["fa"], "-ctk", cfg["ctk"], "-ctv", cfg["ctv"],
         "-np", str(cfg.get("np", 1)), "--fit", cfg.get("fit", "off"), "-lv", "4"]
    if cfg.get("ctx", 8192) != "omit":  # "omit": no -c, --fit on picks the context (baseline A)
        c += ["-c", str(cfg.get("ctx", 8192))]
    if cfg.get("ngl", 999) != "omit":  # "omit": leave the layer count to --fit on
        c += ["-ngl", str(cfg.get("ngl", 999))]
    if cfg.get("ub"):
        c += ["-ub", str(cfg["ub"])]
    if cfg.get("no_warmup"):
        c += ["--no-warmup"]
    if threads:
        c += ["-t", str(threads)]
    if cfg.get("nkvo"):
        c += ["--no-kv-offload"]
    for o in ot:
        c += ["-ot", o]
    if cfg.get("moe_cache_mib") is not None:
        c += ["--moe-cache-mib", str(cfg["moe_cache_mib"])]
    if cfg.get("moe_cache_cpu_share") is not None:
        c += ["--moe-cache-cpu-share", str(cfg["moe_cache_cpu_share"])]
    if cfg.get("spec"):
        c += ["--spec-type", cfg["spec"]]
        if cfg.get("spec_n_max"):
            c += ["--spec-draft-n-max", str(cfg["spec_n_max"])]
        # an external MTP file (gemma-4 mtp-*.gguf) also uses draft-mtp: "use_drafter": true passes it with -md
        if (info_paths.get("drafter") and cfg["spec"].startswith("draft-")
                and (cfg["spec"] != "draft-mtp" or cfg.get("use_drafter"))):  # ngram-mod etc. load no drafter
            c += ["-md", info_paths["drafter"], "-ngld", "99"]
    mm = cfg.get("mmproj")
    if mm in ("cpu", "gpu") and info_paths.get("mmproj"):
        c += ["--mmproj", info_paths["mmproj"]]
        if mm == "cpu":
            c += ["--no-mmproj-offload"]
    else:
        c += ["--no-mmproj"]
    if cfg.get("load_mode"):
        c += ["--load-mode", cfg["load_mode"]]
    if cfg.get("lazy_mode"):
        c += ["--lazy-mode", cfg["lazy_mode"]]
    if cfg.get("ctx_checkpoints") is not None:
        c += ["--ctx-checkpoints", str(cfg["ctx_checkpoints"])]
    if cfg.get("cache_ram") is not None:
        c += ["--cache-ram", str(cfg["cache_ram"])]
    return c + list(cfg.get("extra", []))


def _timing_row(r, t):
    tm = r.get("timings", {})
    return {"tg_ts": tm.get("predicted_per_second"), "pp_ts": tm.get("prompt_per_second"),
            "n_prompt": tm.get("prompt_n"), "n_cached": tm.get("cache_n"), "prompt_ms": tm.get("prompt_ms"),
            "decode_ms": tm.get("predicted_ms"), "n_gen": tm.get("predicted_n"),
            "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"),
            "text_head": (r.get("content") or "")[:160], "wall_s": round(time.time() - t, 3)}


def long_prompt(n_tokens, seed):
    """Deterministic varied text of roughly n_tokens tokens (about 10 tokens per line)."""
    words = ["alpha", "river", "copper", "lantern", "seven", "quiet", "market", "orbit", "maple", "signal",
             "harbor", "violet", "engine", "summit", "pocket", "thunder", "garden", "silver", "anchor", "meadow"]
    lines = []
    x = seed * 7919 + 1
    for i in range(max(1, n_tokens // 10)):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        lines.append(f"{i}: " + " ".join(words[(x >> (3 * k)) % len(words)] for k in range(6)) + ".")
    return "Notes:\n" + "\n".join(lines) + "\n"


def resume_probe(port, cfg, timeout):
    """Resumed conversation: A = long prefix, B = A + its answer + a new turn (same slot: prefix reuse),
    then C = an unrelated prompt of the same length (evicts the slot), then B again (restored only through
    --cache-ram, or through ctx checkpoints for recurrent / SWA models). Records tokens re-processed and ms."""
    n = int(cfg["resume"].get("prefix_tokens", 8000))
    pa = long_prompt(n, 1) + "Question: summarise the notes.\nAnswer:"
    body = {"n_predict": 16, "temperature": 0.0, "top_k": 1, "cache_prompt": True, "seed": 0}
    out = {}
    t = time.time(); ra = http(port, "/completion", {**body, "prompt": pa}, timeout=timeout)
    out["A_first"] = _timing_row(ra, t)
    pb = pa + (ra.get("content") or "") + "\nQuestion: and the first line?\nAnswer:"
    t = time.time(); out["B_same_slot"] = _timing_row(http(port, "/completion", {**body, "prompt": pb}, timeout=timeout), t)
    t = time.time(); out["C_other"] = _timing_row(http(port, "/completion", {**body, "prompt": long_prompt(n, 2)}, timeout=timeout), t)
    t = time.time(); out["B_after_evict"] = _timing_row(http(port, "/completion", {**body, "prompt": pb + " "}, timeout=timeout), t)
    return out


def fill_probe(port, frac, timeout):
    """One request whose prompt + output fill `frac` of the slot context (peak KV / compute residency)."""
    props = http(port, "/props", timeout=10)
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
    if not n_ctx:
        return {"error": "no n_ctx in /props"}
    target = int(n_ctx * frac) - 16
    probe = long_prompt(2000, 3)
    try:
        ntok = len(http(port, "/tokenize", {"content": probe}, timeout=30).get("tokens", []))
        per_line = max(1.0, ntok / max(1, probe.count("\n")))
    except Exception:
        per_line = 12.0
    lines = int(target / per_line)
    text = long_prompt(int(lines * 10), 3)  # long_prompt makes n_tokens // 10 lines
    try:  # trim to the target exactly when tokenize works
        toks = http(port, "/tokenize", {"content": text}, timeout=60).get("tokens", [])
        if len(toks) > target:
            text = http(port, "/detokenize", {"tokens": toks[:target]}, timeout=60).get("content", text)
    except Exception:
        pass
    t = time.time()
    r = http(port, "/completion", {"prompt": text, "n_predict": 8, "temperature": 0.0, "top_k": 1,
                                   "cache_prompt": False, "seed": 0}, timeout=timeout)
    row = _timing_row(r, t)
    row.update({"n_ctx_slot": n_ctx, "frac": frac, "target_tokens": target})
    return row


def server_measure(cfg, port, timeout, smp=None, proc=None):
    """Wait for /health; a 1-token warmup on a different short prompt (CUDA graphs / allocator warm, MoE cache
    nearly empty); then the FIRST full decode is recorded as `cold`, the next r as warm `reps`."""
    t0 = time.time()
    while True:
        try:
            if http(port, "/health", timeout=5).get("status") == "ok":
                break
        except Exception:
            pass
        if proc is not None and proc.poll() is not None:  # died while loading: fail now, not at the timeout
            raise RuntimeError(f"server exited with rc {proc.returncode} before becoming healthy")
        if time.time() - t0 > timeout:
            raise TimeoutError("server did not become healthy")
        time.sleep(0.5)
    load_s = time.time() - t0
    if smp is not None:
        time.sleep(max(1.0, 2 * SAMPLE_PERIOD))  # at least one post-load sample before requests
        smp.phase = "serve"
    out = {"load_s": round(load_s, 2), "reps": []}
    deadline = time.time() + timeout  # the measurement phase gets its own budget after load
    if cfg.get("resume"):
        out["resume"] = resume_probe(port, cfg, max(60, timeout))
        return out
    prompt = PROMPT * max(1, int(cfg.get("prompt_repeat", 1)))
    body = {"prompt": prompt, "n_predict": cfg["n"], "temperature": 0.0, "top_k": 1, "cache_prompt": False,
            "ignore_eos": True, "seed": 0}
    smp_prof = cfg.get("sampling") or "greedy"
    body.update(SAMPLING[smp_prof] if isinstance(smp_prof, str) else smp_prof)
    http(port, "/completion", {**body, "prompt": "Hello", "n_predict": 1}, timeout=max(30, timeout))
    t = time.time()
    out["cold"] = _timing_row(http(port, "/completion", body, timeout=max(30, timeout)), t)
    prompts = PROMPT_SETS[cfg["prompt_set"]] if cfg.get("prompt_set") else [prompt] * cfg["r"]
    k = cfg.get("_round", 0) % len(prompts)  # rotated rounds: each round starts at a different prompt
    idx = list(range(len(prompts)))[k:] + list(range(len(prompts)))[:k]
    for i in idx:
        pr = prompts[i]
        left = deadline - time.time()
        if out["reps"] and left < 5:
            break  # partial: keep what was measured
        t = time.time()
        row = _timing_row(http(port, "/completion", {**body, "prompt": pr}, timeout=max(30, left)), t)
        row["prompt_idx"] = i
        out["reps"].append(row)
    frac = cfg.get("fill_frac", FILL_FRAC)
    if frac and FILL_ROUNDS and cfg.get("_round", 0) >= FILL_ROUNDS:
        frac = 0
    if frac:
        time.sleep(2 * SAMPLE_PERIOD)
        if smp is not None:
            smp.phase = "fill"
        try:
            out["fill"] = fill_probe(port, frac, max(120, timeout))
        except Exception as e:
            out["fill"] = {"error": f"{type(e).__name__}: {e}"}
        time.sleep(2 * SAMPLE_PERIOD)
    return out


LOG_PATTERNS = {
    "buffers": re.compile(r"(\S+) (model|KV|compute|RS) buffer size =\s*([\d.]+) MiB"),
    "moe_cache": re.compile(r"moe[ _]cache[^\n]*", re.I),
    "fit": re.compile(r"(?:common_params_fit_impl|common_fit_params|llama_params_fit)[^\n]*"),
    "kv_size": re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB[^\n]*"),
    "splits": re.compile(r"graph(?: \(pp bs=\d+, tg bs=\d+\))?: nodes = ([\d /]+), splits = ([\d /]+)"),
    "lazy": re.compile(r"[^\n]*(?:lazy|on.demand|per.layer|engram)[^\n]*", re.I),
    "upload": re.compile(r"uploaded = ([\d.]+) MiB"),
    "dev_buffer": re.compile(r"(\w+):\s+(CUDA\d+|ROCm\d+|Vulkan\d+|MTL\d+|Metal)\s+(.+?) buffer size =\s*([\d.]+) MiB[^\n]*"),
    "keep": re.compile(r"[^\n]*(?:buffer size|MoE cache size|offloaded \d+/\d+ layers|projected to use|"
                       r"free memory target|memory breakdown|\|\s+- (?:CUDA|ROCm|Vulkan|Host)|MiB free\)|"
                       r"using device)[^\n]*"),
    "moe_cache_size": re.compile(r"(CUDA\d+|ROCm\d+|Vulkan\d+|MTL\d+) MoE cache size =\s*([\d.]+) MiB"),
}


def device_buffers(text):
    """Every device buffer line (main model, drafter, KV, RS, compute, output, mmproj, MoE cache) and their sum."""
    lines = []
    for m in LOG_PATTERNS["dev_buffer"].finditer(text):
        lines.append({"fn": m.group(1), "dev": m.group(2), "kind": m.group(3).strip(), "mib": float(m.group(4))})
    for m in LOG_PATTERNS["moe_cache_size"].finditer(text):
        lines.append({"fn": "moe_cache", "dev": m.group(1), "kind": "MoE cache", "mib": float(m.group(2))})
    # mmproj weights on the GPU are not printed as a "buffer size" line: the clip loader's model size + backend
    mb = re.search(r"mtmd_backends: using (CUDA\d+|ROCm\d+|Vulkan\d+|MTL\d+) backend", text)
    ms = re.search(r"load_hparams: model size:\s*([\d.]+) MiB", text)
    if mb and ms:
        lines.append({"fn": "mtmd_model", "dev": mb.group(1), "kind": "mmproj model", "mib": float(ms.group(1))})
    # sched_reserve re-reserves the same context's compute buffer later in the run (same size, same buffer):
    # count each (fn, dev, kind, size) once
    seen, total = set(), 0.0
    for x in lines:
        k = (x["fn"], x["dev"], x["kind"], x["mib"])
        if x["dev"] not in ("CUDA0", "ROCm0", "Vulkan0", "MTL0", "Metal") or (x["fn"] == "sched_reserve" and k in seen):
            continue
        seen.add(k); total += x["mib"]
    return {"lines": lines, "sum_mib": round(total, 2)}


def parse_server_log(text):
    bufs = {}
    for m in LOG_PATTERNS["buffers"].finditer(text):
        k = f"{m.group(1)}:{m.group(2)}"
        bufs[k] = round(bufs.get(k, 0.0) + float(m.group(3)), 2)
    return {"buffers_mib": bufs, "moe_cache_lines": LOG_PATTERNS["moe_cache"].findall(text)[:12],
            "fit_lines": [ln for ln in LOG_PATTERNS["fit"].findall(text)
                          if re.search(r"projected|set ngl|moe cache|surplus|overflow|will leave|n_layer=", ln)][-8:],
            "kv_size_lines": [m.group(0) for m in LOG_PATTERNS["kv_size"].finditer(text)][:4],
            "n_ctx": [int(x) for x in re.findall(r"llama_context: n_ctx\s+= (\d+)", text)][:2],
            # FA / any op that falls back to the CPU shows up as extra graph splits in the tg graph
            # (an all-GPU graph has ~2); compare against the f16 row of the same model
            "graph_splits": [m.group(2).strip() for m in LOG_PATTERNS["splits"].finditer(text)][:3],
            "lazy_lines": [ln.strip()[:200] for ln in LOG_PATTERNS["lazy"].findall(text)
                           if "lazy read" in ln.lower() or "engram" in ln.lower()][:8],
            "moe_uploaded_mib": max([float(x) for x in LOG_PATTERNS["upload"].findall(text)] or [0.0]),
            "device_buffers": device_buffers(text),
            # the fitter's whole decision (layers kept / overflowed, tensor overrides it set, ctx, cache), verbatim
            "fit_all": [re.sub(r"^\S+ [A-Z] ", "", ln.strip())[:300] for ln in re.findall(
                r"[^\n]*(?:common_params_fit_impl|common_fit_params|llama_params_fit|tensor_buft_override|buft override|"
                r"overrid|=CPU)[^\n]*", text)][:120],
            "mem_lines": [re.sub(r"^\S+ [A-Z] ", "", ln.strip())[:240] for ln in LOG_PATTERNS["keep"].findall(text)][:80]}


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default=os.path.abspath("placement_models"))
    ap.add_argument("--bin-dir")
    ap.add_argument("--variant", help="prebuilt asset variant (cuda13-newer, cuda12-portable, ...)")
    ap.add_argument("--tag", default=RELEASE_TAG)
    ap.add_argument("--threads", type=int, default=0, help="0 = llama.cpp default")
    ap.add_argument("--cpus", default="auto", help="auto | none | cpulist")
    ap.add_argument("--keep-models", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--models", default="", help="only these model aliases")
    ap.add_argument("--rounds", type=int, default=1, help=">1: rerun the model's configs, alternate order (A/A, drift)")
    ap.add_argument("--timeout", type=int, default=240, help="per config, seconds")
    ap.add_argument("--max-load1", type=float, default=0, help="wait before each run until host load1 <= this "
                    "(0 = off); rows whose load1 (pre / post / max while running) exceeds it get contended=true")
    ap.add_argument("--contended-load1", type=float, default=0, help="mark rows contended above this load1 "
                    "(default: --max-load1); lets the start gate be looser than the contention mark")
    ap.add_argument("--load-wait-s", type=float, default=900, help="longest wait per run for --max-load1")
    ap.add_argument("--own-existing", action="store_true", help="also delete model files already on disk after their block")
    ap.add_argument("--prompt-set", default="", help="server configs: calib | holdout prompt set for every config")
    ap.add_argument("--merge-models", action="store_true", help="one download per (repo, file, mmproj); per-arm drafter kept")
    ap.add_argument("--interleave-by-point", action="store_true", help="run points in sequence, interleave each point's arms")
    ap.add_argument("--fill-rounds", type=int, default=0, help="fill request only in the first N rounds (0 = all)")
    ap.add_argument("--sample-period", type=float, default=0.5, help="NVML / RSS sampling period, s")
    ap.add_argument("--fill-frac", type=float, default=0.0,
                    help="server mode: after the reps, one request filling this fraction of the context (0 = off)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-host-probe", action="store_true", help="skip the disk / memcpy host probe")
    ap.add_argument("--vram-budget-mib", type=int, default=0, help="preflight VRAM when the device reports none "
                    "(unified memory: Strix Halo, Apple); 0 = no VRAM gate when unreadable")
    ap.add_argument("--budget-min", type=float, default=0, help="wall budget for the whole run (0 = none): arms whose "
                    "estimated runtime would overrun it get a SKIP row instead of running")
    ap.add_argument("--expect-gpu", default="", help="substring the GPU name must contain (e.g. T4, L4); else SKIP all")
    ap.add_argument("--dl-mbps", type=float, default=60.0, help="assumed download speed for the budget, MB/s")
    ap.add_argument("--fake-shim", default=os.environ.get("PLACEMENT_FAKE_SHIM") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "ctx_sweep", "shim", "fakemem.so"),
        help="LD_PRELOAD cudaMemGetInfo shim for configs with fake_vram_mib (emulated card for --fit / moe-cache auto)")
    a = ap.parse_args()
    global SAMPLE_PERIOD, FILL_FRAC, FILL_ROUNDS, OWN_EXISTING
    OWN_EXISTING = a.own_existing
    SAMPLE_PERIOD, FILL_FRAC, FILL_ROUNDS = a.sample_period, a.fill_frac, a.fill_rounds

    pts = json.load(open(a.points))
    os.makedirs(a.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    logdir = os.path.splitext(os.path.abspath(a.out))[0] + "_logs"
    os.makedirs(logdir, exist_ok=True)
    a.logdir = logdir
    hw = hardware(a.models_dir)
    log("hardware", json.dumps({k: v for k, v in hw.items() if k != "numa_nodes"}))
    bins = ensure_bins(a.bin_dir, a.variant, a.tag) if not a.dry_run else (a.bin_dir or "")
    if not a.dry_run:
        devs, _ = list_cuda_devices(bins)
        hw["llama_devices"] = devs
        hw["prebuilt"] = os.path.relpath(bins, os.path.abspath(a.bin_dir or "llama_bin")) if not a.bin_dir else os.path.basename(os.path.normpath(bins))
        try:
            hw["prebuilt_info"] = json.load(open(os.path.join(bins, "UNSLOTH_PREBUILT_INFO.json")))
        except Exception:
            hw["prebuilt_info"] = None
        try:
            v = subprocess.run([os.path.join(bins, "llama-server" + EXE), "--version"], env=bin_env(bins),
                               capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
            hw["llama_version"] = (v.stdout + v.stderr).strip()[-400:]
        except Exception:
            hw["llama_version"] = None
        hw["host_probe"] = host_probe(a.models_dir) if not a.no_host_probe else None
        log("host_probe", json.dumps({k: v for k, v in (hw["host_probe"] or {}).items() if k != "lscpu"}))
        json.dump(hw, open(os.path.splitext(os.path.abspath(a.out))[0] + ".host.json", "w"), indent=1, default=str)
    env = bin_env(bins)
    t_start = time.time()
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")  # no second staging copy outside models-dir

    def _on_signal(signum, _frame):
        log(f"signal {signum}: killing run-owned processes, deleting run-owned files")
        cleanup_owned(a.keep_models)
        sys.exit(128 + signum)
    for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sg, _on_signal)
    defaults = {"p": 512, "n": 128, "r": 3, "fa": "on", "ctk": "f16", "ctv": "f16", **pts.get("defaults", {})}
    device_skip = None
    expect = a.expect_gpu or pts.get("expect_gpu", "")
    if expect and expect.lower() not in (hw["gpu"].get("name") or "").lower():
        device_skip = f"device: expected {expect!r}, found {hw['gpu'].get('name')!r}"
    if not a.dry_run and not hw.get("llama_devices"):
        device_skip = "device: the llama.cpp bundle loads no GPU backend on this host (CPU only)"
    try:
        info_json = json.load(open(os.path.join(bins, "UNSLOTH_PREBUILT_INFO.json")))
        sm = str(int(float(hw["gpu"].get("compute_cap") or 0) * 10))
        if hw["gpu"].get("compute_cap") and info_json.get("supported_sms") and sm not in [str(x) for x in info_json["supported_sms"]]:
            device_skip = f"device: sm{sm} not in the prebuilt's supported_sms {info_json['supported_sms']}"
    except Exception:
        pass
    scale = {"v": 1.0}  # observed / estimated runtime, raised after every run that ran longer than estimated
    only = set(filter(None, a.only.split(",")))
    want_models = set(filter(None, a.models.split(",")))
    configs = [{**defaults, **c} for c in pts["configs"] if not only or c["id"] in only]
    if a.merge_models:
        configs = merge_models(pts, configs)
    if a.prompt_set:
        for c in configs:
            if c.get("mode") == "server":
                c["prompt_set"] = a.prompt_set
    order = []
    for c in configs:
        if c["model"] not in order and (not want_models or c["model"] in want_models):
            order.append(c["model"])
    out = open(a.out, "a", encoding="utf-8")

    def skip_row(cfg, reason, rnd=0, **kw):
        row = {"id": cfg["id"], "round": rnd, "model": cfg["model"], "mode": cfg.get("mode", "bench"), "skip": reason,
               "cfg": {k: v for k, v in cfg.items() if k != "model"}, "ts": time.time(), "tag": a.tag,
               "hw": {k: hw[k] for k in ("host", "gpu", "ram_total", "cpu_model", "nproc", "colab")}, **kw}
        out.write(json.dumps(row) + "\n"); out.flush()
        log(f"SKIP {cfg['id']} (round {rnd}): {reason}")

    try:
        _run_models(a, pts, hw, bins, env, configs, order, out, skip_row, device_skip, t_start, scale)
    finally:
        cleanup_owned(a.keep_models)
        out.close()


def model_roles(spec):
    """Files of a model entry: weights, mmproj, and one or more drafters ("drafter", "drafter_2", ... after merging)."""
    return ["file", "mmproj"] + sorted(k for k in spec if k == "drafter" or k.startswith("drafter_"))


def merge_models(pts, configs):
    """Aliases naming the same (repo, file, mmproj) become one model: one download, one interleaved block. Each
    config keeps its own alias's drafter (cfg["_drafter_role"] -> the merged entry's drafter / drafter_N)."""
    canon, merged = {}, {}
    for c in configs:
        al = c["model"]; sp = pts["models"][al]
        key = (sp["repo"], sp["file"], sp.get("mmproj"))
        ca = canon.setdefault(key, al)
        m = merged.setdefault(ca, {k: v for k, v in sp.items() if k != "drafter"})
        d = sp.get("drafter")
        role = None
        if d:
            have = {m[k]: k for k in m if k == "drafter" or k.startswith("drafter_")}
            role = have.get(d) or ("drafter" if "drafter" not in m else f"drafter_{len(have) + 1}")
            m[role] = d
        c["model_alias"] = al; c["model"] = ca; c["_drafter_role"] = role
    pts["models"] = {**pts["models"], **merged}
    return configs


def _run_models(a, pts, hw, bins, env, configs, order, out, skip_row, device_skip, t_start, scale):
    for alias in order:
        spec = pts["models"][alias]
        mcfgs = [c for c in configs if c["model"] == alias]
        n_owned0 = len(OWNED_FILES)
        if device_skip:
            for c in mcfgs:
                skip_row(c, device_skip)
            continue
        # ---- preflight from the header only: disk (incl. staging), RAM / cgroup, free VRAM, runtime budget
        if not a.dry_run:
            mdir = os.path.join(a.models_dir, spec["repo"].replace("/", "__"))
            own_path(mdir)
            os.makedirs(mdir, exist_ok=True)
            sizes, shard_sz = {}, {}
            for role in model_roles(spec):
                if spec.get(role):
                    for nm in shard_names(spec[role]):
                        shard_sz[nm] = remote_size(spec.get(role + "_repo", spec["repo"]), nm)
                    sizes[role] = sum(shard_sz[nm] for nm in shard_names(spec[role]))
            # user download guard: no single weights file over HF_DOWNLOAD_MAX_GB (default 20) unless already on disk
            cap = float(os.environ.get("HF_DOWNLOAD_MAX_GB", "20")) * 1e9
            big = {nm: sz for nm, sz in shard_sz.items() if sz > cap
                   and not (os.path.exists(os.path.join(mdir, nm)) and os.path.getsize(os.path.join(mdir, nm)) == sz)}
            if big:
                for c in mcfgs:
                    skip_row(c, f"download cap: {', '.join(f'{r} {sz / 1e9:.1f} GB' for r, sz in big.items())} > "
                                f"HF_DOWNLOAD_MAX_GB {cap / 1e9:.0f} (needs the user's OK)")
                continue
            hdrs = []
            for nm in shard_names(spec["file"]):
                hdr = fetch_header(spec["repo"], nm, shard_sz[nm], mdir)
                hdrs.append(hdr)
                if hdr not in OWNED_FILES:
                    OWNED_FILES.append(hdr)  # header sidecar is always this harness's own file
            pinfo = model_info(hdrs[0])
            to_dl = sum(sz for nm, sz in shard_sz.items()
                        if not (os.path.exists(os.path.join(mdir, nm)) and os.path.getsize(os.path.join(mdir, nm)) == sz))
            free_disk = shutil.disk_usage(a.models_dir).free
            mi = meminfo(); cg = cgroup_mem_avail()
            ram_avail = min(mi.get("MemAvailable", 0), cg) if cg is not None else mi.get("MemAvailable", 0)
            vf = vram_free_mib()
            vgate = vf is not None or bool(a.vram_budget_mib)  # unreadable device memory: llama.cpp's own failure is the gate
            vfree = (vf if vf is not None else a.vram_budget_mib) << 20
            pre = {"disk_free": free_disk, "to_download": to_dl, "ram_avail": ram_avail, "cgroup_avail": cg,
                   "vram_free": vfree, "file_bytes": sizes["file"]}
            if free_disk < to_dl * 1.05 + (2 << 30):
                for c in mcfgs:
                    skip_row(c, f"disk: need {to_dl * 1.05 / 1e9 + 2:.1f} GB incl. staging, {free_disk / 1e9:.1f} GB free", preflight=pre)
                continue
            if sizes["file"] > ram_avail + vfree and vgate and not spec.get("ssd_ok"):
                for c in mcfgs:
                    skip_row(c, f"fit: file {sizes['file'] / 1e9:.1f} GB > RAM avail + free VRAM {(ram_avail + vfree) / 1e9:.1f} GB", preflight=pre)
                continue
            keep_cfgs = []
            elapsed = time.time() - t_start
            budget_left = a.budget_min * 60 - elapsed - to_dl / (a.dl_mbps * 1e6) if a.budget_min else None
            for c in mcfgs:
                ot = build_overrides(c, pinfo)
                ngl_acc = c.get("ngl", 999) if isinstance(c.get("ngl", 999), int) else 999
                by = placement_bytes(pinfo, ot, ngl_acc, c.get("spec"))
                ctx = c.get("ctx") if isinstance(c.get("ctx"), int) else (c.get("depth", 0) + c["p"] + c["n"]) if not c.get("ctx") else 8192
                kvb = kv_bytes(pinfo, ctx, c["ctk"], c["ctv"])
                mm = sizes.get("mmproj", 0) if c.get("mmproj") == "gpu" else 0
                need_vram = by["w_gpu"] + (0 if c.get("nkvo") else kvb) + mm + (768 << 20)
                need_ram = by["w_cpu"] + (kvb if c.get("nkvo") else 0) + (1 << 30)
                if c.get("ngl") == "omit" or c.get("fit") == "on":
                    need_vram = min(need_vram, (c.get("fake_vram_mib") or (vfree >> 20)) << 20)
                    need_ram = by["w_cpu"] + by["w_gpu"] + (1 << 30) - need_vram + (1 << 30)
                fit_row = c.get("ngl") == "omit" or c.get("fit") == "on"  # the fitter sizes itself to free VRAM
                if vgate and not fit_row and need_vram > vfree - (256 << 20):
                    skip_row(c, f"vram: need {need_vram / 2**20:.0f} MiB, free {vfree / 2**20:.0f} MiB", preflight=pre); continue
                if need_ram > ram_avail and not (c.get("ssd_ok") or spec.get("ssd_ok")):
                    skip_row(c, f"ram: need {need_ram / 2**20:.0f} MiB, available {ram_avail / 2**20:.0f} MiB (cgroup {cg})", preflight=pre); continue
                est = estimate_seconds(c, by, hw, c.get("mode", "bench"), scale["v"]) * a.rounds
                # no preflight budget skip: the header-only estimate ran 2-3x long on the 11-26 T4 / L4 runs and
                # dropped every spill / cache arm of the last block; the per-run check (estimate x the scale learned
                # from this run's own walls) still stops the run before the wall budget
                c["_est_s"] = round(est / a.rounds / scale["v"], 1)  # unscaled single-round estimate
                keep_cfgs.append(c)
            if not keep_cfgs:
                log(f"{alias}: every arm skipped, nothing downloaded")
                cleanup_model(n_owned0, a.keep_models)
                continue
            mcfgs = keep_cfgs
        try:
            paths = {}
            try:
                _fetch_all(a, spec, alias, hw, paths)
            except Exception as e:  # one model's download failing must not end the run: skip its arms, go on
                for c in mcfgs:
                    skip_row(c, f"download: {type(e).__name__}: {str(e)[:300]}")
                continue
            info = model_info(paths["model"]) if os.path.exists(paths.get("model", "")) else None
            seq = []
            groups = [mcfgs]
            if a.interleave_by_point:  # points one after another, each point's arms interleaved over the rounds
                gk = {}
                for c in mcfgs:
                    gk.setdefault(c.get("point") or c["id"].rsplit("_", 1)[0], []).append(c)
                groups = list(gk.values())
            for g in groups:
                for rnd in range(a.rounds):
                    seq += [(rnd, c) for c in (g if rnd % 2 == 0 else g[::-1])]
            for rnd, cfg in seq:
                ot = build_overrides(cfg, info) if info else list(cfg.get("ot", []))
                threads = cfg.get("threads", a.threads) or 0
                cpus, cpu_busy_pre = pick_cpus(cfg.get("cpus", a.cpus), threads or 16, hw)
                if cpus and not threads:
                    threads = len(cpus)
                mode = cfg.get("mode", "bench")
                port = free_port()
                cmd = bench_cmd(bins, paths["model"], cfg, ot, threads) if mode == "bench" else \
                    server_cmd(bins, ({**paths, "drafter": paths.get(cfg["_drafter_role"]) if cfg["_drafter_role"] else None}
                                      if "_drafter_role" in cfg else paths), cfg, ot, threads, port)
                row = {"cpu_node": hw.get("_cpu_node_choice"), "id": cfg["id"], "round": rnd, "model": alias, "repo": spec["repo"], "file": spec["file"],
                       "mode": mode, "cfg": {k: v for k, v in cfg.items() if k not in ("model",)}, "ot": ot,
                       "threads": threads, "cpus": cpus, "cpu_busy_pre": cpu_busy_pre, "cmd": cmd,
                       "load1_pre": (getload() or 0), "ts": time.time(), "tag": a.tag,
                       "hw": {**{k: hw.get(k) for k in ("host", "gpu", "gpu_numa", "ram_total", "cpu_model", "nproc", "colab",
                                              "llama_devices", "prebuilt", "llama_version", "os")},
                              "host_probe": {k: v for k, v in (hw.get("host_probe") or {}).items() if k != "lscpu"}}}
                if info:
                    ctx = cfg.get("ctx") if isinstance(cfg.get("ctx"), int) else 8192 if cfg.get("ctx") else (cfg.get("depth", 0) + cfg["p"] + cfg["n"])
                    ngl_acc = cfg.get("ngl", 999)
                    if not isinstance(ngl_acc, int):  # "omit": --fit decides; accounting assumes all on GPU,
                        ngl_acc = 999                 # the real split is in server_log.buffers_mib
                        row["placement_by_fit"] = True
                    row["bytes"] = placement_bytes(info, ot, ngl_acc, cfg.get("spec"))
                    row["bytes"]["kv_total"] = kv_bytes(info, ctx, cfg["ctk"], cfg["ctv"])
                    depth = cfg.get("depth", 0) + cfg["p"] // 2 if mode == "server" else cfg.get("depth", 0)
                    row["bytes"]["kv_tok_at_depth"] = kv_bytes(info, depth + cfg["n"] // 2, cfg["ctk"], cfg["ctv"])
                    row["model_meta"] = {k: info[k] for k in ("arch", "n_layer", "nextn", "n_expert", "n_used",
                                                              "ctx_train", "file_bytes")}
                    row["model_meta"]["n_kv_layers"] = len(info["kv_layers"])
                if a.dry_run:
                    log("DRY", cfg["id"], " ".join(cmd)); continue
                if cfg.get("fake_vram_mib") and cfg.get("fit") == "on" and not any(
                        d.startswith("CUDA") for d in hw.get("llama_devices") or []):
                    skip_row(cfg, "shim: fake_vram_mib needs the CUDA cudaMemGetInfo shim (not on this backend)", rnd)
                    continue
                if cfg.get("rounds_max") is not None and rnd >= cfg["rounds_max"]:
                    continue
                if a.budget_min and cfg.get("_est_s"):
                    left = a.budget_min * 60 - (time.time() - t_start)
                    if cfg["_est_s"] * scale["v"] > left:
                        skip_row(cfg, f"budget: est {cfg['_est_s'] * scale['v'] / 60:.1f} min > {max(0, left) / 60:.1f} min left", rnd)
                        continue
                if a.max_load1:  # CPU-sensitive rows: wait for a quiet host, else run and flag the row
                    w0 = time.time()
                    while (getload() or 0) > a.max_load1 and time.time() - w0 < a.load_wait_s:
                        time.sleep(15)
                    row["load_wait_s"] = round(time.time() - w0, 1)
                    row["load1_pre"] = (getload() or 0)
                lp = os.path.join(a.logdir, f"{cfg['id']}_r{rnd}.log")
                log(f"run {cfg['id']} (round {rnd}, {mode}) threads={threads} cpus={cpus and len(cpus)}")
                try:
                    tmo = cfg.get("timeout", a.timeout)
                    cenv = env
                    if cfg.get("fake_vram_mib") and cfg.get("fit") == "on" and not (a.fake_shim and os.path.exists(a.fake_shim)):
                        a.fake_shim = build_shim(a.logdir)  # Colab: compile fakemem.c shipped beside this script
                    if cfg.get("fake_vram_mib") and cfg.get("fit") == "on":  # the shim only changes what --fit plans
                        if not (a.fake_shim and os.path.exists(a.fake_shim)):
                            raise RuntimeError("fake_vram_mib needs --fake-shim (cudaMemGetInfo LD_PRELOAD shim)")
                        cenv = {**env, "LD_PRELOAD": a.fake_shim, "FAKE_CUDA_FREE_MIB": str(cfg["fake_vram_mib"]),
                                "FAKE_CUDA_TOTAL_MIB": str(cfg["fake_vram_mib"])}
                    wait_fn = (lambda p, smp, cfg=cfg, port=port, tmo=tmo, rnd=rnd: server_measure({**cfg, "_round": rnd}, port, tmo, smp, p)) if mode == "server" else None
                    row["shim"] = {"on": "LD_PRELOAD" in cenv and cenv is not env,
                                   "fake_mib": cenv.get("FAKE_CUDA_FREE_MIB") if cenv is not env else None}
                    rc, so, tail, cpu_s, wall, mem, extra = run_proc(cmd, cenv, tmo, cpus, lp, wait_fn)
                    if rc == -9:
                        row["timeout_s"] = tmo
                    row.update({"rc": rc, "cpu_s": cpu_s, "wall_s": round(wall, 2), "mem": mem,
                                "load1_post": (getload() or 0), "load_mode": cfg.get("load_mode", "auto(mmap)")})
                    # an all-GPU placement burning many cores = some op (e.g. FA for this KV type) runs on the CPU
                    all_gpu = row.get("bytes", {}).get("tok_cpu", 1e9) < 1e6  # token_embd row only
                    if cpu_s and wall and all_gpu:
                        row["cpu_cores_busy"] = round(cpu_s / wall, 1)
                        row["cpu_fallback_suspect"] = cpu_s / wall > 4.0
                    if mode == "bench":
                        row["result"] = parse_bench(so)
                        if rc != 0 or not row["result"]:
                            row["error"] = (f"timeout after {tmo} s | " if rc == -9 else "") + tail[-1500:]
                    else:
                        row["result"] = extra
                        reps = extra["reps"] or ([extra["cold"]] if extra.get("cold") else [])
                        row["result"]["tg_med"] = median([x["tg_ts"] for x in reps])
                        row["result"]["pp_med"] = median([x["pp_ts"] for x in reps])
                        dn = sum((x["draft_n"] or 0) for x in reps); da = sum((x["draft_acc"] or 0) for x in reps)
                        row["result"]["draft_accept"] = (da / dn) if dn else None
                        row["server_log"] = parse_server_log(open(lp, errors="replace").read())
                        # authoritative check: an all-GPU tg graph has 2 splits; more = some op (FA for this KV
                        # type, e.g. iq4_nl) is scheduled on the CPU backend
                        sp = row["server_log"]["graph_splits"]
                        if sp and row.get("bytes", {}).get("tok_cpu", 1e9) < 1e6:
                            tg_splits = int(sp[-1].split("/")[-1])
                            row["tg_graph_splits"] = tg_splits
                            row["cpu_fallback"] = tg_splits > 2
                        dec_ms = sum((x.get("decode_ms") or 0) for x in reps + ([extra["cold"]] if extra.get("cold") else []))
                        pu = (mem or {}).get("proc_used_mib") or {}
                        bsum = row["server_log"]["device_buffers"]["sum_mib"]
                        # process overhead = per-process peak minus every logged device buffer (CUDA context,
                        # cuBLAS workspace, graphs, allocator slack): what the planner must reserve per host
                        row["overhead_mib"] = {k: (round(pu[k] - bsum, 1) if pu.get(k) is not None else None)
                                               for k in ("peak_after_load", "peak_fill", "peak_all")}
                        row["overhead_mib"]["buffers_sum_mib"] = bsum
                        up = row["server_log"]["moe_uploaded_mib"]
                        row["result"]["moe_upload_GBs_during_decode"] = round(up / 1024 / (dec_ms / 1000), 2) if up and dec_ms else None
                except Exception as e:  # keep going: the row records the failure
                    row["error"] = f"{type(e).__name__}: {e}"
                    try:
                        row["log_tail"] = open(lp, errors="replace").read()[-1500:]
                    except Exception:
                        pass
                if a.max_load1:
                    lm = max(row.get("load1_pre") or 0, row.get("load1_post") or 0, (row.get("mem") or {}).get("load1_max") or 0)
                    row["contended"] = lm > (a.contended_load1 or a.max_load1)
                    row["load1_gate"] = {"start_max": a.max_load1, "contended_over": a.contended_load1 or a.max_load1}
                if cfg.get("_est_s") and row.get("wall_s"):
                    row["est_s"] = round(cfg["_est_s"] * scale["v"], 1)
                    scale["v"] = min(5.0, max(scale["v"], row["wall_s"] / cfg["_est_s"]))
                out.write(json.dumps(row) + "\n"); out.flush()
                r = row.get("result") or {}
                if mode == "bench":
                    log(f"  -> pp {(r.get('pp') or {}).get('avg_ts')} tg {(r.get('tg') or {}).get('avg_ts')} "
                        f"wall {row.get('wall_s')} s cpu {row.get('cpu_s')} s {row.get('error', '')[:200]}")
                else:
                    log(f"  -> tg_med {r.get('tg_med')} pp_med {r.get('pp_med')} acc {r.get('draft_accept')} "
                        f"wall {row.get('wall_s')} s {row.get('error', '')[:300]}")
        finally:
            cleanup_model(n_owned0, a.keep_models)


def _fetch_all(a, spec, alias, hw, paths):
    for role in model_roles(spec):
        if not spec.get(role):
            continue
        if a.dry_run:
            paths["model" if role == "file" else role] = os.path.join(a.models_dir, spec["repo"].replace("/", "__"), spec[role])
            continue
        for nm in shard_names(spec[role])[::-1]:  # shard 1 last: its path is the one passed
            pth, new, _ = fetch(spec.get(role + "_repo", spec["repo"]), nm, a.models_dir, hw, alias)
        paths["model" if role == "file" else role] = pth


def cleanup_model(n_owned0, keep_models):
    """Delete what this run created for one model (frees disk before the next model)."""
    mine = OWNED_FILES[n_owned0:]
    if keep_models:
        for pth in [x for x in mine if x.endswith(".header")]:
            try:
                os.remove(pth); OWNED_FILES.remove(pth)
            except (OSError, ValueError):
                pass
        return
    del OWNED_FILES[n_owned0:]
    for pth in reversed(mine):
        try:
            if os.path.isdir(pth):
                shutil.rmtree(pth)
            elif os.path.exists(pth):
                os.remove(pth)
            log("deleted (run-owned)", pth)
        except OSError:
            pass


if __name__ == "__main__":
    main()
