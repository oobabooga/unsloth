"""ComfyUI, headless, one server per cell, driven through its /prompt API with the family graph from comfy_graphs/.

Cell options (all optional unless marked):
  family         (required, or cell["family"]) qwen-image-2.1 | z-image | flux.1 | sdxl-turbo | wan2.2-5b (+ aliases)
  files          (required) {"dit": path, "te": path | [paths], "vae": path, "checkpoint": path, "diffusers": dir};
                 symlinked into <out>/_comfy/models and passed with --extra-model-paths-config
  comfy          {"commit", "python", "torch", "index", "sage", "kitchen"} -> setup_comfyui.ensure; the env vars
                 DIFFUSION_BENCH_COMFY_PYTHON / DIFFUSION_BENCH_COMFY_DIR reuse an existing venv / clone
  weight_dtype   UNETLoader weight_dtype (default | fp8_e4m3fn | fp8_e4m3fn_fast | fp8_e5m2)
  compile        true | "inductor" | "cudagraphs": TorchCompileModel on the model
  highvram       --highvram (keep models resident; ComfyUI otherwise offloads between prompts)
  comfy_args     extra server flags, a string or a list, e.g. "--fast --use-sage-attention"
  easycache      EasyCache reuse_threshold (e.g. 0.2); easycache_start / easycache_end (0.15 / 0.95)
  sampler / scheduler / shift / cfg / flux_guidance   override the template's (cfg defaults to cell guidance)
  cache_bust     all (default) | sampler | none: see comfy_graphs.apply_nonce
  api_nodes      false (default): start with --disable-api-nodes so the server never calls out
  startup_timeout_s 600, render_timeout_s 3600, keep_raw false

Timing facts per render (Render.extra): exec_s (ComfyUI's own execution_start -> execution_success), step_s from
the sampler's tqdm line in the server log (the last complete N/N rate), easycache skipped steps, and whether the
sampler came from cache (a cache hit is a bug in the protocol, flagged, never silently timed). load() only starts
the server: ComfyUI loads model files lazily on the first prompt, so the model load lands in the warmup (cold_s).
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from .base import Backend, Render

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

IT_RE = re.compile(r"(\d+)/(\d+) \[[^\]]*?,\s*([\d.]+)(it/s|s/it)[,\]]")
EASY_RE = re.compile(r"EasyCache - skipped (\d+)/(\d+) steps")
# Server-log lines that prove an optimisation engaged; a flag ComfyUI ignores leaves no line.
ENGAGED = {"sage_attention": "Using sage attention", "pytorch_attention": "Using pytorch attention",
           "flash_attention": "Using Flash Attention", "xformers_attention": "Using xformers attention",
           "fp16_accumulation": "Enabled fp16 accumulation.", "cublas_ops": "Using cublas ops",
           "highvram": "Set vram state to: HIGH_VRAM", "pinned_memory": "Enabled pinned memory",
           "mixed_precision_ops": "Using mixed precision operations",
           # gfx1151 levers: Comfy Kitchen INT8 attention (--use-ck-attention), its unavailability, a runtime flash
           # fallback (logged per failing call), sub-quadratic default, and DynamicVRAM (comfy-aimdo) on / off.
           "kitchen_attention": "Using Comfy Kitchen attention",
           "kitchen_attention_unavailable": "Comfy Kitchen attention is unavailable",
           "flash_attention_failed": "Flash Attention failed",
           "sub_quadratic_attention": "Using sub quadratic optimization",
           "dynamic_vram": "DynamicVRAM support detected and enabled",
           "dynamic_vram_disabled": "DynamicVRAM support disabled"}
FACT_RE = {"device": re.compile(r"Device: (.+)"), "vram_state": re.compile(r"Set vram state to: (\S+)"),
           "comfyui_version": re.compile(r"ComfyUI version: (\S+)"), "torch": re.compile(r"pytorch version: (\S+)"),
           "comfy_kitchen": re.compile(r"comfy-kitchen version: (\S+)"),
           "native_ops": re.compile(r"Native ops: (.+)"), "model_type": re.compile(r"model_type (\S+)")}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ComfyUIError(RuntimeError):
    pass


class ComfyUIBackend(Backend):
    name = "comfyui"
    in_process = False

    def __init__(self, cell: dict, out: Path):
        super().__init__(cell, out)
        self.srv: Optional[subprocess.Popen] = None
        self.logf = None
        self.port: Optional[int] = None
        self.work = self.out / "_comfy"
        self.log_path = self.work / "server.log"
        self.renders = 0
        self.info: dict = {}

    # ------------------------------------------------------------------------------------------ http
    def api(self, path: str, payload: Optional[dict] = None, timeout: float = 30):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data = None if payload is None else json.dumps(payload).encode(),
                                     headers = {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout = timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors = "replace")
            raise ComfyUIError(f"ComfyUI {path} -> HTTP {exc.code}: {self._explain(body)}") from None

    @staticmethod
    def _explain(body: str) -> str:
        """The node errors of a rejected /prompt, flattened to one line a person can act on."""
        try:
            data = json.loads(body)
        except Exception:  # noqa: BLE001
            return body[:1500]
        parts = []
        err = data.get("error") or {}
        if isinstance(err, dict) and err.get("message"):
            parts.append(f"{err.get('message')} {err.get('details') or ''}".strip())
        for node, info in (data.get("node_errors") or {}).items():
            for e in info.get("errors", []):
                parts.append(f"node {node} ({info.get('class_type')}): {e.get('message')} {e.get('details') or ''}".strip())
        return "; ".join(parts)[:2000] or body[:1500]

    def log_text(self, start: int = 0) -> str:
        if self.logf:
            self.logf.flush()
        try:
            with open(self.log_path, "rb") as f:
                f.seek(start)
                return ANSI.sub("", f.read().decode(errors = "replace"))
        except FileNotFoundError:
            return ""

    def log_size(self) -> int:
        if self.logf:
            self.logf.flush()
        try:
            return self.log_path.stat().st_size
        except FileNotFoundError:
            return 0

    def tail(self, n: int = 40) -> str:
        lines = [ln for ln in self.log_text().replace("\r", "\n").splitlines() if ln.strip()]
        return "\n".join(lines[-n:])

    # ------------------------------------------------------------------------------------------ lifecycle
    def family(self) -> str:
        import comfy_graphs

        return comfy_graphs.canonical(self.opts.get("family") or self.cell.get("family") or "")

    def load(self) -> dict:
        import comfy_graphs
        import setup_comfyui

        fam = self.family()
        mod = comfy_graphs.module(fam)
        files = dict(self.opts.get("files") or {})
        if not files:
            raise ValueError(f"comfyui cell needs options.files ({', '.join(mod.FILES)})")
        unknown = set(files) - set(mod.FILES)
        if unknown:
            raise ValueError(f"comfyui {fam}: unknown file roles {sorted(unknown)}; expected {sorted(mod.FILES)}")
        # Symlink the cell's files first: a missing file fails here, before a server starts.
        self.work.mkdir(parents = True, exist_ok = True)
        wiring = {}
        for role, paths in files.items():
            wiring.setdefault(mod.FILES[role], [])
            wiring[mod.FILES[role]] += [paths] if isinstance(paths, str) else list(paths)
        yaml = setup_comfyui.wire_models(wiring, self.work / "models")
        self.names = {role: ([Path(p).name for p in v] if isinstance(v, list) else Path(v).name)
                      for role, v in files.items()}

        comfy = dict(self.opts.get("comfy") or {})
        self.info = setup_comfyui.ensure(comfy_commit = comfy.get("commit"), python = comfy.get("python"),
                                         torch = comfy.get("torch"), index = comfy.get("index"),
                                         kitchen = comfy.get("kitchen", True), sage = comfy.get("sage", False))
        comfy_dir, py = Path(self.info["comfy_dir"]), self.info["python"]
        self.port = int(self.opts.get("port") or free_port())
        dirs = {k: self.work / k for k in ("output", "temp", "input", "user")}
        for d in dirs.values():
            d.mkdir(parents = True, exist_ok = True)
        cmd = [py, "-u", str(comfy_dir / "main.py"), "--listen", "127.0.0.1", "--port", str(self.port),
               "--disable-auto-launch", "--output-directory", str(dirs["output"]), "--temp-directory", str(dirs["temp"]),
               "--input-directory", str(dirs["input"]), "--user-directory", str(dirs["user"]),
               "--extra-model-paths-config", str(yaml)]
        if not self.opts.get("api_nodes"):
            cmd.append("--disable-api-nodes")
        if self.opts.get("highvram"):
            cmd.append("--highvram")
        extra = self.opts.get("comfy_args") or []
        cmd += extra.split() if isinstance(extra, str) else [str(a) for a in extra]
        self.cmd = cmd
        self.logf = open(self.log_path, "wb")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.srv = subprocess.Popen(cmd, cwd = str(comfy_dir), stdout = self.logf, stderr = subprocess.STDOUT,
                                    stdin = subprocess.DEVNULL, env = env, start_new_session = True)
        deadline = time.time() + float(self.opts.get("startup_timeout_s", 600))
        while True:
            if self.srv.poll() is not None:
                raise ComfyUIError(f"ComfyUI server exited with code {self.srv.returncode} during startup "
                                   f"(log {self.log_path}):\n{self.tail(25)}")
            try:
                stats = self.api("/system_stats", timeout = 5)
                break
            except Exception:  # noqa: BLE001 - not up yet
                if time.time() > deadline:
                    raise ComfyUIError(f"ComfyUI did not answer /system_stats within the startup timeout:\n{self.tail(25)}")
                time.sleep(0.5)
        self._check_nodes(mod)
        text = self.log_text()
        return {"backend": "comfyui", "family": fam, "comfy_commit": self.info.get("commit"),
                "comfy_dir": str(comfy_dir), "python": py, "port": self.port, "files": self.names,
                "cmd": cmd[3:], "weight_dtype": self.opts.get("weight_dtype", "default"),
                "compile": self.opts.get("compile") or False, "easycache": self.opts.get("easycache"),
                "cache_bust": self.opts.get("cache_bust", "all"), "models_load_lazily": True,
                "engaged": {k: v in text for k, v in ENGAGED.items()}, "facts": self._facts(text),
                "system": (stats.get("system") or {}), "devices": [d.get("name") for d in stats.get("devices", [])]}

    def _check_nodes(self, mod) -> None:
        """Every node the family graph uses exists in this ComfyUI, and every file is one its loader lists: a
        missing node (older commit) or an unlisted file becomes one clear error instead of a node_errors blob."""
        import comfy_graphs

        probe = comfy_graphs.GraphParams(prompt = "probe", seed = 0, steps = 1, width = 512, height = 512,
                                         files = self.names, frames = 5, compile = self.opts.get("compile"),
                                         easycache = self.opts.get("easycache"))
        try:
            graph = mod.build(probe)
        except Exception:  # noqa: BLE001 - a bad option surfaces again at render with the real params
            return
        info = self.api("/object_info", timeout = 120)
        missing = sorted({n["class_type"] for n in graph.values() if n["class_type"] not in info})
        if missing:
            raise ComfyUIError(f"this ComfyUI ({self.info.get('commit', '?')[:10]}) has no node(s) {missing}; "
                               f"use a newer comfy.commit")
        for nid, node in graph.items():
            spec = (info[node["class_type"]].get("input") or {}).get("required") or {}
            for key, val in node["inputs"].items():
                if not (isinstance(val, str) and key in spec and
                        (key.endswith("_name") or key.startswith("clip_name") or key == "model_path")):
                    continue
                choices = spec[key][0]
                if choices == "COMBO" and len(spec[key]) > 1:
                    choices = (spec[key][1] or {}).get("options")
                if isinstance(choices, list) and val not in choices:
                    raise ComfyUIError(f"{node['class_type']}.{key}={val!r} is not in ComfyUI's list "
                                       f"({len(choices)} entries, e.g. {choices[:5]})")

    @staticmethod
    def _facts(text: str) -> dict:
        out = {}
        for key, rx in FACT_RE.items():
            m = rx.findall(text)
            if m:
                out[key] = m[-1].strip()
        return out

    def render(self, row: dict, steps: int) -> Render:
        import comfy_graphs

        if self.srv is None or self.srv.poll() is not None:
            raise ComfyUIError(f"ComfyUI server is not running (exit {self.srv and self.srv.returncode}):\n{self.tail(25)}")
        fam = self.family()
        mod = comfy_graphs.module(fam)
        self.renders += 1
        o, c = self.opts, self.cell
        p = comfy_graphs.GraphParams(
            prompt = row["prompt"], seed = int(row["seed"]), steps = int(steps), width = int(c["width"]),
            height = int(c["height"]), negative = c.get("negative_prompt"), cfg = o.get("cfg", c.get("guidance")),
            sampler = o.get("sampler"), scheduler = o.get("scheduler"), shift = o.get("shift"),
            guidance = o.get("flux_guidance"), frames = c.get("frames"), files = self.names,
            weight_dtype = o.get("weight_dtype", "default"), compile = o.get("compile"), easycache = o.get("easycache"),
            easycache_start = float(o.get("easycache_start", 0.15)), easycache_end = float(o.get("easycache_end", 0.95)),
            prefix = f"r{self.renders:04d}_{row.get('id', 'x')}", nonce = self.renders,
            nonce_nodes = o.get("cache_bust", "all"), extra = dict(o.get("graph_extra") or {}))
        graph = comfy_graphs.build(fam, p)
        start = self.log_size()
        resp = self.api("/prompt", {"prompt": graph, "client_id": self.client_id})
        pid = resp["prompt_id"]
        deadline = time.time() + float(o.get("render_timeout_s", 3600))
        while True:
            h = self.api(f"/history/{pid}")
            if pid in h and (h[pid].get("status") or {}).get("completed") is not None:
                break
            if self.srv.poll() is not None:
                raise ComfyUIError(f"ComfyUI server died mid-render (exit {self.srv.returncode}):\n{self.tail(30)}")
            if time.time() > deadline:
                self.api("/interrupt", {})
                raise ComfyUIError(f"ComfyUI render exceeded {o.get('render_timeout_s', 3600)} s")
            time.sleep(0.05)
        st = h[pid]["status"]
        if st.get("status_str") != "success":
            msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            detail = msgs[-1][1] if msgs else st.get("messages")
            if isinstance(detail, dict):
                detail = f"{detail.get('node_type')}: {detail.get('exception_type')}: {detail.get('exception_message')}"
            raise ComfyUIError(f"ComfyUI job failed: {str(detail)[:1500]}")
        chunk = self.log_text(start)
        rates = [m for m in IT_RE.findall(chunk) if m[0] == m[1]]
        step_s = None
        if rates:
            v, unit = float(rates[-1][2]), rates[-1][3]
            step_s = round(1.0 / v if unit == "it/s" else v, 4)
        msgs = {m[0]: m[1] for m in st.get("messages", [])}
        exec_s = None
        if "execution_success" in msgs and "execution_start" in msgs:
            exec_s = round((msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000, 3)
        cached = (msgs.get("execution_cached") or {}).get("nodes") or []
        files = [(i.get("subfolder") or "", i["filename"]) for i in (h[pid]["outputs"].get(comfy_graphs.SAVE) or {})
                 .get("images", [])]
        if not files:
            raise ComfyUIError(f"ComfyUI finished but the save node wrote nothing (cached nodes: {cached})")
        from PIL import Image

        images = []
        for sub, name in files:
            path = self.work / "output" / sub / name
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
            if not o.get("keep_raw"):
                path.unlink(missing_ok = True)
        skipped = EASY_RE.findall(chunk)
        extra = {"exec_s": exec_s, "sampler_cached": "sampler" in cached, "cached_nodes": cached}
        if skipped:
            extra["easycache_skipped"] = f"{skipped[-1][0]}/{skipped[-1][1]}"
        if getattr(mod, "VIDEO", False):
            return Render(frames = images, fps = int(c.get("fps") or 24), step_s = step_s, extra = extra)
        return Render(image = images[0], step_s = step_s, extra = extra)

    @property
    def client_id(self) -> str:
        if not hasattr(self, "_cid"):
            self._cid = str(uuid.uuid4())
        return self._cid

    def status(self) -> dict:
        text = self.log_text()
        return {"engaged": {k: v in text for k, v in ENGAGED.items()}, "facts": self._facts(text),
                "server_alive": bool(self.srv and self.srv.poll() is None)}

    def close(self) -> None:
        if self.srv is not None and self.srv.poll() is None:
            try:
                os.killpg(self.srv.pid, signal.SIGTERM)  # start_new_session: pgid == our child's pid
                self.srv.wait(timeout = 30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(self.srv.pid, signal.SIGKILL)
                    self.srv.wait(timeout = 10)
                except Exception:  # noqa: BLE001
                    pass
        if self.logf:
            self.logf.close()
            self.logf = None

    def trees(self) -> dict:
        return {"comfyui": self.info.get("comfy_dir")} if self.info.get("comfy_dir") else {}

    def sync(self) -> None:  # render() returns only after ComfyUI reports the prompt finished
        pass

    def reset_peak(self) -> None:
        pass
