#!/usr/bin/env python3
"""The home cluster link (#14274) across two physical machines: the real `unsloth` CLI and Studio's planner,
master on one box, worker on another, over the lab LAN.

  worker    --repo R --bins base=DIR,coal=DIR --out-dir D [--window-s 2700] [--local]
            writes D/self.json (addresses, a control port, a per-run nonce) and obeys the master's
            nonce-authenticated commands (join, start/stop the worker, done) until done or the window ends
  master    --repo R --bins ... --gguf M --out-dir D [--peer-dir P | GitHub artifacts] [--local]
            pairs, joins the worker, checks list / LAN discovery, plans with Studio, serves through the
            relay, compares greedy text with a same-box direct RPC run, measures base vs coalesced
            builds, moves the worker's port (discovery must find it), unpaired client, remove
  summarize --in DIR --out DIR      VERDICT.md / verdict.json from the master and worker results

No raw ggml-rpc-server is ever reachable from the LAN: the worker's rpc-server stays on loopback behind
the agent, and the direct-RPC reference runs on the master box over loopback only. The control channel
carries only commands with an HMAC over a per-run nonce (published in the run's own artifact); the
pairing token it carries is single use, expires in minutes, and only adds a worker to this run's
throwaway cluster."""

import argparse
import asyncio
import hashlib
import hmac
import io
import ipaddress
import json
import os
import re
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lanprobe"))
import lanprobe as lp  # noqa: E402  (interfaces, lan_ipv4, all_macs, read_json, write_json)

ENC = "utf-8"
VIRTUAL = ("docker", "veth", "br-", "virbr", "lo", "tailscale", "zt", "wg")
ART_PREFIX = "hl-worker-self-"


def sign(nonce, body):
    return hmac.new(bytes.fromhex(nonce), json.dumps(body, sort_keys = True).encode(ENC), hashlib.sha256).hexdigest()


def free_port(host = "127.0.0.1"):
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def host_identity(info):
    """Hostname and the MACs of physical interfaces (docker / veth bridges repeat across boxes)."""
    macs = sorted({(i.get("mac") or "").lower() for i in info["interfaces"]
                   if i.get("mac") and not str(i.get("name", "")).startswith(VIRTUAL)} - {"", "00:00:00:00:00:00"})
    return {"hostname": socket.gethostname(), "runner": os.environ.get("RUNNER_NAME", ""), "macs": macs}


def lan_nets(info, local):
    """[(ip, network)] for physical IPv4 interfaces; loopback too in --local mode."""
    out = []
    for i in info["interfaces"]:
        name = str(i.get("name", ""))
        for ip, prefix in i["ipv4"]:
            a = ipaddress.ip_address(ip)
            if a.is_loopback and local:
                out.append((ip, str(ipaddress.ip_network(f"{ip}/8", strict = False))))
            elif not a.is_loopback and not a.is_link_local and not name.startswith(VIRTUAL):
                out.append((ip, str(ipaddress.ip_network(f"{ip}/{prefix or 24}", strict = False))))
    return out


def cli_cmd():
    return [sys.executable, "-c", "import sys; sys.argv[0] = 'unsloth'; from unsloth_cli import app; app()"]


def base_env(repo, state, run_dir, bin_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "studio" / "backend")])
    env["UNSLOTH_CLUSTER_STATE_DIR"] = str(state)
    env["UNSLOTH_CLUSTER_RUN_DIR"] = str(run_dir)
    env["LLAMA_SERVER_PATH"] = str(Path(bin_dir) / "llama-server")
    env.pop("UNSLOTH_CLUSTER_NODES", None)
    return env


def parse_bins(text):
    return {k: Path(v).resolve() for k, v in (x.split("=", 1) for x in text.split(",") if x)}


def kill_group(p, wait = 15):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(wait)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ------------------------------------------------------------------ worker

class Worker:
    def __init__(self, a):
        self.a = a
        self.repo = Path(a.repo).resolve()
        self.bins = parse_bins(a.bins)
        self.out = Path(a.out_dir)
        self.out.mkdir(parents = True, exist_ok = True)
        self.state, self.run_dir = self.out / "state_worker", self.out / "run_worker"
        self.state.mkdir(exist_ok = True)
        self.run_dir.mkdir(exist_ok = True)
        self.proc, self.log_path, self.events, self.done = None, None, [], False
        self.nonce = secrets.token_hex(16)

    def env(self, arm):
        return base_env(self.repo, self.state, self.run_dir, self.bins[arm])

    def ev(self, **kw):
        kw["t"] = round(time.time(), 3)
        self.events.append(kw)
        print("worker:", json.dumps(kw, default = str)[:400], flush = True)

    def stop_worker(self):
        kill_group(self.proc)
        self.proc = None

    def handle(self, cmd, args):
        if cmd == "ping":
            return {"ok": True}
        if cmd == "free_ports":
            return {"ok": True, "ports": [free_port("0.0.0.0") for _ in range(int(args.get("n", 2)))]}
        if cmd == "join":
            token = args["token"]
            r = subprocess.run(cli_cmd() + ["cluster", "join", "--token-stdin", "--name", args["name"],
                                            "--port", str(args["port"])],
                               input = token + "\n", capture_output = True, text = True, timeout = 120,
                               env = self.env("base"), cwd = str(self.out))
            text = (r.stdout + r.stderr).replace(token, "<token>")
            self.ev(cmd = "join", rc = r.returncode, out = text[-600:])
            return {"ok": r.returncode == 0, "rc": r.returncode, "out": text[-1500:], "leaked": token in r.stdout + r.stderr}
        if cmd == "start_worker":
            self.stop_worker()
            arm, port = args["arm"], int(args["port"])
            self.log_path = self.out / f"worker_{arm}_{port}_{int(time.time())}.log"
            self.proc = subprocess.Popen(cli_cmd() + ["worker", "--port", str(port), "--rpc-port", str(free_port())],
                                         env = self.env(arm), stdout = open(self.log_path, "w"), stderr = subprocess.STDOUT,
                                         cwd = str(self.out), start_new_session = True)
            up, end = False, time.time() + 180
            while time.time() < end:
                text = self.log_path.read_text(errors = "replace")
                if "serving on port" in text or self.proc.poll() is not None:
                    up = "serving on port" in text and self.proc.poll() is None
                    break
                time.sleep(0.3)
            tail = self.log_path.read_text(errors = "replace")[-1500:]
            self.ev(cmd = "start_worker", arm = arm, port = port, up = up)
            return {"ok": up, "tail": tail}
        if cmd == "stop_worker":
            self.stop_worker()
            return {"ok": True}
        if cmd == "tail":
            return {"ok": True, "tail": self.log_path.read_text(errors = "replace")[-3000:] if self.log_path else ""}
        if cmd == "done":
            self.stop_worker()
            self.done = True
            return {"ok": True}
        return {"ok": False, "error": "unknown command"}

    def serve(self):
        info = lp.interfaces()
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", 0))
        srv.listen(8)
        srv.settimeout(1.0)
        me = {"role": "worker", **host_identity(info), "nets": lan_nets(info, self.a.local),
              "control_port": srv.getsockname()[1], "nonce": self.nonce, "interfaces": info["interfaces"]}
        lp.write_json(str(self.out / "self.json"), me)
        print("worker control on", me["control_port"], me["nets"], flush = True)
        deadline = time.time() + self.a.window_s
        try:
            while not self.done and time.time() < deadline and not (self.out / "STOP").exists():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(600)
                    try:
                        f = conn.makefile("rwb")
                        msg = json.loads(f.readline().decode(ENC))
                        body = {"cmd": msg["cmd"], "args": msg.get("args", {}), "ts": msg["ts"]}
                        if not hmac.compare_digest(sign(self.nonce, body), msg.get("mac", "")) or abs(time.time() - body["ts"]) > 600:
                            self.ev(cmd = "rejected", peer = addr[0])
                            continue
                        # once a master is working, only a long silence ends the run (the job timeout is the cap)
                        deadline = max(deadline, time.time() + 1800)
                        reply = self.handle(body["cmd"], body["args"])
                        f.write((json.dumps(reply, default = str) + "\n").encode(ENC))
                        f.flush()
                    except Exception as exc:
                        self.ev(cmd = "error", error = repr(exc))
        finally:
            self.stop_worker()
            lp.write_json(str(self.out / "worker_result.json"), {"self": me, "events": self.events, "done": self.done,
                                                                  "timed_out": not self.done})
        print("worker exit, done =", self.done, flush = True)
        return 0 if self.done else 2


def cmd_worker(a):
    return Worker(a).serve()


# ------------------------------------------------------------------ master

class Control:
    def __init__(self, host, port, nonce):
        self.host, self.port, self.nonce = host, port, nonce

    def __call__(self, cmd, timeout = 300, **args):
        body = {"cmd": cmd, "args": args, "ts": time.time()}
        with socket.create_connection((self.host, self.port), timeout = 10) as s:
            s.settimeout(timeout)
            f = s.makefile("rwb")
            f.write((json.dumps({**body, "mac": sign(self.nonce, body)}) + "\n").encode(ENC))
            f.flush()
            return json.loads(f.readline().decode(ENC) or "{}")


def api(path, token, raw = False):
    url = os.environ.get("GITHUB_API_URL", "https://api.github.com") + path
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout = 30) as r:
        body = r.read()
    return body if raw else json.loads(body.decode(ENC))


def wait_peer(a):
    end = time.time() + a.wait_s
    while time.time() < end:
        try:
            if a.peer_dir:
                p = Path(a.peer_dir) / "self.json"
                if p.exists():
                    return lp.read_json(str(p))
            else:
                token = os.environ["GH_API_TOKEN"]
                repo, run_id = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
                arts = api(f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", token).get("artifacts", [])
                mine = [x for x in arts if x["name"].startswith(ART_PREFIX) and not x.get("expired")]
                if mine:
                    best = max(mine, key = lambda x: int(x["name"][len(ART_PREFIX):] or 0) if x["name"][len(ART_PREFIX):].isdigit() else 0)
                    blob = api(f"/repos/{repo}/actions/artifacts/{best['id']}/zip", token, raw = True)
                    with zipfile.ZipFile(io.BytesIO(blob)) as z:
                        return json.loads(z.read("self.json").decode(ENC))
        except Exception as exc:
            print("waiting for worker:", repr(exc)[:200], flush = True)
        time.sleep(5)
    return None


class Master:
    def __init__(self, a):
        self.a = a
        self.repo = Path(a.repo).resolve()
        self.bins = parse_bins(a.bins)
        self.out = Path(a.out_dir)
        self.out.mkdir(parents = True, exist_ok = True)
        self.state, self.run_dir = self.out / "state_master", self.out / "run_master"
        self.state.mkdir(exist_ok = True)
        self.run_dir.mkdir(exist_ok = True)
        self.R = {"steps": [], "perf": [], "texts": {}}
        self.procs = []

    def step(self, name, ok, detail = ""):
        self.R["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[-1500:]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {'' if ok else str(detail)[-400:]}", flush = True)
        return ok

    def env(self, arm):
        return base_env(self.repo, self.state, self.run_dir, self.bins[arm])

    def cli(self, *args, stdin = None, timeout = 120, arm = "base"):
        return subprocess.run(cli_cmd() + list(args), env = self.env(arm), input = stdin, capture_output = True,
                              text = True, timeout = timeout, cwd = str(self.out))

    def use_arm(self, arm):
        os.environ.update(self.env(arm))

    def plan(self, arm):
        self.use_arm(arm)
        self.cs.reset_for_tests()
        req = SimpleNamespace(model_path = self.a.gguf, gguf_variant = None, llama_extra_args = None,
                              max_seq_length = 512, cache_type_kv = None, disable_vision = False)
        out = asyncio.run(self.ss.before_load(req, 4))
        status = self.ss.status()
        args = list(out.llama_extra_args or [])
        rpc = args[args.index("--rpc") + 1] if "--rpc" in args else ""
        return args, [e for e in rpc.split(",") if e], status

    def llama(self, arm, args, env, tag):
        port = free_port()
        log = self.out / f"llama_{tag}.log"
        p = subprocess.Popen([str(self.bins[arm] / "llama-server"), "-m", self.a.gguf, "--port", str(port), "-c", "512",
                              "-ngl", "99", "-t", str(self.a.threads), "-np", "1", "--no-warmup", "-lv", "4"] + args,
                             env = env, stdout = open(log, "w"), stderr = subprocess.STDOUT, start_new_session = True)
        self.procs.append(p)
        end = time.time() + 300
        while time.time() < end and p.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 2) as r:
                    if r.status == 200:
                        return p, port, log
            except Exception:
                time.sleep(0.5)
        return p, None, log

    def ask(self, port, n, timeout = 600):
        body = json.dumps({"prompt": "The capital of France is", "n_predict": n, "temperature": 0,
                           "cache_prompt": False, "ignore_eos": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return json.loads(r.read())

    def complete(self, arm, args, env, tag):
        p, port, log = self.llama(arm, args, env, tag)
        if port is None:
            kill_group(p)
            return None, [], log.read_text(errors = "replace")
        texts, per_tok = [], []
        try:
            for _ in range(1 + self.a.repeats):
                o = self.ask(port, self.a.n_predict)
                texts.append(o["content"])
                per_tok.append(o["timings"]["predicted_per_token_ms"])
        finally:
            kill_group(p)
        return texts[0], per_tok[1:], log.read_text(errors = "replace")

    def linked_env(self, args, arm):
        rpc = args[args.index("--rpc") + 1] if "--rpc" in args else ""
        return {**os.environ, **self.cs.secure_rpc_env([str(self.bins[arm] / "llama-server"), "--rpc", rpc])}

    def direct(self, arm, args, endpoints, tag):
        dp = free_port()
        rp = subprocess.Popen([str(self.bins[arm] / "ggml-rpc-server"), "-H", "127.0.0.1", "-p", str(dp)],
                              stdout = open(self.out / f"direct_rpc_{tag}.log", "w"), stderr = subprocess.STDOUT,
                              start_new_session = True, env = {**os.environ, "GGML_RPC_NO_RDMA": "1"})
        self.procs.append(rp)
        for _ in range(300):
            try:
                socket.create_connection(("127.0.0.1", dp), timeout = 1).close()
                break
            except OSError:
                time.sleep(0.2)
        rpc = ",".join(endpoints)
        dargs = [x if x != rpc else f"127.0.0.1:{dp}" for x in args]
        try:
            return self.complete(arm, dargs, {**os.environ, "GGML_RPC_NO_RDMA": "1"}, tag)
        finally:
            kill_group(rp)

    def run(self):
        a, R = self.a, self.R
        info = lp.interfaces()
        me = host_identity(info)
        R["master"] = {**me, "nets": lan_nets(info, a.local)}
        peer = wait_peer(a)
        if peer is None:
            R["verdict"] = "NOT_CONCURRENT"
            self.step("worker published its control address", False, "no worker artifact in time")
            return 3
        self.peer = peer
        R["worker"] = {k: peer.get(k) for k in ("hostname", "runner", "macs", "nets", "control_port")}
        same = peer.get("hostname") == me["hostname"] or bool(set(peer.get("macs") or []) & set(me["macs"]))
        R["same_machine"] = same
        pair = [(m, w) for m, mn in lan_nets(info, a.local) for w, wn in peer.get("nets", []) if mn == wn and m != w or (a.local and m == w)]
        if not pair:
            self.step("master and worker share a LAN subnet", False, json.dumps([R["master"]["nets"], peer.get("nets")]))
            R["verdict"] = "NO_SHARED_SUBNET"
            return 3
        m_ip, w_ip = pair[0]
        R["addresses"] = {"master": m_ip, "worker": w_ip}
        ctl = Control(w_ip, peer["control_port"], peer["nonce"])
        try:
            rtts = []
            for _ in range(20):
                t = time.perf_counter()
                ctl("ping", timeout = 10)
                rtts.append((time.perf_counter() - t) * 1e6)
            R["control_rtt_us"] = {"median": round(statistics.median(rtts), 1), "min": round(min(rtts), 1)}
            if same and not a.allow_same_machine:
                R["verdict"] = "SAME_MACHINE"
                self.step("master and worker are different machines", False, f"{me} vs {peer.get('hostname')}")
                return 3
            self.step("master and worker are different machines", not same or a.allow_same_machine,
                      f"{me['hostname']} / {peer.get('hostname')}, {m_ip} -> {w_ip}")
            return self.flow(ctl, m_ip, w_ip)
        finally:
            try:
                ctl("done", timeout = 60)
            except Exception as exc:
                print("done not delivered:", repr(exc), flush = True)

    def flow(self, ctl, m_ip, w_ip):
        a, R = self.a, self.R
        name = "box-w"
        # agent ports that are free on the worker box (the paired one, then the one only discovery can find)
        p1, p2 = ctl("free_ports", n = 2)["ports"]
        # 1. pairing through the real CLI, token handed to the worker over the control channel
        pport = free_port("0.0.0.0")
        plog = self.out / "pair.log"
        pairer = subprocess.Popen(cli_cmd() + ["cluster", "pair", "--ttl", "5m", "--addr", m_ip, "--port", str(pport)],
                                  env = self.env("base"), stdout = open(plog, "w"), stderr = subprocess.STDOUT,
                                  cwd = str(self.out), start_new_session = True)
        self.procs.append(pairer)
        token = None
        for _ in range(300):
            m = re.search(r"^\s+(unsloth-pair-1\.\S+)\s*$", plog.read_text(), re.M)
            if m:
                token = m.group(1)
                break
            time.sleep(0.1)
        if not self.step("cluster pair prints a one-time token", token is not None, plog.read_text()):
            return 1
        j = ctl("join", token = token, name = name, port = p1)
        self.step("cluster join on the worker box (token over stdin, never echoed)", j.get("ok") and not j.get("leaked"), j.get("out", ""))
        try:
            pairer.wait(60)
        except subprocess.TimeoutExpired:
            pass
        ptxt = plog.read_text()
        self.step("cluster pair on the master reports the worker", f"Paired {name}" in ptxt, ptxt.replace(token, "<token>"))
        # 2. the worker serves (base build first)
        s = ctl("start_worker", arm = "base", port = p1)
        if not self.step("unsloth worker serves on the worker box (rpc-server on its loopback)", s.get("ok"), s.get("tail", "")):
            return 1
        time.sleep(6.5)
        lst = self.cli("cluster", "list")
        R["cluster_list"] = lst.stdout
        self.step("cluster list shows the worker ready", lst.returncode == 0 and name in lst.stdout and " ready" in lst.stdout, lst.stdout + lst.stderr)
        disc = self.cli("cluster", "discover", timeout = 60)
        R["cluster_discover"] = disc.stdout
        worker_ips = {ip for ip, _ in self.peer.get("nets", [])}
        heard = [l for l in disc.stdout.splitlines() if l.rstrip().endswith(f"paired as {name}")]
        self.step("cluster discover hears the worker over LAN multicast",
                  disc.returncode == 0 and any(l.split()[1].rsplit(":", 1)[0] in worker_ips for l in heard if len(l.split()) > 1),
                  disc.stdout + disc.stderr)
        # 3. Studio's planner in this process
        sys.path[:0] = [str(self.repo / "studio" / "backend"), str(self.repo)]
        self.use_arm("base")
        from core.inference import cluster_serving as cs
        from core.inference import spark_serving as ss
        self.cs, self.ss = cs, ss
        args, eps, status = self.plan("base")
        R["plan"] = {"args": args, "status": {k: status.get(k) for k in ("topology", "nodes", "reason")}}
        nodes = status.get("nodes") or []
        self.step("Studio plans the model onto the remote worker through a loopback relay",
                  len(eps) == 1 and eps[0].startswith("127.0.0.1:") and any(str(n.get("host")) == w_ip for n in nodes),
                  json.dumps(R["plan"], default = str))
        if len(eps) != 1:
            return 1
        # 4. same-box direct reference (loopback rpc-server on the master box only)
        ref, ref_tok, _ = self.direct("base", args, eps, "ref")
        R["texts"]["direct_base"] = ref
        # 5. arms: relayed over the LAN, base vs coalesced, alternating; plus same-box direct per arm
        order = [("base", "coal")[i % 2] for i in range(2 * a.rounds)]
        if "coal" not in self.bins:
            order = ["base"] * a.rounds
        served_all, equal_all = True, True
        for i, arm in enumerate(order):
            if arm != "base" or i:
                s = ctl("start_worker", arm = arm, port = p1)
                if not s.get("ok"):
                    served_all = False
                    self.step(f"worker restarts on the {arm} build", False, s.get("tail", ""))
                    continue
            args, eps, _ = self.plan(arm)
            text, tok, log = self.complete(arm, args, self.linked_env(args, arm), f"linked_{arm}_{i}")
            buf = bool(eps) and (f"[{eps[0]}] model buffer size" in log or re.search(rf"RPC\d*\[{re.escape(eps[0])}\].*buffer size", log))
            dtext, dtok, _ = self.direct(arm, args, eps, f"direct_{arm}_{i}")
            served_all &= bool(text) and bool(buf)
            equal_all &= text == ref and dtext == ref
            R["perf"].append({"round": i, "arm": arm, "linked_ms_per_token": tok, "direct_same_box_ms_per_token": dtok,
                              "text_equal": text == ref, "direct_text_equal": dtext == ref})
            R["texts"][f"linked_{arm}_{i}"] = text
            print("ROUND", json.dumps(R["perf"][-1]), flush = True)
        self.step("llama-server on the master serves through the relay, model buffer on the remote worker (every round)", served_all,
                  json.dumps(R["perf"])[-1200:])
        self.step("greedy text through the LAN link equals same-box direct RPC (every round, both builds)", equal_all,
                  json.dumps({k: v for k, v in R["texts"].items()}, ensure_ascii = False)[-1200:])
        R["perf_summary"] = summarize_perf(R["perf"])
        # 6. the worker moves to another port: only LAN discovery can find it
        s = ctl("start_worker", arm = "base", port = p2)
        moved = False
        if s.get("ok"):
            time.sleep(6.5)
            args, eps, status = self.plan("base")
            nodes = status.get("nodes") or []
            found = any(int(n.get("rpc_port") or 0) == p2 for n in nodes)
            text, _, _ = self.complete("base", args, self.linked_env(args, "base"), "moved") if eps else (None, [], "")
            moved = found and text == ref
            R["moved"] = {"port": p2, "nodes": nodes, "text_equal": text == ref}
        self.step("worker restarted on another port is found through LAN discovery and serves", moved, json.dumps(R.get("moved"), default = str))
        # 7. a client with no cluster certificate gets nothing from the worker agent
        import ssl
        from studio.cluster_link import tls

        async def unpaired():
            r, w = await asyncio.open_connection(w_ip, p2, ssl = tls.bootstrap_client_context())
            w.write(b"\x00\x00\x00\x13" + b'{"type":"open_rpc"}')
            await w.drain()
            return await asyncio.wait_for(r.read(64), 5)

        try:
            got = asyncio.run(unpaired())
        except (ssl.SSLError, ConnectionError, asyncio.IncompleteReadError, OSError, asyncio.TimeoutError):
            got = b""
        self.step("a client without a cluster certificate gets nothing from the worker", got == b"", repr(got))
        # 8. remove while a generation is running: the live session is cut, the next plan has no worker
        args, eps, _ = self.plan("base")
        cut, detail = False, ""
        if eps:
            p, port, log = self.llama("base", args, self.linked_env(args, "base"), "remove")
            if port:
                box = {}

                def long_gen():
                    t = time.time()
                    try:
                        o = self.ask(port, 4000, timeout = 900)
                        box["n"] = o.get("timings", {}).get("predicted_n")
                    except Exception as exc:
                        box["err"] = repr(exc)[:200]
                    box["t"] = time.time() - t

                th = threading.Thread(target = long_gen, daemon = True)
                th.start()
                time.sleep(3)
                t_rm = time.time()
                rm = self.cli("cluster", "remove", name)
                th.join(60)
                ended = time.time() - t_rm
                cut = rm.returncode == 0 and not th.is_alive() and (box.get("err") or (box.get("n") or 0) < 4000) and ended < 30
                detail = json.dumps({"remove_rc": rm.returncode, "remove_out": rm.stdout[-300:], "gen": box, "ended_s_after_remove": round(ended, 1),
                                     "llama_alive": p.poll() is None})
            kill_group(p)
        self.step("cluster remove cuts the live session within seconds", cut, detail)
        try:
            args, eps, _ = self.plan("base")
            detail = json.dumps(args)
        except Exception as exc:  # no worker left: refusing to plan is also "no remote worker"
            eps, detail = [], f"planner refused: {exc!r}"[:600]
        self.step("after remove the plan has no remote worker", not eps, detail)
        return 0 if all(s["ok"] for s in R["steps"]) else 1

    def main(self):
        rc = 1
        try:
            rc = self.run()
        except Exception:
            import traceback
            self.step("unexpected error", False, traceback.format_exc())
        finally:
            for p in reversed(self.procs):
                kill_group(p)
            self.R.setdefault("verdict", "PASS" if rc == 0 and self.R["steps"] and all(s["ok"] for s in self.R["steps"]) else "FAIL")
            lp.write_json(str(self.out / "master_result.json"), self.R)
            print("VERDICT", self.R["verdict"], sum(s["ok"] for s in self.R["steps"]), "/", len(self.R["steps"]), flush = True)
        return rc


def summarize_perf(perf):
    out = {}
    for arm in sorted({r["arm"] for r in perf}):
        rows = [r for r in perf if r["arm"] == arm]
        lk = [statistics.median(r["linked_ms_per_token"]) for r in rows if r["linked_ms_per_token"]]
        dr = [statistics.median(r["direct_same_box_ms_per_token"]) for r in rows if r["direct_same_box_ms_per_token"]]
        out[arm] = {"rounds": len(rows),
                    "linked_lan_ms_per_token": {"median": round(statistics.median(lk), 3), "min": round(min(lk), 3), "max": round(max(lk), 3)} if lk else None,
                    "direct_same_box_ms_per_token": {"median": round(statistics.median(dr), 3), "min": round(min(dr), 3), "max": round(max(dr), 3)} if dr else None}
    return out


def cmd_master(a):
    return Master(a).main()


def cmd_summarize(a):
    found = {}
    for root, _, files in os.walk(a.input):
        for f in files:
            if f in ("master_result.json", "worker_result.json"):
                found[f] = lp.read_json(os.path.join(root, f))
    m, w = found.get("master_result.json"), found.get("worker_result.json")
    lines = []
    if not m:
        verdict = "NO_MASTER_RESULT"
    else:
        verdict = m.get("verdict", "FAIL")
    lines.append(f"# Home cluster link across two machines: VERDICT: {verdict}\n")
    if m:
        lines.append(f"- master: {m.get('master', {}).get('hostname')} ({(m.get('addresses') or {}).get('master')})")
        lines.append(f"- worker: {(m.get('worker') or {}).get('hostname')} ({(m.get('addresses') or {}).get('worker')})")
        lines.append(f"- control channel RTT: {m.get('control_rtt_us')}")
        lines.append("")
        lines.append("| step | result |")
        lines.append("|---|---|")
        for s in m.get("steps", []):
            lines.append(f"| {s['step']} | {'PASS' if s['ok'] else 'FAIL'} |")
        ps = m.get("perf_summary")
        if ps:
            lines.append("")
            lines.append("| build | rounds | relayed over LAN ms/token (median, min-max) | same-box direct ms/token (median, min-max) |")
            lines.append("|---|---|---|---|")
            for arm, r in ps.items():
                lk, dr = r.get("linked_lan_ms_per_token") or {}, r.get("direct_same_box_ms_per_token") or {}
                lines.append(f"| {arm} | {r['rounds']} | {lk.get('median')} ({lk.get('min')}-{lk.get('max')}) | {dr.get('median')} ({dr.get('min')}-{dr.get('max')}) |")
    if w:
        lines.append("")
        lines.append(f"- worker side: done={w.get('done')}, events={len(w.get('events', []))}, rejected={sum(1 for e in w.get('events', []) if e.get('cmd') == 'rejected')}")
    lines.append("")
    lines.append("Not covered: a Windows worker (inbound blocked on the Windows boxes), GPUs (CPU builds; the GPU group serializes GPU jobs), a raw direct-RPC run across the LAN (never exposed).")
    os.makedirs(a.out, exist_ok = True)
    Path(a.out, "VERDICT.md").write_text("\n".join(lines) + "\n", encoding = ENC)
    lp.write_json(os.path.join(a.out, "verdict.json"), {"verdict": verdict, "master": m, "worker": w})
    print("\n".join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest = "cmd", required = True)
    for role in ("worker", "master"):
        p = sub.add_parser(role)
        p.add_argument("--repo", required = True)
        p.add_argument("--bins", required = True, help = "base=DIR[,coal=DIR]")
        p.add_argument("--out-dir", required = True)
        p.add_argument("--local", action = "store_true", help = "self-test on one host over loopback")
        if role == "worker":
            p.add_argument("--window-s", type = float, default = 2700)
        else:
            p.add_argument("--gguf", required = True)
            p.add_argument("--peer-dir", default = None)
            p.add_argument("--wait-s", type = float, default = 1800)
            p.add_argument("--threads", type = int, default = 8)
            p.add_argument("--n-predict", type = int, default = 64)
            p.add_argument("--repeats", type = int, default = 3)
            p.add_argument("--rounds", type = int, default = 3)
            p.add_argument("--allow-same-machine", action = "store_true")
    p = sub.add_parser("summarize")
    p.add_argument("--in", dest = "input", required = True)
    p.add_argument("--out", required = True)
    a = ap.parse_args()
    return {"worker": cmd_worker, "master": cmd_master, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
