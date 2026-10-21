#!/usr/bin/env python3
"""Pipeline parallelism across N physical machines over llama.cpp RPC (CPU builds, lab LAN).

  worker    --bin DIR --out-dir D [--index K] [--window-s S] [--threads T]
            writes D/self.json (addresses, a control port, a per-run nonce) and obeys the master's
            nonce-authenticated commands: csr (a fresh key stays on this box), start_rpc (ggml-rpc-server
            on the LAN with llama.cpp's mutual TLS, the master's certificate the only one allowed),
            stop_rpc, window_begin / window_end (CPU use of the rpc-server over a measured window), done
  master    --bin DIR --gguf M --workers N --out-dir D
            single-box baseline X1 at the per-group batch, then S = 2, 4, N + 1 stages with G = 1, S, S + 1
            pipeline groups, saturating load; efficiency = X(S, G) / (S x X1(P / G)). The master is a stage
            too: -ngl n_layer - k keeps the first k layers AND the output layer on its CPU (llama.cpp takes no
            CPU in --device), so the logits never cross the LAN; the other layers split evenly over S - 1
            workers. A logits arm compares, on 2 workers holding every repeating layer, the output layer
            on the last worker (-ngl 999: the logits cross the LAN each step) against the output layer on
            the master's CPU (-ngl n_layer), single-user and saturated.
  summarize --in DIR --out DIR      VERDICT.md / verdict.json

No plaintext RPC on the LAN: every rpc-server listens with mutual TLS 1.3 and an allowlist holding only
this run's master certificate (signed by a CA made for this run; worker keys never leave their box). The
control channel carries commands with an HMAC over a per-run nonce published in the run's own artifact."""

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import hmac
import io
import ipaddress
import json
import os
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lanprobe"))
import lanprobe as lp  # noqa: E402  (interfaces, read_json, write_json)

ENC = "utf-8"
VIRTUAL = ("docker", "veth", "br-", "virbr", "lo", "tailscale", "zt", "wg")
ART_PREFIX = "pp-worker-self-"


def sign(nonce, body):
    return hmac.new(bytes.fromhex(nonce), json.dumps(body, sort_keys = True).encode(ENC), hashlib.sha256).hexdigest()


def free_port(host = "127.0.0.1"):
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def host_identity(info):
    macs = sorted({(i.get("mac") or "").lower() for i in info["interfaces"]
                   if i.get("mac") and not str(i.get("name", "")).startswith(VIRTUAL)} - {"", "00:00:00:00:00:00"})
    return {"hostname": socket.gethostname(), "runner": os.environ.get("RUNNER_NAME", ""), "macs": macs,
            "cpus": os.cpu_count(), "load": open("/proc/loadavg").read().split()[:3]}


# the output head (a tied model duplicates token_embd for it) on the master CPU
OUTPUT_ON_CPU = r"(output|token_embd)\.weight=CPU"
HEAD_LOCAL = ["-ot", OUTPUT_ON_CPU]
TRACE = os.environ.get("PP_TRACE") == "1"  # #191 event traces on the master and every rpc-server
LOCAL = os.environ.get("PP_LOCAL") == "1"  # smoke test on one host: loopback counts as the LAN


def lan_nets(info):
    if LOCAL:
        return [("127.0.0.1", "127.0.0.0/8")]
    out = []
    for i in info["interfaces"]:
        name = str(i.get("name", ""))
        for ip, prefix in i["ipv4"]:
            a = ipaddress.ip_address(ip)
            if not a.is_loopback and not a.is_link_local and not name.startswith(VIRTUAL):
                out.append((ip, str(ipaddress.ip_network(f"{ip}/{prefix or 24}", strict = False))))
    return out


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


def cpu_seconds(pid):
    """utime + stime of a process and its threads, in seconds."""
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def cert_fp(pem):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    cert = x509.load_pem_x509_certificate(pem.encode() if isinstance(pem, str) else pem)
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


# ------------------------------------------------------------------ worker

class Worker:
    def __init__(self, a):
        self.a = a
        self.bin = Path(a.bin).resolve()
        self.out = Path(a.out_dir)
        self.out.mkdir(parents = True, exist_ok = True)
        self.tls = self.out / "tls"
        self.tls.mkdir(exist_ok = True)
        self.nonce = secrets.token_hex(16)
        self.proc, self.events, self.done = None, [], False
        self.samples, self.sampling = [], False
        self.lock = threading.Lock()

    def ev(self, **kw):
        kw["t"] = round(time.time(), 3)
        self.events.append(kw)
        print("worker:", json.dumps(kw, default = str)[:400], flush = True)

    def stop_rpc(self):
        kill_group(self.proc)
        self.proc = None

    def sampler(self):
        last = None
        while not self.done:
            time.sleep(0.25)
            p = self.proc
            if p is None or p.poll() is not None:
                last = None
                continue
            try:
                now, cpu = time.monotonic(), cpu_seconds(p.pid)
            except OSError:
                last = None
                continue
            if last is not None and self.sampling:
                with self.lock:
                    self.samples.append((now, (cpu - last[1]) / max(now - last[0], 1e-6)))
            last = (now, cpu)

    def handle(self, cmd, args):
        if cmd == "ping":
            return {"ok": True, "load": open("/proc/loadavg").read().split()[:3]}
        if cmd == "csr":
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.x509.oid import NameOID
            key = ec.generate_private_key(ec.SECP256R1())
            (self.tls / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                                                 serialization.PrivateFormat.PKCS8,
                                                                 serialization.NoEncryption()))
            os.chmod(self.tls / "key.pem", 0o600)
            csr = (x509.CertificateSigningRequestBuilder()
                   .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"pp-worker-{self.a.index}")]))
                   .sign(key, hashes.SHA256()))
            return {"ok": True, "csr": csr.public_bytes(serialization.Encoding.PEM).decode()}
        if cmd == "start_rpc":
            self.stop_rpc()
            (self.tls / "cert.pem").write_text(args["cert"])
            (self.tls / "ca.pem").write_text(args["ca"])
            (self.tls / "allow.txt").write_text("\n".join(args["allow"]) + "\n")
            host, port = args["host"], int(args["port"])
            argv = [str(self.bin / "ggml-rpc-server"), "-H", host, "-p", str(port), "-t", str(args.get("threads", self.a.threads)),
                    "--tls-cert", str(self.tls / "cert.pem"), "--tls-key", str(self.tls / "key.pem"),
                    "--tls-ca", str(self.tls / "ca.pem"), "--tls-allow", str(self.tls / "allow.txt")]
            if args.get("cache"):
                argv += ["-c"]
            if TRACE:
                argv += ["--trace", str(self.out / f"trace_rpc_{port}.jsonl")]
            log = self.out / f"rpc_{port}_{int(time.time())}.log"
            env = dict(os.environ, HOME = str(self.out / "home"), LD_LIBRARY_PATH = str(self.bin))
            self.proc = subprocess.Popen(argv, stdout = open(log, "w"), stderr = subprocess.STDOUT, env = env,
                                         start_new_session = True)
            up, end = False, time.time() + 60
            while time.time() < end and self.proc.poll() is None:
                try:
                    with socket.create_connection((host, port), timeout = 1):
                        up = True
                        break
                except OSError:
                    time.sleep(0.2)
            self.ev(cmd = "start_rpc", host = host, port = port, up = up)
            return {"ok": up, "fp": cert_fp(args["cert"]), "tail": log.read_text(errors = "replace")[-800:]}
        if cmd == "stop_rpc":
            self.stop_rpc()
            return {"ok": True}
        if cmd == "window_begin":
            with self.lock:
                self.samples = []
            self.sampling = True
            return {"ok": True}
        if cmd == "window_end":
            self.sampling = False
            with self.lock:
                s = list(self.samples)
            t0 = s[0][0] if s else 0.0
            return {"ok": True, "samples": [(round(t - t0, 3), round(c, 3)) for t, c in s],
                    "load": open("/proc/loadavg").read().split()[:3]}
        if cmd == "tail":
            logs = sorted(self.out.glob("rpc_*.log"))
            return {"ok": True, "tail": logs[-1].read_text(errors = "replace")[-3000:] if logs else ""}
        if cmd == "done":
            self.stop_rpc()
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
        me = {"role": "worker", "index": self.a.index, **host_identity(info), "nets": lan_nets(info),
              "control_port": srv.getsockname()[1], "nonce": self.nonce, "rpc_port": free_port("0.0.0.0")}
        lp.write_json(str(self.out / "self.json"), me)
        print("worker control on", me["control_port"], me["nets"], flush = True)
        threading.Thread(target = self.sampler, daemon = True).start()
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
                        deadline = max(deadline, time.time() + 1800)
                        reply = self.handle(body["cmd"], body["args"])
                        f.write((json.dumps(reply, default = str) + "\n").encode(ENC))
                        f.flush()
                    except Exception as exc:
                        self.ev(cmd = "error", error = repr(exc))
        finally:
            self.done = True
            self.stop_rpc()
            lp.write_json(str(self.out / "worker_result.json"), {"self": me, "events": self.events,
                                                                  "timed_out": not self.done})
        return 0


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


def wait_peers(n, wait_s, attempt, peer_dirs = None):
    """The newest self.json of each worker index 1..n from this run's artifacts (or --peer-dirs)."""
    if peer_dirs:
        end = time.time() + wait_s
        while time.time() < end:
            paths = [Path(d) / "self.json" for d in peer_dirs]
            if all(p.exists() for p in paths):
                return [lp.read_json(str(p)) for p in paths]
            time.sleep(1)
        return [lp.read_json(str(p)) for p in paths if p.exists()]
    token = os.environ["GH_API_TOKEN"]
    repo, run_id = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
    found, end = {}, time.time() + wait_s
    while time.time() < end and len(found) < n:
        try:
            arts = api(f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", token).get("artifacts", [])
            for x in arts:
                name = x["name"]
                if not name.startswith(ART_PREFIX) or x.get("expired") or not name.endswith(f"-{attempt}"):
                    continue
                idx = int(name[len(ART_PREFIX):].split("-")[0])
                if idx in found:
                    continue
                blob = api(f"/repos/{repo}/actions/artifacts/{x['id']}/zip", token, raw = True)
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    found[idx] = json.loads(z.read("self.json").decode(ENC))
        except Exception as exc:
            print("waiting for workers:", repr(exc)[:200], flush = True)
        if len(found) < n:
            print(f"workers published: {sorted(found)} of {n}", flush = True)
            time.sleep(10)
    return [found[i] for i in sorted(found)]


class CA:
    """A CA made for this run: signs the master's client certificate and each worker's CSR."""

    def __init__(self, d):
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
        self.x509, self.hashes, self.ser, self.ec, self.oid = x509, hashes, serialization, ec, NameOID
        self.d = Path(d)
        self.d.mkdir(parents = True, exist_ok = True)
        self.key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "pp-lan run CA")])
        now = datetime.datetime.now(datetime.timezone.utc)
        self.cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
                     .public_key(self.key.public_key()).serial_number(x509.random_serial_number())
                     .not_valid_before(now - datetime.timedelta(minutes = 5)).not_valid_after(now + datetime.timedelta(days = 1))
                     .add_extension(x509.BasicConstraints(ca = True, path_length = 0), critical = True)
                     .add_extension(x509.KeyUsage(digital_signature = False, content_commitment = False, key_encipherment = False,
                                                  data_encipherment = False, key_agreement = False, key_cert_sign = True,
                                                  crl_sign = True, encipher_only = False, decipher_only = False), critical = True)
                     .sign(self.key, hashes.SHA256()))
        self.pem = self.cert.public_bytes(serialization.Encoding.PEM).decode()
        (self.d / "ca.pem").write_text(self.pem)

    def sign_csr(self, csr_pem):
        x509 = self.x509
        csr = x509.load_pem_x509_csr(csr_pem.encode())
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(csr.subject).issuer_name(self.cert.subject)
                .public_key(csr.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(minutes = 5)).not_valid_after(now + datetime.timedelta(days = 1))
                .add_extension(x509.BasicConstraints(ca = False, path_length = None), critical = True)
                .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                                                      x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]), critical = False)
                .sign(self.key, self.hashes.SHA256()))
        return cert.public_bytes(self.ser.Encoding.PEM).decode()

    def client(self):
        key = self.ec.generate_private_key(self.ec.SECP256R1())
        (self.d / "client.key").write_bytes(key.private_bytes(self.ser.Encoding.PEM, self.ser.PrivateFormat.PKCS8,
                                                              self.ser.NoEncryption()))
        csr = (self.x509.CertificateSigningRequestBuilder()
               .subject_name(self.x509.Name([self.x509.NameAttribute(self.oid.COMMON_NAME, "pp-master")]))
               .sign(key, self.hashes.SHA256()))
        pem = self.sign_csr(csr.public_bytes(self.ser.Encoding.PEM).decode())
        (self.d / "client.pem").write_text(pem)
        return str(self.d / "client.pem"), str(self.d / "client.key"), cert_fp(pem)


def make_prompt(i):
    words = ["river", "stone", "cloud", "lantern", "harbor", "meadow", "copper", "violet", "signal", "orchard"]
    body = " ".join(words[(i * 7 + k * 3) % len(words)] for k in range(90))
    return f"Request {i}. Continue this list of words with more words in the same style: {body}"


class Master:
    def __init__(self, a):
        self.a = a
        self.bin = Path(a.bin).resolve()
        self.out = Path(a.out_dir)
        self.out.mkdir(parents = True, exist_ok = True)
        self.R = {"cases": [], "steps": [], "args": vars(a)}

    def step(self, name, ok, detail = ""):
        self.R["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[-1500:]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {'' if ok else str(detail)[-400:]}", flush = True)
        return ok

    def server(self, tag, extra, env_extra = None):
        port = free_port()
        argv = [str(self.bin / "llama-server"), "-m", self.a.gguf, "--host", "127.0.0.1", "--port", str(port),
                "--no-warmup", "--temp", "0"] + extra
        log = self.out / f"server_{tag}.log"
        env = dict(os.environ, LD_LIBRARY_PATH = str(self.bin), **(env_extra or {}))
        if TRACE:
            env["GGML_RPC_TRACE"] = str(self.out / f"trace_{tag}.jsonl")
        p = subprocess.Popen(argv, stdout = open(log, "w"), stderr = subprocess.STDOUT, env = env, start_new_session = True)
        end = time.time() + self.a.load_timeout_s
        while time.time() < end and p.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 2) as r:
                    if r.status == 200:
                        return p, port, log
            except Exception:
                pass
            time.sleep(1)
        kill_group(p)
        return None, None, log

    def one(self, port, i, n_predict):
        body = json.dumps({"prompt": make_prompt(i), "n_predict": n_predict, "temperature": 0,
                           "ignore_eos": True, "cache_prompt": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data = body,
                                     headers = {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout = 1800) as r:
            out = json.loads(r.read())
        t = out.get("timings", {})
        return int(t.get("predicted_n", 0)), float(t.get("prompt_n", 0))

    def load(self, port, conc, n_req, n_predict, base):
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(lambda i: self.one(port, base + i, n_predict), range(n_req)))
        dt = time.perf_counter() - t0
        gen = sum(r[0] for r in res)
        return {"seconds": round(dt, 2), "gen_tokens": gen, "tok_s": round(gen / dt, 2), "requests": n_req}

    def measure(self, tag, extra, conc, ctls = (), env_extra = None):
        """Warm up with one full wave, then a saturating window of `waves` x conc requests."""
        p, port, log = self.server(tag, extra, env_extra)
        if p is None:
            return {"tag": tag, "error": "server did not come up", "log": log.read_text(errors = "replace")[-1500:]}
        try:
            self.load(port, conc, conc, 8, 10_000)
            mine, stop = [], threading.Event()

            def sample_self():  # the master's own stage: its llama-server's CPU use
                last = (time.monotonic(), cpu_seconds(p.pid))
                while not stop.wait(0.25):
                    now, cpu = time.monotonic(), cpu_seconds(p.pid)
                    mine.append((now - last[0], (cpu - last[1]) / max(now - last[0], 1e-6)))
                    last = (now, cpu)

            th = threading.Thread(target = sample_self, daemon = True)
            for c in ctls:
                c("window_begin")
            th.start()
            r = self.load(port, conc, self.a.waves * conc, self.a.n_predict, 20_000)
            stop.set()
            th.join(5)
            busy = [mine] + [c("window_end").get("samples", []) for c in ctls]
            r["busy"] = busy_stats(busy, self.a.threads) if ctls else busy_stats([mine], self.a.threads)
            return {"tag": tag, **r}
        except Exception as exc:
            return {"tag": tag, "error": repr(exc)[:400], "log": log.read_text(errors = "replace")[-1500:]}
        finally:
            kill_group(p)

    def run(self):
        a, R = self.a, self.R
        info = lp.interfaces()
        R["master"] = {**host_identity(info), "nets": lan_nets(info)}
        peers = wait_peers(a.workers, a.wait_s, os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
                           a.peer_dirs.split(",") if a.peer_dirs else None)
        R["workers"] = [{k: p.get(k) for k in ("index", "hostname", "runner", "macs", "nets", "cpus", "load")} for p in peers]
        need = min(a.min_workers or a.workers, a.workers)
        if not self.step(f"at least {need} of {a.workers} workers published their control address", len(peers) >= need,
                         f"got {[p.get('index') for p in peers]}"):
            R["verdict"] = "NOT_CONCURRENT"
            return 3
        # a lab box that never got scheduled shrinks the sweep instead of voiding the boxes that did
        R["workers_requested"], a.workers = a.workers, len(peers)
        boxes = {R["master"]["hostname"], *(p["hostname"] for p in peers)}
        if not self.step("master and every worker are different machines", LOCAL or len(boxes) == a.workers + 1, sorted(boxes)):
            R["verdict"] = "SHARED_MACHINE"
            return 3
        ctls, addrs = [], []
        for p in peers:
            pair = [(m, w) for m, mn in R["master"]["nets"] for w, wn in p.get("nets", []) if mn == wn and (m != w or LOCAL)]
            if not pair:
                self.step(f"worker {p['index']} shares a LAN subnet with the master", False, p.get("nets"))
                R["verdict"] = "NO_SHARED_SUBNET"
                return 3
            ctls.append(Control(pair[0][1], p["control_port"], p["nonce"]))
            addrs.append((pair[0][1], p["rpc_port"]))
        try:
            return self.flow(ctls, addrs)
        finally:
            for c in ctls:
                try:
                    c("done", timeout = 60)
                except Exception as exc:
                    print("done not delivered:", repr(exc), flush = True)

    def flow(self, ctls, addrs):
        a, R = self.a, self.R
        rtts = []
        for _ in range(10):
            t = time.perf_counter()
            ctls[0]("ping", timeout = 10)
            rtts.append((time.perf_counter() - t) * 1e6)
        R["control_rtt_us"] = round(statistics.median(rtts), 1)
        ca = CA(self.out / "tls")
        cert, key, master_fp = ca.client()
        endpoints = []
        for i, (c, (host, port)) in enumerate(zip(ctls, addrs)):
            csr = c("csr")["csr"]
            r = c("start_rpc", cert = ca.sign_csr(csr), ca = ca.pem, allow = [master_fp], host = host, port = port,
                  threads = a.threads, cache = True)
            if not self.step(f"worker {i + 1} serves TLS RPC on {host}:{port}", r.get("ok"), r.get("tail", "")):
                R["verdict"] = "RPC_DOWN"
                return 1
            endpoints.append(f"tls://{r['fp']}@{host}:{port}")
        tls_env = {"GGML_RPC_TLS_CERT": cert, "GGML_RPC_TLS_KEY": key, "GGML_RPC_TLS_CA": str(ca.d / "ca.pem")}
        B = a.slots_per_group
        ctx = lambda slots: str(slots * (a.ctx_per_slot))  # noqa: E731

        def local(slots):
            return ["-t", str(a.threads), "-np", str(slots), "-c", ctx(slots), "-ngl", "0"]

        def split(s, g, head_local = True):
            # s stages: the master (first k layers + output, on its CPU) and s - 1 workers. -ngl counts the
            # output layer first (i_gpu_start = n_layer + 1 - ngl), so -ngl alone would put it on the last
            # worker and ship vocab x rows x 4 bytes of logits per step; OUTPUT_ON_CPU keeps it here.
            eps = endpoints[:s - 1]
            k = round(a.n_layer / s)
            p = g * B
            args = ["-t", str(a.threads), "-np", str(p), "-c", ctx(B), "--rpc", ",".join(eps),
                    "--device", ",".join(f"RPC{j}" for j in range(s - 1)),
                    "-ngl", str(a.n_layer - k + (1 if head_local else 0)),
                    "-sm", "layer", "-ts", ",".join(["1"] * (s - 1))] + (HEAD_LOCAL if head_local else [])
            if g > 1:
                args += ["--pipeline-groups", str(g)]
            return args, p

        def logits(out_remote, g):
            # 2 workers hold every repeating layer; the output layer on the last worker or on this CPU
            p = g * B
            args = ["-t", str(a.threads), "-np", str(p), "-c", ctx(B), "--rpc", ",".join(endpoints[:2]),
                    "--device", "RPC0,RPC1", "-ngl", "999", "-sm", "layer", "-ts", "1,1"]
            if not out_remote:
                args += HEAD_LOCAL
            if g > 1:
                args += ["--pipeline-groups", str(g)]
            return args, p

        plan = []
        stages = sorted({int(x) for x in a.stages.split(",")} if a.stages else {2, min(4, a.workers + 1), a.workers + 1})
        stages = [x for x in stages if 2 <= x <= a.workers + 1]
        heads = {"on": [True], "off": [False], "both": [False, True]}[a.head_local]
        for rnd in range(a.rounds):
            plan.append(("x1", 1, 1, B))
            if rnd == 0:
                plan.append(("x1_big", 1, 1, (a.workers + 2) * B))
            for s in stages:
                for g in sorted({1, s, s + 1}):
                    for hl in heads:
                        plan.append(("rpc" if hl else "rpc_hr", s, g, g * B))
            if a.workers >= 2:
                for out_remote in (True, False):
                    for g, conc in ((1, 1), (1, B), (3, 3 * B)):
                        plan.append(("logits_remote" if out_remote else "logits_local", 2, g, conc))
        for rnd_i, (kind, s, g, p) in enumerate(plan):
            if kind.startswith("x1"):
                res = self.measure(f"{kind}_{p}_{rnd_i}", local(p), p)
            elif kind.startswith("logits"):
                args, slots = logits(kind == "logits_remote", g)
                res = self.measure(f"{kind}_g{g}_c{p}_{rnd_i}", args, p, ctls[:2], tls_env)
                res["conc"] = p
                p = slots
            else:
                args, p = split(s, g, kind == "rpc")
                res = self.measure(f"{kind}_s{s}_g{g}_{rnd_i}", args, p, ctls[:s - 1], tls_env)
            res.update({"kind": kind, "stages": s, "groups": g, "slots": p})
            R["cases"].append(res)
            print("CASE", json.dumps({k: v for k, v in res.items() if k != "log"}, default = str)[:700], flush = True)
            lp.write_json(str(self.out / "master_result.json"), R)
        R["summary"] = summarize_cases(R["cases"], B)
        for row in R["summary"]["rows"]:
            print("SUMMARY", json.dumps(row), flush = True)
        ok = all("error" not in c for c in R["cases"])
        self.step("every case served", ok, [c.get("tag") for c in R["cases"] if "error" in c])
        return 0 if ok else 1

    def main(self):
        rc = 1
        try:
            rc = self.run()
        except Exception:
            import traceback
            self.step("unexpected error", False, traceback.format_exc())
        finally:
            self.R.setdefault("verdict", "DONE" if rc == 0 else "FAIL")
            lp.write_json(str(self.out / "master_result.json"), self.R)
            print("VERDICT", self.R["verdict"], flush = True)
        return rc


def busy_stats(series, threads):
    """Per-stage mean CPU use (fraction of its thread budget; master first) over the window, and the share of
    samples in which every stage computes at once (each above a third of its own mean)."""
    if not series or not all(series):
        return None
    n = min(len(s) for s in series)
    util = [[c / threads for _, c in s[:n]] for s in series]
    means = [statistics.mean(u) for u in util]
    all_busy = sum(all(u[k] > m / 3 for u, m in zip(util, means)) for k in range(n)) / max(n, 1)
    return {"mean_util": [round(m, 3) for m in means], "all_busy_share": round(all_busy, 3), "samples": n}


def summarize_cases(cases, B):
    ok = [c for c in cases if "error" not in c]
    x1 = [c["tok_s"] for c in ok if c["kind"] == "x1"]
    x1_med = statistics.median(x1) if x1 else None
    big = [c["tok_s"] for c in ok if c["kind"] == "x1_big"]
    rows, logit_rows = [], []
    for kind in ("logits_remote", "logits_local"):
        for (g, conc) in sorted({(c["groups"], c.get("conc")) for c in ok if c["kind"] == kind}):
            xs = [c["tok_s"] for c in ok if c["kind"] == kind and c["groups"] == g and c.get("conc") == conc]
            logit_rows.append({"output": "last worker" if kind == "logits_remote" else "master CPU (-ot)",
                               "groups": g, "concurrency": conc, "tok_s": round(statistics.median(xs), 2), "runs": xs})
    for (kind, s, g) in sorted({(c["kind"], c["stages"], c["groups"]) for c in ok if c["kind"] in ("rpc", "rpc_hr")}):
        same = [c for c in ok if c["kind"] == kind and c["stages"] == s and c["groups"] == g]
        xs = [c["tok_s"] for c in same]
        busy = [c.get("busy") for c in same if c.get("busy")]
        med = statistics.median(xs)
        rows.append({"head": "master" if kind == "rpc" else "last worker",
                     "stages": s, "groups": g, "slots": g * B, "tok_s": round(med, 2), "runs": xs,
                     "ideal": round(s * x1_med, 2) if x1_med else None,
                     "efficiency": round(med / (s * x1_med), 3) if x1_med else None,
                     "mean_util": busy[-1]["mean_util"] if busy else None,
                     "all_busy_share": busy[-1]["all_busy_share"] if busy else None})
    return {"x1_tok_s": x1, "x1_median": x1_med, "x1_big_tok_s": big, "slots_per_group": B, "rows": rows,
            "logits": logit_rows}


def cmd_summarize(a):
    src = Path(a.in_dir)
    res = next((json.loads(p.read_text()) for p in src.rglob("master_result.json")), None)
    lines = ["# N-machine pipeline parallelism over llama.cpp RPC (CPU, lab LAN, mutual TLS)", ""]
    if not res:
        lines.append("No master result.")
    else:
        lines.append(f"Verdict: {res.get('verdict')}  ")
        lines.append(f"Master: {res.get('master', {}).get('hostname')}, workers: "
                     f"{', '.join(str(w.get('hostname')) for w in res.get('workers', []))}  ")
        sm = res.get("summary") or {}
        lines.append(f"Single box X1 at {sm.get('slots_per_group')} slots: {sm.get('x1_tok_s')} tok/s "
                     f"(median {sm.get('x1_median')}); single box at {(len(res.get('workers', [])) + 1) * (sm.get('slots_per_group') or 0)} "
                     f"slots: {sm.get('x1_big_tok_s')}")
        lines += ["", "| head on | stages | groups | slots | tok/s (median) | ideal S x X1 | efficiency | stage CPU use | all stages busy |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sm.get("rows", []):
            lines.append(f"| {r.get('head', 'last worker')} | {r['stages']} | {r['groups']} | {r['slots']} | {r['tok_s']} {r['runs']} | {r['ideal']} | "
                         f"{r['efficiency']} | {r['mean_util']} | {r['all_busy_share']} |")
        if sm.get("logits"):
            lines += ["", "Logits arm (2 workers hold every repeating layer):", "",
                      "| output layer on | groups | concurrency | tok/s (median) |", "|---|---|---|---|"]
            for r in sm["logits"]:
                lines.append(f"| {r['output']} | {r['groups']} | {r['concurrency']} | {r['tok_s']} {r['runs']} |")
        lines += ["", "Steps:"] + [f"- {'PASS' if s['ok'] else 'FAIL'} {s['step']}" for s in res.get("steps", [])]
    out = Path(a.out_dir)
    out.mkdir(parents = True, exist_ok = True)
    (out / "VERDICT.md").write_text("\n".join(lines) + "\n")
    lp.write_json(str(out / "verdict.json"), res or {})
    print("\n".join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest = "cmd", required = True)
    w = sub.add_parser("worker")
    w.add_argument("--bin", required = True)
    w.add_argument("--out-dir", required = True)
    w.add_argument("--index", type = int, default = 1)
    w.add_argument("--window-s", type = float, default = 5400)
    w.add_argument("--threads", type = int, default = 16)
    m = sub.add_parser("master")
    m.add_argument("--bin", required = True)
    m.add_argument("--gguf", required = True)
    m.add_argument("--workers", type = int, default = 4)
    m.add_argument("--stages", default = "", help = "comma list of stage counts (default 2, min(4, w + 1), w + 1)")
    m.add_argument("--head-local", default = "on", choices = ["on", "off", "both"],
                   help = "output head on the master CPU (-ot) in the rpc cases; 'both' adds rpc_hr cases with it on the last worker")
    m.add_argument("--min-workers", type = int, default = 0, help = "run with this many if the rest never start (0 = all)")
    m.add_argument("--out-dir", required = True)
    m.add_argument("--wait-s", type = float, default = 2400)
    m.add_argument("--threads", type = int, default = 16)
    m.add_argument("--n-layer", type = int, default = 36)
    m.add_argument("--peer-dirs", default = "")
    m.add_argument("--slots-per-group", type = int, default = 16)
    m.add_argument("--ctx-per-slot", type = int, default = 512)
    m.add_argument("--n-predict", type = int, default = 64)
    m.add_argument("--waves", type = int, default = 2)
    m.add_argument("--rounds", type = int, default = 2)
    m.add_argument("--load-timeout-s", type = float, default = 600)
    s = sub.add_parser("summarize")
    s.add_argument("--in", dest = "in_dir", required = True)
    s.add_argument("--out", dest = "out_dir", required = True)
    a = ap.parse_args()
    if a.cmd == "worker":
        return Worker(a).serve()
    if a.cmd == "master":
        return Master(a).main()
    return cmd_summarize(a)


if __name__ == "__main__":
    sys.exit(main())
