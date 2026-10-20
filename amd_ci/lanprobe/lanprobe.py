#!/usr/bin/env python3
"""Two-machine LAN probe for the strix-halo pool: can two jobs reach each other, and does UDP multicast
(239.255.77.77:50079, TTL 1, the cluster discovery group) cross between them?

Stdlib only (Linux python3 and the Windows boxes' Python 3.12).

  serve     --role R --out-dir D [--window-s 900]   listeners + multicast beacons, writes D/self.json
  client    --role R --out-dir D --peers a,b [--linux-peers b] [--peer-dir P]
            finds the peers' self.json (GitHub run artifacts, or P for a local test), tests TCP / UDP /
            RTT / bandwidth against them, waits until each peer has tested us, then stops serve
  collect   --out-dir D                              merges self / client / serve logs into result.json
  summarize --in DIR --out DIR                       VERDICT.md + verdict.json from every result.json

Everything a listener answers must carry a proof of the per-job random nonce (an HMAC over a
challenge or the message); the nonce itself is never sent on the wire, only published in the job's
own artifact, so nothing else on the lab network can use these listeners."""

import argparse
import hashlib
import hmac
import io
import ipaddress
import json
import os
import platform
import secrets
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

GROUP = "239.255.77.77"
GROUP_PORT = 50079
ENC = "utf-8"


def mac(nonce: str, text: str) -> str:
    return hmac.new(bytes.fromhex(nonce), text.encode(ENC), hashlib.sha256).hexdigest()


def read_json(path):
    with open(path, encoding = ENC) as f:
        return json.load(f)


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding = ENC) as f:
        json.dump(data, f, indent = 1, default = str)
    os.replace(tmp, path)


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output = True, text = True, timeout = 20)
        return r.stdout
    except Exception as exc:
        return f"ERROR {exc}"


def interfaces():
    """[{name, mac, mtu, up, ipv4: [(ip, prefix)]}], default route, raw text for the record."""
    out = {"interfaces": [], "default_route": "", "raw": ""}
    if sys.platform.startswith("linux"):
        try:
            for i in json.loads(run(["ip", "-j", "addr", "show"]) or "[]"):
                out["interfaces"].append({
                    "name": i.get("ifname"), "mac": i.get("address"), "mtu": i.get("mtu"),
                    "up": "UP" in (i.get("flags") or []),
                    "ipv4": [(a["local"], a.get("prefixlen")) for a in i.get("addr_info", []) if a.get("family") == "inet"],
                })
        except Exception as exc:
            out["raw"] = f"ip -j addr failed: {exc}"
        out["default_route"] = run(["ip", "route", "show", "default"]).strip()
    else:
        ips = []
        try:
            ips = sorted({ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
        except OSError:
            pass
        macs = [line.split(",")[0].strip('"') for line in run(["getmac", "/fo", "csv", "/nh"]).splitlines() if line.strip()]
        out["interfaces"] = [{"name": f"if{n}", "mac": None, "mtu": None, "up": True, "ipv4": [(ip, None)]} for n, ip in enumerate(ips)]
        out["macs"] = macs
        out["default_route"] = run(["route", "print", "-4", "0.0.0.0"])[-1500:]
        out["raw"] = run(["ipconfig", "/all"])[-6000:]
    return out


def lan_ipv4(info):
    ips = []
    for i in info["interfaces"]:
        for ip, _ in i["ipv4"]:
            a = ipaddress.ip_address(ip)
            if not a.is_loopback and not a.is_link_local and ip not in ips:
                ips.append(ip)
    return ips


def all_macs(info):
    macs = set(m.lower().replace("-", ":") for m in info.get("macs", []) if m and ":" in m.replace("-", ":"))
    for i in info["interfaces"]:
        m = (i.get("mac") or "").lower()
        if m and m != "00:00:00:00:00:00":
            macs.add(m)
    return sorted(macs)


# ---------------------------------------------------------------- serve

class Server:
    def __init__(self, role, out_dir, window_s):
        self.role, self.out_dir = role, out_dir
        self.nonce = secrets.token_hex(16)
        self.deadline = time.time() + window_s
        self.events = []
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def log(self, **ev):
        ev["t"] = round(time.time(), 3)
        with self.lock:
            if len(self.events) < 20000:
                self.events.append(ev)

    def stopped(self):
        return self.stop.is_set() or time.time() > self.deadline or os.path.exists(os.path.join(self.out_dir, "STOP"))

    def start(self):
        info = interfaces()
        self.ips = lan_ipv4(info)
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.bind(("0.0.0.0", 0))
        self.tcp.listen(16)
        self.tcp.settimeout(0.5)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(("0.0.0.0", 0))
        self.udp.settimeout(0.5)
        self.mrecv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.mrecv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self.mrecv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        self.mrecv.bind(("", GROUP_PORT))
        joined = []
        for ip in self.ips + ["0.0.0.0"]:
            try:
                self.mrecv.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                                      struct.pack("4s4s", socket.inet_aton(GROUP), socket.inet_aton(ip)))
                joined.append(ip)
            except OSError as exc:
                self.log(kind = "join_error", ip = ip, error = str(exc))
        self.mrecv.settimeout(0.5)
        self.self_info = {
            "role": self.role, "nonce": self.nonce, "os": platform.system(), "platform": platform.platform(),
            "hostname": socket.gethostname(), "runner_name": os.environ.get("RUNNER_NAME", ""),
            "tcp_port": self.tcp.getsockname()[1], "udp_port": self.udp.getsockname()[1],
            "ipv4": self.ips, "macs": all_macs(info), "joined": joined, "net": info,
            "started": time.time(), "deadline": self.deadline, "pid": os.getpid(),
        }
        write_json(os.path.join(self.out_dir, "self.json"), self.self_info)
        for fn in (self.tcp_loop, self.udp_loop, self.mcast_recv_loop, self.mcast_send_loop, self.flush_loop):
            threading.Thread(target = fn, daemon = True).start()

    def tcp_loop(self):
        while not self.stopped():
            try:
                conn, addr = self.tcp.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target = self.tcp_conn, args = (conn, addr), daemon = True).start()

    def tcp_conn(self, conn, addr):
        conn.settimeout(15)
        try:
            challenge = secrets.token_hex(16)
            conn.sendall((challenge + "\n").encode(ENC))
            f = conn.makefile("rb")
            req = json.loads(f.readline(4096).decode(ENC))
            if not hmac.compare_digest(req.get("mac", ""), mac(self.nonce, f"tcp|{req.get('from')}|{req.get('op')}|{challenge}")):
                self.log(kind = "tcp_reject", peer = addr[0])
                return
            op, who = req.get("op"), req.get("from")
            self.log(kind = "tcp_" + str(op), peer = addr[0], from_role = who)
            if op == "hello":
                conn.sendall((json.dumps({"ok": True, "role": self.role}) + "\n").encode(ENC))
            elif op == "ping":
                for _ in range(1000):
                    b = f.read(8)
                    if len(b) != 8:
                        break
                    conn.sendall(b)
            elif op == "bw":
                total, t0 = 0, time.time()
                while time.time() - t0 < 30:
                    b = f.read1(1 << 16) if hasattr(f, "read1") else f.read(1 << 16)
                    if not b:
                        break
                    total += len(b)
                conn.sendall((json.dumps({"bytes": total}) + "\n").encode(ENC))
        except Exception as exc:
            self.log(kind = "tcp_error", peer = addr[0], error = str(exc)[:200])
        finally:
            conn.close()

    def udp_loop(self):
        while not self.stopped():
            try:
                data, addr = self.udp.recvfrom(2048)
                msg = json.loads(data.decode(ENC))
            except (socket.timeout, OSError, ValueError):
                continue
            who, seq = msg.get("from"), msg.get("seq")
            if not hmac.compare_digest(str(msg.get("mac", "")), mac(self.nonce, f"udp|{who}|{seq}")):
                self.log(kind = "udp_reject", peer = addr[0])
                continue
            self.log(kind = "udp", peer = addr[0], from_role = who, seq = seq)
            try:
                self.udp.sendto(json.dumps({"ack": seq, "from": self.role, "mac": mac(self.nonce, f"ack|{who}|{seq}")}).encode(ENC), addr)
            except OSError:
                pass

    def mcast_recv_loop(self):
        while not self.stopped():
            try:
                data, addr = self.mrecv.recvfrom(2048)
                msg = json.loads(data.decode(ENC))
            except (socket.timeout, OSError, ValueError):
                continue
            if not isinstance(msg, dict) or msg.get("lanprobe") != 1 or msg.get("from") == self.role:
                continue
            self.log(kind = "beacon", peer = addr[0], from_role = msg.get("from"), seq = msg.get("seq"),
                     ifip = msg.get("ifip"), tag = msg.get("tag"))

    def mcast_send_loop(self):
        seq = 0
        socks = []
        for ip in self.ips:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
                socks.append((ip, s))
            except OSError as exc:
                self.log(kind = "beacon_sock_error", ip = ip, error = str(exc))
        while not self.stopped():
            seq += 1
            for ip, s in socks:
                tag = mac(self.nonce, f"beacon|{self.role}|{seq}|{ip}")
                try:
                    s.sendto(json.dumps({"lanprobe": 1, "from": self.role, "seq": seq, "ifip": ip, "tag": tag}).encode(ENC),
                             (GROUP, GROUP_PORT))
                except OSError as exc:
                    if seq < 3:
                        self.log(kind = "beacon_send_error", ip = ip, error = str(exc))
            time.sleep(1.0)

    def flush_loop(self):
        while not self.stopped():
            self.flush()
            time.sleep(2.0)

    def flush(self):
        with self.lock:
            events = list(self.events)
        write_json(os.path.join(self.out_dir, "serve_log.json"), {"role": self.role, "events": events})


def cmd_serve(a):
    os.makedirs(a.out_dir, exist_ok = True)
    s = Server(a.role, a.out_dir, a.window_s)
    s.start()
    print(f"serving role={a.role} tcp={s.self_info['tcp_port']} udp={s.self_info['udp_port']} ips={s.ips}", flush = True)
    while not s.stopped():
        time.sleep(0.5)
    s.stop.set()
    time.sleep(0.6)
    s.flush()
    print("serve done", flush = True)


# ---------------------------------------------------------------- client

def api(path, token, raw = False):
    url = os.environ.get("GITHUB_API_URL", "https://api.github.com") + path
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    # Unredirected: the artifact zip redirects to a pre-signed blob URL that rejects (and must not see) the token.
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout = 30) as r:
        body = r.read()
    return body if raw else json.loads(body.decode(ENC))


def fetch_peer_from_artifacts(peer, token):
    repo, run_id = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
    arts = api(f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", token).get("artifacts", [])
    mine = [x for x in arts if x["name"].startswith(f"lanprobe-self-{peer}-") and not x.get("expired")]
    if not mine:
        return None
    best = max(mine, key = lambda x: int(x["name"].rsplit("-", 1)[1]) if x["name"].rsplit("-", 1)[1].isdigit() else 0)
    blob = api(f"/repos/{repo}/actions/artifacts/{best['id']}/zip", token, raw = True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        return json.loads(z.read("self.json").decode(ENC))


def tcp_open(ip, port, role, op, nonce, timeout = 3.0):
    s = socket.create_connection((ip, port), timeout = timeout)
    s.settimeout(15)
    f = s.makefile("rb")
    challenge = f.readline(64).decode(ENC).strip()
    s.sendall((json.dumps({"from": role, "op": op, "mac": mac(nonce, f"tcp|{role}|{op}|{challenge}")}) + "\n").encode(ENC))
    return s, f


def test_peer(me, peer, linux_bw):
    res = {"role": peer["role"], "runner_name": peer.get("runner_name"), "hostname": peer.get("hostname"),
           "found_at": time.time(), "tcp": {}, "udp": {}}
    nonce = peer["nonce"]
    ok_ip = None
    for ip in peer["ipv4"]:
        t0 = time.time()
        try:
            s, f = tcp_open(ip, peer["tcp_port"], me["role"], "hello", nonce)
            reply = json.loads(f.readline(4096).decode(ENC))
            s.close()
            res["tcp"][ip] = {"ok": bool(reply.get("ok")), "connect_ms": round((time.time() - t0) * 1000, 2)}
            ok_ip = ok_ip or ip
        except Exception as exc:
            res["tcp"][ip] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.settimeout(0.3)
    for ip in peer["ipv4"]:
        acks = 0
        for seq in range(5):
            try:
                u.sendto(json.dumps({"from": me["role"], "seq": f"{ip}#{seq}", "mac": mac(nonce, f"udp|{me['role']}|{ip}#{seq}")}).encode(ENC),
                         (ip, peer["udp_port"]))
                data, _ = u.recvfrom(2048)
                msg = json.loads(data.decode(ENC))
                if hmac.compare_digest(msg.get("mac", ""), mac(nonce, f"ack|{me['role']}|{msg.get('ack')}")):
                    acks += 1
            except (OSError, ValueError):
                pass
        res["udp"][ip] = {"sent": 5, "acked": acks}
    u.close()
    if ok_ip:
        res["best_ip"] = ok_ip
        try:
            s, f = tcp_open(ok_ip, peer["tcp_port"], me["role"], "ping", nonce)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            rtts = []
            for i in range(100):
                t0 = time.perf_counter()
                s.sendall(struct.pack("!Q", i))
                if f.read(8) != struct.pack("!Q", i):
                    raise RuntimeError("bad echo")
                rtts.append((time.perf_counter() - t0) * 1e6)
            s.close()
            rtts.sort()
            res["rtt_us"] = {"n": len(rtts), "median": round(statistics.median(rtts), 1), "p90": round(rtts[89], 1),
                             "min": round(rtts[0], 1)}
        except Exception as exc:
            res["rtt_error"] = str(exc)[:200]
        if linux_bw:
            try:
                s, f = tcp_open(ok_ip, peer["tcp_port"], me["role"], "bw", nonce)
                buf, sent, t0 = b"\0" * (1 << 16), 0, time.time()
                while time.time() - t0 < 10:
                    s.sendall(buf)
                    sent += len(buf)
                s.shutdown(socket.SHUT_WR)
                got = json.loads(f.readline(4096).decode(ENC))["bytes"]
                dt = time.time() - t0
                s.close()
                res["bandwidth"] = {"sent": sent, "received": got, "seconds": round(dt, 2), "mbit_s": round(got * 8 / dt / 1e6, 1)}
            except Exception as exc:
                res["bandwidth_error"] = str(exc)[:200]
    return res


def tested_by(out_dir):
    try:
        ev = read_json(os.path.join(out_dir, "serve_log.json"))["events"]
    except Exception:
        return set()
    return {e.get("from_role") for e in ev if e.get("kind") == "tcp_hello"}


def cmd_client(a):
    me = read_json(os.path.join(a.out_dir, "self.json"))
    peers = [p for p in a.peers.split(",") if p and p != a.role]
    linux_peers = set(p for p in a.linux_peers.split(",") if p)
    token = os.environ.get("GH_API_TOKEN", "")
    found, results, errors = {}, {}, []
    wait_until = time.time() + a.peer_wait_s
    while time.time() < wait_until and len(found) < len(peers):
        for p in peers:
            if p in found:
                continue
            try:
                if a.peer_dir:
                    path = os.path.join(a.peer_dir, p, "self.json")
                    info = read_json(path) if os.path.exists(path) else None
                else:
                    info = fetch_peer_from_artifacts(p, token)
            except Exception as exc:
                errors.append(f"{time.strftime('%H:%M:%S')} {p}: {type(exc).__name__}: {exc}"[:300])
                info = None
            if info:
                found[p] = info
                print(f"found peer {p} on {info.get('runner_name') or info.get('hostname')} ips={info.get('ipv4')}", flush = True)
                bw = p in linux_peers and info.get("os") == "Linux" and me.get("os") == "Linux"
                results[p] = test_peer(me, info, bw)
                write_json(os.path.join(a.out_dir, "client.json"), {"results": results, "errors": errors[-50:]})
        time.sleep(5)
    # Stay up until every found peer has tested us too (or our window ends), then stop serving.
    while time.time() < me["deadline"] - 5 and not set(found) <= tested_by(a.out_dir):
        time.sleep(3)
    write_json(os.path.join(a.out_dir, "client.json"), {
        "results": results, "errors": errors[-50:], "missing_peers": [p for p in peers if p not in found],
        "tested_by": sorted(x for x in tested_by(a.out_dir) if x), "finished": time.time()})
    open(os.path.join(a.out_dir, "STOP"), "w", encoding = ENC).close()
    print(f"client done: found={sorted(found)} missing={[p for p in peers if p not in found]}", flush = True)


def cmd_collect(a):
    out = {}
    for name in ("self", "client", "serve_log"):
        p = os.path.join(a.out_dir, name + ".json")
        try:
            out[name] = read_json(p)
        except Exception as exc:
            out[name] = {"missing": str(exc)}
    write_json(os.path.join(a.out_dir, "result.json"), out)
    print("collected", {k: ("missing" not in v) for k, v in out.items()})


# ---------------------------------------------------------------- summarize

def valid_beacons(receiver, sender):
    """Beacons in receiver's serve log that prove sender's nonce (so nothing else on the LAN counts)."""
    nonce = sender["self"].get("nonce")
    n, srcs = 0, set()
    for e in receiver["serve_log"].get("events", []):
        if e.get("kind") != "beacon" or e.get("from_role") != sender["self"].get("role") or not nonce:
            continue
        if hmac.compare_digest(str(e.get("tag")), mac(nonce, f"beacon|{sender['self']['role']}|{e.get('seq')}|{e.get('ifip')}")):
            n += 1
            srcs.add(e.get("peer"))
    return n, sorted(srcs)


def pair_verdict(x, y):
    rx, ry = x["self"].get("role"), y["self"].get("role")
    cx = x.get("client", {}).get("results", {})
    cy = y.get("client", {}).get("results", {})
    v = {"pair": f"{rx}-{ry}", "hosts": [x["self"].get("runner_name") or x["self"].get("hostname"),
                                          y["self"].get("runner_name") or y["self"].get("hostname")]}
    same_host = x["self"].get("hostname") == y["self"].get("hostname") or bool(set(x["self"].get("macs", [])) & set(y["self"].get("macs", [])))
    tcp_xy = any(r.get("ok") for r in cx.get(ry, {}).get("tcp", {}).values())
    tcp_yx = any(r.get("ok") for r in cy.get(rx, {}).get("tcp", {}).values())
    udp_xy = any(r.get("acked") for r in cx.get(ry, {}).get("udp", {}).values())
    udp_yx = any(r.get("acked") for r in cy.get(rx, {}).get("udp", {}).values())
    mc_xy, src_xy = valid_beacons(y, x)
    mc_yx, src_yx = valid_beacons(x, y)
    v.update(tcp = {f"{rx}->{ry}": tcp_xy, f"{ry}->{rx}": tcp_yx}, udp = {f"{rx}->{ry}": udp_xy, f"{ry}->{rx}": udp_yx},
             multicast_beacons = {f"{rx}->{ry}": mc_xy, f"{ry}->{rx}": mc_yx}, beacon_sources = {rx: src_xy, ry: src_yx},
             rtt_us = cx.get(ry, {}).get("rtt_us") or cy.get(rx, {}).get("rtt_us"),
             bandwidth = cx.get(ry, {}).get("bandwidth") or cy.get(rx, {}).get("bandwidth"),
             found = {rx: ry in cx, ry: rx in cy}, same_host = same_host,
             subnets = {rx: x["self"].get("ipv4"), ry: y["self"].get("ipv4")})
    if ry not in cx and rx not in cy:
        v["verdict"] = "NOT_CONCURRENT"
    elif same_host:
        v["verdict"] = "SAME_MACHINE"
    elif not (tcp_xy or tcp_yx):
        v["verdict"] = "ISOLATED"
    elif mc_xy and mc_yx:
        v["verdict"] = "SAME_L2_MULTICAST_OK"
    else:
        v["verdict"] = "UNICAST_ONLY"
    return v


def cmd_summarize(a):
    results = {}
    for root, _, files in os.walk(a.inp):
        if "result.json" in files:
            r = read_json(os.path.join(root, "result.json"))
            role = r.get("self", {}).get("role")
            if role:
                results[role] = r
    os.makedirs(a.out, exist_ok = True)
    pairs = []
    roles = sorted(results)
    for i, x in enumerate(roles):
        for y in roles[i + 1:]:
            pairs.append(pair_verdict(results[x], results[y]))
    linux = next((p for p in pairs if p["pair"] == "a-b"), None)
    verdict = linux["verdict"] if linux else ("INCONCLUSIVE" if len(results) < 2 else "NOT_CONCURRENT")
    data = {"verdict": verdict, "roles_reported": roles, "pairs": pairs,
            "runners": {r: results[r]["self"].get("runner_name") for r in roles}}
    write_json(os.path.join(a.out, "verdict.json"), data)
    lines = [f"# LAN probe: VERDICT: {verdict}", "", f"Jobs reporting: {', '.join(roles) or 'none'}", ""]
    for p in pairs:
        lines += [f"## {p['pair']}: {p['verdict']}", f"- hosts: {p['hosts']}", f"- tcp: {p['tcp']}", f"- udp: {p['udp']}",
                  f"- multicast beacons (valid, TTL 1): {p['multicast_beacons']}", f"- rtt_us: {p['rtt_us']}",
                  f"- bandwidth: {p['bandwidth']}", f"- addresses: {p['subnets']}", ""]
    if "w" not in results:
        lines.append("Windows job (w) did not report: no mixed-OS answer from this run.")
    with open(os.path.join(a.out, "VERDICT.md"), "w", encoding = ENC) as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest = "cmd", required = True)
    s = sub.add_parser("serve"); s.add_argument("--role", required = True); s.add_argument("--out-dir", required = True)
    s.add_argument("--window-s", type = float, default = 900)
    c = sub.add_parser("client"); c.add_argument("--role", required = True); c.add_argument("--out-dir", required = True)
    c.add_argument("--peers", required = True); c.add_argument("--linux-peers", default = "")
    c.add_argument("--peer-dir", default = ""); c.add_argument("--peer-wait-s", type = float, default = 660)
    k = sub.add_parser("collect"); k.add_argument("--out-dir", required = True)
    m = sub.add_parser("summarize"); m.add_argument("--in", dest = "inp", required = True); m.add_argument("--out", required = True)
    a = ap.parse_args()
    {"serve": cmd_serve, "client": cmd_client, "collect": cmd_collect, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
