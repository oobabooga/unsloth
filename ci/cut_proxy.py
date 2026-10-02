"""An HTTPS CONNECT proxy that closes each Hugging Face connection after CUT_BYTES downstream.

Stands in for a flaky connection: the same cut applies to Studio's own downloader and to
FastFlowLM's curl, which both honour HTTPS_PROXY.
"""

import json
import select
import socket
import threading

CUT_HOSTS = ("huggingface.co", "hf.co")


class CutProxy:
    def __init__(self, cut_bytes: int) -> None:
        self.cut_bytes = cut_bytes
        self.connections: list[dict] = []
        self._lock = threading.Lock()
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target = self._accept, daemon = True).start()

    def summary(self) -> dict:
        with self._lock:
            rows = list(self.connections)
        hf = [r for r in rows if r["cuttable"]]
        return {
            "connections": len(rows),
            "hf_connections": len(hf),
            "hf_cut": sum(1 for r in hf if r["cut"]),
            "hf_bytes": sum(r["down"] for r in hf),
            "hosts": sorted({r["host"] for r in rows}),
        }

    def _accept(self) -> None:
        while True:
            client, _ = self.sock.accept()
            threading.Thread(target = self._handle, args = (client,), daemon = True).start()

    def _handle(self, client: socket.socket) -> None:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = client.recv(4096)
            if not chunk:
                client.close()
                return
            data += chunk
        line = data.split(b"\r\n", 1)[0].decode(errors = "replace")
        parts = line.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            client.sendall(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")
            client.close()
            return
        host, _, port = parts[1].rpartition(":")
        row = {
            "host": host,
            "down": 0,
            "cut": False,
            "cuttable": any(host == h or host.endswith("." + h) for h in CUT_HOSTS),
        }
        with self._lock:
            self.connections.append(row)
        try:
            upstream = socket.create_connection((host, int(port)), timeout = 30)
        except OSError:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            client.close()
            return
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        upstream.settimeout(None)
        try:
            while True:
                ready, _, _ = select.select([client, upstream], [], [], 300)
                if not ready:
                    break
                if client in ready:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    upstream.sendall(chunk)
                if upstream in ready:
                    chunk = upstream.recv(65536)
                    if not chunk:
                        break
                    client.sendall(chunk)
                    row["down"] += len(chunk)
                    if row["cuttable"] and self.cut_bytes and row["down"] >= self.cut_bytes:
                        row["cut"] = True
                        break
        except OSError:
            pass
        finally:
            for s in (client, upstream):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()


if __name__ == "__main__":
    import sys
    import time

    proxy = CutProxy(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(proxy.url, flush = True)
    while True:
        time.sleep(30)
        print(json.dumps(proxy.summary()), flush = True)
