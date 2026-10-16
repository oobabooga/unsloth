"""HTTPS CONNECT proxy that throttles GitHub asset downloads while a flag file exists.

usage: proxy.py PORT RATE_BYTES_PER_S FLAG_PATH LOG_PATH

Only the remote-to-client direction of connections to *.githubusercontent.com is paced,
and only for bytes relayed while FLAG_PATH exists. Everything else passes untouched.
"""

import asyncio
import os
import sys
import time

PORT = int(sys.argv[1])
RATE = int(sys.argv[2])
FLAG = sys.argv[3]
LOG = sys.argv[4]
T0 = time.monotonic()
_next = 0.0
_conn_id = 0


def log(msg: str) -> None:
    with open(LOG, "a", encoding = "utf-8") as fh:
        fh.write(f"[{time.monotonic() - T0:7.1f}s] {msg}\n")


async def pace(n: int) -> None:
    global _next
    now = time.monotonic()
    start = max(now, _next)
    _next = start + n / RATE
    await asyncio.sleep(_next - now)


async def relay(reader, writer, throttled_host: bool, stats: dict, key: str) -> None:
    try:
        while True:
            data = await reader.read(16384 if throttled_host else 262144)
            if not data:
                break
            if throttled_host and RATE > 0 and os.path.exists(FLAG):
                stats["throttled"] += len(data)
                await pace(len(data))
            stats[key] += len(data)
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(creader, cwriter) -> None:
    global _conn_id
    _conn_id += 1
    cid = _conn_id
    try:
        head = await creader.readuntil(b"\r\n\r\n")
    except Exception:
        cwriter.close()
        return
    line = head.split(b"\r\n", 1)[0].decode("latin-1")
    parts = line.split()
    if len(parts) < 2 or parts[0].upper() != "CONNECT":
        cwriter.write(b"HTTP/1.1 501 Not Implemented\r\nContent-Length: 0\r\n\r\n")
        await cwriter.drain()
        cwriter.close()
        log(f"#{cid} rejected non-CONNECT: {line}")
        return
    hostport = parts[1]
    host, _, port = hostport.rpartition(":")
    try:
        rreader, rwriter = await asyncio.open_connection(host, int(port))
    except Exception as exc:
        cwriter.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await cwriter.drain()
        cwriter.close()
        log(f"#{cid} CONNECT {hostport} failed: {exc}")
        return
    cwriter.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await cwriter.drain()
    throttled_host = host.endswith("githubusercontent.com")
    stats = {"down": 0, "up": 0, "throttled": 0}
    started = time.monotonic()
    log(f"#{cid} open {hostport} flag={os.path.exists(FLAG)}")
    await asyncio.gather(
        relay(rreader, cwriter, throttled_host, stats, "down"),
        relay(creader, rwriter, False, stats, "up"),
    )
    log(
        f"#{cid} close {hostport} down={stats['down']} up={stats['up']} "
        f"throttled={stats['throttled']} secs={time.monotonic() - started:.1f}"
    )


async def main() -> None:
    server = await asyncio.start_server(handle, "127.0.0.1", PORT)
    log(f"listening on 127.0.0.1:{PORT} rate={RATE} B/s flag={FLAG}")
    async with server:
        await server.serve_forever()


asyncio.run(main())
