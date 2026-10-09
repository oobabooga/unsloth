# compare.py A B : frames (*.png) and audio (*.wav) of two sd-cli outputs.
import sys, glob, os, struct, math, hashlib
import numpy as np
from PIL import Image

def read_wav(p):
    b = open(p, "rb").read()
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE"
    i, fmt, data = 12, None, None
    while i + 8 <= len(b):
        cid, sz = b[i:i+4], struct.unpack("<I", b[i+4:i+8])[0]
        body = b[i+8:i+8+sz]
        if cid == b"fmt ": fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data": data = body
        i += 8 + sz + (sz & 1)
    tag, ch, sr, _, _, bits = fmt
    if tag == 3: a = np.frombuffer(data, "<f4")
    elif bits == 16: a = np.frombuffer(data, "<i2").astype(np.float64) / 32768
    elif bits == 32 and tag == 1: a = np.frombuffer(data, "<i4").astype(np.float64) / 2**31
    elif bits == 32: a = np.frombuffer(data, "<f4")
    else: raise SystemExit(f"wav fmt {fmt}")
    return a.astype(np.float64)

def psnr(x, y):
    m = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    return float("inf") if m == 0 else 10 * math.log10(255 ** 2 / m)

A, B = sys.argv[1], sys.argv[2]
fa = sorted(glob.glob(os.path.join(A, "*.png"))); fb = sorted(glob.glob(os.path.join(B, "*.png")))
out = []
if len(fa) != len(fb) or not fa:
    out.append(f"frames: count {len(fa)} vs {len(fb)}")
else:
    same = sum(open(x, "rb").read() == open(y, "rb").read() for x, y in zip(fa, fb))
    ps = [psnr(np.asarray(Image.open(x).convert("RGB")), np.asarray(Image.open(y).convert("RGB"))) for x, y in zip(fa, fb)]
    fin = [p for p in ps if p != float("inf")]
    out.append(f"frames {len(fa)}: identical {same}/{len(fa)}" + (f", PSNR mean {np.mean(fin):.2f} min {min(fin):.2f} dB" if fin else ", bit-identical"))
wa = sorted(glob.glob(os.path.join(A, "*.wav"))); wb = sorted(glob.glob(os.path.join(B, "*.wav")))
if wa and wb:
    x, y = read_wav(wa[0]), read_wav(wb[0])
    n = min(len(x), len(y))
    if open(wa[0], "rb").read() == open(wb[0], "rb").read(): out.append("audio: bit-identical")
    else:
        e = np.sum((x[:n] - y[:n]) ** 2); s = np.sum(x[:n] ** 2)
        out.append(f"audio: SNR {10*math.log10(s/e):.2f} dB (len {len(x)} vs {len(y)})")
elif wa or wb:
    out.append(f"audio: present {bool(wa)} vs {bool(wb)}")
print(" | ".join(out))
