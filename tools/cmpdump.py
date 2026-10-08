# Compare two SDREV_DUMP directories: bit-identity per file, else max|d| and SNR.
import sys, os, numpy as np
a, b = sys.argv[1], sys.argv[2]
out = []
for n in ["latent.f32", "vid.f32", "audio_latent.f32", "audio.f32"]:
    pa, pb = os.path.join(a, n), os.path.join(b, n)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        out.append(f"{n}: missing"); continue
    xa, xb = np.fromfile(pa, np.float32), np.fromfile(pb, np.float32)
    if xa.shape != xb.shape:
        out.append(f"{n}: shape {xa.shape} vs {xb.shape}"); continue
    if xa.tobytes() == xb.tobytes():
        out.append(f"{n}: BIT-IDENTICAL ({xa.size})"); continue
    d = xa.astype(np.float64) - xb.astype(np.float64)
    nd = int((xa.view(np.uint32) != xb.view(np.uint32)).sum())
    snr = 10 * np.log10((xa.astype(np.float64) ** 2).sum() / max((d ** 2).sum(), 1e-300))
    out.append(f"{n}: DIFF {nd}/{xa.size} elems, max|d|={np.abs(d).max():.3g}, SNR={snr:.1f} dB")
print(f"{os.path.basename(a.rstrip('/'))} vs {os.path.basename(b.rstrip('/'))}: " + "; ".join(out))
