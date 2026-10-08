import sys, glob, os, math
import numpy as np
from PIL import Image
def load(d):
    fs = sorted(glob.glob(os.path.join(d, "*.png")))
    return fs
a, b = sys.argv[1], sys.argv[2]
fa, fb = load(a), load(b)
assert len(fa) == len(fb) and fa, (len(fa), len(fb))
ps, ident = [], 0
for x, y in zip(fa, fb):
    A = np.asarray(Image.open(x).convert("RGB"), dtype=np.float64)
    B = np.asarray(Image.open(y).convert("RGB"), dtype=np.float64)
    mse = ((A - B) ** 2).mean()
    ident += mse == 0
    ps.append(float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse))
fin = [p for p in ps if p != float("inf")]
wa = [w for w in glob.glob(os.path.join(a, "*.wav"))]
wb = [w for w in glob.glob(os.path.join(b, "*.wav"))]
audio = "n/a"
if wa and wb:
    audio = "identical" if open(wa[0], "rb").read() == open(wb[0], "rb").read() else "differs"
print(f"{os.path.basename(a)} vs {os.path.basename(b)}: frames={len(fa)} identical_frames={ident} "
      f"psnr_mean={'inf' if not fin else round(sum(fin)/len(fin),2)} psnr_min={'inf' if not fin else round(min(fin),2)} audio={audio}")
