#!/usr/bin/env python3
"""Judge for probes/r9_m3.py (M3, Vulkan halo CONV_2D / CONV_3D, llama.cpp #32629) on gfx1151.

Kill lines (pre-registered, triage_0.md M3):
  K1 any op failure (test-backend-ops CONV_2D / CONV_3D on the halo build B, against the CPU backend);
  K2 Studio's vae=auto picks im2col on gfx1151 (`conv2d direct auto: Vulkan0 -> off`): the halo shader cannot engage;
  K3 VAE decode under 15% faster. Measured here as an estimate, because the PR cannot be applied to the fork without
     porting its upstream prerequisites: est = (CONV_2D share of the fork's VAE graphs, perf logger) x (1 - B/A) over
     the 1024^2 VAE decoder perf cases, weighted by how many convs of that shape the FLUX / Z-Image VAE decoder runs.
     The B/A delta must also lie outside the A/A spread and B must differ from Boff (kill switch) for it to count;
  K4 output not bit-identical: test-backend-ops is NMSE-gated, so bit identity is NOT measured here (stays unproven;
     the halo path with split_k reorders the K reduction, so bit identity vs the mm shader is not expected).
"""
from __future__ import annotations

import math
import re
import statistics as st

# FLUX.1 ae (Z-Image's VAE) decoder at 1024^2: (IW, IH, Cin, Cout, K) -> same-shape 3x3 convs in one decode (mid 4 +
# level-3 resnets 6 at 128^2; upsample + 6 at 256^2; 5 at 512^2 / 256ch; 5 at 1024^2 / 128ch). The two upsample convs
# at 512^2x512ch and 1024^2x256ch and the channel-changing first convs have no perf case and are left out.
VAE_CASES = {(128, 128, 16, 512, 3): 1, (128, 128, 512, 512, 3): 10, (256, 256, 512, 512, 3): 7,
             (512, 512, 256, 256, 3): 5, (1024, 1024, 128, 128, 3): 5, (1024, 1024, 128, 3, 3): 1}
DECODE_KILL = 0.15
RX_CONV2D = re.compile(r"ne_input=\[(\d+),(\d+),(\d+),(\d+)\],ne_kernel=\[(\d+),(\d+),(\d+),(\d+)\]")


def _vae_key(case: str):
    c = case.replace(" ", "")
    m = RX_CONV2D.search(c)
    if not m or "type_kernel=f16" not in c or "stride0=1," not in c or "cwhn=0" not in c:
        return None
    iw, ih, cin, n, kw, kh, _kc, cout = (int(x) for x in m.groups())
    key = (iw, ih, cin, cout, kw)
    return key if key in VAE_CASES and n == 1 and kw == kh else None


def _per_arm(m3: dict, op: str) -> dict:
    """arm -> case -> [us/run, ...]"""
    out: dict = {}
    for r in m3.get("perf") or []:
        if r.get("op") != op:
            continue
        for case, us in (r.get("cases") or {}).items():
            out.setdefault(r["arm"], {}).setdefault(case, []).append(us)
    return out


def analyse(h: dict) -> dict:
    m3 = h.get("m3") or {}
    a: dict = {}
    t = m3.get("test") or {}
    a["op_fail"] = {k: (v.get("passed"), v.get("total"), v.get("rc")) for k, v in t.items()
                    if k.endswith("_B") and (v.get("passed") is None or v.get("passed") != v.get("total"))}
    a["op_fail_A"] = {k: (v.get("passed"), v.get("total")) for k, v in t.items()
                      if k.endswith("_A") and v.get("passed") != v.get("total")}
    a["device"] = next((v.get("device") for v in t.values() if v.get("device")), None)
    a["coopmat_khr"] = bool(a["device"] and "KHR_coopmat" in a["device"])
    for op in ("CONV_2D", "CONV_3D"):
        arms = _per_arm(m3, op)
        A, B, Boff = arms.get("A") or {}, arms.get("B") or {}, arms.get("Boff") or {}
        cases = sorted(set(A) & set(B))
        ratios, aa, boff = [], [], []
        for c in cases:
            ma, mb = st.median(A[c]), st.median(B[c])
            ratios.append(ma / mb if mb else None)
            aa.append((max(A[c]) - min(A[c])) / ma if ma and len(A[c]) > 1 else None)
            if Boff.get(c):
                boff.append(st.median(Boff[c]) / mb if mb else None)
        good = [r for r in ratios if r]
        a[op] = {"n_cases": len(cases), "geomean_speedup": round(math.exp(sum(map(math.log, good)) / len(good)), 4)
                 if good else None, "aa_med": round(st.median([x for x in aa if x is not None]), 4)
                 if any(x is not None for x in aa) else None,
                 "boff_over_b_geomean": round(math.exp(sum(math.log(x) for x in boff if x) / len([x for x in boff if x])), 4)
                 if any(boff) else None}
        if op == "CONV_2D":
            ta = tb = 0.0
            vae_rows = []
            for c in cases:
                k = _vae_key(c)
                if k is None:
                    continue
                w = VAE_CASES[k]
                ma, mb = st.median(A[c]), st.median(B[c])
                spread = (max(A[c]) - min(A[c])) / ma if len(A[c]) > 1 else None
                ta += w * ma
                tb += w * mb
                vae_rows.append((k, w, round(ma, 1), round(mb, 1), round(ma / mb, 3), round(spread, 3) if spread else None,
                                 round(st.median(Boff[c]), 1) if Boff.get(c) else None))
            a["vae_rows"] = vae_rows
            a["vae_conv_saved"] = round(1 - tb / ta, 4) if ta else None
    fork = m3.get("fork") or {}
    a["vae_conv2d_direct"] = fork.get("vae_conv2d_direct")
    pl = ((fork.get("runs") or {}).get("studio_auto_perflog") or {}).get("perflog") or {}
    a["conv2d_share"] = pl.get("conv2d_share")
    a["decode_est"] = round(a["conv2d_share"] * a["vae_conv_saved"], 4) \
        if a.get("conv2d_share") is not None and a.get("vae_conv_saved") is not None else None
    runs = fork.get("runs") or {}
    a["decode_s"] = {k: v.get("decode_s") for k, v in runs.items()}
    a["fork_sha"] = {k: v.get("sha256") for k, v in runs.items()}
    return a


def verdict(h: dict) -> str:
    m3 = h.get("m3") or {}
    if not m3:
        return "M3: NOT RUN"
    if m3.get("stop") and not m3.get("perf"):
        return f"M3: NOT MEASURED ({m3['stop']})"
    a = analyse(h)
    kills = []
    if a["op_fail"]:
        kills.append(f"K1 op failure on the halo build {a['op_fail']}")
    if a["vae_conv2d_direct"] == "off":
        kills.append("K2 vae=auto picks im2col on gfx1151 (conv2d direct auto -> off): halo cannot engage")
    c2 = a.get("CONV_2D") or {}
    engaged = c2.get("boff_over_b_geomean") is not None and abs(c2["boff_over_b_geomean"] - 1) > (c2.get("aa_med") or 0)
    if a.get("decode_est") is not None and a["decode_est"] < DECODE_KILL:
        kills.append(f"K3 estimated VAE decode gain {a['decode_est']:.1%} < 15% (CONV_2D share {a['conv2d_share']:.1%} "
                     f"x VAE-shape conv time saved {a['vae_conv_saved']:.1%})")
    if not a["coopmat_khr"]:
        kills.append(f"halo needs KHR coopmat; device line: {a['device']}")
    if kills:
        return "M3: KILL: " + "; ".join(kills)
    if a.get("decode_est") is None:
        return f"M3: INCONCLUSIVE (decode estimate missing: share {a.get('conv2d_share')}, saved {a.get('vae_conv_saved')}; " \
               f"vae=auto {a['vae_conv2d_direct']})"
    if not engaged:
        return "M3: INCONCLUSIVE (kill switch GGML_VK_DISABLE_CONV_HALO did not move CONV_2D beyond the A/A spread: halo not shown engaged)"
    return (f"M3: PORT WARRANTED, not proven: estimated decode gain {a['decode_est']:.1%} >= 15% at op level; needs the "
            f"PR + its prerequisites ported into the fork's ggml and a real sd-cli decode ABBA; bit identity unmeasured")


def gates(h: dict, items: list) -> list:
    if "m3" not in items:
        return []
    m3 = h.get("m3") or {}
    out = [("m3 ran", bool(m3), "obs['m3'] present" if m3 else "missing")]
    if not m3:
        return out
    b = m3.get("build") or {}
    out.append(("m3 builds A and B", all((b.get(k) or {}).get("exe") for k in ("A", "B")),
                m3.get("stop") or {k: (v.get("error") or "ok") for k, v in b.items()}))
    out.append(("m3 perf arms recorded", any(r.get("n_cases") for r in m3.get("perf") or []),
                f"{len(m3.get('perf') or [])} perf runs"))
    fork = m3.get("fork") or {}
    out.append(("m3 fork vae=auto decision read", fork.get("vae_conv2d_direct") in ("on", "off"),
                fork.get("stop") or fork.get("error") or fork.get("vae_conv2d_direct")))
    return out


def table(h: dict) -> list:
    m3 = h.get("m3") or {}
    if not m3:
        return []
    L = ["", "M3 Vulkan halo conv (llama.cpp #32629) on gfx1151:", verdict(h)]
    try:
        a = analyse(h)
    except Exception as exc:  # noqa: BLE001
        return L + [f"m3 analyse error {exc!r}"]
    L.append(f"m3 toolchain {m3.get('toolchain')}")
    L.append(f"m3 device {a['device']}; KHR coopmat {a['coopmat_khr']}")
    L.append(f"m3 test-backend-ops failures B {a['op_fail'] or 'none'}; A {a['op_fail_A'] or 'none'}")
    for op in ("CONV_2D", "CONV_3D"):
        L.append(f"m3 {op} perf: {a.get(op)} (speedup = A/B us; boff_over_b = kill-switch arm / B)")
    L.append("m3 VAE 1024^2 rows (shape, weight, A us, B us, A/B, A/A spread, Boff us): " + str(a.get("vae_rows")))
    L.append(f"m3 VAE-shape conv time saved {a.get('vae_conv_saved')}; fork vae=auto conv2d direct "
             f"{a['vae_conv2d_direct']}; CONV_2D share of VAE graphs {a['conv2d_share']}; decode estimate "
             f"{a['decode_est']}")
    L.append(f"m3 fork decode s {a['decode_s']}; png sha {a['fork_sha']}")
    L.append("m3 not measured: bit identity of the halo output and a real sd-cli decode with the halo shader "
             "(the PR does not apply to the fork's ggml without its upstream prerequisites)")
    return L
