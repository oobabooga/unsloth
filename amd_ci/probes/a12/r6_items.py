"""a12_probe items for round-6 R6-A: ``fa`` (n6#1 flash-attn Triton AMD configs vs Studio SDPA) and ``tune``
(n6#6 TunableOp tables for Wan 5B / LTX-2.5 / H3). Called as ITEMS[name](ctx, obs, run, rounds, HERE)."""
from __future__ import annotations

import json

FA_SHAPES = ("wan5b_27280", "wan5b_5070", "qwen21_4096")


def item_fa(ctx, obs, run, rounds, HERE):
    work, log = ctx["work"], ctx["log"]
    res, winners = [], {}
    # 1. one autotune process per shape: tuning cost, its own steady-state timing, the winning config
    for shape in FA_SHAPES:
        o = work / f"fa_{shape}_fa_autotune.json"
        r = run([HERE / "fa_amd.py", "--shape", shape, "--arm", "fa_autotune", "--out", o], o, log, timeout = 5400)
        r.update({"round": "tune", "label": "fa_autotune"})
        res.append(r)
        if r.get("best_config"):
            winners[shape] = json.dumps(r["best_config"])
        obs["fa"], obs["fa_winners"] = res, winners
    # 2. timed rounds: Studio SDPA, FA default config, FA with this shape's winner, FA with the Wan 5B winner
    #    (one global env var is all flash-attn reads, so the Wan winner is what a shipped setting would apply)
    wan = winners.get("wan5b_27280")
    for shape in FA_SHAPES:
        arms = [("sdpa_studio", None), ("fa_default", None)]
        if winners.get(shape):
            arms.append(("fa_json", winners[shape]))
        if wan and shape != "wan5b_27280":
            arms.append(("fa_json_wan", wan))
        for rnd, (label, cfg) in rounds(arms):
            o = work / f"fa_{shape}_{label}_{rnd}.json"
            args = [HERE / "fa_amd.py", "--shape", shape, "--arm", "fa_json" if cfg else label, "--out", o]
            if cfg:
                args += ["--config-json", cfg]
            r = run(args, o, log, timeout = 1800)
            r.update({"round": rnd, "label": label})
            res.append(r)
            obs["fa"] = res


def item_tune(ctx, obs, run, rounds, HERE):
    work, log = ctx["work"], ctx["log"]
    table = work / "r6_tunableop_gfx1151.csv"
    o = work / "tune_tune.json"
    r = run([HERE / "tunable_r6.py", "tune", "--out", o, "--table", table], o, log, timeout = 7200)
    obs["tune_tune"] = {k: v for k, v in r.items() if k != "results"}
    obs["tune_table_rows"] = r.get("results")
    if not table.is_file():
        obs["tune_error"] = "no table written"
        return
    res = []
    for rnd in range(3):
        o = work / f"tune_bench_{rnd}.json"
        r = run([HERE / "tunable_r6.py", "bench", "--out", o, "--table", table], o, log, timeout = 3600)
        r["round"] = rnd
        res.append(r)
        obs["tune_bench"] = res
    try:
        obs["tune_table_text"] = table.read_text(encoding = "utf-8")
    except OSError:
        pass


ITEMS = {"fa": item_fa, "tune": item_tune}
