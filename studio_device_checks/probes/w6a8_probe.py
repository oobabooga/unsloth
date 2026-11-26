"""#19588 the sm120 fused W6A8 phase tile: does every candidate tile compile and match decode + reference
(`check_tile`), which rules the per-card probe keeps (`fused_config`), and does a packed GEMM call take the fused
path (`fused_call_count`)? usage: w6a8_probe.py --out FILE"""

from __future__ import annotations

import _dc


def main() -> None:
    out = _dc.out_path()
    rec = {"tree": _dc.enter_tree(), **_dc.cuda_info()}
    try:
        import torch

        try:
            import triton

            rec["triton"] = getattr(triton, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            rec["not_run"] = f"Triton is not importable here: {_dc.err(exc, 300)}"
            _dc.write(out, rec)
            return
        if not rec.get("cuda_available"):
            rec["not_run"] = "no CUDA device"
            _dc.write(out, rec)
            return
        from core.inference import diffusion_grouped_int8_gemm as fg

        try:
            from core.inference import diffusion_grouped_int8_gemm_kernels as kernels

            rec["block_ptr_used"] = bool(getattr(kernels, "BLOCK_PTR", True))
            rec["has_block_ptr_removed"] = hasattr(kernels, "block_ptr_removed")
            if hasattr(kernels, "block_ptr_removed"):
                rec["block_ptr_removed"] = bool(kernels.block_ptr_removed())
        except Exception as exc:  # noqa: BLE001
            rec["kernels_import_error"] = _dc.err(exc, 1200)
        cap = tuple(torch.cuda.get_device_capability(0))
        rules = fg.arch_tiles(cap)
        rec["arch_tiles"] = [list(map(str, r)) for r in rules]
        strict = fg.spill_check_needed(cap) if hasattr(fg, "spill_check_needed") else False
        dev = torch.device("cuda", 0)
        tiles = {}
        for rule in rules:
            tile = rule[1]
            key = str(tile)
            if key in tiles:
                continue
            try:
                ok, spills = fg.check_tile(dev, tile, strict)
                tiles[key] = {"ok": bool(ok), "spills": spills}
            except Exception as exc:  # noqa: BLE001
                tiles[key] = {"ok": False, "error": _dc.err(exc, 1500)}
        rec["check_tile"] = tiles
        rec["check_tile_ok"] = bool(tiles) and all(t["ok"] for t in tiles.values())
        rec["fused_config"] = str(fg.fused_config(0))
        # One eligible packed GEMM per probe case: does it run the fused tile, and match the reference?
        before = fg.fused_call_count()
        calls = []
        from core.inference.diffusion_grouped_int8 import decode_int8_torch

        for case in fg.probe_cases(dev):
            args = (case["xq"], case["packed"], case["s_rel"], case["codebook"], case["xs"], case["ws"],
                    case["bias"], case["bits"], case["group"])
            got = fg.grouped_mm_dequant(*args)
            entry = {"rows": int(case["xq"].shape[0]), "bits": case["bits"], "fused": got is not None}
            if got is not None:
                q8 = decode_int8_torch(case["packed"], case["s_rel"], case["codebook"], case["group"], case["bits"])
                ref = fg.reference(case["xq"], q8, case["xs"], case["ws"], case["bias"])
                entry["bit_equal"] = bool(torch.equal(got, ref))
            calls.append(entry)
        rec["packed_calls"] = calls
        rec["fused_call_count_delta"] = fg.fused_call_count() - before
        rec["kernel_state_error"] = getattr(fg, "_STATE", {}).get("error")
    except Exception as exc:  # noqa: BLE001
        rec["fatal"] = _dc.err(exc)
        rec["tb"] = _dc.tb()
    _dc.write(out, rec)


if __name__ == "__main__":
    main()
