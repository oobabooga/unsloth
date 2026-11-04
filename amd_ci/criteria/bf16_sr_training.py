TITLE = "BF16 stochastic optimizer updates"
MODE = "differential"
NEEDS = ["gpu", "rocm", "nvidia", "linux", "windows", "multi_gpu"]


def gates(observations):
    result = []
    for name in ("base", "head"):
        data = observations.get(name, {})
        result.append((f"{name} executed the update", not data.get("error") and data.get("elements") == 65536, str(data.get("error") or data.get("device"))))
        result.append((f"{name} FP32 control updated every weight", data.get("reference_changed") == 65536, str(data.get("reference_changed"))))
    return result


def base_shows_defect(base):
    return base.get("changed") == 0 and base.get("finite") is True


def head_is_fixed(head):
    return (0 < head.get("changed", 0) < 1000 and head.get("finite") is True
            and head.get("variance_dtype") == "torch.float32" and head.get("tests_rc") == 0)


def table(observations):
    rows = ["| state | device | changed BF16 weights | changed FP32 control | variance dtype | test exit |", "|---|---|---:|---:|---|---:|"]
    for name in ("base", "head", "merge"):
        if name in observations:
            data = observations[name]
            rows.append(f"| {name} | {data.get('device')} | {data.get('changed')} | {data.get('reference_changed')} | {data.get('variance_dtype')} | {data.get('tests_rc')} |")
    return "\n".join(rows)
