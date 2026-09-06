#!/usr/bin/env python3
"""Plan and render a prebuilt-release differential for scaffold.py --prebuilt:
two release tags on one asset as the states, control arms on the head build,
the fetch/state/control lines for both templates (bash and PowerShell), and
the resolution of a Hub folder into pinned shards. Only `hf_resolve` touches
the network.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import urllib.request
from dataclasses import dataclass, field

DEFAULT_RELEASE_REPO = "unslothai/llama.cpp"
DEFAULT_SENTINEL = ("unsloth/Qwen3-0.6B-GGUF@50968a4468ef4233ed78cd7c3de230dd1d61a56b"
                    ":Qwen3-0.6B-Q4_K_M.gguf")
DEFAULT_PROBE = "amd_ci/probes/llamacpp_server_probe.py"
DEFAULT_CRITERIA = "amd_ci/criteria/llamacpp_server_clean.py"
STEP_INDENT = " " * 10


@dataclass
class Spec:
    """`REPO[@REVISION]:PATH[,PATH...]`; PATH is a file, or a folder to resolve."""
    repo: str
    revision: str
    paths: list[str]
    sizes: dict[str, int] = field(default_factory = dict)

    @property
    def entry(self) -> str:
        """The file llama.cpp is pointed at: shard 1 of a split, or the only file."""
        files = sorted(self.paths)
        for f in files:
            if "-00001-of-" in f:
                return f
        return files[0]

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes.get(p, 0) for p in self.paths)


def parse_spec(spec: str) -> Spec:
    if ":" not in spec:
        raise SystemExit(f"model spec {spec!r} wants REPO[@REVISION]:FILE[,FILE...] or REPO[@REVISION]:FOLDER")
    head, _, tail = spec.partition(":")
    repo, _, rev = head.partition("@")
    if repo.count("/") != 1:
        raise SystemExit(f"model spec {spec!r}: {repo!r} is not an owner/name Hub repo")
    paths = [p.strip().strip("/") for p in tail.split(",") if p.strip()]
    if not paths:
        raise SystemExit(f"model spec {spec!r} names no file or folder")
    return Spec(repo = repo, revision = rev, paths = paths)


def needs_resolution(spec: Spec) -> bool:
    """True when the spec names a folder or pins no revision, so the Hub must be asked."""
    return not spec.revision or any(not p.lower().endswith(".gguf") for p in spec.paths)


def _hub_json(url: str):
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout = 60) as r:
        return json.loads(r.read().decode("utf-8"))


def hf_resolve(spec: Spec) -> Spec:
    """Pin the revision to a commit and expand a folder into its sorted .gguf files."""
    rev = spec.revision
    if not rev:
        try:
            rev = _hub_json(f"https://huggingface.co/api/models/{spec.repo}")["sha"]
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"could not resolve the current revision of {spec.repo}: "
                             f"{type(e).__name__}: {e}. Pin one with REPO@SHA:... or work offline "
                             f"by naming every file") from None
    paths: list[str] = []
    sizes: dict[str, int] = {}
    for p in spec.paths:
        if p.lower().endswith(".gguf"):
            paths.append(p)
            continue
        try:
            tree = _hub_json(f"https://huggingface.co/api/models/{spec.repo}/tree/{rev}/{p}")
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"could not list {spec.repo}@{rev}:{p}: {type(e).__name__}: {e}. "
                             f"Name the .gguf files explicitly to scaffold offline") from None
        found = sorted(e["path"] for e in tree
                       if e.get("type") == "file" and e["path"].lower().endswith(".gguf"))
        if not found:
            raise SystemExit(f"{spec.repo}@{rev}:{p} holds no .gguf file")
        for e in tree:
            if e.get("path") in found and e.get("size"):
                sizes[e["path"]] = int(e["size"])
        paths += found
    return Spec(repo = spec.repo, revision = rev, paths = paths, sizes = sizes)


def asset_file(tag: str, asset: str, windows: bool) -> str:
    """The release asset name for a tag: `app-<tag>-<os>-x64-<asset>.<ext>`.

    An asset that already names its OS, or its extension, is taken as given.
    """
    if asset.endswith((".tar.gz", ".zip", ".tgz")):
        return asset.replace("{tag}", tag)
    if asset.startswith(("linux-", "windows-", "macos-")):
        stem = asset
    else:
        stem = f"{'windows' if windows else 'linux'}-x64-{asset}"
    return f"app-{tag}-{stem}.{'zip' if windows else 'tar.gz'}"


def short_tag(tag: str) -> str:
    return tag.split("-", 1)[0]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def arm_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", text).strip("_")


@dataclass
class Plan:
    base_tag: str
    head_tag: str
    asset: str
    windows: bool
    release_repo: str
    model: Spec
    sentinel: Spec
    env: list[str]
    unset_env: list[str]
    control_env: list[str]
    control_assets: list[str]
    control_tags: list[str]
    probe: str
    criteria: str
    cells: str
    n_predict: int
    load_timeout: int
    timeout_minutes: int
    min_free_gb: int
    title: str = ""
    branch: str = ""

    # ---- derived
    @property
    def rel(self) -> str:
        return f"https://github.com/{self.release_repo}/releases/download"

    @property
    def os_word(self) -> str:
        return "windows" if self.windows else "linux"

    def binaries(self) -> dict[str, tuple[str, str]]:
        """key -> (tag, asset). base/head first, then one per distinct control binary."""
        out = {"base": (self.base_tag, self.asset), "head": (self.head_tag, self.asset)}
        for a in self.control_assets:
            out[arm_name(f"head_{a}").lower()] = (self.head_tag, a)
        for t in self.control_tags:
            out[arm_name(f"{short_tag(t)}_{self.asset}").lower()] = (t, self.asset)
        return out

    def d1_args(self) -> list[str]:
        args: list[str] = []
        for kv in self.env:
            args += ["--env", kv]
        for k in self.unset_env:
            args += ["--unset-env", k]
        return args

    def controls(self) -> list[tuple[str, str, list[str]]]:
        """(arm name, binary key, probe args). Numbered in the order given."""
        out: list[tuple[str, str, list[str]]] = []
        d1_keys = [kv.partition("=")[0] for kv in self.env]
        for spec in self.control_env:
            if spec == "absent":
                if not d1_keys:
                    raise SystemExit("--control-env absent needs at least one --env to be absent")
                args = [x for k in d1_keys + list(self.unset_env) for x in ("--unset-env", k)]
                out.append((f"head_{self.asset}_env_absent", "head", args))
            elif "=" in spec:
                # D1's environment with this one variable overridden.
                k = spec.partition("=")[0]
                args: list[str] = []
                for kv in self.env:
                    args += ["--env", spec if kv.partition("=")[0] == k else kv]
                if k not in d1_keys:
                    args += ["--env", spec]
                for u in self.unset_env:
                    if u != k:
                        args += ["--unset-env", u]
                out.append((f"head_{self.asset}_{spec}", "head", args))
            else:
                raise SystemExit(f"--control-env wants `absent` or K=V, got {spec!r}")
        for a in self.control_assets:
            out.append((f"head_{a}", arm_name(f"head_{a}").lower(), self.d1_args()))
        for t in self.control_tags:
            out.append((f"{short_tag(t)}_{self.asset}", arm_name(f"{short_tag(t)}_{self.asset}").lower(), self.d1_args()))
        return [(f"K{i + 1}_{arm_name(n)}", key, args) for i, (n, key, args) in enumerate(out)]

    def probe_common(self) -> list[str]:
        return ["--gpu-var", "NONE", "--load-timeout", str(self.load_timeout),
                "--cells", self.cells, "--n-predict", str(self.n_predict)]

    def default_title(self) -> str:
        return (f"{short_tag(self.base_tag)} vs {short_tag(self.head_tag)}, {self.os_word} {self.asset}, "
                f"{self.model.repo.rsplit('/', 1)[-1]} {self.model.entry.rsplit('/', 1)[-1]}")

    def default_branch(self) -> str:
        b = f"amd-ci-prebuilt-{short_tag(self.base_tag)}-vs-{short_tag(self.head_tag)}"
        return b + ("-windows" if self.windows else "")


# ---- rendering -----------------------------------------------------------------

def _bash_fetch_prebuilts(plan: Plan) -> str:
    lines = []
    for key, (tag, asset) in plan.binaries().items():
        lines.append(f'python3 amd_ci/lib/fetch_llamacpp.py --url "$REL/{tag}/{asset_file(tag, asset, False)}" \\')
        lines.append(f'  --dest "$AMD_CI_WORK/lcpp/{key}" --out "$O/bin_{key}.json"')
    return "\n".join(STEP_INDENT + l for l in lines)


def _ps_fetch_prebuilts(plan: Plan) -> str:
    lines = []
    for key, (tag, asset) in plan.binaries().items():
        lines.append(f'& $env:AMD_CI_PY "$env:GITHUB_WORKSPACE\\amd_ci\\lib\\fetch_llamacpp.py" '
                     f'--url "$env:REL/{tag}/{asset_file(tag, asset, True)}" '
                     f'--dest "$env:AMD_CI_WORK\\lcpp\\{key}" --out "$O\\bin_{key}.json"')
        lines.append("if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }")
    return "\n".join(STEP_INDENT + l for l in lines)


def _bash_fetch_model(plan: Plan) -> str:
    lines = []
    for i, p in enumerate(sorted(plan.model.paths), 1):
        lines.append(f'python3 amd_ci/lib/fetch_gguf.py --repo "$MODEL_REPO" --file {shlex.quote(p)} \\')
        lines.append(f'  --revision "$MODEL_REV" --local-dir "$M" --out "$AMD_CI_WORK/out/model_shard{i}.json"')
    return "\n".join(STEP_INDENT + l for l in lines)


def _ps_fetch_model(plan: Plan) -> str:
    lines = []
    for i, p in enumerate(sorted(plan.model.paths), 1):
        lines.append(f'& $env:AMD_CI_PY "$env:GITHUB_WORKSPACE\\amd_ci\\lib\\fetch_gguf.py" --repo "$env:MODEL_REPO" '
                     f'--file "{p}" --revision "$env:MODEL_REV" --local-dir "$M" '
                     f'--out "$env:AMD_CI_WORK\\out\\model_shard{i}.json"')
        lines.append("if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }")
    return "\n".join(STEP_INDENT + l for l in lines)


def _bash_controls(plan: Plan) -> str:
    ctl = plan.controls()
    if not ctl:
        return STEP_INDENT + 'echo "no control arms were requested"'
    lines = []
    for name, key, args in ctl:
        lines.append(f'run {name} "${key.upper()}_BIN" ' + " ".join(shlex.quote(a) for a in args))
    return "\n".join(STEP_INDENT + l for l in lines)


def _ps_controls(plan: Plan) -> str:
    ctl = plan.controls()
    if not ctl:
        return STEP_INDENT + 'Write-Host "no control arms were requested"'
    lines = []
    for name, key, args in ctl:
        arr = ", ".join('"' + a.replace('"', '`"') + '"' for a in args) or ""
        lines.append(f'Run-Arm "{name}" $env:{key.upper()}_BIN @({arr})')
    return "\n".join(STEP_INDENT + l for l in lines)


def render(template: str, plan: Plan) -> str:
    title = plan.title or plan.default_title()
    branch = plan.branch or plan.default_branch()
    gb = plan.model.total_bytes / 1e9
    model_desc = (f"{gb:.1f} GB, {len(plan.model.paths)} file(s)" if gb else f"{len(plan.model.paths)} file(s)")
    model_timeout = max(60, int(gb * 2.5)) if gb else 240
    d1_title = (f"D1 {plan.base_tag} vs {plan.head_tag}, {plan.os_word} {plan.asset}"
                + (", " + " ".join(plan.env) if plan.env else "")
                + (", unset " + " ".join(plan.unset_env) if plan.unset_env else ""))
    quote = (lambda xs: " ".join(shlex.quote(x) for x in xs)) if not plan.windows \
        else (lambda xs: " ".join('"' + x + '"' if any(c in x for c in " =") else x for x in xs))
    sub = {
        "__TITLE__": title,
        "__BRANCH__": branch,
        "__SLUG__": slug(branch),
        "__REL__": plan.rel,
        "__MODEL_REPO__": plan.model.repo,
        "__MODEL_REV__": plan.model.revision,
        "__MODEL_ENTRY__": plan.model.entry.rsplit("/", 1)[-1],
        "__MODEL_DESC__": model_desc,
        "__MODEL_TIMEOUT__": str(model_timeout),
        "__SENTINEL_REPO__": plan.sentinel.repo,
        "__SENTINEL_REV__": plan.sentinel.revision,
        "__SENTINEL_FILE__": plan.sentinel.paths[0],
        "__TIMEOUT__": str(plan.timeout_minutes),
        "__MIN_FREE_GB__": str(plan.min_free_gb),
        "__BASE_LABEL__": f"{plan.base_tag} {plan.os_word} {plan.asset}",
        "__HEAD_LABEL__": f"{plan.head_tag} {plan.os_word} {plan.asset}",
        "__PROBE__": plan.probe,
        "__CRITERIA__": plan.criteria,
        "__D1_TITLE__": d1_title,
        "__D1_ARGS__": quote(plan.d1_args()),
        "__PROBE_COMMON__": quote(plan.probe_common()),
        "__FETCH_PREBUILTS__": _ps_fetch_prebuilts(plan) if plan.windows else _bash_fetch_prebuilts(plan),
        "__FETCH_MODEL__": _ps_fetch_model(plan) if plan.windows else _bash_fetch_model(plan),
        "__CONTROLS__": _ps_controls(plan) if plan.windows else _bash_controls(plan),
    }
    text = template
    for k, v in sub.items():
        text = text.replace(k, v)
    left = sorted(set(re.findall(r"__[A-Z_]+__", text)))
    if left:
        raise SystemExit(f"template placeholders left unfilled: {left}")
    return text
