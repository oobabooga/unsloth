# SPDX-License-Identifier: AGPL-3.0-only
"""Compare local socket filter forms with the live sandbox probe."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studio/backend"))
from core.inference import sandbox_macos as backend
from core.inference.sandbox_probe import probe

original = backend.build_profile
for variant in ("original", "individual-remote", "path-outbound", "path-bind-outbound"):
    def build_profile(*, workdir, private_tmp, runtime_paths, developer_paths=()):
        profile = original(workdir=workdir, private_tmp=private_tmp,
                           runtime_paths=runtime_paths, developer_paths=developer_paths)
        spellings = tuple(p for root in (private_tmp, workdir) for p in backend._sbpl_spellings(root))
        if variant == "individual-remote":
            for path in spellings:
                profile += f'(allow network-outbound (remote unix-socket (subpath {backend._sbpl_string(path)})))\n'
        if variant in ("path-outbound", "path-bind-outbound"):
            profile += backend._rule("allow network-outbound", backend._path_filters((private_tmp, workdir))) + "\n"
        if variant == "path-bind-outbound":
            profile += backend._rule("allow network-bind", backend._path_filters((private_tmp, workdir))) + "\n"
        return profile
    backend.build_profile = build_profile
    print(variant, probe(backend, force=True), flush=True)
backend.build_profile = original
