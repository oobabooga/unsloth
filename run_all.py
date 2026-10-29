"""Runs every A/B scenario through ab_probe.py, each in its own process and Studio home.

usage: run_all.py <runner_temp> <driver_dir> <out_dir> <scenario,...>
"""

import os
import subprocess
import sys
from pathlib import Path

temp, driver, out, scenarios = sys.argv[1:5]
log = Path(out) / "run_all.log"
Path(out).mkdir(parents = True, exist_ok = True)
with open(log, "a", encoding = "utf-8") as fh:
    for scenario in scenarios.split(","):
        for label in ("A", "B"):
            env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(Path(temp) / f"home-{label}-{scenario}"))
            fh.write(f"\n##### {label} {scenario}\n")
            fh.flush()
            subprocess.run(
                [
                    sys.executable,
                    str(Path(driver) / "ab_probe.py"),
                    str(Path(temp) / label / "studio" / "backend"),
                    label,
                    str(Path(temp) / "pristine"),
                    out,
                    scenario,
                ],
                stdout = fh,
                stderr = subprocess.STDOUT,
                env = env,
            )
(Path(out) / "done").write_text("done")
