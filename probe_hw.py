import json, os, platform, shutil, subprocess, sys

out = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout[-6000:]
    except Exception as exc:  # noqa: BLE001
        return f"error {exc}"


if sys.platform == "win32":
    out["cpu"] = sh("wmic cpu get name")
    out["gpu"] = sh("wmic path win32_VideoController get name,driverversion")
    out["npu"] = sh('powershell -NoProfile -Command "Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match \'NPU|IPU|Compute Accelerator\' } | Format-List FriendlyName,Status,InstanceId | Out-String"')
    out["default_home_exists"] = os.path.isdir(os.path.expanduser("~/.unsloth/studio"))
else:
    out["cpu"] = sh("lscpu | head -20")
    out["mem"] = sh("free -g")
    out["lspci"] = sh("lspci -nn | grep -iE 'vga|display|accelerator|signal'")
    out["accel"] = sh("ls -l /dev/accel* ; lsmod | grep -i xdna")
    out["lemonade_pkg"] = sh("dpkg -l | grep -iE 'lemonade|xrt|fastflow'")
    out["which"] = {k: shutil.which(k) for k in ("lemonade-server", "lemond", "flm", "uv", "unsloth")}
open(sys.argv[1], "w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
