#!/usr/bin/env python3
"""Probe: inventory of the XDNA2 NPU software stack on this runner. Observes only."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

CMDS = {
    "uname": "uname -a",
    "os_release": "cat /etc/os-release | head -4",
    "lspci_npu": "lspci -nn 2>/dev/null | grep -i -E '1022:17f0|1022:1502|signal processing|npu' || true",
    "dev_accel": "ls -la /dev/accel 2>&1",
    "lsmod_xdna": "lsmod 2>/dev/null | grep -i -E 'amdxdna|xdna' || echo none",
    "modinfo_xdna": "modinfo amdxdna 2>&1 | head -12",
    "sys_accel": "for d in /sys/class/accel/*; do echo $d; cat $d/device/vbnv $d/device/fw_version 2>/dev/null; readlink -f $d/device/driver; done 2>&1",
    "fw_files": "ls /lib/firmware/amdnpu/* 2>&1 | head -20",
    "memlock": "ulimit -l",
    "groups": "id",
    "xrt_dirs": "ls -d /opt/xilinx/xrt /opt/xilinx/* /usr/xrt 2>&1",
    "xrt_smi_which": "command -v xrt-smi || ls /opt/xilinx/xrt/bin/xrt-smi 2>&1",
    "xrt_smi_examine": "(xrt-smi examine || /opt/xilinx/xrt/bin/xrt-smi examine) 2>&1 | head -40",
    "flm_which": "command -v flm 2>&1; ls -la $(command -v flm 2>/dev/null) 2>/dev/null",
    "flm_version": "flm version 2>&1 | head -10; flm --version 2>&1 | head -5",
    "flm_validate": "timeout 60 flm validate 2>&1 | head -40",
    "flm_list": "timeout 60 flm list 2>&1 | head -80",
    "dpkg_npu": "dpkg -l 2>/dev/null | grep -i -E 'xrt|xdna|fastflow|flm|ryzen|lemonade|vitis|aie|onnxruntime' || echo none",
    "snap_flatpak": "snap list 2>/dev/null | grep -i -E 'lemonade|flm' ; true",
    "pip_npu": "python3 -m pip list 2>/dev/null | grep -i -E 'onnxruntime|vitis|ryzen|lemonade|aie|iron|mlir|amd|quark|optimum|diffusers|torch' || echo none",
    "lemonade_which": "command -v lemonade-server lemonade 2>&1; lemonade-server --version 2>&1 | head -3",
    "ryzen_ai_dirs": "ls -d /opt/amd* /opt/ryzen* /opt/RyzenAI* $HOME/ryzen* $HOME/RyzenAI* /usr/local/ryzen* 2>&1",
    "mlir_aie": "ls -d /opt/mlir-aie* /opt/mlir_aie* $HOME/mlir-aie* 2>&1; command -v aiecc.py 2>&1",
    "fastflowlm_dirs": "ls -d /opt/fastflowlm* /usr/share/fastflowlm* /opt/flm* $HOME/.config/flm $HOME/Documents/flm 2>&1; ls $HOME/.config/flm 2>&1 | head",
    "ps_npu": "ps -eo user,pid,etime,cmd 2>/dev/null | grep -i -E 'flm|lemonade|xrt' | grep -v grep || echo none",
    "rocm": "ls -d /opt/rocm* 2>&1; cat /opt/rocm/.info/version 2>&1",
    "cpu": "grep -m1 'model name' /proc/cpuinfo",
    "xdna_params": "for f in /sys/module/amdxdna/parameters/*; do echo $f=$(cat $f 2>&1); done",
    "lemonade_help": "lemonade --help 2>&1 | head -40; lemonade --version 2>&1 | head -3",
    "lemonade_files": "dpkg -L lemonade-server 2>/dev/null | grep -v -E '/share/(doc|man)' | head -60",
    "lemond_ports": "ss -ltnp 2>/dev/null | head -20",
    "lemond_sysinfo": "for p in 8000 13305 8080; do echo port $p; curl -s -m 5 http://127.0.0.1:$p/api/v1/system-info | head -c 3000; echo; curl -s -m 5 http://127.0.0.1:$p/api/v1/health | head -c 1500; echo; done",
    "lemond_models": "for p in 8000 13305; do curl -s -m 5 'http://127.0.0.1:'$p'/api/v1/models?show_all=true' | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(m.get(\"id\"), m.get(\"recipe\"), m.get(\"labels\"), m.get(\"downloaded\")) for m in d.get(\"data\",[])]' 2>&1 | head -200; done",
    "flm_anywhere": "ls -la /usr/bin/flm /usr/local/bin/flm /opt/fastflowlm/bin/flm /var/lib/lemonade 2>&1; find / -xdev -maxdepth 6 \\( -name 'flm' -o -name 'ryzenai-sd-server*' -o -name '*.xclbin' \\) -not -path '/proc/*' 2>/dev/null | head -30",
    "runner": "echo ${RUNNER_NAME:-?}",
}

def run(c):
    try:
        p = subprocess.run(["bash", "-lc", c], capture_output=True, text=True, timeout=150)
        return (p.stdout + p.stderr).strip()[-4000:]
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    obs = {"state": a.state, "cmds": {k: run(v) for k, v in CMDS.items()}}
    obs["has_accel_node"] = os.path.exists("/dev/accel/accel0")
    obs["has_flm"] = shutil.which("flm") is not None
    a.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    sys.exit(main())
