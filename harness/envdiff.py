import os, sys, json
sys.path.insert(0, os.path.join(sys.argv[1], "studio"))
import install_python_stack as ips
py = sys.executable
P = "https://download.pytorch.org/whl/cu128"
shapes = {
 "uv_pinned": ips._build_uv_cmd(("torch", "--index-url", P)),
 "uv_plain": ips._build_uv_cmd(("torch",)),
 "pip_pinned_install": ips._build_pip_cmd(("torch", "--index-url", P)),
 "pip_plain_install": ips._build_pip_cmd(("torch",)),
 "pip_pinned_download": [py, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "-d", "t", "x", "--index-url", P],
 "pip_wheel": [py, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", "s", "unsloth"],
 "pip_uninstall_wheel": [py, "-m", "pip", "uninstall", "-y", "wheel"],
 "pip_check": [py, "-m", "pip", "check"],
 "uv_uninstall": ["uv", "pip", "uninstall", "--python", py, "x"],
 "pip_cachedir_install": [py, "-m", "pip", "--cache-dir", "c", "install", "x"],
}
out = {}
for k, cmd in shapes.items():
    if hasattr(ips, "_uv_cmd_and_env") and cmd[:1] == ["uv"]:
        cmd, env = ips._uv_cmd_and_env(cmd)
    else:
        env = ips._install_env_for_cmd(cmd)
    out[k] = {"argv": cmd, "env": env}
root = os.path.abspath(sys.argv[1])
text = json.dumps(out, sort_keys=True)
# The module points UV_OVERRIDE (macOS arm64) at files inside its own checkout; compare trees, not their paths.
for spelling in {root, root.replace("\\", "/"), json.dumps(root)[1:-1]}:
    text = text.replace(spelling, "<TREE>")
print("RESULT " + text)
