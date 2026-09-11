#!/usr/bin/env bash
# Docker-user checks for unsloth/unsloth on a host WITHOUT an NVIDIA GPU (GitHub runners,
# AMD DevLab, macOS via colima). Every check records a RESULT line and the suite continues.
#
#   OUT=dir IMAGE=unsloth/unsloth:latest CORE=unsloth/unsloth:core bash docker_cpu_suite.sh
#
# Skips via env: SKIP_UI=1 SKIP_SSH=1 SKIP_TUNNEL=1 SKIP_OFFLINE=1 SKIP_CORE=1 SKIP_RUNSH=1
# Bash 3.2 compatible (macOS).
set -u

OUT="${OUT:-$PWD/suite_out}"
IMAGE="${IMAGE:-unsloth/unsloth:latest}"
CORE="${CORE:-unsloth/unsloth:core}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
READY_TIMEOUT="${READY_TIMEOUT:-1500}"
SP=18000   # host ports, away from anything the runner uses
JP=18888
mkdir -p "$OUT"
RES="$OUT/results.txt"
: > "$RES"

log() { printf '[suite %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
result() { # id PASS|FAIL|INFO detail
    printf 'RESULT %s %s %s\n' "$1" "$2" "$3" | tee -a "$RES"
}
section() { echo; echo "::group::$*" 2>/dev/null; log "== $*"; }
endsection() { echo "::endgroup::" 2>/dev/null; }

wait_ready() { # container timeout -> 0 ready, 1 incomplete, 2 died/timeout
    local c="$1" t="$2" start now
    start=$(date +%s)
    while :; do
        if docker logs "$c" 2>&1 | grep -q 'Unsloth container ready'; then return 0; fi
        if docker logs "$c" 2>&1 | grep -q 'startup incomplete'; then return 1; fi
        if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" != "true" ]; then return 2; fi
        now=$(date +%s)
        [ $((now - start)) -ge "$t" ] && return 2
        sleep 5
    done
}
studio_pw() { docker logs "$1" 2>&1 | sed -n 's/.*Unsloth Studio login -> username: unsloth   password: \([^ ]*\).*/\1/p' | tail -1; }
jupyter_pw() { docker logs "$1" 2>&1 | sed -n 's/.*generated password: \([^ )]*\).*/\1/p' | head -1; }

section "host"
uname -a
docker version --format 'client {{.Client.Version}} server {{.Server.Version}} os {{.Server.Os}}/{{.Server.Arch}}' 2>&1
docker info 2>&1 | grep -E 'Server Version|Docker Root Dir|Storage Driver|Runtimes|Architecture|CPUs|Total Memory|Operating System|Cgroup' || true
df -h / 2>/dev/null | tail -1
HOST_ARCH="$(docker info --format '{{.Architecture}}' 2>/dev/null)"
case "$HOST_ARCH" in x86_64|amd64) WANT_ARCH=amd64;; aarch64|arm64) WANT_ARCH=arm64;; *) WANT_ARCH="$HOST_ARCH";; esac
result host INFO "arch=$HOST_ARCH $(uname -sr)"
endsection

section "pull"
t0=$(date +%s)
if docker pull -q "$IMAGE" >/dev/null 2>"$OUT/pull.err"; then
    ia="$(docker image inspect -f '{{.Architecture}}' "$IMAGE")"
    rev="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE")"
    sz="$(docker image inspect -f '{{.Size}}' "$IMAGE")"
    [ "$ia" = "$WANT_ARCH" ] && st=PASS || st=FAIL
    result pull_latest "$st" "$(( $(date +%s) - t0 ))s arch=$ia want=$WANT_ARCH rev=$rev size=$((sz/1000000))MB"
else
    result pull_latest FAIL "$(tail -3 "$OUT/pull.err")"
    exit 1
fi
if [ "${SKIP_CORE:-0}" != 1 ]; then
    t0=$(date +%s)
    docker pull -q "$CORE" >/dev/null 2>&1 && result pull_core PASS "$(( $(date +%s) - t0 ))s arch=$(docker image inspect -f '{{.Architecture}}' "$CORE")" || result pull_core FAIL "pull failed"
fi
endsection

if [ "${SKIP_CORE:-0}" != 1 ]; then
section "core without a GPU"
docker run --rm "$CORE" python -c 'print(1)' > "$OUT/core_nogpu.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] && grep -q 'No GPU visible' "$OUT/core_nogpu.log" && grep -q 'UNSLOTH_ALLOW_CPU=1' "$OUT/core_nogpu.log"; then
    result core_refuses_without_gpu PASS "exit=$rc with actionable message"
else
    result core_refuses_without_gpu FAIL "exit=$rc $(head -c 300 "$OUT/core_nogpu.log")"
fi
docker run --rm -e UNSLOTH_ALLOW_CPU=1 -e UNSLOTH_SKIP_NOTEBOOK_SYNC=1 "$CORE" python -c '
import torch, platform, importlib.metadata as m
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), platform.machine())
for p in ("unsloth","unsloth_zoo","transformers","trl","peft","bitsandbytes","xformers","vllm","triton"):
    try: print(p, m.version(p))
    except Exception as e: print(p, "MISSING")
x = torch.randn(256,256); print("cpu matmul", float((x@x).abs().mean()))
' > "$OUT/core_cpu.log" 2>&1
rc=$?
grep -q 'cpu matmul' "$OUT/core_cpu.log" && result core_allow_cpu_torch PASS "$(tr '\n' ' ' < "$OUT/core_cpu.log" | tail -c 400)" || result core_allow_cpu_torch FAIL "rc=$rc $(tail -c 400 "$OUT/core_cpu.log")"
# What a CPU user gets from `import unsloth`
docker run --rm -e UNSLOTH_ALLOW_CPU=1 -e UNSLOTH_SKIP_NOTEBOOK_SYNC=1 "$CORE" python -c 'import unsloth; print("IMPORT_OK")' > "$OUT/core_import_unsloth_cpu.log" 2>&1
result core_import_unsloth_on_cpu INFO "exit=$? $(tail -c 300 "$OUT/core_import_unsloth_cpu.log" | tr '\n' ' ')"
# llama.cpp CPU inference with the bundled binaries
docker run --rm -e UNSLOTH_ALLOW_CPU=1 -e UNSLOTH_SKIP_NOTEBOOK_SYNC=1 "$CORE" bash -c '
set -e
cd /tmp && python - <<EOF
from huggingface_hub import hf_hub_download
p = hf_hub_download("unsloth/gemma-3-270m-it-GGUF", "gemma-3-270m-it-Q4_K_M.gguf", local_dir="/tmp")
print(p)
EOF
/opt/unsloth/llama.cpp/llama-server --list-devices 2>&1 | tail -3
/opt/unsloth/llama.cpp/llama-bench -m /tmp/gemma-3-270m-it-Q4_K_M.gguf -p 64 -n 32 2>&1 | tail -4
' > "$OUT/core_llama_cpu.log" 2>&1
rc=$?
grep -q 'tg32' "$OUT/core_llama_cpu.log" && result core_llama_cpp_cpu PASS "$(grep -E 'tg32|pp64' "$OUT/core_llama_cpu.log" | tr -s ' ' | tr '\n' ' ')" || result core_llama_cpp_cpu FAIL "rc=$rc $(tail -c 600 "$OUT/core_llama_cpu.log")"
# `--user` on core, the documented non-root path, with a bind mount
mkdir -p "$OUT/userwork" && chmod 777 "$OUT/userwork"
docker run --rm --user "$(id -u):$(id -g)" -e UNSLOTH_ALLOW_CPU=1 -v "$OUT/userwork":/workspace/host "$CORE" bash -c 'id; python -c "import torch; open(\"/workspace/host/from_container.txt\",\"w\").write(\"ok\")"; ls -ld /workspace/unsloth-notebooks 2>&1 | head -2; python -c "import IPython.paths as p; print(\"ipythondir\", p.get_ipython_dir())"' > "$OUT/core_user.log" 2>&1
rc=$?
own="$(ls -ln "$OUT/userwork/from_container.txt" 2>/dev/null | awk '{print $3":"$4}')"
[ "$rc" -eq 0 ] && [ "$own" = "$(id -u):$(id -g)" ] && result core_user_flag PASS "rc=0 owner=$own $(tr '\n' ' ' < "$OUT/core_user.log" | tail -c 300)" || result core_user_flag FAIL "rc=$rc owner=$own $(tail -c 500 "$OUT/core_user.log")"
endsection
fi

section "latest: plain docker run -d, no flags beyond ports"
docker rm -f uq >/dev/null 2>&1
t0=$(date +%s)
docker run -d --name uq -p $SP:8000 -p $JP:8888 "$IMAGE" >/dev/null
wait_ready uq "$READY_TIMEOUT"; rc=$?
docker logs uq > "$OUT/latest_first_boot.log" 2>&1
if [ $rc -eq 0 ]; then
    result latest_cpu_ready PASS "$(( $(date +%s) - t0 ))s to ready block"
else
    result latest_cpu_ready FAIL "rc=$rc after $(( $(date +%s) - t0 ))s; $(grep -E 'ERROR|FATAL|Traceback|exited' "$OUT/latest_first_boot.log" | tail -5 | tr '\n' ' ')"
fi
grep -q 'continuing on CPU' "$OUT/latest_first_boot.log" && result latest_cpu_warning PASS "CPU-mode warning printed" || result latest_cpu_warning FAIL "no CPU-mode warning"
SPW="$(studio_pw uq)"; JPW="$(jupyter_pw uq)"
[ -n "$SPW" ] && [ -n "$JPW" ] && result generated_passwords_in_logs PASS "studio=${#SPW} chars jupyter=${#JPW} chars" || result generated_passwords_in_logs FAIL "studio='$SPW' jupyter='$JPW'"
# auth is enforced
c1=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$SP/api/models/list")
c2=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$JP/api/contents")
c3=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$SP/api/health")
{ [ "$c1" = 401 ] || [ "$c1" = 403 ]; } && { [ "$c2" = 403 ] || [ "$c2" = 401 ] || [ "$c2" = 302 ]; } && result auth_enforced PASS "studio models/list=$c1 jupyter contents=$c2 health=$c3" || result auth_enforced FAIL "studio=$c1 jupyter=$c2 health=$c3"
docker exec uq bash -c 'supervisorctl status' > "$OUT/supervisor_status.txt" 2>&1
result supervisor_status INFO "$(tr -s ' ' < "$OUT/supervisor_status.txt" | tr '\n' ';')"
endsection

if [ "${SKIP_UI:-0}" != 1 ]; then
section "Playwright UI (Studio + JupyterLab)"
STUDIO_URL="http://127.0.0.1:$SP" JUPYTER_URL="http://127.0.0.1:$JP" STUDIO_PW="$SPW" \
    STUDIO_NEW_PW="Docker-Suite-2026" JUPYTER_PW="$JPW" EXPECT_GPU=cpu OUT="$OUT/ui" \
    TURN_TIMEOUT_S="${TURN_TIMEOUT_S:-600}" LOAD_TIMEOUT_S="${LOAD_TIMEOUT_S:-1200}" \
    "$PY" "$HERE/docker_ui_probe.py" > "$OUT/ui.log" 2>&1
rc=$?
grep -E 'PASS |FAIL |SUMMARY' "$OUT/ui.log"
[ $rc -eq 0 ] && result ui_probe PASS "$(grep SUMMARY "$OUT/ui.log")" || result ui_probe FAIL "$(grep -E 'FAIL|SUMMARY' "$OUT/ui.log" | tail -4 | tr '\n' ' ' | head -c 900)"
docker exec uq bash -c 'ps -eo pid,pcpu,rss,args | grep -E "[l]lama-server" | cut -c1-250' > "$OUT/llama_server_ps.txt" 2>&1
result llama_server_process INFO "$(head -c 300 "$OUT/llama_server_ps.txt")"
endsection
NEWPW="Docker-Suite-2026"
else
NEWPW=""
fi

section "restart persistence"
t0=$(date +%s)
docker stop uq >/dev/null
stop_s=$(( $(date +%s) - t0 ))
result docker_stop_time "$([ $stop_s -lt 10 ] && echo PASS || echo FAIL)" "${stop_s}s exit=$(docker inspect -f '{{.State.ExitCode}}' uq)"
T1=$(date +%s)
docker start uq >/dev/null
st=0; while ! docker logs --since "$T1" uq 2>&1 | grep -qE 'container ready|startup incomplete'; do sleep 3; st=$((st+3)); [ $st -gt "$READY_TIMEOUT" ] && break; done
docker logs --since "$T1" uq > "$OUT/latest_restart.log" 2>&1
if [ -n "$NEWPW" ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$SP/api/auth/login" -H 'Content-Type: application/json' -d "{\"username\":\"unsloth\",\"password\":\"$NEWPW\"}")
    grep -q 'password set on an earlier boot' "$OUT/latest_restart.log" && [ "$code" = 200 ] && result restart_keeps_password PASS "login=$code after restart ${st}s" || result restart_keeps_password FAIL "login=$code $(grep -E 'Studio  ' "$OUT/latest_restart.log")"
fi
grep -q 'existing jupyter config reused' "$OUT/latest_restart.log" && result restart_keeps_jupyter_pw PASS "" || result restart_keeps_jupyter_pw FAIL "$(grep JupyterLab "$OUT/latest_restart.log" | head -2)"
endsection

section "in-container tools"
docker exec uq unsloth-llama-update --check > "$OUT/llama_update_check.log" 2>&1
result llama_update_check "$([ $? -eq 0 ] && echo PASS || echo FAIL)" "$(tr '\n' ' ' < "$OUT/llama_update_check.log" | tail -c 300)"
docker exec uq bash -c 'unsloth --help >/dev/null 2>&1 && echo cli-ok; /opt/unsloth-studio/bin/unsloth --version 2>&1 | tail -1; ls /opt/unsloth-studio/whisper.cpp/build/bin | head -5' > "$OUT/tools.log" 2>&1
result in_container_tools INFO "$(tr '\n' ' ' < "$OUT/tools.log")"
endsection

if [ "${SKIP_SSH:-0}" != 1 ] && command -v ssh-keygen >/dev/null 2>&1; then
section "SSH (PUBLIC_KEY)"
rm -f "$OUT/ukey" "$OUT/ukey.pub"
ssh-keygen -q -t ed25519 -N '' -f "$OUT/ukey"
docker rm -f uqssh >/dev/null 2>&1
docker run -d --name uqssh -p 12222:22 -e PUBLIC_KEY="$(cat "$OUT/ukey.pub")" -e HF_TOKEN=secret-should-not-leak -e UNSLOTH_MARKER=visible "$IMAGE" >/dev/null
for i in $(seq 1 60); do docker logs uqssh 2>&1 | grep -q 'sshd.*RUNNING\|Server listening' && break; sleep 3; done
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -i $OUT/ukey -p 12222"
ssh $SSHO root@127.0.0.1 'bash -lc "echo WHOAMI=\$(whoami); echo MARKER=\$UNSLOTH_MARKER; echo HFTOK=\${HF_TOKEN:-unset}; which python; python -c \"import torch;print(torch.__version__)\""' > "$OUT/ssh_root.log" 2>&1
rc=$?
grep -q 'WHOAMI=root' "$OUT/ssh_root.log" && result ssh_key_root_login PASS "$(tr '\n' ' ' < "$OUT/ssh_root.log")" || result ssh_key_root_login FAIL "rc=$rc $(tail -c 400 "$OUT/ssh_root.log")"
grep -q 'HFTOK=unset' "$OUT/ssh_root.log" && result ssh_env_secret_filtered PASS "" || result ssh_env_secret_filtered INFO "$(grep HFTOK "$OUT/ssh_root.log")"
# the old docs connect as unsloth@
ssh $SSHO unsloth@127.0.0.1 true > "$OUT/ssh_unsloth.log" 2>&1
result ssh_old_docs_unsloth_user "$([ $? -eq 0 ] && echo PASS || echo FAIL)" "$(tail -c 200 "$OUT/ssh_unsloth.log" | tr '\n' ' ')"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -o PreferredAuthentications=password -o PubkeyAuthentication=no -p 12222 root@127.0.0.1 true > "$OUT/ssh_pw.log" 2>&1
grep -qi 'permission denied' "$OUT/ssh_pw.log" && result ssh_password_auth_refused PASS "$(tail -c 150 "$OUT/ssh_pw.log" | tr '\n' ' ')" || result ssh_password_auth_refused FAIL "$(tail -c 200 "$OUT/ssh_pw.log")"
docker rm -f uqssh >/dev/null 2>&1
endsection
fi

section "old docs full example: -e JUPYTER_PORT=8000 -p 8000:8000 -p 2222:22 + USER_PASSWORD"
docker rm -f uqold >/dev/null 2>&1
mkdir -p "$OUT/work"
docker run -d --name uqold -e JUPYTER_PORT=8000 -e JUPYTER_PASSWORD=mypassword -e USER_PASSWORD=unsloth2024 -p 19000:8000 -p 12223:22 -v "$OUT/work":/workspace/work "$IMAGE" >/dev/null
wait_ready uqold 900; rc=$?
docker logs uqold > "$OUT/old_docs.log" 2>&1
code=$(curl -s -o "$OUT/old_docs_root.html" -w '%{http_code}' "http://127.0.0.1:19000/")
title="$(grep -o '<title>[^<]*' "$OUT/old_docs_root.html" | head -1)"
result old_docs_jupyter_port_8000 INFO "ready_rc=$rc http=$code title='$title' $(grep -E 'Address already in use|port 8000|Errno 98|FATAL|gave up' "$OUT/old_docs.log" | head -4 | tr '\n' ' ')"
docker exec uqold bash -c 'id unsloth 2>&1; ls /home 2>&1' > "$OUT/old_docs_user.log" 2>&1
result old_docs_unsloth_user INFO "$(tr '\n' ' ' < "$OUT/old_docs_user.log")"
docker rm -f uqold >/dev/null 2>&1
endsection

if [ "${SKIP_TUNNEL:-0}" != 1 ]; then
section "Cloudflare tunnel for JupyterLab"
docker rm -f uqcf >/dev/null 2>&1
docker run -d --name uqcf -e UNSLOTH_JUPYTER_CLOUDFLARE=1 -e JUPYTER_PASSWORD=tunnelpw "$IMAGE" >/dev/null
url=""
for i in $(seq 1 100); do url="$(docker logs uqcf 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1)"; [ -n "$url" ] && break; sleep 3; done
if [ -n "$url" ]; then
    sleep 15
    code=000; for i in 1 2 3 4 5 6; do code=$(curl -s -o /dev/null -w '%{http_code}' -L "$url/login"); [ "$code" = 200 ] && break; sleep 10; done
    result jupyter_cloudflare_tunnel "$([ "$code" = 200 ] && echo PASS || echo FAIL)" "url=$url /login=$code"
else
    result jupyter_cloudflare_tunnel FAIL "no trycloudflare URL in logs: $(docker logs uqcf 2>&1 | grep -i 'tunnel\|cloudflared' | tail -3 | tr '\n' ' ')"
fi
docker rm -f uqcf >/dev/null 2>&1
endsection
fi

if [ "${SKIP_OFFLINE:-0}" != 1 ]; then
section "offline start (--network none)"
docker rm -f uqoff >/dev/null 2>&1
t0=$(date +%s)
docker run -d --name uqoff --network none "$IMAGE" >/dev/null
wait_ready uqoff 1200; rc=$?
docker logs uqoff > "$OUT/offline.log" 2>&1
result offline_start "$([ $rc -eq 0 ] && echo PASS || echo FAIL)" "rc=$rc $(( $(date +%s) - t0 ))s $(grep -E 'unsloth-nb|WARN|ERROR' "$OUT/offline.log" | head -4 | tr '\n' ' ')"
docker rm -f uqoff >/dev/null 2>&1
endsection
fi

if [ "${SKIP_RUNSH:-0}" != 1 ]; then
section "docker/run.sh and install_nvidia_toolkit.sh from main"
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/run.sh -o "$OUT/run.sh"
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh -o "$OUT/install_nvidia_toolkit.sh"
# default UNSLOTH_GPUS=all on a host with no NVIDIA GPU: must drop --gpus and still run
( cd "$OUT/work" && UNSLOTH_IMAGE="$CORE" UNSLOTH_ALLOW_CPU=1 HF_HOME="$OUT/hf" TRITON_CACHE_DIR="$OUT/triton" timeout 600 bash "$OUT/run.sh" python -c 'import torch,os;print("RUNSH_OK", torch.cuda.is_available(), os.path.exists("/dev/kfd"), sorted(os.listdir("/dev/dri")) if os.path.isdir("/dev/dri") else None)' ) > "$OUT/runsh.log" 2>&1
rc=$?
grep -q RUNSH_OK "$OUT/runsh.log" && result run_sh_no_nvidia PASS "rc=$rc $(grep -E 'WARN|AMD|RUNSH_OK' "$OUT/runsh.log" | tr '\n' ' ' | head -c 400)" || result run_sh_no_nvidia FAIL "rc=$rc $(tail -c 500 "$OUT/runsh.log")"
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    UNSLOTH_TOOLKIT_VERIFY=1 sudo -E bash "$OUT/install_nvidia_toolkit.sh" > "$OUT/toolkit.log" 2>&1
    rc=$?
else
    bash "$OUT/install_nvidia_toolkit.sh" > "$OUT/toolkit.log" 2>&1
    rc=$?
fi
result toolkit_installer_no_nvidia INFO "rc=$rc $(tr '\n' ' ' < "$OUT/toolkit.log" | tail -c 400)"
endsection
fi

section "cleanup"
docker rm -f uq >/dev/null 2>&1
echo
echo "================ SUITE RESULTS ================"
cat "$RES"
fails=$(grep -c ' FAIL ' "$RES")
echo "fails=$fails"
exit 0
