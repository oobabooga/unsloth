# Inside WSL: first boot of the Studio image with generated passwords, the way the README
# quick start runs it (ports only). Writes the passwords into $1 for the Windows side.
set -u
OUTDIR="$1"
mkdir -p "$OUTDIR"
for i in 1 2 3 4 5; do
    docker pull -q unsloth/unsloth:latest && break
    echo "pull attempt $i failed; retrying"; sleep 20
done
docker rm -f uqw >/dev/null 2>&1
t0=$(date +%s)
docker run -d --name uqw -p 18000:8000 -p 18888:8888 unsloth/unsloth:latest
for i in $(seq 1 400); do
    docker logs uqw 2>&1 | grep -q 'container ready' && break
    sleep 3
done
echo "ready after $(( $(date +%s) - t0 ))s"
docker logs uqw 2>&1 | grep -E 'login ->|generated password|container ready|WARN'
docker logs uqw 2>&1 | sed -n 's/.*Unsloth Studio login -> username: unsloth   password: \([^ ]*\).*/\1/p' | tail -1 | tr -d '\n' > "$OUTDIR/studio_pw.txt"
docker logs uqw 2>&1 | sed -n 's/.*generated password: \([^ )]*\).*/\1/p' | head -1 | tr -d '\n' > "$OUTDIR/jupyter_pw.txt"
ls -la "$OUTDIR"
ip -4 addr show eth0 2>/dev/null | grep inet || true
