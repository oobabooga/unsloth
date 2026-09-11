# Run inside the container: start a real ipykernel with the image's IPython profile
# (startup hook, pip magic, sidecar selection) and execute one cell headlessly.
cat > /tmp/k.ipynb <<'EOF'
{"cells":[{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],
  "source":["import torch, platform, sys; print('KERNEL_OK', torch.__version__, platform.machine(), sys.executable)"]}],
 "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}
EOF
cd /tmp
s=$(date +%s)
timeout 600 jupyter nbconvert --to notebook --execute --output /tmp/k_out.ipynb /tmp/k.ipynb 2>&1 | tail -5
echo "rc=${PIPESTATUS[0]} secs=$(( $(date +%s) - s ))"
python - <<'EOF'
import json
nb = json.load(open("/tmp/k_out.ipynb"))
for o in nb["cells"][0].get("outputs", []):
    print("".join(o.get("text", "")) or o)
EOF
