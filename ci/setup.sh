#!/usr/bin/env bash
# Install Studio once (no torch), build the main and branch frontends, and set up Playwright.
set -euxo pipefail
W="$RUNNER_TEMP/w"
CI="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$W"/{tmp,cache}
export TMPDIR="$W/tmp" UV_CACHE_DIR="$W/cache/uv" npm_config_cache="$W/cache/npm"
export PLAYWRIGHT_BROWSERS_PATH="$W/cache/pw" PIP_CACHE_DIR="$W/cache/pip"
bash "$CI/checkouts.sh" "$W"

curl -fsSL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.xz | tar -xJ -C "$W"
export PATH="$W/node-v22.12.0-linux-x64/bin:$PATH"
for v in main fix; do
  (cd "$W/$v/studio/frontend" && npm ci --no-audit --no-fund --loglevel=error && npx vite build --outDir "$W/dist-$v" --emptyOutDir > "$W/build-$v.log" 2>&1)
  tail -2 "$W/build-$v.log"
done

python3 -m venv "$W/pw"
"$W/pw/bin/pip" -q install playwright pillow
"$W/pw/bin/python" -m playwright install chromium
sudo -n "$W/pw/bin/python" -m playwright install-deps chromium || echo "no sudo for playwright deps"

cd "$W/main"
# The runner's package index can lag the floor main asks for; --local overlays this checkout anyway.
time (UNSLOTH_DESKTOP_BACKEND_VERSION=2026.9.12 UNSLOTH_STUDIO_HOME="$W/inst" UNSLOTH_SKIP_AUTOSTART=1 UNSLOTH_NO_TORCH=1 ./install.sh --local --no-torch < /dev/null > "$W/install.log" 2>&1) || { tail -80 "$W/install.log"; exit 1; }
tail -15 "$W/install.log"
ls "$W/inst/unsloth_studio/bin/python"
