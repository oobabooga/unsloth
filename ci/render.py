# Render .github/workflows/sdrev.yml from ci/jobs.json (static matrices, DevLab runners only).
import json
jobs = json.load(open("ci/jobs.json"))
[j.setdefault('script','run.sh') for j in jobs]
lin = [j for j in jobs if j.get("os") != "windows"]; win = [j for j in jobs if j.get("os") == "windows"]
keys = ["name","backend","base","head","tbo","tbo_env","h3","h3_envs","zimg","head_cmake","src_repo","script","variants","steps","reps","h3_steps","h3_ab_reps"]
def mat(js):
    return "\n".join("          - " + "\n            ".join(f"{k}: {json.dumps(str(j.get(k,'')))}" for k in keys) for j in js)
out = """name: sd.cpp PR A/B on AMD

on:
  push:
    branches: [sdrev-amd]

permissions: {}

jobs:
"""
if lin:
    out += """  ab:
    name: ${{ matrix.name }}
    strategy:
      fail-fast: false
      matrix:
        include:
""" + mat(lin) + """
    runs-on: [self-hosted, Linux, strix-halo, devlab-dispatch]
    timeout-minutes: 360
    steps:
      - name: Fetch harness
        run: git clone -q --depth 1 --branch sdrev-amd https://github.com/oobabooga/unsloth "$RUNNER_TEMP/h-$GITHUB_RUN_ID-$$" && echo "H=$RUNNER_TEMP/h-$GITHUB_RUN_ID-$$" >> "$GITHUB_ENV"
      - name: Run
        env:
          BACKEND: ${{ matrix.backend }}
          BASE: ${{ matrix.base }}
          HEAD: ${{ matrix.head }}
          TBO_OPS: ${{ matrix.tbo }}
          TBO_ENV_VARIANTS: ${{ matrix.tbo_env }}
          H3: ${{ matrix.h3 }}
          H3_HEAD_ENVS: ${{ matrix.h3_envs }}
          ZIMG: ${{ matrix.zimg }}
          HEAD_CMAKE: ${{ matrix.head_cmake }}
          SRC_REPO: ${{ matrix.src_repo }}
          VARIANTS: ${{ matrix.variants }}
          STEPS: ${{ matrix.steps }}
          REPS: ${{ matrix.reps }}
          H3_STEPS: ${{ matrix.h3_steps }}
          H3_AB_REPS: ${{ matrix.h3_ab_reps }}
        run: bash "$H/ci/${{ matrix.script }}"
      - name: Upload
        if: always() && env.ART != ''
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
        with:
          name: ${{ matrix.name }}
          path: ${{ env.ART }}
          if-no-files-found: ignore
"""
if win:
    out += """  win:
    name: ${{ matrix.name }}
    strategy:
      fail-fast: false
      matrix:
        include:
""" + mat(win) + """
    runs-on: [self-hosted, Windows, strix-halo, devlab-dispatch]
    timeout-minutes: 360
    steps:
      - name: Fetch harness
        shell: powershell
        run: |
          $h = Join-Path $env:RUNNER_TEMP ("h-" + $env:GITHUB_RUN_ID + "-" + $PID)
          git clone -q --depth 1 --branch sdrev-amd https://github.com/oobabooga/unsloth $h
          Add-Content $env:GITHUB_ENV "H=$h"
      - name: Run
        shell: powershell
        env:
          BASE: ${{ matrix.base }}
          HEAD: ${{ matrix.head }}
          TBO_OPS: ${{ matrix.tbo }}
          H3: ${{ matrix.h3 }}
          H3_HEAD_ENVS: ${{ matrix.h3_envs }}
          ZIMG: ${{ matrix.zimg }}
          HEAD_CMAKE: ${{ matrix.head_cmake }}
        run: '& "$env:H\\\\ci\\\\run.ps1"'
      - name: Upload
        if: always() && env.ART != ''
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
        with:
          name: ${{ matrix.name }}
          path: ${{ env.ART }}
          if-no-files-found: ignore
"""
open(".github/workflows/sdrev.yml","w").write(out)
