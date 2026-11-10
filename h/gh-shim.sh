#!/usr/bin/env bash
# Serves the newest llama Windows cpu zips with the OpenMP DLL renamed to libomp.dll (FAKE_MODE=rename)
# or fails those two downloads (FAKE_MODE=fail). Everything else goes to the real gh.
args="$*"
case "$args" in
  "release download "*"--repo unslothai/llama.cpp --pattern app-"*"-windows-"*"-cpu.zip"*)
    [ "${FAKE_MODE:-rename}" = fail ] && exit 1
    f="$(printf '%s' "$args" | grep -oE 'app-[^ ]*-cpu\.zip')"; cat "$RENAMED_DIR/$f"; exit 0 ;;
esac
exec "$REAL_GH" "$@"
