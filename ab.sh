#!/usr/bin/env bash
# Usage: ab.sh WORK SHA ROUNDS CPUS VARIANTS...
# CPUS=real uses the runner's CPU count; a number simulates pre-commit partitioning for that many CPUs.
set -euo pipefail
WORK=$1 SHA=$2 ROUNDS=$3 CPUS=$4; shift 4; VARIANTS=("$@")
H=$(cd "$(dirname "$0")" && pwd)
PY=${AB_PY:-python3}
mkdir -p "$WORK"; cd "$WORK"
if [ ! -d base ]; then
  git init -q base && git -C base fetch -q --depth 1 https://github.com/unslothai/unsloth "$SHA" && git -C base checkout -q FETCH_HEAD
fi
EXCLUDE=$($PY - <<'P'
import re; t=open("base/.pre-commit-config.yaml").read()
b=t[t.index("- id: ruff-format-with-kwargs"):]
print(re.search(r"exclude: '(.*)'", b).group(1))
P
)
for v in "${VARIANTS[@]}"; do
  [ -d "$v" ] && continue
  cp -a base "$v"
  ( cd "$v"
    git -c user.name=ab -c user.email=ab@example.com checkout -q -b ab
    $PY "$H/make_variant.py" "$v" .pre-commit-config.yaml
    git add .pre-commit-config.yaml
    git -c user.name=ab -c user.email=ab@example.com commit -qm "variant $v" --allow-empty
    git tag clean
    $PY "$H/perturb.py" . "$EXCLUDE" 400
    git -c user.name=ab -c user.email=ab@example.com commit -qam perturbed
    git tag dirty
    git reset -q --hard clean
    $PY "$H/pc.py" install-hooks >/dev/null )
done
[ "$CPUS" = real ] && CPUS_ENV="" || CPUS_ENV=$CPUS
export SKIP=ruff,sync-allow-scripts-pins
RES="$WORK/results-$CPUS.tsv"
[ -f "$RES" ] || printf "cpus\tscenario\tround\tvariant\twall\tuser\tsys\trc\tpeak_ruff\tpeak_ruff_threads\tpeak_kwargs_py\tdiff_sha\n" > "$RES"
sample() {  # peak concurrent ruff procs, ruff threads, kwarg-pass pythons while $1 lives
  local pid=$1 pr=0 pt=0 pk=0 r t k
  while kill -0 "$pid" 2>/dev/null; do
    read -r r t < <(ps -eo nlwp=,comm= | awk '$2=="ruff"{n++; s+=$1} END{print n+0, s+0}')
    k=$(pgrep -fc 'enforce_kwargs_spacing|run_ruff_format' || true)
    (( r > pr )) && pr=$r; (( t > pt )) && pt=$t; (( k > pk )) && pk=$k
    sleep 0.02
  done
  echo "$pr $pt $pk"
}
for scen in clean dirty; do
  for r in $(seq 1 "$ROUNDS"); do
    order=("${VARIANTS[@]}")
    (( r % 2 == 0 )) && order=($(printf '%s\n' "${VARIANTS[@]}" | tac))
    for v in "${order[@]}"; do
      cd "$WORK/$v"; git reset -q --hard "$scen"; rm -rf .ruff_cache
      extra=()
      [ "$v" = single_par ] && extra=(env UNSLOTH_RUFF_FORMAT_JOBS=1 RAYON_NUM_THREADS=1)
      [ "$v" = main ] && [ -n "$CPUS_ENV" ] && extra=(env UNSLOTH_RUFF_FORMAT_JOBS=$(( CPUS_ENV < 8 ? CPUS_ENV : 8 )))
      set +e
      AB_CPUS=$CPUS_ENV /usr/bin/time -o /tmp/ab_time -f "%e %U %S" "${extra[@]}" $PY "$H/pc.py" run --all-files > /tmp/ab_out 2>&1 &
      tpid=$!
      peaks=$(sample $tpid)
      wait $tpid; rc=$?
      set -e
      read -r wall usr sys < <(tail -1 /tmp/ab_time)
      dsha=$(git diff "$scen" -- . ':!.ruff_cache' | sha256sum | cut -c1-12)
      read -r pr pt pk <<< "$peaks"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$CPUS" "$scen" "$r" "$v" "$wall" "$usr" "$sys" "$rc" "$pr" "$pt" "$pk" "$dsha" | tee -a "$RES"
      if [ "$r" = 1 ]; then echo "--- $v $scen output:"; grep -E "Passed|Failed|Skipped|files were modified" /tmp/ab_out | head -12; fi
    done
  done
done
