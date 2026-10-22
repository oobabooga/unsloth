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
    git add -A
    git -c user.name=ab -c user.email=ab@example.com commit -qm "variant $v" --allow-empty
    if [ "$v" = ruff_only ]; then  # steady state for plain ruff: the tree ruff itself produces
      SKIP=ruff,sync-allow-scripts-pins $PY "$H/pc.py" run --all-files >/dev/null 2>&1 || true
      git -c user.name=ab -c user.email=ab@example.com commit -qam "ruff-formatted tree"
      echo "ruff_only: reformatted $(git diff --name-only HEAD~1 | wc -l) files"
    fi
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
    if [ "$(uname)" = Linux ]; then read -r r t < <(ps -eo nlwp=,comm= | awk '$2=="ruff"{n++; s+=$1} END{print n+0, s+0}')
    else r=$(ps -Ao comm= | awk '$1 ~ /(^|\/)ruff$/{n++} END{print n+0}'); t=0; fi
    k=$(pgrep -f 'enforce_kwargs_spacing|run_ruff_format|ab_kwargs_both' | wc -l | tr -d ' ')
    (( r > pr )) && pr=$r; (( t > pt )) && pt=$t; (( k > pk )) && pk=$k
    sleep 0.02
  done
  echo "$pr $pt $pk"
}
for scen in clean dirty; do
  for r in $(seq 1 "$ROUNDS"); do
    order=("${VARIANTS[@]}")
    (( r % 2 == 0 )) && order=($(printf '%s\n' "${VARIANTS[@]}" | awk '{a[NR]=$0} END{for(i=NR;i>0;i--)print a[i]}'))
    for v in "${order[@]}"; do
      cd "$WORK/$v"; git reset -q --hard "$scen"; rm -rf .ruff_cache
      extra=()
      [ "$v" = single_par ] && extra=(env UNSLOTH_RUFF_FORMAT_JOBS=1 RAYON_NUM_THREADS=1)
      [ "$v" = main ] && [ -n "$CPUS_ENV" ] && extra=(env UNSLOTH_RUFF_FORMAT_JOBS=$(( CPUS_ENV < 8 ? CPUS_ENV : 8 )))
      set +e
      AB_CPUS=$CPUS_ENV $PY "$H/timeit.py" "$WORK/ab_time" ${extra[@]+"${extra[@]}"} $PY "$H/pc.py" run --all-files > "$WORK/ab_out" 2>&1 &
      tpid=$!
      peaks=$(sample $tpid)
      wait $tpid; rc=$?
      set -e
      read -r wall usr sys < <(tail -1 "$WORK/ab_time")
      dsha=$(git diff "$scen" -- . ':!.ruff_cache' | $PY -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:12])')
      read -r pr pt pk <<< "$peaks"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$CPUS" "$scen" "$r" "$v" "$wall" "$usr" "$sys" "$rc" "$pr" "$pt" "$pk" "$dsha" | tee -a "$RES"
      if [ "$r" = 1 ]; then echo "--- $v $scen output:"; grep -E "Passed|Failed|Skipped|files were modified" "$WORK/ab_out" | head -12; fi
    done
  done
done
