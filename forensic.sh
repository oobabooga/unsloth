#!/usr/bin/env bash
# Usage: forensic.sh VARIANT_DIR SCENARIO -- runs each formatter hook alone, in config order,
# and reports how many files that hook changed and whether the tree ends where main's hook ends.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); PY=${AB_PY:-python3}
cd "$1"; git reset -q --hard "$2"; rm -rf .ruff_cache
ids=$($PY -c "
import re; t=open('.pre-commit-config.yaml').read()
print(' '.join(i for i in re.findall(r'- id: (\S+)', t) if i not in ('ruff','sync-allow-scripts-pins')))")
for id in $ids; do
  out=$($PY "$H/pc.py" run "$id" --all-files 2>&1); rc=$?
  n=$(git diff --name-only | wc -l)
  echo "$(basename "$1") $2 hook=$id rc=$rc files_changed_by_this_hook=$n"
  git -c user.name=ab -c user.email=ab@example.com commit -qam "after $id" --allow-empty
done
echo "$(basename "$1") $2 final_tree_sha=$(git diff "$2" | sha256sum | cut -c1-12) files_differing_from_start=$(git diff --name-only "$2" | wc -l)"
git reset -q --hard "$2"
