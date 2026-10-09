# gen_leg.py <workflow.yml> <job id> <src repo> <ref> <tag> <out.yml> <new job id>: copy a prebuilt leg, source cloned instead of downloaded, ccache dropped.
import sys, yaml, copy, re
wf, job, repo, ref, tag, out, newid = sys.argv[1:8]
j = copy.deepcopy(yaml.safe_load(open(wf))['jobs'][job])
steps = []
for s in j['steps']:
    n = s.get('name', '')
    if n == 'Checkout mirror (tooling)':
        steps.append({'name': 'Checkout tree (tooling)', 'uses': s['uses'], 'with': {'repository': repo, 'ref': ref, 'path': 'tooling', 'fetch-depth': 1}})
    elif n.startswith('Download source'):
        continue
    elif n == 'Extract source':
        steps.append({'name': 'Source tree (as resolve prepares it)', 'run': f'set -eux\ngit clone -q https://github.com/{repo} src\ncd src\ngit checkout -q {ref}\ngit submodule update --init --recursive --depth 1 ggml\nfor p in scripts/unsloth/ggml-patches/*.patch; do [ -e "$p" ] || continue; git -C ggml apply --verbose "../$p"; done\ngit log --oneline -1\n'})
    elif n in ('ccache key', 'ccache', 'Evict stale ccache files', 'Save ccache'):
        continue
    else:
        s = copy.deepcopy(s)
        if 'run' in s:
            s['run'] = re.sub(r' \\\n\s*-DCMAKE_(C|CXX|HIP|CUDA)_COMPILER_LAUNCHER=ccache(?= \\\n|\n|$)', '', s['run'])
            s['run'] = re.sub(r'\s*-DCMAKE_(C|CXX|HIP|CUDA)_COMPILER_LAUNCHER=ccache \\\n', '\n', s['run'])
        steps.append(s)
j['steps'] = steps
for k in ('needs', 'if', 'continue-on-error'): j.pop(k, None)
j['name'] = newid; j['timeout-minutes'] = 360
txt = yaml.safe_dump({newid: j}, sort_keys=False, width=200)
txt = txt.replace('${{ needs.resolve.outputs.tag }}', tag).replace('${{ needs.resolve.outputs.commit }}', ref)
open(out, 'w').write(txt)
