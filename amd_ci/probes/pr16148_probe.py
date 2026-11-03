import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True)
    p.add_argument('--checkout', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.checkout).resolve()
    backend = root / 'studio/backend'
    test = 'test_rocm_kfd_runtime_filtered_nodes.py'
    head_test = root.parent / "head" / "studio/backend/tests" / test
    if head_test.resolve() != (backend / "tests" / test).resolve():
        shutil.copy(head_test, backend / "tests" / test)
    live_code = '''import json, pathlib, sys
sys.path.insert(0, 'studio/backend')
from utils.hardware import hardware
nodes = pathlib.Path('/sys/class/kfd/kfd/topology/nodes')
props = {p.parent.name: p.read_text(encoding='utf-8') for p in nodes.glob('*/properties')}
fn = getattr(hardware, '_rocm_kfd_render_nodes_openable', None)
print('OBS=' + json.dumps({'module': hardware.__file__, 'properties': props, 'openable': fn() if fn else None}))
'''
    live = subprocess.run([sys.executable, '-c', live_code], cwd=root, text=True, capture_output=True, timeout=60)
    obs = {'state': a.state, 'live_rc': live.returncode, 'live_stderr': live.stderr[-2000:]}
    for line in live.stdout.splitlines():
        if line.startswith('OBS='):
            obs['live'] = json.loads(line[4:])
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rf', '-o', 'addopts=', 'tests/' + test, 'tests/test_sd_cpp_vram_budget.py'], cwd=backend, text=True, capture_output=True, timeout=300)
    obs.update(rc=result.returncode, stdout=result.stdout[-20000:], stderr=result.stderr[-3000:])
    Path(a.out).write_text(json.dumps(obs, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
