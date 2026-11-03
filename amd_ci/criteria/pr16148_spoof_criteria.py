TITLE = 'AMD HIP device multiplier: two logical ordinals and filtered-topology differential'
MODE = 'differential'
NEEDS = ['rocm', 'gpu', 'multi_gpu']


def gates(obs):
    checks = []
    for name in ('base', 'head'):
        o = obs.get(name, {})
        w = o.get('live', {}).get('wiring', {})
        devices = w.get('devices', [])
        checks.append((name + ' tests ran', o.get('rc') in (0, 1) and ' passed' in o.get('stdout', ''), o.get('stderr', '')))
        engaged = w.get('spoofed_devices') == '1' and w.get('device_count') == 2
        checks.append((name + ' HIP multiplier engaged', engaged, str(w)))
        correct = len(devices) == 2 and all(d.get('passed') and d.get('current_device') == i and d.get('result_device') == f'cuda:{i}' for i, d in enumerate(devices))
        checks.append((name + ' real matmul on both logical ordinals', correct, str(devices)))
    head = obs.get('head', {}).get('live', {})
    checks.append(('head opens actual render nodes', bool(head.get('properties')) and head.get('openable') is True, str(head.get('openable'))))
    return checks


def base_shows_defect(o):
    return o.get('rc') == 1 and 'FAILED tests/test_rocm_kfd_runtime_filtered_nodes.py::test_a_render_node_missing_from_the_container_declines' in o.get('stdout', '')


def head_is_fixed(o):
    return o.get('rc') == 0 and o.get('live', {}).get('openable') is True


def table(obs):
    rows = ['| state | pytest rc | real render nodes openable |', '|---|---|---|']
    for name in ('base', 'head'):
        o = obs.get(name, {})
        rows.append(f"| {name} | {o.get('rc')} | {o.get('live', {}).get('openable')} |")
    return '\n'.join(rows) + '\n\nThe two-board filtered topology is simulated. Actual host render nodes are opened on gfx1151; heterogeneous physical HIP renumbering is not measured. Both logical ordinals must execute a real matmul; they alias the same physical GPU.'
