TITLE = 'Render-node routing on AMD: synthetic filtered topology and real host opens'
MODE = 'differential'
NEEDS = ['rocm', 'gpu', 'multi_gpu']


def gates(obs):
    return [(name + ' tests ran', o.get('rc') in (0, 1) and ' passed' in o.get('stdout', ''), o.get('stderr', '')) for name, o in obs.items() if name in ('base', 'head')] + [('head opens actual render nodes', bool(obs.get('head', {}).get('live', {}).get('properties')) and obs.get('head', {}).get('live', {}).get('openable') is True, str(obs.get('head', {}).get('live', {})))]


def base_shows_defect(o):
    return o.get('rc') == 1 and 'FAILED tests/test_rocm_kfd_runtime_filtered_nodes.py::test_a_render_node_missing_from_the_container_declines' in o.get('stdout', '')


def head_is_fixed(o):
    return o.get('rc') == 0 and o.get('live', {}).get('openable') is True


def table(obs):
    rows = ['| state | pytest rc | real render nodes openable |', '|---|---|---|']
    for name in ('base', 'head'):
        o = obs.get(name, {})
        rows.append(f"| {name} | {o.get('rc')} | {o.get('live', {}).get('openable')} |")
    return '\n'.join(rows) + '\n\nThe two-board filtered topology is simulated. Actual host render nodes are opened on gfx1151; heterogeneous HIP renumbering is not measured.'
