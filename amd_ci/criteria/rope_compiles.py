# SPDX-License-Identifier: AGPL-3.0-only
TITLE = 'Stock RoPE fullgraph forward and backward on the accelerator'
MODE = 'differential'
NEEDS = ['rocm', 'windows', 'nvidia', 'multi_gpu']


def gates(obs):
    return [(f'{arm} accelerator observed', bool((obs.get(arm) or {}).get('gpu')),
             str((obs.get(arm) or {}).get('device') or (obs.get(arm) or {}).get('error')))
            for arm in ('base', 'head')]


def base_shows_defect(base):
    cases = base.get('cases', [])
    failed = len(cases) == 6 and all('error' in c for c in cases)
    return failed, 'Base fullgraph errors: ' + str([c.get('error') for c in cases])


def head_is_fixed(head):
    cases = head.get('cases', [])
    ok = len(cases) == 6 and all(c.get('executed') and c.get('finite_grad')
            and c.get('max_error', float('inf')) <= (2e-3 if '16' in c['dtype'] else 1e-5)
            and c.get('grad_error', float('inf')) <= 2e-3 for c in cases)
    return ok, str(cases)


def table(obs):
    return '\n'.join(['| state | cases |', '|---|---|'] +
                     [f'| {arm} | {value.get("cases")} |' for arm, value in obs.items()
                      if isinstance(value, dict) and 'cases' in value])
