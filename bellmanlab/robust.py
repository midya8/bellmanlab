"""Exact finite-scenario (state,action)-rectangular robust Bellman operators.

Scenario rows are selectable independently, NOT a globally fixed model index.
"""
from fractions import Fraction as F
from math import lcm
from .core import (_keys, _compile, _canonical, _values, _policy, _pack_values,
                   pack, rational, MAX_DENOMINATOR_LCM)


def _compile_robust(problem):
    _keys(problem, ['version', 'scenarios'])
    if type(problem['version']) is not int or problem['version'] != 1:
        raise ValueError('robust version must be 1')
    scenarios = problem['scenarios']
    if type(scenarios) is not list or not 1 <= len(scenarios) <= 8:
        raise ValueError('1..8 complete scenario models required')
    compiled = [_compile(m) for m in scenarios]
    g, rows, _, _ = compiled[0]
    shape = [len(row) for row in rows]
    den = 1
    entries = 0
    for h, other, d, count in compiled:
        if h != g or [len(row) for row in other] != shape:
            raise ValueError('scenarios must share discount and state/action shape')
        den = lcm(den, d)
        entries += count
    if entries > 2048 or den > MAX_DENOMINATOR_LCM:
        raise ValueError('aggregate uncertainty exceeds 2048 entries or LCM 1000000')
    return g, [c[1] for c in compiled], den, entries


def _canonical_robust(g, scenarios):
    return {'version': 1, 'scenarios': [_canonical(g, rows) for rows in scenarios]}


def _certificate(g, scenarios, v, policy):
    # [state][action][scenario]; each listed row is read exactly once per audit.
    q = [[[rows[i][a][0] + g * sum(p * v[j] for j, p in rows[i][a][1])
           for rows in scenarios] for a in range(len(scenarios[0][i]))]
         for i in range(len(v))]
    worst = [[min(action) for action in state] for state in q]
    greedy = [[a for a, value in enumerate(row) if value == max(row)] for row in worst]
    tv = [max(row) if policy is None else row[policy[i]] for i, row in enumerate(worst)]
    residual = max(abs(x - y) for x, y in zip(v, tv))
    return {'values': _pack_values(v),
            'scenario_q_values': [[_pack_values(action) for action in state] for state in q],
            'worst_case_q_values': [_pack_values(row) for row in worst],
            'worst_case_scenarios': [[[k for k, x in enumerate(action) if x == min(action)]
                                     for action in state] for state in q],
            'greedy_actions': greedy, 'greedy_policy': [row[0] for row in greedy],
            'bellman_values': _pack_values(tv), 'residual_sup': pack(residual),
            'value_error_upper_bound': pack(residual / (1 - g)),
            'target': 'robust optimal value' if policy is None else 'robust value of supplied fixed policy',
            'policy': None if policy is None else policy[:],
            'certificate': 'exact contraction residual/(1-discount) for the stated target; not a greedy-policy loss bound, statistical confidence, or a guarantee outside the supplied rectangular uncertainty set'}


def robust_audit(problem, values, *, policy=None):
    g, scenarios, den, entries = _compile_robust(problem)
    p = None if policy is None else _policy(policy, scenarios[0])
    v = _values(values, scenarios[0], den)
    return {'format': 'bellmanlab-robust-audit-v1',
            'problem': _canonical_robust(g, scenarios), **_certificate(g, scenarios, v, p)}


def robust_value_iteration(problem, *, policy=None, initial=None, sweeps=100, tolerance=None):
    g, scenarios, den, entries = _compile_robust(problem)
    p = None if policy is None else _policy(policy, scenarios[0])
    if type(sweeps) is not int or not 0 <= sweeps <= 256:
        raise ValueError('sweeps must be integer in 0..256')
    tol = rational([1, 1000] if tolerance is None else tolerance)
    if tol < 0:
        raise ValueError('tolerance must be nonnegative')
    v = _values(initial, scenarios[0], den)
    history = []
    for step in range(sweeps + 1):
        cert = _certificate(g, scenarios, v, p)
        history.append({'sweep': step, **cert})
        if F(*cert['value_error_upper_bound']) <= tol or step == sweeps:
            break
        v = [F(*x) for x in cert['bellman_values']]
    return {'format': 'bellmanlab-robust-value-iteration-v1',
            'problem': _canonical_robust(g, scenarios), 'policy': p,
            'tolerance': pack(tol), 'sweeps_completed': len(history) - 1,
            'converged': F(*history[-1]['value_error_upper_bound']) <= tol,
            'history': history, 'result': history[-1],
            'planned_backup_transition_reads': (sweeps + 1) * entries,
            'executed_backup_transition_reads': len(history) * entries,
            'scope': 'independent finite scenario choice at every state/action, with nature observing the selected action; rewards and transitions may both vary; this is the rectangular closure, not a single global scenario; fixed-policy mode still audits all actions but its residual targets only that policy; counters exclude arithmetic bit costs, validation, comparisons and serialization'}
