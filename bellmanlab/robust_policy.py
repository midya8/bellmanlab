"""Exact nested controller/nature policy iteration with explicit inner caps."""
from .core import _policy, _evaluate
from .robust import _compile_robust, _canonical_robust, _certificate
from fractions import Fraction as F


def _limit(value, name):
    if type(value) is not int or not 1 <= value <= 64:
        raise ValueError(name + ' must be integer in 1..64')


def _nature(g, scenarios, controller, initial, iterations):
    # One ordinary action per scenario at each state, restricted to controller.
    rows = [tuple(s[i][a] for s in scenarios) for i, a in enumerate(controller)]
    nature = _policy([0] * len(rows) if initial is None else initial, rows)
    history = []
    previous = None
    exact = False
    for step in range(iterations):
        v = _evaluate(g, rows, nature)
        if previous is not None:
            assert all(x <= y for x, y in zip(v, previous))
        cert = _certificate(g, scenarios, v, controller)
        minimizing = [cert['worst_case_scenarios'][i][a] for i, a in enumerate(controller)]
        new = [old if old in ties else ties[0] for old, ties in zip(nature, minimizing)]
        stable = new == nature
        history.append({'iteration': step, 'nature_policy': nature[:], **cert,
                        'minimizing_scenarios_for_policy': minimizing,
                        'improved_nature_policy': new, 'stable': stable,
                        'linear_policy_equation_residual': [0, 1]})
        if stable:
            assert cert['residual_sup'] == [0, 1]
            exact = True
            break
        if step + 1 < iterations:
            nature = new
            previous = v
    return {'policy': controller[:], 'nature_policy': history[-1]['nature_policy'],
            'values': history[-1]['values'], 'exact': exact,
            'iterations_completed': len(history), 'history': history,
            'result': history[-1],
            'value_status': 'exact robust value of supplied controller' if exact else
                            'last evaluated stationary nature value; componentwise upper bound on supplied controller robust value, not exact',
            'scope': 'nature minimizes for a fixed deterministic controller; exact rational linear equations are solved and checked each iteration; keep old scenario on a minimizing tie; an inner cap returns the last evaluated nature policy, not its unevaluated proposal; residual bound targets the fixed controller robust value, not robust optimality'}


def robust_evaluate_policy(problem, policy, *, initial_nature=None, iterations=64):
    g, scenarios, den, entries = _compile_robust(problem)
    controller = _policy(policy, scenarios[0])
    _limit(iterations, 'iterations')
    return {'format': 'bellmanlab-robust-exact-policy-v1',
            'problem': _canonical_robust(g, scenarios),
            **_nature(g, scenarios, controller, initial_nature, iterations)}


def robust_policy_iteration(problem, *, initial=None, iterations=64, nature_iterations=64):
    g, scenarios, den, entries = _compile_robust(problem)
    _limit(iterations, 'iterations')
    _limit(nature_iterations, 'nature_iterations')
    controller = _policy([0] * len(scenarios[0]) if initial is None else initial, scenarios[0])
    history = []
    previous = None
    optimal = False
    reason = 'controller evaluation cap'
    for step in range(iterations):
        evaluated = _nature(g, scenarios, controller, None, nature_iterations)
        v = [F(*x) for x in evaluated['values']]
        cert = _certificate(g, scenarios, v, None)
        if evaluated['exact'] and previous is not None:
            assert all(x >= y for x, y in zip(v, previous))
        # Do not use an unfinished adversary solve to improve the controller.
        new = None
        stable = False
        if evaluated['exact']:
            new = [old if old in ties else ties[0]
                   for old, ties in zip(controller, cert['greedy_actions'])]
            stable = new == controller
        history.append({'iteration': step, 'policy': controller[:],
                        'evaluation': evaluated, 'optimality_audit': cert,
                        'improved_policy': new, 'stable': stable})
        if not evaluated['exact']:
            reason = 'nature evaluation cap; controller improvement withheld'
            break
        if stable:
            assert cert['residual_sup'] == [0, 1]
            optimal = True
            reason = 'both nature and controller policies stable'
            break
        if step + 1 < iterations:
            controller = new
            previous = v
    last = history[-1]
    return {'format': 'bellmanlab-robust-policy-iteration-v1',
            'problem': _canonical_robust(g, scenarios), 'history': history,
            'iterations_completed': len(history),
            'nature_evaluations_completed': sum(h['evaluation']['iterations_completed'] for h in history),
            'policy': last['policy'], 'nature_policy': last['evaluation']['nature_policy'],
            'values': last['evaluation']['values'],
            'policy_value_exact': last['evaluation']['exact'], 'optimal': optimal,
            'termination': reason,
            'scope': 'nested exact policy evaluation for rectangular finite-scenario control; no controller improvement before a stable inner nature solve; old actions/scenarios retained on exact ties; each cap returns evaluated policies/values and never an unevaluated proposal; stable means robust optimal, not unique; at most 64 controller evaluations and 64 nature evaluations each; no statistical confidence, coupled-uncertainty, online RL or runtime guarantee'}
