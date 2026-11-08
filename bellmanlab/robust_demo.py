"""Specified robust fixtures. No parameter or seed search."""
from .core import evaluate_policy, pack
from .robust import robust_audit, robust_value_iteration
from fractions import Fraction as F


def one_state(rewards):
    return {'version': 1, 'discount': [1, 2], 'states': [{'actions': [
        {'reward': r, 'transitions': [{'next': 0, 'prob': 1}]} for r in rewards]}]}


def robust_demo():
    problem = {'version': 1, 'scenarios': [one_state([[2, 5], [1, 5]]),
                                         one_state([[-2, 5], [1, 5]])]}
    nominal = evaluate_policy(problem['scenarios'][0], [0])
    risky = robust_value_iteration(problem, policy=[0], sweeps=32, tolerance=[1, 1000000])
    safe = robust_value_iteration(problem, policy=[1], sweeps=32, tolerance=[1, 1000000])
    optimal = robust_value_iteration(problem, sweeps=32, tolerance=[1, 1000000])
    assert nominal['values'] == [[4, 5]]
    assert optimal['result']['greedy_policy'] == [1]
    assert all(x['converged'] for x in [risky, safe, optimal])
    # These exact fixed points are independently checked by zero residuals.
    risky_exact = robust_audit(problem, [[-4, 5]], policy=[0])
    safe_exact = robust_audit(problem, [[2, 5]])
    assert risky_exact['residual_sup'] == safe_exact['residual_sup'] == [0, 1]
    coupled = {'version': 1, 'scenarios': []}
    for rewards in ([1, -1], [-1, 1]):
        coupled['scenarios'].append({'version': 1, 'discount': [1, 2], 'states': [
            {'actions': [{'reward': r, 'transitions': [{'next': 1-i, 'prob': 1}]}]}
            for i, r in enumerate(rewards)]})
    globals_ = [evaluate_policy(m, [0, 0])['values'] for m in coupled['scenarios']]
    rectangular = robust_audit(coupled, [-2, -2])
    assert rectangular['residual_sup'] == [0, 1]
    assert min(F(*v[0]) for v in globals_) == F(-2, 3)
    return {'decision_case': {'problem': problem, 'nominal_risky_value': nominal,
                             'robust_risky_iteration': risky, 'robust_safe_iteration': safe,
                             'robust_optimal_iteration': optimal,
                             'exact_risky_audit': risky_exact, 'exact_optimal_audit': safe_exact,
                             'exact_robust_loss_of_nominal_policy': pack(F(6, 5))},
            'rectangularity_case': {'problem': coupled, 'fixed_global_scenario_values': globals_,
                                   'global_scenario_worst_start_value': [-2, 3],
                                   'rectangular_fixed_point_audit': rectangular},
            'selection': 'direct one-state risky +/-2/5 and safe 1/5, gamma1/2; separate alternating two-state rewards +/-1 with opposite global scenarios; no search',
            'scope': 'supplied finite-scenario rectangular closure, not the original coupled global-model uncertainty, a learned confidence region, or real-world safety'}
