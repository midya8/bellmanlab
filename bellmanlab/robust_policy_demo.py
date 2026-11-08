"""One prescribed model, three budget settings. Complete evaluated witnesses."""
from .robust_demo import robust_demo
from .robust_policy import robust_evaluate_policy, robust_policy_iteration


def robust_policy_demo():
    problem=robust_demo()['decision_case']['problem']
    inner=robust_policy_iteration(problem,nature_iterations=1)
    outer=robust_policy_iteration(problem,iterations=1)
    complete=robust_policy_iteration(problem)
    risky=robust_evaluate_policy(problem,[0])
    assert not inner['optimal'] and not inner['policy_value_exact'] and inner['values']==[[4,5]]
    assert inner['history'][0]['improved_policy'] is None
    assert not outer['optimal'] and outer['policy_value_exact'] and outer['values']==[[-4,5]]
    assert outer['policy']==[0] and outer['history'][0]['improved_policy']==[1]
    assert complete['optimal'] and complete['values']==[[2,5]] and complete['policy']==[1]
    assert risky['exact'] and risky['values']==[[-4,5]]
    return {'problem':problem,'inner_capped':inner,'controller_capped':outer,
            'complete':complete,'exact_risky_evaluation':risky,
            'selection':'same prescribed one-state risky +/-2/5, safe1/5, discount1/2 model as robust-demo; no parameter search; only evaluation caps change',
            'scope':'nature cap1 returns evaluated upper bound, withholds controller proposal; controller cap1 returns exact risky value and unevaluated safe proposal separately; completion returns stable controller/nature pair; this is not a convergence-speed comparison'}
