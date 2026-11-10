"""Exact finite discounted expected-cost planning, separate from robust APIs."""
from fractions import Fraction as F
from itertools import product, combinations
from math import lcm
from .core import _compile, _canonical, rational, pack, _keys, _linear_solve


def compile_problem(p):
    _keys(p, ['version', 'model', 'initial', 'costs', 'budget'])
    if type(p['version']) is not int or p['version'] != 1: raise ValueError('version must be 1')
    g, rows, den, _ = _compile(p['model']); n = len(rows)
    if n > 4 or sum(map(len, rows)) > 8: raise ValueError('occupancy limited to 4 states and 8 actions total')
    mu = p['initial']; costs = p['costs']
    if type(mu) is not list or len(mu) != n: raise ValueError('one initial probability per state')
    mu = list(map(rational, mu))
    if any(x < 0 for x in mu) or sum(mu) != 1: raise ValueError('initial distribution must be nonnegative and sum exactly to one')
    if type(costs) is not list or len(costs) != n: raise ValueError('cost rows must match states')
    cs = []
    for costs_row, row in zip(costs, rows):
        if type(costs_row) is not list or len(costs_row) != len(row): raise ValueError('one cost per action')
        cr = list(map(rational, costs_row))
        if any(abs(x) > 100 for x in cr): raise ValueError('absolute costs limited to 100')
        cs.append(cr)
    b = rational(p['budget'])
    if abs(b) > 100: raise ValueError('absolute normalized budget limited to 100')
    for x in mu + [b] + sum(cs, []): den = lcm(den, x.denominator)
    if den > 1000000: raise ValueError('combined denominator LCM exceeds 1000000')
    return g, rows, mu, cs, b


def _canon(g, rows, mu, cs, b):
    return dict(version=1, model=_canonical(g, rows), initial=list(map(pack, mu)), costs=[list(map(pack, c)) for c in cs], budget=pack(b))


def _deterministic(g, rows, mu, cs, solve=_linear_solve):
    n = len(rows); out = []
    for policy in product(*(range(len(row)) for row in rows)):
        a = [[F(i == j) for j in range(n)] for i in range(n)]
        for s, k in enumerate(policy):
            for t, prob in rows[s][k][1]: a[t][s] -= g * prob
        d = solve(a, [(1-g)*x for x in mu])
        x = [d[s] if k == policy[s] else F(0) for s, row in enumerate(rows) for k in range(len(row))]
        r = sum(d[s]*rows[s][k][0] for s, k in enumerate(policy)); c = sum(d[s]*cs[s][k] for s, k in enumerate(policy))
        out.append(dict(actions=list(policy), occupancy=x, reward=r, cost=c))
    return out


def _cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])


def _hull(points):
    points = sorted(set(points))
    if len(points) <= 1: return points
    chains = []
    for seq in (points, list(reversed(points))):
        chain = []
        for p in seq:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], p) <= 0: chain.pop()
            chain.append(p)
        chains.append(chain)
    return chains[0][:-1]+chains[1][:-1]


def _clip(hull, b):
    pts = [p for p in hull if p[0] <= b]
    for p, q in zip(hull, hull[1:]+hull[:1]):
        if min(p[0], q[0]) < b < max(p[0], q[0]):
            w = (b-p[0])/(q[0]-p[0]); pts.append((b, p[1]+w*(q[1]-p[1])))
    return _hull(pts)


def _witness(point, det):
    direct = [i for i, d in enumerate(det) if (d['cost'], d['reward']) == point]
    if direct: return [(direct[0], F(1))]
    for i, j in combinations(range(len(det)), 2):
        p = det[i]['cost'], det[i]['reward']; q = det[j]['cost'], det[j]['reward']
        axis = 0 if p[0] != q[0] else 1
        if p[axis] == q[axis]: continue
        w = (point[axis]-p[axis])/(q[axis]-p[axis])
        if 0 < w < 1 and all(p[k]+w*(q[k]-p[k]) == point[k] for k in (0,1)): return [(i, 1-w), (j, w)]
    raise AssertionError('missing convex witness')


def _value(g, rows, policy, payoffs, solve):
    n = len(rows); a = [[F(i == j) for j in range(n)] for i in range(n)]
    for s, k in enumerate(policy):
        for t, prob in rows[s][k][1]: a[s][t] -= g*prob
    return solve(a, [payoffs[s][k] for s, k in enumerate(policy)])


def _assemble(g, rows, mu, cs, b, det, hull, clipped, solve):
    n = len(rows); flat = [(s,k) for s,row in enumerate(rows) for k in range(len(row))]
    def point_row(p):
        mix = _witness(p, det)
        return dict(cost=pack(p[0]), reward=pack(p[1]), mixture=[dict(policy=i, weight=pack(w)) for i,w in mix])
    cost_values = [_value(g, rows, d['actions'], cs, solve) for d in det]
    min_values = [min(v[s] for v in cost_values) for s in range(n)]
    lower = (1-g)*sum(m*v for m,v in zip(mu,min_values))
    min_slacks = [cs[s][k]+g*sum(prob*min_values[t] for t,prob in rows[s][k][1])-min_values[s] for s,k in flat]
    assert all(v >= 0 for v in min_slacks) and lower == min(d['cost'] for d in det)
    min_cert = dict(values=list(map(pack,min_values)), slacks=list(map(pack,min_slacks)), lower_bound=pack(lower), budget_gap=pack(lower-b))
    det_rows = [dict(index=i, actions=d['actions'], occupancy=list(map(pack,d['occupancy'])), cost=pack(d['cost']), reward=pack(d['reward']), feasible=d['cost'] <= b) for i,d in enumerate(det)]
    feasible_det = [d['reward'] for d in det if d['cost'] <= b]
    report = dict(format='bellmanlab-occupancy-v1', problem=_canon(g,rows,mu,cs,b), actions=[dict(state=s,action=k) for s,k in flat], deterministic=det_rows, hull=[point_row(p) for p in hull], feasible_hull=[point_row(p) for p in clipped], min_cost_certificate=min_cert, feasible=bool(clipped), best_deterministic_reward=pack(max(feasible_det)) if feasible_det else None, optimal_vertices=[], selected=None)
    if not clipped: return report
    optimum = max(p[1] for p in clipped); ties = [i for i,p in enumerate(clipped) if p[1] == optimum]
    chosen = min(ties, key=lambda i: (clipped[i][0], i)); p = clipped[chosen]; mix = _witness(p,det)
    x = [sum(w*det[i]['occupancy'][j] for i,w in mix) for j in range(len(flat))]
    d = [sum(x[j] for j,(s,k) in enumerate(flat) if s == t) for t in range(n)]
    pi = [[x[flat.index((s,k))]/d[s] if d[s] else F(k == 0) for k in range(len(row))] for s,row in enumerate(rows)]
    balances = []
    for t in range(n):
        inj = (1-g)*mu[t]; incoming = g*sum(x[j]*dict(rows[s][k][1]).get(t,F(0)) for j,(s,k) in enumerate(flat))
        balances.append(dict(state=t, initial=pack(inj), incoming=pack(incoming), outgoing=pack(d[t]), residual=pack(d[t]-inj-incoming)))
    candidates = {F(0)}
    for a, z in combinations(hull,2):
        if a[0] != z[0]:
            lam = (a[1]-z[1])/(a[0]-z[0])
            if lam >= 0: candidates.add(lam)
    lam = next(l for l in sorted(candidates) if max(z[1]-l*z[0] for z in hull)+l*b == optimum)
    payoffs = [[r-lam*cs[s][k] for k,(r,t) in enumerate(row)] for s,row in enumerate(rows)]
    values = [_value(g,rows,a['actions'],payoffs,solve) for a in det]
    v = [max(z[s] for z in values) for s in range(n)]
    slacks = [v[s]-rows[s][k][0]+lam*cs[s][k]-g*sum(prob*v[t] for t,prob in rows[s][k][1]) for s,k in flat]
    upper = lam*b+(1-g)*sum(m*w for m,w in zip(mu,v))
    # Re-solve the extracted stationary policy, not the same global mixture weights per state.
    a = [[F(i == j) for j in range(n)] for i in range(n)]
    for s,row in enumerate(rows):
        for k,(r,trans) in enumerate(row):
            for t,prob in trans: a[t][s] -= g*pi[s][k]*prob
    replay_d = solve(a,[(1-g)*m for m in mu])
    replay_x = [replay_d[s]*pi[s][k] for s,k in flat]
    assert x == replay_x and all(z >= 0 for z in slacks) and not any(a*z for a,z in zip(x,slacks)) and not lam*(b-p[0]) and upper == optimum
    assert all(row['residual'] == [0,1] for row in balances)
    dual = dict(multiplier=pack(lam), values=list(map(pack,v)), action_slacks=list(map(pack,slacks)), occupied_slack_products=list(map(pack,[a*z for a,z in zip(x,slacks)])), budget_slack=pack(b-p[0]), budget_product=pack(lam*(b-p[0])), upper_bound=pack(upper), gap=pack(upper-optimum))
    report['optimal_vertices'] = ties
    report['selected'] = dict(vertex=chosen, **point_row(p), occupancy=list(map(pack,x)), state_mass=list(map(pack,d)), policy=[list(map(pack,row)) for row in pi], zero_mass_states=[s for s in range(n) if not d[s]], balances=balances, replay_occupancy=list(map(pack,replay_x)), discounted_reward=pack(p[1]/(1-g)), discounted_cost=pack(p[0]/(1-g)), discounted_budget=pack(b/(1-g)), gain_over_deterministic=pack(optimum-max(feasible_det)) if feasible_det else None, dual=dual)
    return report


def occupancy(problem):
    g,rows,mu,cs,b = compile_problem(problem)
    det = _deterministic(g,rows,mu,cs); hull = _hull([(d['cost'],d['reward']) for d in det])
    return _assemble(g,rows,mu,cs,b,det,hull,_clip(hull,b),_linear_solve)
