"""Two simultaneous normalized expected-cost budgets, exact full flow vertices.

Separate workflow: no changes to ordinary, robust, or one-budget APIs.
"""
from fractions import Fraction as F
from itertools import combinations, product
from math import lcm
from .core import _compile, _canonical, _keys, rational, pack, _linear_solve


def compile_problem(p):
    _keys(p, ['version', 'model', 'initial', 'costs', 'budgets'])
    if type(p['version']) is not int or p['version'] != 1: raise ValueError('version must be 1')
    g, rows, den, _ = _compile(p['model']); n = len(rows)
    if n > 4 or sum(map(len, rows)) > 8: raise ValueError('coupled limits: 4 states, 8 actions total')
    if type(p['initial']) is not list or len(p['initial']) != n: raise ValueError('one initial probability per state')
    mu = list(map(rational, p['initial']))
    if any(v < 0 for v in mu) or sum(mu) != 1: raise ValueError('initial distribution must sum to one')
    if type(p['costs']) is not list or len(p['costs']) != 2: raise ValueError('exactly two cost tables')
    cs = []
    for table in p['costs']:
        if type(table) is not list or len(table) != n: raise ValueError('cost rows must match states')
        out = []
        for cr, row in zip(table, rows):
            if type(cr) is not list or len(cr) != len(row): raise ValueError('one cost per action')
            out.append(list(map(rational, cr)))
        cs.append(out)
    if type(p['budgets']) is not list or len(p['budgets']) != 2: raise ValueError('exactly two budgets')
    bs = list(map(rational, p['budgets']))
    extras = mu + bs + [v for table in cs for row in table for v in row]
    if any(abs(v) > 100 for v in extras): raise ValueError('absolute costs/budgets <=100')
    for v in extras: den = lcm(den, v.denominator)
    if den > 1000000: raise ValueError('combined denominator LCM exceeds 1000000')
    return g, rows, mu, cs, bs


def canonical(g, rows, mu, cs, bs):
    return dict(version=1, model=_canonical(g, rows), initial=list(map(pack, mu)), costs=[[list(map(pack, row)) for row in table] for table in cs], budgets=list(map(pack, bs)))


def _dot(a, b): return sum((x*y for x, y in zip(a, b)), F(0))


def _system(g, rows, mu, cs):
    flat = [(s,k) for s,row in enumerate(rows) for k in range(len(row))]
    a = [[F(s == t)-g*dict(rows[s][k][1]).get(t,F(0)) for s,k in flat] for t in range(len(rows))]
    return flat, a, [(1-g)*v for v in mu], [[table[s][k] for s,k in flat] for table in cs], [rows[s][k][0] for s,k in flat]


def _vertices(a, b, c, budgets):
    # Add two nonnegative budget slacks. A has full row rank because gamma<1.
    n, m = len(a), len(a[0]); mat = [row+[F(0),F(0)] for row in a]
    mat += [c[j]+[F(j == k) for k in range(2)] for j in range(2)]
    rhs = b+budgets; out = set()
    for basis in combinations(range(m+2), n+2):
        try: z = _linear_solve([[row[j] for j in basis] for row in mat], rhs)
        except ValueError: continue
        if min(z) < 0: continue
        full = [F(0)]*(m+2)
        for j,v in zip(basis,z): full[j] = v
        out.add(tuple(full[:m]))
    return sorted(out)


def _det(g, rows, mu, a, b, c, rewards, flat):
    out = []
    for policy in product(*(range(len(row)) for row in rows)):
        cols = [flat.index((s,k)) for s,k in enumerate(policy)]
        mass = _linear_solve([[row[j] for j in cols] for row in a],b)
        x = [F(0)]*len(flat)
        for j,v in zip(cols,mass): x[j] = v
        out.append(dict(actions=list(policy), x=x, costs=[_dot(row,x) for row in c], reward=_dot(rewards,x)))
    return out


def _values(g, rows, policy, payoff):
    n = len(rows)
    a = [[F(s == t)-g*dict(rows[s][policy[s]][1]).get(t,F(0)) for t in range(n)] for s in range(n)]
    return _linear_solve(a,[payoff[s][k] for s,k in enumerate(policy)])


def _mixture(x, det):
    # Complete full-occupancy witness, not merely matching projected returns.
    for size in (1,2,3):
        for ids in combinations(range(len(det)),size):
            eq = [[F(1)]*size]+[[det[i]['x'][j] for i in ids] for j in range(len(x))]
            rhs = [F(1)]+list(x)
            for rs in combinations(range(len(eq)),size):
                try: w = _linear_solve([eq[j] for j in rs],[rhs[j] for j in rs])
                except ValueError: continue
                if min(w) > 0 and all(_dot(row,w) == v for row,v in zip(eq,rhs)):
                    return [dict(policy=i, weight=pack(v)) for i,v in zip(ids,w)]
    raise AssertionError('no three-policy full-occupancy witness')


def _dual(g, rows, mu, cs, bs, a, c, rewards, det, x, reward):
    n, m = len(rows), len(x)
    constraints = [[a[s][j] for s in range(n)]+[c[k][j] for k in range(2)] for j in range(m)]
    constraints += [[F(0)]*n+[F(k == j) for k in range(2)] for j in range(2)]
    rhs = rewards+[F(0),F(0)]; choices = []
    required = {j for j,v in enumerate(x) if v}
    required.update(m+k for k in range(2) if _dot(c[k],x) < bs[k])
    for ids in combinations(range(m+2),n+2):
        if not required.issubset(ids): continue
        try: z = _linear_solve([constraints[j] for j in ids],[rhs[j] for j in ids])
        except ValueError: continue
        if any(_dot(row,z) < v for row,v in zip(constraints,rhs)): continue
        upper = (1-g)*_dot(mu,z[:n])+_dot(bs,z[n:])
        if upper == reward: choices.append(tuple(z[n:]))
    if not choices: raise AssertionError('dual vertex missing')
    lam = min(choices,key=lambda z:(sum(z),z))
    pay = [[r-sum(lam[j]*cs[j][s][k] for j in range(2)) for k,(r,_) in enumerate(row)] for s,row in enumerate(rows)]
    vals = [_values(g,rows,d['actions'],pay) for d in det]
    v = [max(row[s] for row in vals) for s in range(n)]
    slack = [_dot([a[s][j] for s in range(n)],v)+sum(c[k][j]*lam[k] for k in range(2))-rewards[j] for j in range(m)]
    budget_slack = [bs[k]-_dot(c[k],x) for k in range(2)]
    upper = (1-g)*_dot(mu,v)+_dot(bs,lam)
    assert min(slack) >= 0 and upper == reward
    return dict(multipliers=list(map(pack,lam)),values=list(map(pack,v)),action_slacks=list(map(pack,slack)),occupied_slack_products=[pack(x[j]*slack[j]) for j in range(m)],budget_slacks=list(map(pack,budget_slack)),budget_products=[pack(lam[k]*budget_slack[k]) for k in range(2)],upper_bound=pack(upper),gap=pack(upper-reward))


def _separator(g, rows, mu, cs, bs, det):
    # A separating nonnegative normal of the two-cost projection. Enumerating
    # switch points of min_i lambda.C_i on lambda0+lambda1=1 is complete.
    candidates = {(F(1),F(0)),(F(0),F(1))}
    for p,q in combinations(det,2):
        d = [p['costs'][k]-q['costs'][k] for k in range(2)]
        if d[0] != d[1]:
            t = -d[1]/(d[0]-d[1])
            if 0 <= t <= 1: candidates.add((t,1-t))
    valid = []
    for lam in candidates:
        lower = min(_dot(lam,d['costs']) for d in det)
        gap = lower-_dot(lam,bs)
        if gap > 0: valid.append((gap,lam))
    if not valid: raise AssertionError('joint infeasibility separator missing')
    gap,lam = min(valid,key=lambda z:(-z[0],z[1]))
    pay = [[sum(lam[j]*cs[j][s][k] for j in range(2)) for k in range(len(row))] for s,row in enumerate(rows)]
    vals = [_values(g,rows,d['actions'],pay) for d in det]
    u = [min(v[s] for v in vals) for s in range(len(rows))]
    slacks = [pay[s][k]+g*sum(prob*u[t] for t,prob in trans)-u[s] for s,row in enumerate(rows) for k,(_,trans) in enumerate(row)]
    lower = (1-g)*_dot(mu,u); ceiling = _dot(lam,bs)
    assert min(slacks) >= 0 and lower-ceiling == gap
    return dict(weights=list(map(pack,lam)),values=list(map(pack,u)),action_slacks=list(map(pack,slacks)),lower_bound=pack(lower),budget_ceiling=pack(ceiling),gap=pack(gap))


def coupled(problem):
    g,rows,mu,cs,bs = compile_problem(problem)
    flat,a,b,c,rewards = _system(g,rows,mu,cs)
    det = _det(g,rows,mu,a,b,c,rewards,flat); vertices = _vertices(a,b,c,bs)
    fd = [d['reward'] for d in det if all(d['costs'][k] <= bs[k] for k in range(2))]
    vr = [dict(index=i,occupancy=list(map(pack,x)),costs=[pack(_dot(row,x)) for row in c],reward=pack(_dot(rewards,x)),active_budgets=[k for k in range(2) if _dot(c[k],x) == bs[k]]) for i,x in enumerate(vertices)]
    out = dict(format='bellmanlab-coupled-v1',problem=canonical(g,rows,mu,cs,bs),actions=[dict(state=s,action=k) for s,k in flat],deterministic=[dict(index=i,actions=d['actions'],occupancy=list(map(pack,d['x'])),costs=list(map(pack,d['costs'])),reward=pack(d['reward']),feasible=all(d['costs'][k] <= bs[k] for k in range(2))) for i,d in enumerate(det)],vertices=vr,feasible=bool(vertices),best_deterministic_reward=pack(max(fd)) if fd else None,optimal_vertices=[],selected=None,infeasibility=None)
    if not vertices:
        out['infeasibility'] = _separator(g,rows,mu,cs,bs,det)
        return out
    optimum = max(_dot(rewards,x) for x in vertices)
    ties = [i for i,x in enumerate(vertices) if _dot(rewards,x) == optimum]
    chosen = min(ties,key=lambda i:tuple(_dot(row,vertices[i]) for row in c)+(i,)); x = vertices[chosen]
    costs = [_dot(row,x) for row in c]
    mass = [sum(x[j] for j,(s,k) in enumerate(flat) if s == t) for t in range(len(rows))]
    pi = [[x[flat.index((s,k))]/mass[s] if mass[s] else F(k == 0) for k in range(len(row))] for s,row in enumerate(rows)]
    replay = [[F(s == t)-g*sum(pi[s][k]*dict(rows[s][k][1]).get(t,F(0)) for k in range(len(row))) for s,row in enumerate(rows)] for t in range(len(rows))]
    rd = _linear_solve(replay,b); rx = [rd[s]*pi[s][k] for s,k in flat]
    assert rx == list(x)
    balances = []
    for t in range(len(rows)):
        incoming = g*sum(x[j]*dict(rows[s][k][1]).get(t,F(0)) for j,(s,k) in enumerate(flat))
        balances.append(dict(state=t,initial=pack(b[t]),incoming=pack(incoming),outgoing=pack(mass[t]),residual=pack(mass[t]-b[t]-incoming)))
    out['optimal_vertices'] = ties
    out['selected'] = dict(vertex=chosen,occupancy=list(map(pack,x)),reward=pack(optimum),costs=list(map(pack,costs)),mixture=_mixture(x,det),state_mass=list(map(pack,mass)),policy=[list(map(pack,row)) for row in pi],zero_mass_states=[s for s in range(len(rows)) if not mass[s]],balances=balances,replay_occupancy=list(map(pack,rx)),discounted_reward=pack(optimum/(1-g)),discounted_costs=[pack(v/(1-g)) for v in costs],discounted_budgets=[pack(v/(1-g)) for v in bs],gain_over_deterministic=pack(optimum-max(fd)) if fd else None,dual=_dual(g,rows,mu,cs,bs,a,c,rewards,det,x,optimum))
    return out
