"""Separate determinant, supporting-line and primal/dual certificate checker.

Only input compilation/canonicalization, fraction packing and theory are shared.
No policy, hull, clipping, witness, assembly or Gaussian solve helpers imported.
"""
from fractions import Fraction as F
from itertools import permutations, product, combinations
from .core import pack
from .occupancy import compile_problem, _canon


def determinant(a):
    n = len(a)
    if not n: return F(1)
    total = F(0)
    for p in permutations(range(n)):
        v = F(-1 if sum(p[i] > p[j] for i in range(n) for j in range(i+1,n)) % 2 else 1)
        for i,j in enumerate(p): v *= a[i][j]
        total += v
    return total


def cramer(a,b):
    den = determinant(a)
    if not den: raise ValueError('singular determinant')
    return [determinant([[b[i] if j == k else a[i][j] for j in range(len(b))] for i in range(len(b))])/den for k in range(len(b))]


def supporting_hull(points):
    points = sorted(set(points))
    if len(points) < 2: return points
    hull = []; current = points[0]
    while True:
        hull.append(current); nxt = next(p for p in points if p != current)
        for p in points:
            cross = (nxt[0]-current[0])*(p[1]-current[1])-(nxt[1]-current[1])*(p[0]-current[0])
            dist = lambda q: sum((q[k]-current[k])**2 for k in (0,1))
            if cross < 0 or cross == 0 and dist(p) > dist(nxt): nxt = p
        current = nxt
        if current == hull[0]: break
        if len(hull) > len(points): raise ValueError('supporting hull failed')
    return hull


def _expected(p):
    g,rows,mu,costs,b = compile_problem(p); n = len(rows)
    flat = [(s,k) for s,row in enumerate(rows) for k in range(len(row))]
    # Enumerate action choices recursively through product, reconstruct each
    # adjoint occupancy using determinant expansions, not Gauss-Jordan.
    det = []
    for policy in product(*(range(len(row)) for row in rows)):
        matrix = [[F(i == j)-g*dict(rows[j][policy[j]][1]).get(i,F(0)) for j in range(n)] for i in range(n)]
        mass = cramer(matrix,[(1-g)*m for m in mu])
        occ = [mass[s] if k == policy[s] else F(0) for s,k in flat]
        det.append(dict(actions=list(policy), x=occ, c=sum(occ[j]*costs[s][k] for j,(s,k) in enumerate(flat)), r=sum(occ[j]*rows[s][k][0] for j,(s,k) in enumerate(flat))))
    pts = [(z['c'],z['r']) for z in det]; hull = supporting_hull(pts)
    # All pair crossings, not the generator's polygon-edge clipping.
    candidate = [z for z in pts if z[0] <= b]
    for a,z in combinations(pts,2):
        if min(a[0],z[0]) < b < max(a[0],z[0]):
            t = (b-a[0])/(z[0]-a[0]); candidate.append((b,a[1]+t*(z[1]-a[1])))
    clipped = supporting_hull(candidate)
    def mixture(point):
        for i,z in enumerate(pts):
            if z == point: return [(i,F(1))]
        for i,j in combinations(range(len(pts)),2):
            for axis in (0,1):
                if pts[i][axis] != pts[j][axis]: break
            else: continue
            t = (point[axis]-pts[i][axis])/(pts[j][axis]-pts[i][axis])
            if 0 < t < 1 and tuple((1-t)*pts[i][k]+t*pts[j][k] for k in (0,1)) == point: return [(i,1-t),(j,t)]
        raise ValueError('no witness')
    def point_row(z): return dict(cost=pack(z[0]),reward=pack(z[1]),mixture=[dict(policy=i,weight=pack(w)) for i,w in mixture(z)])
    def policy_values(payoffs):
        ans = []
        for z in det:
            policy = z['actions']; a = [[F(i == j)-g*dict(rows[i][policy[i]][1]).get(j,F(0)) for j in range(n)] for i in range(n)]
            ans.append(cramer(a,[payoffs[s][k] for s,k in enumerate(policy)]))
        return ans
    cvs = policy_values(costs); cv = [min(z[s] for z in cvs) for s in range(n)]
    lower = (1-g)*sum(mu[s]*cv[s] for s in range(n))
    slack_c = [costs[s][k]+g*sum(pr*cv[t] for t,pr in rows[s][k][1])-cv[s] for s,k in flat]
    if any(v < 0 for v in slack_c) or lower != min(d['c'] for d in det): raise ValueError('minimum cost replay failed')
    min_cert = dict(values=list(map(pack,cv)),slacks=list(map(pack,slack_c)),lower_bound=pack(lower),budget_gap=pack(lower-b))
    bestdet = max((z['r'] for z in det if z['c'] <= b),default=None)
    out = dict(format='bellmanlab-occupancy-v1',problem=_canon(g,rows,mu,costs,b),actions=[dict(state=s,action=k) for s,k in flat],deterministic=[dict(index=i,actions=z['actions'],occupancy=list(map(pack,z['x'])),cost=pack(z['c']),reward=pack(z['r']),feasible=z['c'] <= b) for i,z in enumerate(det)],hull=[point_row(z) for z in hull],feasible_hull=[point_row(z) for z in clipped],min_cost_certificate=min_cert,feasible=bool(clipped),best_deterministic_reward=None if bestdet is None else pack(bestdet),optimal_vertices=[],selected=None)
    if not clipped:
        if not lower > b: raise ValueError('no infeasibility lower bound')
        return out
    opt = max(z[1] for z in clipped); tied = [i for i,z in enumerate(clipped) if z[1] == opt]; chosen = min(tied,key=lambda i:clipped[i][0]); z = clipped[chosen]
    mix = mixture(z); x = [sum(w*det[i]['x'][j] for i,w in mix) for j in range(len(flat))]
    mass = [sum(x[j] for j,(s,k) in enumerate(flat) if s == t) for t in range(n)]
    pi = [[x[flat.index((s,k))]/mass[s] if mass[s] else F(k == 0) for k in range(len(row))] for s,row in enumerate(rows)]
    # Slopes from all deterministic pairs, not hull adjacency.
    slopes = {F(0)}
    for a,q in combinations(pts,2):
        if a[0] != q[0] and (l := (a[1]-q[1])/(a[0]-q[0])) >= 0: slopes.add(l)
    lam = next(l for l in sorted(slopes) if all(q[1]+l*(b-q[0]) <= opt for q in pts) and max(q[1]+l*(b-q[0]) for q in pts) == opt)
    vals = policy_values([[r-lam*costs[s][k] for k,(r,t) in enumerate(row)] for s,row in enumerate(rows)])
    v = [max(q[s] for q in vals) for s in range(n)]
    slack = [v[s]-rows[s][k][0]+lam*costs[s][k]-g*sum(pr*v[t] for t,pr in rows[s][k][1]) for s,k in flat]
    upper = lam*b+(1-g)*sum(m*w for m,w in zip(mu,v))
    balances = []
    for t in range(n):
        inj = (1-g)*mu[t]; inc = g*sum(x[j]*dict(rows[s][k][1]).get(t,F(0)) for j,(s,k) in enumerate(flat))
        balances.append(dict(state=t,initial=pack(inj),incoming=pack(inc),outgoing=pack(mass[t]),residual=pack(mass[t]-inj-inc)))
    matrix = [[F(i == j)-g*sum(pi[j][k]*dict(row[1]).get(i,F(0)) for k,row in enumerate(rows[j])) for j in range(n)] for i in range(n)]
    rd = cramer(matrix,[(1-g)*m for m in mu]); rx = [rd[s]*pi[s][k] for s,k in flat]
    if any(t < 0 for t in slack) or any(a*t for a,t in zip(x,slack)) or lam*(b-z[0]) or upper != opt or rx != x: raise ValueError('primal-dual replay failed')
    dual = dict(multiplier=pack(lam),values=list(map(pack,v)),action_slacks=list(map(pack,slack)),occupied_slack_products=list(map(pack,[a*t for a,t in zip(x,slack)])),budget_slack=pack(b-z[0]),budget_product=pack(lam*(b-z[0])),upper_bound=pack(upper),gap=pack(upper-opt))
    out['optimal_vertices'] = tied
    out['selected'] = dict(vertex=chosen,**point_row(z),occupancy=list(map(pack,x)),state_mass=list(map(pack,mass)),policy=[list(map(pack,row)) for row in pi],zero_mass_states=[s for s in range(n) if not mass[s]],balances=balances,replay_occupancy=list(map(pack,rx)),discounted_reward=pack(z[1]/(1-g)),discounted_cost=pack(z[0]/(1-g)),discounted_budget=pack(b/(1-g)),gain_over_deterministic=None if bestdet is None else pack(opt-bestdet),dual=dual)
    return out


def _exact(a,b):
    if type(a) is not type(b): return False
    if type(a) is dict: return a.keys() == b.keys() and all(_exact(a[k],b[k]) for k in a)
    if type(a) is list: return len(a) == len(b) and all(_exact(x,y) for x,y in zip(a,b))
    return a == b


def verify_occupancy(report):
    try:
        if type(report) is not dict or not _exact(report,_expected(report['problem'])): raise ValueError('occupancy certificate mismatch')
    except (KeyError, TypeError, StopIteration, ZeroDivisionError) as e: raise ValueError('invalid occupancy certificate') from e
    return True


def flow_vertices(problem, budget=True):
    """Additional direct LP oracle, no policy enumeration or projected hull."""
    g,rows,mu,costs,b = compile_problem(problem); n = len(rows); flat = [(s,k) for s,row in enumerate(rows) for k in range(len(row))]
    a = [[F(t == s)-g*dict(rows[s][k][1]).get(t,F(0)) for s,k in flat] for t in range(n)]; rhs = [(1-g)*m for m in mu]
    cs = [costs[s][k] for s,k in flat]; vertices = set()
    for active in ([False,True] if budget else [False]):
        matrix = a+[cs] if active else a; target = rhs+[b] if active else rhs
        for count in range(1,min(len(matrix),len(flat))+1):
            for support in combinations(range(len(flat)),count):
                for selected_rows in combinations(range(len(matrix)),count):
                    square = [[matrix[i][j] for j in support] for i in selected_rows]
                    if not determinant(square): continue
                    vals = cramer(square,[target[i] for i in selected_rows])
                    if any(v < 0 for v in vals): break
                    x = [F(0)]*len(flat)
                    for j,v in zip(support,vals): x[j] = v
                    if all(sum(v*y for v,y in zip(row,x)) == r for row,r in zip(matrix,target)) and (not budget or sum(c*y for c,y in zip(cs,x)) <= b): vertices.add(tuple(x))
                    break
    return sorted(vertices)
