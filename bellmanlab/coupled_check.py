"""Independent policy-simplex/minor replay. No flow-vertex or generator helpers.

Shared: input compiler/canonical input, classical mathematics, output schema.
Dual/separator multipliers are proof witnesses: any valid exact witness is allowed.
"""
from fractions import Fraction as F
from itertools import combinations, permutations, product
from .coupled import compile_problem, canonical
from .core import pack


def determinant(a):
    n=len(a); total=F(0)
    for p in permutations(range(n)):
        term=F(-1 if sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))%2 else 1)
        for i in range(n): term*=a[i][p[i]]
        total+=term
    return total


def cramer(a,b):
    d=determinant(a)
    if not d: raise ValueError('singular minor')
    return [determinant([[b[i] if j==k else a[i][j] for j in range(len(b))] for i in range(len(b))])/d for k in range(len(b))]


def dot(a,b): return sum((x*y for x,y in zip(a,b)),F(0))


def _q(value):
    if type(value) is not list or len(value)!=2 or any(type(v) is not int for v in value) or value[1]<=0 or any(v.bit_length()>4096 for v in value): raise ValueError('invalid proof rational')
    q=F(*value)
    if pack(q)!=value: raise ValueError('noncanonical proof rational')
    return q


def _same(actual,expected):
    if type(actual) is not type(expected): raise ValueError('primitive/container type mismatch')
    if type(expected) is dict:
        if actual.keys()!=expected.keys(): raise ValueError('certificate keys differ')
        for k in expected: _same(actual[k],expected[k])
    elif type(expected) is list:
        if len(actual)!=len(expected): raise ValueError('certificate length differs')
        for a,b in zip(actual,expected): _same(a,b)
    elif actual!=expected: raise ValueError('certificate value differs')


def _replay(problem):
    g,rows,mu,cs,bs=compile_problem(problem); n=len(rows)
    flat=[(s,k) for s,row in enumerate(rows) for k in range(len(row))]; m=len(flat)
    trans=[[dict(rows[s][k][1]).get(t,F(0)) for s,k in flat] for t in range(n)]
    a=[[F(t==s)-g*trans[t][j] for j,(s,k) in enumerate(flat)] for t in range(n)]
    b=[(1-g)*v for v in mu]; c=[[table[s][k] for s,k in flat] for table in cs]; reward=[rows[s][k][0] for s,k in flat]
    det=[]
    for policy in product(*(range(len(row)) for row in rows)):
        cols=[flat.index((s,k)) for s,k in enumerate(policy)]
        mass=cramer([[row[j] for j in cols] for row in a],b)
        x=[F(0)]*m
        for j,v in zip(cols,mass): x[j]=v
        det.append((list(policy),x,[dot(row,x) for row in c],dot(reward,x)))
    # Independently intersect every one-, two- and three-policy simplex with
    # both budgets. A new vertex lies in a <=2-dimensional original face.
    candidates={}
    def keep(ids,weights):
        if min(weights)<=0 or sum(weights)!=1: return
        x=tuple(sum(w*det[i][1][j] for i,w in zip(ids,weights)) for j in range(m))
        costs=[dot(row,x) for row in c]
        if any(v>z for v,z in zip(costs,bs)): return
        active=[k for k in range(2) if costs[k]==bs[k]]; support=[j for j,v in enumerate(x) if v]
        equations=a+[c[k] for k in active]
        if len(support)>len(equations): return
        # Full rank restricted to positive columns is exactly the vertex test.
        if not any(determinant([[equations[i][j] for j in support] for i in rs]) for rs in combinations(range(len(equations)),len(support))): return
        candidates.setdefault(x,(ids,weights))
    for i in range(len(det)): keep((i,),(F(1),))
    for i,j in combinations(range(len(det)),2):
        for k in range(2):
            d=det[j][2][k]-det[i][2][k]
            if d:
                w=(bs[k]-det[i][2][k])/d
                keep((i,j),(1-w,w))
    for ids in combinations(range(len(det)),3):
        mat=[[F(1)]*3]+[[det[i][2][k] for i in ids] for k in range(2)]
        try: weights=cramer(mat,[F(1)]+bs)
        except ValueError: continue
        keep(ids,weights)
    vertices=sorted(candidates)
    # Canonical smallest-size lexicographic full-occupancy mixture, not the
    # incidental budget-intersection construction above.
    def witness(x):
        for size in (1,2,3):
            for ids in combinations(range(len(det)),size):
                equations=[[F(1)]*size]+[[det[i][1][j] for i in ids] for j in range(m)]
                targets=[F(1)]+list(x)
                for rs in combinations(range(m+1),size):
                    try: weights=cramer([equations[i] for i in rs],[targets[i] for i in rs])
                    except ValueError: continue
                    if min(weights)>0 and all(dot(row,weights)==v for row,v in zip(equations,targets)):
                        return [dict(policy=i,weight=pack(w)) for i,w in zip(ids,weights)]
        raise ValueError('no full occupancy mixture')
    def values(policy,pay):
        mat=[[F(s==t)-g*dict(rows[s][policy[s]][1]).get(t,F(0)) for t in range(n)] for s in range(n)]
        return cramer(mat,[pay[s][k] for s,k in enumerate(policy)])
    return g,rows,mu,cs,bs,flat,a,b,c,reward,det,vertices,witness,values,trans


def verify_coupled(report):
    if type(report) is not dict or 'problem' not in report: raise ValueError('report object required')
    g,rows,mu,cs,bs,flat,a,b,c,reward,det,vertices,witness,values,trans=_replay(report['problem']); n=len(rows);m=len(flat)
    fd=[d[3] for d in det if all(v<=z for v,z in zip(d[2],bs))]
    vr=[dict(index=i,occupancy=list(map(pack,x)),costs=[pack(dot(row,x)) for row in c],reward=pack(dot(reward,x)),active_budgets=[k for k in range(2) if dot(c[k],x)==bs[k]]) for i,x in enumerate(vertices)]
    expected=dict(format='bellmanlab-coupled-v1',problem=canonical(g,rows,mu,cs,bs),actions=[dict(state=s,action=k) for s,k in flat],deterministic=[dict(index=i,actions=d[0],occupancy=list(map(pack,d[1])),costs=list(map(pack,d[2])),reward=pack(d[3]),feasible=all(v<=z for v,z in zip(d[2],bs))) for i,d in enumerate(det)],vertices=vr,feasible=bool(vertices),best_deterministic_reward=pack(max(fd)) if fd else None,optimal_vertices=[],selected=None,infeasibility=None)
    if not vertices:
        cert=report.get('infeasibility')
        if type(cert) is not dict or type(cert.get('weights')) is not list or len(cert['weights'])!=2: raise ValueError('separator required')
        lam=list(map(_q,cert['weights']))
        if min(lam)<0 or sum(lam)!=1: raise ValueError('separator weights must be nonnegative and sum to one')
        pay=[[sum(lam[j]*cs[j][s][k] for j in range(2)) for k in range(len(row))] for s,row in enumerate(rows)]
        vals=[values(d[0],pay) for d in det];u=[min(v[s] for v in vals) for s in range(n)]
        slacks=[pay[s][k]+g*sum(prob*u[t] for t,prob in z)-u[s] for s,row in enumerate(rows) for k,(_,z) in enumerate(row)]
        lower=(1-g)*dot(mu,u);ceiling=dot(lam,bs)
        if min(slacks)<0 or lower<=ceiling: raise ValueError('not an infeasibility proof')
        expected['infeasibility']=dict(weights=list(map(pack,lam)),values=list(map(pack,u)),action_slacks=list(map(pack,slacks)),lower_bound=pack(lower),budget_ceiling=pack(ceiling),gap=pack(lower-ceiling))
    else:
        optimum=max(dot(reward,x) for x in vertices);ties=[i for i,x in enumerate(vertices) if dot(reward,x)==optimum]
        chosen=min(ties,key=lambda i:tuple(dot(row,vertices[i]) for row in c)+(i,));x=vertices[chosen];costs=[dot(row,x) for row in c]
        mass=[sum(x[j] for j,(s,k) in enumerate(flat) if s==t) for t in range(n)]
        pi=[[x[flat.index((s,k))]/mass[s] if mass[s] else F(k==0) for k in range(len(row))] for s,row in enumerate(rows)]
        mat=[[F(s==t)-g*sum(pi[s][k]*dict(rows[s][k][1]).get(t,F(0)) for k in range(len(row))) for s,row in enumerate(rows)] for t in range(n)]
        rd=cramer(mat,b);rx=[rd[s]*pi[s][k] for s,k in flat]
        if list(x)!=rx: raise ValueError('extracted policy does not replay')
        balances=[]
        for t in range(n):
            incoming=g*dot(trans[t],x)
            balances.append(dict(state=t,initial=pack(b[t]),incoming=pack(incoming),outgoing=pack(mass[t]),residual=pack(mass[t]-b[t]-incoming)))
        sel=report.get('selected');dual=sel.get('dual') if type(sel) is dict else None
        if type(dual) is not dict or type(dual.get('multipliers')) is not list or len(dual['multipliers'])!=2: raise ValueError('dual required')
        lam=list(map(_q,dual['multipliers']))
        if min(lam)<0: raise ValueError('negative multiplier')
        pay=[[r-sum(lam[j]*cs[j][s][k] for j in range(2)) for k,(r,_) in enumerate(row)] for s,row in enumerate(rows)]
        vals=[values(d[0],pay) for d in det];v=[max(z[s] for z in vals) for s in range(n)]
        slack=[v[s]-reward[j]+sum(lam[k]*c[k][j] for k in range(2))-g*dot([trans[t][j] for t in range(n)],v) for j,(s,k) in enumerate(flat)]
        bs_slack=[bs[k]-costs[k] for k in range(2)];upper=(1-g)*dot(mu,v)+dot(lam,bs)
        if min(slack)<0 or upper!=optimum: raise ValueError('dual does not certify optimum')
        proof=dict(multipliers=list(map(pack,lam)),values=list(map(pack,v)),action_slacks=list(map(pack,slack)),occupied_slack_products=[pack(x[j]*slack[j]) for j in range(m)],budget_slacks=list(map(pack,bs_slack)),budget_products=[pack(lam[k]*bs_slack[k]) for k in range(2)],upper_bound=pack(upper),gap=pack(upper-optimum))
        expected['optimal_vertices']=ties
        expected['selected']=dict(vertex=chosen,occupancy=list(map(pack,x)),reward=pack(optimum),costs=list(map(pack,costs)),mixture=witness(x),state_mass=list(map(pack,mass)),policy=[list(map(pack,row)) for row in pi],zero_mass_states=[s for s in range(n) if not mass[s]],balances=balances,replay_occupancy=list(map(pack,rx)),discounted_reward=pack(optimum/(1-g)),discounted_costs=[pack(z/(1-g)) for z in costs],discounted_budgets=[pack(z/(1-g)) for z in bs],gain_over_deterministic=pack(optimum-max(fd)) if fd else None,dual=proof)
    _same(report,expected)
    return True
