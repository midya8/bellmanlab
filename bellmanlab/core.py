"""Exact rational discounted finite-MDP planning and residual certificates."""
from fractions import Fraction as F
from math import lcm
MAX_DENOMINATOR_LCM=1000000


def _keys(d,expected):
    if type(d) is not dict or set(d)!=set(expected):raise ValueError('object keys must be '+','.join(expected))
def rational(x):
    if type(x) is int:x=[x,1]
    if type(x) is not list or len(x)!=2 or any(type(v) is not int for v in x) or abs(x[0])>1000000 or not 1<=x[1]<=1000000:raise ValueError('rational must be integer or [numerator,positive denominator], bounded at 1000000')
    return F(*x)
def pack(x):return [x.numerator,x.denominator]
def _pack_values(v):return list(map(pack,v))
def _compile(p):
    _keys(p,['version','discount','states'])
    if type(p['version']) is not int or p['version']!=1:raise ValueError('version must be 1')
    g=rational(p['discount'])
    if not 0<=g<1:raise ValueError('discount must be in [0,1)')
    states=p['states']
    if type(states) is not list or not 1<=len(states)<=16:raise ValueError('1..16 states required')
    rows=[];den=g.denominator;entries=0;actions=0
    for state in states:
        _keys(state,['actions']);a=state['actions']
        if type(a) is not list or not 1<=len(a)<=8:raise ValueError('1..8 actions per state required')
        row=[];actions+=len(a)
        for action in a:
            _keys(action,['reward','transitions']);r=rational(action['reward']);den=lcm(den,r.denominator)
            if abs(r)>100:raise ValueError('absolute reward must be <=100')
            t=action['transitions']
            if type(t) is not list or not 1<=len(t)<=len(states):raise ValueError('1..n transitions required per action')
            seen=set();trans=[]
            for e in t:
                _keys(e,['next','prob']);j=e['next'];q=rational(e['prob'])
                if type(j) is not int or not 0<=j<len(states) or j in seen or not 0<=q<=1:raise ValueError('invalid/duplicate successor or probability')
                seen.add(j);trans.append((j,q));den=lcm(den,q.denominator);entries+=1
            if sum(q for j,q in trans)!=1:raise ValueError('each transition row must sum exactly to one')
            row.append((r,tuple(sorted(trans))))
        rows.append(tuple(row))
    if actions>64 or entries>512:raise ValueError('model limited to 64 actions and 512 transition entries')
    if den>MAX_DENOMINATOR_LCM:raise ValueError('model rational denominator LCM exceeds 1000000')
    return g,tuple(rows),den,entries

def _canonical(g,rows):
    return {'version':1,'discount':pack(g),'states':[{'actions':[{'reward':pack(r),'transitions':[{'next':j,'prob':pack(q)} for j,q in t]} for r,t in row]} for row in rows]}
def _policy(policy,rows):
    if type(policy) not in (list,tuple) or len(policy)!=len(rows) or any(type(a) is not int or not 0<=a<len(row) for a,row in zip(policy,rows)):raise ValueError('policy must choose one valid integer action per state')
    return list(policy)
def _values(values,rows,den):
    if values is None:return [F(0)]*len(rows)
    if type(values) not in (list,tuple) or len(values)!=len(rows):raise ValueError('one initial value per state required')
    v=list(map(rational,values))
    for x in v:den=lcm(den,x.denominator)
    if den>MAX_DENOMINATOR_LCM or any(abs(x)>1000000 for x in v):raise ValueError('initial values exceed denominator or magnitude budget')
    return v

def _q(g,rows,v):return [[r+g*sum(p*v[j] for j,p in t) for r,t in row] for row in rows]
def _certificate(g,rows,v):
    q=_q(g,rows,v);tv=[max(row) for row in q];residual=max(abs(a-b) for a,b in zip(tv,v));greedy=[[i for i,a in enumerate(row) if a==max(row)] for row in q]
    return {'values':_pack_values(v),'q_values':[_pack_values(row) for row in q],'bellman_values':_pack_values(tv),
            'residual_sup':pack(residual),'value_error_upper_bound':pack(residual/(1-g)),
            'greedy_actions':greedy,'greedy_policy':[row[0] for row in greedy],
            'certificate':'exact Bellman optimality residual; ||v-v*||_infinity <= residual/(1-discount); this certifies value error, not the greedy policy loss or a unique optimal policy'}

def audit(model,values):
    g,rows,den,entries=_compile(model);v=_values(values,rows,den)
    return {'format':'bellmanlab-audit-v1','model':_canonical(g,rows),**_certificate(g,rows,v)}

def _linear_solve(a,b):
    # Exact Gauss-Jordan with a nonzero pivot, no floating tolerance.
    n=len(b);m=[list(row)+[v] for row,v in zip(a,b)]
    for i in range(n):
        k=next((k for k in range(i,n) if m[k][i]),None)
        if k is None:raise ValueError('singular policy equations')
        m[i],m[k]=m[k],m[i];pivot=m[i][i];m[i]=[x/pivot for x in m[i]]
        for k in range(n):
            if k!=i:
                scale=m[k][i];m[k]=[x-scale*y for x,y in zip(m[k],m[i])]
    return [row[-1] for row in m]
def _evaluate(g,rows,policy):
    n=len(rows);a=[[F(i==j) for j in range(n)] for i in range(n)];b=[]
    for i,k in enumerate(policy):
        r,t=rows[i][k];b.append(r)
        for j,p in t:a[i][j]-=g*p
    v=_linear_solve(a,b)
    q=_q(g,rows,v)
    assert all(v[i]==q[i][k] for i,k in enumerate(policy))
    return v

def evaluate_policy(model,policy):
    g,rows,den,entries=_compile(model);policy=_policy(policy,rows);v=_evaluate(g,rows,policy)
    result=_certificate(g,rows,v)
    return {'format':'bellmanlab-policy-v1','model':_canonical(g,rows),'policy':policy,'policy_equation_residual':pack(F(0)),'is_optimal':result['residual_sup']==[0,1],**result}

def value_iteration(model,*,initial=None,sweeps=100,tolerance=None):
    g,rows,den,entries=_compile(model)
    if type(sweeps) is not int or not 0<=sweeps<=256:raise ValueError('sweeps must be integer in 0..256')
    tol=rational([1,1000] if tolerance is None else tolerance)
    if tol<0:raise ValueError('tolerance must be nonnegative')
    v=_values(initial,rows,den);history=[]
    for step in range(sweeps+1):
        cert=_certificate(g,rows,v);bound=F(*cert['value_error_upper_bound']);history.append({'sweep':step,**cert})
        if bound<=tol or step==sweeps:break
        v=[F(*x) for x in cert['bellman_values']]
    return {'format':'bellmanlab-value-iteration-v1','model':_canonical(g,rows),'tolerance':pack(tol),'sweeps_completed':len(history)-1,
            'converged':F(*history[-1]['value_error_upper_bound'])<=tol,'history':history,'result':history[-1],
            'planned_backup_transition_reads':(sweeps+1)*entries,'executed_backup_transition_reads':len(history)*entries,
            'scope':'synchronous exact rational Bellman backups, including a final residual audit; at most 256 sweeps, plus initial audit; counters exclude rational bit costs, validation and serialization; tolerance gates certified value error, not raw residual, policy loss or runtime'}

def policy_iteration(model,*,initial=None,iterations=64):
    g,rows,den,entries=_compile(model)
    if type(iterations) is not int or not 1<=iterations<=64:raise ValueError('iterations must be in 1..64')
    policy=_policy([0]*len(rows) if initial is None else initial,rows);history=[];optimal=False;previous=None
    for step in range(iterations):
        v=_evaluate(g,rows,policy);cert=_certificate(g,rows,v)
        if previous is not None:assert all(a>=b for a,b in zip(v,previous))
        new=[old if old in greedy else greedy[0] for old,greedy in zip(policy,cert['greedy_actions'])];stable=new==policy
        history.append({'iteration':step,'policy':policy[:],**cert,'improved_policy':new,'stable':stable})
        if stable:optimal=True;break
        if step+1<iterations:policy=new;previous=v
    return {'format':'bellmanlab-policy-iteration-v1','model':_canonical(g,rows),'history':history,'iterations_completed':len(history),'optimal':optimal,
            'policy':history[-1]['policy'],'values':history[-1]['values'],
            'scope':'exact evaluation of deterministic stationary policies; retain old action on exact ties to avoid gratuitous policy switches; stable greedy policy satisfies Bellman optimality; hitting the cap returns the last evaluated policy with optimal=false, never an unevaluated improved policy; no learned environment, online RL or unique-policy claim'}
