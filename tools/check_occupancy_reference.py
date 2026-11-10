"""Actual SciPy/HiGHS LP reference. Optional, not a runtime dependency."""
import itertools
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scipy
from scipy.optimize import linprog
from bellmanlab.occupancy import occupancy,compile_problem
from bellmanlab.occupancy_check import verify_occupancy
from bellmanlab.occupancy_demo import controls
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_occupancy import seeded


def cases():
    yield from controls().values()
    yield from seeded()
    for r0,r1,c0,c1,b,g in itertools.product([-1,0,1],repeat=6):
        p=controls()['randomization_gap'];p['model']['discount']={-1:0,0:[1,2],1:[2,3]}[g]
        for a,r in zip(p['model']['states'][0]['actions'],[r0,r1]):a['reward']=r
        p['costs']=[[c0,c1]];p['budget']=b;yield p


def main():
    count=feasible=infeasible=0;worst=0.0
    for p in cases():
        report=occupancy(p);assert verify_occupancy(report)
        g,rows,mu,cs,b=compile_problem(p);flat=[(s,k) for s,row in enumerate(rows) for k in range(len(row))]
        eq=[[float((t==s)-g*dict(rows[s][k][1]).get(t,0)) for s,k in flat] for t in range(len(rows))]
        rhs=[float((1-g)*m) for m in mu];rewards=[float(rows[s][k][0]) for s,k in flat];costs=[float(cs[s][k]) for s,k in flat]
        result=linprog([-r for r in rewards],A_ub=[costs],b_ub=[float(b)],A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs')
        count+=1
        if not report['feasible']:
            assert result.status==2,(count,result.message);infeasible+=1
        else:
            assert result.success,(count,result.message)
            target=report['selected']['reward'];target=target[0]/target[1];error=abs(-result.fun-target);worst=max(worst,error)
            assert error<=1e-8,(count,error)
            assert min(result.x)>=-1e-8
            assert max(abs(sum(v*x for v,x in zip(row,result.x))-t) for row,t in zip(eq,rhs))<=1e-8
            assert sum(c*x for c,x in zip(costs,result.x))<=float(b)+1e-8
            feasible+=1
        minimum=linprog(costs,A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs');assert minimum.success
        target=report['min_cost_certificate']['lower_bound'];assert abs(minimum.fun-target[0]/target[1])<=1e-8
    print(json.dumps(dict(reference='scipy.optimize.linprog(method=highs)',scipy=scipy.__version__,cases=count,feasible=feasible,infeasible=infeasible,actual_lp_solves=count*2,absolute_tolerance=1e-8,worst_objective_error=worst),sort_keys=True))

if __name__=='__main__':main()
