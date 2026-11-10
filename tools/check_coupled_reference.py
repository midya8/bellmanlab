"""Optional actual HiGHS reference, full state/action variables, two budgets."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fractions import Fraction as F
import numpy as np
import scipy
from scipy.optimize import linprog
from bellmanlab.coupled import coupled,compile_problem
from coupled_cases import cases


def main():
    count=feasible=infeasible=0;worst=0.0
    for p in cases():
        g,rows,mu,cs,bs=compile_problem(p);flat=[(s,k) for s,row in enumerate(rows) for k in range(len(row))]
        A=np.array([[float(F(t==s)-g*dict(rows[s][k][1]).get(t,F(0))) for s,k in flat] for t in range(len(rows))]);b=np.array([float((1-g)*v) for v in mu]);C=np.array([[float(table[s][k]) for s,k in flat] for table in cs]);B=np.array(list(map(float,bs)));reward=np.array([float(rows[s][k][0]) for s,k in flat])
        ref=linprog(-reward,A_ub=C,b_ub=B,A_eq=A,b_eq=b,bounds=(0,None),method='highs');r=coupled(p);count+=1
        if r['feasible']:
            assert ref.success,(count,ref.message);feasible+=1;x=ref.x
            err=abs(-ref.fun-float(F(*r['selected']['reward'])));worst=max(worst,err)
            assert err<=1e-8,(count,err)
            assert max(abs(A@x-b))<=1e-8 and min(x)>=-1e-8 and max(C@x-B)<=1e-8
        else:
            assert ref.status==2,(count,ref.message);infeasible+=1
    print(json.dumps(dict(scipy=scipy.__version__,method='highs',solves=count,feasible=feasible,infeasible=infeasible,absolute_tolerance=1e-8,worst_reward_error=worst)))


if __name__=='__main__':main()
