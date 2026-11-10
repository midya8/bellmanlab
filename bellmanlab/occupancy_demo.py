"""Original fully specified controls, not fitted environments or safety tests."""
from copy import deepcopy


def controls():
    def model(g, rows):
        return dict(version=1,discount=g,states=[dict(actions=[dict(reward=r,transitions=[dict(next=j,prob=p) for j,p in t]) for r,t in row]) for row in rows])
    def problem(m, costs, initial, b): return dict(version=1,model=m,costs=costs,initial=initial,budget=b)
    base = problem(model([1,2],[[(0,[(0,1)]),(2,[(0,1)])]]),[[0,1]],[1],[1,2])
    visit = problem(model([1,2],[[(0,[(0,1)]),(0,[(1,1)])],[(2,[(1,1)])]]),[[0,0],[1]],[1,0],[1,4])
    unreachable = problem(model([2,3],[[(1,[(0,1)]),(2,[(0,1)])],[(99,[(1,1)]),(-99,[(1,1)])]]),[[0,2],[100,-100]],[1,0],1)
    zero = deepcopy(visit); zero['model']['discount']=0
    impossible = deepcopy(base); impossible['costs']=[[1,2]]; impossible['budget']=[1,2]
    negative = problem(model([3,4],[[(-3,[(0,1)]),(-1,[(0,1)])]]),[[-2,0]],[1],-1)
    slack = deepcopy(base); slack['budget']=3
    ties = problem(model([1,2],[[(1,[(0,1)]),(1,[(0,1)])]]),[[0,1]],[1],1)
    duplicate = problem(model([1,2],[[(1,[(0,1)]),(1,[(0,1)]),(2,[(0,1)])]]),[[0,0,2]],[1],1)
    polygon = problem(model([2,3],[[(0,[(0,1)]),(2,[(1,1)])],[(1,[(0,[1,2]),(1,[1,2])]),(3,[(1,1)])]]),[[0,2],[1,3]],[[1,2],[1,2]],[3,2])
    four = problem(model([3,4],[[(0,[(0,[1,2]),(1,[1,2])]),(1,[(2,1)])],[(2,[(1,[1,2]),(3,[1,2])]),(-1,[(0,1)])],[(3,[(2,1)]),(1,[(3,1)])],[(0,[(3,1)]),(2,[(0,1)])]]),[[0,2],[3,-1],[2,0],[0,1]],[[1,2],0,[1,2],0],1)
    singleton = problem(model([99,100],[[(1,[(0,1)])]]),[[1]],[1],1)
    return dict(randomization_gap=base, visitation_reweighting=visit, unreachable_states=unreachable, zero_discount=zero, infeasible_budget=impossible, signed_costs=negative, slack_budget=slack, flat_optimal_face=ties, duplicate_points=duplicate, stochastic_polygon=polygon, four_state=four, singleton_boundary=singleton)
