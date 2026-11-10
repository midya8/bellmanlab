"""Original supplied teaching controls, no benchmark or learned dynamics."""
from copy import deepcopy


def one(rewards, costs, budgets, discount=(1,2)):
    return dict(version=1,model=dict(version=1,discount=list(discount),states=[dict(actions=[dict(reward=r,transitions=[dict(next=0,prob=1)]) for r in rewards])]),initial=[1],costs=[[c] for c in costs],budgets=budgets)


def controls():
    three = one([0,3,2],[[0,1,0],[0,0,1]],[[1,3],[1,4]])
    joint = one([1,2],[[1,0],[0,1]],[[1,3],[1,3]])
    wait = dict(version=1,model=dict(version=1,discount=[1,2],states=[dict(actions=[dict(reward=0,transitions=[dict(next=0,prob=1)]),dict(reward=0,transitions=[dict(next=1,prob=1)]),dict(reward=0,transitions=[dict(next=2,prob=1)])]),dict(actions=[dict(reward=4,transitions=[dict(next=1,prob=1)])]),dict(actions=[dict(reward=3,transitions=[dict(next=2,prob=1)])])]),initial=[1,0,0],costs=[[[0,0,0],[1],[0]],[[0,0,0],[0],[1]]],budgets=[[1,5],[1,6]])
    unreachable = deepcopy(wait); unreachable['model']['states'].append(dict(actions=[dict(reward=100,transitions=[dict(next=3,prob=1)]),dict(reward=-100,transitions=[dict(next=3,prob=1)])]));unreachable['initial'].append(0)
    for table in unreachable['costs']:table.append([0,0])
    dup = one([0,3,3,2],[[0,1,1,0],[0,0,0,1]],[[1,3],[1,4]])
    zero = deepcopy(three);zero['model']['discount']=0
    stochastic = deepcopy(wait)
    stochastic['model']['states'][0]['actions'][1]['transitions']=[dict(next=0,prob=[1,3]),dict(next=1,prob=[2,3])]
    stochastic['model']['states'][0]['actions'][2]['transitions']=[dict(next=1,prob=[1,4]),dict(next=2,prob=[3,4])]
    return dict(three_way=three,jointly_impossible=joint,visitation_reweighting=wait,unreachable_state=unreachable,duplicate_returns=dup,zero_discount=zero,stochastic_routes=stochastic,redundant_budgets=one([0,2],[[0,1],[0,2]],[[1,2],1]),one_slack=one([0,2],[[0,1],[3,0]],[[1,2],4]),both_slack=one([0,2],[[0,1],[3,0]],[5,5]),tied_face=one([1,1,1],[[0,1,0],[0,0,1]],[[1,2],[1,2]]),signed_budget=one([-2,3,1],[[-2,1,0],[1,-2,0]],[[-1,4],[-1,4]]))
