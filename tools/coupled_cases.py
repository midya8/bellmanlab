"""Deterministic external-check inputs, including denominator-bound stress."""
from itertools import product
import random
from bellmanlab.coupled_demo import controls,one


def seeded(count=80):
    rng=random.Random(20957)
    for _ in range(count):
        n=rng.randint(1,4);states=[];costs=[[],[]]
        for s in range(n):
            acts=[];cs=[[],[]]
            for k in range(2):
                w=[rng.randint(0,3) for _ in range(n)]
                if not sum(w):w[s]=1
                acts.append(dict(reward=rng.randint(-4,4),transitions=[dict(next=t,prob=[v,sum(w)]) for t,v in enumerate(w)]))
                for j in range(2):cs[j].append(rng.randint(-3,3))
            states.append(dict(actions=acts))
            for j in range(2):costs[j].append(cs[j])
        yield dict(version=1,model=dict(version=1,discount=rng.choice([0,[1,2],[2,3],[9,10]]),states=states),initial=[1]+[0]*(n-1),costs=costs,budgets=[rng.choice([-2,-1,0,[1,2],2,4]) for j in range(2)])


def stress():
    states=[]
    for s in range(4):
        acts=[]
        for k in range(2):
            w=[[3,1,1,1],[1,2,2,1],[1,3,1,1],[2,1,2,1],[1,1,3,1],[2,1,1,2],[1,1,1,3],[1,2,1,2]][2*s+k]
            acts.append(dict(reward=(s+1)*(1 if k else -1),transitions=[dict(next=t,prob=[v,6]) for t,v in enumerate(w)]))
        states.append(dict(actions=acts))
    return dict(version=1,model=dict(version=1,discount=[999983,999984],states=states),initial=[[1,4]]*4,costs=[[[s-1,2-s] for s in range(4)],[[1-s,s] for s in range(4)]],budgets=[5,5])


def cases():
    return list(controls().values())+list(seeded())+[one([0,2,3],[list(c[:3]),list(c[3:])],[0,0]) for c in product([-1,0,1],repeat=6)]+[stress()]
