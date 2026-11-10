import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from unittest.mock import patch,Mock
from bellmanlab.coupled import coupled
from bellmanlab.coupled_check import verify_coupled,cramer
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


class CoupledTests(unittest.TestCase):
    def test_twelve_controls(self):
        for name,p in controls().items():
            with self.subTest(name=name): self.assertTrue(verify_coupled(coupled(p)))
    def test_three_way_is_necessary_and_both_prices(self):
        r=coupled(controls()['three_way']);s=r['selected']
        self.assertEqual(s['reward'],[3,2]);self.assertEqual(s['mixture'],[dict(policy=0,weight=[5,12]),dict(policy=1,weight=[1,3]),dict(policy=2,weight=[1,4])]);self.assertEqual(s['dual']['multipliers'],[[3,1],[2,1]])
        self.assertEqual(r['best_deterministic_reward'],[0,1]);self.assertEqual(s['gain_over_deterministic'],[3,2]);self.assertEqual(s['dual']['gap'],[0,1])
        self.assertEqual(s['discounted_costs'],[[2,3],[1,2]])
    def test_individually_feasible_jointly_impossible(self):
        r=coupled(controls()['jointly_impossible']);self.assertFalse(r['feasible']);self.assertIsNone(r['selected']);self.assertEqual(r['vertices'],[])
        self.assertEqual(r['infeasibility']['weights'],[[1,2],[1,2]]);self.assertEqual(r['infeasibility']['gap'],[1,6])
        self.assertTrue(any(F(*d['costs'][0])<=F(1,3) for d in r['deterministic']));self.assertTrue(any(F(*d['costs'][1])<=F(1,3) for d in r['deterministic']))
    def test_visitation_not_local_mixture(self):
        s=coupled(controls()['visitation_reweighting'])['selected']
        self.assertEqual(s['policy'][0],[[8,19],[6,19],[5,19]])
        self.assertEqual(s['state_mass'],[[19,30],[1,5],[1,6]])
        self.assertEqual(s['occupancy'],s['replay_occupancy']);self.assertEqual(s['reward'],[13,10])
        # Averaging the deterministic local rows with the episode-level weights
        # yields a different occupancy, and fails both normalized budgets.
        pi=[F(4,15),F(2,5),F(1,3)]
        d=cramer([[1-F(1,2)*pi[0],0,0],[-F(1,2)*pi[1],F(1,2),0],[-F(1,2)*pi[2],0,F(1,2)]],[F(1,2),F(0),F(0)])
        self.assertGreater(d[1],F(1,5));self.assertGreater(d[2],F(1,6))
    def test_full_occupancies_not_projected_points(self):
        r=coupled(controls()['duplicate_returns']);self.assertEqual(len(r['vertices']),6)
        projections={(tuple(tuple(v) for v in z['costs']),tuple(z['reward'])) for z in r['vertices']}
        self.assertEqual(len(projections),4);self.assertEqual(len(r['optimal_vertices']),2)
    def test_zero_mass_and_slack(self):
        r=coupled(controls()['unreachable_state']);self.assertEqual(r['selected']['zero_mass_states'],[3]);self.assertEqual(r['selected']['policy'][3],[[1,1],[0,1]])
        r=coupled(controls()['one_slack']);self.assertEqual(r['selected']['dual']['multipliers'][1],[0,1])
        r=coupled(controls()['both_slack']);self.assertEqual(r['selected']['dual']['multipliers'],[[0,1],[0,1]])
    def test_seeded_80_complete_vertices(self):
        for i,p in enumerate(seeded()):
            with self.subTest(i=i):self.assertTrue(verify_coupled(coupled(p)))
    def test_exhaustive_729_one_state_tables(self):
        for costs in product([-1,0,1],repeat=6):
            p=one([0,2,3],[list(costs[:3]),list(costs[3:])],[0,0])
            self.assertTrue(verify_coupled(coupled(p)))
    def test_bridge_to_existing_single_budget(self):
        from bellmanlab.occupancy import occupancy
        for p in list(controls().values())+list(seeded(12)):
            onep=dict(version=1,model=p['model'],initial=p['initial'],costs=p['costs'][0],budget=p['budgets'][0])
            p=copy.deepcopy(p);p['costs'][1]=[[0]*len(s['actions']) for s in p['model']['states']];p['budgets'][1]=0
            a=coupled(p);b=occupancy(onep);self.assertEqual(a['feasible'],b['feasible'])
            if a['feasible']:self.assertEqual(a['selected']['reward'],b['selected']['reward'])
    def test_checker_no_generator_helpers(self):
        r=coupled(controls()['stochastic_routes']);import bellmanlab.coupled as module
        names=['coupled','_dot','_system','_vertices','_det','_values','_mixture','_dual','_separator','_linear_solve']
        with patch.multiple(module,**{name:Mock(side_effect=AssertionError(name)) for name in names}): self.assertTrue(verify_coupled(r))
    def test_each_noninput_leaf_mutation(self):
        count=0
        for name in ['three_way','jointly_impossible','unreachable_state','duplicate_returns']:
            r=coupled(controls()[name]);paths=[]
            def walk(v,path):
                if type(v) is dict:
                    for k,x in v.items():
                        if path or k!='problem':walk(x,path+[k])
                elif type(v) is list:
                    for i,x in enumerate(v):walk(x,path+[i])
                else:paths.append(path)
            walk(r,[])
            for path in paths:
                q=copy.deepcopy(r);t=q
                for k in path[:-1]:t=t[k]
                old=t[path[-1]];t[path[-1]]=not old if type(old) is bool else old+1 if type(old) is int else 'forged'
                with self.assertRaises(ValueError,msg=str(path)):verify_coupled(q)
            count+=len(paths)
        self.assertGreater(count,900)
    def test_invalid_schema(self):
        for field,value in [('version',True),('budgets',[0]),('budgets',[0.5,0]),('budgets',[[1,0],0]),('initial',[False]),('costs',[[[0,1,0]]])]:
            p=controls()['three_way'];p[field]=value
            with self.assertRaises(ValueError):coupled(p)
        p=controls()['three_way'];p['extra']=0
        with self.assertRaises(ValueError):coupled(p)
    def test_cli_atomic_and_forgery(self):
        cmd=[sys.executable,'-m','bellmanlab.coupled_cli']
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.json';p.write_text(json.dumps(controls()['three_way']))
            run=subprocess.run(cmd+['run',str(p)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
            r=json.loads(run.stdout);p.write_text(run.stdout)
            run=subprocess.run(cmd+['verify',str(p)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
            r['selected']['dual']['gap']=[1,1];p.write_text(json.dumps(r));run=subprocess.run(cmd+['verify',str(p)],capture_output=True,text=True);self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'')
            for text in ['{"version":1,"version":1}','{"x":1e0}','{"x":0.5}','NaN','x'*65537]:
                p.write_text(text);run=subprocess.run(cmd+['run',str(p)],capture_output=True,text=True);self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'')
    def test_alternative_valid_witness_not_claimed_canonical(self):
        # Duplicate constraints admit different nonnegative multiplier splits.
        r=coupled(controls()['redundant_budgets']);s=r['selected'];lam=s['dual']['multipliers']
        self.assertEqual(sum(F(*q)*k for q,k in zip(lam,[1,2])),2)
        r['selected']['dual']['multipliers']=[[2,1],[0,1]] if lam!=[[2,1],[0,1]] else [[0,1],[1,1]]
        self.assertTrue(verify_coupled(r))
    def test_operational_bounds_and_large_discount(self):
        p=controls()['unreachable_state'];self.assertTrue(verify_coupled(coupled(p)))
        p=controls()['three_way'];p['model']['discount']=[999983,999984];r=coupled(p);self.assertTrue(verify_coupled(r));self.assertEqual(r['selected']['discounted_reward'],[1499976,1])
