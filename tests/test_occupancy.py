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
from unittest.mock import patch
from bellmanlab.occupancy import occupancy,compile_problem
from bellmanlab.occupancy_check import verify_occupancy,flow_vertices,cramer
from bellmanlab.occupancy_demo import controls


def seeded(count=120):
    rng=random.Random(91843)
    for _ in range(count):
        n=rng.randint(1,4);states=[];costs=[]
        for s in range(n):
            acts=[];cs=[]
            for k in range(2):
                w=[rng.randint(0,3) for j in range(n)]
                if not sum(w):w[s]=1
                acts.append(dict(reward=rng.randint(-4,4),transitions=[dict(next=j,prob=[v,sum(w)]) for j,v in enumerate(w)]));cs.append(rng.randint(-3,3))
            states.append(dict(actions=acts));costs.append(cs)
        yield dict(version=1,model=dict(version=1,discount=rng.choice([0,[1,2],[2,3],[9,10]]),states=states),initial=[1]+[0]*(n-1),costs=costs,budget=rng.choice([-3,-1,0,[1,2],2,4]))


class OccupancyTests(unittest.TestCase):
    def test_controls(self):
        for name,p in controls().items():
            with self.subTest(name=name):self.assertTrue(verify_occupancy(occupancy(p)))
    def test_original_wait_exit_counterexample(self):
        r=occupancy(controls()['visitation_reweighting']);s=r['selected'];self.assertEqual(s['policy'],[[[2,3],[1,3]],[[1,1]]]);self.assertEqual(s['reward'],[1,2]);self.assertEqual(s['cost'],[1,4]);self.assertEqual(s['discounted_cost'],[1,2]);self.assertEqual(s['occupancy'],[[1,2],[1,4],[1,4]])
        # Direct half exit: flow (1-gamma)/(1-gamma/2)=2/3 at s0.
        d=cramer([[F(3,4),0],[F(-1,4),F(1,2)]],[F(1,2),0]);self.assertEqual(d,[F(2,3),F(1,3)]);self.assertGreater(d[1],F(1,4))
    def test_randomization_gap_and_tie(self):
        r=occupancy(controls()['randomization_gap']);self.assertEqual(r['best_deterministic_reward'],[0,1]);self.assertEqual(r['selected']['gain_over_deterministic'],[1,1]);self.assertEqual(r['selected']['dual']['multiplier'],[2,1]);self.assertEqual(r['selected']['dual']['action_slacks'],[[0,1],[0,1]])
    def test_infeasibility_and_zero_mass(self):
        r=occupancy(controls()['infeasible_budget']);self.assertFalse(r['feasible']);self.assertIsNone(r['selected']);self.assertEqual(r['min_cost_certificate']['budget_gap'],[1,2])
        r=occupancy(controls()['unreachable_states']);self.assertEqual(r['selected']['zero_mass_states'],[1]);self.assertEqual(r['selected']['policy'][1],[[1,1],[0,1]])
    def test_face_duplicates_slack(self):
        r=occupancy(controls()['flat_optimal_face']);self.assertEqual(r['optimal_vertices'],[0,1]);self.assertEqual(r['selected']['cost'],[0,1]);self.assertEqual(r['selected']['dual']['multiplier'],[0,1])
        r=occupancy(controls()['duplicate_points']);self.assertEqual(len(r['deterministic']),3);self.assertEqual(len(r['hull']),2)
        r=occupancy(controls()['slack_budget']);self.assertEqual(r['selected']['dual']['budget_slack'],[2,1]);self.assertEqual(r['selected']['dual']['multiplier'],[0,1])
    def test_seeded_120_complete_reports(self):
        for i,p in enumerate(seeded()):
            with self.subTest(i=i):self.assertTrue(verify_occupancy(occupancy(p)))
    def test_exhaustive_one_state_729(self):
        count=0
        for r0,r1,c0,c1,b,g in product([-1,0,1],repeat=6):
            p=controls()['randomization_gap'];p['model']['discount']={-1:0,0:[1,2],1:[2,3]}[g]
            for a,r in zip(p['model']['states'][0]['actions'],[r0,r1]):a['reward']=r
            p['costs']=[[c0,c1]];p['budget']=b
            self.assertTrue(verify_occupancy(occupancy(p)));count+=1
        self.assertEqual(count,729)
    def test_direct_flow_lp_vertex_oracle(self):
        for p in list(controls().values())+list(seeded(16)):
            r=occupancy(p);vertices=flow_vertices(p);self.assertEqual(bool(vertices),r['feasible'])
            g,rows,mu,cs,b=compile_problem(p);rewards=[a[0] for row in rows for a in row]
            if vertices:self.assertEqual(max(sum(a*b for a,b in zip(x,rewards)) for x in vertices),F(*r['selected']['reward']))
    def test_checker_helpers_independent(self):
        r=occupancy(controls()['four_state'])
        import bellmanlab.occupancy as m
        with patch.multiple(m,**{name:unittest.mock.Mock(side_effect=AssertionError(name)) for name in ['occupancy','_deterministic','_hull','_clip','_witness','_value','_assemble','_linear_solve','_cross']}):self.assertTrue(verify_occupancy(r))
    def test_mutations_every_noninput_primitive(self):
        r=occupancy(controls()['visitation_reweighting']);paths=[]
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
            with self.assertRaises(ValueError):verify_occupancy(q)
        self.assertGreater(len(paths),100)
    def test_strict_types_keys_order_and_canonical_fractions(self):
        r=occupancy(controls()['visitation_reweighting'])
        for path,value in [(['feasible'],1),(['selected','vertex'],True),(['selected','cost'],[2,8]),(['selected','occupancy'],list(reversed(r['selected']['occupancy']))),(['selected','dual','gap'],[0,2])]:
            q=copy.deepcopy(r);t=q
            for k in path[:-1]:t=t[k]
            t[path[-1]]=value
            with self.assertRaises(ValueError):verify_occupancy(q)
        for change in [lambda q:q.update(extra=0),lambda q:q['selected'].pop('balances'),lambda q:q['hull'].reverse()]:
            q=copy.deepcopy(r);change(q)
            with self.assertRaises(ValueError):verify_occupancy(q)
    def test_input_validation(self):
        for path,value in [(['version'],True),(['budget'],0.5),(['budget'],[1,0]),(['budget'],101),(['initial'],[1,1]),(['initial'],[False]),(['costs'],[[0]]),(['costs'],[[0,101]]),(['model','discount'],1)]:
            q=controls()['randomization_gap'];t=q
            for k in path[:-1]:t=t[k]
            t[path[-1]]=value
            with self.assertRaises(ValueError):occupancy(q)
    def test_cli_atomic(self):
        with tempfile.TemporaryDirectory()as td:
            p=Path(td)/'in.json';p.write_text(json.dumps(controls()['visitation_reweighting']));cmd=[sys.executable,'-m','bellmanlab.occupancy_cli']
            z=subprocess.run(cmd+['run',str(p)],capture_output=True,text=True);self.assertEqual(z.returncode,0,z.stderr);report=json.loads(z.stdout);p.write_text(z.stdout)
            z=subprocess.run(cmd+['verify',str(p)],capture_output=True,text=True);self.assertEqual(z.returncode,0,z.stderr)
            for body in ['{"version":1,"version":1}','{"budget":1e0}','{"budget":0.5}','NaN','x'*65537]:
                p.write_text(body);z=subprocess.run(cmd+['run',str(p)],capture_output=True,text=True);self.assertEqual(z.returncode,2);self.assertEqual(z.stdout,'')
            report['selected']['reward']=[999,1];p.write_text(json.dumps(report));z=subprocess.run(cmd+['verify',str(p)],capture_output=True,text=True);self.assertEqual(z.returncode,2);self.assertEqual(z.stdout,'')
    def test_operational_bounds(self):
        p=controls()['four_state'];self.assertTrue(verify_occupancy(occupancy(p)))
        p['model']['states'].append(copy.deepcopy(p['model']['states'][0]));p['costs'].append([0,0]);p['initial'].append(0)
        with self.assertRaises(ValueError):occupancy(p)
        p=controls()['four_state'];p['model']['states'][0]['actions'].append(copy.deepcopy(p['model']['states'][0]['actions'][0]));p['costs'][0].append(0)
        with self.assertRaises(ValueError):occupancy(p)
    def test_combined_denominator_guard(self):
        p=controls()['visitation_reweighting'];p['model']['discount']=[1,999983];p['initial']=[[1,2],[1,2]]
        with self.assertRaises(ValueError):occupancy(p)
        p=controls()['randomization_gap'];p['model']['discount']=[999999,1000000];self.assertTrue(verify_occupancy(occupancy(p)));self.assertEqual(occupancy(p)['selected']['discounted_reward'],[1000000,1])
    def test_canonicalization_zero_entries_and_permutations(self):
        p=controls()['visitation_reweighting'];p['model']['states'][0]['actions'][0]['transitions'].append(dict(next=1,prob=0));p['model']['states'][0]['actions'][0]['reward']=[0,2]
        r=occupancy(p);self.assertTrue(verify_occupancy(r));self.assertEqual(r['selected']['policy'][0],[[2,3],[1,3]])
        p['model']['states'][0]['actions'].reverse();p['costs'][0].reverse();r=occupancy(p);self.assertTrue(verify_occupancy(r));self.assertEqual(r['selected']['policy'][0],[[1,3],[2,3]])
    def test_exact_output_rationals_exceed_input_budget(self):
        # Output is exact and must not be revalidated as a bounded input scalar.
        p=controls()['four_state'];p['model']['discount']=[999999,1000000];r=occupancy(p);self.assertTrue(verify_occupancy(r))
        integers=[]
        def walk(v):
            if type(v)is dict:
                for x in v.values():walk(x)
            elif type(v)is list:
                for x in v:walk(x)
            elif type(v)is int:integers.append(abs(v))
        walk(r);self.assertGreater(max(integers),2**53)
    def test_isolation(self):
        p=controls()['four_state'];original=copy.deepcopy(p);r=occupancy(p);self.assertEqual(p,original);r['deterministic'][0]['actions'][0]=999;self.assertNotEqual(occupancy(p)['deterministic'][0]['actions'][0],999)

if __name__=='__main__':unittest.main()
