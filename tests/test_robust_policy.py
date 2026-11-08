import copy,json,subprocess,sys,unittest
from fractions import Fraction as F
from itertools import product
from bellmanlab import robust_evaluate_policy,robust_policy_iteration
from bellmanlab.robust_demo import one_state,robust_demo
from test_robust import model,cramers,pair


class RobustPolicyTests(unittest.TestCase):
    def test_243_models_independent_exact_minmax_witnesses(self):
        patterns=[[[F(1),F(0)],[F(0),F(1)]],[[F(1,2),F(1,3)],[F(2,3),F(1,2)]],[[F(0),F(1)],[F(1),F(0)]]]
        policies=list(product(range(2),repeat=2));count=0
        for k,probs in enumerate(patterns):
            for rewards in product([-1,0,1],repeat=4):
                problem={'version':1,'scenarios':[model(rewards,probs),model([-r for r in rewards],patterns[(k+1)%3])]}
                truths=[]
                for p in policies:
                    candidates=[cramers(problem,p,n) for n in policies];count+=len(candidates)
                    truth=[min(v[i] for v in candidates) for i in range(2)];truths.append(truth)
                    run=robust_evaluate_policy(problem,p);self.assertTrue(run['exact']);self.assertEqual(run['values'],list(map(pair,truth)))
                    self.check_nature(problem,run,truth)
                optimum=[max(v[i] for v in truths) for i in range(2)]
                run=robust_policy_iteration(problem);self.assertTrue(run['optimal']);self.assertTrue(run['policy_value_exact']);self.assertEqual(run['values'],list(map(pair,optimum)))
                self.assertEqual(list(map(pair,cramers(problem,run['policy'],run['nature_policy']))),run['values'])
                previous=None
                for h in run['history']:
                    truth=truths[policies.index(tuple(h['policy']))];self.check_nature(problem,h['evaluation'],truth)
                    if previous is not None:self.assertTrue(all(a>=b for a,b in zip(truth,previous)))
                    previous=truth
                    ties=h['optimality_audit']['greedy_actions'];self.assertEqual(h['improved_policy'],[old if old in t else t[0] for old,t in zip(h['policy'],ties)])
                self.assertEqual(run['nature_evaluations_completed'],sum(len(h['evaluation']['history']) for h in run['history']))
        self.assertEqual(count,3888)

    def check_nature(self,problem,run,truth):
        previous=None;p=run['policy']
        for h in run['history']:
            v=cramers(problem,p,h['nature_policy']);self.assertEqual(h['values'],list(map(pair,v)));self.assertEqual(h['linear_policy_equation_residual'],[0,1]);self.assertTrue(all(a>=b for a,b in zip(v,truth)))
            if previous is not None:self.assertTrue(all(a<=b for a,b in zip(v,previous)))
            previous=v
            minimizing=h['minimizing_scenarios_for_policy'];q=h['scenario_q_values'];self.assertEqual(minimizing,[[j for j,x in enumerate(q[i][a]) if F(*x)==min(F(*z) for z in q[i][a])] for i,a in enumerate(p)])
            self.assertEqual(h['improved_nature_policy'],[old if old in t else t[0] for old,t in zip(h['nature_policy'],minimizing)])
            self.assertLessEqual(max(abs(a-b) for a,b in zip(v,truth)),F(*h['value_error_upper_bound']))
        self.assertTrue(run['history'][-1]['stable']);self.assertEqual(run['result']['residual_sup'],[0,1])

    def test_inner_cap_returns_evaluated_upper_bound_not_proposal(self):
        p=robust_demo()['decision_case']['problem'];r=robust_evaluate_policy(p,[0],iterations=1)
        self.assertFalse(r['exact']);self.assertEqual(r['nature_policy'],[0]);self.assertEqual(r['values'],[[4,5]]);self.assertEqual(r['history'][-1]['improved_nature_policy'],[1]);self.assertEqual(r['result']['value_error_upper_bound'],[8,5]);self.assertIn('upper bound',r['value_status'])

    def test_outer_inner_cap_withholds_controller_improvement(self):
        p=robust_demo()['decision_case']['problem'];r=robust_policy_iteration(p,nature_iterations=1)
        self.assertFalse(r['optimal']);self.assertFalse(r['policy_value_exact']);self.assertEqual(r['iterations_completed'],1);self.assertEqual(r['policy'],[0]);self.assertEqual(r['nature_policy'],[0]);self.assertEqual(r['values'],[[4,5]]);self.assertIsNone(r['history'][0]['improved_policy']);self.assertIn('withheld',r['termination'])

    def test_controller_cap_returns_last_evaluated_policy(self):
        p=robust_demo()['decision_case']['problem'];r=robust_policy_iteration(p,iterations=1)
        self.assertFalse(r['optimal']);self.assertTrue(r['policy_value_exact']);self.assertEqual(r['policy'],[0]);self.assertEqual(r['nature_policy'],[1]);self.assertEqual(r['values'],[[-4,5]]);self.assertEqual(r['history'][0]['improved_policy'],[1]);self.assertIn('controller',r['termination'])

    def test_complete_fixture_and_joint_stability(self):
        p=robust_demo()['decision_case']['problem'];r=robust_policy_iteration(p);self.assertTrue(r['optimal']);self.assertEqual(r['policy'],[1]);self.assertEqual(r['values'],[[2,5]]);self.assertEqual(r['iterations_completed'],2);self.assertEqual(r['nature_evaluations_completed'],3)
        for h in r['history']:
            self.assertTrue(h['evaluation']['exact']);self.assertEqual(h['evaluation']['result']['residual_sup'],[0,1])
        self.assertEqual(r['history'][-1]['optimality_audit']['residual_sup'],[0,1])

    def test_retain_controller_and_nature_on_all_ties(self):
        m=one_state([1,1]);p={'version':1,'scenarios':[m,copy.deepcopy(m)]}
        r=robust_evaluate_policy(p,[1],initial_nature=[1]);self.assertEqual(r['iterations_completed'],1);self.assertEqual(r['nature_policy'],[1]);self.assertEqual(r['history'][0]['minimizing_scenarios_for_policy'],[[0,1]])
        r=robust_policy_iteration(p,initial=[1]);self.assertEqual(r['policy'],[1]);self.assertEqual(r['iterations_completed'],1);self.assertTrue(r['optimal'])

    def test_zero_discount_negative_and_tiny_advantage(self):
        m=one_state([-1,[-999999,1000000]]);m['discount']=0;p={'version':1,'scenarios':[m]};r=robust_policy_iteration(p);self.assertTrue(r['optimal']);self.assertEqual(r['policy'],[1]);self.assertEqual(r['values'],[[-999999,1000000]])

    def test_invalid_caps_and_policies(self):
        p={'version':1,'scenarios':[one_state([1])]}
        for arg in [True,0,65,1.5]:
            with self.assertRaises(ValueError):robust_evaluate_policy(p,[0],iterations=arg)
            for k in ['iterations','nature_iterations']:
                with self.assertRaises(ValueError):robust_policy_iteration(p,**{k:arg})
        for policy in [[],[True],[1],None]:
            with self.assertRaises(ValueError):robust_evaluate_policy(p,policy)
        for nature in [[],[True],[1]]:
            with self.assertRaises(ValueError):robust_evaluate_policy(p,[0],initial_nature=nature)

    def test_input_output_history_isolation(self):
        p=robust_demo()['decision_case']['problem'];before=copy.deepcopy(p);r=robust_policy_iteration(p);r['history'][0]['policy'][0]=99;r['history'][0]['evaluation']['history'][0]['values'][0][0]=99
        self.assertEqual(p,before);self.assertEqual(r['policy'],[1]);self.assertEqual(r['values'],[[2,5]]);self.assertNotEqual(r['history'][0]['evaluation']['history'][1]['values'][0][0],99)

    def test_cli_exact_evaluate_and_policy_caps(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'problem.json';path.write_text(json.dumps(robust_demo()['decision_case']['problem']))
            for args,flag,value in [(['robust-evaluate',str(path),'--policy','[0]'],'exact',True),(['robust-policy',str(path)],'optimal',True),(['robust-policy',str(path),'--nature-iterations','1'],'policy_value_exact',False),(['robust-evaluate',str(path),'--policy','[0]','--iterations','1'],'exact',False)]:
                r=subprocess.run([sys.executable,'-m','bellmanlab',*args],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)[flag],value)

    def test_large_chain_exact_and_all_scenario_witnesses(self):
        m={'version':1,'discount':[9,10],'states':[{'actions':[{'reward':int(i==15),'transitions':[{'next':min(i+1,15),'prob':1}]}]} for i in range(16)]};p={'version':1,'scenarios':[m,copy.deepcopy(m),copy.deepcopy(m)]};r=robust_policy_iteration(p)
        self.assertTrue(r['optimal']);self.assertEqual(r['values'],[pair(F(10)*F(9,10)**(15-i)) for i in range(16)]);self.assertEqual(r['nature_policy'],[0]*16);self.assertEqual(r['history'][0]['evaluation']['result']['minimizing_scenarios_for_policy'],[[0,1,2]]*16)

    def test_policy_demo_and_original_demos_unchanged(self):
        from bellmanlab.robust_policy_demo import robust_policy_demo
        r=subprocess.run([sys.executable,'-m','bellmanlab','robust-policy-demo'],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),robust_policy_demo())
