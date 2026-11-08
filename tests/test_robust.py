import copy,itertools,json,subprocess,sys,unittest
from fractions import Fraction as F
from math import lcm
from bellmanlab import robust_audit,robust_value_iteration,audit,value_iteration
from bellmanlab.robust_demo import robust_demo,one_state


def pair(x):return [x.numerator,x.denominator]
def model(rewards,probs):
    return {'version':1,'discount':[1,2],'states':[{'actions':[
        {'reward':rewards[2*i+a], 'transitions':[{'next':0,'prob':pair(probs[i][a])},{'next':1,'prob':pair(1-probs[i][a])}]}
        for a in range(2)]} for i in range(2)]}

def cramers(problem,policy,nature):
    # Independent closed 2x2 equations, no production compiler or solver.
    rows=[problem['scenarios'][k]['states'][i]['actions'][policy[i]] for i,k in enumerate(nature)]
    rewards=[F(r['reward']) for r in rows]
    p=[[F(*t['prob']) for t in r['transitions']] for r in rows]
    a,b,c,d=1-p[0][0]/2,-p[0][1]/2,-p[1][0]/2,1-p[1][1]/2
    determinant=a*d-b*c
    return [(rewards[0]*d-b*rewards[1])/determinant,(a*rewards[1]-c*rewards[0])/determinant]


class RobustTests(unittest.TestCase):
    def test_243_models_exhaustive_controller_and_adversary_oracle(self):
        patterns=[[[F(1),F(0)],[F(0),F(1)]],[[F(1,2),F(1,3)],[F(2,3),F(1,2)]],[[F(0),F(1)],[F(1),F(0)]]]
        policies=list(itertools.product(range(2),repeat=2));count=0
        for k,probs in enumerate(patterns):
            for rewards in itertools.product([-1,0,1],repeat=4):
                other=[-r for r in rewards]
                problem={'version':1,'scenarios':[model(rewards,probs),model(other,patterns[(k+1)%3])]}
                fixed=[]
                for p in policies:
                    candidates=[cramers(problem,p,n) for n in policies];count+=len(candidates)
                    truth=[min(v[i] for v in candidates) for i in range(2)];fixed.append(truth)
                    exact=robust_audit(problem,[pair(x) for x in truth],policy=p)
                    self.assertEqual(exact['residual_sup'],[0,1])
                    run=robust_value_iteration(problem,policy=p,sweeps=4,tolerance=0)
                    self.check_history(problem,run,truth,p)
                truth=[max(v[i] for v in fixed) for i in range(2)]
                self.assertEqual(robust_audit(problem,[pair(x) for x in truth])['residual_sup'],[0,1])
                self.check_history(problem,robust_value_iteration(problem,sweeps=6,tolerance=0),truth,None)
        self.assertEqual(count,3888)

    def check_history(self,problem,run,truth,policy):
        g=F(1,2);previous=None
        entries=sum(len(a['transitions']) for m in problem['scenarios'] for s in m['states'] for a in s['actions'])
        self.assertEqual(run['executed_backup_transition_reads'],len(run['history'])*entries)
        for h in run['history']:
            v=list(map(lambda p:F(*p),h['values']));q=[]
            for i,state in enumerate(problem['scenarios'][0]['states']):
                q.append([[F(problem['scenarios'][k]['states'][i]['actions'][a]['reward'])+g*sum(F(*t['prob'])*v[t['next']] for t in problem['scenarios'][k]['states'][i]['actions'][a]['transitions']) for k in range(2)] for a in range(len(state['actions']))])
            self.assertEqual(h['scenario_q_values'],[[[pair(x) for x in a] for a in s] for s in q])
            worst=[[min(a) for a in s] for s in q]
            tv=[max(s) if policy is None else s[policy[i]] for i,s in enumerate(worst)]
            self.assertEqual(h['bellman_values'],list(map(pair,tv)))
            self.assertEqual(h['worst_case_scenarios'],[[[k for k,x in enumerate(a) if x==min(a)] for a in s] for s in q])
            self.assertEqual(h['greedy_actions'],[[a for a,x in enumerate(s) if x==max(s)] for s in worst])
            residual=max(abs(x-y) for x,y in zip(v,tv));self.assertEqual(h['residual_sup'],pair(residual))
            self.assertEqual(h['value_error_upper_bound'],pair(residual/(1-g)))
            self.assertLessEqual(max(abs(x-y) for x,y in zip(v,truth)),F(*h['value_error_upper_bound']))
            if previous is not None:self.assertEqual(h['values'],previous)
            previous=h['bellman_values']

    def test_single_scenario_matches_nominal_every_checkpoint(self):
        m=one_state([[2,5],[1,5]]);r=robust_value_iteration({'version':1,'scenarios':[m]},sweeps=9,tolerance=0);n=value_iteration(m,sweeps=9,tolerance=0)
        for a,b in zip(r['history'],n['history']):
            for key in ['values','bellman_values','residual_sup','value_error_upper_bound','greedy_actions','greedy_policy']:self.assertEqual(a[key],b[key])

    def test_fixture_and_coupled_distinction(self):
        d=robust_demo();c=d['decision_case'];self.assertEqual(c['exact_robust_loss_of_nominal_policy'],[6,5]);self.assertEqual(c['robust_optimal_iteration']['result']['greedy_policy'],[1]);r=d['rectangularity_case'];self.assertEqual(r['global_scenario_worst_start_value'],[-2,3]);self.assertEqual(r['rectangular_fixed_point_audit']['values'],[[-2,1],[-2,1]])

    def test_ties_duplicates_permutation(self):
        m=one_state([1,1]);p={'version':1,'scenarios':[m,copy.deepcopy(m)]};a=robust_audit(p,[0]);self.assertEqual(a['greedy_actions'],[[0,1]]);self.assertEqual(a['worst_case_scenarios'],[[[0,1],[0,1]]])
        p={'version':1,'scenarios':[one_state([1,2]),one_state([-1,3])]};a=robust_audit(p,[0]);p['scenarios'].reverse();b=robust_audit(p,[0]);self.assertEqual(a['worst_case_q_values'],b['worst_case_q_values']);self.assertEqual(b['worst_case_scenarios'],[[[0],[1]]])

    def test_fixed_policy_target_is_not_optimality(self):
        p={'version':1,'scenarios':[one_state([0,1])]};a=robust_audit(p,[0],policy=[0]);self.assertEqual(a['residual_sup'],[0,1]);self.assertEqual(a['greedy_policy'],[1]);self.assertIn('fixed policy',a['target']);self.assertEqual(robust_audit(p,[0])['residual_sup'],[1,1])

    def test_caps_zero_sweeps_and_initial_early_stop(self):
        p={'version':1,'scenarios':[one_state([1])]};a=robust_value_iteration(p,sweeps=0,tolerance=0);self.assertFalse(a['converged']);self.assertEqual(a['sweeps_completed'],0);self.assertEqual(len(a['history']),1)
        a=robust_value_iteration(p,initial=[2],sweeps=256,tolerance=0);self.assertTrue(a['converged']);self.assertEqual(a['sweeps_completed'],0);self.assertEqual(a['executed_backup_transition_reads'],1);self.assertEqual(a['planned_backup_transition_reads'],257)

    def test_zero_discount_negative_values(self):
        m=one_state([-2,-1]);m['discount']=0;p={'version':1,'scenarios':[m]};r=robust_value_iteration(p,tolerance=0);self.assertTrue(r['converged']);self.assertEqual(r['sweeps_completed'],1);self.assertEqual(r['result']['values'],[[-1,1]])

    def test_no_aliasing(self):
        p={'version':1,'scenarios':[one_state([1])]};before=copy.deepcopy(p);a=robust_value_iteration(p,sweeps=2,tolerance=0);a['history'][0]['values'][0][0]=999;a['problem']['scenarios'][0]['states'][0]['actions'][0]['reward'][0]=99;self.assertEqual(p,before);self.assertNotEqual(a['history'][1]['values'][0][0],999)

    def test_strict_schema_and_shape(self):
        good={'version':1,'scenarios':[one_state([1])]};bad=[]
        for x in [None,[],0,True]:bad.append(x)
        for k,v in [('version',True),('version',2),('scenarios',[]),('scenarios',[one_state([1])]*9),('scenarios',{})]:p=copy.deepcopy(good);p[k]=v;bad.append(p)
        p=copy.deepcopy(good);p['extra']=1;bad.append(p)
        for change in ['discount','actions','states']:
            p=copy.deepcopy(good);m=one_state([1])
            if change=='discount':m['discount']=[1,3]
            elif change=='actions':m['states'][0]['actions']*=2
            else:m['states']*=2
            p['scenarios'].append(m);bad.append(p)
        for p in bad:
            with self.assertRaises(ValueError):robust_audit(p,[0])

    def test_combined_lcm_guard_and_initial(self):
        p={'version':1,'scenarios':[one_state([[1,997]]),one_state([[1,991]]),one_state([[1,983]])]}
        with self.assertRaises(ValueError):robust_audit(p,[0])
        p={'version':1,'scenarios':[one_state([[1,997]]),one_state([[1,991]])]}
        with self.assertRaises(ValueError):robust_audit(p,[[1,983]])

    def test_aggregate_entry_boundary(self):
        m={'version':1,'discount':0,'states':[{'actions':[{'reward':0,'transitions':[{'next':j,'prob':[1,16]} for j in range(16)]} for a in range(2)]} for i in range(16)]}
        p={'version':1,'scenarios':[m]*4};self.assertEqual(robust_value_iteration(p,sweeps=0)['executed_backup_transition_reads'],2048)
        p['scenarios'].append(m)
        with self.assertRaises(ValueError):robust_audit(p,[0]*16)

    def test_parameter_rejections(self):
        p={'version':1,'scenarios':[one_state([1])]}
        for k,values in [('sweeps',[True,-1,257,1.5]),('tolerance',[-1,True,0.01]),('policy',[[True],[],[1]]),('initial',[[],[True],[[1000001,1]]])]:
            for v in values:
                with self.assertRaises(ValueError):robust_value_iteration(p,**{k:v})

    def test_read_zero_probability_entries(self):
        m={'version':1,'discount':0,'states':[{'actions':[{'reward':0,'transitions':[{'next':0,'prob':1},{'next':1,'prob':0}]}]},{'actions':[{'reward':0,'transitions':[{'next':1,'prob':1}]}]}]};p={'version':1,'scenarios':[m,m]};r=robust_value_iteration(p);self.assertEqual(r['executed_backup_transition_reads'],6)

    def test_cli(self):
        p=subprocess.run([sys.executable,'-m','bellmanlab','robust-demo'],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['decision_case']['exact_robust_loss_of_nominal_policy'],[6,5])

    def test_cli_model_modes_and_duplicate_keys(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.json';path.write_text(json.dumps({'version':1,'scenarios':[one_state([0,1])]}))
            for args in [['robust-audit',str(path),'--values','[0]','--policy','[0]'],['robust-value',str(path),'--policy','[0]','--tolerance','0']]:
                p=subprocess.run([sys.executable,'-m','bellmanlab',*args],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertIn('robust',json.loads(p.stdout)['format'])
            path.write_text('{"version":1,"version":1,"scenarios":[]}');p=subprocess.run([sys.executable,'-m','bellmanlab','robust-value',str(path)],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertIn('duplicate JSON key',p.stderr)

    def test_cli_budget_and_invalid_policy(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.json';path.write_bytes(b' '*262145);p=subprocess.run([sys.executable,'-m','bellmanlab','robust-value',str(path)],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertIn('256 KiB',p.stderr)
            path.write_text(json.dumps({'version':1,'scenarios':[one_state([0])]}));p=subprocess.run([sys.executable,'-m','bellmanlab','robust-audit',str(path),'--values','[0]','--policy','[1]'],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertIn('policy',p.stderr)
