import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from bellmanlab import audit,evaluate_policy,value_iteration,policy_iteration
from bellmanlab.core import _linear_solve
from bellmanlab.demo import demo


def f(x):return F(x) if type(x) is int else F(*x)
def vec(v):return list(map(f,v))
def mdp(rewards,g=[1,2],variant=0):
    # Three fixed stochastic transition patterns, two states/two actions.
    patterns=[[(1,0),(0,1),(0,1),(1,0)],[(F(1,2),F(1,2))]*4,[(F(1,3),F(2,3)),(1,0),(F(3,4),F(1,4)),(0,1)]]
    rows=patterns[variant];states=[]
    for i in range(2):
        actions=[]
        for a in range(2):
            probs=rows[2*i+a];actions.append({'reward':rewards[2*i+a],'transitions':[{'next':j,'prob':[F(q).numerator,F(q).denominator]} for j,q in enumerate(probs)]})
        states.append({'actions':actions})
    return {'version':1,'discount':g,'states':states}
def cramer_two(m,policy):
    g=f(m['discount']);a=[];b=[]
    for i,k in enumerate(policy):
        row=m['states'][i]['actions'][k];p={x['next']:f(x['prob']) for x in row['transitions']};a.append([F(i==j)-g*p.get(j,0) for j in range(2)]);b.append(f(row['reward']))
    det=a[0][0]*a[1][1]-a[0][1]*a[1][0]
    return [(b[0]*a[1][1]-a[0][1]*b[1])/det,(a[0][0]*b[1]-b[0]*a[1][0])/det]
def tiny(reward=1,g=[1,2]):return {'version':1,'discount':g,'states':[{'actions':[{'reward':reward,'transitions':[{'next':0,'prob':1}]}]}]}


class CoreTests(unittest.TestCase):
    def test_243_independent_policy_enumerations(self):
        count=0
        for rewards in itertools.product([-1,0,1],repeat=4):
            for variant in range(3):
                m=mdp(rewards,variant=variant);policies=list(itertools.product(range(2),repeat=2));expected=[cramer_two(m,p) for p in policies];best=[max(v[i] for v in expected) for i in range(2)]
                for p,v in zip(policies,expected):self.assertEqual(vec(evaluate_policy(m,p)['values']),v)
                pi=policy_iteration(m);self.assertTrue(pi['optimal']);self.assertEqual(vec(pi['values']),best)
                for row in pi['history']:
                    self.assertEqual(vec(row['values']),cramer_two(m,row['policy']))
                vi=value_iteration(m,sweeps=4,tolerance=0)
                for row in vi['history']:
                    error=max(abs(a-b) for a,b in zip(vec(row['values']),best));self.assertLessEqual(error,f(row['value_error_upper_bound']))
                count+=1
        self.assertEqual(count,243)
    def test_each_synchronous_backup(self):
        m=mdp([-1,1,0,2],variant=2);r=value_iteration(m,initial=[1,-1],sweeps=4,tolerance=0)
        for a,b in zip(r['history'],r['history'][1:]):
            v=vec(a['values']);expected=[]
            for state in m['states']:expected.append(max(f(x['reward'])+f(m['discount'])*sum(f(t['prob'])*v[t['next']] for t in x['transitions']) for x in state['actions']))
            self.assertEqual(vec(b['values']),expected)
    def test_residual_and_greedy_reconstruction(self):
        m=mdp([1,-1,2,0],variant=2);v=[F(1,2),F(-1,2)];r=audit(m,[[1,2],[-1,2]])
        for i,state in enumerate(m['states']):
            q=[f(x['reward'])+F(1,2)*sum(f(t['prob'])*v[t['next']] for t in x['transitions']) for x in state['actions']]
            self.assertEqual(vec(r['q_values'][i]),q);self.assertEqual(r['greedy_actions'][i],[j for j,x in enumerate(q) if x==max(q)])
        residual=max(abs(max(vec(row))-x) for row,x in zip(r['q_values'],v));self.assertEqual(f(r['residual_sup']),residual);self.assertEqual(f(r['value_error_upper_bound']),2*residual)
    def test_demo_both_residual_gates_and_delayed_return(self):
        r=demo();self.assertEqual(r['gates'],{'raw_residual_within_value_tolerance':True,'initial_certified_value_error_within_tolerance':False,'final_certified_value_error_within_tolerance':True,'improved_policy_is_optimal':True})
        x=r['residual_case'];self.assertEqual(x['initial_audit']['residual_sup'],[1,100]);self.assertEqual(x['initial_audit']['value_error_upper_bound'],[1,1]);self.assertEqual(x['iteration']['sweeps_completed'],230)
        self.assertEqual(r['delayed_case']['policy_iteration']['values'][0],[9,5])
    def test_exact_one_state_closed_form(self):
        for g in [0,[1,2],[99,100]]:
            for reward in [-1,0,[1,100]]:
                r=evaluate_policy(tiny(reward,g),[0]);self.assertEqual(f(r['values'][0]),f(reward)/(1-f(g)));self.assertTrue(r['is_optimal'])
    def test_gamma_zero(self):
        m=mdp([1,2,4,3],g=0);r=value_iteration(m,sweeps=1,tolerance=0);self.assertEqual(r['result']['values'],[[2,1],[4,1]]);self.assertTrue(r['converged'])
    def test_16_state_chain_closed_form(self):
        m={'version':1,'discount':[1,2],'states':[{'actions':[{'reward':1 if i==15 else 0,'transitions':[{'next':min(i+1,15),'prob':1}]}]} for i in range(16)]}
        v=vec(evaluate_policy(m,[0]*16)['values']);self.assertEqual(v,[F(2)*F(1,2)**(15-i) for i in range(16)])
    def test_ties_retain_old_policy(self):
        m=mdp([0]*4);r=policy_iteration(m,initial=[1,1]);self.assertEqual(r['policy'],[1,1]);self.assertEqual(r['history'][0]['greedy_actions'],[[0,1],[0,1]]);self.assertEqual(r['iterations_completed'],1)
    def test_exact_small_advantage(self):
        m=mdp([0,[1,1000000],0,0]);r=policy_iteration(m);self.assertEqual(r['policy'][0],1);self.assertTrue(r['optimal'])
    def test_budget_returns_evaluated_not_proposed_policy(self):
        m=demo()['delayed_case']['model'];r=policy_iteration(m,iterations=1);self.assertFalse(r['optimal']);self.assertEqual(r['policy'],[0,0,0]);self.assertEqual(r['history'][0]['improved_policy'],[1,0,0]);self.assertEqual(r['values'][0],[1,1])
    def test_value_cap_not_fake_convergence(self):
        r=value_iteration(tiny(1,[99,100]),sweeps=1,tolerance=0);self.assertFalse(r['converged']);self.assertEqual(r['sweeps_completed'],1)
    def test_zero_sweeps_and_initial_exact_solution(self):
        r=value_iteration(tiny(),initial=[2],sweeps=0,tolerance=0);self.assertTrue(r['converged']);self.assertEqual(r['sweeps_completed'],0);self.assertEqual(len(r['history']),1)
    def test_backup_counters_include_final_audit(self):
        m=mdp([1]*4,variant=1);r=value_iteration(m,sweeps=3,tolerance=0);self.assertEqual(r['planned_backup_transition_reads'],4*8);self.assertEqual(r['executed_backup_transition_reads'],len(r['history'])*8)
    def test_gauss_jordan_nonzero_pivot_and_singularity(self):
        self.assertEqual(_linear_solve([[F(0),F(1)],[F(1),F(0)]],[F(2),F(3)]),[F(3),F(2)])
        with self.assertRaises(ValueError):_linear_solve([[F(0)]],[F(1)])
    def test_policy_values_monotone(self):
        r=policy_iteration(mdp([-1,1,0,2],variant=2));values=[vec(x['values']) for x in r['history']]
        for a,b in zip(values,values[1:]):self.assertTrue(all(y>=x for x,y in zip(a,b)))
    def test_canonical_roundtrip_and_successor_order(self):
        m=mdp([1,0,0,1],variant=2);a=policy_iteration(m)
        for state in m['states']:
            for row in state['actions']:row['transitions'].reverse()
        self.assertEqual(a,policy_iteration(m));self.assertEqual(a,policy_iteration(a['model']))
    def test_model_and_output_rng_isolation(self):
        m=mdp([1,0,0,1]);before=copy.deepcopy(m);random.seed(92);state=random.getstate();r=policy_iteration(m);r['history'][0]['values'][0][0]=999
        self.assertEqual(m,before);self.assertEqual(random.getstate(),state);self.assertNotEqual(r,policy_iteration(m))
    def test_model_keys_version_states(self):
        for m in [{},dict(tiny(),extra=1),dict(tiny(),version=True),dict(tiny(),states=[]),dict(tiny(),states=tiny()['states']*17)]:
            with self.assertRaises(ValueError):policy_iteration(m)
    def test_rational_validation(self):
        for x in [True,.5,'1/2',[1,0],[1,-1],[True,2],[1000001,1],[1,1000001],[1,2,3],float('nan')]:
            with self.assertRaises(ValueError):policy_iteration(tiny(g=x))
    def test_discount_validation(self):
        for g in [-1,1,2]:
            with self.assertRaises(ValueError):policy_iteration(tiny(g=g))
    def test_reward_and_action_validation(self):
        for reward in [101,-101]:
            with self.assertRaises(ValueError):policy_iteration(tiny(reward))
        for actions in [[],tiny()['states'][0]['actions']*9,None]:
            m=tiny();m['states'][0]['actions']=actions
            with self.assertRaises(ValueError):policy_iteration(m)
    def test_probability_row_and_duplicate_validation(self):
        for transitions in [[],[{'next':0,'prob':[1,2]}],[{'next':True,'prob':1}],[{'next':1,'prob':1}],[{'next':0,'prob':-1}],[{'next':0,'prob':1},{'next':0,'prob':0}]]:
            m=tiny();m['states'][0]['actions'][0]['transitions']=transitions
            with self.assertRaises(ValueError):policy_iteration(m)
    def test_denominator_lcm_guard_before_solve(self):
        m=mdp([[1,997],[1,991],[1,983],0]);
        with patch('bellmanlab.core._linear_solve',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError,'LCM'):policy_iteration(m)
    def test_initial_lcm_and_length_guard(self):
        for initial in [[],[True],[[1,999983]]]:
            with self.assertRaises(ValueError):value_iteration(tiny(g=[1,997]),initial=initial)
    def test_global_action_entry_caps(self):
        m={'version':1,'discount':0,'states':[{'actions':[{'reward':0,'transitions':[{'next':j,'prob':[1,16]} for j in range(16)]} for a in range(4)]} for i in range(16)]}
        with self.assertRaisesRegex(ValueError,'512'):policy_iteration(m)
        for state in m['states']:
            for row in state['actions']:row['transitions']=[{'next':0,'prob':1}]
        policy_iteration(m)
        m['states'][0]['actions'].append(copy.deepcopy(m['states'][0]['actions'][0]))
        with self.assertRaisesRegex(ValueError,'64 actions'):policy_iteration(m)
    def test_policy_and_budget_validation(self):
        for policy in [None,[],[True],[1]]:
            with self.assertRaises(ValueError):evaluate_policy(tiny(),policy)
        for k in [True,-1,257,1.0]:
            with self.assertRaises(ValueError):value_iteration(tiny(),sweeps=k)
        for k in [True,0,65,1.0]:
            with self.assertRaises(ValueError):policy_iteration(tiny(),iterations=k)
        with self.assertRaises(ValueError):value_iteration(tiny(),tolerance=-1)
    def test_json_and_repeat_determinism(self):
        m=mdp([1,0,0,1],variant=1);r=policy_iteration(m);self.assertEqual(json.loads(json.dumps(r)),r);self.assertEqual(policy_iteration(m),r)
    def test_cli_equal_all_commands(self):
        m=tiny()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.json';path.write_text(json.dumps(m))
            for args,expected in [(['audit','--values','[0]'],audit(m,[0])),(['evaluate','--policy','[0]'],evaluate_policy(m,[0])),(['value','--sweeps','2'],value_iteration(m,sweeps=2)),(['policy'],policy_iteration(m))]:
                out=subprocess.run([sys.executable,'-m','bellmanlab',args[0],str(path)]+args[1:],text=True,capture_output=True);self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout),expected)
    def test_cli_errors_size_and_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.json'
            for raw in ['{"version":1,"version":1}', ' '*262145, json.dumps(tiny(.5))]:
                path.write_text(raw);out=subprocess.run([sys.executable,'-m','bellmanlab','policy',str(path)],text=True,capture_output=True);self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'')
    def test_cli_demo(self):
        out=subprocess.run([sys.executable,'-m','bellmanlab','demo'],text=True,capture_output=True);self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout),demo())
