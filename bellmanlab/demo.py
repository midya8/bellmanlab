from fractions import Fraction as F
from .core import audit,value_iteration,policy_iteration,evaluate_policy


def demo():
    residual={'version':1,'discount':[99,100],'states':[{'actions':[{'reward':[1,100],'transitions':[{'next':0,'prob':1}]}]}]}
    initial=audit(residual,[0]);truth=evaluate_policy(residual,[0]);vi=value_iteration(residual,sweeps=256,tolerance=[1,10])
    raw=F(*initial['residual_sup'])<=F(1,10);certified=F(*initial['value_error_upper_bound'])<=F(1,10)
    assert raw and not certified and truth['values']==[[1,1]] and vi['converged']
    delayed={'version':1,'discount':[9,10],'states':[
        {'actions':[{'reward':1,'transitions':[{'next':2,'prob':1}]},{'reward':0,'transitions':[{'next':1,'prob':1}]}]},
        {'actions':[{'reward':[1,5],'transitions':[{'next':1,'prob':1}]}]},
        {'actions':[{'reward':0,'transitions':[{'next':2,'prob':1}]}]}]}
    immediate=evaluate_policy(delayed,[0,0,0]);optimal=policy_iteration(delayed);assert immediate['values'][0]==[1,1] and optimal['values'][0]==[9,5] and optimal['policy']==[1,0,0] and optimal['optimal']
    return {'residual_case':{'model':residual,'initial_audit':initial,'exact_policy_value':truth,'iteration':vi},
            'delayed_case':{'model':delayed,'immediate_policy':immediate,'policy_iteration':optimal},
            'gates':{'raw_residual_within_value_tolerance':raw,'initial_certified_value_error_within_tolerance':certified,'final_certified_value_error_within_tolerance':vi['converged'],'improved_policy_is_optimal':optimal['optimal']},
            'selection':'directly specified one-state reward 1/100, discount 99/100; separate three-state immediate-one versus delayed reward 1/5 with discount 9/10; no parameter, seed or policy search for fixture selection',
            'scope':'complete supplied discounted finite models, not learned transitions, statistical confidence, sample efficiency or an online reinforcement-learning benchmark'}
