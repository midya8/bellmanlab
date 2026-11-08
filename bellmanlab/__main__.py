import argparse,json
from .core import audit,evaluate_policy,value_iteration,policy_iteration
from .demo import demo

def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out

def main(argv=None):
    p=argparse.ArgumentParser(description='Exact rational planning for supplied discounted finite MDPs');s=p.add_subparsers(dest='command',required=True);s.add_parser('demo')
    for name in ['audit','evaluate','value','policy']:
        a=s.add_parser(name);a.add_argument('model')
        if name=='audit':a.add_argument('--values',required=True)
        if name=='evaluate':a.add_argument('--policy',required=True)
        if name=='value':a.add_argument('--initial');a.add_argument('--sweeps',type=int,default=100);a.add_argument('--tolerance',default='[1,1000]')
        if name=='policy':a.add_argument('--initial');a.add_argument('--iterations',type=int,default=64)
    args=p.parse_args(argv)
    try:
        if args.command=='demo':out=demo()
        else:
            with open(args.model,'rb') as f:raw=f.read(262145)
            if len(raw)>262144:raise ValueError('model JSON exceeds 256 KiB')
            m=json.loads(raw,object_pairs_hook=pairs)
            if args.command=='audit':out=audit(m,json.loads(args.values))
            elif args.command=='evaluate':out=evaluate_policy(m,json.loads(args.policy))
            elif args.command=='value':out=value_iteration(m,initial=json.loads(args.initial) if args.initial is not None else None,sweeps=args.sweeps,tolerance=json.loads(args.tolerance))
            else:out=policy_iteration(m,initial=json.loads(args.initial) if args.initial is not None else None,iterations=args.iterations)
        text=json.dumps(out,indent=2,allow_nan=False)
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,'error: '+str(e)+'\n')
    print(text)
if __name__=='__main__':main()
