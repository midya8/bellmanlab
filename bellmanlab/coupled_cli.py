"""Strict new entrypoint, output appears only after complete computation."""
import argparse
import json
from pathlib import Path
from .coupled import coupled
from .coupled_check import verify_coupled
from .coupled_demo import controls
from .occupancy_cli import _object


def load(path,limit):
    data=Path(path).read_bytes()
    if len(data)>limit: raise ValueError('JSON byte limit exceeded')
    def no_float(x): raise ValueError('use integer or rational pair, no decimal/exponent/nonfinite token')
    return json.loads(data,object_pairs_hook=_object,parse_float=no_float,parse_constant=no_float)


def main():
    p=argparse.ArgumentParser(description='Two expected-cost budgets, not per-trajectory safety')
    p.add_argument('command',choices=['run','verify','demo']);p.add_argument('file',nargs='?');p.add_argument('--control',choices=list(controls()),default='three_way');args=p.parse_args()
    try:
        if args.command=='demo':
            if args.file: raise ValueError('demo takes no file')
            out=coupled(controls()[args.control])
        else:
            if not args.file: raise ValueError('file required')
            data=load(args.file,2097152 if args.command=='verify' else 65536)
            out=dict(verified=verify_coupled(data),format='bellmanlab-coupled-verification-v1') if args.command=='verify' else coupled(data)
        print(json.dumps(out,sort_keys=True,separators=(',',':')))
    except (ValueError,UnicodeError,OSError,RecursionError,KeyError,TypeError,OverflowError) as e: p.exit(2,'error: '+str(e)+'\n')


if __name__=='__main__': main()
