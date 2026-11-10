"""Strict independent occupancy entrypoint; existing CLI remains unchanged."""
import argparse
import json
from pathlib import Path
from .occupancy import occupancy
from .occupancy_check import verify_occupancy
from .occupancy_demo import controls


def _object(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise ValueError('duplicate JSON key')
        d[k]=v
    return d


def load(path):
    data=Path(path).read_bytes()
    if len(data)>65536: raise ValueError('JSON limited to 65536 bytes')
    def no_float(x): raise ValueError('use integer or rational pairs, not decimal/exponent/nonfinite tokens')
    return json.loads(data,object_pairs_hook=_object,parse_float=no_float,parse_constant=no_float)


def main():
    p=argparse.ArgumentParser(description='Finite normalized discounted expected-cost planning, not hard safety')
    p.add_argument('command',choices=['run','verify','demo']);p.add_argument('file',nargs='?');p.add_argument('--control',choices=list(controls()),default='visitation_reweighting');a=p.parse_args()
    try:
        if a.command=='demo':
            if a.file: raise ValueError('demo accepts no file')
            out=occupancy(controls()[a.control])
        else:
            if not a.file: raise ValueError('file required')
            data=load(a.file)
            out={'verified':verify_occupancy(data),'format':'bellmanlab-occupancy-verification-v1'} if a.command=='verify' else occupancy(data)
        print(json.dumps(out,sort_keys=True,separators=(',',':')))
    except (ValueError,UnicodeError,OSError,RecursionError) as e: p.exit(2,'error: '+str(e)+'\n')


if __name__=='__main__':main()
