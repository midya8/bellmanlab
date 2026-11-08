"""Exact certificate payloads; linear common-zero value-error scale."""
import argparse,html,json
from fractions import Fraction as F
from pathlib import Path
from bellmanlab.demo import demo


def frozen(d):
    case=d['residual_case'];trace=case['iteration']
    return {'model':case['model'],'initial':case['initial_audit'],'exact_policy_value':case['exact_policy_value'],
            'final':trace['result'],'sweeps_completed':trace['sweeps_completed'],'tolerance':trace['tolerance'],
            'selected_history':[r for r in trace['history'] if r['sweep'] in [0,1,10,100,229,230]],
            'delayed_case':d['delayed_case'],'gates':d['gates'],'selection':d['selection'],'scope':d['scope'],
            'metadata_scope':'exact initial/final certificates, six explicitly selected trace checkpoints, full delayed-policy trace; full 231-audit value history remains in CLI demo, no data omitted from the API'}

def render(d):
    if d!=demo():raise ValueError('requires complete unchanged demo')
    fixed=frozen(d);initial=d['residual_case']['initial_audit'];values=[F(*initial['residual_sup']),F(*initial['value_error_upper_bound']),F(1)]
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="700" viewBox="0 0 1120 700">','<rect width="1120" height="700" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(fixed,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(35,43,'BellmanLab | a small residual is not a small value error',29)
    text(35,80,'One-state supplied MDP: reward 1/100, discount 99/100, initial value 0. Exact value = 1.',17)
    text(35,114,'Same linear scale, shared zero. The raw residual misses the value-error tolerance of 1/10.',16)
    for numerator in range(0,11,2):
        y=450-260*numerator/10;out.append(f'<path d="M 80 {y:g} H 1070" stroke="#33445e"/>');text(30,y+5,str(numerator/10),14)
    out.append('<path data-tolerance="1/10" d="M 80 424 H 1070" stroke="#efbd68" stroke-dasharray="6 5"/>');text(90,414,'value tolerance = 0.1',16,'#efbd68')
    for i,(v,label) in enumerate(zip(values,['Raw residual','Certified error bound','Actual value error'])):
        x=170+330*i;height=260*v;y=450-height;color='#63d8bc' if i==0 else '#e58f99'
        out.append(f'<rect data-value="{v}" x="{x}" y="{float(y):g}" width="90" height="{float(height):g}" fill="{color}"/>');text(x-5,float(y)-14,str(v),20,color);text(x-35,488,label,17)
    text(35,533,'Certificate: ||v - v*|| infinity <= residual / (1 - discount). Exact fractions, no stopping tolerance in equality.',16)
    text(35,568,'230 synchronous sweeps bring the certified value error below 1/10; every backup and audit is returned.',16)
    text(35,603,'Separate delayed-return fixture: immediate policy value 1; exact policy iteration reaches 9/5 at the start state.',16)
    text(35,639,'Complete supplied discounted model, not online RL, learned transitions or a statistical confidence bound.',16,'#b7c4d9')
    text(35,676,'Exact initial/final certificates and selected checkpoints in SVG metadata; full histories remain in CLI demo.',15,'#b7c4d9')
    out.append('</svg>');return '\n'.join(out)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(demo());path=Path(__file__).resolve().parents[1]/'assets'/'bellman-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.write_text(s)
