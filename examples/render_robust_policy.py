"""Evaluated versus proposed policies in nested robust planning."""
import argparse,html,json
from pathlib import Path
from bellmanlab.robust_policy_demo import robust_policy_demo


def render(d):
    if d!=robust_policy_demo():raise ValueError('requires complete unchanged policy demo')
    cases=[d[k] for k in ['inner_capped','controller_capped','complete']]
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="700" viewBox="0 0 1120 700">','<rect width="1120" height="700" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(d,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8',**attrs):
        extra=' '.join(k.replace('_','-')+'="'+html.escape(str(v))+'"' for k,v in attrs.items())
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}" {extra}>'+html.escape(s)+'</text>')
    text(30,42,'BellmanLab | a proposed policy is not an evaluated policy',28)
    text(30,80,'Same one-state uncertainty set in every row. Only the evaluation caps change. Discount = 1/2.',17)
    for x,s in [(30,'Termination'),(400,'Controller'),(545,'Nature'),(660,'Value'),(790,'Exact?'),(920,'Optimal?')]:text(x,133,s,18)
    labels=['Inner cap = 1','Controller cap = 1','Both policies stable']
    for i,(case,label) in enumerate(zip(cases,labels)):
        y=196+110*i;out.append(f'<path d="M 30 {y-34} H 1090" stroke="#33445e"/>')
        text(30,y,label,19)
        for field,x in [('policy',400),('nature_policy',545),('values',660),('policy_value_exact',790),('optimal',920)]:
            value=case[field]
            if field in ['policy','nature_policy']:s=str(value)
            elif field=='values':s='/'.join(map(str,value[0]))
            else:s='yes' if value else 'no'
            text(x,y,s,20,'#63d8bc' if value is True else '#e7edf8',data_case=i,data_field=field,data_exact=json.dumps(value,separators=(',',':')))
        notes=['Value is an upper bound for risky policy. Controller improvement is withheld.',
               'Risky value is exact. Proposed safe controller [1] is NOT returned as evaluated.',
               'Safe controller [1] and nature [0] are evaluated. Both stability checks pass.']
        text(30,y+37,notes[i],16,'#b7c4d9')
    text(30,537,'Inner exact solve: evaluate nature [0] at 4/5, improve to [1], then evaluate -4/5 and verify stability.',17)
    text(30,576,'Outer improvement then selects safe [1]. Its exact value is 2/5; total: 2 controller / 3 nature evaluations.',17)
    text(30,615,'Full nested histories contain solved equations, worst-case ties, residuals and separate unevaluated proposals.',17)
    text(30,654,'Stability proves robust optimality, not policy uniqueness. Inner caps never manufacture outer improvements.',16,'#b7c4d9')
    text(30,686,'Full exact demo embedded in SVG metadata. Prescribed model and budgets, not a speed or learning benchmark.',15,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(robust_policy_demo());path=Path(__file__).resolve().parents[1]/'assets'/'robust-policy-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.write_text(s)
