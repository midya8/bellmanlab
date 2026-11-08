"""Two signed-value panels; exact full demo metadata; existing project style."""
import argparse,html,json
from fractions import Fraction as F
from pathlib import Path
from bellmanlab.robust_demo import robust_demo


def render(d):
    if d != robust_demo():raise ValueError('requires complete unchanged robust demo')
    c=d['decision_case'];r=d['rectangularity_case']
    values=[F(*c['nominal_risky_value']['values'][0]),F(*c['exact_risky_audit']['values'][0]),F(*c['exact_optimal_audit']['values'][0]),F(*r['global_scenario_worst_start_value']),F(*r['rectangular_fixed_point_audit']['values'][0])]
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="760" viewBox="0 0 1120 760">','<rect width="1120" height="760" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(d,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(30,42,'BellmanLab | the uncertainty set changes the answer',29)
    text(30,78,'Exact supplied scenarios, discount 1/2. Signed values on linear scales with a visible common zero.',17)
    text(30,122,'A nominal winner loses under ambiguity',21)
    text(610,122,'Rectangular is not globally coupled',21)
    panels=[(60,550,290,145,[-1,0,1]),(640,1080,290,72.5,[-2,-1,0,1,2])]
    for panel,(left,right,zero,scale,ticks) in enumerate(panels):
        for tick in ticks:
            y=zero-scale*tick;out.append(f'<path data-panel="{panel}" data-tick="{tick}" d="M {left} {y:g} H {right}" stroke="'+('#b7c4d9' if tick==0 else '#33445e')+'"/>');text(left-27,y+5,str(tick),14)
    for i,(value,label,x,panel,color) in enumerate(zip(values,['Nominal risky','Robust risky','Robust safe','One global model','Row-wise closure'],[115,275,435,720,930],[0,0,0,1,1],['#9ba9c0','#e58f99','#63d8bc','#9ba9c0','#e58f99'])):
        zero,scale=290,panels[panel][3];height=abs(value)*F(str(scale));y=F(zero)-max(value,F(0))*F(str(scale))
        out.append(f'<rect data-panel="{panel}" data-value="{value}" x="{x}" y="{float(y):.12g}" width="65" height="{float(height):.12g}" fill="{color}"/>')
        text(x-1,float(y)-12 if value>0 else float(y+height)+24,str(value),20,color);text(x-32,483,label,16)
    text(30,532,'Risky reward: +2/5 or -2/5. Safe: +1/5.',17)
    text(30,563,'Nominal choice: risky. Robust choice: safe.',17)
    text(30,594,'Exact robust loss of nominal policy: 6/5.',17,'#63d8bc')
    text(610,532,'Two alternating states; global rewards +/-1.',17)
    text(610,563,'One model fixed forever: worst start value -2/3.',17)
    text(610,594,'Independent row choices: start value -2.',17)
    text(30,646,'The max-action/min-scenario operator certifies its own rectangular target, not a globally coupled model.',17)
    text(30,681,'Full scenario Q-values, worst-case ties and exact residual/(1-discount) certificates remain inspectable.',17)
    text(30,716,'Both fixtures are directly specified. No learned confidence set, online RL or real-world safety claim.',16,'#b7c4d9')
    text(30,748,'Full exact robust demo, including every iteration checkpoint, is embedded in this SVG metadata.',15,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(robust_demo());path=Path(__file__).resolve().parents[1]/'assets'/'robust-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.write_text(s)
