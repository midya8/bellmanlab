"""Build a wheel; run old demos and every new control outside the source tree."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(args,**kwargs):return subprocess.run(args,check=True,capture_output=True,text=True,**kwargs)

def main():
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory()as td:
        tmp=Path(td);run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','-w',str(tmp),str(root)])
        run([sys.executable,'-m','venv',str(tmp/'env')]);py=tmp/'env'/('Scripts/python.exe'if os.name=='nt'else'bin/python');cli=py.with_name('bellmanlab-occupancy');oldcli=py.with_name('bellmanlab')
        wheel=next(tmp.glob('*.whl'));run([str(py),'-m','pip','install','--no-deps',str(wheel)],cwd=tmp)
        proof=run([str(py),'-c','import bellmanlab, bellmanlab.core, bellmanlab.robust, bellmanlab.robust_policy, bellmanlab.occupancy, bellmanlab.occupancy_check, bellmanlab.occupancy_demo, bellmanlab.occupancy_cli; import pathlib; modules=[bellmanlab.core,bellmanlab.robust,bellmanlab.robust_policy,bellmanlab.occupancy,bellmanlab.occupancy_check,bellmanlab.occupancy_demo,bellmanlab.occupancy_cli];assert all("site-packages" in m.__file__ for m in modules);print(len(modules))'],cwd=tmp);assert proof.stdout.strip()=='7'
        for command in ['demo','robust-demo','robust-policy-demo']:
            a=json.loads(run([str(py),'-m','bellmanlab',command],cwd=tmp).stdout);b=json.loads(run([str(oldcli),command],cwd=tmp).stdout);assert a==b
        names=json.loads(run([str(py),'-c','import json;from bellmanlab.occupancy_demo import controls;print(json.dumps(list(controls())))'],cwd=tmp).stdout)
        direct=forged=0
        for name in names:
            a=json.loads(run([str(py),'-m','bellmanlab.occupancy_cli','demo','--control',name],cwd=tmp).stdout);b=json.loads(run([str(cli),'demo','--control',name],cwd=tmp).stdout);assert a==b
            p=tmp/'input.json';p.write_text(json.dumps(a['problem']));report=tmp/'report.json';report.write_text(json.dumps(a))
            for cmd in [[str(py),'-m','bellmanlab.occupancy_cli'],[str(cli)]]:
                assert json.loads(run(cmd+['run',str(p)],cwd=tmp).stdout)==a
                assert json.loads(run(cmd+['verify',str(report)],cwd=tmp).stdout)['verified']
            for mutate in [lambda r:r.update(feasible=not r['feasible']),lambda r:r['min_cost_certificate'].update(lower_bound=[999,1])]:
                bad=json.loads(json.dumps(a));mutate(bad);report.write_text(json.dumps(bad))
                for cmd in [[str(py),'-m','bellmanlab.occupancy_cli'],[str(cli)]]:
                    z=subprocess.run(cmd+['verify',str(report)],capture_output=True,text=True,cwd=tmp);assert z.returncode==2 and z.stdout=='';forged+=1
                z=run([str(py),'-c','import json,sys;from bellmanlab.occupancy_check import verify_occupancy\ntry:verify_occupancy(json.load(open(sys.argv[1])))\nexcept ValueError: print("rejected")\nelse:raise AssertionError("forgery accepted")',str(report)],cwd=tmp);assert z.stdout.strip()=='rejected';direct+=1
        print(json.dumps(dict(wheel=wheel.name,installed_module_origins=7,previous_demo_families=3,controls=len(names),entrypoints=2,cli_forgery_rejections=forged,direct_forgery_rejections=direct),sort_keys=True))
if __name__=='__main__':main()
