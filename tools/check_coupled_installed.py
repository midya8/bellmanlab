"""Build a wheel; run old demos and every new control outside the source tree."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(args,**kwargs):
    try:return subprocess.run(args,check=True,capture_output=True,text=True,**kwargs)
    except subprocess.CalledProcessError as e:
        print(e.stdout or '',file=sys.stderr);print(e.stderr or '',file=sys.stderr)
        raise

def main():
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory()as td:
        tmp=Path(td);run([sys.executable,'-m','pip','wheel','--no-deps','-w',str(tmp),str(root)])
        run([sys.executable,'-m','venv',str(tmp/'env')]);py=tmp/'env'/('Scripts/python.exe'if os.name=='nt'else'bin/python');cli=py.with_name('bellmanlab-coupled');oldcli=py.with_name('bellmanlab')
        wheel=next(tmp.glob('*.whl'));run([str(py),'-m','pip','install','--no-deps',str(wheel)],cwd=tmp)
        origins=run([str(py),'-c','import importlib,json;mods=[importlib.import_module("bellmanlab."+x) for x in ["core","robust","robust_policy","occupancy","occupancy_check","occupancy_demo","occupancy_cli","coupled","coupled_check","coupled_demo","coupled_cli"]];assert all("site-packages" in m.__file__ for m in mods);print(len(mods))'],cwd=tmp);assert origins.stdout.strip()=='11'
        names=json.loads(run([str(py),'-c','import json;from bellmanlab.coupled_demo import controls;print(json.dumps(list(controls())))'],cwd=tmp).stdout)
        direct=forged=0
        for name in names:
            a=json.loads(run([str(py),'-m','bellmanlab.coupled_cli','demo','--control',name],cwd=tmp).stdout);b=json.loads(run([str(cli),'demo','--control',name],cwd=tmp).stdout);assert a==b
            p=tmp/'input.json';p.write_text(json.dumps(a['problem']));report=tmp/'report.json';report.write_text(json.dumps(a))
            for cmd in [[str(py),'-m','bellmanlab.coupled_cli'],[str(cli)]]:
                assert json.loads(run(cmd+['run',str(p)],cwd=tmp).stdout)==a
                assert json.loads(run(cmd+['verify',str(report)],cwd=tmp).stdout)['verified']
            for which in [0,1]:
                bad=json.loads(json.dumps(a))
                if not which:bad['feasible']=not bad['feasible']
                elif bad['selected']:bad['selected']['dual']['upper_bound']=[999,1]
                else:bad['infeasibility']['gap']=[999,1]
                report.write_text(json.dumps(bad))
                for cmd in [[str(py),'-m','bellmanlab.coupled_cli'],[str(cli)]]:
                    z=subprocess.run(cmd+['verify',str(report)],capture_output=True,text=True,cwd=tmp);assert z.returncode==2 and z.stdout=='';forged+=1
                z=run([str(py),'-c','import json,sys;from bellmanlab.coupled_check import verify_coupled\ntry:verify_coupled(json.load(open(sys.argv[1])))\nexcept ValueError: print("rejected")\nelse:raise AssertionError("forgery accepted")',str(report)],cwd=tmp);assert z.stdout.strip()=='rejected';direct+=1
        print(json.dumps(dict(wheel=wheel.name,installed_module_origins=11,controls=len(names),entrypoints=2,cli_forgery_rejections=forged,direct_forgery_rejections=direct),sort_keys=True))
if __name__=='__main__':main()
