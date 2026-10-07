#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('start',type=int); p.add_argument('end',type=int)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
files=sorted((root/'tests').glob('test_*.py'))
selected=files[a.start:a.end]
cmd=[sys.executable,'-m','pytest','-q',*[str(f.relative_to(root)) for f in selected]]
env=dict(os.environ); env['PYTHONPATH']=str(root/'src')
t0=time.time(); proc=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True)
runtime=time.time()-t0
text=proc.stdout+'\n'+proc.stderr
m_pass=re.search(r'(\d+) passed',text)
m_fail=re.search(r'(\d+) failed',text)
report={
 'schema':'komorebi-math.episode009-source-batch','schema_version':1,
 'start':a.start,'end':a.end,'files':[str(f.relative_to(root)) for f in selected],
 'exit_status':proc.returncode,'runtime_seconds':runtime,
 'passed_tests':int(m_pass.group(1)) if m_pass else 0,
 'failed_tests':int(m_fail.group(1)) if m_fail else 0,
 'stdout_tail':'\n'.join(proc.stdout.splitlines()[-30:]),
 'stderr_tail':'\n'.join(proc.stderr.splitlines()[-30:]),
 'passed':proc.returncode==0,
}
out=root/'release'/f'EPISODE009_SOURCE_QUALIFICATION_{a.start:03d}_{a.end-1:03d}.json'
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
sys.exit(proc.returncode)
