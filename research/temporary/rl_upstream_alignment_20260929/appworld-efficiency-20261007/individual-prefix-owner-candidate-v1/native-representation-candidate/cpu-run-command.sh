set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import os,json,hashlib,time,subprocess,psutil
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');d=r/'candidates/appworld-row-cuts-finite-20261007-v1/native-representation-candidate';python=os.environ['VENV_PYTHON'];commands=[[python,str(d/'check_saved_target_rows_cpu.py')],[python,str(d/'check_contracts.py'),'--torch-interface','--output',str(d/'cpu-cache-interface.json')]];records=[]
for index,cmd in enumerate(commands):
 begin=time.time();out=d/f'cpu-command{index}.stdout.txt';err=d/f'cpu-command{index}.stderr.txt';peak_pss=peak_rss=0
 with out.open('wb') as stdout,err.open('wb') as stderr:
  proc=subprocess.Popen(cmd,env=dict(os.environ),cwd=d,stdout=stdout,stderr=stderr);birth=psutil.Process(proc.pid).create_time()
  while proc.poll() is None:
   try:
    tree=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True);mem=[p.memory_full_info() for p in tree];peak_pss=max(peak_pss,sum(m.pss for m in mem));peak_rss=max(peak_rss,sum(m.rss for m in mem))
   except psutil.NoSuchProcess:pass
   time.sleep(.1)
 records.append({'command':cmd,'pid':proc.pid,'birth':birth,'started_unix':begin,'finished_unix':time.time(),'returncode':proc.returncode,'sampled_peak_tree_PSS_bytes':peak_pss,'sampled_peak_tree_RSS_bytes':peak_rss,'scope_resource':'Independent CPU subprocess tree only; RSS is not claimed as physical usage; PSS is recorded separately.','stdout':str(out),'stderr':str(err),'stdout_text':out.read_text(),'stderr_text':err.read_text()})
 if proc.returncode:break
result={'scope':'Isolated CPU real-request target interfaces and small original DynamicCache interface only. No model/DT/FA/FLA/GPU/checkpoint execution or production changes.','environment':{k:os.environ.get(k) for k in ['VENV_PYTHON','CUDA_VISIBLE_DEVICES','MACA_VISIBLE_DEVICES']},'commands':records,'source_sha256':{str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*.py') if 'cpu-attempt1-assumption-error' not in str(p)},'all_commands_completed':len(records)==2,'all_returncodes_zero':all(x['returncode']==0 for x in records),'prior_attempt':'cpu-attempt1-assumption-error preserves the test-only erroneous assumption that the capture was already sorted. The original stable context sort is used now; owner sources unchanged.'}
(d/'cpu-execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
raise SystemExit(0 if result['all_commands_completed'] and result['all_returncodes_zero'] else 1)
PY
