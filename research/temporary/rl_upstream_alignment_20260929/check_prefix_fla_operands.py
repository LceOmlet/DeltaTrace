"""Run the existing FA/FLA verification entry on newly recorded FLA operands.

No model is loaded. The original script, recurrence and assertions are used
unchanged. This bounded operator-only process shares our TextCraft GPU4.
"""
from stage_environment_entry import remote, ROOT, ENTRY

remote(r'''source @ENTRY@/metax-entry.env.sh
export CUDA_VISIBLE_DEVICES=4
unset MACA_VISIBLE_DEVICES
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,resource,runpy,sys,time,torch
from pathlib import Path
root=Path('@ROOT@');receipt=root/'receipts/owner-b8-dispatch-20260930/native-dt-gathers-1790848462'
script=root/'receipts/upstream-alignment-20260929/verify_saved_fla_dtypes.py'
sources=root/'receipts/training-setup/official-kernel-tests'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='386a389fd6fbbc3a80aa7ab499610a8164ac7902578e65cda64629049ce97748'
sys.path.insert(0,str(root/'releases/c9cd147/clean/qwen35'))
sys.path.insert(0,str(root/'releases/c9cd147/clean/qwen3'))
import finite_fla_gpu,verify_official_kernel_tolerances,profiles.qwen35_gdn_symmetric
baseline=json.loads((root/'receipts/environment-only-20260930/entry/verified_runtime.json').read_text())
for relative in ['clean/qwen35/finite_fla_gpu.py','profiles/qwen35_gdn_symmetric.py']:
 p=root/'releases/c9cd147'/relative
 assert hashlib.sha256(p.read_bytes()).hexdigest()==baseline['dt_source_sha256'][relative]
record=dict(pid=os.getpid(),started_unix=time.time(),physical_device=4,scope=__doc__,
 script=str(script),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),runs=[],
 imports={m.__name__:dict(path=m.__file__,sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())
  for m in [finite_fla_gpu,verify_official_kernel_tolerances,profiles.qwen35_gdn_symmetric]})
path=receipt/'original-fla-checks.json'
assert not path.exists(), 'Inspect any existing operand check; do not overwrite'
path.write_text(json.dumps(record,indent=2)+'\n')
for label in ['native_first','observed_cut_320']:
 output=receipt/f'original-fla-{label}.json'
 operands=receipt/f'finite-{label}-rank0.pt'
 print(json.dumps(dict(phase='original_fla_check',label=label,pid=os.getpid(),operands=str(operands))),flush=True)
 sys.argv=[str(script),'--operands',str(operands),'--sources',str(sources),'--output',str(output)]
 torch.cuda.reset_peak_memory_stats();started=time.time()
 runpy.run_path(str(script),run_name='__main__')
 result=json.loads(output.read_text())
 record['runs'].append(dict(label=label,seconds=time.time()-started,output=str(output),
  max_allocated_bytes=torch.cuda.max_memory_allocated(),process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  passed=all(c['status']=='passed' for c in result['cases'])))
 path.write_text(json.dumps(record,indent=2)+'\n')
record['finished_unix']=time.time()
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
