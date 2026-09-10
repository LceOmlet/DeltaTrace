from pathlib import Path
import json,hashlib,time,subprocess,importlib.util,inspect,traceback
P=Path(__file__).resolve().parent;plan=json.loads((P/'postflight_memory_v4_plan.json').read_bytes());started=time.perf_counter()
target=P/'postflight_memory_v4.json';assert not target.exists()
report={'status':'running','model_calls':0,'attribution_calls':0,'files':[],'GPU_program_calls':0}
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def record(path,expected,kind):
 actual=digest(path);report['files'].append({'path':str(path),'sha256':expected,'actual_sha256':actual,'bytes':path.stat().st_size,'kind':kind,'matches':actual==expected});save();assert actual==expected,str(path)
save()
try:
 assert json.loads((P/'memory_v4_author/results.json').read_bytes())['status']=='complete'
 q=json.loads((P/'memory_v4_rollout/queue.json').read_bytes());assert q['status']=='complete' and len(q['jobs'])==21
 assert all(j['status']!='running' and 'ended' in j for j in q['jobs'])
 for row in plan['files']:record(Path(row['path']),row['sha256'],row['kind'])
 for name,expected in plan['normalized_FT'].items():
  path=Path('/root/flashtrace-vjp-official')/name;actual=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
  report['files'].append({'path':str(path),'sha256':expected,'actual_sha256':actual,'kind':'fixed_FT_normalized_source','matches':actual==expected});save();assert actual==expected,str(path)
 package=Path(importlib.util.find_spec('transformers').origin).parent
 record(package/'models/qwen3/modeling_qwen3.py',plan['native_model_sha256'],'original_native_model_source')
 from torch.utils.checkpoint import create_selective_checkpoint_contexts
 record(Path(inspect.getfile(create_selective_checkpoint_contexts)),plan['native_SAC_sha256'],'original_public_SAC_source')
 env=json.loads((P/'memory_production_v4_release/environment.json').read_bytes())['qwen3']
 record(Path(env['finite_library']),env['finite_library_sha256'],'unchanged_original_finite_FA_binary')
 smi=subprocess.run(['/usr/bin/mx-smi'],capture_output=True,text=True,check=True)
 (P/'final_memory_v4_mx_smi.txt').write_text(smi.stdout);report['GPU_status_text']=smi.stdout
 assert 'no process found' in smi.stdout.lower()
 report['status']='verified'
except BaseException:report['status']='failed';report['error']=traceback.format_exc()
report['elapsed_seconds']=time.perf_counter()-started;save();print(json.dumps({'status':report['status'],'files':len(report['files']),'error':report.get('error')}),flush=True)
