from pathlib import Path
import json,hashlib,time,subprocess,importlib.util,traceback
P=Path(__file__).resolve().parent;plan=json.loads((P/'postflight_plan.json').read_bytes());started=time.perf_counter()
report={'status':'running','model_calls':0,'attribution_calls':0,'files':[],'GPU_program_calls':0}
def save():(P/'postflight.json').write_text(json.dumps(report,indent=2)+'\n')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
save()
try:
 assert json.loads((P/'author/results.json').read_bytes())['status']=='complete'
 for row in plan['files']:
  path=Path(row['path']);actual=digest(path);record=dict(row,actual_sha256=actual,bytes=path.stat().st_size,matches=actual==row['sha256'])
  report['files'].append(record);save();assert record['matches'],str(path)
 for name,expected in plan['normalized_FT'].items():
  path=Path('/root/flashtrace-vjp-official')/name;actual=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
  report['files'].append({'path':str(path),'sha256':expected,'actual_sha256':actual,'kind':'fixed_FT_normalized_source','matches':actual==expected});assert actual==expected,str(path)
 package=Path(importlib.util.find_spec('transformers').origin).parent
 native=package/'models/qwen3/modeling_qwen3.py';actual=digest(native)
 report['files'].append({'path':str(native),'sha256':plan['native_model_sha256'],'actual_sha256':actual,'kind':'native_Qwen3_model_source','matches':actual==plan['native_model_sha256']});assert actual==plan['native_model_sha256']
 env=json.loads((P/'compact_production_release/environment.json').read_bytes())['qwen3'];native=Path(env['finite_library']);actual=digest(native)
 report['files'].append({'path':str(native),'sha256':env['finite_library_sha256'],'actual_sha256':actual,'kind':'unchanged_finite_FA_library','matches':actual==env['finite_library_sha256']});assert actual==env['finite_library_sha256']
 report['native_FA_current_identity']=[]
 for module in ['flash_attn','flash_attn_2_cuda']:
  spec=importlib.util.find_spec(module);path=Path(spec.origin)
  if module=='flash_attn':path=path.parent/'flash_attn_interface.py'
  report['native_FA_current_identity'].append({'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size,'scope':'current file identity recorded; native callable identity and actual operands checked in every comparison'})
 smi=subprocess.run(['/usr/bin/mx-smi'],capture_output=True,text=True,check=True);(P/'final_mx_smi.txt').write_text(smi.stdout)
 report['GPU_status_text']=smi.stdout;report['status']='verified'
except BaseException:report['status']='failed';report['error']=traceback.format_exc()
report['elapsed_seconds']=time.perf_counter()-started;save();print(json.dumps({'status':report['status'],'files':len(report['files']),'error':report.get('error')}),flush=True)
