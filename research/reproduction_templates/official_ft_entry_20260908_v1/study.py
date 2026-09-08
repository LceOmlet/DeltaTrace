import os,sys,hashlib,json,subprocess,time,traceback,zipfile,importlib.metadata as md
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
r={'status':'running','protocol':p,'model_loads':0,'model_forwards':0,'quality_queries':0};start=time.perf_counter()
def save():
 q=A/'results.partial';q.write_text(json.dumps(r,indent=2));q.replace(A/'results.json')
def sources():
 root=Path(p['root']);result={}
 assert {str(f.relative_to(root)) for f in (root/'flashtrace').rglob('*.py')}==set(p['package_blob_sha1'])
 for name,digest in p['package_blob_sha1'].items():
  raw=(root/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
  assert blob==digest,name
  result[name]=hashlib.sha256(raw).hexdigest()
 return result
try:
 r['official_sources_before']=sources();r['status']='installing_declared_dependency';save()
 with (A/'install.log').open('w') as f:
  result=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--target',str(A/'deps'),'--report',str(A/'pip_report.json'),'wordfreq==3.1.1'],stdout=f,stderr=subprocess.STDOUT,timeout=180)
 r['dependency_install_exit']=result.returncode;assert result.returncode==0
 sys.path.insert(0,str(A/'deps'));sys.path.insert(0,p['root'])
 import flashtrace,torch
 from flashtrace import FlashTrace
 from flashtrace.improved import LLMIFRAttributionBoth,evaluate_attr_recovery_skip_tokens
 r.update(status='complete_official_package_imported',module=flashtrace.__file__,
  versions={n:md.version(n) for n in ['wordfreq','torch','transformers','flash-linear-attention','fla-core']},
  gpu_free_total=list(torch.cuda.mem_get_info()),device=torch.cuda.get_device_name())
 r['official_sources_after']=sources();assert r['official_sources_before']==r['official_sources_after']
except Exception:r.update(status='failed',error=traceback.format_exc())
finally:
 r['seconds']=time.perf_counter()-start;save()
 with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in ['study.py','protocol.json','results.json','install.log','pip_report.json']:
   if (A/name).exists():z.write(A/name,name)
 print(json.dumps({'status':r['status'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)
