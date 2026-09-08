"""Freeze direct official import prerequisites; no attribution algorithm extraction."""
import base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
tree=json.loads((A/'official_FT_e81_source_tree_20260908.json').read_bytes())
assert tree['sha']=='e81b3be50a48dcfc652fbf1b530069b552736e66' and not tree['truncated']
p={'commit':tree['sha'],'root':'${PRIVATE_MOUNT_PATH}',
   'package_blob_sha1':{x['path']:x['sha'] for x in tree['tree'] if x['path'].startswith('flashtrace/') and x['path'].endswith('.py')},
   'prior_import_failure':'ModuleNotFoundError: wordfreq, from full official package import; no model loaded.',
   'budget':{'dependency_install_attempts':1,'model_loads':0,'model_forwards':0,'quality_queries':0},
   'policy':'Install declared dependency wordfreq==3.1.1 into a private overlay; never stub imports or extract/rewrite author control. Preserve actual resolved dependencies and installer log.'}
s='''import os,sys,hashlib,json,subprocess,time,traceback,zipfile,importlib.metadata as md
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
  raw=(root/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\\0'+raw).hexdigest()
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
'''
import ast;ast.parse(s)
files={'study.py':s.encode()};p['study_sha256']=sha(files['study.py']);files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'official_ft_entry_protocol_20260908.json').write_bytes(files['protocol.json'])
(A/'official_ft_entry_20260908.py').write_bytes(files['study.py'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_official_ft_entry_20260908_v1");d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_official_ft_entry_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_files':len(p['package_blob_sha1']),'budget':p['budget']}))
