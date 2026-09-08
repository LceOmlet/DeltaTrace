"""Complete the next declared missing dependency, preserving numerical packages."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
prev=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_entry_20260908_v1/results.json').read_bytes())
assert prev['dependency_install_exit']==0 and "No module named 'spacy'" in prev['error']
p=prev['protocol'];p['previous_result_sha256']=sha((A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_entry_20260908_v1/results.json').read_bytes())
p['prior_import_failure']='wordfreq installed; next declared missing dependency is spacy. No model loaded.'
p['policy']='Resolve spacy>=3.8,<3.9 against installed packages, forbid replacement of numerical stack, install only missing resolved packages in a new private overlay; no FT edits, fake imports or AST execution.'
s=(A/'official_ft_entry_20260908.py').read_text()
old=""" with (A/'install.log').open('w') as f:
  result=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--target',str(A/'deps'),'--report',str(A/'pip_report.json'),'wordfreq==3.1.1'],stdout=f,stderr=subprocess.STDOUT,timeout=180)
 r['dependency_install_exit']=result.returncode;assert result.returncode==0
 sys.path.insert(0,str(A/'deps'));sys.path.insert(0,p['root'])"""
new=""" overlay='${ARTIFACT_ROOT}/codex_official_ft_entry_20260908_v1/deps'
 env=dict(os.environ,PYTHONPATH=overlay)
 with (A/'install.log').open('w') as f:
  plan=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--dry-run','--report',str(A/'resolve_report.json'),'spacy>=3.8,<3.9'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
  r['resolver_exit']=plan.returncode;assert plan.returncode==0
  planned=json.loads((A/'resolve_report.json').read_bytes())['install']
  forbidden={'torch','transformers','numpy','fla-core','flash-linear-attention','flash-attn'}
  assert not any(x['metadata']['name'].lower().replace('_','-') in forbidden for x in planned)
  requirements=[x['metadata']['name']+'=='+x['metadata']['version'] for x in planned]
  r['resolved_missing_packages']=requirements;save()
  result=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--target',str(A/'deps'),'--report',str(A/'pip_report.json'),*requirements],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 r['dependency_install_exit']=result.returncode;assert result.returncode==0
 sys.path.insert(0,overlay);sys.path.insert(0,str(A/'deps'));sys.path.insert(0,p['root'])"""
assert s.count(old)==1;s=s.replace(old,new).replace("'install.log','pip_report.json'","'install.log','pip_report.json','resolve_report.json'")
ast.parse(s);files={'study.py':s.encode()};p['study_sha256']=sha(files['study.py']);files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'official_ft_entry_v2_protocol_20260908.json').write_bytes(files['protocol.json']);(A/'official_ft_entry_v2_20260908.py').write_bytes(files['study.py'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode();python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_official_ft_entry_20260908_v2");d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_official_ft_entry_v2_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'budget':p['budget']}))
