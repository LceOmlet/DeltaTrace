"""Verify actual official execution and preserved failures; never fill missing scores."""
import ast,hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
names=['codex_official_ft_entry_20260908_v1','codex_official_ft_entry_20260908_v2',
       'codex_official_ft_pristine_20260908_v1','codex_official_ft_exp2_default_20260908_v1']
jobs=[]
for name in names:
 d=A/'snapshot/tmp'/name
 z=zipfile.ZipFile(d/'review_bundle.zip');assert z.testzip() is None
 for item in z.namelist():
  assert item in ['study.py','protocol.json','results.json','install.log','pip_report.json','resolve_report.json','official_outputs.npz']
  raw=z.read(item);path=d/item
  if path.exists():assert path.read_bytes()==raw,(name,item)
  else:path.write_bytes(raw)
 p=json.loads((d/'protocol.json').read_bytes());r=json.loads((d/'results.json').read_bytes())
 assert sha((d/'study.py').read_bytes())==p['study_sha256'];ast.parse((d/'study.py').read_bytes())
 receipt=json.loads((d/'terminal_receipt.json').read_bytes());assert not receipt['proc_exists']
 assert receipt['results_sha256']==sha((d/'results.json').read_bytes())
 assert receipt['bundle_sha256']==sha((d/'review_bundle.zip').read_bytes())
 jobs.append({'directory':name,'status':r['status'],'results_sha256':receipt['results_sha256'],
              'bundle_sha256':receipt['bundle_sha256'],'seconds':r.get('seconds',r.get('seconds_before_bundle'))})
 if name==names[-1]:actual=r;last=d
assert json.loads((A/'snapshot/tmp'/names[0]/'results.json').read_bytes())['dependency_install_exit']==0
assert json.loads((A/'snapshot/tmp'/names[1]/'results.json').read_bytes())['status']=='complete_official_package_imported'
previous=json.loads((A/'snapshot/tmp'/names[2]/'results.json').read_bytes())
assert previous['root_forwards_completed']==0 and previous['actual_input']['shape']==[1,598]
contract=json.loads((A/'snapshot/tmp/qwen35_official_input_contract_20260908.json').read_bytes())
assert contract['current_keep_equals_historical'] and contract['current_gold_equals_historical']
up=A/'snapshot${PRIVATE_MOUNT_PATH}'
verified={}
tree=json.loads((A/'official_FT_e81_source_tree_20260908.json').read_bytes())
for name,blob in actual['protocol']['package_blob_sha1'].items():
 raw=(up/name).read_bytes()
 assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==blob,name
 assert sha(raw)==actual['official_sources_before'][name]
 verified[name]=sha(raw)
needle=[]
if actual['status']=='unchanged_official_FT_NI0_trace_completed':
 assert actual['official_sources_before']==actual['official_sources_after']==verified
 assert actual['actual_input']['sha256']==contract['modes']['current_exp2_default']['input_ids_sha256']
 assert actual['root_forwards_completed']==1
 arrays=np.load(last/'official_outputs.npz')
 assert sha((last/'official_outputs.npz').read_bytes())==actual['artifacts']['official_outputs.npz']
 assert arrays['scores'].shape==(4,340) and np.isfinite(arrays['scores']).all()
 expected=arrays['base'].copy();prefix=[expected.copy()]
 for v in arrays['per_hop']:
  expected=expected+v;prefix.append(expected.copy())
 assert np.array_equal(np.stack(prefix),arrays['cumulative_projected'])
 assert np.array_equal(expected,arrays['sum'])
 keep=contract['current_keep'];gold=set(contract['current_gold']) & set(keep)
 assert len(keep)==310 and len(gold)==40
 for item in actual['needle']:
  selected=item['selected'];score=arrays['scores'][item['hop']]
  assert len(set(selected))==31 and set(selected)<=set(keep)
  hits=set(selected)&gold;assert sorted(hits)==item['hits'] and len(hits)/len(gold)==item['recovery']
  cutoff=min(score[selected]);assert all(score[j]<=cutoff for j in set(keep)-set(selected))
  needle.append({**item,'cutoff_ties':int(np.sum(score[keep]==cutoff))})
result={'status':actual['status'],'jobs':jobs,'all_jobs_terminal':True,'official_commit':actual['protocol']['commit'],
 'official_package_sha256':verified,'complete_official_package_verified':len(verified)==16,
 'input_contract':{k:v for k,v in contract.items() if k not in ['current_keep','current_gold']},
 'original_needle':needle,'actual_calls':actual.get('calls',{}),
 'actual_root_completed':actual['root_forwards_completed'],'error':actual.get('error'),
 'peak_allocated':actual.get('peak_allocated'),'peak_reserved':actual.get('peak_reserved'),
 'timings':actual.get('timings'),'FT_source_modified':False,
 'comparison_limit':'Original FT uses current official raw588/eager/B1. Historical DT605/FA results are not a paired comparison. No original FT quality failure or DT victory inferred from custom FT derivatives or driver-input failures.',
 'resource_accounting':{'model_loads_including_driver_stop':previous['model_loads']+actual['model_loads'],
  'completed_root_forwards':actual['root_forwards_completed'],'new_generation_calls':0,
  'original_recovery_calls':actual['original_recovery_calls'],'dependency_installs':2},
 'next':'Keep complete original FT untouched; use its verified input contract for future same-input DT comparison. Do not resume withdrawn FT content/controller modifications.'}
(A/'official_ft_pristine_summary_20260908.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:result[k] for k in ['status','all_jobs_terminal','original_needle','actual_calls','error','resource_accounting']},ensure_ascii=False))
