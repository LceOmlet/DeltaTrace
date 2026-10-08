"""Describe frozen development artifacts; no model or credit recomputation.

Negative d is not a sign error. Squared saved coefficients are not gradients.
The held-out rows are never selected for statistics. This only reads original
native d and TextCraft's already-whitened original actor coefficients.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ENTRY, SSH

BODY = r'''
import hashlib,json,resource,time
from pathlib import Path
import torch
torch.set_num_threads(1)
started=time.perf_counter()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def tensor_sha(t):
 t=t.detach().cpu().contiguous()
 h=hashlib.sha256(str((str(t.dtype),tuple(t.shape))).encode())
 h.update(t.numpy().tobytes());return h.hexdigest()
def describe(t):
 t=t.double();finite=t[torch.isfinite(t)]
 out={'tokens':t.numel(),'nonfinite':int((~torch.isfinite(t)).sum())}
 if not finite.numel():return out
 out.update(negative_tokens=int((finite<0).sum()),negative_fraction=float((finite<0).double().mean()),
   mean=float(finite.mean()),mean_absolute=float(finite.abs().mean()),squared_sum=float(finite.square().sum()),
   quantiles=dict(zip(('min','p01','median','p95','p99','max'),torch.quantile(finite,torch.tensor([0.,.01,.5,.95,.99,1.],dtype=torch.float64)).tolist())))
 return out
def mean(items):return sum(items)/len(items) if items else None
out={'scope':'Frozen development only; saved native d and original actor coefficients; no Q/V/A, whitening, model or gradient recomputation',
 'tasks':{},'inputs':[],'limitations':[
 'Nonzero-reward first-action-target DT request set, not all rollouts or historical degradation window.',
 'Previously exposed initial-state groups were forced into development; exposed and unexposed groups reported separately.',
 'Negative d prevalence is not error prevalence: representative native single-deletion references do not exist here.',
 'Squared actor coefficients are descriptive, not gradient contributions, because score-function vectors are absent.',
 'AppWorld has no complete actor/readout artifact after the original OOM; no actor coefficients inferred.',
 'No held-out test values are inspected or used to choose a candidate.'
 ]}
for task,spec in SPECS.items():
 records=spec['records'];groups=spec['groups'];native={};sources={};bindings=[]
 wanted={r['native_occurrences'][0]['file_sha256'] for r in records.values() if r['native_occurrences']}
 for binding in spec['native_files']:
  if binding['sha256'] not in wanted:continue
  p=Path(binding['path']);assert sha(p)==binding['sha256'];bindings.append(binding)
  v=torch.load(p,map_location='cpu',weights_only=False)
  for row in v['rows']:
   uid=str(row['traj_uid'])
   if uid not in records or uid in native:continue
   expected=records[uid]['native_occurrences'][0]
   if expected['file_sha256']!=binding['sha256'] or expected['batch_row']!=row['batch_row']:continue
   suffix=row['suffix_positions'];original=suffix[row['prior'][suffix].bool()]
   positions=row['prompt_length']+row['prior'][suffix].nonzero().flatten()
   d=v['native_signed'][row['batch_row'],positions].clone()
   assert tensor_sha(d)==expected['native_signed_sha256']
   native[uid]=d;sources[uid]=original.clone()
  del v
 rows={uid:{'native_d':describe(d)} for uid,d in native.items()}
 actor_vectors={};actor_duplicates=0
 if task=='textcraft':
  base=Path(spec['native_files'][0]['path']).parent
  expected_actor=['1563ce74f298769893b360398fd1bacf16c376439b1d462e56ef9dc6f905be59','a1970da0bbf462b203314cbf29cc8ecd8c81e526fe1b6ee944978a76bbadd34e']
  for rank in (0,1):
   p=base/f'rank{rank}-pre-update.pt';digest=sha(p);assert digest==expected_actor[rank]
   out['inputs'].append({'path':str(p),'sha256':digest,'bytes':p.stat().st_size})
   v=torch.load(p,map_location='cpu',weights_only=False);ts=v['tensors']
   for i,u in enumerate(v['non_tensors']['traj_uid']):
    uid=str(u)
    if uid not in records:continue
    policy=ts['response_mask'][i].bool()
    raw=ts['dt_token_advantages'][i,policy].clone();white=ts['advantages'][i,policy].clone()
    ids=ts['responses'][i,policy].clone()
    if uid in actor_vectors:
     actor_duplicates+=1
     assert all(torch.equal(a,b) for a,b in zip(actor_vectors[uid],(raw,white,ids)))
     continue
    actor_vectors[uid]=(raw,white,ids)
    artifact=v['non_tensors']['dt_direct_target_artifact'][i]
    retained=torch.as_tensor(artifact['retained_response_positions'])[policy]
    source=torch.isin(retained,sources[uid])
    target=torch.as_tensor(artifact['target_mask'])[retained].bool()
    assert not bool((source & target).any())
    record=rows[uid];record['actor_raw_A']=describe(raw);record['actor_whitened_coefficient']=describe(white)
    for name,vector in [('raw_A',raw),('whitened_coefficient',white)]:
     denom=float(vector.double().square().sum())
     record[name+'_source_squared_share']=float(vector[source].double().square().sum())/denom if denom else None
     record[name+'_target_squared_share']=float(vector[target].double().square().sum())/denom if denom else None
     record[name+'_source_squared_sum']=float(vector[source].double().square().sum())
     record[name+'_target_squared_sum']=float(vector[target].double().square().sum())
   del v
 result={'development_trajectories':len(records),'completed_native':len(native),
   'missing_native_uids':sorted(set(records)-set(native)),'native_inputs':bindings,
   'unique_source_tokens':sum(v.numel() for v in native.values()),
   'token_pooled_native_d':describe(torch.cat(list(native.values()))),
   'actor_unique_trajectories':len(actor_vectors),'actor_duplicate_rows_not_recounted':actor_duplicates,
   'strata':{},'trajectory_statistics':rows}
 for stratum in ('all_development','previously_examined_groups','unexamined_groups'):
  chosen=[g for g in groups if stratum=='all_development' or g['previously_examined']==(stratum=='previously_examined_groups')]
  group_stats=[]
  for g in chosen:
   rs=[rows[u] for u in g['trajectory_uids'] if u in rows]
   metrics={'negative_d_fraction':mean([r['native_d']['negative_fraction'] for r in rs])}
   for key in ('actor_raw_A','actor_whitened_coefficient'):
    valid=[r[key]['negative_fraction'] for r in rs if key in r]
    if valid:metrics[key+'_negative_fraction']=mean(valid)
   for key in ('raw_A_source_squared_share','raw_A_target_squared_share','whitened_coefficient_source_squared_share','whitened_coefficient_target_squared_share'):
    valid=[r[key] for r in rs if r.get(key) is not None]
    if valid:metrics[key]=mean(valid)
   group_stats.append({'initial_state_sha256':g['initial_state_sha256'],'trajectories':len(g['trajectory_uids']),
      'completed_native':len(rs),'metrics':metrics})
  keys={k for g in group_stats for k in g['metrics']}
  result['strata'][stratum]={'groups':len(chosen),'trajectories':sum(len(g['trajectory_uids']) for g in chosen),
    'completed_native':sum(g['completed_native'] for g in group_stats),
    'equal_group_mean':{k:mean([g['metrics'][k] for g in group_stats if g['metrics'].get(k) is not None]) for k in keys},
    'group_statistics':group_stats}
 if actor_vectors:
  result['token_pooled_actor_raw_A']=describe(torch.cat([v[0] for v in actor_vectors.values()]))
  result['token_pooled_actor_whitened_coefficient']=describe(torch.cat([v[1] for v in actor_vectors.values()]))
  for prefix,field in [('raw_A','actor_raw_A'),('whitened_coefficient','actor_whitened_coefficient')]:
   denom=sum(r[field]['squared_sum'] for r in rows.values())
   result[prefix+'_source_squared_share_pooled']=sum(r[prefix+'_source_squared_sum'] for r in rows.values())/denom
   result[prefix+'_target_squared_share_pooled']=sum(r[prefix+'_target_squared_sum'] for r in rows.values())/denom
 out['tasks'][task]=result
out['runtime']={'unix':time.time(),'seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 'cuda_initialized':torch.cuda.is_initialized(),'model_calls':0,'DT_calls':0,'gradient_calls':0,'optimizer_steps':0}
assert not out['runtime']['cuda_initialized']
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
'''

if __name__ == '__main__':
    corpus=json.loads((HERE/'corpus.json').read_bytes())
    manifest=json.loads((HERE/'manifest.json').read_bytes())
    specs={}
    for task,data in manifest['tasks'].items():
        groups=[g for g in data['groups'] if g['split']=='development']
        uids={u for g in groups for u in g['trajectory_uids']}
        specs[task]={'groups':groups,'records':{r['traj_uid']:r for r in corpus['tasks'][task]['records'] if r['traj_uid'] in uids},
                     'native_files':corpus['tasks'][task]['native_files']}
    body='SPECS='+repr(specs)+'\n'+BODY
    script='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    run=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
    (HERE/'development-evidence.stderr.txt').write_bytes(run.stderr)
    run.check_returncode()
    result=json.loads(run.stdout)
    result['local_provenance']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('corpus.json','manifest.json',Path(__file__).name)}
    (HERE/'development-evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'runtime':result['runtime'],'tasks':{t:{k:v for k,v in d.items() if k not in ('trajectory_statistics','native_inputs')} for t,d in result['tasks'].items()}},ensure_ascii=False))
