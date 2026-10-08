"""Read frozen development d, stratifying its exponential amplification.

Bins describe p_deleted(Y)/p_factual(Y), never a clipping rule or tolerance.
Do not infer existence/nonexistence of a population moment from finite data.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ENTRY,SSH

BODY=r'''
import hashlib,json,math,resource,time
from pathlib import Path
import torch
torch.set_num_threads(1);started=time.perf_counter()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(t):
 t=t.contiguous();h=hashlib.sha256(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.numpy().tobytes());return h.hexdigest()
names=['ratio_le_1','ratio_1_to_2','ratio_2_to_10','ratio_10_to_100','ratio_gt_100']
out={'scope':'Development only. Separate frequencies and conditional magnitudes by exponential amplification; no pooled raw advantage mean or new model call.',
 'bins':{'variable':'c=exp(-d)=p_deleted(Y)/p_factual(Y); A/r=1-c for nonzero r',
         'intervals':['[0,1]','(1,2]','(2,10]','(10,100]','(100,infinity)'],
         'reason':'Positive A/r is bounded by 1. The central negative band is bounded by -1. Larger amplification is separated by decades; the final interval is unbounded. These are diagnostic strata, not official numerical tolerances or training thresholds.'},
 'moment_status':'No claim about population finite/infinite expectation or variance; finite captured values cannot establish these.',
 'tasks':{}}
for task,spec in SPECS.items():
 records=spec['records'];first=set(spec['first_stage_uids']);rows={};tail=[]
 wanted={r['native_occurrences'][0]['file_sha256'] for r in records.values() if r['native_occurrences']}
 for binding in spec['native_files']:
  if binding['sha256'] not in wanted:continue
  assert sha(binding['path'])==binding['sha256']
  value=torch.load(binding['path'],map_location='cpu',weights_only=False)
  for row in value['rows']:
   uid=str(row['traj_uid'])
   if uid not in records or uid in rows:continue
   expected=records[uid]['native_occurrences'][0]
   if expected['file_sha256']!=binding['sha256'] or expected['batch_row']!=row['batch_row']:continue
   pos=row['prompt_length']+row['prior'][row['suffix_positions']].nonzero().flatten()
   d=value['native_signed'][row['batch_row'],pos]
   assert tensor_sha(d)==expected['native_signed_sha256']
   assert bool(torch.isfinite(d).all())
   boundaries=torch.tensor([0.,-math.log(2),-math.log(10),-math.log(100)],dtype=d.dtype)
   masks=[d>=boundaries[0],(d<boundaries[0])&(d>=boundaries[1]),(d<boundaries[1])&(d>=boundaries[2]),
          (d<boundaries[2])&(d>=boundaries[3]),d<boundaries[3]]
   r={'traj_uid':uid,'initial_state_sha256':records[uid]['initial_state_sha256'],
      'first_stage':uid in first,'source_tokens':d.numel(),'strata':{}}
   for name,mask in zip(names,masks):
    selected=d[mask].double();credit=-torch.expm1(-selected)
    r['strata'][name]={'tokens':selected.numel(),'frequency':selected.numel()/d.numel(),
      'conditional_mean_A_over_r':float(credit.mean()) if selected.numel() else None,
      'conditional_mean_abs_A_over_r':float(credit.abs().mean()) if selected.numel() else None,
      'conditional_mean_square_A_over_r':float(credit.square().mean()) if selected.numel() else None,
      'd_min':float(selected.min()) if selected.numel() else None,'d_max':float(selected.max()) if selected.numel() else None}
   rows[uid]=r
   for index in (d<boundaries[1]).nonzero().flatten().tolist():
    tail.append({'traj_uid':uid,'initial_state_sha256':r['initial_state_sha256'],'first_stage':uid in first,
      'native':binding,'batch_row':row['batch_row'],'packed_slot':int(pos[index]),
      'source_index':index,'token_id':int(row['selected'][pos[index]]),'d':float(d[index]),
      'counterfactual_ratio':math.exp(-float(d[index])),'reward':records[uid]['reward'],
      'stratum':names[2+int(float(d[index])<float(boundaries[2]))+int(float(d[index])<float(boundaries[3]))]})
  del value
 group_stats=[]
 for group in spec['groups']:
  available=[rows[u] for u in group['trajectory_uids'] if u in rows]
  g={'initial_state_sha256':group['initial_state_sha256'],'previously_examined':group['previously_examined'],
     'declared_trajectories':len(group['trajectory_uids']),'completed':len(available),'strata':{}}
  for name in names:
   entries=[r['strata'][name] for r in available];present=[e for e in entries if e['tokens']]
   g['strata'][name]={'tokens':sum(e['tokens'] for e in entries),'trajectories_present':len(present),
     'frequency_trajectory_mean':sum(e['frequency'] for e in entries)/len(entries),
     'conditional_abs_A_over_r_trajectory_mean':sum(e['conditional_mean_abs_A_over_r'] for e in present)/len(present) if present else None}
  group_stats.append(g)
 summary={}
 for name in names:
  all_entries=[r['strata'][name] for r in rows.values()];present_groups=[g['strata'][name] for g in group_stats if g['strata'][name]['trajectories_present']]
  summary[name]={'tokens':sum(e['tokens'] for e in all_entries),'trajectories_present':sum(bool(e['tokens']) for e in all_entries),
   'groups_present':len(present_groups),'groups_total':len(group_stats),
   'frequency_equal_group_mean':sum(g['strata'][name]['frequency_trajectory_mean'] for g in group_stats)/len(group_stats),
   'conditional_abs_A_over_r_equal_present_group_mean':sum(g['conditional_abs_A_over_r_trajectory_mean'] for g in present_groups)/len(present_groups) if present_groups else None}
 out['tasks'][task]={'declared':len(records),'completed':len(rows),'missing_uids':sorted(set(records)-set(rows)),
   'summary':summary,'groups':group_stats,'rows':rows,'complete_observed_ratio_gt_2_tail':tail,
   'tail_scope':'Census of ratio>2 positions in this completed frozen development capture set only; not a population moment or error estimate.'}
out['runtime']={'seconds':time.perf_counter()-started,'unix':time.time(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 'cuda_initialized':torch.cuda.is_initialized(),'model_calls':0,'DT_calls':0,'gradient_calls':0,'optimizer_steps':0}
assert not out['runtime']['cuda_initialized']
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
'''

if __name__=='__main__':
    corpus=json.loads((HERE/'corpus.json').read_bytes());manifest=json.loads((HERE/'manifest.json').read_bytes())
    specs={}
    for task,data in manifest['tasks'].items():
        groups=[g for g in data['groups'] if g['split']=='development']
        uids={u for g in groups for u in g['trajectory_uids']}
        specs[task]={'groups':groups,'first_stage_uids':[u for g in groups for u in g['first_stage_uids']],
                     'records':{r['traj_uid']:r for r in corpus['tasks'][task]['records'] if r['traj_uid'] in uids},
                     'native_files':corpus['tasks'][task]['native_files']}
    body='SPECS='+repr(specs)+'\n'+BODY
    shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=60)
    (HERE/'credit-strata.stderr.txt').write_bytes(run.stderr);run.check_returncode()
    data=json.loads(run.stdout)
    data['provenance']={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('corpus.json','manifest.json',Path(__file__).name)}
    (HERE/'credit-strata.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'runtime':data['runtime'],'tasks':{t:{'declared':v['declared'],'completed':v['completed'],'summary':v['summary'],
       'tail_positions':len(v['complete_observed_ratio_gt_2_tail']),'tail_first_stage':sum(p['first_stage'] for p in v['complete_observed_ratio_gt_2_tail'])} for t,v in data['tasks'].items()}},ensure_ascii=False))
