"""Read frozen development d, stratifying its exponential amplification.

Bins describe p_deleted(Y)/p_factual(Y), never a clipping rule or tolerance.
Do not infer existence/nonexistence of a population moment from finite data.
"""
import argparse
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
 records=spec['records'];first=set(spec['first_stage_uids']);rows={};tail=[];bound_points=[]
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
   if PROBABILITY_BOUNDS:
    sample=value['detail'].get('per_sample',[])
    factual=sample[row['batch_row']].get('factual_target_logp') if row['batch_row']<len(sample) else None
    available=factual is not None and math.isfinite(factual)
    r['probability_bound']={'available':available,'factual_target_logp':factual}
    implied=float(factual)-d.double() if available else None
   for name,mask in zip(names,masks):
    selected=d[mask].double();credit=-torch.expm1(-selected)
    r['strata'][name]={'tokens':selected.numel(),'frequency':selected.numel()/d.numel(),
      'conditional_mean_A_over_r':float(credit.mean()) if selected.numel() else None,
      'conditional_mean_abs_A_over_r':float(credit.abs().mean()) if selected.numel() else None,
      'conditional_mean_square_A_over_r':float(credit.square().mean()) if selected.numel() else None,
      'd_min':float(selected.min()) if selected.numel() else None,'d_max':float(selected.max()) if selected.numel() else None}
    if PROBABILITY_BOUNDS:
     violations=mask&(implied>0) if available else None
     r['strata'][name]['probability_bound_violations']=int(violations.sum()) if available else None
     if available:
      for index in violations.nonzero().flatten().tolist():
       bound_points.append({'traj_uid':uid,'initial_state_sha256':r['initial_state_sha256'],
        'packed_slot':int(pos[index]),'source_index':index,'token_id':int(row['selected'][pos[index]]),
        'd':float(d[index]),'factual_target_logp':float(factual),
        'implied_deleted_logp':float(implied[index]),'stratum':name,'native':binding,
        'previously_examined':spec['state_exposure'][r['initial_state_sha256']]})
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
 if PROBABILITY_BOUNDS:
  out['tasks'][task]['probability_bound_violation_points']=bound_points
out['runtime']={'seconds':time.perf_counter()-started,'unix':time.time(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 'cuda_initialized':torch.cuda.is_initialized(),'model_calls':0,'DT_calls':0,'gradient_calls':0,'optimizer_steps':0}
assert not out['runtime']['cuda_initialized']
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probability-bounds',action='store_true',
                        help='Read-only implied-deletion probability bound on the same development captures; no training behavior')
    args=parser.parse_args()
    corpus=json.loads((HERE/'corpus.json').read_bytes());manifest=json.loads((HERE/'manifest.json').read_bytes())
    specs={}
    for task,data in manifest['tasks'].items():
        groups=[g for g in data['groups'] if g['split']=='development']
        uids={u for g in groups for u in g['trajectory_uids']}
        specs[task]={'groups':groups,'first_stage_uids':[u for g in groups for u in g['first_stage_uids']],
                     'state_exposure':{g['initial_state_sha256']:g['previously_examined'] for g in groups},
                     'records':{r['traj_uid']:r for r in corpus['tasks'][task]['records'] if r['traj_uid'] in uids},
                     'native_files':corpus['tasks'][task]['native_files']}
    body='SPECS='+repr(specs)+'\nPROBABILITY_BOUNDS='+repr(args.probability_bounds)+'\n'+BODY
    shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=60)
    stem='credit-probability-bounds' if args.probability_bounds else 'credit-strata'
    (HERE/(stem+'.stderr.txt')).write_bytes(run.stderr);run.check_returncode()
    data=json.loads(run.stdout)
    if args.probability_bounds:
        from summarize_author_collection import quantiles
        for task,taskdata in data['tasks'].items():
            groups=[g for g in manifest['tasks'][task]['groups'] if g['split']=='development']
            rows=taskdata['rows'];points=taskdata['probability_bound_violation_points']
            cells={}
            names=list(taskdata['summary'])
            for name in names:
                selected=[p for p in points if p['stratum']==name]
                group_rows=[]
                for group in groups:
                    available=[rows[u] for u in group['trajectory_uids'] if u in rows
                               and rows[u]['probability_bound']['available']]
                    present=[r for r in available if r['strata'][name]['tokens']]
                    gs=[p for p in selected if p['initial_state_sha256']==group['initial_state_sha256']]
                    group_rows.append(dict(initial_state_sha256=group['initial_state_sha256'],
                        previously_examined=group['previously_examined'],
                        available_trajectories=len(available),trajectories_in_stratum=len(present),
                        violations=len(gs),positive_logp_excess=quantiles([p['implied_deleted_logp'] for p in gs]),
                        conditional_frequency_trajectory_mean=sum(
                            r['strata'][name]['probability_bound_violations']/r['strata'][name]['tokens']
                            for r in present)/len(present) if present else None))
                present_groups=[g for g in group_rows if g['trajectories_in_stratum']]
                cells[name]=dict(violations=len(selected),
                    source_tokens=taskdata['summary'][name]['tokens'],
                    trajectories=len({p['traj_uid'] for p in selected}),
                    states=len({p['initial_state_sha256'] for p in selected}),
                    positive_logp_excess=quantiles([p['implied_deleted_logp'] for p in selected]),
                    conditional_frequency_equal_present_group_mean=sum(
                        g['conditional_frequency_trajectory_mean'] for g in present_groups)/len(present_groups)
                        if present_groups else None,groups=group_rows)
            taskdata['probability_bound_summary']=dict(strata=cells,
                unavailable_factual_scores=[u for u,r in rows.items() if not r['probability_bound']['available']],
                total_violations=len(points),violation_states=len({p['initial_state_sha256'] for p in points}))
        data['scope']='Frozen development saved captures only. Necessary probability bound; not single-delete ground truth, a clipping rule, training gate, component-cause test or new acceptance tolerance.'
        data['probability_bound_definition']='For the same actual target Y and captured factual logp l_F, a valid deletion logp is l_deleted=l_F-d<=0. Positive l_F-d is an impossible implied probability, regardless of whether the advantage is negative. No credit is corrected. Passing this bound does not establish accuracy.'
    data['provenance']={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('corpus.json','manifest.json',Path(__file__).name)}
    (HERE/(stem+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'runtime':data['runtime'],'tasks':{t:{'declared':v['declared'],'completed':v['completed'],'summary':v['summary'],
       'tail_positions':len(v['complete_observed_ratio_gt_2_tail']),'tail_first_stage':sum(p['first_stage'] for p in v['complete_observed_ratio_gt_2_tail'])} for t,v in data['tasks'].items()}},ensure_ascii=False))
