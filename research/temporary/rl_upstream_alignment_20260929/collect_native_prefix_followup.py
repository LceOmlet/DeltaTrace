"""Collect completed, original bounded probe artifacts without rerunning them."""
import argparse
import json
import subprocess
from stage_environment_entry import ENTRY, REPO, SSH


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--local',required=True)
    args=parser.parse_args()
    script=r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,json,pathlib,psutil,time
out=pathlib.Path(@OUT@)
def read(name):
 p=out/name
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),value=json.loads(p.read_bytes()))
records={n:read(n) for n in ('prepared.json','job.json','result.json','rank0.json','rank1.json',
 'projection-inputs-rank0.json','projection-inputs-rank1.json',
 'reverse-prefetch-rank0.json','reverse-prefetch-rank1.json',
 'root-capture-inventory-rank0.json','root-capture-inventory-rank1.json') if (out/n).is_file()}
job=records['job.json']['value']
try:
 p=psutil.Process(job['pid']);status=p.status();birth=p.create_time()
except psutil.NoSuchProcess:
 status='exited';birth=None
for rank in (0,1):
 v=records.get(f'rank{rank}.json',{}).get('value',{})
 for case in v.get('first_operator_checks',v.get('first_attention_checks',[])):
  p=pathlib.Path(case['saved_actual_operands']['path'])
  assert hashlib.sha256(p.read_bytes()).hexdigest()==case['saved_actual_operands']['sha256']
summary=[]
for rank in (0,1):
 v=records.get(f'rank{rank}.json',{}).get('value',{});item=dict(rank=rank,phase=v.get('phase'))
 if 'reports' in v:
  item['warm_phases']=[]
  for variant,r in v['reports'].items():
   if not variant.endswith('_warm'):continue
   phases=r['original_runner_phase_seconds']
   item['warm_phases'].append(dict(variant=variant,total_wall_seconds=r['total_wall_seconds'],
    original_attribute_wall_seconds=r.get('original_attribute_wall_seconds'),
    phase_seconds_sum=sum(phases.values()),
    repeated_prefix_seconds=phases.get('native_shared_prefix'),
    cache_materialization_seconds=phases.get('shared_native_prefix_cache'),
    endpoint_root_seconds=phases['native_root_with_CPU_checkpoints'],
    native_replay_seconds=sum(s for k,s in phases.items() if k.startswith('native_replay_')),
    finite_decoder_seconds=sum(s for k,s in phases.items() if k.startswith('finite_decoder_')),
    fa_lse_seconds=sum(s for k,s in phases.items() if k.startswith('public_FA_LSE_')),
    root_capture_observations=r.get('root_tape_observations'),
    phase_counts=r.get('original_runner_phase_counts'),
    peak_torch_allocated_bytes=r.get('peak_torch_allocated_bytes'),
    physical_free_bytes=r.get('physical_free_bytes')))
 if 'first_operator_checks' in v:
  item['operator_checks']=[{k:x[k] for k in ('case','layer','tokens','actual_operand_dtypes','output_error_ratio','state_error_ratio','original_fla_forward_assertions') if k in x} for x in v['first_operator_checks']]
 for k in ('request_index','source_row','prefix','capture_input_shape','native_input_shape','first_observed_unequal_layer','shared_observation_layer'):
  if k in v:item[k]=v[k]
 projection=records.get(f'projection-inputs-rank{rank}.json',{}).get('value')
 if projection is not None:
  item['projection_input_summary']={k:x for k,x in projection.items() if k!='projections'}
 inventory=records.get(f'root-capture-inventory-rank{rank}.json',{}).get('value')
 if inventory is not None:
  item['root_capture_inventory']={k:x for k,x in inventory.items() if k!='rows'}
 prefetch=records.get(f'reverse-prefetch-rank{rank}.json',{}).get('value')
 if prefetch is not None:
  item['reverse_prefetch_summary']={k:x for k,x in prefetch.items() if k not in ('layers','calls','topology')}
  item['reverse_prefetch_summary']['original_layer_count']=len(prefetch.get('topology',[]))
  item['reverse_prefetch_summary']['original_replay_count']=len(prefetch.get('calls',[]))
  item['reverse_prefetch_summary']['settings_restored']=all(x.get('restored') for x in prefetch.get('calls',[]))
  if 'shared_warm' in v.get('reports',{}) and 'prefetch_warm' in v['reports']:
   item['prefetch_only_value_comparison']=[x for x in v.get('raw_value_observations',[])
    if x.get('variant')=='prefetch_warm' and x.get('comparison_reference')=='shared_warm']
 if any(k.startswith('root_tape') for k in v.get('reports',{})):
  item['root_capture_value_comparison']=[x for x in v.get('raw_value_observations',[])
   if x.get('variant','').startswith('root_tape') and x.get('comparison_reference')=='shared_warm']
 summary.append(item)
completed='result.json' in records
# Full arrays remain in their SHA-bound remote owner receipts. This local
# index keeps phase/operand evidence, without duplicating large failure-input
# token lists or the same rank result a second time in result.json.
if completed:
 records['result.json'].pop('value')
for rank in (0,1):
 v=records.get(f'rank{rank}.json',{}).get('value',{})
 # Full parameter topology/buffer pointers stay in the SHA-bound raw files;
 # the rank report duplicates that observation, so retain its receipt only.
 if 'report' in v and 'receipt' in v:
  v.pop('report')
 prefetch=records.get(f'reverse-prefetch-rank{rank}.json')
 if prefetch is not None:
  prefetch.pop('value')
 for r in v.get('reports',{}).values():
  readout=r.get('original_readout_report')
  if readout is not None:
   r['original_readout_report']={k:x for k,x in readout.items()
     if k not in ('minimum_log_ratio_batch','traces')}
log=out/'probe.log'
print(json.dumps(dict(observed_unix=time.time(),remote_root=str(out),formal_deployment=False,
 completed=completed,
 scope='Bounded original B4 operator/phase observations; incomplete runs do not prove numerical acceptance or performance. No new whole-DT threshold or production acceptance.',
 failure_log=None if completed else dict(path=str(log),sha256=hashlib.sha256(log.read_bytes()).hexdigest(),tail=log.read_text(errors='replace')[-12000:]),
 driver=dict(pid=job['pid'],pid_birth=job['pid_birth'],status=status,observed_birth=birth),
 records=records,summary=summary)))
PY
'''.replace('@ENTRY@',ENTRY).replace('@OUT@',repr(args.out))
    result=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
    if result.returncode:
        print(result.stdout.decode('utf8','replace'));raise SystemExit(result.returncode)
    value=json.loads(result.stdout)
    path=REPO/args.local
    if path.exists():
        raise FileExistsError(f'Preserve completed evidence: {path}')
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(local=str(path),driver=value['driver'],summary=value['summary']),indent=2))
