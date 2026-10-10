"""Audit the real frozen probability sample through its existing statistics owner.

No new sample, counterfactual, model, gradient, or training computation is made.
This is not a numerical tolerance test or evidence about the current policy.
"""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'experiments/rl/current_runtime.json').exists())
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import hashlib,inspect,json,math,os,psutil,resource,sys,time
from pathlib import Path
import scipy
from scipy import stats
started=time.perf_counter()
root=Path(ROOT)/'receipts/credit-tail-probability-sample-20261009-v1'
analysis=root/'analysis-c9e78c9b'
def artifact(p):
 blob=p.read_bytes()
 return dict(path=str(p),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
sample_path=root/'sample.json'
assert artifact(sample_path)['sha256']=='60eed5f19c37320056d53186f0d802051b73ef55085846fa700a8d8f1264c1d4'
assert artifact(analysis/'tail_probability_statistics.py')['sha256']=='6eb3411743b2765327b616e61fc1e92f22898a851ff27263ea31a51c04a762bd'
sys.path.insert(0,str(analysis))
from tail_probability_statistics import summarize_task
sample=json.loads(sample_path.read_bytes())
results={};checked=[artifact(sample_path),artifact(analysis/'tail_probability_statistics.py')]
for task,frame in sample['tasks'].items():
 path=analysis/(task+'.json');saved=json.loads(path.read_bytes());data=saved['tasks'][task]
 checked.append(artifact(path))
 for binding in saved['sources']:
  actual=artifact(Path(binding['path']))
  assert actual['bytes']==binding['bytes'] and actual['sha256']==binding['sha256']
  checked.append(actual)
 assert scipy.__version__==saved['sampling_owner']['version']
 assert artifact(Path(inspect.getsourcefile(stats.hypergeom.__class__)))['sha256']==saved['sampling_owner']['source']['sha256']
 queries={(entry['traj_uid'],q['packed_slot']):(entry,q) for entry in frame['entries'] for q in entry['queries']}
 assert len(queries)==frame['sampled_sources']==len(data['observations'])
 assert sum(s['population'] for s in frame['strata'])==frame['completed_frame_sources']
 assert sum(e['source_tokens'] for e in frame['entries'])==frame['completed_frame_sources']
 for stratum in frame['strata']:
  name,N,n=stratum['name'],stratum['population'],stratum['sample']
  assert sum(e['source_strata_populations'][name] for e in frame['entries'])==N
  assert sum(q['prediction_stratum']==name for e,q in queries.values())==n
  if name not in ('ratio_le_1','ratio_1_to_2'):assert n==N
 for point in data['observations']:
  entry,q=queries[(point['traj_uid'],point['packed_slot'])]
  for key in ('token_id','saved_d','prediction_stratum','inclusion_probability','token_total_weight','state_trajectory_source_weight'):
   assert point[key]==q[key]
  assert point['initial_state_sha256']==entry['initial_state_sha256']
  pi=q['stratum_sample']/q['stratum_population']
  assert q['inclusion_probability']==pi and q['token_total_weight']==1/pi
  assert q['state_trajectory_source_weight']==1/(frame['initial_states']*frame['state_completed_trajectories'][entry['initial_state_sha256']]*entry['source_tokens']*pi)
  observed=[score['d'] for score in point['scores'].values()]
  if point['prior_native_d_interval'] is not None:observed.extend(point['prior_native_d_interval'])
  assert point['native_d_interval']==[min(observed),max(observed)]
 tables={str(c):summarize_task(frame,data['observations'],c) for c in (2,10,100)}
 assert tables==data['confusion_by_cumulative_threshold']
 states={s:{str(c):summarize_task(frame,data['observations'],c,domain_state=s) for c in (2,10,100)} for s in data['by_initial_state']}
 assert states==data['by_initial_state']
 results[task]=dict(
  frame={k:frame[k] for k in ('declared_trajectories','completed_trajectories','missing_uids','initial_states','completed_frame_sources','sampled_sources','sampled_states','strata')},
  population_counts_and_original_weights_verified=True,
  complete_observation_identities_verified=True,
  all_saved_reference_views_preserved=True,
  original_tables_recomputed_exactly=True,
  original_state_tables_recomputed_exactly=True,
  cumulative_thresholds=tables,by_initial_state=states)
print(json.dumps(dict(unix=time.time(),scope='Actual frozen frame, original statistics and saved reference views; no newly sampled position or model execution.',
 tasks=results,sources=list({p['path']:p for p in checked}.values()),
 scipy_version=scipy.__version__,scipy_hypergeom_source=artifact(Path(inspect.getsourcefile(stats.hypergeom.__class__))),
 elapsed_seconds=time.perf_counter()-started,cpu_auditor_PSS_bytes=psutil.Process().memory_full_info().pss,
 cpu_auditor_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_imported='torch' in sys.modules,
 operations=dict(model=0,DT=0,native_forward=0,backward=0,optimizer=0,new_sample=0,production_changes=0),
 interpretation=[
  'Precision ranges reflect the saved numerical reference envelope, not a confidence interval.',
  'Recall intervals use the original finite-population hypergeometric inversion; statistical coverage is conditional on reference labels being bounded by that observed envelope.',
  'Each interval is pointwise for one threshold/domain; it is not simultaneous across all states and thresholds.',
  'These base-model development captures are not current-policy trajectories or held-out evidence.',
  'Single-token deletion agreement does not replace the original author cumulative deletion, RISE or MAS metrics.',
  'No implication of a current training-gradient bias or a cause of the iteration14 NaN is established.'])))
'''.replace('ROOT', repr(transport.ROOT), 1)
command = 'source ' + transport.ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
r = subprocess.run(transport.SSH + ['bash', '-s'], input=command.encode(), capture_output=True, timeout=55)
if r.returncode:
    raise RuntimeError(r.stderr.decode(errors='replace'))
result = json.loads(r.stdout)
out = REPO / 'experiments/rl/results_frozen_tail_frame_audit_20261010.json'
assert not out.exists(), 'Preserve a completed audit; inspect it rather than rerunning'
out.write_bytes(r.stdout)
print(json.dumps(dict(saved=str(out),seconds=result['elapsed_seconds'],
    tasks={t:dict(positions=d['frame']['sampled_sources'],state_tables=len(d['by_initial_state']),
        recomputed=d['original_tables_recomputed_exactly']) for t,d in result['tasks'].items()},
    operations=result['operations']), ensure_ascii=False))
