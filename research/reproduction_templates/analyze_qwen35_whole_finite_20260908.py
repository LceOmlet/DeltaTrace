"""CPU independent full-chain accounting from immutable saved output vectors."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_whole_finite_20260908_v1';sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        f=(D/name).resolve();assert f.is_relative_to(D.resolve());f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(name))
raw=(D/'results.json').read_bytes();r=json.loads(raw)
assert r['status']=='one_author_answer_32_layer_finite_propagation_executed'
for name,digest in r['protocol']['files_sha256'].items():
    source=(D/name).read_bytes();assert sha(source)==digest;ast.parse(source)
for a in r['artifacts']:
    if a['download']:assert sha((D/a['file']).read_bytes())==a['sha256'] and (D/a['file']).stat().st_size==a['bytes']
def metric(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'relative_L2':float(np.linalg.norm(b-a))/na if na else None,'max_abs':float(np.abs(b-a).max()),'cosine':float(a@b)/(na*nb) if na*nb else None}
head=dict(np.load(D/'answer_seed_review.npz',allow_pickle=False));whole=dict(np.load(D/'whole_input_review.npz',allow_pickle=False))
for group in [head,whole]:assert all(np.isfinite(v).all() for v in group.values())
def allocation(group):return (group['finite'].astype(np.float64)*(group['input1'].astype(np.float64)-group['input0'].astype(np.float64))).sum(-1)
head_sum=allocation(head).sum(1);signed=allocation(whole);sums=signed.sum(1)
assert np.max(np.abs(head_sum-r['head_and_norm_finite_effect']))<1e-7
assert np.max(np.abs(sums-r['signed_sums']))<1e-7
limit=metric(head['native_gradient'],head['equal']);assert abs(limit['relative_L2']-r['head_norm_equal_vs_native']['relative_L2'])<1e-10
assert np.count_nonzero(whole['finite'][1,368:])==0
root=np.asarray(r['original_root_answer_logprobs'],dtype=np.float64).reshape(-1,2)
replay=np.asarray(r['replayed_answer_logprobs'],dtype=np.float64).reshape(-1,2)
cuts=[0,38,59]
sum_samples=lambda values:np.array([values[cuts[b]:cuts[b+1]].sum() for b in range(2)])
root_delta=sum_samples(root[:,1]-root[:,0]);replay_delta=sum_samples(replay[:,1]-replay[:,0])
assert np.max(np.abs(root_delta-r['original_root_answer_delta']))<1e-10
assert np.max(np.abs(replay_delta-r['head_replay_delta']))<1e-10
rows=[];last=head_sum;local=[];discontinuity=[]
for i in reversed(range(32)):
    row=r['layers'][str(i)];before=np.array(row['root_output_effect']);actual=np.array(row['replay_output_effect']);after=np.array(row['input_effect'])
    assert np.max(np.abs(before-last))<1e-7
    local.append(after-actual);discontinuity.append(actual-before)
    rows.append({'layer':i,'type':row['block_type'],'root_output_effect':before.tolist(),'replay_output_effect':actual.tolist(),
        'input_effect':after.tolist(),'local_finite_residual':(after-actual).tolist(),
        'replay_discontinuity':(actual-before).tolist(),'local_relative_residual':np.divide(after-actual,np.abs(actual),out=np.full_like(actual,np.nan),where=actual!=0).tolist(),
        'native_replay_vs_root':row['replay_vs_root'],'padding_max_abs':row['padding_max_abs']})
    last=after
assert np.max(np.abs(last-sums))<1e-7
local_sum=np.sum(local,axis=0);discontinuity_sum=np.sum(discontinuity,axis=0)
head_rounding=head_sum-replay_delta;head_replay=replay_delta-root_delta
ledger=head_replay+head_rounding+local_sum+discontinuity_sum
assert np.max(np.abs(ledger-(sums-root_delta)))<1e-7
costs={}
for call in r['calls']:
    kind=call['kind']
    key=('weight_load' if kind.startswith('load_original_') else 'saved_capture_load' if kind.startswith('load_saved_') else
        'finite_FA_decoder' if kind.startswith('finite_decoder') and int(kind.removeprefix('finite_decoder'))%4==3 else
        'finite_GDN_decoder' if kind.startswith('finite_decoder') else 'auxiliary_FA' if 'public_FA_auxiliary' in kind else
        'original_decoder_replay' if kind.startswith('original_decoder') else 'answer_and_norm')
    group=costs.setdefault(key,{'calls':0,'seconds':0.0,'largest_incremental_peak_bytes':0,'largest_absolute_peak_bytes':0})
    group['calls']+=1;group['seconds']+=call['seconds'];group['largest_incremental_peak_bytes']=max(group['largest_incremental_peak_bytes'],call['peak_bytes']-call['before_bytes'])
    group['largest_absolute_peak_bytes']=max(group['largest_absolute_peak_bytes'],call['peak_bytes'])
summary={'status':'author_answer_32_layer_finite_pass_executed_and_accounted_quality_pending',
    'raw_sha256':sha(raw),'protocol_sha256':sha((D/'protocol.json').read_bytes()),'source_sha256':r['protocol']['files_sha256'],
    'target_selection':r['target_selection'],'target_scope_change':r['protocol']['target_scope_change'],
    'head_logprobs_vs_root':metric(root,replay),'head_norm_equal_vs_native':limit,
    'root_answer_delta':root_delta.tolist(),'signed_sums':sums.tolist(),'unassigned_root_effect':(root_delta-sums).tolist(),
    'relative_total_residual':((sums-root_delta)/np.abs(root_delta)).tolist(),
    'positive_mass':np.maximum(signed,0).sum(1).tolist(),'negative_mass':np.minimum(signed,0).sum(1).tolist(),
    'ledger':{'head_replay':head_replay.tolist(),'head_norm_finite_and_rounding':head_rounding.tolist(),
        'decoder_local_finite_sum':local_sum.tolist(),'decoder_replay_discontinuity_sum':discontinuity_sum.tolist(),
        'ledger_sum':ledger.tolist(),'reconciles':True,'per_layer':rows},
    'costs':costs,'calls':r['calls'],'job_seconds':r['job_seconds'],'budget':r['protocol']['budget'],
    'compiler_benchmark_observations':r['compiler_benchmark_observations'],'compiler_counters':r['compiler_counters'],
    'artifacts':r['artifacts'],'versions':r['versions'],'source_tree':r['sources_before'],
    'limits':['One engineering pass on two historical official trajectories, no quality or independent-confirmation result.',
        'Target scope changed from old8B full-response seeds to explicit author answer sink. Do not merge quality numbers.',
        'Original root was saved in an earlier paid run; current job uses30 original decoder replays and two saved original captures.',
        'Cost includes new compiler graphs, diagnostic/weight IO and replay; it is not matched warm FT cost or a throughput claim.',
        'A reconciled finite residual ledger is accounting, not proof of useful token ranking or conditional deletion signs.']}
assert r['sources_before']==r['sources_after']
(A/'qwen35_whole_finite_summary_20260908.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
print(json.dumps({k:summary[k] for k in ['status','root_answer_delta','signed_sums','relative_total_residual','head_norm_equal_vs_native','costs']}))
