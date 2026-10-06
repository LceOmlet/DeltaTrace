source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
export CUDA_VISIBLE_DEVICES=""
export MACA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$VENV_PYTHON" - <<'PY'

from pathlib import Path
import hashlib,importlib.util,json,os,resource,time
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('MACA_VISIBLE_DEVICES')==''
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=root/'receipts/sql-paired-real-b4-20261006-v1'
target=out/'sql-original-588-b4-partition-workload.json'
assert out.is_dir() and not target.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=root/'runs/sql-padding-restart-20261001/sql-dt/source.json'
assert sha(source)=='fe54f11730b1d9d06da6364edc7a09a6524fb7f8f02510c59e8e7ca4c99b04ff'
prepared_path=root/'candidates/sql-native-host-cache-20261005-v2/audit/prepared.json'
if not prepared_path.exists():
 prepared_path=root/'candidates/sql-native-host-cache-20261005-v2/prepared.json'
if not prepared_path.exists():
 prepared_path=root/'receipts/sql-native-host-cache-20261005-v2/prepared.json'
assert sha(prepared_path)=='630ed1f67bb7b6dc6a9d0b16a5c91c69773c0c47c5da11e9dfb59795cdf87c92'
prepared=json.loads(prepared_path.read_text())
owner_path=Path(prepared['verl_root'])/'verl/utils/seqlen_balancing.py'
assert sha(owner_path)==prepared['owner_sha256']['verl/utils/seqlen_balancing.py']
entry=Path(prepared['entry'])
dispatch_path=entry/'dt_training_batch.py'
readout_path=entry/'reward_readout.py'
assert sha(dispatch_path)==prepared['entry_sha256']['dt_training_batch.py']=='da9b8a01c3bb9ba00fe3388fd95f93961d33c44bdf0eccf32e7b5846e2ffcae2'
assert sha(readout_path)==prepared['entry_sha256']['reward_readout.py']=='228afbc7a10841d482c3d73def59dfe9ef192c057a97d76dca57369502a10137'
spec=importlib.util.spec_from_file_location('sql_frozen_owner_seqlen_balancing',owner_path)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
import torch
torch.set_num_threads(1)
session=Path('/tmp/ray/session_2026-10-01_05-44-42_236554_552842/logs')
expected_logs={562595:'31630f5ba4b794eba442d2cb978684edba61cf3d789e68528544db86ea5e7ce7',
 564368:'7f77661c933868a363d508b8e07733353e78fcaa0ed1e84c637a35e5da788a18'}
ranks=[]
for rank,pid in enumerate((562595,564368)):
 path=next(session.glob(f'worker-*-{pid}.out'));hasher=hashlib.sha256();offset=0;found=[]
 with path.open('rb') as handle:
  for line_number,raw in enumerate(handle,1):
   start=offset;offset+=len(raw);hasher.update(raw)
   marker=b'[DeltaTrace readout]'
   if marker not in raw:continue
   report=json.loads(raw.split(marker,1)[1].strip())
   if report['finite_trace_calls']!=588:continue
   retained={s['trace']['owner_batch_index']*4+i:s['traj_uid']
    for i,s in enumerate(report['minimum_log_ratio_batch']['samples'])}
   rows=[]
   for index,t in enumerate(report['traces']):
    assert t['owner_batch_index']==index//4
    rows.append(dict(original_rank=rank,original_trace_index=index,
     original_owner_batch_index=t['owner_batch_index'],original_slot=index%4,
     traj_uid=retained.get(index),source_step=t['source_step'],observed_return=t['observed_return'],
     context_tokens=t['context_tokens'],compute_tokens=t['compute_tokens'],query_tokens=t['query_tokens'],
     target_tokens=1,attention_tokens=t['context_tokens']-t['query_tokens']-1))
   assert len(rows)==2352
   assert all(len(s['selected_input_ids'])-s['source_end']-s['trace']['query_tokens']==1
    for s in report['minimum_log_ratio_batch']['samples'])
   assert [x['context_tokens'] for x in rows]==sorted(x['context_tokens'] for x in rows)
   for begin in range(0,len(rows),4):
    group=rows[begin:begin+4]
    assert {x['compute_tokens'] for x in group}=={max(x['context_tokens'] for x in group)}
   # actual_row_lengths was emitted before the readout's context sort. Compare
   # its multiset only; do not pair it with trace indices or invent trajectory IDs.
   assert sorted(report['actual_row_lengths'])==sorted(x['attention_tokens'] for x in rows)
   # Invert the owner's stable rank-local context sort using the original
   # pre-sort actual_row_lengths. Query length is constant here, so equal
   # attention lengths preserve the owner's equal-context tie order.
   assert len({x['query_tokens'] for x in rows})==1
   original_attention=list(report['actual_row_lengths'])
   sorted_source_indices=sorted(range(len(original_attention)),key=original_attention.__getitem__)
   source_rows=[None]*len(rows)
   for trace_row,source_index in zip(rows,sorted_source_indices):
    assert trace_row['attention_tokens']==original_attention[source_index]
    source_rows[source_index]=dict(trace_row,original_rank_source_index=source_index)
   found.append(dict(rank=rank,worker_pid=pid,original_report=dict(line=line_number,
    byte_start=start,byte_end=offset,line_sha256=hashlib.sha256(raw).hexdigest(),
    seconds=report['seconds'],finite_trace_calls=588,event_contrasts=report['event_contrasts']),
    rows=rows,original_attention_tokens_source_order=original_attention,
    original_rank_source_order_trace_indices=[x['original_trace_index'] for x in source_rows],
    source_rows=source_rows,actual_row_lengths_multiset_matches_attention_tokens=True,
    original_log=dict(path=str(path),sha256=None,bytes=None)))
 assert len(found)==1 and hasher.hexdigest()==expected_logs[pid]
 found[0]['original_log'].update(sha256=hasher.hexdigest(),bytes=offset)
 ranks.append(found[0])
flat=ranks[0]['source_rows']+ranks[1]['source_rows']
context=[x['context_tokens'] for x in flat];attention=[x['attention_tokens'] for x in flat]
partitions=owner.get_seqlen_balanced_partitions(attention,k_partitions=2,equal_size=True)
context_partitions=owner.get_seqlen_balanced_partitions(context,k_partitions=2,equal_size=True)
balanced=[[flat[i] for i in part] for part in partitions]
# The frozen readout sorts contexts on each rank. Record that native operation
# explicitly; this is not a new scheduler or a changed minibatch.
balanced=[sorted(rows,key=lambda row:row['context_tokens']) for rows in balanced]
def layout(rows_by_rank):
 groups=[];rank_stats=[]
 for rank,rows in enumerate(rows_by_rank):
  widths=[max(x['context_tokens'] for x in rows[i:i+4]) for i in range(0,len(rows),4)]
  true=sum(x['context_tokens'] for x in rows)
  packed=4*sum(widths)
  rank_stats.append(dict(rank=rank,requests=len(rows),b4_calls=len(widths),
   attention_tokens=sum(x['attention_tokens'] for x in rows),true_context_slots=true,
   unpaired_packed_slots=packed,unpaired_right_padding_slots=packed-true,
   paired_packed_slots=2*packed,paired_right_padding_slots=2*(packed-true),
   b4_max_context_tokens=widths))
 for index in range(588):
  pair=[x['b4_max_context_tokens'][index] for x in rank_stats]
  groups.append(dict(batch_one_based=index+1,rank_b4_max_context_tokens=pair,
   synchronized_max_context_tokens=max(pair),absolute_rank_width_difference=abs(pair[0]-pair[1])))
 return dict(ranks=rank_stats,groups=groups,
  total_unpaired_packed_slots=sum(x['unpaired_packed_slots'] for x in rank_stats),
  total_paired_packed_slots=sum(x['paired_packed_slots'] for x in rank_stats),
  total_unpaired_right_padding_slots=sum(x['unpaired_right_padding_slots'] for x in rank_stats),
  total_paired_right_padding_slots=sum(x['paired_right_padding_slots'] for x in rank_stats),
  synchronized_max_context_length_sum=sum(x['synchronized_max_context_tokens'] for x in groups),
  absolute_rank_width_difference_sum=sum(x['absolute_rank_width_difference'] for x in groups))
before=layout([r['rows'] for r in ranks]);after=layout(balanced)
metrics={key:dict(before=before[key],after=after[key],difference=after[key]-before[key],
 reduction_fraction=(1-after[key]/before[key]) if before[key] else None)
 for key in ['total_paired_packed_slots','total_paired_right_padding_slots',
             'synchronized_max_context_length_sum','absolute_rank_width_difference_sum']}
result=dict(observed_unix=time.time(),scope='Complete original stopped SQL 588 B4/rank length accounting; frozen official partitioner called on actual attention lengths. No wall-time prediction.',
 source=dict(path=str(source),sha256=sha(source)),prepared=dict(path=str(prepared_path),sha256=sha(prepared_path)),
 owner=dict(path=str(owner_path),sha256=sha(owner_path),function='get_seqlen_balanced_partitions',
  module=owner.__name__,arguments=dict(k_partitions=2,equal_size=True)),
 integration_sources={str(p):sha(p) for p in (dispatch_path,readout_path)},
 field_semantics=dict(context_tokens='Original per-request prompt+current action+query+single categorical target length.',
  compute_tokens='Original B4 maximum context_tokens, before cached-prefix slicing.',
  attention_tokens='context_tokens-query_tokens-1; multiset exactly matches original actual_row_lengths on each rank.',
  actual_row_lengths='Original prompt+action lengths emitted before rank-local readout sort, not directly mapped to trace indices.',
  partition_input_order='Original actual_row_lengths DP transport order; original stable context sort is inverted per rank. This preserves the final native padded row instead of globally sorting it into a different position.',
  traj_uid='Only original retained minimum 4 rows/rank have IDs; other row IDs are missing, never reconstructed.',
  query_token_lengths=sorted(set(x['query_tokens'] for x in flat)),
  attention_vs_context_official_partitions_exact_equal=partitions==context_partitions),
 original_rank_rows=ranks,official_partitions=partitions,before=before,official_balanced=after,comparison=metrics,
 operations=dict(model=0,tokenizer=0,ray=0,rollout=0,DT=0,backward=0,optimizer=0,official_partitioner_calls=2),
 cpu=dict(cuda_initialized=torch.cuda.is_initialized(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
  maca_visible_devices=os.environ['MACA_VISIBLE_DEVICES'],max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
 analyzer_source=dict(local_path='D:\\Users\\Administrator\\Documents\\ChatGPT\\DeltaTrace\\research\\temporary\\rl_upstream_alignment_20260929\\sql-paired-real-b4-20261006\\v1\\analyze_sql_partition_workload.py',sha256='d81d72cb1a03339b1cd213b9fda8ea4069a0a1c05f6f6bcb5abcc801b3d4799d'),
 limitations=['Packed slots count original paired input widths before prefix slicing. Full source_start/prefix cuts are not saved for all traces, so actual suffix FLOPs cannot be derived.',
  'The synchronized maximum length sum and width difference describe batch geometry, not measured device waiting seconds or promised speedup.',
  'Native UID tie order is unavailable outside eight retained rows; equal-length identity ties do not affect this length accounting.',
  'SQL prepared balance/scope/host cache fixes were not deployed in these stopped logs.'])
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,
 comparison=metrics,rank_true_slots=[x['true_context_slots'] for x in before['ranks']],
 rank_original_paired_slots=[x['paired_packed_slots'] for x in before['ranks']],
 rank_balanced_paired_slots=[x['paired_packed_slots'] for x in after['ranks']],
 owner_sha256=sha(owner_path),field_semantics=result['field_semantics'],cpu=result['cpu'])))

PY
