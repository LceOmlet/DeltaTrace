"""Summarize recorded tests without expanding their claims."""
import hashlib,json,statistics
from pathlib import Path

repo=Path(__file__).resolve().parents[3]
base=Path(__file__).resolve().parent
b8=base/'owner-b8-dispatch-20260930';head=base/'owner-entropy-20260930'
read=lambda p:json.loads(p.read_text())
r=read(b8/'result.json')
rows=[json.loads(s) for s in (b8/'physical-resources.jsonl').read_text().splitlines()]
assert r['phase']=='passed' and r['submitted_shape']==[8,32768]
evidence=[b8/n for n in ['result.json','physical-resources.jsonl','launch-config.yaml','job.json']]
dtype=[];performance=[]
for n in ['final-fused-temperature1.json','materialized-temperature09.json']:
    p=head/n;v=read(p);assert all(a['passed'] for c in v['cases'] for a in c['assertions'])
    dtype.append(v);evidence.append(p)
for n in ['fused-head-performance.json','fused-head-performance09.json']:
    p=head/n;v=read(p)
    performance.append(dict(shape=v['shape'],vocab=v['vocab'],temperature=v.get('temperature',1.0),
        warm_mean_seconds={c['mode']:statistics.mean(x['total'] for x in c['warm']) for c in v['cases']}))
    evidence.append(p)
for p in [head/'final-official-head-case2.log',b8/'final-cpu-tests.xml',b8/'formal-status.json',
          b8/'live-TextCraft-complete.json',b8/'live-AppWorld-complete.json',b8/'sql-pause-timeout.json']:
    if p.exists():evidence.append(p)
report=dict(
    scope='Original VERL output-head comparisons and global B8 x 32768 update capacity; not a new whole-network tolerance or a task training-success claim',
    owner=dict(commit='20bd331bdbc9026a5668e11362178e10ab7400c8',
        test='tests/kernels/test_linear_cross_entropy.py',unmodified_test='TestLinearCrossEntropy(2).verify_correctness(): five repetitions passed',
        forward_atol=1e-4,forward_rtol=1e-4,backward_atol=1e-2,backward_rtol=1e-4,
        production_path='Original RayWorkerGroup DP dispatch -> original dp_actor update/loss -> original Qwen torch wrapper -> patched owner FusedLinearForPPO',
        changed='Qwen3.5 text dispatch; FP32 head output storage; preserve BF16 gradient branches and nonunit-temperature storage boundary; compile original chunks',
        unchanged='DT Q/V/A, PPO loss/optimizer/update, FA/FLA, vLLM, task budgets, LoRA rank8 alpha16'),
    dtype_comparisons=dtype,
    capacity=dict(submitted_shape=r['submitted_shape'],microbatch_per_gpu=4,lora_rank=8,lora_alpha=16,
        prompt_tokens=512,response_tokens=32256,synthetic_advantages=True,
        scope='One shared B8 DataProto; original dispatcher splits to two actual B4 workers; full context for every row',
        old_logprob_seconds=r['old_logprob_seconds'],update_seconds=r['update_seconds'],
        grad_norm=r['metrics']['actor/grad_norm'],workers=[dict(rank=w['rank'],calls=w['calls'],
            changed_parameter_tensors=w['changed_parameter_tensors'],
            post_update_sync_sleep_seconds=w['post_update_sync_sleep_seconds']) for w in r['workers']],
        physical_gpu_sample_max_mib={str(i):max(x['physical_gpu_mib'][i] for x in rows if x.get('physical_gpu_mib')) for i in [6,7]},
        process_tree_pss_sample_max_gib=max(sum(j['pss_bytes'] for j in x['jobs']) for x in rows)/2**30,
        resource_note='5-second physical samples, not guaranteed instantaneous peaks; virtual torch allocator and whole-machine RAM are not job physical usage'),
    head_performance=performance,
    performance_scope='Same operands, warm vocabulary-head forward/backward only; not a whole-model slowdown ratio. Full B8 update above includes original offload, checkpoint and synchronization.',
    formal_status=read(b8/'formal-status.json') if (b8/'formal-status.json').exists() else None,
    receipts={str(p.relative_to(repo)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence})
(repo/'experiments/rl/results_actor_b8.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
old=repo/'experiments/rl/results_actor_entropy.json';v=read(old)
v['historical_scope']='Earlier microbatch1 entropy repair; superseded for live status and B4 capacity by results_actor_b8.json'
v['historical_formal_status']=v.pop('formal_status',v.get('historical_formal_status'))
v['current_results']='results_actor_b8.json'
old.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(report['capacity'],indent=2))
