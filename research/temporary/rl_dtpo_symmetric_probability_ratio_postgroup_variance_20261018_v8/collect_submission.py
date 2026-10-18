"""Read the six-run receipts and original log collector; no model or training calls."""
import hashlib
import json
from pathlib import Path
import runpy
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert sha(parser)=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(parser),run_name='original_v8_submission_log_reader')
plan=json.loads((AUDIT/'six-combination-manifest.json').read_bytes())
controller=json.loads((AUDIT/'controller-job.json').read_bytes())
p=psutil.Process(controller['pid']);assert p.create_time()==controller['birth']
rows=[]
for segment in plan['segments']:
    folder=Path(segment['prepared']).parent
    prepared=json.loads(Path(segment['prepared']).read_bytes())
    state=json.loads(Path(segment['state']).read_bytes()) if Path(segment['state']).exists() else {'jobs':[]}
    live={r['method']:r for r in state['jobs']}
    for candidate in prepared['jobs']:
        row=dict(task=candidate['task'],method=candidate['method'],version=candidate['version'],
            run=candidate['run'],checkpoints=candidate['checkpoints'],budget=candidate['budget'],
            effective_config=str(folder/(candidate['method']+'-effective-config.yaml')),
            effective_config_SHA256=candidate['effective_config_SHA256'],
            save_freq=candidate['options']['trainer.save_freq'],resume_mode=candidate['options']['trainer.resume_mode'],
            method_mode=candidate['effective_environment'].get('DT_CREDIT_MODE'),
            source_bindings=candidate['source_bindings'],local_patch_commit=candidate['local_patch_commit'],
            actor_gradient_norm=candidate['actor_gradient_norm'],local_shard_key=candidate['local_shard_key'],
            prepared_source=str(folder/(candidate['method']+'-prepared.json')))
        assert sha(row['effective_config'])==row['effective_config_SHA256']
        for path,digest in candidate['source_bindings'].items():assert sha(path)==digest,path
        if candidate['method'] not in live:
            row.update(status='queued_not_started',completed_iterations=0,pid=None,birth=None)
        else:
            actual=live[candidate['method']]
            process=psutil.Process(actual['pid']);assert process.create_time()==actual['birth']
            snapshot=owner['collect'](str(ROOT),[actual['output']]);job,=snapshot['jobs']
            snap_path=AUDIT/(candidate['task'].lower()+'-'+candidate['method']+'-current-source-snapshot.json')
            snap_path.write_text(json.dumps(snapshot,indent=2)+'\n')
            row.update(status='native_training_in_progress' if owner['completed'](job) else 'native_initialization_or_first_iteration_in_progress',
                completed_iterations=owner['completed'](job),pid=process.pid,birth=process.create_time(),
                checkpoint_step=job.get('checkpoint_step'),source_snapshot=str(snap_path),
                last_metric=job['metrics'][-1] if job['metrics'] else None,
                workers=[dict(pid=q.pid,birth=q.create_time(),phase=q.name())
                    for q in process.children(recursive=True) if 'WorkerDict' in q.name()],
                worker_binding_receipts=[str(p) for p in (Path(actual['run'])/'runtime-overrides').glob('actual-worker-imports-*.json')])
        rows.append(row)
result=dict(collected=time.time(),status='six_run_native_queue_submitted',manifest=str(AUDIT/'six-combination-manifest.json'),
    controller={k:controller[k] for k in ('pid','birth','source_commit')},
    method='DTPO symmetric-probability-ratio postgroup-variance v8',credit='tanh(d/2)',credit_variance_normalization='post_group_sum_no_mean_subtraction',
    actor_gradient_norm='preclip_global_native_DTensor_full_tensor',historical_shards_not_converted=True,
    official_GRPO_group_normalization_preserved=True,extra_model_or_environment_samples=0,
    source_commit=plan['source_commit'],verification=str(AUDIT/'CPU-behavior-verification.json'),
    superseded_stop=str(AUDIT/'superseded-v7-stopped.json'),
    original_parser=dict(path=str(parser),SHA256=sha(parser)),jobs=rows)
(AUDIT/'submission-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(collected=result['collected'],controller=result['controller'],
    jobs=[{k:r.get(k) for k in ('task','method','version','status','completed_iterations','pid','birth','workers')} for r in rows])))
