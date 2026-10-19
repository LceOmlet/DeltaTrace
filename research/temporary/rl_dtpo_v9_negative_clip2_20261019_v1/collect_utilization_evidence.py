"""Compose saved native phase, stack and first-iteration observations only."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python

report = json.loads(run_python(r'''
import collections, hashlib, json, time
from pathlib import Path
import psutil
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
segment=R/'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v4'
run=R/'runs/textcraft-dtpo-v9-negative-clip2-8gpu-200-20261019-v4'
bindings=run/'runtime-overrides/actual-worker-imports-after-first-update-v1.json'
metadata=json.loads(bindings.read_bytes())
assert len(metadata['results'])==8
assert all(x['actor_optimizer_entries']==496 and x['actor_optimizer_step_values']==[4.0]
    and x['actor_scheduler_epoch']==1 and x['DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP']=='2'
    and x['DT_NATIVE_HOST_CACHE_BATCH']=='1' for x in metadata['results'])

profiles=[]
for directory,actual_phase in [
    ('bounded-generation-phase-1792395393','old_forward_boundary'),
    ('bounded-generation-phase-1792396154','generation'),
    ('bounded-generation-phase-1792396588','DT')]:
    path=A/directory/'observed.json'
    raw=json.loads(path.read_bytes())
    summary=[]
    for job in raw['jobs']:
        for worker in job['workers']:
            native=Path(worker['raw'])
            rows=[]
            for number,line in enumerate(native.read_text().splitlines(),1):
                if not line.strip() or 'MainThread' not in line:continue
                stack,count=line.rsplit(' ',1)
                rows.append(dict(stack=stack,samples=int(count),source_line=number))
            rows.sort(key=lambda item:item['samples'],reverse=True)
            summary.append(dict(pid=worker['pid'],process_title=worker['process_title'],
                native_stack_source=str(native),native_stack_SHA256=hashlib.sha256(native.read_bytes()).hexdigest(),
                main_thread_samples=sum(item['samples'] for item in rows),top_main_thread_stacks=rows[:12]))
    profiles.append(dict(actual_phase=actual_phase,path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         workers=summary,physical=raw['physical']))

phase_path=A/'native-utilization-phase-summary.json'
phases=json.loads(phase_path.read_bytes())
first_path=A/'first-complete-iteration-efficiency-audit.json'
first=json.loads(first_path.read_bytes())
generation_path=R/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v3/bounded-native-generation-observation.json'
generation=json.loads(generation_path.read_bytes())
generation_reference=dict(path=str(generation_path),SHA256=hashlib.sha256(generation_path.read_bytes()).hexdigest(),
    status='Historical bounded generation before withdrawn v3 DT allocation failure; not current v4 progress',
    values={key:generation[key] for key in ('run','observed_native_calls','sampled_requests','generated_tokens',
        'generated_length_median','generated_length_max','outputs_with_512_tokens','outputs_with_512_token_fraction',
        'native_cached_prompt_fraction','mean_inter_call_gap_seconds','mean_native_engine_call_coverage')})
observer=segment/'existing-DT-metadata-observer-completed.json'
assert json.loads(observer.read_bytes())['source_SHA256']=='20c0de1e0d2b7f89082e372fbdac292cab0bea41866612f8f0555d3afe94649c'
copy=A/'existing-DT-metadata-observer-completed.json'
if not copy.exists():copy.write_bytes(observer.read_bytes())
assert copy.read_bytes()==observer.read_bytes()
live=[]
for item in [dict(pid=2503202,birth=1792394518.8),dict(pid=2503287,birth=1792394519.29)]+metadata['results']:
    try:
        p=psutil.Process(item['pid'])
        live.append(dict(pid=p.pid,birth=p.create_time(),matches_recorded_birth=p.create_time()==item['birth'],title=p.name()))
    except psutil.NoSuchProcess:
        live.append(dict(pid=item['pid'],recorded_birth=item['birth'],gone=True))
result=dict(collected=time.time(),run=str(run),execution_source_commit=first['execution_source_commit'],
    method_source_commit=first['method_source_commit'],upstream_commit=first['upstream_commit'],
    method='u=clip(-expm1(-d),-2,1); multiply official normalized group/event credit, then native post-group variance scaling without centering',
    first_complete_iteration=first['current_summary'],DT_first_iteration_work=first['DT_actual_work'],
    actual_first_update_binding=dict(path=str(bindings),SHA256=hashlib.sha256(bindings.read_bytes()).hexdigest(),
        workers=[{k:x[k] for k in ('pid','birth','rank','actor_optimizer_entries','actor_optimizer_step_values','actor_scheduler_epoch','existing_DT_timing_observer_installed')}for x in metadata['results']]),
    physical_phase_summary=dict(path=str(phase_path),SHA256=hashlib.sha256(phase_path.read_bytes()).hexdigest(),content=phases),
    bounded_observations=profiles,existing_DT_metadata_observer=dict(original_receipt=str(observer),copied_receipt=str(copy)),
    previous_bounded_native_generation=generation_reference,
    live_processes=live,host_MemAvailable_GiB=psutil.virtual_memory().available/2**30,
    progress_at_least=dict(textcraft_DTPO=1,textcraft_PPO=0,textcraft_GRPO=0,appworld_DTPO=0,appworld_PPO=0,appworld_GRPO=0),
    interpretation=[
        'Generation caller waits on the complete synchronous eight-rank native turn; then native decode/environment/tokenization run before the next turn.',
        'Worker waits on vLLM native model futures are not by themselves evidence of idle GPU computation.',
        'DT window shows capture-stream synchronization, device-host transfer, finite propagation and B4 native pinned-cache release; no one window represents the whole iteration.',
        'Installed vLLM already compares _last_mapping before rebuilding LoRA metadata; no redundant local mapping cache was added.',
        'No-cross-B4-release trial was withdrawn after an eight-rank pinned allocation failure; current original B4 release boundary is preserved.',
        'GPU utilization is native device activity, not percentage of peak FLOPS. Separate-command bandwidth samples are not simultaneous kernel profiles.'
    ],new_training_method_changes=0,new_model_calls=0,new_optimizer_calls=0,new_environment_calls=0)
target=A/'current-GPU-utilization-evidence.json'
target.write_text(json.dumps(result,indent=2)+'\n')
result['remote_receipt']=str(target)
print(json.dumps(result))
'''))
path=Path('experiments/rl/results_dtpo_v9_negative2_current_GPU_util_20261019_v4.json')
path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(result=str(path),collected=report['collected'],
    live_processes=report['live_processes'],host_MemAvailable_GiB=report['host_MemAvailable_GiB']),ensure_ascii=False))
