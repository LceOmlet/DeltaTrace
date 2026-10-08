"""Summarize original-owner observations without declaring an unreproduced fix."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def receipt(path):
    raw = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    observations = [json.loads(p.read_bytes()) for p in sorted((HERE/'observations').glob('[0-9]*.json'))]
    final = observations[-1]
    files = final['files']
    phase = files['phase.json']
    assert phase['phase'] in ('complete', 'failed'), 'Diagnostic is still active or timed out'
    ranks = [files.get(f'rank{i}.json', {}) for i in (0, 1)]
    summaries = []
    for rank in ranks:
        steps = rank.get('optimizer_steps', [])
        completed = [s for s in steps if s.get('completed_unix')]
        nonfinite = []
        for row in rank.get('microbatches', []):
            for key, value in row.items():
                values = value if key == 'policy_outputs' else [value]
                for item in values:
                    if isinstance(item, dict) and item.get('nonfinite', 0):
                        nonfinite.append(dict(microbatch=row['index'], field=key, stats=item))
        summaries.append(dict(rank=rank.get('rank'),phase=rank.get('phase'),
            microbatches_started=len(rank.get('microbatches', [])),completed_optimizer_steps=len(completed),
            native_grad_norms=[s['native_grad_norm'] for s in completed],
            nonfinite_gradients=[s.get('nonfinite_gradients') for s in steps],
            nonfinite_outputs=nonfinite,exception=rank.get('error'),traceback=rank.get('traceback'),
            metrics=rank.get('metrics')))
    all_finite = phase['phase']=='complete' and all(
        r['phase']=='complete' and r['microbatches_started']==32 and r['completed_optimizer_steps']==4
        and all(math.isfinite(v) for v in r['native_grad_norms']) and not any(r['nonfinite_gradients'])
        and not r['nonfinite_outputs'] and not r['exception'] for r in summaries)
    state_changes = []
    for rank in (0, 1):
        before = files[f'rank{rank}-before_vllm_init-handoff.json']
        for name in ('after_vllm_init', 'after_vllm_generation_sleep'):
            after = files[f'rank{rank}-{name}-handoff.json']
            ignored = {'phase','unix','allocated','reserved','pss_bytes','fsdp_states','lora'}
            delta = {k:[before[k], after.get(k)] for k in before if k not in ignored and before[k]!=after.get(k)}
            fsdp_delta = [dict(name=a['name'],fields={k:[a[k],b.get(k)] for k in a if a[k]!=b.get(k)})
                          for a,b in zip(before['fsdp_states'],after['fsdp_states']) if a!=b]
            state_changes.append(dict(rank=rank,phase=name,changed_fields=delta,fsdp_changes=fsdp_delta,
                changed_trainable_parameters=[k for k in before['lora'] if before['lora'][k]!=after['lora'].get(k)]))
    stage_times = {}
    # Phase lines are emitted by the original diagnostic helper. Preserve their
    # observed timestamps; do not infer unobserved boundaries from polling.
    for observation in observations:
        for line in observation.get('driver_log', {}).get('text','').splitlines():
            if line.startswith('{"phase":'):
                row = json.loads(line)
                stage_times.setdefault(row['phase'],row['unix'])
    pairs = [('native_actor_rollout_init_begin','native_rollout_context_begin'),
             ('native_generate_begin','native_rollout_context_end'),
             ('native_old_log_prob_begin','native_ref_log_prob_begin'),
             ('native_ref_log_prob_begin','original_DT_begin'),
             ('original_DT_begin','native_update_begin'),('native_update_begin','complete')]
    durations = {a:stage_times[b]-stage_times[a] for a,b in pairs if a in stage_times and b in stage_times}
    memory = [int(x) for o in observations for x in re.findall(r'(\d+)/65536 MiB',o['physical'])]
    kl_values = [v for r in summaries for v in (r['metrics'] or {}).get('actor/ppo_kl',[])]
    result = dict(status='original_rollout_cycle_DT_full_PPO_finite' if all_finite else 'diagnostic_failed',
        root_cause_located=False,fixed=False,observed_unix=final['observed_unix'],
        launch=final['launch'],upstream_commit='20bd331',
        diagnostic_source=receipt(HERE/'diagnose_rollout_dt_to_actor.py'),
        reused_observer_source=receipt(HERE.parent/'v2/diagnose_dt_to_actor.py'),
        owner_handoff_sources=receipt(HERE/'rollout-handoff-sources.json'),
        native_offload_sources=receipt(HERE/'native-offload-sources.json'),
        original_incident_metrics=receipt(HERE/'original-Ray-metrics.json'),
        original_function_identity=receipt(HERE.parent/'v2/original-PPO-function-identity.json'),
        terminal_observation=receipt(HERE/'observations'/f"{int(final['observed_unix'])}.json"),
        full_fresh_update_finite=all_finite,ranks=summaries,handoff_states=state_changes,
        DT_output=files.get('DT-output-observation.json'),
        original_reported_ppo_kl=0.288,diagnostic_aggregate_ppo_kl=statistics.mean(kl_values) if kl_values else None,
        probability_difference_interpretation='A measured replay difference, not proof of a causal mechanism or a numerical acceptance threshold.',
        retained_initial_LoRA=[files[f'rank{i}-initial.json']['initial_tensor_capture'] for i in (0,1)],
        retained_old_ref_input=files.get('actor-input-with-native-old-ref.json'),
        phase_seconds=durations,maximum_observed_tree_pss_bytes=max(sum(p['pss_bytes'] for p in o['processes']) for o in observations),
        maximum_observed_physical_GPU_MiB=max(memory) if memory else None,
        remaining_owned_processes=[p['pid'] for p in final['processes'] if p['status']!='zombie'],
        formal_training_restarted=False,checkpoint_restore=False,deployed_repair=False,
        limits=[
            'One native VERL/vLLM base-sampling generation cycle on32 saved initial prompts, not the original30-turn environment rollout.',
            'Original saved actor advantages and DT carrier are preserved separately; generated diagnostic tokens do not enter training.',
            'Original initial LoRA A, old/ref log-probs and actor position_ids were not saved; positions are reconstructed through the original helper.',
            'Diagnostic physical4/5 differs from incident physical2/3.',
            'Native anomaly detection and synchronous observations can affect timing; this is not formal training throughput.',
            'Recorded parameter hashes cover trainable local shards, not every frozen base parameter.',
            'No causal fix is established by a finite fresh replay; official PPO and DT numerical tolerances were not changed.',
        ])
    path = ROOT/'experiments/rl/results_rollout_dt_to_native_actor_nonfinite_20261008.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','root_cause_located','fixed','diagnostic_aggregate_ppo_kl','phase_seconds','maximum_observed_tree_pss_bytes','maximum_observed_physical_GPU_MiB','remaining_owned_processes')},indent=2))


if __name__=='__main__':
    main()
