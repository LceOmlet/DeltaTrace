"""Record actual diagnostic evidence without changing formal launch identities."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
REPOSITORY = AUDIT.parents[2]


def binding(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=path.as_posix(), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    endpoints_path = AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/endpoint-analysis.json'
    endpoints = json.loads(endpoints_path.read_bytes())
    memory_path = HERE/'candidate-analysis-v6/analysis.json'
    memory = json.loads(memory_path.read_bytes())
    observation_path = max(HERE.glob('observation-*.json'), key=lambda p: int(p.stem.split('-')[-1]))
    observation = json.loads(observation_path.read_bytes())
    assert all(item['hold_entered'] and not item['release_file_exists'] for item in observation['text_holds'])
    assert next(item for item in observation['processes'] if item['pid'] == 2786671)['status'] == 'NoSuchProcess'
    launch_path = HERE/'actual-results-v6/launch.json'
    launch = json.loads(launch_path.read_bytes())
    assert launch['formal_restart'] is False and launch['optimizer_steps_requested'] == 0
    cases = [{key: row[key] for key in ('task','source_token','candidate','native_d','saved_DT_d',
        'native_deleted_to_factual_ratio','DT_deleted_to_factual_ratio',
        'same_reward1_native_coefficient','saved_DT_reward1_coefficient',
        'factual_drift_from_original_saved','identity_controls','earlier_target_maxabs','top_target_effects')}
        for row in endpoints['cases']]
    code_commit = subprocess.check_output(['git','-c','core.longpaths=true','rev-parse','HEAD'],
        cwd=REPOSITORY, text=True).strip()
    source_path = AUDIT/'direct-target-prefix-runtime-20261007/v1/current-runtime-metadata/appworld-source.json'
    source = json.loads(source_path.read_bytes())
    first_phase = json.loads((HERE/'actual-results-v6/results/rank0-phases.jsonl').read_bytes().splitlines()[0])
    assert first_phase['runner_sha256'] == launch['candidate_preparation']['changed_sha256']
    assert all(row['full_failed_B4_previous_comparison']['signed_exact_equal'] and
        all(row['full_failed_B4_previous_comparison']['QVA_exact_equal'].values()) for row in memory['ranks'])
    now = datetime.datetime.now(datetime.timezone.utc)
    record = dict(observed_utc=now.isoformat(), observed_unix=now.timestamp(),
        scope='Actual extreme-token credibility and actual failed AppWorld B4 phase-memory diagnosis; no method or task configuration change',
        code_binding=dict(local_patch_commit=code_commit,upstream_lock=binding(REPOSITORY/'experiments/rl/upstream.lock'),
            formal_source=binding(source_path),formal_source_local_patch_commit=source['local_patch_commit'],
            actual_imported_runner=first_phase,original_owners=endpoints['cases'][1]['owners'],
            effective_configuration=binding(AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/appworld/effective-config.yaml')),
        endpoints=dict(binding=binding(endpoints_path), cases=cases,
            scope=endpoints['numerical_scope']),
        memory=dict(binding=binding(memory_path), ranks=memory['ranks'],
            source_transport=binding(HERE/'actual-results-v6/transport.json'),
            actual_launch=binding(launch_path), candidate=launch['candidate_preparation'],
            status='bounded_real_B4_validation_only_not_formal_deployment',
            scope=memory['physical_scope'], tolerances=memory['tolerances']),
        formal_state=dict(observation=binding(observation_path), textcraft='Same formal process, first update held, no release',
            appworld='Original formal process terminal; no checkpoint restored and no formal restart',
            physical=observation['physical'], processes=observation['processes'],
            text_holds=observation['text_holds']),
        unchanged=dict(PLAN=True,QVA=True,whitening=True,PPO=True,task_configuration=True,
            lora_rank=8,lora_alpha=16,per_GPU_actual_microbatch=4,diagnostic_devices=[4,5],
            credit_clipping=False,numerical_correction=False,checkpoint_restore=False),
        diagnostic_initialization_failures=dict(v2='Missing task environment at diagnostic launch, before DT',
            v4='Synchronous context was invalid for the task async actor, before DT',
            v5='Direct worker construction bypassed the trainer WorkerDict spawn interface, before DT',
            v6='Uses original create_colocated_worker_cls/spawn and AsyncLLMServerManager; no executor or registry patch'),
        evidence_scope='Two single-token probes supplement attribution evaluation; not a replacement for cumulative deletion, RISE or MAS. Exact output comparison concerns memory lifetime only, not a certificate of individual DT counterfactual accuracy.')
    result_path=REPOSITORY/'experiments/rl/results_extreme_credit_memory_20261007.json'
    result_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    current_path=REPOSITORY/'experiments/rl/current_runtime.json'
    current=json.loads(current_path.read_bytes())
    current['latest_extreme_credit_memory_diagnostic']=dict(
        observed_utc=record['observed_utc'],receipt=binding(result_path),
        formal_launches_unchanged=True,memory_candidate_not_formally_deployed=True,
        textcraft_update_held=True,appworld_formal_terminal=True)
    current_path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'runtime-snapshot-20261007.json').write_bytes(current_path.read_bytes())
    print(json.dumps(dict(result=binding(result_path),runtime=binding(current_path))))


if __name__ == '__main__':
    main()
