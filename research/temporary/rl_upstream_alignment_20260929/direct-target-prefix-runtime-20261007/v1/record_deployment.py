"""Index original submission, source and read-only worker receipts locally.

No remote, model, training, checkpoint or process operation. This is provenance
aggregation, not a launcher or a replacement for remote active manifests.
"""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=path.as_posix(), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    paired_path = HERE / 'paired-controller-readonly-1791370527.json'
    paired = read(paired_path)
    phase_path = HERE / 'paired-first-dt-readonly-1791371004.json'
    phase = read(phase_path)
    observed = phase['observed_unix']
    metadata = HERE / 'current-runtime-metadata'
    active = read(metadata / 'active-training.json')
    records = []
    for task, slug, submission, stage in [
        ('AppWorld', 'appworld', 'submit.stdout.json', 'stage-receipt.json'),
        ('TextCraft', 'textcraft', 'textcraft-submit.stdout.json', 'textcraft-stage-receipt.json'),
    ]:
        receipt = read(HERE / submission)
        job = receipt['job']
        original = next(row for row in active['jobs'] if row['task'] == task)
        assert original == job
        source_path = metadata / (slug + '-source.json')
        source = read(source_path)
        assert binding(source_path)['sha256'] == receipt['source_sha256']
        observation = next(row for row in paired['jobs'] if row['task'] == slug)
        assert observation['expected']['pid'] == job['pid']
        assert observation['expected']['birth'] == job['observed_process_created_unix']
        assert observation['expected']['source_sha256'] == receipt['source_sha256']
        current = next(row for row in phase['jobs'] if row['task'] == slug)
        assert current['expected']['driver_pid'] == job['pid']
        assert current['expected']['birth'] == job['observed_process_created_unix']
        assert current['source']['sha256'] == receipt['source_sha256']
        controller = next(row['json'] for row in observation['metadata']
                          if row['path'].endswith('/controller-status.json'))
        assert controller['status'] == 'observer_armed_not_training_health_proof'
        assert controller['submissions'] == 2
        assert sorted(row['owner']['rank'] for row in controller['workers']) == [0, 1]
        driver = current['processes'][0]
        records.append(dict(**job, job=job,
            source=dict(path=job['source_receipt'], local_path=binding(source_path)['path'],
                        sha256=receipt['source_sha256'], bytes=source_path.stat().st_size),
            process=dict(alive=True, created_unix=driver['actual_birth'],
                         pid_identity_matches=True, status=driver['status'],
                         workers=[row['pid'] for row in current['processes'][1:]]),
            remote_source=job['source_receipt'], startup_options=source['startup_options'],
            source_code_commit=source['local_patch_commit'],
            source_preparation=binding(HERE / stage), submission=binding(HERE / submission),
            controller=controller, original_worker_processes=current['processes'][1:],
            phase='original formal sampling; first DT not yet activated at this observation',
            completed_DT=False, completed_PPO_update=False, joint32k_capacity_verified=False,
            runtime_overrides_scope=
                'Original WorkerDict RPC armed one-shot passive input/native-credit/MLP metadata observation '
                'and Event hold before first update; no numerical or objective modification'))
    result = dict(observed_unix=observed,
        observed_utc=datetime.datetime.fromtimestamp(observed, datetime.timezone.utc).isoformat(),
        status='two_fresh_formal_jobs_sampling_observers_armed_not_training_health_proof',
        tasks=records, original_resource_observation=binding(phase_path),
        original_controller_observation=binding(paired_path),
        prepared_CPU_verification=binding(HERE / 'parent-prepared-verification.json'),
        head_owner_readonly_audit=binding(HERE / 'textcraft-appworld-head-owner-readonly.json'),
        native_MLP_owner_API_readonly_audit=binding(HERE.parents[1] /
            'direct-target-native-mlp-memory-20261007/v1/native-mlp-existing-owner-api-readonly-20261007.json'),
        observation_code_commit='56e4d778b222b3ab4dd64500e810cfa9812e65fd',
        textcraft_preparation_commit='40dc2a7a6d1bf289fab1111010406e17a7603acc',
        recorder=binding(__file__),
        invariants='Original task budgets, generation/environment/PPO/whitening/LoRA8/16/B4 per card/32k ceilings unchanged. Only producer/readout prefix wiring changed per task.',
        evidence_scope='Old saved B8 input contains one current response per row, not the full multiround joint target. New original full DataProto will be saved at first DT entry; not yet present.',
        observer_owner_scope='v4 armed.owners enumerates callable actor and ref objects; only the actual assigned producer matches. v2 installs only on the owner with that producer/direct_readout. Two rank RPCs, not duplicated updates.',
        unresolved='Native HF/PEFT MLP peak and extreme per-token credit remain unverified; no credit clipping/correction, checkpoint recovery, backup or unrelated task started.')
    result_path = REPO / 'experiments/rl/results_prefix_runtime_20261007.json'
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    snapshot_path = REPO / 'experiments/rl/current_runtime.json'
    snapshot = read(snapshot_path)
    retired = snapshot.setdefault('retired_runtime_jobs', [])
    for prior in snapshot['jobs']:
        if prior['task'] in ('AppWorld', 'TextCraft'):
            current = next(row for row in records if row['task'] == prior['task'])
            if prior['pid'] != current['pid']:
                retired.append(dict(prior, retirement_scope='Original terminal job before fresh prefix runtime; see 2026-10-07 17:56 ledger'))
    untouched = [dict(row, current_scope='inactive historical task; not restarted')
                 for row in snapshot['jobs'] if row['task'] not in ('AppWorld', 'TextCraft')]
    snapshot['jobs'] = untouched + records
    snapshot.update(observed_unix=observed, observed_utc=result['observed_utc'],
        latest_observation_unix=observed, latest_observation_utc=result['observed_utc'],
        latest_readonly_observation=dict(receipt=binding(result_path),
            textcraft='New PID2833207 sampling; one-shot observer armed; no first DT/update',
            appworld='New PID2786671 sampling; one-shot observer armed; native MLP OOM repair not yet proved'),
        latest_runtime_deployment=dict(receipt=binding(result_path), scope=result['status']),
        authoritative_remote={name:dict(path=str(PurePosixPath(records[0]['job']['output']).parents[3] / name),
            exists=True, sha256=binding(metadata / name)['sha256'],
            bytes=binding(metadata / name)['bytes'], local_path=binding(metadata / name)['path'])
            for name in ('active-training.json', 'active-source.json')},
        manifest=dict(path=active['manifest'], exists=True,
            sha256=binding(metadata / 'formal-training.json')['sha256'],
            local_path=binding(metadata / 'formal-training.json')['path']),
        code_repository_commit_at_collection='40dc2a7a6d1bf289fab1111010406e17a7603acc',
        recording_command='C:/Users/Administrator/miniconda3/python.exe -X utf8 ' + Path(__file__).as_posix(),
        recorder_source=binding(__file__))
    # Preserve the existing snapshot line endings; old dated evidence remains.
    snapshot_path.write_bytes((json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n').replace('\n', '\r\n').encode())
    print(json.dumps(dict(result=binding(result_path), snapshot=binding(snapshot_path))))


if __name__ == '__main__':
    main()
