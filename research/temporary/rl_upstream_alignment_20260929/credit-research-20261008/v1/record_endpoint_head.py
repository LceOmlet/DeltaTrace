"""Bind the single bounded head-rule experiment to its actual artifacts."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    folder=HERE/'endpoint-head-owner-v1'
    observations=HERE/'endpoint-head-textcraft-observations'
    observed=max((p for p in observations.glob('*.json') if p.stem.isdecimal()),key=lambda p:int(p.stem))
    snapshot=json.loads(observed.read_bytes())
    launch=json.loads((folder/'textcraft-launch.json').read_bytes())
    ranks=[]
    for rank in (0,1):
        binding=snapshot['files'].get(f'results/rank{rank}.json',{})
        value=binding.get('value',{})
        if value.get('summary_only'):
            raw_path=observations/f'rank{rank}.json'
            # The observer reserializes JSON with Windows newlines. Bind the
            # original remote byte hash separately; compare parsed contents.
            full=json.loads(raw_path.read_bytes())
            candidates=[json.loads(p.read_bytes())['files'].get(f'results/rank{rank}.json',{})
                for p in observations.glob('*.json') if p.stem.isdecimal()]
            assert any(c.get('sha256')==binding['sha256'] and c.get('value')==full for c in candidates)
            value=full
        ranks.append(dict(rank=rank,phase=value.get('phase'),operations=value.get('operations'),
            elapsed_seconds=value.get('elapsed_seconds'),actual_dtype_check=value.get('candidate',{}).get('actual_dtype_check'),
            imported_owners=value.get('owners'),candidate=value.get('candidate'),
            remote_raw_sha256=binding.get('sha256'),
            raw=ref(observations/f'rank{rank}.json') if (observations/f'rank{rank}.json').exists() else None))
    complete=all(r['phase']=='complete' for r in ranks)
    analysis=folder/'textcraft-analysis.json'
    result=dict(scope=__doc__,status='completed_unaccepted' if complete else 'running_unaccepted',
        observation=ref(observed),launch=dict(artifact=ref(folder/'textcraft-launch.json'),**launch),
        source_commit=launch['code_commit'],ranks=ranks,
        derivation=ref(HERE/'endpoint-head-derivation.json'),
        comparison_inputs=ref(folder/'comparison-inputs-textcraft.json'),
        analysis=ref(analysis) if analysis.exists() else None,
        physical=snapshot.get('physical'),host_memory=snapshot.get('host_memory'),
        driver_alive=snapshot.get('driver_alive'),
        official_FA_FLA_kernels_unchanged=True,
        numerical_scope='Actual first-call primitive FP32/FP64 residuals are diagnostics; no new or borrowed official tolerance.',
        operations_excluded=['optimizer','rollout','checkpoint_restore','reference_token_sampling'],
        accepted_candidate=False,production_modified=False,formal_restart=False,
        recorder=ref(Path(__file__)))
    if complete and analysis.exists():
        data=json.loads(analysis.read_bytes())
        result['primary_metrics']={k:{n:v for n,v in metric.items()
            if n not in ('paired_trajectories','state_means')} for k,metric in data['primary_metrics'].items()}
        result['decision']='Whole-collection quality and cross-cell tail diagnostics must be assessed together; no automatic deployment.'
        summary=folder/'textcraft-summary.json'
        if summary.exists():
            verdict=json.loads(summary.read_bytes())
            result.update(status=verdict['decision'],summary=ref(summary),
                          decision=verdict['decision_evidence'],next_task_GPU_calls=0)
    preservation=folder/'preserved-artifacts/manifest.json'
    if preservation.exists():
        manifest=json.loads(preservation.read_bytes())
        result['preserved_debug_artifacts']=dict(manifest=ref(preservation),
            files=len(manifest['files']),uncompressed_bytes=sum(f['bytes'] for f in manifest['files']),
            archive=manifest['archive'],local_archive=manifest['local_archive'],
            all_file_SHA256_verified=manifest['all_file_SHA256_verified'])
    output=ROOT/'experiments/rl/results_endpoint_head_20261009.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    current=ROOT/'experiments/rl/current_runtime.json'
    value=json.loads(current.read_bytes())
    value['latest_head_candidate']=dict(status=result['status'],receipt=ref(output),
        observed_unix=snapshot['unix'],
        source_commit=launch['code_commit'],pid=launch['pid'],birth=launch['birth'],
        devices=launch['devices'],driver_alive=result['driver_alive'],
        production_modified=False,formal_restart=False)
    value['latest_readonly_observation']['diagnostic']='Isolated endpoint-head candidate: '+result['status']
    current.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(receipt=str(output),status=result['status'],ranks=[r['phase'] for r in ranks])))


if __name__=='__main__':
    main()
