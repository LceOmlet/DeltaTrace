"""Record actual single-background diagnostic identity and completed evidence."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('textcraft','appworld'),required=True)
    args = parser.parse_args()
    folder = HERE/f'single-background-{args.task}-observations'
    observation = max((p for p in folder.glob('*.json') if p.stem.isdecimal()),key=lambda p:int(p.stem))
    snapshot = json.loads(observation.read_bytes())
    launch = json.loads((HERE/f'single-background-{args.task}-launch.json').read_bytes())
    ranks = []
    for rank in (0,1):
        binding = snapshot['files'].get(f'results/rank{rank}.json',{})
        value = binding.get('value',{})
        if value.get('summary_only'):
            value = json.loads((folder/f'rank{rank}.json').read_bytes())
            candidates = [json.loads(p.read_bytes())['files'].get(f'results/rank{rank}.json',{})
                          for p in folder.glob('*.json') if p.stem.isdecimal()]
            assert any(c.get('sha256') == binding.get('sha256') and c.get('value') == value for c in candidates), \
                'Fetch a full observation of the actual current rank result before recording it.'
        ranks.append(dict(rank=rank,phase=value.get('phase'),pid=value.get('pid'),birth=value.get('birth'),
            elapsed_seconds=value.get('elapsed_seconds'),operations=value.get('operations'),
            finite_runner=value.get('finite_runner'),owners=value.get('owners'),
            remote_raw_sha256=binding.get('sha256'),
            local=ref(folder/f'rank{rank}.json') if (folder/f'rank{rank}.json').exists() else None,
            completed_points=sum(len(b['points']) for b in value.get('batches',[]))))
    complete = all(r['phase']=='complete' for r in ranks)
    failed = any(r['phase']=='failed' for r in ranks)
    status = ('completed_diagnostic' if complete else 'failed_partial_diagnostic' if failed else
              'terminal_incomplete_diagnostic' if snapshot.get('driver_alive') is False else 'running_diagnostic')
    result = dict(scope=__doc__,task=args.task,status=status,
        launch=launch,launch_artifact=ref(HERE/f'single-background-{args.task}-launch.json'),
        observation=ref(observation),ranks=ranks,physical=snapshot.get('physical'),host=snapshot.get('host_memory'),
        driver_alive=snapshot.get('driver_alive'),plan=ref(HERE/'layer-collection-inputs.json'),
        production_modified=False,formal_restart=False,credit_repaired=False,official_tolerance_claim=False,
        source=ref(Path(__file__)),
        scope_limit='Original finite calls at single-source endpoints diagnose background allocation and native numerical residual separately. They are not a permitted training-time per-source replacement or a new whole-method quality test.')
    analysis = HERE/f'single-background-{args.task}-analysis.json'
    if analysis.exists():
        a = json.loads(analysis.read_bytes())
        result['analysis'] = ref(analysis)
        result['diagnostic_counts'] = {k:a[k] for k in ('complete','point_count','original_spurious_tail','original_spurious_tail_still_spurious')}
    preserved = HERE/f'single-background-{args.task}/preserved-artifacts/manifest.json'
    if preserved.exists():
        m = json.loads(preserved.read_bytes())
        result['preserved'] = dict(manifest=ref(preserved),files=len(m['files']),
            uncompressed_bytes=sum(v['bytes'] for v in m['files']),archive=m['archive'],
            all_file_SHA256_verified=m['all_file_SHA256_verified'])
    output = REPO/f'experiments/rl/results_single_background_{args.task}_20261009.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    p = REPO/'experiments/rl/current_runtime.json'
    current = json.loads(p.read_bytes())
    if current.get('latest_single_background_collection_20261009',{}).get('task'):
        current['latest_single_background_collection_20261009'] = {
            k:v for k,v in current['latest_single_background_collection_20261009'].items()
            if k in ('textcraft','appworld')}
    current.setdefault('latest_single_background_collection_20261009',{})[args.task] = dict(
        status=result['status'],receipt=ref(output),observed_unix=snapshot['unix'],
        driver_alive=result['driver_alive'],pid=launch['pid'],birth=launch['birth'],devices=[4,5],
        base_commit=launch['code_commit'],actual_script_sha256=launch['scripts']['inspect_single_background_collection.py'],
        production_modified=False,formal_restart=False)
    p.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(receipt=str(output),status=result['status'],ranks=[r['phase'] for r in ranks])))


if __name__ == '__main__':
    main()
