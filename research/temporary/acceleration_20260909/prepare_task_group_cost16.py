"""Freeze cost groups using the actual formal helper and its real quality run.

CPU only. Reconstructs token packing for identity checks, never model or metric
calculations. The resulting plan does not assert any new timing measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/official'))
from batching import group_cases

sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.quality.read_bytes()
    report = json.loads(raw)
    assert report['status'] == 'complete' and report['family'] == 'qwen35'
    assert report['selection'] == 'development16' and report['sample_batch'] == 2
    assert report['dt_backend'] == 'accelerated_qwen35'
    assert report['driver_sha256'] == sha((ROOT / 'experiments/official/evaluate.py').read_bytes())
    assert report['batching_sha256'] == sha((ROOT / 'experiments/official/batching.py').read_bytes())
    assert report['selected_counts'] == {'niah_mq_q2': 8, 'morehopqa': 8}
    assert len(report['cases']) == 16
    prepared = []
    for row in report['cases']:
        ids = np.asarray(row['input_ids'], dtype=np.int64)[None]
        assert sha(ids.tobytes()) == row['input_sha256']
        assert row['metrics']['DT']['views'] == {'rise': 'signed', 'mas': 'positive_part', 'needle': 'positive_part'}
        prepared.append({'key': f"{row['dataset']}_{row['index']}", 'ids': ids, 'row': row})
    groups = []
    for task in report['selected_counts']:
        groups.extend(group_cases([c for c in prepared if c['row']['dataset'] == task], 2))
    assert len(groups) == len(report['DT_batches']) == 8
    batches = []
    for group, original in zip(groups, report['DT_batches']):
        assert [c['key'] for c in group] == original['cases']
        lengths = [c['ids'].shape[1] for c in group]
        eos = int(group[0]['ids'][0, -1])
        packed = np.full((4, max(lengths)), eos, dtype=np.int64)
        mask = np.zeros_like(packed)
        for j, case in enumerate(group):
            row = case['row']
            assert int(case['ids'][0, -1]) == eos
            packed[2*j:2*j+2, :lengths[j]] = case['ids']
            packed[2*j, [row['user_positions'][k] for k in row['keep']]] = eos
            mask[2*j:2*j+2, :lengths[j]] = 1
        root = original['actual_root']
        assert sha(packed.tobytes()) == root['input_sha256']
        assert sha(mask.tobytes()) == root['mask_sha256']
        assert root['sample_batch'] == 2 and root['endpoint_batch'] == 4
        batches.append({'cases': original['cases'], 'lengths': lengths,
                        'actual_root': root, 'padding_per_endpoint': 2*max(lengths)-sum(lengths)})
    plan = {'status': 'prepared_not_measured',
            'scope': 'Original Qwen3.5 development16; exact task-local groups from formal signed-view quality execution',
            'quality_report_sha256': sha(raw), 'quality_vectors_sha256': report['vectors_sha256'],
            'formal_driver_sha256': report['driver_sha256'], 'batching_sha256': report['batching_sha256'],
            'clean_sources_sha256': report['clean_sources_sha256'], 'acceleration_sources': report['acceleration_sources'],
            'cases': [{k: row[k] for k in ('dataset', 'index', 'input_sha256', 'prompt_length', 'target_length')}
                      for row in report['cases']],
            'batches': batches, 'warm_every_shape': True, 'interleaved_rounds': 2,
            'baseline': 'clean sample B1 through original formal attribute_batch helper',
            'candidate': 'accelerated sample B2 through same original formal attribute_batch helper',
            'metric_calls': 0, 'launch_only_after_paper_GPU_is_idle': True,
            'do_not_infer': 'No new speedup or quality result is produced by this CPU input-identity check.'}
    args.output.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': plan['status'], 'cases': len(prepared), 'batches': len(batches),
                      'padding_per_endpoint': sum(b['padding_per_endpoint'] for b in batches)}))


if __name__ == '__main__':
    main()
