"""Prepare bounded factual-EOS comparisons on the frozen development collection.

Research only. Four most-negative source d values are the proposed refinement
positions; eight uniform remaining positions diagnose the rest of the vector.
The first group does NOT estimate overall accuracy. Together their known
inclusion probabilities permit population estimates without treating an
outlier-only query set as representative. No reference-policy tokens sampled.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import resource
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(corpus, manifest):
    results = []
    for task, data in manifest['tasks'].items():
        selected = {uid for group in data['groups'] if group['split'] == 'development'
                    for uid in group['first_stage_uids']}
        records = {row['traj_uid']: row for row in corpus['tasks'][task]['records']}
        files = {value['sha256']: value for value in corpus['tasks'][task]['native_files']}
        loaded_path, native = None, None
        for uid in sorted(selected):
            record = records[uid]
            result = {'task': task, 'traj_uid': uid, 'initial_state_sha256': record['initial_state_sha256'],
                      'status': 'needs_original_baseline_DT', 'queries': []}
            if not record['native_occurrences']:
                results.append(result)
                continue
            occurrence = record['native_occurrences'][0]
            binding = files[occurrence['file_sha256']]
            path = Path(binding['path'])
            if path != loaded_path:
                if sha(path) != binding['sha256']:
                    raise ValueError('Original native artifact changed')
                native = torch.load(path, map_location='cpu', weights_only=False)
                loaded_path = path
            row = next(value for value in native['rows'] if value['batch_row'] == occurrence['batch_row'])
            assert row['traj_uid'] == uid
            positions = row['prompt_length']+row['prior'][row['suffix_positions']].nonzero().flatten()
            signed = native['native_signed'][row['batch_row'], positions]
            if not bool(torch.isfinite(signed).all()):
                raise ValueError('Nonfinite saved d is a numerical failure, not a selection score')
            negatives = [int(i) for i in signed.lt(0).nonzero().flatten()]
            refined = sorted(negatives, key=lambda index: (float(signed[index]), index))[:4]
            remaining = sorted(set(range(len(positions)))-set(refined))
            seed = int(hashlib.sha256((manifest['seed']+'\0'+task+'\0'+uid).encode()).hexdigest(), 16)
            controls = random.Random(seed).sample(remaining, min(8, len(remaining)))
            for index in refined+controls:
                packed = int(positions[index])
                selected_for_refinement = index in refined
                result['queries'].append({
                    'source_index': index, 'packed_slot': packed,
                    'token_id': int(row['selected'][packed]), 'saved_d': float(signed[index]),
                    'purpose': 'refinement_candidate' if selected_for_refinement else 'uniform_remaining_diagnostic',
                    'inclusion_probability': 1.0 if selected_for_refinement else len(controls)/len(remaining),
                    'deletion': 'Original EOS substitution at this one existing source position; all other factual IDs unchanged',
                })
            result.update(status='prepared_only', native=binding, batch_row=row['batch_row'],
                          source_count=len(positions), negative_source_count=len(negatives),
                          proposed_extra_deletions=len(refined),
                          diagnostic_remaining_population=len(remaining),
                          diagnostic_remaining_samples=len(controls))
            results.append(result)
    return results


if __name__ == '__main__':
    raise SystemExit('Withdrawn by user: this query-selection candidate must not be prepared or launched.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_bytes())
    assert sha(args.corpus) == manifest['corpus']['sha256']
    corpus = json.loads(args.corpus.read_bytes())
    rows = build(corpus, manifest)
    result = {'scope': __doc__, 'corpus_sha256': sha(args.corpus),
              'manifest_sha256': sha(args.manifest), 'source_sha256': sha(Path(__file__)),
              'candidate_is_production': False, 'rows': rows,
              'runtime': {'unix': time.time(), 'seconds': time.perf_counter()-started,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'cuda_initialized': torch.cuda.is_initialized(), 'model_calls': 0,
                  'DT_calls': 0, 'optimizer_steps': 0}}
    assert not result['runtime']['cuda_initialized']
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'runtime': result['runtime'],
                      'trajectories': len(rows), 'needs_baseline': sum(not row['queries'] for row in rows),
                      'source_queries_prepared': sum(len(row['queries']) for row in rows)}))
