"""Read immutable companion-branch evidence; do not run or alter its experiments.

These observations constrain a GDN-only explanation. They do not causally
identify a finite operator, compare cross-model absolute metrics, or adopt a fix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
COMMIT = 'e624b88'
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    args = p.parse_args()
    commit = subprocess.check_output(['git', '-C', str(args.repository), 'rev-parse', COMMIT]).decode().strip()
    receipts = {}

    def blob(path):
        data = subprocess.check_output(['git', '-C', str(args.repository), 'show', commit + ':' + path])
        receipts[path] = sha(data)
        return data

    base = 'research/temporary/'
    full = base + 'five_dataset_disagreement_full_20260909/'
    part = base + 'target_partition_diagnosis_20260909/'
    f = json.loads(blob(full + 'results/summary.json'))
    s = json.loads(blob(part + 'results/summary.json'))
    assert f['status'] == s['status'] == 'verified'
    assert sha(blob(full + 'results/raw_results.json')) == f['raw_report_sha256']
    assert sha(blob(part + 'results/raw_results.json')) == s['raw_sha256']
    assert s['native_effects_sha256'] == receipts[full + 'results/summary.json']
    assert f['actual_input_hashes_reconstructed'] == 2688 and f['unchanged_original_native_scores'] == 448
    assert s['count'] == 40 and s['original_full_vectors_bitwise_equal']
    rows = []
    for dataset, d in f['datasets'].items():
        effects = d['effects']['full']
        roots = s['datasets'][dataset]['roots']
        rows.append({'dataset': dataset, 'full_count': d['count'],
                     'full_target_deletion_A_larger': effects['positive_zero_negative']['deletion_A_minus_B'][0],
                     'full_target_insertion_A_larger': effects['positive_zero_negative']['insertion_A_minus_B'][0],
                     'full_target_A_larger_at_both': effects['endpoint_direction_categories']['A_larger_at_both'],
                     'partition_indices': s['datasets'][dataset]['indices'],
                     'thinking_DT_A_larger': roots['thinking']['DT_A_larger'],
                     'thinking_native_balanced_A_larger': roots['thinking']['native_balanced_A_larger']})
    report = {'status': 'committed_raw_and_summary_hashes_verified', 'companion_commit': commit,
              'source_hashes': receipts, 'rows': rows,
              'definitions': 'A is a fixed matched-size FT K3-only gold group; B is DT-only non-gold. These are selected after rankings, not random treatment groups. Balanced means the average of the two endpoint group effects, not a replacement paper metric.',
              'sample_scope': 'All 448 original Qwen3 examples include the first40 previously viewed pilot; same-objective partition uses indices8..15 of each of five tasks. No Qwen3.5 data added by this audit.',
              'implication': 'Qwen3 has consequential allocation/deletion mismatches without GDN. In the three single-chain VT groups, the thinking-only component still orders the selected groups oppositely to native balanced effects in most of the next-eight cases. Sink+EOS target mismatch alone is insufficient.',
              'limits': 'This imports immutable companion evidence, not an independent rerun or mechanistic causal proof. It neither identifies the cause of Qwen3.5 NI gaps nor establishes a uniform successful repair. Do not modify FT, remove target terms, or attribute the whole discrepancy to GDN on this basis.'}
    (HERE / 'shared_quality_evidence.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
