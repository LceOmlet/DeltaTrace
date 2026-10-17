"""Read native outputs and execute the pinned official hybrid TP comparator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import warnings

import torch

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/native-rollout-TP-efficiency-20261018-v1'


def main():
    records = {}
    for tp in (1, 2):
        directory = AUDIT/f'TP{tp}-state10-v2'
        value = json.loads((directory/'status.json').read_bytes())
        assert value['status'] == 'completed_native_generation_probe', value.get('error')
        assert value['formal_iterations'] == value['DT_calls'] == value['optimizer_updates'] == 0
        assert len(value['results']) == 8
        assert all(r['tp'] == tp and r['scheduler'] == 10 and r['optimizer_entries'] == 496 for r in value['results'])
        assert value['input_SHA256'] == hashlib.sha256((directory/'inputs.snapshot.json').read_bytes()).hexdigest()
        records[tp] = value
    assert records[1]['input_SHA256'] == records[2]['input_SHA256']
    assert records[1]['diagnostic_nonzero_model_state'] == records[2]['diagnostic_nonzero_model_state']
    cases = json.loads((AUDIT/'TP1-state10-v2/inputs.snapshot.json').read_bytes())['cases']
    owner = ROOT/'receipts/upstream-alignment-20260929/verify_vllm_batching_owner.py'
    spec = importlib.util.spec_from_file_location('pinned_native_vllm_tests', owner)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    checks, cost = [], []
    for case, prompts in cases.items():
        trials = ('cold', 'hot') if case in ('dense', 'tail') else ('cold',)
        for trial in trials:
            outputs, times = {}, {}
            for tp, value in records.items():
                packed = {}
                times[tp] = []
                for row in value['results']:
                    metadata, = [r for r in row['cases'] if (r['case'],r['trial']) == (case,trial)]
                    times[tp].append(metadata['seconds'])
                    data = torch.load(metadata['outputs'], map_location='cpu', weights_only=False)
                    assert data['prompts'] == [prompts[i] for i in data['global_indices']]
                    assert all(len(x[0]) == 128 for x in data['outputs'])
                    if row['rank'] % tp == 0:
                        for index, output in zip(data['global_indices'], data['outputs']):
                            assert index not in packed
                            packed[index] = output
                assert sorted(packed) == list(range(len(prompts)))
                outputs[tp] = [packed[i] for i in range(len(prompts))]
            cost.append(dict(case=case, trial=trial, exact_global_requests=len(prompts),
                exact_generated_tokens=128*len(prompts),
                TP1_global_synchronous_seconds=max(times[1]), TP2_global_synchronous_seconds=max(times[2]),
                TP1_rank_seconds=times[1], TP2_rank_seconds=times[2],
                observed_TP2_speed_ratio=max(times[1])/max(times[2]),
                includes_cold_initialization_effects=(trial == 'cold'),
                scope='Matched native engine generation only, not whole training speed'))
            if case in ('numeric_short', 'capacity_32768'):
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter('always')
                    harness.compare(outputs_0_lst=outputs[1], outputs_1_lst=outputs[2],
                        name_0='native_TP1', name_1='native_TP2')
                checks.append(dict(case=case, status='passed_original_hybrid_TP_test_check_logprobs_close',
                    warnings=[str(w.message) for w in seen], new_thresholds=0))
    result = dict(status='native_generation_cost_and_original_TP_protocol_completed_not_deployed',
        input_SHA256=records[1]['input_SHA256'], fixed_native_state=records[1]['diagnostic_nonzero_model_state'],
        results=cost, numerical_checks=checks, original_test_sources=harness.report['sources'],
        official_contract=str(AUDIT/'official-validation-contract.json'),
        formal_iterations=0, optimizer_updates=0, method_changes=0,
        scope='Official sampler TP1/TP2 consistency and real generation/storage capacity; no absolute-logprob-error or training-quality claim')
    (AUDIT/'comparison-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
