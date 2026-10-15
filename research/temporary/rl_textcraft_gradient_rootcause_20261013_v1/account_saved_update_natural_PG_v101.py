"""Account the completed actual native PG readouts from saved update states.

Pure arithmetic on native receipts. No model, optimizer, normalization or
gradient implementation is supplied, and no cross-run vector subtraction is
claimed from previously unmatched fixed-adjoint experiments.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / 'native-saved-update-natural-PG-v99-status-final.json'
status = json.loads(source.read_bytes())['report']
assert not status['alive']
assert status['state']['phase'] == 'complete_native_gradient_chain_localization_not_repair'
manifest = json.loads((HERE / 'prepare_saved_update_natural_PG_v99-dt.json').read_bytes())['report']
specs = {c['name']: c for c in manifest['cases']}
assert len(status['state']['observations']) == len(specs) == 6
inputs = {}
rows = []
for name, spec in specs.items():
    chains = [item['value'] for item in status['chain_reports'] if item['value']['iteration'] == name]
    assert {item['rank'] for item in chains} == {0, 1}
    gram = [[0.] * 8 for _ in range(8)]
    head = []
    for chain in chains:
        assert chain['trainable_SHA256_before'] == chain['trainable_SHA256_after'] == spec['expected_trainable_SHA256_by_rank'][str(chain['rank'])]
        assert chain['scheduler_before'] == chain['scheduler_after'] == 80
        actual_inputs = chain['native_forward_inputs']
        assert len(actual_inputs) == 8
        if name == 'old80_before':
            inputs[chain['rank']] = actual_inputs
        else:
            assert actual_inputs == inputs[chain['rank']]
        for record in chain['records']:
            assert record.get('input_adjoint_nonfinite_positions', 0) == 0
            assert record.get('output_adjoint_nonfinite_positions', 0) == 0
        acc = chain['native_accumulation_readout']
        assert acc['original_backward_calls'] == 8
        for i in range(8):
            for j in range(8):
                gram[i][j] += acc['local_increment_Gram'][i][j]
        seeds = chain['diagnostic_boundary_seeds']
        assert len(seeds) == 8
        for value in seeds:
            assert value['applied_norm'] == value['original_incoming_norm']
            assert value['mode'] in ('capture_unmodified_native_seed', 'observe_native_head_adjoint_no_replacement')
        head += [dict(rank=chain['rank'], **value) for value in seeds]
    observation = next(item for item in status['state']['observations'] if item['iteration'] == name)
    assert observation['diagnostic_seed_intervention'] is False
    assert set(observation['gradient_statistics']['norms']) == {'full_DT'}
    norm = observation['gradient_statistics']['norms']['full_DT']
    reconstructed = math.sqrt(max(0., math.fsum(math.fsum(row) for row in gram)))
    rss = math.sqrt(math.fsum(gram[i][i] for i in range(8)))
    native_head_rss = math.sqrt(math.fsum(item['original_incoming_norm'] ** 2 for item in head))
    head_comparison = None
    if name != 'old80_before':
        aa = math.fsum(item['reference_sumsq'] for item in head)
        bb = math.fsum(item['current_sumsq'] for item in head)
        ab = math.fsum(item['dot_with_reference'] for item in head)
        head_comparison = dict(reference_RSS=math.sqrt(aa), current_RSS=math.sqrt(bb),
            norm_ratio=math.sqrt(bb / aa), cosine=ab / math.sqrt(aa * bb),
            difference_RSS=math.sqrt(max(0., aa + bb - 2 * ab)),
            scope='Stack of the16 native rank-local B4 BF16 adjoints under the original local backward scales, not a globally reduced hidden tensor or an operator norm.')
    rows.append(dict(case=name, actual_global_native_PG_norm=norm,
        B4_increment_RSS=rss, B4_direction_factor=norm / rss,
        Gram_reconstruction_difference=reconstructed - norm,
        native_head_adjoint_stack_RSS=native_head_rss, head_comparison=head_comparison,
        source_chains=[dict(path=item['path'], SHA256=item['SHA256']) for item in status['chain_reports'] if item['value']['iteration'] == name]))
before = rows[0]['actual_global_native_PG_norm']
for row in rows:
    row['ratio_to_same_session_original80'] = row['actual_global_native_PG_norm'] / before
    row['change_to_same_session_original80'] = row['ratio_to_same_session_original80'] - 1
bindings = status['state']['diagnostic_parameter_partition_bindings']
assert len(bindings) == 12
for binding in bindings:
    wanted = specs[binding['case']]['diagnostic_parameter_partition']['rank_artifacts'][str(binding['rank'])]
    assert binding['artifact_SHA256'] == wanted['SHA256']
    assert binding['result_trainable_SHA256'] == wanted['expected_result_trainable_SHA256']
    assert binding['scheduler_before'] == binding['scheduler_after'] == 80
    assert binding['optimizer_steps_before'] == binding['optimizer_steps_after'] == [320.]
forward_bindings = status['forward_bindings']
assert len(forward_bindings) == 6
buffers = {}
for entry in forward_bindings:
    for binding in entry['bindings']:
        rank = binding['rank']
        if entry['iteration'] == 'old80_before':
            buffers[rank] = binding['native_buffers']
        else:
            assert binding['native_buffers'] == buffers[rank]
native_owner_readouts = []
for item in status['native_PG_reports']:
    actual = item['value']
    config = actual['effective_config']
    assert config['ppo_mini_batch_size'] == 32
    assert config['ppo_micro_batch_size_per_gpu'] == 4
    assert config['ppo_epochs'] == 1 and config['gradient_accumulation'] == 8
    assert config['native_backward_scale'] == .125 and config['loss_agg_mode'] == 'token-mean'
    assert actual['optimizer_step_executed'] is False
    assert actual['native_gradient_clipping_executed'] is True
    assert actual['sources']['actor']['sha256'].startswith('3a65e173')
    assert actual['sources']['core']['sha256'].startswith('fc2f992b')
    native_pass = actual['passes']['full_DT']
    assert native_pass['optimizer_boundary']['optimizer_step_no_op_calls'] == 1
    microbatches = native_pass['microbatch_losses']
    assert len(microbatches) == 8
    assert all(m['ppo_kl'] == m['pg_clipfrac'] == m['pg_clipfrac_lower'] == 0. for m in microbatches)
    native_owner_readouts.append(dict(path=item['path'], SHA256=item['SHA256'],
        effective_config=config, native_mean_PPO_KL_and_clip_scalars_exact_zero=True,
        optimizer_step_executed=False, native_gradient_clipping_executed=True))
assert len(native_owner_readouts) == 12
report = dict(recorded_at=datetime.now().astimezone().isoformat(),
    status='completed_actual_native_PG_at_saved_after_states_not_root_repair',
    job=status['job'], rows=rows, actual_native_inputs_equal=True, actual_native_buffers_equal=True,
    actual_parameter_bindings=bindings, native_owner_readouts=native_owner_readouts,
    source=dict(path=str(source), SHA256=hashlib.sha256(source.read_bytes()).hexdigest()),
    rootcause_repair_verified=False, optimizer_updates=0, DT_calls=0,
    limits=manifest['limits'] + [
        'The v90 fixed-adjoint values belong to a separate native execution with an observed forward-point discrepancy. Do not subtract those vectors or attribute exact causal fractions to the head change.',
        'These actual PGs use unchanged old rollout inputs/credits and reset rho1 at each saved model point. They do not reconstruct the next historical on-policy batch or prove a from-scratch trend.',
        'Same-point repeated values and Gram/adjoint arithmetic are reported as measured; no invented numerical acceptance tolerance or all-layer decline requirement.'])
path = HERE / 'completed-saved-update-natural-PG-v101.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(path.resolve()), rows=rows), ensure_ascii=False))
