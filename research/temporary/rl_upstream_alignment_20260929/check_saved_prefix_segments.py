"""Read the original capture plan, then check saved native FLA state segments.

The metadata recorder calls the unchanged lease factory and observes its
capture arguments. It never executes a model. Numerical work uses saved real
B4 operands, the original FLA fwd_h and the pinned original ht assertion.
There is no new cache, capture policy, model, tolerance or whole-DT gate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


LEASE_SHA = 'd5b539c7ffb8a431374cf79f4b995ef6f4138c3e6e289890409b6b79dbad393e'
ARTIFACT_SHA = '06fda7843c4120cb1406b671f06cb19f29921b61b5d524dc28539550446aef16'
FLA_TEST_SHA = '35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path, expected_sha):
    path = Path(path)
    if sha(path) != expected_sha:
        raise ValueError(f'Original owner source does not match: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_ht_assertion(path):
    path = Path(path)
    if sha(path) != FLA_TEST_SHA:
        raise ValueError('Original FLA test source SHA256 mismatch.')
    tree = ast.parse(path.read_bytes())
    test = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'test_chunk')
    statements = [n for n in test.body if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id == 'assert_close'
        and isinstance(n.value.args[0], ast.Constant)
        and n.value.args[0].value == 'ht']
    if len(statements) != 1:
        raise ValueError('Original ht assertion interface changed.')
    code = compile(ast.fix_missing_locations(ast.Module(body=statements,
        type_ignores=[])), str(path), 'exec')
    return code, [ast.unparse(n) for n in statements]


def read_capture_plan(torch, requests, geometry, component, lease_source,
                      artifact_source, rank):
    """Observe original factory metadata; dummy artifacts are never consumed."""
    owner = load_module('qwen35_native_prefix_artifacts', artifact_source, ARTIFACT_SHA)
    lease_owner = load_module('_saved_prefix_original_lease_owner', lease_source, LEASE_SHA)
    all_requests = sorted(requests['requests'], key=lambda r: r['context_tokens'])
    metadata = geometry['ranks'][rank]
    batches = metadata['batches']
    if len(all_requests) != component['complete_capture_bank_rows']:
        raise ValueError('Full original request bank is required.')
    position = 0
    captures = []

    def synchronize(value):
        nonlocal position
        batch = batches[position]
        position += 1
        if value != batch['local_common_prefix_boundary']:
            raise ValueError('Owner request order differs from recorded geometry.')
        # Consume the original recorded cross-rank MIN, rather than replacing
        # its algorithm or making a rank-local choice during this offline read.
        return batch['frozen_v2_cross_rank_MIN_prefix']

    def record_capture(model, input_ids, prefix_lengths, *, config, **kwargs):
        frame = inspect.currentframe().f_back
        # unittest.mock invokes this passive recorder through its own frames.
        # Find the unchanged factory frame; do not reproduce its grouping.
        while frame is not None and frame.f_code is not lease_owner.prepare_native_prefix_leases.__code__:
            frame = frame.f_back
        if frame is None:
            raise ValueError('Metadata did not come from the original factory.')
        local = frame.f_locals
        artifact = owner.NativePrefixArtifacts(config, input_ids.detach().cpu(), [])
        captures.append(dict(artifact=artifact, prefix_lengths=list(prefix_lengths),
            capture_index=len(captures), canonical_indices=list(local['rows']),
            row_uids=[local['records'][row][0] for row in local['rows']],
            input_shape=list(input_ids.shape)))
        return artifact

    model = SimpleNamespace(execution_device='cpu',
        lm_head=SimpleNamespace(weight=torch.zeros(1)),
        _conditional=SimpleNamespace(config=None), synchronize_prefix_start=synchronize)
    # Only this CPU metadata call suppresses its final CUDA completion fence.
    # The GPU kernel/reference phase below never patches Torch or FLA.
    with patch.object(owner.NativePrefixArtifacts, 'capture', side_effect=record_capture), \
         patch.object(torch.cuda, 'synchronize', return_value=None):
        leases, preparation = lease_owner.prepare_native_prefix_leases(
            SimpleNamespace(model=model), all_requests, minibatch_size=4,
            eos_token_id=requests['eos_token_id'])
    if position != len(batches):
        raise ValueError('Original capture metadata did not consume all batches.')
    batch_index = component['consumer_batch_index']
    request = all_requests[component['request_index']]
    lease = leases[batch_index]
    source, source_row = lease.sources[component['local_row']]
    selected = next(c for c in captures if c['artifact'] is source)
    if source_row != component['source_row'] or request['traj_uid'] != component['traj_uid']:
        raise ValueError('Recorded source row/UID differs from original factory.')
    if lease.prefix_length != component['prefix']:
        raise ValueError('Recorded prefix differs from original factory.')
    if selected['input_shape'] != component['capture_input_shape']:
        raise ValueError('Recorded original full capture B4 shape differs.')
    if selected['row_uids'][source_row] != component['traj_uid']:
        raise ValueError('Source UID is not at the recorded complete B4 row.')
    if not torch.equal(source.input_ids[source_row, :lease.prefix_length],
                       request['prompt'][:lease.prefix_length]):
        raise ValueError('Exact factual prefix IDs do not match.')
    selected.pop('artifact')
    ids = source.input_ids.contiguous()
    selected.update(prefix=lease.prefix_length, source_row=source_row,
        request_index=component['request_index'], traj_uid=request['traj_uid'],
        full_capture_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        preparation=preparation, total_capture_calls=len(captures),
        model_forwards=0, metadata_only=True)
    return selected


def run(args, save):
    import torch
    import torch.nn.functional as F
    import fla.utils as fla_owner
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    if fla_owner.FLA_CI_ENV:
        raise AssertionError('Original FLA CI warning exemption must be disabled.')
    raw_component = json.loads(args.component_report.read_bytes())
    component = (raw_component['records'][f'rank{args.rank}.json']['value']
                 if 'records' in raw_component else raw_component)
    geometry = json.loads(args.geometry_report.read_bytes())
    request_source = geometry['ranks'][args.rank]
    if sha(args.requests) != request_source['sha256'] or \
            sha(args.requests) != component['original_request_sha256']:
        raise ValueError('Original full request bank SHA256 mismatch.')
    recorded = next(row for row in component['first_operator_checks']
                    if row['case'] == 'long_readout')['saved_actual_operands']
    if sha(args.operands) != recorded['sha256']:
        raise ValueError('Original long-source operands SHA256 mismatch.')
    requests = torch.load(args.requests, map_location='cpu', weights_only=False)
    saved = torch.load(args.operands, map_location='cpu', weights_only=True, mmap=True)
    if saved['case'] != 'long_readout' or saved['layer'] != component['gdn_layer_index']:
        raise ValueError('Operands do not describe the recorded original long source.')
    save('reading_original_capture_plan', requests_sha256=sha(args.requests),
         operands_sha256=sha(args.operands))
    plan = read_capture_plan(torch, requests, geometry, component,
                             args.lease_source, args.artifact_source, args.rank)
    boundaries = [n for n in plan['prefix_lengths'] if n <= plan['prefix']]
    if not boundaries or boundaries[-1] != plan['prefix']:
        raise ValueError('Captured target boundary is absent from the original plan.')
    if len(boundaries) < 2:
        raise ValueError('This original B4 does not exercise an interior state segment.')
    tensors = saved['tensors']
    for name, meta in recorded['tensors'].items():
        actual = tensors[name]
        if list(actual.shape) != meta['shape'] or str(actual.dtype) != meta['dtype']:
            raise ValueError(f'Recorded actual tensor metadata differs: {name}')
    if tensors['q'].shape[0] != 4 or tensors['q'].shape[1] != plan['prefix']:
        raise ValueError('The full original B4 prefix operands are required.')
    save('original_capture_plan_read', capture_plan=plan, boundaries_checked=boundaries)
    previous_state, previous_length = None, 0
    segment_records = []
    final_initial = None
    for boundary in boundaries:
        save('original_fwd_h_segment_start', segment_start=previous_length,
             segment_end=boundary, initial_state_dtype=(
                 None if previous_state is None else str(previous_state.dtype)))
        tick = time.perf_counter()
        fields = {name: tensors[record_name][:, previous_length:boundary]
                  .to(args.device).contiguous()
                  for name, record_name in (('k', 'stage_k'), ('w', 'stage_w'),
                                            ('u', 'u'), ('g', 'stage_g'))}
        initial = previous_state
        h, v_new, state = chunk_gated_delta_rule_fwd_h(**fields,
            initial_state=initial, output_final_state=True, save_new_value=False)
        del h, v_new, fields
        torch.cuda.synchronize()
        segment_records.append(dict(start=previous_length, end=boundary,
            initial_state_dtype=None if initial is None else str(initial.dtype),
            final_state_dtype=str(state.dtype), seconds=time.perf_counter()-tick,
            original_cumulative_stage_g_used=True))
        if boundary == plan['prefix']:
            final_initial = initial
            final_start = previous_length
        previous_state, previous_length = state, boundary
    save('original_fwd_h_segments_complete', segments=segment_records)
    assertion, original_statements = original_ht_assertion(args.official_source)
    reference = load_module('_saved_prefix_official_fla_reference',
                            args.official_source, FLA_TEST_SHA)
    raw = {name: tensors[name][:, final_start:plan['prefix']].to(args.device)
           for name in ('q', 'k', 'v', 'beta', 'g')}
    # The saved diagnostic explicitly records raw_q/raw_k; its original
    # reference applies ordinary normalization once. This is an ht-only check:
    # the original reference scale multiplies q, which contributes only to o.
    # It does not affect the recurrent state update. Saved data has no scale
    # field, so preserve the original reference default instead of guessing it.
    q, k = F.normalize(raw['q'], p=2, dim=-1), F.normalize(raw['k'], p=2, dim=-1)
    save('original_recurrent_reference_start', segment_start=final_start,
         segment_end=plan['prefix'], reference_initial_state_dtype=str(final_initial.dtype),
         reference_scale='original_default_None_state_is_scale_independent',
         normalization_applied_once=True, output_o_checked=False)
    with torch.no_grad():
        unused_o, ref_ht = reference.recurrent_gated_delta_rule_ref(
            q=q, k=k, v=raw['v'], beta=raw['beta'], g=raw['g'],
            initial_state=final_initial, output_final_state=True)
        del unused_o
        exec(assertion, dict(assert_close=fla_owner.assert_close,
                            ref_ht=ref_ht, tri_ht=previous_state))
    torch.cuda.synchronize()
    save('original_ht_assertion_passed', original_assertions=original_statements,
         official_source=dict(path=str(args.official_source), sha256=sha(args.official_source)),
         state_error_ratio=float(fla_owner.get_err_ratio(ref_ht, previous_state)))
    # Let the actual HF layer own its storage dtype and copy, as in the bank.
    # This is a representation observation, not a new numerical acceptance gate.
    from transformers.cache_utils import LinearAttentionLayer
    cache_layer = LinearAttentionLayer()
    local_row, source_row = component['local_row'], plan['source_row']
    cache_layer.update_conv_state(tensors['prepared_prefix_conv_states'][local_row:local_row+1]
                                  .to(args.device))
    actual_stored = cache_layer.update_recurrent_state(previous_state[source_row:source_row+1])
    recorded_stored = tensors['prepared_prefix_recurrent_states'][local_row:local_row+1].to(args.device)
    save('complete', status='passed_original_interior_segment_ht_assertion',
         native_raw_final_state_dtype=str(previous_state.dtype),
         reference_final_state_dtype=str(ref_ht.dtype),
         hf_storage_observation=dict(actual_dtype=str(actual_stored.dtype),
            recorded_dtype=str(recorded_stored.dtype),
            equal=bool(torch.equal(actual_stored, recorded_stored)),
            maximum_absolute_difference=float((actual_stored.float()-recorded_stored.float()).abs().max()),
            acceptance_threshold=None), model_forwards=0, whole_dt_acceptance=False,
         original_operator_source=dict(path=inspect.getsourcefile(chunk_gated_delta_rule_fwd_h),
            sha256=sha(inspect.getsourcefile(chunk_gated_delta_rule_fwd_h))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('requests', 'operands', 'component-report', 'geometry-report',
                 'lease-source', 'artifact-source', 'official-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = dict(scope=__doc__, rank=args.rank, started_unix=time.time(),
        phases=[], script_source=dict(path=__file__, sha256=sha(__file__)),
        provenance={name:dict(path=str(getattr(args,name)), sha256=sha(getattr(args,name)))
                    for name in ('requests','operands','component_report','geometry_report',
                                 'lease_source','artifact_source','official_source')})
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)

    def save(phase, **values):
        result.update(phase=phase, observed_unix=time.time(), **values)
        result['phases'].append(dict(phase=phase, observed_unix=result['observed_unix']))
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
        print(json.dumps(dict(rank=args.rank, phase=phase, **values)), flush=True)
    try:
        run(args, save)
    except Exception as error:
        save('failed', status='failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
