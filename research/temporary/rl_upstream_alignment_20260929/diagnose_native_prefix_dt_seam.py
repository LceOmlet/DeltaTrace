"""Consume native prefix artifacts in an isolated, otherwise unchanged DT.

No PPO, value model, new finite formula or production deployment. Compare
the same real endpoint IDs and weights; report raw differences and costs
without introducing a whole-DT numerical tolerance.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import types

import torch


def diagnose(runner, producer, inputs, out, save, *, cache_tensors):
    from deltatrace_credit import trace_token_attribution
    from native_prefix_artifacts_candidate import NativePrefixArtifacts

    path = Path(out) / 'qwen35_dense_finite_runner_candidate.py'
    spec = importlib.util.spec_from_file_location('_prepared_prefix_runner', path)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    samples = inputs['samples']
    assert len(samples) == 4 and runner.reuse_native_prefix
    length = max(len(sample['selected_input_ids']) for sample in samples)
    selected = torch.full((4, length), inputs['eos_token_id'], device='cuda', dtype=torch.long)
    reference = selected.clone()
    cases = []
    for row, sample in enumerate(samples):
        ids = torch.tensor(sample['selected_input_ids'], device='cuda', dtype=torch.long)
        selected[row, :ids.numel()] = ids
        reference[row, :ids.numel()] = ids
        reference[row, sample['source_start']:sample['source_end']] = inputs['eos_token_id']
        cases.append({'target_ids': ids[-1:].cpu(), 'prompt_length': ids.numel()-1})
    kwargs = dict(packed_answer_targets=producer.packed_answer_targets,
                  outcome_token_ids=inputs['outcome_token_ids'])
    original_attribute = runner.attribute
    original_forward = runner.model.forward_root
    original_prefix = {}

    def observe_prefix(**options):
        value = original_forward(**options)
        original_prefix.update(cache_tensors(value.past_key_values))
        return value

    save('native_dt_cold_start', endpoint_shape=list(selected.shape),
         numerical_scope=__doc__, candidate_owner_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    try:
        runner.model.forward_root = observe_prefix
        cold_signed, _, cold_detail = trace_token_attribution(runner, reference, selected, cases,
                                                            [[0]] * 4, **kwargs)
    finally:
        runner.model.forward_root = original_forward
    prefix = cold_detail['native_shared_prefix_length']
    assert prefix > 0
    factual_prefix = selected[:, :prefix]
    save('native_dt_warm_start', native_prefix=prefix,
         cold_dt_seconds=cold_detail['complete_attribution_seconds_with_diagnostics'])
    baseline, _, baseline_detail = trace_token_attribution(runner, reference, selected, cases,
                                                         [[0]] * 4, **kwargs)
    native_fields = {}
    start = time.perf_counter()
    shared = NativePrefixArtifacts.capture(runner.model, factual_prefix, [prefix],
        config=runner.model._conditional.config,
        observe_native_cache=lambda cache: native_fields.update(cache_tensors(cache)))
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter()-start
    composed = shared.materialize(prefix, device='cuda')
    composed_fields = cache_tensors(composed)
    same_capture = []
    repeat_native = []
    for key, native in native_fields.items():
        actual = composed_fields[key]
        same_capture.append(dict(layer=key[0], field=key[1], dtype=str(actual.dtype),
                                 shape=list(actual.shape), equal=bool(torch.equal(actual, native))))
        prior = original_prefix[key]
        repeat_native.append(dict(layer=key[0], field=key[1], equal=bool(torch.equal(actual, prior)),
            maximum_absolute_difference=float((actual.float()-prior.float()).abs().max())))
    # Same-forward data transfer and original HF conversion must preserve the
    # exact owner artifact. Repeated native computation is reported separately.
    assert all(item['equal'] for item in same_capture)
    del composed, composed_fields, native_fields, original_prefix
    provider_calls = []

    def provide(ids):
        assert torch.equal(ids, factual_prefix), 'Consume only the exact already prepared factual prefix'
        start = time.perf_counter()
        cache = shared.materialize(ids.shape[1], device=ids.device)
        provider_calls.append(dict(prefix=int(ids.shape[1]), batch=int(ids.shape[0]),
                                   host_seconds=time.perf_counter()-start))
        return cache

    save('provided_native_dt_start', capture_seconds=capture_seconds,
         same_capture_fields=same_capture, repeated_native_fields=repeat_native,
         warm_baseline_seconds=baseline_detail['complete_attribution_seconds_with_diagnostics'])
    measurements = []
    vectors = {'baseline': baseline, 'cold_baseline': cold_signed}
    try:
        runner.attribute = types.MethodType(candidate.Qwen35DenseFiniteRunner.attribute, runner)
        for iteration in range(2):
            signed, _, detail = trace_token_attribution(runner, reference, selected, cases, [[0]] * 4,
                prefix_cache_provider=provide, **kwargs)
            vectors[f'provided_{iteration}'] = signed
            difference = signed.double()-baseline.double()
            row = dict(iteration=iteration,
                observed_signed_equal=bool(torch.equal(signed, baseline)),
                max_signed_difference=float(difference.abs().max()),
                rms_signed_difference=float(difference.square().mean().sqrt()),
                baseline_endpoint_scores=[baseline_detail['target_logp0'], baseline_detail['target_logp1']],
                provided_endpoint_scores=[detail['target_logp0'], detail['target_logp1']],
                dt_seconds=detail['complete_attribution_seconds_with_diagnostics'],
                calls=detail['calls'], peak_torch_allocated_bytes=detail['peak_allocated'],
                physical_free_bytes=torch.cuda.mem_get_info()[0])
            assert not any(call['kind']=='native_shared_prefix' for call in detail['calls'])
            measurements.append(row)
            save('provided_native_dt_complete', iteration=iteration,
                 dt_seconds=row['dt_seconds'], signed_equal=row['observed_signed_equal'],
                 max_signed_difference=row['max_signed_difference'])
    finally:
        runner.attribute = original_attribute
    tensor_path = Path(out) / f"owner-prefix-vectors-rank{torch.distributed.get_rank()}.pt"
    torch.save(vectors, tensor_path)
    save('native_dt_seam_diagnostic_complete', baseline_calls=baseline_detail['calls'],
         baseline_dt_seconds=baseline_detail['complete_attribution_seconds_with_diagnostics'],
         provided_measurements=measurements, provider_calls=provider_calls,
         vectors=dict(path=str(tensor_path), sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest()),
         capture_seconds=capture_seconds, same_capture_fields=same_capture,
         repeated_native_fields=repeat_native)
