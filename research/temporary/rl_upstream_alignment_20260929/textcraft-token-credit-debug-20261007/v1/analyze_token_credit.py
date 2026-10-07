"""CPU-only statistics of saved original DT/readout and pre-update tensors.

This program never evaluates a model, DT, Q/V credit, or whitening. It selects
and reports saved values. Cross-stage links require the original artifact's
full IDs/masks, retained positions, and an exact token-ID check. Multiple
matches remain multiple matches. Only load trusted captures from this task.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch


CREDIT = ('dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates')


def scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def plain(value):
    if isinstance(value, torch.Tensor):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    return scalar(value)


def tensor(value, *, dtype=None):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('map_location=cpu did not produce CPU tensors')
        return value.detach().to(dtype=dtype) if dtype is not None else value.detach()
    return torch.as_tensor(value, dtype=dtype, device='cpu')


def file_source(path, payload):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), bytes=path.stat().st_size,
                pid=payload.get('pid'), captured_unix=payload.get('captured_unix'),
                provenance=plain(payload.get('provenance')), scope=payload.get('scope'))


def signature(prompt, response, policy, target):
    result = tuple(tuple(plain(value)) for value in (prompt, response, policy, target))
    if not (len(result[1]) == len(result[2]) == len(result[3])):
        raise ValueError('full response IDs and original policy/target masks differ in length')
    return result


def signature_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode('utf-8')).hexdigest()


def full_artifact_signature(saved):
    """Read complete artifact axes from saved IDs/masks, before causal packing."""
    original = saved['row']
    ids, responses = tensor(original['input_ids']), tensor(original['responses'])
    attention = tensor(original['attention_mask']).bool()
    boundary = ids.numel() - responses.numel()
    suffix = attention[boundary:].nonzero().flatten()
    return signature(ids[:boundary][attention[:boundary]], responses[suffix],
                     tensor(saved['policy'])[suffix], tensor(saved['target'])[suffix]), suffix


class Moments:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.tokens = self.finite = self.nan = self.posinf = self.neginf = 0
        self.negative = self.zero = self.positive = 0
        self.total = self.sumsq = 0.0
        self.extrema = {}
        self.nonfinite_examples = {}
        self.tails = {str(x): dict(abs_ge=0, negative_le=0, positive_ge=0)
                      for x in thresholds}

    def add(self, values, slots, position):
        values, slots = tensor(values).flatten(), tensor(slots, dtype=torch.long).flatten()
        if values.numel() != slots.numel():
            raise ValueError('statistic values and original slots differ in length')
        self.tokens += values.numel()
        for name, test in [('nan', torch.isnan(values)),
                           ('posinf', torch.isposinf(values)),
                           ('neginf', torch.isneginf(values))]:
            count = int(test.sum())
            setattr(self, name, getattr(self, name) + count)
            if count and name not in self.nonfinite_examples:
                index = int(test.nonzero()[0])
                self.nonfinite_examples[name] = dict(position(int(slots[index])),
                                                     value=scalar(values[index]))
        keep = torch.isfinite(values)
        values, slots = values[keep], slots[keep]
        self.finite += values.numel()
        if not values.numel():
            return
        double = values.double()
        self.total += float(double.sum())
        self.sumsq += float(double.square().sum())
        self.negative += int((values < 0).sum())
        self.zero += int((values == 0).sum())
        self.positive += int((values > 0).sum())
        for threshold in self.thresholds:
            counts = self.tails[str(threshold)]
            counts['abs_ge'] += int((values.abs() >= threshold).sum())
            counts['negative_le'] += int((values <= -threshold).sum())
            counts['positive_ge'] += int((values >= threshold).sum())
        for name, scores, minimize in [('min', values, True), ('max', values, False),
                                       ('max_abs', values.abs(), False)]:
            index = int(scores.argmin() if minimize else scores.argmax())
            score, value = float(scores[index]), float(values[index])
            ties = int((scores == scores[index]).sum())
            old = self.extrema.get(name)
            better = old is None or (score < old['score'] if minimize else score > old['score'])
            if better:
                self.extrema[name] = dict(position(int(slots[index])), value=value,
                                          score=score, exact_tie_count=ties)
            elif score == old['score']:
                old['exact_tie_count'] += ties

    def result(self):
        output = dict(tokens=self.tokens, finite=self.finite, nan=self.nan,
                      positive_infinity=self.posinf, negative_infinity=self.neginf,
                      finite_negative=self.negative, finite_zero=self.zero,
                      finite_positive=self.positive, finite_sum=self.total,
                      finite_sumsq=self.sumsq, finite_extrema=self.extrema,
                      nonfinite_examples=self.nonfinite_examples, tail_counts=self.tails,
                      tie_policy='first saved file/row/original slot; exact ties counted')
        maximum = self.extrema.get('max_abs')
        output['minimum_non_NaN'] = (self.nonfinite_examples.get('neginf') or
                                     self.extrema.get('min') or
                                     self.nonfinite_examples.get('posinf'))
        output['NaN_prevents_an_ordered_minimum_over_all_values'] = bool(self.nan)
        output['one_max_abs_squared_over_finite_sumsq'] = (
            maximum['value'] * maximum['value'] / self.sumsq if maximum and self.sumsq else None)
        output['moment_accumulation_finite'] = math.isfinite(self.total) and math.isfinite(self.sumsq)
        return output


def readout_position(record, slot):
    saved, original = record['saved'], record['saved']['row']
    responses = tensor(original['responses']).flatten()
    ids = tensor(original['input_ids']).flatten()
    attention = tensor(original['attention_mask']).bool().flatten()
    mapping = tensor(saved['response_to_packed'], dtype=torch.long)
    packed = int(mapping[slot])
    prompt = int(saved['prompt_length'])
    absolute = ids.numel() - responses.numel() + slot
    native = saved.get('native_signed_packed')
    role = ('self_target_literal_boundary' if bool(saved['target'][slot]) else
            'estimated_prior_source' if bool(saved['prior'][slot]) and native is not None else
            'prior_source_without_saved_native_trace' if bool(saved['prior'][slot]) else
            'other_policy_causal_zero' if bool(saved['policy'][slot]) else 'masked')
    trace = saved.get('trace') or {}
    valid = attention.nonzero().flatten()
    effective = (valid == absolute).nonzero().flatten()
    ratios, outputs = saved.get('ratios'), saved.get('outputs', {})
    output = dict(record['identity'], response_slot=slot, original_input_slot=absolute,
                  packed_input_slot=packed if packed >= 0 else None,
                  attention_effective_index=int(effective[0]) if effective.numel() == 1 else None,
                  compressed_suffix_offset=packed - prompt if packed >= 0 else None,
                  dt_target_offset=packed - prompt if packed >= 0 and bool(saved['target'][slot]) else None,
                  target_predictor_slot=packed - 1 if packed >= 0 and bool(saved['target'][slot]) else None,
                  token_id=int(responses[slot]), input_token_id=int(ids[absolute]),
                  attention_valid=bool(attention[absolute]), d_role=role,
                  consumed_d_storage=scalar(ratios[slot]) if ratios is not None else None,
                  consumed_d_dtype=str(ratios.dtype) if ratios is not None else None,
                  native_signed=scalar(native[packed]) if native is not None and 0 <= packed < native.numel() else None,
                  native_signed_dtype=str(native.dtype) if native is not None else None,
                  raw_QVA={key: scalar(outputs[key][slot]) for key in CREDIT if key in outputs},
                  joint_factual_target_logp=trace.get('factual_target_logp'),
                  joint_all_prior_EOS_reference_target_logp=trace.get('reference_target_logp'),
                  joint_root_effect=trace.get('root_effect'),
                  logp_scope='saved joint endpoint sums; not single-token deletion logp',
                  value_scope=('native trace only; no consumed FP32 d or Q/V/A saved'
                               if record.get('native_only') else 'completed original readout'),
                  native_signed_role=('native trace component, not literal self-target deletion d'
                                      if bool(saved['target'][slot]) else 'original native trace component'))
    if packed >= 0:
        selected = tensor(saved['selected']).flatten()
        output['axis_token_ID_checks'] = dict(
            response_equals_original_input=bool(responses[slot] == ids[absolute]),
            response_equals_selected=bool(responses[slot] == selected[packed]),
            valid_attention_index_equals_original_input=bool(valid[packed] == absolute))
    return output


def add_stat(table, key, values, slots, position, thresholds):
    if key not in table:
        table[key] = Moments(thresholds)
    table[key].add(values, slots, position)


def validate_readout(saved, *, native_only=False):
    for key in ('row', 'selected', 'suffix_positions', 'response_to_packed', 'prompt_length',
                'valid_policy', 'valid_target', 'policy', 'target', 'prior',
                'group_masks', 'target_offsets') + (() if native_only else ('ratios', 'outputs')):
        if key not in saved:
            raise ValueError('missing saved readout field ' + key)
    original = saved['row']
    for key in ('input_ids', 'responses', 'attention_mask'):
        if key not in original:
            raise ValueError('missing original row field ' + key)
    responses, selected = tensor(original['responses']), tensor(saved['selected'])
    if responses.ndim != 1 or selected.ndim != 1:
        raise ValueError('readout IDs are not original one-dimensional rows')
    width, prompt = responses.numel(), int(saved['prompt_length'])
    for key in ('policy', 'target', 'prior', 'response_to_packed') + (() if native_only else ('ratios',)):
        if tensor(saved[key]).shape != responses.shape:
            raise ValueError(key + ' does not preserve original response slots')
    for key in (() if native_only else CREDIT):
        if key not in saved['outputs'] or tensor(saved['outputs'][key]).shape != responses.shape:
            raise ValueError('missing or misaligned original ' + key)
    for key, mask in saved['group_masks'].items():
        if tensor(mask).shape != responses.shape:
            raise ValueError('misaligned original group mask ' + key)
    suffix = tensor(saved['suffix_positions'], dtype=torch.long)
    if suffix.ndim != 1 or bool(((suffix < 0) | (suffix >= width)).any()):
        raise ValueError('saved suffix_positions is outside original responses')
    mapping = tensor(saved['response_to_packed'], dtype=torch.long)
    if not torch.equal(mapping[suffix], torch.arange(suffix.numel()) + prompt):
        raise ValueError('saved packed/response mapping is inconsistent')
    if not torch.equal(responses[suffix], selected[prompt:]):
        raise ValueError('saved packed/response token IDs differ')
    ids, attention = tensor(original['input_ids']), tensor(original['attention_mask']).bool()
    effective = ids[attention]
    full_suffix = attention[-width:].nonzero().flatten()
    if (ids.ndim != 1 or attention.shape != ids.shape or
            not torch.equal(effective[:selected.numel()], selected) or
            not torch.equal(full_suffix[:suffix.numel()], suffix) or
            prompt != int(attention[:-width].sum())):
        raise ValueError('saved selected IDs/slots are not an exact original attention-effective prefix')
    for key in ('valid_policy', 'valid_target'):
        if tensor(saved[key]).shape != suffix.shape:
            raise ValueError(key + ' does not align with attention-effective response IDs')
    native = saved.get('native_signed_packed')
    if native is not None and (native.ndim != 1 or native.numel() < selected.numel()):
        raise ValueError('saved original signed vector does not cover selected input')


def native_batch_record(saved, native, trace):
    """Adapt the v2 native-return snapshot schema; do not synthesize credit."""
    saved = dict(saved, native_signed_packed=native, trace=trace)
    suffix = tensor(saved['suffix_positions'], dtype=torch.long)
    mapping = torch.full((saved['width'],), -1, dtype=torch.long)
    mapping[suffix] = torch.arange(suffix.numel()) + saved['prompt_length']
    saved.update(response_to_packed=mapping,
                 valid_policy=tensor(saved['policy'])[suffix],
                 valid_target=tensor(saved['target'])[suffix],
                 group_masks=dict(self_target=saved['target'], prior_source=saved['prior'],
                                  other_policy=saved['policy'] & ~saved['target'] & ~saved['prior'],
                                  masked=~saved['policy'], all_slots=torch.ones_like(saved['policy'])))
    return saved


def add_native_statistics(record, table, thresholds):
    saved = record['saved']
    native, mapping = saved['native_signed_packed'], tensor(saved['response_to_packed'], dtype=torch.long)
    add_stat(table, 'native_signed/all_packed_input', native, torch.arange(native.numel()),
             lambda slot: packed_position(record, slot), thresholds)
    for group, mask in dict(saved['group_masks'], policy=saved['policy']).items():
        slots = (tensor(mask).bool() & (mapping >= 0)).nonzero().flatten()
        add_stat(table, 'native_signed/' + group, native[mapping[slots]], slots,
                 lambda slot: readout_position(record, slot), thresholds)


def packed_position(record, packed):
    saved = record['saved']
    mapping = tensor(saved['response_to_packed'], dtype=torch.long)
    matches = (mapping == packed).nonzero().flatten()
    if matches.numel() == 1:
        return readout_position(record, int(matches[0]))
    selected = tensor(saved['selected'])
    valid = tensor(saved['row']['attention_mask']).bool().nonzero().flatten()
    trace = saved.get('trace') or {}
    return dict(record['identity'], packed_input_slot=packed,
                original_input_slot=int(valid[packed]) if packed < selected.numel() else None,
                attention_effective_index=packed if packed < selected.numel() else None,
                response_slot=None, compressed_suffix_offset=None, dt_target_offset=None,
                token_id=int(selected[packed]) if packed < selected.numel() else None,
                d_role='prompt_or_packed_padding; not an estimated policy-source d',
                native_signed=scalar(saved['native_signed_packed'][packed]),
                native_signed_dtype=str(saved['native_signed_packed'].dtype),
                joint_factual_target_logp=trace.get('factual_target_logp'),
                joint_all_prior_EOS_reference_target_logp=trace.get('reference_target_logp'),
                logp_scope='saved joint endpoint sums; not single-token deletion logp')


def analyze(directory, thresholds, expected_ranks=(0, 1)):
    files = (sorted(directory.glob('rank*-readout.pt')) + sorted(directory.glob('rank*-pre-update.pt')) +
             sorted(directory.glob('rank*-readout-native-batch-*.pt')))
    result = dict(scope='CPU analysis of actual saved original tensors only', analyzed_unix=time.time(),
                  input_directory=str(directory.resolve()), sources=[], capture_metadata=[],
                  errors=[], missing_inputs=[],
                  tail_thresholds=thresholds,
                  tail_scope='descriptive counts only; no clipping, acceptance threshold, or gradient estimate',
                  old_step7_vector_status='No original step7 vectors were saved; scalar/group receipts cannot recover token locations or d. These captures are not step7 replay.',
                  d_scope='native FP64 signed and training-consumed FP32 d are separate; self-target zero is a storage placeholder, not a finite measured d',
                  second_moment_scope='uncentered finite-token second moment of saved rows, including actual DP duplicates; not gradient share',
                  readout_rows=[], actor_rows=[], actor_extreme_links=[], native_batch_rows=[],
                  native_batch_scope='separate original native-return snapshots, possibly also present in completed readout; never added to readout/actor totals; no Q/V/A or consumed FP32 d inferred')
    expected = [f'rank{rank}-{suffix}.pt' for rank in expected_ranks
                for suffix in ('readout', 'pre-update')]
    existing = {path.name for path in files}
    result['expected_capture_files'] = expected
    result['missing_inputs'] = [name for name in expected if name not in existing]
    result['all_expected_files_present'] = not result['missing_inputs']
    payloads = []
    for path in files:
        try:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            source = file_source(path, payload)
            result['sources'].append(source)
            result['capture_metadata'].append(dict(file=path.name,
                missing_tensor_fields=payload.get('missing_tensor_fields', []),
                missing_non_tensor_fields=payload.get('missing_non_tensor_fields', []),
                observer_errors=plain(payload.get('observer_errors', [])),
                original_readout_report=plain(payload.get('report'))))
            payloads.append((path, payload, source))
        except Exception as error:
            result['errors'].append(dict(path=str(path), stage='load', error=repr(error)))
    records, lookup, readout_stats, actor_stats, native_stats = [], defaultdict(list), {}, {}, {}
    for path, payload, source in payloads:
        if '-readout-native-batch-' not in path.name:
            continue
        try:
            native = tensor(payload['native_signed'])
            if native.ndim != 2:
                raise ValueError('native batch signed tensor must retain original batch and packed axes')
            detail = payload.get('detail') or {}
            for index, saved in enumerate(payload.get('rows', [])):
                identity = dict(file=path.name, file_sha256=source['sha256'], saved_row=index,
                                batch=payload.get('batch'), batch_row=saved['batch_row'],
                                trajectory_index=saved.get('trajectory_index'), traj_uid=str(saved.get('traj_uid', '')))
                try:
                    batch_row = int(saved['batch_row'])
                    if not 0 <= batch_row < native.shape[0]:
                        raise ValueError('saved batch_row is outside original native batch')
                    samples = detail.get('per_sample')
                    trace = (samples[batch_row] if isinstance(samples, (list, tuple)) and batch_row < len(samples) else
                             detail if native.shape[0] == 1 and samples is None else None)
                    saved = native_batch_record(saved, native[batch_row], trace)
                    validate_readout(saved, native_only=True)
                    sig, full_suffix = full_artifact_signature(saved)
                    record = dict(saved=saved, identity=identity, signature=sig,
                                  full_suffix_positions=full_suffix, native_only=True)
                    result['native_batch_rows'].append(dict(identity, full_artifact_sha256=signature_sha(sig),
                        native_signed_dtype=str(native.dtype), target_offsets=plain(saved['target_offsets']),
                        trace=plain(trace), available=dict(native_signed=True, consumed_FP32_d=False,
                                                          QVA=False, actor=False)))
                    add_native_statistics(record, native_stats, thresholds)
                except Exception as error:
                    result['errors'].append(dict(identity, stage='native_batch_row', error=repr(error)))
        except Exception as error:
            result['errors'].append(dict(file=path.name, stage='native_batch', error=repr(error)))
    for path, payload, source in payloads:
        if not path.name.endswith('-readout.pt'):
            continue
        for index, saved in enumerate(payload.get('rows', [])):
            identity = dict(file=path.name, file_sha256=source['sha256'], saved_row=index,
                            trajectory_index=saved.get('trajectory_index'), traj_uid=str(saved.get('traj_uid', '')))
            try:
                validate_readout(saved)
                sig, full_suffix = full_artifact_signature(saved)
                record = dict(saved=saved, identity=identity, signature=sig, full_suffix_positions=full_suffix)
                records.append(record)
                lookup[(identity['traj_uid'], sig)].append(record)
                result['readout_rows'].append(dict(identity, full_artifact_sha256=signature_sha(sig),
                    native_signed_dtype=saved.get('native_signed_dtype'), ratios_dtype=saved.get('ratios_dtype'),
                    target_offsets=plain(saved['target_offsets']), trace=plain(saved.get('trace'))))
                groups = dict(saved['group_masks'], policy=saved['policy'])
                vectors = dict(consumed_d_storage=saved['ratios'], **saved['outputs'])
                native = saved.get('native_signed_packed')
                mapping = tensor(saved['response_to_packed'], dtype=torch.long)
                for name, values in vectors.items():
                    for group, mask in groups.items():
                        slots = tensor(mask).bool().nonzero().flatten()
                        add_stat(readout_stats, name + '/' + group, values[slots], slots,
                                 lambda slot, record=record: readout_position(record, slot), thresholds)
                if native is not None:
                    add_native_statistics(record, readout_stats, thresholds)
                    slots = tensor(saved['prior']).bool().nonzero().flatten()
                    add_stat(readout_stats, 'consumed_d_estimated/prior_source', saved['ratios'][slots], slots,
                             lambda slot, record=record: readout_position(record, slot), thresholds)
            except Exception as error:
                result['errors'].append(dict(identity, stage='readout_row', error=repr(error)))

    actor_identities = []
    actor_contexts = {}
    for path, payload, source in payloads:
        if not path.name.endswith('-pre-update.pt'):
            continue
        batch, metadata = payload.get('tensors', {}), payload.get('non_tensors', {})
        if 'responses' not in batch or 'response_mask' not in batch:
            result['errors'].append(dict(file=path.name, stage='actor_fields',
                                         missing=[key for key in ('responses', 'response_mask') if key not in batch]))
            continue
        responses, policy = tensor(batch['responses']), tensor(batch['response_mask']).bool()
        if responses.shape != policy.shape or responses.ndim != 2:
            result['errors'].append(dict(file=path.name, stage='actor_shapes',
                                         responses=list(responses.shape), response_mask=list(policy.shape)))
            continue
        invalid = [key for key in (*CREDIT, 'advantages')
                   if key in batch and tensor(batch[key]).shape != responses.shape]
        if invalid:
            result['errors'].append(dict(file=path.name, stage='actor_shapes', invalid_fields=invalid))
            continue
        uids, artifacts = metadata.get('traj_uid'), metadata.get('dt_direct_target_artifact')
        for index in range(responses.shape[0]):
            identity = dict(file=path.name, file_sha256=source['sha256'], saved_row=index,
                            traj_uid=str(uids[index]) if uids is not None else None)
            actor_identities.append(identity)
            context = dict(identity=identity, batch=batch, row=index, candidates=[], artifact=None,
                           retained=None, matches='missing original artifact or traj_uid')
            try:
                if artifacts is not None and uids is not None:
                    artifact = artifacts[index]
                    sig = signature(*(artifact[key] for key in ('prompt_ids', 'response_ids', 'policy_mask', 'target_mask')))
                    context['artifact'] = artifact
                    context['retained'] = tensor(artifact['retained_response_positions'], dtype=torch.long)
                    if context['retained'].shape != responses[index].shape:
                        raise ValueError('retained_response_positions does not match original actor response width')
                    context['candidates'] = lookup.get((identity['traj_uid'], sig), [])
                    context['signature_sha256'] = signature_sha(sig)
                    context['matches'] = ('unique full IDs/masks and UID match' if len(context['candidates']) == 1 else
                                          'multiple full IDs/masks and UID matches; none selected' if context['candidates'] else
                                          'no full IDs/masks and UID match; no d inferred')
                kept = context['retained'] >= 0 if context['retained'] is not None else torch.zeros_like(policy[index])
                valid = torch.zeros_like(kept)
                if context['artifact'] is not None:
                    original = tensor(context['artifact']['response_ids'])
                    within = kept & (context['retained'] < original.numel())
                    slots = within.nonzero().flatten()
                    valid[slots] = responses[index, slots] == original[context['retained'][slots]]
                context['valid_retained'] = valid
                group_masks = dict(policy=policy[index], masked=~policy[index],
                                   all_slots=torch.ones_like(policy[index]),
                                   self_target=torch.zeros_like(policy[index]),
                                   prior_source=torch.zeros_like(policy[index]),
                                   other_policy=torch.zeros_like(policy[index]))
                slots = (valid & policy[index]).nonzero().flatten()
                if slots.numel():
                    original_slots = context['retained'][slots]
                    if context['candidates']:
                        roles = []
                        for candidate in context['candidates']:
                            saved = candidate['saved']
                            response_slots = candidate['full_suffix_positions'][original_slots]
                            roles.append(torch.stack([tensor(saved[name]).bool()[response_slots]
                                                      for name in ('target', 'prior', 'policy')]))
                        agree = torch.ones(slots.numel(), dtype=torch.bool)
                        for role in roles[1:]:
                            agree &= (role == roles[0]).all(0)
                        target, prior, native_policy = roles[0]
                        agree &= native_policy
                        group_masks['self_target'][slots[agree & target]] = True
                        group_masks['prior_source'][slots[agree & prior & ~target]] = True
                        group_masks['other_policy'][slots[agree & ~target & ~prior]] = True
                    else:
                        target = tensor(context['artifact']['target_mask']).bool()[original_slots]
                        group_masks['self_target'][slots[target]] = True
                assigned = group_masks['self_target'] | group_masks['prior_source'] | group_masks['other_policy']
                group_masks['policy_without_saved_source_group'] = policy[index] & ~assigned
                actor_contexts[(path.name, index)] = context
                result['actor_rows'].append(dict(identity, matching_status=context['matches'],
                    full_artifact_sha256=context.get('signature_sha256'),
                    matching_readout_rows=[candidate['identity'] for candidate in context['candidates']],
                    valid_retained_token_IDs=int(valid.sum()),
                    retained_token_ID_mismatches=int((kept & ~valid).sum()),
                    synthetic_or_padding_positions=int((~kept).sum()),
                    policy_tokens=int(policy[index].sum())))
                for name in (*CREDIT, 'advantages'):
                    if name not in batch:
                        continue
                    values = tensor(batch[name])
                    if values.shape != responses.shape:
                        raise ValueError(name + ' does not match original actor responses shape')
                    for group, mask in group_masks.items():
                        slots = mask.nonzero().flatten()
                        add_stat(actor_stats, name + '/' + group, values[index, slots], slots,
                                 lambda slot, context=context: actor_position(context, slot), thresholds)
            except Exception as error:
                result['errors'].append(dict(identity, stage='actor_row', error=repr(error)))
    result['readout_statistics'] = {key: value.result() for key, value in readout_stats.items()}
    result['native_batch_statistics'] = {key: value.result() for key, value in native_stats.items()}
    result['actor_statistics'] = {key: value.result() for key, value in actor_stats.items()}
    for name in ('advantages/policy', 'dt_token_advantages/policy'):
        stats = actor_stats.get(name)
        if stats and (stats.extrema or stats.nonfinite_examples):
            extreme = (stats.nonfinite_examples.get('neginf') or stats.extrema.get('min') or
                       stats.nonfinite_examples.get('posinf'))
            if extreme is None:
                continue
            context = actor_contexts[(extreme['file'], extreme['saved_row'])]
            result['actor_extreme_links'].append(dict(measure=name, **actor_link(context, extreme['response_slot'])))
    result['counts'] = dict(readout_work_rows=len(records),
                          readout_unique_UIDs=len({record['identity']['traj_uid'] for record in records}),
                          native_batch_work_rows=len(result['native_batch_rows']),
                          actor_work_rows=len(actor_identities),
                          actor_unique_UIDs=len({row['traj_uid'] for row in actor_identities if row['traj_uid'] is not None}),
                          repeated_readout_UID_counts={key: count for key, count in Counter(
                              record['identity']['traj_uid'] for record in records).items() if count > 1})
    return plain(result)


def actor_position(context, slot):
    batch, index = context['batch'], context['row']
    responses = batch['responses']
    absolute = batch['input_ids'].shape[-1] - responses.shape[-1] + slot if 'input_ids' in batch else None
    retained = context.get('retained')
    original_slot = int(retained[slot]) if retained is not None else None
    return dict(context['identity'], response_slot=slot, original_input_slot=absolute,
                full_native_response_slot=original_slot if original_slot is not None and original_slot >= 0 else None,
                token_id=int(responses[index, slot]),
                input_token_id=int(batch['input_ids'][index, absolute]) if absolute is not None else None,
                response_equals_original_input=bool(responses[index, slot] == batch['input_ids'][index, absolute]) if absolute is not None else None,
                retained_token_ID_verified=bool(context['valid_retained'][slot]) if context.get('valid_retained') is not None else False,
                actual_whitened_advantage=scalar(batch['advantages'][index, slot]) if 'advantages' in batch else None,
                raw_QVA={key: scalar(batch[key][index, slot]) for key in CREDIT if key in batch},
                matching_status=context['matches'])


def actor_link(context, slot):
    output = actor_position(context, slot)
    output['matching_readout_extremal_slot_records'] = []
    if not output['retained_token_ID_verified']:
        output['cross_stage_status'] = 'unverified retained token identity; no cross-stage value assigned'
        return output
    original_slot = output['full_native_response_slot']
    for candidate in context['candidates']:
        response_slot = int(candidate['full_suffix_positions'][original_slot])
        entry = readout_position(candidate, response_slot)
        entry['artifact_full_native_response_slot'] = original_slot
        entry['actor_token_ID_equals_readout'] = entry['token_id'] == output['token_id']
        entry['actor_raw_QVA_exact_equals_readout'] = {
            key: bool(torch.equal(context['batch'][key][context['row'], slot],
                                  candidate['saved']['outputs'][key][response_slot]))
            for key in CREDIT if key in context['batch']}
        output['matching_readout_extremal_slot_records'].append(entry)
    output['cross_stage_status'] = context['matches']
    entries = output['matching_readout_extremal_slot_records']
    output['exact_token_and_raw_QVA_match_candidates'] = sum(
        entry['actor_token_ID_equals_readout'] and
        len(entry['actor_raw_QVA_exact_equals_readout']) == len(CREDIT) and
        all(entry['actor_raw_QVA_exact_equals_readout'].values()) for entry in entries)
    output['no_candidate_selected_when_multiple'] = len(entries) > 1
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-ranks', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--tail-thresholds', type=float, nargs='+', default=[1., 5., 10., 50., 100.])
    args = parser.parse_args()
    thresholds = sorted(set(args.tail_thresholds))
    if any(not math.isfinite(value) or value <= 0 for value in thresholds):
        parser.error('tail thresholds must be positive finite descriptive magnitudes')
    if args.cpu_threads < 1:
        parser.error('cpu threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    output = args.output or args.input_dir / 'token-credit-analysis.json'
    result = analyze(args.input_dir, thresholds, args.expected_ranks)
    result['analyzer'] = dict(path=str(Path(__file__).resolve()),
                              sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output.resolve()), sources=len(result['sources']),
                          counts=result['counts'], missing_inputs=result['missing_inputs'],
                          errors=result['errors']), ensure_ascii=True))


if __name__ == '__main__':
    main()
