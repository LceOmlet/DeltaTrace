"""Opt-in, one-shot observation of existing TextCraft DT and actor inputs.

No model, credit, scatter, whitening, or optimizer implementation lives here.
The readout proxy retains the owner's exact prepared-item references until its
unchanged return. Native signed effects and training-consumed FP32 ratios are
different artifacts. Self-target ratios are storage placeholders, not d=0.
The actor proxy saves the original DataProto before its first update and can
suspend that SAME process; resuming continues the original bound method.
Importing this module installs nothing. Parent controls installation/resume.
"""
from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path
import time

import torch


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _binding(obj, name):
    had, previous = name in vars(obj), vars(obj).get(name)

    def restore():
        if had:
            setattr(obj, name, previous)
        elif name in vars(obj):
            delattr(obj, name)
    return restore


def _position(item, slot):
    row, width = item['row'], item['width']
    matches = (item['suffix_positions'] == slot).nonzero().flatten()
    offset = int(matches[0]) if matches.numel() else None
    packed = item['prompt_length'] + offset if offset is not None else None
    return dict(response_slot=slot,
                original_input_slot=row['input_ids'].numel() - width + slot,
                packed_input_slot=packed, compressed_suffix_offset=offset,
                target_predictor_slot=(packed - 1 if packed is not None and
                                       bool(item['target'][slot]) else None),
                token_id=int(row['responses'][slot]))


def _extrema(values, mask, item):
    slots = mask.nonzero().flatten()
    result = dict(tokens=int(slots.numel()), nonfinite=0,
                  tie_policy='first original response slot; tie_count retained')
    if not slots.numel():
        return result
    subset = values[slots]
    result['nonfinite'] = int((~torch.isfinite(subset)).sum())
    # Raw tensors retain all NaN/Inf. Summaries do not invent a failure gate.
    finite = torch.isfinite(subset)
    slots, subset = slots[finite], subset[finite]
    if not slots.numel():
        return result
    for name, scores, index in [('min', subset, int(subset.argmin())),
                                ('max', subset, int(subset.argmax())),
                                ('max_abs', subset.abs(), int(subset.abs().argmax()))]:
        entry = _position(item, int(slots[index]))
        entry.update(value=float(subset[index]), magnitude=float(subset[index].abs()),
                     tie_count=int((scores == scores[index]).sum()))
        result[name] = entry
    return result


def _row_record(item, output, trace, native):
    groups = dict(self_target=item['target'], prior_source=item['prior'],
                  other_policy=item['policy'] & ~item['target'] & ~item['prior'],
                  masked=~item['policy'], all_slots=torch.ones_like(item['policy']))
    values = dict(ratios=item['ratios'], **output)
    record = dict(trajectory_index=item['index'],
                  traj_uid=str(item['row'].get('traj_uid', '')),
                  alignment_scope='full_native_trajectory_before_actor_retained_scatter',
                  reward=item['reward'], width=item['width'],
                  prompt_length=item['prompt_length'],
                  row=_snapshot(item['row']), selected=_snapshot(item['selected']),
                  suffix_positions=_snapshot(item['suffix_positions']),
                  policy=_snapshot(item['policy']), target=_snapshot(item['target']),
                  prior=_snapshot(item['prior']), valid_policy=_snapshot(item['valid_policy']),
                  valid_target=_snapshot(item['valid_target']),
                  target_offsets=copy.deepcopy(item['target_offsets']),
                  case=_snapshot(item['case']),
                  ratios=_snapshot(item['ratios']), outputs=_snapshot(output),
                  ratios_dtype=str(item['ratios'].dtype),
                  d_scope='ratios is the actual FP32 owner-consumed response-slot vector',
                  self_target_branch=dict(ratios_value_role='zero storage placeholder only',
                                          literal_boundary='p_without_i(Y)=0; Q=r,V=0,A=r',
                                          finite_d_measured=False),
                  group_masks=_snapshot(groups), trace=_snapshot(trace),
                  native_signed_packed=_snapshot(native),
                  native_signed_dtype=str(native.dtype) if native is not None else None,
                  native_signed_scope='original trace output row, full packed/padded input axis',
                  extrema={key: {group: _extrema(value, mask, item)
                                 for group, mask in groups.items()}
                           for key, value in values.items()})
    # Compressed-slot mapping is captured without decoding or reconstructing IDs.
    record['response_to_packed'] = torch.full((item['width'],), -1, dtype=torch.long)
    record['response_to_packed'][item['suffix_positions']] = (
        torch.arange(item['suffix_positions'].numel()) + item['prompt_length'])
    return record


class ReadoutObservation:
    """Explicit installation; original return objects and owner exceptions survive."""
    def __init__(self, readout, destination, *, provenance=None):
        self.readout, self.destination = readout, Path(destination)
        self.provenance = _snapshot(provenance or {})
        self.items, self.native, self.errors = [], {}, []
        self.installed = self.restored = self.saved = False

    def restore(self):
        for restore in reversed(getattr(self, '_restores', [])):
            restore()
        self._restores = []
        self.restored = True

    def install(self):
        if self.installed:
            return self
        readout = self.readout
        prepare, trajectories = readout._prepare_row, readout.trajectories
        owner_func = inspect.unwrap(getattr(trajectories, '__func__', trajectories))
        namespace = owner_func.__globals__
        self._restores = [_binding(readout, '_prepare_row'), _binding(readout, 'trajectories')]

        def prepared(*args, **kwargs):
            item = prepare(*args, **kwargs)
            self.items.append(item)  # Exact original object, no copied preparation.
            return item

        def observed(*args, **kwargs):
            original_trace = namespace['trace_token_attribution']
            trace_installed = False

            def trace(*trace_args, **trace_kwargs):
                result = original_trace(*trace_args, **trace_kwargs)
                try:
                    cases = (trace_args[3] if len(trace_args) > 3
                             else trace_kwargs['target_case'])
                    identities = {id(item['case']): item for item in self.items}
                    for batch_row, case in enumerate(cases):
                        item = identities.get(id(case))
                        if item is not None:
                            self.native[id(item)] = _snapshot(result[0][batch_row])
                except Exception as error:
                    self.errors.append(dict(stage='native_signed_observation', error=repr(error)))
                return result

            try:
                namespace['trace_token_attribution'] = trace
                trace_installed = True
                output = trajectories(*args, **kwargs)
            except BaseException:
                self.items.clear()
                self.native.clear()
                raise
            finally:
                if trace_installed:
                    namespace['trace_token_attribution'] = original_trace
                self.restore()
            try:
                report = readout.last_report
                traces = {entry['trajectory_index']: entry for entry in report.get('traces', [])}
                payload = dict(scope='pure passive one-shot observation; no numeric acceptance claim',
                               pid=os.getpid(), captured_unix=time.time(), provenance=self.provenance,
                               report=_snapshot(report), observer_errors=self.errors,
                               rows=[_row_record(item, value, traces.get(item['index']),
                                                 self.native.get(id(item)))
                                     for item, value in zip(self.items, output)],
                               original_item_references_used=True,
                               original_output_returned_unchanged=True)
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, self.destination)
                self.saved = True
            except Exception as error:
                self.errors.append(dict(stage='save_readout', error=repr(error)))
            finally:
                self.items.clear()
                self.native.clear()
            return output

        readout._prepare_row, readout.trajectories = prepared, observed
        self.installed = True
        return self


def _actor_snapshot(data, provenance, raw_path):
    fields = ('dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates', 'advantages',
              'loss_mask', 'response_mask', 'input_ids', 'responses', 'attention_mask',
              'policy_mask', 'target_mask', 'token_level_rewards')
    non_tensors = ('uid', 'traj_uid', 'dt_direct_target_artifact')
    return dict(scope='original actor DataProto after native scatter and whole-batch whitening',
                pid=os.getpid(), captured_unix=time.time(), provenance=_snapshot(provenance),
                raw_readout_path=str(raw_path),
                tensors={key: _snapshot(data.batch[key]) for key in fields if key in data.batch},
                missing_tensor_fields=[key for key in fields if key not in data.batch],
                non_tensors={key: _snapshot(data.non_tensor_batch[key]) for key in non_tensors
                             if key in data.non_tensor_batch},
                missing_non_tensor_fields=[key for key in non_tensors if key not in data.non_tensor_batch])


def install_on_worker_dict(worker, out, *, provenance=None, hold=True, suspend=None):
    """Parent calls through the existing execute_with_func_generator RPC only.

    WorkerDict's official dispatch uses getattr(w, name) at invocation, so an
    instance update_actor proxy intercepts its next original call. Suspension is
    BEFORE original update_actor, AFTER the completed compute_dt/finally and
    TaskRunner's native scatter/whitening. A mocked suspend is only for CPU tests.
    Production uses psutil.Process(os.getpid()).suspend(); root resumes both ranks.
    """
    owners, seen = [], set()
    for owner in worker.worker_dict.values():
        producer = getattr(owner, '_deltatrace_producer', None)
        if producer is not None and getattr(producer, 'direct_readout', None) is not None:
            if id(owner) not in seen:
                owners.append(owner)
                seen.add(id(owner))
    records = []
    for owner in owners:
        rank = owner.rank
        raw_path = Path(out) / f'rank{rank}-readout.pt'
        actor_path = Path(out) / f'rank{rank}-pre-update.pt'
        ReadoutObservation(owner._deltatrace_producer.direct_readout,
                           raw_path, provenance=provenance).install()
        original, restore = owner.update_actor, _binding(owner, 'update_actor')
        record = dict(pid=os.getpid(), rank=rank, installed=True,
                      raw_readout_path=str(raw_path), actor_path=str(actor_path),
                      hold=hold)

        def observed_update(*args, _original=original, _restore=restore,
                            _record=record, _raw=raw_path, _path=actor_path, **kwargs):
            _restore()  # One-shot; owner binding restored even while suspended.
            data = kwargs.get('data', args[0] if args else None)
            try:
                payload = _actor_snapshot(data, provenance or {}, _raw)
                _path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, _path)
                _record['actor_saved'] = True
            except Exception as error:
                _record['observation_error'] = repr(error)
            if hold:
                # Hold even if observation I/O failed: do not race an update.
                _record['hold_entered'] = True
                if suspend is None:
                    import psutil
                    psutil.Process(os.getpid()).suspend()
                else:
                    suspend()
            # Reached only after parent SIGCONT (or a CPU-test suspend callback).
            return _original(*args, **kwargs)

        owner.update_actor = observed_update
        records.append(record)
    return records
