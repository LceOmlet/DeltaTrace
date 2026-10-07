"""Prepared-only first-DT input capture and optional passive replay scope.

Compose the original producer call with the unchanged v2 token observation and
pre-update file-release hold. No producer/model construction, extra DT call,
training implementation, process suspension, or default installation is added.
The parent supplies source-bound modules through the existing VERL worker RPC.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time


def install_before_first_dt(worker, out, *, observer_module, producer_class=None,
                            provenance=None, hold=True, replay_scope_factory=None):
    """Arm once after original worker initialization, during native sampling.

    The original lazy producer is assigned before its first prepared call.
    Restore this class binding, install v2, save the received DataProto, then
    invoke that same original method once. The optional passive context owns
    its own restoration; it must preserve the owner's return and exception.
    Distinct rank release paths remain owned by the unchanged v2 observer.
    """
    if producer_class is None:
        from deltatrace_rollout import DeltaTraceRolloutProducer
        producer_class = DeltaTraceRolloutProducer
    owners, identities = [], set()
    for owner in worker.worker_dict.values():
        if (callable(getattr(owner, 'compute_dt_token_advantages', None)) and
                callable(getattr(owner, 'update_actor', None)) and id(owner) not in identities):
            owners.append(owner)
            identities.add(id(owner))
    provenance = dict(provenance or {})
    observer_path = Path(observer_module.__file__)
    provenance.update(observer_source=str(observer_path),
                      observer_sha256=hashlib.sha256(observer_path.read_bytes()).hexdigest(),
                      first_dt_seam_source=str(Path(__file__).resolve()),
                      first_dt_seam_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record = dict(status='armed_not_yet_activated', pid=os.getpid(), armed_unix=time.time(),
                  provenance=provenance, replay_scope_enabled=replay_scope_factory is not None,
                  owners=[dict(rank=owner.rank,
                               prepared_input=str(Path(out) / f'rank{owner.rank}-input-prepared.pt'),
                               release_file=str(Path(out) / f'rank{owner.rank}-release-update'),
                               installation_sidecar=str(Path(out) / f'rank{owner.rank}-first-dt-install.json'))
                          for owner in owners])
    if not owners:
        record['status'] = 'no_original_DT_owner_found_no_binding_changed'
        return record
    original = producer_class.attribute_prepared_batch
    restore = observer_module._binding(producer_class, 'attribute_prepared_batch')

    def first(producer, *args, **kwargs):
        matched = [owner for owner in owners
                   if getattr(owner, '_deltatrace_producer', None) is producer]
        if not matched:
            return original(producer, *args, **kwargs)
        restore()
        try:
            installed = observer_module.install_on_worker_dict(
                worker, out, provenance=provenance, hold=hold)
            record.update(status='activated_before_original_prepared_call',
                          activated_unix=time.time(), workers=installed)
        except Exception as error:
            record['installation_observation_error'] = repr(error)
        data = kwargs.get('data', args[0] if args else None)
        record['prepared_inputs'] = []
        for owner in matched:
            path = Path(out) / f'rank{owner.rank}-input-prepared.pt'
            try:
                batch = getattr(data, 'batch', None)
                payload = dict(scope='exact original DataProto before first attribute_prepared_batch',
                               pid=os.getpid(), rank=owner.rank, captured_unix=time.time(),
                               provenance=observer_module._snapshot(provenance),
                               batch=observer_module._snapshot(
                                   None if batch is None else dict(batch.items())),
                               non_tensor_batch=observer_module._snapshot(
                                   getattr(data, 'non_tensor_batch', None)),
                               meta_info=observer_module._snapshot(getattr(data, 'meta_info', None)))
                path.parent.mkdir(parents=True, exist_ok=True)
                observer_module.torch.save(payload, path)
                record['prepared_inputs'].append(dict(rank=owner.rank, path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), saved=True))
            except Exception as error:
                record['prepared_inputs'].append(dict(rank=owner.rank, path=str(path),
                                                      saved=False, observer_error=repr(error)))
            try:
                sidecar = Path(out) / f'rank{owner.rank}-first-dt-install.json'
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            except Exception as error:
                record.setdefault('observer_errors', []).append(repr(error))
        scope = (nullcontext() if replay_scope_factory is None else
                 replay_scope_factory(matched[0], producer))
        with scope:
            return original(producer, *args, **kwargs)

    producer_class.attribute_prepared_batch = first
    return record
