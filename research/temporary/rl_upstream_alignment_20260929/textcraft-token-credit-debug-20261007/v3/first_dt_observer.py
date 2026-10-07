"""Prepared-only first-DT seam, installed by the original VERL worker RPC.

Compose v2 observation with the existing producer.attribute_prepared_batch.
Original FSDP compute_dt creates/assigns its producer before invoking this
method, so the proxy can install v2 before the FIRST actual readout. No early
producer/model, FSDP clone, scheduling, extra DT, or OS suspension is added.
The parent supplies the already source-bound v2 module and uses the existing
execute_with_func_generator; importing this file changes nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


def install_before_first_dt(worker, out, *, observer_module, producer_class=None,
                            provenance=None, hold=True):
    """Arm once during original sampling, after worker init_model has returned.

    RPC timeouts do not mean this pending installation failed. Parent must not
    retry it blindly or stop any Ray process. The first matched original call
    restores its class binding before installing v2 and invoking the same owner
    method. The v2 update proxy holds only that first original actor call, using
    distinct rank release paths in this run's new output directory.
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
                  provenance=provenance,
                  owners=[dict(rank=owner.rank,
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
        restore()  # Original class method restored even if the owner later raises.
        try:
            installed = observer_module.install_on_worker_dict(
                worker, out, provenance=provenance, hold=hold)
            record.update(status='activated_before_original_prepared_call',
                          activated_unix=time.time(), workers=installed)
            for owner in matched:
                path = Path(out) / f'rank{owner.rank}-first-dt-install.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        except Exception as error:
            # Passive observation never replaces an original return/exception.
            # If sidecar writing failed AFTER v2 installed, its actor hold remains.
            record['installation_observation_error'] = repr(error)
        return original(producer, *args, **kwargs)

    producer_class.attribute_prepared_batch = first
    return record  # Only light metadata crosses RPC, never the actor/model object.
