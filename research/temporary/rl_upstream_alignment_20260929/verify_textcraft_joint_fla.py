"""Observe same-pair FLA contractions through the original capture extension.

The original FLA recipe keeps its seven B4/rank layouts and all finite/native
calls. Only the full-trace collector gains a passive public-output observation.
No model, propagation rule, target, reference or training setting is changed.
"""
from contextlib import contextmanager
import json
import sys

import ray
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

import verify_textcraft_conditional_fla as prior
import verify_textcraft_conditional_boundaries as entry
from verify_textcraft_conditional_boundaries import registered_body
from verify_textcraft_native_readout import identity
from observe_textcraft_joint_fla import collect_actual_joint_fla


ORIGINAL_FLA_RECIPE = prior.ConditionalFLAWorker.__ray_metadata__.modified_class.inspect_matched_layout
ORIGINAL_RECIPE = prior.ORIGINAL_RECIPE


@ray.remote
class JointFLAWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_matched_layout(self):
        owner_globals = registered_body(ORIGINAL_FLA_RECIPE).__globals__
        previous = owner_globals['collect_actual_fla']

        @contextmanager
        def collect(runner, bank, slots):
            with collect_actual_joint_fla(runner, bank, slots) as receipt:
                yield receipt
            joint = bank['metadata']['conditional_fla']['joint_fla']
            if (joint['diagnostics'] or
                    joint['recorded_layer_slot_scalars'] != joint['expected_layer_slot_scalars']):
                raise RuntimeError('Incomplete same-pair observation: '+repr(joint['diagnostics']))

        owner_globals['collect_actual_fla'] = collect
        try:
            receipt = ORIGINAL_FLA_RECIPE(self)
            receipt['joint_fla_observer_source'] = identity(collect_actual_joint_fla)
            receipt['original_matched_recipe'] = identity(ORIGINAL_RECIPE)
            return receipt
        finally:
            owner_globals['collect_actual_fla'] = previous


if __name__ == '__main__':
    entry.ConditionalBoundaryWorker = JointFLAWorker
    entry.main()
    if '--inspect-only' in sys.argv:
        path = entry.OUT / 'native-owner-inspection.json'
        receipt = json.loads(path.read_bytes())
        receipt['joint_fla_observer'] = identity(collect_actual_joint_fla)
        receipt['delegated_FLA_recipe'] = identity(ORIGINAL_FLA_RECIPE)
        path.write_text(json.dumps(receipt, indent=2)+'\n')
