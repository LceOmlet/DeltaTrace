"""Delegate the measured primitive worker with passive GDN observations.

The original worker owns all paired layouts, model calls, finite propagation,
checkpoint loading and cleanup.  Only its passive observation contexts are
composed here; no new model, reference, finite rule or PPO path is introduced.
"""
from contextlib import contextmanager
import json
import sys

import ray
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

import verify_textcraft_conditional_primitives as prior
from observe_textcraft_conditional_gdn import (
    collect_actual_gdn, contract_saved_native_gdn)


ORIGINAL_PRIMITIVE_RECIPE = prior.ConditionalPrimitiveWorker.__ray_metadata__.modified_class.inspect_matched_layout
ORIGINAL_RECIPE = prior.ORIGINAL_RECIPE


@ray.remote
class ConditionalGDNWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_matched_layout(self):
        body = prior.existing.registered_body(ORIGINAL_PRIMITIVE_RECIPE)
        worker_globals = body.__globals__
        old_collector = worker_globals['collect_actual_primitives']
        old_contractor = worker_globals['contract_saved_native_primitives']
        retained_banks = []

        @contextmanager
        def collect(runner, bank, slots):
            retained_banks.append(bank)
            with old_collector(runner, bank, slots):
                with collect_actual_gdn(runner, bank, slots):
                    yield
            metadata = bank['metadata']['conditional_gdn']
            errors = metadata['diagnostics'] + [error
                for layer in metadata['layers'].values() for error in layer['diagnostics']]
            if errors or metadata['captured_layer_indices'] != metadata['layer_indices']:
                raise RuntimeError('Incomplete original GDN coefficient observation: ' + repr(errors))
            if any(layer['norm_gate_calls'] != 1 for layer in metadata['layers'].values()):
                raise RuntimeError('Missing actual original GDN norm-gate call.')

        @contextmanager
        def contract(runner, bank, slots):
            with old_contractor(runner, bank, slots) as report:
                with contract_saved_native_gdn(runner, bank, slots) as gdn:
                    yield report
            report['conditional_gdn'] = gdn
            if gdn['diagnostics'] or gdn['recorded_layer_slot_scalars'] != gdn['expected_layer_slot_scalars']:
                raise RuntimeError('Incomplete original GDN root observation: ' + repr(gdn['diagnostics']))
            bank['gdn_single_calls'] = bank.get('gdn_single_calls', 0) + 1
            if bank['gdn_single_calls'] == 2:
                bank.get('gdn_bank', {}).clear()

        worker_globals['collect_actual_primitives'] = collect
        worker_globals['contract_saved_native_primitives'] = contract
        try:
            receipt = ORIGINAL_PRIMITIVE_RECIPE(self)
            receipt['gdn_observer_source'] = prior.existing.existing.identity(collect_actual_gdn)
            receipt['original_matched_recipe'] = prior.existing.existing.identity(ORIGINAL_RECIPE)
            return receipt
        finally:
            worker_globals['collect_actual_primitives'] = old_collector
            worker_globals['contract_saved_native_primitives'] = old_contractor
            for bank in retained_banks:
                bank.get('gdn_bank', {}).clear()
            retained_banks.clear()


if __name__ == '__main__':
    prior.existing.ConditionalBoundaryWorker = ConditionalGDNWorker
    prior.existing.main()
    if '--inspect-only' in sys.argv:
        path = prior.existing.OUT / 'native-owner-inspection.json'
        receipt = json.loads(path.read_bytes())
        receipt['gdn_observer'] = prior.existing.existing.identity(collect_actual_gdn)
        receipt['gdn_contractor'] = prior.existing.existing.identity(contract_saved_native_gdn)
        receipt['delegated_primitive_recipe'] = prior.existing.existing.identity(ORIGINAL_PRIMITIVE_RECIPE)
        path.write_text(json.dumps(receipt, indent=2) + '\n')
