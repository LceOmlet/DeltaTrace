"""Compose the measured GDN worker with passive original FLA observations.

The original recipe retains all model/finite calls and the same seven B4
layouts per rank.  This only composes observation contexts and telescopes
recorded scalar projections; it adds no model, recurrence, reference or loss.
"""
from contextlib import contextmanager
import json
import sys

import ray
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

import verify_textcraft_conditional_gdn as prior
from observe_textcraft_conditional_fla import (
    collect_actual_fla, contract_saved_native_fla)


ORIGINAL_GDN_RECIPE = prior.ConditionalGDNWorker.__ray_metadata__.modified_class.inspect_matched_layout
ORIGINAL_RECIPE = prior.ORIGINAL_RECIPE


@ray.remote
class ConditionalFLAWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_matched_layout(self):
        body = prior.prior.existing.registered_body(ORIGINAL_GDN_RECIPE)
        worker_globals = body.__globals__
        old_collector = worker_globals['collect_actual_gdn']
        old_contractor = worker_globals['contract_saved_native_gdn']
        retained_banks = []

        @contextmanager
        def collect(runner, bank, slots):
            retained_banks.append(bank)
            with old_collector(runner, bank, slots):
                with collect_actual_fla(runner, bank, slots):
                    yield
            metadata = bank['metadata']['conditional_fla']
            if metadata['diagnostics'] or any(not layer['groups']
                    for layer in metadata['layers'].values()):
                raise RuntimeError('Incomplete original FLA coefficient observation: '
                    + repr(metadata['diagnostics']))

        @contextmanager
        def contract(runner, bank, slots):
            with old_contractor(runner, bank, slots) as gdn:
                with contract_saved_native_fla(runner, bank, slots) as fla:
                    yield gdn
            gdn['conditional_fla'] = fla
            if fla['diagnostics'] or fla['recorded_layer_slot_scalars'] != fla['expected_layer_slot_scalars']:
                raise RuntimeError('Incomplete original FLA root observation: ' + repr(fla['diagnostics']))
            original = {(row['decoder_index'], row['original_slot']): row for row in gdn['layers']}
            for row in fla['layers']:
                gdn_row = original[row['decoder_index'], row['original_slot']]
                effects = gdn_row['effects']
                f = row['F_input_projection']
                y = row['Y_public_output_projection']
                terms = dict(input_chain=effects['mixer_input']-effects['z_input']-f,
                    FLA_projection=f-y, native_cast_bridge=y-effects['o_input'])
                row['original_GDN_effects'] = effects
                row['remaining_three_terms'] = terms
                row['remaining_three_term_sum'] = sum(terms.values())
                row['original_GDN_remaining'] = gdn_row['three_terms']['remaining_input_FLA']
                row['remaining_closure'] = sum(terms.values())-row['original_GDN_remaining']
            bank['fla_single_calls'] = bank.get('fla_single_calls', 0) + 1
            if bank['fla_single_calls'] == 2:
                bank.get('fla_bank', {}).clear()

        worker_globals['collect_actual_gdn'] = collect
        worker_globals['contract_saved_native_gdn'] = contract
        try:
            receipt = ORIGINAL_GDN_RECIPE(self)
            receipt['fla_observer_source'] = prior.prior.existing.existing.identity(collect_actual_fla)
            receipt['original_matched_recipe'] = prior.prior.existing.existing.identity(ORIGINAL_RECIPE)
            return receipt
        finally:
            worker_globals['collect_actual_gdn'] = old_collector
            worker_globals['contract_saved_native_gdn'] = old_contractor
            for bank in retained_banks:
                bank.get('fla_bank', {}).clear()
            retained_banks.clear()


if __name__ == '__main__':
    prior.prior.existing.ConditionalBoundaryWorker = ConditionalFLAWorker
    prior.prior.existing.main()
    if '--inspect-only' in sys.argv:
        path = prior.prior.existing.OUT / 'native-owner-inspection.json'
        receipt = json.loads(path.read_bytes())
        receipt['fla_observer'] = prior.prior.existing.existing.identity(collect_actual_fla)
        receipt['fla_contractor'] = prior.prior.existing.existing.identity(contract_saved_native_fla)
        receipt['delegated_GDN_recipe'] = prior.prior.existing.existing.identity(ORIGINAL_GDN_RECIPE)
        path.write_text(json.dumps(receipt, indent=2) + '\n')
