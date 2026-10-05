"""Observe the two already-computed owner endpoint orders; no extra model call."""
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
from observe_textcraft_fla_orders import (
    collect_actual_fla_orders, contract_saved_native_fla_orders)


ORIGINAL_FLA_RECIPE = prior.ConditionalFLAWorker.__ray_metadata__.modified_class.inspect_matched_layout
ORIGINAL_RECIPE = prior.ORIGINAL_RECIPE


@ray.remote
class EndpointOrderWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_matched_layout(self):
        original_globals = registered_body(ORIGINAL_FLA_RECIPE).__globals__
        original_collector = original_globals['collect_actual_fla']
        original_contractor = original_globals['contract_saved_native_fla']
        banks = []

        @contextmanager
        def collect(runner, bank, slots):
            banks.append(bank)
            with collect_actual_fla_orders(runner, bank, slots) as receipt:
                yield receipt
            orders = bank['metadata']['conditional_fla']['conditional_orders']
            if orders['diagnostics'] or orders['recorded_original_avg_returns'] != orders['expected_original_avg_returns']:
                raise RuntimeError('Incomplete original endpoint-order observation: '+repr(orders['diagnostics']))

        @contextmanager
        def contract(runner, bank, slots):
            with contract_saved_native_fla_orders(runner, bank, slots) as receipt:
                yield receipt
            orders = receipt['conditional_orders']
            if (orders['diagnostics'] or orders['recorded_layer_slot_scalars'] != orders['expected_layer_slot_scalars']
                    or orders['reused_average_contraction_calls'] != orders['expected_reused_average_contraction_calls']):
                raise RuntimeError('Incomplete original endpoint-order contraction: '+repr(orders['diagnostics']))

        original_globals['collect_actual_fla'] = collect
        original_globals['contract_saved_native_fla'] = contract
        try:
            receipt = ORIGINAL_FLA_RECIPE(self)
            receipt['endpoint_order_observer_source'] = identity(collect_actual_fla_orders)
            receipt['original_matched_recipe'] = identity(ORIGINAL_RECIPE)
            return receipt
        finally:
            original_globals['collect_actual_fla'] = original_collector
            original_globals['contract_saved_native_fla'] = original_contractor
            for bank in banks:
                bank.get('fla_order_bank', {}).clear()
            banks.clear()


if __name__ == '__main__':
    entry.ConditionalBoundaryWorker = EndpointOrderWorker
    entry.main()
    if '--inspect-only' in sys.argv:
        path = entry.OUT / 'native-owner-inspection.json'
        receipt = json.loads(path.read_bytes())
        receipt['endpoint_order_observer'] = identity(collect_actual_fla_orders)
        receipt['endpoint_order_contractor'] = identity(contract_saved_native_fla_orders)
        receipt['delegated_FLA_recipe'] = identity(ORIGINAL_FLA_RECIPE)
        path.write_text(json.dumps(receipt, indent=2)+'\n')
