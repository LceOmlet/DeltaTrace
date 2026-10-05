"""Delegate the verified matched-layout worker with primitive observations.

The existing worker still owns the same seven B4 groups, two single-token
probes per group, full EOS trace, model initialization and cleanup.  This
module substitutes only the two passive observation callbacks, using the
actual serialized Ray/VERL method globals already resolved by that worker.
No forward, finite rule, reference, PPO or training implementation is added.
"""
from contextlib import contextmanager
import sys

import ray
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

import verify_textcraft_conditional_boundaries as existing
from observe_textcraft_conditional_primitives import (
    collect_actual_primitives, contract_saved_native_primitives)


ORIGINAL_WORKER_RECIPE = existing.ConditionalBoundaryWorker.__ray_metadata__.modified_class.inspect_matched_layout
# These names are consumed by the unchanged original CPU-binding inspection.
ORIGINAL_RECIPE = existing.ORIGINAL_RECIPE
collect_actual_coefficients = collect_actual_primitives


@ray.remote
class ConditionalPrimitiveWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_matched_layout(self):
        body = existing.registered_body(ORIGINAL_WORKER_RECIPE)
        worker_globals = body.__globals__
        complete = worker_globals['observe_complete_original_trace']
        complete_globals = complete.__globals__
        old_collector = complete_globals['collect_actual_coefficients']
        old_contractor = worker_globals['contract_saved_native_roots']
        old_metadata_collector = worker_globals['collect_actual_coefficients']
        retained_banks = []

        @contextmanager
        def collect(runner, slots):
            with old_collector(runner, slots) as bank:
                with collect_actual_primitives(runner, bank, slots):
                    yield bank
            retained_banks.append(bank)

        @contextmanager
        def contract(runner, bank, slots):
            with old_contractor(runner, bank, slots) as report:
                with contract_saved_native_primitives(runner, bank, slots) as primitives:
                    yield report
            report['conditional_primitives'] = primitives
            # The original recipe makes exactly two singles per group and
            # releases the original bank afterward.  Release this observation
            # bank at the same already-owned group boundary.
            bank['primitive_single_calls'] = bank.get('primitive_single_calls', 0) + 1
            if bank['primitive_single_calls'] == 2:
                bank['primitive_bank'].clear()

        complete_globals['collect_actual_coefficients'] = collect
        worker_globals['collect_actual_coefficients'] = collect
        worker_globals['contract_saved_native_roots'] = contract
        try:
            receipt = ORIGINAL_WORKER_RECIPE(self)
            receipt['primitive_observer_source'] = existing.existing.identity(collect_actual_primitives)
            return receipt
        finally:
            complete_globals['collect_actual_coefficients'] = old_collector
            worker_globals['collect_actual_coefficients'] = old_metadata_collector
            worker_globals['contract_saved_native_roots'] = old_contractor
            for bank in retained_banks:
                bank.get('primitive_bank', {}).clear()
                bank.get('by_boundary', {}).clear()
            retained_banks.clear()


if __name__ == '__main__':
    existing.ConditionalBoundaryWorker = ConditionalPrimitiveWorker
    existing.main()
