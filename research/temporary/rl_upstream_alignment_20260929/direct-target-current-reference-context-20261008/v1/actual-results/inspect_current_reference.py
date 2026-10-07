"""Bind the existing original native reference-context diagnostic to current B4.

Only the evidenced case metadata is supplied. Original owner constructs the
EOS/restore pair, initializes VERL and scores original joint Y once per rank.
No scoring, finite DT, credit, environment, inference or training copy.
"""
import json
from pathlib import Path

import inspect_reference_context as owner

CASE=json.loads((Path(__file__).resolve().parent/'case.json').read_bytes())

def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch,register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class CurrentReferenceWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_reference_context(self,source_path,output):
            owner.CASES['appworld']=CASE
            original=owner.make_worker
            assert original.__module__=='inspect_reference_context'
            worker=original().__ray_metadata__.modified_class
            return worker.inspect_reference_context(self,source_path,output)
    return CurrentReferenceWorker

if __name__=='__main__':
    owner.CASES['appworld']=CASE
    owner.make_worker=make_worker
    owner.main()
