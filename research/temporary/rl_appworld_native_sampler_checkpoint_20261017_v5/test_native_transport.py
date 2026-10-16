"""Exercise native two-rank CPU sampler checkpoint transport without rollouts."""
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-native-sampler-checkpoint-20261017-v5'


def main():
    patch=json.loads((AUDIT/'owner-patch.json').read_bytes())
    assert json.loads((AUDIT/'CPU-sampler-checkpoint-verification.json').read_bytes())['status']=='passed'
    source=json.loads((ROOT/'runs/appworld-dtpo-reward-target-exponential-formal-20261017-v4/source.json').read_bytes())
    if '--native' not in sys.argv:
        env=dict(os.environ,**source['effective_environment'])
        for key in ('loop','entry','verl_dt'):
            env={k:v.replace(patch['baselines'][key],patch['owner_roots'][key]) for k,v in env.items()}
        env['CUDA_VISIBLE_DEVICES']=''
        env.pop('MACA_VISIBLE_DEVICES',None)
        env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(AUDIT/'native-transport-worker-receipts')
        raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--native'],env=env))

    site.addsitedir(os.environ['LOOP_EXTRAS'])
    import psutil
    import torch
    from omegaconf import OmegaConf
    from loop_owner_recipe import compose
    from loop_owner_worker import OwnerProcesses
    from loop_owner_rollout import LoopOwner
    from phi_agents.rl.parallel_scenario_sampler import ParallelScenarioSampler
    from phi_agents.rl.appworld_scenario_runner import AppWorldScenarioSampler

    cfg=compose(Path(patch['owner_roots']['loop']),[])
    config=OmegaConf.to_container(cfg,resolve=True)
    record=dict(started=time.time(),status='starting_native_CPU_owners',owner_roots=patch['owner_roots'],
                world_size=2,native_runners_per_rank=cfg.rl.num_scenario_runners,
                model_calls=0,DT_calls=0,optimizer_updates=0,rollout_requests=0,
                formal_training_checkpoint_create_calls=0)
    pool=None
    references=[]
    try:
        pool=OwnerProcesses(config,source['options']['actor_rollout_ref.model.path'],0,False,2)
        identities=[dict(pid=p.pid,birth=psutil.Process(p.pid).create_time()) for p in pool.processes]
        proxy=LoopOwner.__new__(LoopOwner)
        proxy.processes=pool
        initial=proxy.checkpoint_state_dict()
        assert len(initial)==2
        expected_states=[]
        expected_next=[]
        dataset=cfg.rl.scenario_sampler.dataset_name
        for rank in range(2):
            native=ParallelScenarioSampler(lambda:AppWorldScenarioSampler(dataset),num_threads=1)
            references.append(native)
            native.load_state_dict(initial[rank])
            native.ensure_scenarios_requested(31)
            for _ in range(7):next(native)
            # Deliberately quiescent prefetch snapshots make the physical state
            # readback exact; the pending-request case is tested separately.
            with native._mutex:
                while native._n_scenarios_requested or native._n_scenarios_in_progress:
                    native._generation_requested.wait(timeout=1.0)
                state=native.state_dict()
            expected_states.append(state)
            count=2*len(state['samplers'][0]['all_task_ids'])+11
            expected_next.append([next(native).task_id for _ in range(count)])

        with tempfile.TemporaryDirectory(prefix='appworld-sampler-transport-state-only-') as directory:
            path=Path(directory)/'loop-sampler-state-only.pt'
            torch.save(expected_states,path)
            proxy.load_checkpoint_state_dict(torch.load(path,weights_only=False))
            actual=proxy.checkpoint_state_dict()
            assert actual==expected_states
            comparisons=[]
            for rank,state in enumerate(actual):
                restored=ParallelScenarioSampler(lambda:AppWorldScenarioSampler(dataset),num_threads=1)
                references.append(restored)
                restored.load_state_dict(state)
                actual_next=[next(restored).task_id for _ in range(len(expected_next[rank]))]
                assert actual_next==expected_next[rank]
                comparisons.append(dict(rank=rank,actual_CPU_owner=identities[rank],
                    numpy_RNG_and_remaining_task_order_exact=True,prefetch_queue_and_request_count_exact=True,
                    next_scenarios_exact=len(actual_next),prefetched_scenarios=len(state['scenarios'])))

            pending=LoopOwner.__new__(LoopOwner)
            pending.processes=None
            pending.load_checkpoint_state_dict(actual)
            assert pending._sampler_checkpoint_to_restore==actual

        assert not torch.cuda.is_initialized()
        record.update(status='native_state_checks_passed_cleanup_pending',comparisons=comparisons,
                      initial_native_states_read=True,actual_two_rank_IPC_save_load_readback_exact=True,
                      driver_pending_restore_preserves_same_state=True)
    finally:
        for sampler in references:sampler.stop()
        try:
            if pool is not None:pool.close()
        except BaseException as error:
            record['cleanup_error']=repr(error)
            raise
        finally:
            if pool is not None:
                record['CPU_owner_cleanup']=[dict(pid=p.pid,alive=p.is_alive(),exitcode=p.exitcode) for p in pool.processes]
            record['finished']=time.time()
            (AUDIT/'native-two-rank-sampler-checkpoint-verification.json').write_text(json.dumps(record,indent=2)+'\n')
    assert len(record['CPU_owner_cleanup'])==2 and not any(p['alive'] for p in record['CPU_owner_cleanup'])
    record['status']='passed'
    (AUDIT/'native-two-rank-sampler-checkpoint-verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    main()
