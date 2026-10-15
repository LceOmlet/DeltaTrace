"""Apply the verified minimal owner patch at original Ray RPC boundaries.

No training restart, optimizer/scheduler call, hyperparameter or credit change.
The code is an in-place patch of the bound pinned VERL file, with its original
bytes backed up and actual imports/worker births recorded. This script is
prepared only until the owner parity/cost verification has actually passed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DEST=ROOT/'receipts/textcraft-efficiency-20261015-v1'
BASELINE='e285c3353bddbffae38df346b44014ee5038b606531874ddbeb10ee1241f77de'
CANDIDATE='bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2'
JOBS={
    'DTPO':dict(run='textcraft-dt-grpo-multiplier-variance-only-20261014-v3',pid=3140624,birth=1791993570.92,
                workers={3151306:1791993626.33,3153050:1791993643.98}),
    'PPO':dict(run='textcraft-ppo-control-fixed-verl-fresh-20261014-v2',pid=3946004,birth=1792041120.19,
               workers={3953367:1792041179.18,3955296:1792041198.10}),
}
LABEL=sys.argv[sys.argv.index('--job')+1] if '--job' in sys.argv else None


def deploy(worker):
    import hashlib,importlib,inspect,json,os,time
    from pathlib import Path
    import psutil,torch
    from verl.utils.experimental import torch_functional as owner
    from verl.workers.actor import dp_actor
    from verl.trainer.ppo import core_algos
    assert psutil.Process().create_time()==JOBS[LABEL]['workers'][os.getpid()]
    w=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
    heads=[m for n,m in w.actor.actor_module.named_modules() if n.endswith('.lm_head')]
    assert len(heads)==1 and type(heads[0]) is torch.nn.Linear and not heads[0].weight.requires_grad
    assert hashlib.sha256(Path(dp_actor.__file__).read_bytes()).hexdigest()=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
    assert hashlib.sha256(Path(core_algos.__file__).read_bytes()).hexdigest()=='fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
    assert hasattr(owner._fused_linear_for_ppo_bwd,'_torchdynamo_orig_callable'),'Remove one-shot observation before deploying'
    assert hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()==CANDIDATE
    scheduler=w.actor_lr_scheduler.last_epoch
    importlib.invalidate_caches()
    importlib.reload(owner)
    original_callable=owner._fused_linear_for_ppo_bwd._torchdynamo_orig_callable
    assert inspect.signature(original_callable).parameters['compute_vocab_grad'].default is True
    assert 'compute_vocab_grad=vocab_weights.requires_grad' in inspect.getsource(owner.FusedLinearForPPOFunction.backward)
    assert scheduler==w.actor_lr_scheduler.last_epoch
    record=dict(unix=time.time(),job=LABEL,pid=os.getpid(),birth=psutil.Process().create_time(),rank=w.rank,
        source_path=owner.__file__,previous_source_sha256=BASELINE,source_sha256=CANDIDATE,
        compiled_helper_source=inspect.getsourcefile(original_callable),
        public_backward_source=inspect.getsourcefile(owner.FusedLinearForPPOFunction.backward),
        scheduler_before=scheduler,scheduler_after=w.actor_lr_scheduler.last_epoch,
        frozen_head=True,original_actor_core_unchanged=True,
        additional_model_DT_backward_optimizer_scheduler_calls=0,
        training_hyperparameter_credit_changes=0,status='patched_bound_VERL_owner_reloaded_at_RPC_boundary')
    (DEST/f'{LABEL}-rank{w.rank}-frozen-head-deployed.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    assert LABEL in JOBS
    job=JOBS[LABEL];run=ROOT/'runs'/job['run']
    if '--client' in sys.argv:
        import ray
        p=psutil.Process(job['pid']);assert p.create_time()==job['birth']
        gcs=next(c for c in p.children(recursive=True) if c.name()=='gcs_server')
        host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
        port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
        ray.init(address=host+':'+port,log_to_driver=False)
        try:
            names=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            assert len(names)==2
            values=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=deploy) for a in names],timeout=7200)
            assert {v['pid']:v['birth'] for v in values}==job['workers']
            record=dict(unix=time.time(),job=LABEL,driver_pid=job['pid'],driver_birth=job['birth'],workers=values,
                version='textcraft-native-frozen-head-efficiency-20261015-v1',
                source_patch_commit=os.environ.get('DT_EFFICIENCY_PATCH_COMMIT'),
                verification_receipts=[str(DEST/n) for n in ('head-forward-benchmark.json','native-frozen-head-chunk.json')],
                status='both_actual_workers_reloaded_verified_frozen_head_owner_patch',
                method_unchanged=True,model_optimizer_scheduler_calls_added=0)
            (run/'runtime-overrides/native-frozen-head-efficiency-20261015-v1.json').write_text(json.dumps(record,indent=2)+'\n')
            (DEST/f'{LABEL}-frozen-head-deployment.json').write_text(json.dumps(record,indent=2)+'\n')
        finally:ray.shutdown()
    elif '--prepare' in sys.argv:
        # The isolated check uses actual native hidden states, the original
        # public autograd owner, and both frozen/trainable input contracts.
        for name in ('head-forward-benchmark.json',):
            v=json.loads((DEST/name).read_bytes())
            assert v['phase']=='completed' and v['hidden_gradient_equal']
            assert all(all(c['returned_logp_entropy_and_requested_gradients_equal']) for c in v['public_autograd_checks'])
        assert json.loads((DEST/'native-frozen-head-chunk.json').read_bytes())['original_function_restored']
        launch=json.loads((run/'launch.json').read_bytes())
        assert psutil.Process(job['pid']).create_time()==job['birth']
        path=Path(launch['effective_environment']['VERL_ROOT'])/'verl/utils/experimental/torch_functional.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==BASELINE
        backup=DEST/f'{LABEL}-pinned-head-before-efficiency-v1.py'
        with backup.open('xb') as f:f.write(path.read_bytes())
        candidate=ROOT/'candidates/textcraft-native-frozen-head-efficiency-20261015-v1/verl/utils/experimental/torch_functional.py'
        assert hashlib.sha256(candidate.read_bytes()).hexdigest()==CANDIDATE
        compile(candidate.read_text(),str(path),'exec')
        path.write_bytes(candidate.read_bytes())
        env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
        env['DT_EFFICIENCY_PATCH_COMMIT']=sys.argv[sys.argv.index('--code-commit')+1]
        stdout,stderr=DEST/f'{LABEL}-deploy.stdout.log',DEST/f'{LABEL}-deploy.stderr.log'
        with stdout.open('xb') as out,stderr.open('xb') as err:
            p=subprocess.Popen([env['VENV_PYTHON'],'-u',__file__,'--job',LABEL,'--client'],env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        record=dict(unix=time.time(),job=LABEL,client_pid=p.pid,client_birth=psutil.Process(p.pid).create_time(),
            previous_sha256=BASELINE,sha256=CANDIDATE,owner_path=str(path),backup_path=str(backup),
            status='verified_owner_file_patched_waiting_actual_RPC_reload',
            stdout=str(stdout),stderr=str(stderr))
        (DEST/f'{LABEL}-deployment-client.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record))
