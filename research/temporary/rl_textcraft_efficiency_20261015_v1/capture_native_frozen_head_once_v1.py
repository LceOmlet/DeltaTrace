"""Snapshot one already scheduled native fused-head backward chunk, rank0 only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
DEST=ROOT/'receipts/textcraft-efficiency-20261015-v1'
WORKERS={3151306:1791993626.33,3153050:1791993643.98}
OWNER_SHA='e285c3353bddbffae38df346b44014ee5038b606531874ddbeb10ee1241f77de'


def install(worker):
    import hashlib,json,os,time
    from pathlib import Path
    import psutil,torch
    from verl.utils.experimental import torch_functional as owner
    assert psutil.Process().create_time()==WORKERS[os.getpid()]
    actors=[w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False)]
    assert len(actors)==1
    w=actors[0]
    assert hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()==OWNER_SHA
    heads=[(n,m) for n,m in w.actor.actor_module.named_modules() if n.endswith('.lm_head')]
    assert len(heads)==1
    name,head=heads[0]
    info=dict(unix=time.time(),pid=os.getpid(),birth=psutil.Process().create_time(),rank=w.rank,
        head_path=name,head_class=type(head).__module__+'.'+type(head).__qualname__,
        head_weight_requires_grad=head.weight.requires_grad,head_weight_shape=list(head.weight.shape),
        owner_path=owner.__file__,owner_sha256=OWNER_SHA,scheduler=w.actor_lr_scheduler.last_epoch)
    # No other rank snapshots a checkpoint-sized weight or changes a callable.
    if w.rank!=0:return info
    assert not head.weight.requires_grad
    assert type(head) is torch.nn.Linear
    assert not hasattr(owner,'_efficiency_frozen_head_capture_v1')
    original=owner._fused_linear_for_ppo_bwd
    owner._efficiency_frozen_head_capture_v1=True
    def observed(*args,**kwargs):
        result=original(*args,**kwargs)
        owner._fused_linear_for_ppo_bwd=original
        try:
            assert not args
            artifact=dict(hidden=kwargs['hidden_states'].detach().cpu(),
                vocab_weights=kwargs['vocab_weights'].detach().cpu(),
                labels=kwargs['input_ids'].detach().cpu(),
                dlog_probs=None if kwargs['dlog_probs'] is None else kwargs['dlog_probs'].detach().cpu(),
                dentropy=None if kwargs['dentropy'] is None else kwargs['dentropy'].detach().cpu(),
                native_hidden_gradient=result[0].detach().cpu(),
                temperature=kwargs.get('temperature',1.0),metadata=info)
            torch.save(artifact,DEST/'native-frozen-head-chunk.pt')
            record=dict(info,captured_unix=time.time(),status='original_backward_chunk_saved',
                original_function_restored=owner._fused_linear_for_ppo_bwd is original,
                snapshot=str(DEST/'native-frozen-head-chunk.pt'),
                snapshot_bytes=(DEST/'native-frozen-head-chunk.pt').stat().st_size,
                hidden_shape=list(artifact['hidden'].shape),labels_shape=list(artifact['labels'].shape),
                input_requires_grad_hidden=kwargs['hidden_states'].requires_grad,
                input_requires_grad_vocab=kwargs['vocab_weights'].requires_grad,
                additional_model_DT_backward_optimizer_calls=0,training_math_changes=0)
            (DEST/'native-frozen-head-chunk.json').write_text(json.dumps(record,indent=2)+'\n')
        except Exception as exc:
            (DEST/'native-frozen-head-capture-error.json').write_text(json.dumps(dict(error=repr(exc),
                original_function_restored=owner._fused_linear_for_ppo_bwd is original)))
        return result
    owner._fused_linear_for_ppo_bwd=observed
    return dict(info,snapshot_queued=True)


def install_forward(worker):
    import hashlib,json,os,time
    from pathlib import Path
    import psutil,torch
    from verl.utils.experimental import torch_functional as owner
    assert psutil.Process().create_time()==WORKERS[os.getpid()]
    w=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
    if w.rank!=0:return dict(rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),installed=False)
    assert hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()==OWNER_SHA
    assert not hasattr(owner,'_efficiency_head_forward_capture_v1')
    original=owner._fused_linear_for_ppo_fwd
    owner._efficiency_head_forward_capture_v1=True
    def observed(*args,**kwargs):
        result=original(*args,**kwargs)
        owner._fused_linear_for_ppo_fwd=original
        try:
            assert not args
            artifact={k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in kwargs.items()}
            artifact['log_probs'],artifact['entropy']=[v.detach().cpu() for v in result]
            torch.save(artifact,DEST/'native-frozen-head-forward-chunk.pt')
            (DEST/'native-frozen-head-forward-chunk.json').write_text(json.dumps(dict(
                unix=time.time(),rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
                source_sha256=OWNER_SHA,hidden_shape=list(artifact['hidden_states'].shape),
                vocab_shape=list(artifact['vocab_weights'].shape),
                original_function_restored=owner._fused_linear_for_ppo_fwd is original,
                additional_model_DT_backward_optimizer_calls=0,training_math_changes=0),indent=2)+'\n')
        except Exception as exc:
            (DEST/'native-frozen-head-forward-error.json').write_text(json.dumps(dict(error=repr(exc))))
        return result
    owner._fused_linear_for_ppo_fwd=observed
    return dict(rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),installed=True)


def install_full_forward(worker):
    import hashlib,json,os,sys,time
    from pathlib import Path
    import psutil,torch
    from verl.utils.experimental import torch_functional as owner
    assert psutil.Process().create_time()==WORKERS[os.getpid()]
    w=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
    if w.rank!=0:return dict(rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),installed=False)
    assert hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()=='bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2'
    assert not hasattr(owner,'_efficiency_head_full_forward_capture_v2')
    owner._efficiency_head_full_forward_capture_v2=True
    original=owner.FusedLinearForPPO.forward
    from verl.models.transformers.qwen3_vl import forward_with_torch_backend
    def observed(instance,*args,**kwargs):
        frame=sys._getframe(1)
        original_ids=frame.f_locals.get('input_ids')
        keep=frame.f_locals.get('kwargs',{}).get('logits_to_keep',0)
        try:result=original(instance,*args,**kwargs)
        finally:owner.FusedLinearForPPO.forward=original
        try:
            assert frame.f_code is forward_with_torch_backend.__code__
            assert original_ids is not None
            assert not args
            artifact={k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in kwargs.items()}
            artifact['log_probs'],artifact['entropy']=[v.detach().cpu() for v in result]
            artifact['original_input_ids']=original_ids.detach().cpu()
            artifact['logits_to_keep']=keep.detach().cpu() if isinstance(keep,torch.Tensor) else keep
            torch.save(artifact,DEST/'native-frozen-head-full-chunk.pt')
            (DEST/'native-frozen-head-full-chunk.json').write_text(json.dumps(dict(
                unix=time.time(),rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
                hidden_shape=list(artifact['hidden_states'].shape),vocab_shape=list(artifact['vocab_weights'].shape),
                original_input_shape=list(artifact['original_input_ids'].shape),
                requested_logits_to_keep=keep if isinstance(keep,int) else 'exact_saved_index_tensor',
                actual_returned_head_shape=list(result[0].shape),
                original_function_restored=owner.FusedLinearForPPO.forward is original,
                additional_model_DT_backward_optimizer_calls=0,training_math_changes=0),indent=2)+'\n')
        except Exception as exc:
            (DEST/'native-frozen-head-full-v2-error.json').write_text(json.dumps(dict(error=repr(exc))))
        return result
    owner.FusedLinearForPPO.forward=observed
    return dict(rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),installed=True)


full_mode=any(x in sys.argv for x in ('--full','--full-client','--full-v2','--full-v2-client'))
forward_mode=any(x in sys.argv for x in ('--forward','--forward-client'))
prefix='native-frozen-head-full-v2' if full_mode else 'native-frozen-head-forward' if forward_mode else 'native-frozen-head'
if '--client' in sys.argv or '--forward-client' in sys.argv or '--full-client' in sys.argv or '--full-v2-client' in sys.argv:
    import ray
    launch=json.loads((RUN/'launch.json').read_bytes());p=psutil.Process(launch['pid']);assert p.create_time()==launch['birth']
    gcs=next(c for c in p.children(recursive=True) if c.name()=='gcs_server')
    host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    ray.init(address=host+':'+port,log_to_driver=False)
    try:
        names=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
        assert len(names)==2
        values=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install_full_forward if full_mode else install_forward if forward_mode else install) for a in names],timeout=7200)
        assert {v['pid']:v['birth'] for v in values}==WORKERS
        (DEST/(prefix+'-installation.json')).write_text(json.dumps(dict(unix=time.time(),workers=values,
            source=__file__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    finally:ray.shutdown()
elif '--read' not in sys.argv and not (DEST/(prefix+'-client.json')).exists():
    launch=json.loads((RUN/'launch.json').read_bytes());assert psutil.Process(launch['pid']).create_time()==launch['birth']
    env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
    stdout,stderr=DEST/(prefix+'-client.stdout.log'),DEST/(prefix+'-client.stderr.log')
    with stdout.open('xb') as out,stderr.open('xb') as err:
        p=subprocess.Popen([env['VENV_PYTHON'],'-u',__file__,'--full-v2-client' if full_mode else '--forward-client' if forward_mode else '--client'],env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    (DEST/(prefix+'-client.json')).write_text(json.dumps(dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),
        stdout=str(stdout),stderr=str(stderr)),indent=2)+'\n')
value={}
for name in (prefix+'-client.json',prefix+'-installation.json',prefix+'-chunk.json',prefix+'-capture-error.json',prefix+'-error.json'):
    if (DEST/name).exists():value[name]=json.loads((DEST/name).read_bytes())
print(json.dumps(value))
