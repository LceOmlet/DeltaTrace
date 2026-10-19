"""Extend the existing read-only metadata RPC with the actual clip/cache flags."""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/inspect_native_workers.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='7f3c940abc65da9317a9e5f74182f5080cae066a116ddd6beaebc7476b36bb1d'
code=owner.read_text().replace('textcraft-eight-gpu-sequential-20261017-v1',
    'textcraft-eight-gpu-v9-negative-clip2-20261019-v4')
code=code.replace("else 'eight-GPU-initial-v1')","else 'v9-negative2-eight-GPU-initial-v1')")
anchor="    result['DP_world_size'] = actor.world_size"
assert code.count(anchor)==1
code=code.replace(anchor,anchor+'''
    result['DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP']=os.environ.get('DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP')
    result['DT_NATIVE_HOST_CACHE_BATCH']=os.environ.get('DT_NATIVE_HOST_CACHE_BATCH')
    result['PYTORCH_CUDA_ALLOC_CONF']=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    result['actor_optimizer_step_values']=sorted({float(s['step'].item()) for s in actor.actor_optimizer.state.values() if 'step' in s})
    engine=getattr(getattr(actor,'rollout',None),'inference_engine',None)
    engine=getattr(engine,'llm_engine',None)
    cache=getattr(engine,'cache_config',None)
    scheduler=getattr(engine,'scheduler_config',None)
    result['rollout_native_cache_config']={key:getattr(cache,key,None) for key in ('enable_prefix_caching','mamba_cache_mode','block_size')}
    result['rollout_native_scheduler_config']={key:getattr(scheduler,key,None) for key in ('max_num_seqs','max_num_batched_tokens','enable_chunked_prefill')}
''')
exec(compile(code,str(owner),'exec'),globals())
