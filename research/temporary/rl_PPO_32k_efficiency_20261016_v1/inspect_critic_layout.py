"""Read the existing PPO critic's actual module layout; no computation or mutation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1'
job=json.loads((ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json').read_bytes())
driver=psutil.Process(job['pid'])
assert driver.create_time()==job['birth']
if '--client' not in sys.argv:
    env=dict(os.environ);env.update(driver.environ());env['CUDA_VISIBLE_DEVICES']=''
    env.pop('MACA_VISIBLE_DEVICES',None)
    AUDIT.mkdir(exist_ok=True)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--client'],env=env))

import ray

def inspect_worker(worker):
    from collections import Counter
    import inspect
    import os
    from pathlib import Path
    import hashlib
    import psutil
    from verl.utils.fsdp_utils import FSDPModule
    from verl.utils import activation_offload
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf
    critic=worker.worker_dict['critic']
    model=critic.critic_module
    nodes=[dict(name=name,type=type(module).__name__,mro=[cls.__name__ for cls in type(module).__mro__[:4]])
           for name,module in model.named_modules() if isinstance(module,FSDPModule)]
    paths={name:Path(inspect.getsourcefile(obj)) for name,obj in (
        ('activation_offload',activation_offload),('HF_Qwen3_5',hf))}
    sources={name:dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()}
    first=next(model.parameters())
    return dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=critic.rank,
        model_type=type(model).__name__,config_model_type=model.config.model_type,
        no_split_modules=sorted(model._no_split_modules),wrapped_nodes=nodes,
        first_parameter_dtype=str(first.dtype),first_parameter_device=str(first.device),
        total_parameters=sum(p.numel() for p in model.parameters()),
        text_submodule_type=type(getattr(getattr(model,'model',None),'language_model',None)).__name__,
        native_activation_offload=critic.config.model.enable_activation_offload,
        native_gradient_checkpointing=critic.config.model.enable_gradient_checkpointing,
        sources=sources,model_calls=0,DT_calls=0,optimizer_updates=0,training_mutations=0)

gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
ray.init(address=host+':'+port,log_to_driver=False)
try:
    named=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
    assert len(named)==2
    results=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=inspect_worker) for a in named])
    path=AUDIT/'actual-critic-layout.json'
    path.write_text(json.dumps(dict(collected_unix=time.time(),job=dict(pid=job['pid'],birth=job['birth']),
        results=results,scope=__doc__,metadata_attempt1='Original no_split_modules set was read successfully but JSON serialization failed; normalized the metadata list, no training or model mutation.'),indent=2)+'\n')
    print(json.dumps(dict(receipt=str(path),results=[{**{k:v for k,v in item.items() if k!='wrapped_nodes'},
        'wrapped_counts':{branch:sum(branch in node['name'] for node in item['wrapped_nodes'])
                          for branch in ('visual.blocks','language_model.layers')}} for item in results]),indent=2))
finally:
    ray.shutdown()
