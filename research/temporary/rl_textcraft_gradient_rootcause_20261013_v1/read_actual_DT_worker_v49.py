"""Read already constructed DT runner and capture bindings via native Ray RPC."""
import hashlib
import json
import os
from pathlib import Path
import psutil
import subprocess
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1'
launch=json.loads((RUN/'launch.json').read_bytes())
if '--native-env' not in sys.argv:
    env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
    raise SystemExit(subprocess.run([env['VENV_PYTHON'],'-u',__file__,'--native-env'],env=env).returncode)

def read_worker(worker):
    import hashlib,inspect,marshal,os,sys,time,psutil
    from pathlib import Path
    owners=[w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False)]
    assert len(owners)==1
    w=owners[0];producer=getattr(w,'_deltatrace_producer',None)
    assert producer is not None, 'Do not construct or import an uninitialized DT path'
    runner=producer.runner;backend=runner.capture_backend
    modules={}
    for name,m in sorted(sys.modules.items()):
        if (name in ('deltatrace_rollout','reward_readout','deltatrace_credit','finite_fla_gpu',
                     'vendor_fa_finite_bf16_d256','verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos')
            or name.endswith(('qwen35_gdn_finite','qwen35_dense_finite_runner','qwen35_decoder_finite',
                              'qwen35_retained_capture','qwen35_code_local_capture','profiles.official'))):
            path=getattr(m,'__file__',None)
            modules[name]=dict(path=path,SHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None)
    functions={}
    for label,f in [('actual_readout',producer.direct_readout.trajectories),
                    ('retained_decoder',backend.NativeDecoderCapture.retain),
                    ('retained_attention',backend.NativeDenseAttentionCapture.retain),
                    ('retained_GDN',backend.NativeGDNCapture.copy),
                    ('runner',type(runner).__init__)]:
        c=f.__code__
        functions[label]=dict(module=f.__module__,filename=c.co_filename,first_line=c.co_firstlineno,
                              code_SHA256=hashlib.sha256(marshal.dumps(c)).hexdigest(),names=list(c.co_names),
                              source=inspect.getsource(f) if label=='retained_decoder' else None)
    flags={name:getattr(runner,name,None) for name in (
        'copy_replay_captures','offload_replay_mixer','gdn_head_batch_size','defer_diagnostics',
        'compile_gdn_scalar_rules','pin_replay_host','gdn_gpu_capture_names','fa_coefficient_suffix',
        'gdn_coefficient_suffix','compact_gdn_captures','pin_root_host','reuse_native_prefix','native_conv_initial_states')}
    flags['boundary_scalar_attributes']={k:v for k,v in vars(runner.boundaries).items()
        if isinstance(v,(int,float,str,bool,type(None)))}
    return dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=w.rank,unix=time.time(),
        modules=modules,functions=functions,flags=flags,runner_class=type(runner).__module__+'.'+type(runner).__name__,
        capture_backend=backend.__name__,readout_class=type(producer.direct_readout).__name__,
        last_readout_report={k:v for k,v in producer.direct_readout.last_report.items() if k not in ('traces','group_stats','actual_context_lengths','causal_context_lengths','original_response_rows')},
        model_calls=0,DT_calls=0,optimizer_calls=0)

import ray
p=psutil.Process(launch['pid']);assert p.create_time()==launch['birth']
gcs=next(c for c in p.children(recursive=True) if c.name()=='gcs_server')
port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
ray.init(address=host+':'+port,log_to_driver=False)
try:
    names=[a for a in ray.util.list_named_actors(all_namespaces=True)
           if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
    assert len(names)==2
    refs=[ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=read_worker) for a in names]
    # A read-only RPC queues behind the existing 17-minute DT call.
    # This is a client wait, not a training timeout or a retry of DT.
    results=ray.get(refs,timeout=1800)
    report=dict(unix=time.time(),run=str(RUN),workers=sorted(results,key=lambda w:w['rank']),
                model_calls=0,DT_calls=0,optimizer_calls=0,worker_mutations=0)
    dest=OUT/'actual-DT-worker-bindings-v49.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(receipt=str(dest),report=report)))
finally:
    ray.shutdown()
