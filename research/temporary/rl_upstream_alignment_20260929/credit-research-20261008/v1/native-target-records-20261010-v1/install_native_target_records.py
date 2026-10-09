"""Update only the existing passive recorder through original VERL generic RPC."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import psutil
import ray

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
FORMAL=ROOT/'runs/textcraft-formal-stable-20261009-v1'
SOURCE=ROOT/'receipts/native-target-records-20261010-v1/reward_readout.py'
OLD_SHA='2609d93f91e9e16b8963ed5ca2226a00e28190005dbd51e56a214d3d35696eb2'
NEW_SHA='f0fbae610f1a8c22b65c6486cefb73eb5ff04c154178e0f822e679b600fa33bc'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_worker(worker):
    import qwen35_gdn_finite as gdn
    import reward_readout as module
    expected_births={987808:1791553850.00,989860:1791553867.51}
    assert psutil.Process().create_time()==expected_births[os.getpid()]
    assert digest(module.__file__)=='814cfe929b4afcc5ce3570450ea8aca269b1097db7abed83dc917df3f1d1cc9b'
    assert digest(gdn.__file__)=='7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656'
    assert digest(SOURCE)==NEW_SHA
    owners=[x for x in worker.worker_dict.values() if getattr(x,'_is_actor',False)]
    assert len(owners)==1
    owner=owners[0];readout=owner._deltatrace_producer.direct_readout
    assert type(readout) is module.DirectActionTargetReadout
    before=type(readout)._record_joint_dt_batch
    assert digest(before.__code__.co_filename)==OLD_SHA
    assert digest(type(readout).trajectories.__code__.co_filename)==OLD_SHA
    assert (owner.config.model.lora_rank,owner.config.model.lora_alpha)==(8,16)
    assert owner.config.actor.ppo_micro_batch_size_per_gpu==4
    expected_directory=FORMAL/'credit-records'/f'rank{owner.rank}-pid{os.getpid()}'
    assert readout._diagnostic_directory==str(expected_directory)
    tree=ast.parse(SOURCE.read_bytes())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DirectActionTargetReadout')
    helper=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_record_joint_dt_batch')
    namespace=dict(before.__globals__,__file__=str(SOURCE))
    exec(compile(ast.Module(body=[helper],type_ignores=[]),str(SOURCE),'exec'),namespace)
    trajectory=type(readout).trajectories
    type(readout)._record_joint_dt_batch=namespace[helper.name]
    assert type(readout).trajectories is trajectory
    return dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=owner.rank,unix=time.time(),
      numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8',
      GDN_path=gdn.__file__,GDN_resolved=str(Path(gdn.__file__).resolve()),GDN_sha256=digest(gdn.__file__),
      previous_recorder_source=before.__code__.co_filename,previous_recorder_sha256=OLD_SHA,
      active_recorder_source=str(SOURCE),active_recorder_sha256=NEW_SHA,
      trajectory_method_source=trajectory.__code__.co_filename,trajectory_method_sha256=OLD_SHA,
      trajectory_method_object_unchanged=True,record_directory=str(expected_directory),
      lora_rank=8,lora_alpha=16,per_card_microbatch=4,model_DT_optimizer_calls=0)


if __name__=='__main__':
    assert psutil.Process(982372).create_time()==1791553809.84
    assert digest(FORMAL/'source.json')=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
    verified=json.loads((SOURCE.parent/'cpu-result.json').read_bytes())
    assert verified['status']=='passed_existing_real_B4_CPU_serialization_only'
    assert verified['candidate_sha256']==NEW_SHA and verified['original_math_AST_unchanged']
    actual=json.loads((FORMAL/'source.json').read_bytes())['environment']
    for key in ['PYTHONPATH','LD_LIBRARY_PATH','MACA_PATH']:
        if key in actual:assert os.environ.get(key)==actual[key],key
    out=FORMAL/'runtime-overrides/native-target-records-20261010-v1.json'
    assert not out.exists()
    record=dict(started_unix=time.time(),pid=os.getpid(),birth=psutil.Process().create_time(),
      source_commit=sys.argv[1],script_sha256=digest(__file__),formal_pid=982372,
      formal_birth=1791553809.84,source_sha256=digest(FORMAL/'source.json'),complete=False,
      scope=__doc__,numerical_tolerance_changed=False,QVA_whitening_PPO_changed=False)
    out.write_text(json.dumps(record,indent=2)+'\n')
    ray.init(address='127.0.0.1:58837',log_to_driver=False,ignore_reinit_error=False)
    try:
        named=[x for x in ray.util.list_named_actors(all_namespaces=True)
               if x['name'].startswith('zagXiM') and 'register_center' not in x['name']]
        assert len(named)==2
        refs=[ray.get_actor(x['name'],namespace=x['namespace']).execute_with_func_generator.remote(install_worker) for x in named]
        record['named_actors']=named;out.write_text(json.dumps(record,indent=2)+'\n')
        results=ray.get(refs)
        record.update(complete=True,completed_unix=time.time(),results=results)
        out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    finally:
        ray.shutdown()
