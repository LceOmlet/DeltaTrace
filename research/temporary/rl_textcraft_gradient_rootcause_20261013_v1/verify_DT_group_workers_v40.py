"""Reuse existing read/passive-record RPCs without changing native update math."""
import hashlib
import json
import os
from pathlib import Path
import psutil
import runpy
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
job = json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())
RUN = Path(job['run'])
launch = json.loads((RUN/'launch.json').read_bytes())
if '--native-env' not in sys.argv:
    env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
    r=subprocess.run([env['VENV_PYTHON'],'-u',__file__,'--native-env'],env=env)
    raise SystemExit(r.returncode)
import ray

def main():
    p=psutil.Process(job['pid']);assert p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE
    dest=RUN/'runtime-overrides/initial-state-and-existing-passive-records-v40.json'
    assert not dest.exists(), 'Read completed receipt rather than install again'
    gcs=next(c for c in p.children(recursive=True) if c.name()=='gcs_server')
    port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    reader_path=OUT/'read_worker_state.py'
    passive_path=OUT/'install_passive_records_native_graph.py'
    reader=runpy.run_path(str(reader_path),run_name='existing_state_reader_v40')['read_worker']
    passive=runpy.run_path(str(passive_path),run_name='existing_passive_recorder_v40')['install_worker']
    ray.init(address=host+':'+port,log_to_driver=False)
    try:
        names=[a for a in ray.util.list_named_actors(all_namespaces=True)
               if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
        assert len(names)==2, 'Workers not yet registered; do not call initialization healthy'
        actors=[ray.get_actor(a['name'],namespace=a['namespace']) for a in names]
        states=ray.get([a.execute_with_func_generator.remote(func=reader) for a in actors],timeout=600)
        states.sort(key=lambda w:w['rank'])
        for w in states:
            assert (w['optimizer_states'],w['scheduler_last_epoch'])==(0,0), 'Initial state not observed'
            c=w['configuration']
            assert (c['lora_rank'],c['lora_alpha'],c['microbatch'],c['minibatch_per_rank'],c['context'],c['enforce_eager'])==(8,16,4,32,32768,False)
            assert c['entropy']==.001 and c['dual_clip']==3 and c['ppo_epochs']==1
            for name,expected in [('verl.workers.actor.dp_actor','3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'),
                ('verl.trainer.ppo.core_algos','fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'),
                ('verl.workers.fsdp_workers','9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'),
                ('verl.utils.checkpoint.fsdp_checkpoint_manager','f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01')]:
                assert w['modules'][name]['sha256']==expected,name
        report=dict(unix=time.time(),job=job,workers=states,initial_optimizer0_scheduler0_verified=True,
            callback_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (reader_path,passive_path)},
            rootcause_repair_verified=False,complete=False)
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(report,indent=2)+'\n')
        passive.__globals__.update(FORMAL=str(RUN),WORKERS={w['pid']:w['birth'] for w in states})
        report['passive_records']=ray.get([a.execute_with_func_generator.remote(func=passive) for a in actors],timeout=600)
        report.update(complete=True,completed_unix=time.time(),status='actual_initial_owner_and_existing_passive_recorder_verified')
        dest.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(receipt=str(dest),report=report)),flush=True)
    finally:ray.shutdown()

if __name__=='__main__':main()
