"""Set an existing DT option through VERL's existing serialized worker RPC.

Operational command, not a training hook or a replacement scheduler. The
current synchronous actor call finishes first. No model, numerical function,
trajectory, optimizer or upstream source is replaced. Keep this client alive
until the original Ray ObjectRef returns; never resubmit on an observation
timeout. Root manifests are updated by the operator after inspecting receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import ray
from ray.core.generated.gcs_pb2 import ActorTableData


def set_existing_option(worker, expected_pid, environment_path):
    import os
    import torch

    assert os.getpid() == expected_pid, (os.getpid(), expected_pid)
    producer = worker._deltatrace_producer
    runner = producer.runner
    before = runner.offload_replay_mixer
    runner.offload_replay_mixer = False
    os.environ['DT_ENVIRONMENT_JSON'] = environment_path
    return dict(pid=os.getpid(), before=before, after=runner.offload_replay_mixer,
                torch_cpu_threads=torch.get_num_threads(),
                environment_json=os.environ['DT_ENVIRONMENT_JSON'],
                applied_at=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', choices=['Webshop', 'Sokoban', 'AppWorld'], required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    args = parser.parse_args()
    jobs = json.loads((args.root/'formal-training.json').read_text())['jobs']
    job, = [j for j in jobs if j['task'] == args.task]
    source = Path(job['settings']['DT_ENVIRONMENT_JSON'])
    config = json.loads(source.read_text())
    assert config['qwen35']['dt_offload_replay_mixer'] is True
    config['qwen35']['dt_offload_replay_mixer'] = False
    target = Path(job['run_dir'])/'environment-mixer-resident.json'
    encoded = json.dumps(config, indent=2)+'\n'
    if target.exists():
        assert target.read_text() == encoded
    else:
        target.write_text(encoded)
    receipt = Path(job['run_dir'])/'mixer-residency-rpc.json'
    assert not receipt.exists(), 'Inspect the original request; do not submit twice'
    result = dict(task=args.task, worker_pid=args.worker_pid, started=time.time(),
                  previous_environment=str(source), environment_json=str(target),
                  environment_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  only_config_change={'qwen35.dt_offload_replay_mixer': [True, False]},
                  owner_method='actor_rollout_execute_with_func_generator')
    session = Path(job['ray_session'])
    host = json.loads((session/'node_ip_address.json').read_text())['node_ip_address']
    port = next(iter(json.loads((session/'ports_by_node.json').read_text()).values()))['gcs_server_port']
    ray.init(address=f'{host}:{port}', logging_level='ERROR', log_to_driver=False)
    try:
        actors = ray._private.state.actors()
        actor_id, = [aid for aid, a in actors.items()
                     if a['Pid'] == args.worker_pid and a['State'] == 'ALIVE']
        # Use Ray's existing GCS identity to obtain the owning namespace.
        raw = ray._private.state.state._global_state_accessor.get_actor_info(ray.ActorID.from_hex(actor_id))
        actor_data = ActorTableData.FromString(raw)
        actor = ray.get_actor(actor_data.name, namespace=actor_data.ray_namespace)
        result.update(actor_id=actor_id, actor_name=actor_data.name,
                      ray_namespace=actor_data.ray_namespace, status='submitting')
        receipt.write_text(json.dumps(result, indent=2)+'\n')
        ref = actor.actor_rollout_execute_with_func_generator.remote(
            func=set_existing_option, expected_pid=args.worker_pid, environment_path=str(target))
        result.update(status='queued_owner_rpc', object_ref=ref.hex())
        receipt.write_text(json.dumps(result, indent=2)+'\n')
        result.update(status='applied', result=ray.get(ref))
        receipt.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    except Exception as exc:
        result.update(status='request_failed', error_type=type(exc).__name__, error=str(exc))
        receipt.write_text(json.dumps(result, indent=2)+'\n')
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
