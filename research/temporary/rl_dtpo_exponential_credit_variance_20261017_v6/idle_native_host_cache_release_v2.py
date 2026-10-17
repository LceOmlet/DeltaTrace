"""Release unused pinned buffers on the same idle native PPO workers.

Use the existing Ray execution hook and the already accepted PyTorch owner API.
No model, loss, optimizer, sampler, or checkpoint implementation is provided.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
job, = [j for j in manifest['jobs'] if j.get('task') == 'TextCraft' and j.get('method') == 'ppo']
source = json.loads(Path(job['source_receipt']).read_bytes())
receipt = Path(job['output']).parent / 'runtime-overrides/native-idle-host-cache-release-after-critic103-20261017-v2.json'

if receipt.exists():
    print(json.dumps(dict(status='completed_resource_action_retained', receipt=str(receipt))))
    raise SystemExit(0)

if '--client' not in sys.argv:
    env = dict(os.environ)
    env.update(source['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    result = subprocess.run([env['VENV_PYTHON'], '-X', 'utf8', __file__, '--client'],
                            env=env, capture_output=True, text=True)
    (AUDIT / 'idle-native-host-cache-release-v2-client.stdout').write_text(result.stdout)
    (AUDIT / 'idle-native-host-cache-release-v2-client.stderr').write_text(result.stderr)
    print(result.stdout, end='')
    if result.returncode:
        print(result.stderr, file=sys.stderr, end='')
    raise SystemExit(result.returncode)

import psutil
import ray

driver = psutil.Process(job['pid'])
assert driver.pid == 3823147 and driver.create_time() == job['birth'] == 1792200692.49
workers = [p for p in driver.children(recursive=True) if 'WorkerDict' in p.name()]
assert {p.pid for p in workers} == {3874156, 3876616}
assert all(p.name() == 'ray::WorkerDict' for p in workers), 'Only idle workers may be queried'
gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))


def release_native_unused_host_buffers(worker):
    import os
    from pathlib import Path
    import time
    import psutil
    import torch

    expected = {3874156: 1792201135.09, 3876616: 1792201155.53}
    process = psutil.Process()
    assert process.create_time() == expected[process.pid]
    assert os.environ.get('VERL_RELEASE_UNUSED_HOST_CACHE') == '1'
    assert torch.cuda.is_initialized()

    def roles():
        result = []
        for name, item in worker.worker_dict.items():
            entry = dict(name=name)
            for prefix in ('actor', 'critic'):
                scheduler = getattr(item, prefix + '_lr_scheduler', None)
                optimizer = getattr(item, prefix + '_optimizer', None)
                if scheduler is not None:
                    entry[prefix + '_scheduler_last_epoch'] = scheduler.last_epoch
                if optimizer is not None:
                    entry[prefix + '_optimizer_state_entries'] = len(optimizer.state)
            result.append(entry)
        return result

    def physical():
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))
        xtt = int(Path('/sys/class/drm/card1/device/mem_info_xtt_used').read_text().split()[0])
        return dict(collected=time.time(), MemAvailable_KiB=available, XTT_single_pool_KiB=xtt)

    before_roles = roles()
    before = dict(torch.cuda.memory.host_memory_stats())
    before_physical = physical()
    started = time.monotonic()
    torch._C._host_emptyCache()
    seconds = time.monotonic() - started
    after = dict(torch.cuda.memory.host_memory_stats())
    after_physical = physical()
    after_roles = roles()
    assert before_roles == after_roles
    return dict(pid=process.pid, birth=process.create_time(), before=before, after=after,
                before_physical=before_physical, after_physical=after_physical,
                native_release_seconds=seconds, roles_before=before_roles, roles_after=after_roles,
                role_scheduler_epochs_and_state_entry_counts_unchanged=True,
                model_calls=0, optimizer_updates=0, checkpoint_calls=0)


request = AUDIT / 'idle-native-host-cache-release-v2-job.json'
request.write_text(json.dumps(dict(pid=os.getpid(), birth=psutil.Process().create_time(),
                                  collected=time.time(), status='native_resource_rpc_pending',
                                  driver_pid=driver.pid, driver_birth=driver.create_time()), indent=2) + '\n')
ray.init(address=host + ':' + port, log_to_driver=False)
try:
    actors = [a for a in ray.util.list_named_actors(all_namespaces=True)
              if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
    assert len(actors) == 2
    results = ray.get([ray.get_actor(a['name'], namespace=a['namespace'])
                      .execute_with_func_generator.remote(func=release_native_unused_host_buffers)
                      for a in actors])
    record = dict(status='completed_native_unused_host_buffer_release', collected=time.time(),
                  task='TextCraft', method='ppo', driver_pid=driver.pid, driver_birth=driver.create_time(),
                  source=str(Path(__file__)), source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_owner_API='torch._C._host_emptyCache', results=results,
                  model_calls=0, optimizer_updates=0, checkpoint_calls=0, training_method_changes=0,
                  scope='Inactive pinned buffers only. Native reserved counters are bookkeeping; '
                        'physical host/XTT samples overlap other running jobs and are not exclusive attribution.')
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    request.write_text(json.dumps(dict(status=record['status'], completed=time.time(), receipt=str(receipt)), indent=2) + '\n')
    print(json.dumps(dict(status=record['status'], receipt=str(receipt), results=results)))
finally:
    ray.shutdown()
