"""One bounded read-only snapshot of the existing fresh AppWorld job."""
import collections
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import time

import psutil


ROOT = pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')


def log_observation(path):
    patterns = {
        'loop_collection': r'n_rollouts_collected|loop_transport|rollouts collected|cancelled|canceled',
        'generation': r'sample_tokens|generated_tokens|generation|requests.*tokens',
        'dt': r'\[DeltaTrace readout\]|\[DT EOS plan\]|\[DT EOS minibatch\]',
        'actor_update': r'\[native_host_cache\].*update_actor|timing_s/update_actor|actor/grad_norm',
        'whitening_named': r'masked_whiten|whitening|whitened',
        'errors': r'Traceback \(most recent call last\)|OutOfMemoryError|out of memory|CUDA error|RuntimeError:|AssertionError:|FATAL|\bgrad_norm[:=].*nan',
        'warnings': r'WARNING|Warning:|recompile limit|graph break|Graph break',
        'step_metric': r'\bstep:[0-9]+|\bstep: [0-9]+|timing_s/step|critic/score/mean',
        'rpc': r'compute_log_prob|compute_ref_log_prob|compute_dt_token_advantages|update_actor|Setting global step',
    }
    compiled = {key: re.compile(value) for key, value in patterns.items()}
    counts = collections.Counter()
    recent = {key: collections.deque(maxlen=8) for key in patterns}
    tail = collections.deque(maxlen=10)
    sha = hashlib.sha256()
    limit = path.stat().st_size
    offset = 0
    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            if offset >= limit:
                break
            raw = raw[:limit-offset]
            sha.update(raw)
            text = raw.decode(errors='replace').rstrip()
            line = dict(line=number, byte_offset=offset, text=text[:2600],
                        original_line_bytes=len(raw), truncated_text=len(text)>2600)
            tail.append(line)
            for key, pattern in compiled.items():
                if pattern.search(text):
                    counts[key] += 1
                    recent[key].append(line)
            offset += len(raw)
    return dict(path=str(path), read_prefix_bytes=offset, sha256=sha.hexdigest(),
        scope='Fixed prefix size at file-open observation; later append excluded.',
        match_counts=dict(counts), recent={key:list(value) for key,value in recent.items()},
        tail=list(tail))


def main():
    started = time.time()
    manifest_path = ROOT / 'active-training.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    app = next(item for item in manifest['jobs'] if item['task']=='AppWorld')
    PID = int(app['pid'])
    BIRTH = float(app['observed_process_created_unix'])
    OUT = pathlib.Path(app['output'])
    process = psutil.Process(PID)
    birth = process.create_time()
    assert abs(birth-BIRTH) < .02, (PID, birth, BIRTH)
    children = process.children(recursive=True)
    selected = []
    for child in children:
        try:
            name = child.name()
            if 'TaskRunner' in name or 'WorkerDict' in name:
                memory = child.memory_full_info()
                selected.append(dict(pid=child.pid, birth=child.create_time(), name=name,
                    status=child.status(), rss_bytes=memory.rss, pss_bytes=memory.pss))
        except psutil.Error:
            pass
    source_path = OUT / 'source.json'
    source_raw = source_path.read_bytes()
    source = json.loads(source_raw)
    owner_sources = {}
    for name in ('verl/trainer/ppo/ray_trainer.py', 'verl/workers/actor/dp_actor.py',
                 'verl/workers/fsdp_workers.py'):
        path = pathlib.Path(app['verl_root']) / name
        owner_sources[str(path)] = dict(actual_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_receipt_expected_sha256=source['owner_head_sha256'][name])
    ray_sessions = list(pathlib.Path('/tmp/ray').glob('session_*_'+str(PID)))
    files = [OUT / 'train.log']
    selected_pids = {str(item['pid']) for item in selected}
    for session in ray_sessions:
        for path in (session/'logs').glob('worker-*'):
            if path.suffix in ('.out','.err') and path.stem.rsplit('-',1)[-1] in selected_pids:
                files.append(path)
    logs = [log_observation(path) for path in files if path.is_file()]
    cgroup = {}
    for line in pathlib.Path('/proc',str(PID),'cgroup').read_text().splitlines():
        hierarchy, controllers, relative = line.split(':',2)
        if hierarchy=='0':
            base = pathlib.Path('/sys/fs/cgroup') / relative.lstrip('/')
            cgroup['path'] = str(base)
            for name in ('memory.current','memory.max','memory.events','memory.stat'):
                path = base / name
                if path.exists():
                    text = path.read_text().strip()
                    cgroup[name] = text if name!='memory.stat' else {
                        key:int(value) for key,value in (row.split() for row in text.splitlines())
                        if key in ('anon','file','kernel','slab','pagetables','sock')}
        elif 'memory' in controllers.split(','):
            base = pathlib.Path('/sys/fs/cgroup/memory')
            cgroup['path'] = str(base)
            for name in ('memory.usage_in_bytes','memory.limit_in_bytes','memory.failcnt',
                         'memory.oom_control','memory.stat'):
                path = base / name
                if path.exists():
                    text = path.read_text().strip()
                    cgroup[name] = text if name!='memory.stat' else {
                        key:int(value) for key,value in (row.split() for row in text.splitlines())}
    stacks = []
    profiler = shutil.which('py-spy')
    if profiler:
        for child in selected:
            if 'TaskRunner' in child['name']:
                call = subprocess.run([profiler,'dump','--nonblocking','--pid',str(child['pid'])],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=12)
                stacks.append(dict(pid=child['pid'], returncode=call.returncode,
                                   stdout=call.stdout, stderr=call.stderr))
    physical = subprocess.run(['mx-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,timeout=12)
    checkpoint = OUT / 'checkpoints/latest_checkpointed_iteration.txt'
    result = dict(observed_unix=time.time(), duration_seconds=time.time()-started,
        scope='Read existing process, official logs, files and resource counters only; no model, DT, rollout, backward, checkpoint loading or optimizer call.',
        driver=dict(pid=PID, expected_birth=BIRTH, observed_birth=birth, status=process.status(),
                    cmdline=process.cmdline()), selected_processes=selected,
        source=dict(path=str(source_path), sha256=hashlib.sha256(source_raw).hexdigest()),
        active_manifest=dict(path=str(manifest_path), sha256=hashlib.sha256(manifest_raw).hexdigest(),
                             AppWorld_job=app), original_owner_sources=owner_sources,
        ray_sessions=[str(path) for path in ray_sessions], logs=logs, TaskRunner_stacks=stacks,
        host=dict(available_bytes=psutil.virtual_memory().available, total_bytes=psutil.virtual_memory().total),
        cgroup=cgroup, physical=dict(returncode=physical.returncode,stdout=physical.stdout,stderr=physical.stderr),
        latest_checkpoint_marker=dict(path=str(checkpoint),exists=checkpoint.exists(),
            value=checkpoint.read_text().strip() if checkpoint.exists() else None),
        output_file_names=[path.name for path in OUT.iterdir() if path.is_file()])
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
