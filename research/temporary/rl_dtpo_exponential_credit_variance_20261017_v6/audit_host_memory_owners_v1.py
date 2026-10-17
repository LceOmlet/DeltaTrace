"""One read-only whole-host PSS inventory, including environment processes.

Do not sum fork RSS, create a training logger, stop processes, or release memory.
Keep the original observer's physical driver pool separate from process PSS.
Command lines/environment are inspected only for known ownership markers; their
raw contents are never saved or printed.
"""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
started = time.time()
manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
owners = {}
roots = []
historical_roots = {3805315: 'AppWorld_DTPO_failed_v6'}
known_run_paths = {str(ROOT / 'runs/appworld-dtpo-reward-target-exponential-variance-formal-20261017-v6'): 'AppWorld_DTPO_failed_v6'}
for job in manifest['jobs']:
    if job.get('task') not in ('TextCraft', 'AppWorld') or job.get('method') not in ('dt', 'ppo'):
        continue
    label = job['task'] + '_' + job['method']
    source = json.loads(Path(job['source_receipt']).read_bytes()) if job.get('source_receipt') else {}
    for previous in (job.get('previous_job_identity'), source.get('previous_job_identity')):
        if isinstance(previous, dict) and previous.get('pid'):
            historical_roots[previous['pid']] = label + '_previous'
    if job.get('output'):
        known_run_paths[str(Path(job['output']).parent)] = label
    try:
        process = psutil.Process(job['pid'])
        if process.create_time() != job['birth']:
            continue
        roots.append(dict(owner=label, pid=process.pid, birth=process.create_time(), version=job.get('version')))
        for member in [process, *process.children(recursive=True)]:
            owners[member.pid] = label
    except psutil.Error:
        pass
for pid, birth, label in [(2993563, 1792233768.81, 'DT_capacity_diagnostic_execution4')]:
    try:
        process = psutil.Process(pid)
        if process.create_time() == birth:
            roots.append(dict(owner=label, pid=pid, birth=birth))
            for member in [process, *process.children(recursive=True)]:
                owners[member.pid] = label
    except psutil.Error:
        pass
root_pids = {r['pid'] for r in roots}
rows = []
unreadable = []
for process in psutil.process_iter(['pid', 'ppid', 'name', 'create_time']):
    try:
        info = process.info
        args = process.cmdline()
        joined = ' '.join(args).lower()
        name = info['name']
        metrics = {}
        for line in Path(f'/proc/{process.pid}/smaps_rollup').read_text().splitlines():
            key, sep, value = line.partition(':')
            if sep and key in ('Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Pss_Shmem', 'Private_Clean', 'Private_Dirty', 'SwapPss'):
                metrics[key + '_KiB'] = int(value.split()[0])
        if process.create_time() != info['create_time']:
            raise psutil.NoSuchProcess(process.pid)
        owner = owners.get(process.pid, 'outside_current_training_trees')
        markers = [s for s in ('compile_worker', 'torch._inductor', 'uvicorn', 'appworld.serve',
                               'phi_agents', 'appworld_evals', 'rollout_rpc', 'gcs_server',
                               'raylet', 'dashboard', 'resource_tracker') if s in joined]
        safe_scripts = sorted(set(Path(a).name for a in args if len(a) < 250 and a.endswith('.py')
                                  and all(c.isalnum() or c in '._-' for c in Path(a).name)))
        category = 'other_or_unresolved_process'
        if process.pid in root_pids:
            category = 'training_or_diagnostic_driver'
        elif 'WorkerDict' in name or 'ResourceWorker' in name:
            category = 'GPU_worker_including_inprocess_misc'
        elif 'TaskRunner' in name:
            category = 'training_controller_shared_rollout_environment_records'
        elif 'compile_worker' in joined or 'torch._inductor' in joined:
            category = 'Torch_compile_process'
        elif ('uvicorn' in joined and 'appworld' in joined) or 'appworld.serve' in joined or ('appworld' in [Path(a).name.lower() for a in args[:3]] and 'serve' in args):
            category = 'AppWorld_environment_server'
        elif 'appworld_evals' in joined or 'rollout_rpc' in joined or ('phi_agents' in joined and 'appworld' in joined):
            category = 'environment_or_sampling_owner'
        elif any(s in joined for s in ('gcs_server', '/raylet', 'ray/dashboard', 'ray/_private', 'resource_tracker')):
            category = 'Ray_or_Python_infrastructure'
        session, group = os.getsid(process.pid), os.getpgid(process.pid)
        matches = [label for path, label in known_run_paths.items() if path in ' '.join(args)]
        # Look for exact run paths only. Do not retain any environment value.
        if owner == 'outside_current_training_trees' and (metrics.get('Pss_KiB', 0) >= 262144 or 'environment' in category):
            try:
                values = process.environ().values()
                matches.extend(label for path, label in known_run_paths.items() if any(path in value for value in values))
            except psutil.Error:
                pass
        old_session = historical_roots.get(session) or historical_roots.get(group)
        rows.append(dict(pid=process.pid, ppid=info['ppid'], birth=info['create_time'], name=name,
                         owner=owner, category=category, session_id=session, process_group=group,
                         exact_historical_session=old_session, exact_run_path_markers=sorted(set(matches)),
                         safe_script_basenames=safe_scripts, command_markers=markers, **metrics))
    except (psutil.Error, OSError, ValueError) as exc:
        unreadable.append(dict(pid=process.pid, error=type(exc).__name__))

by_pid = {r['pid']: r for r in rows}
# Forked compiler children sometimes inherit a generic Python command line.
for row in rows:
    if row['category'] not in ('other_or_unresolved_process', 'Ray_or_Python_infrastructure'):
        continue
    parent = row['ppid']
    seen = set()
    while parent in by_pid and parent not in seen:
        seen.add(parent)
        ancestor = by_pid[parent]
        if ancestor['category'] == 'Torch_compile_process':
            row['category'] = 'Torch_compile_process'
            row['category_evidence'] = 'descendant_of_explicit_compile_worker_command'
            break
        parent = ancestor['ppid']
grouped = defaultdict(lambda: dict(processes=0, Pss_KiB=0, Pss_Anon_KiB=0, Pss_File_KiB=0, Pss_Shmem_KiB=0))
for row in rows:
    group = grouped[(row['owner'], row['category'])]
    group['processes'] += 1
    for field in ('Pss_KiB', 'Pss_Anon_KiB', 'Pss_File_KiB', 'Pss_Shmem_KiB'):
        group[field] += row.get(field, 0)
original_observer_path = ROOT / 'receipts/appworld-formal-training-20261017-v1/latest-host-RSS-PSS-XTT-v1.json'
observer = json.loads(original_observer_path.read_bytes())
meminfo = {k: int(v.strip().split()[0]) for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
result = dict(status='completed_read_only_whole_host_PSS_inventory', started=started, finished=time.time(),
              source=str(Path(__file__)), source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              roots=roots, historical_root_markers=historical_roots,
              process_PSS_total_KiB=sum(r.get('Pss_KiB', 0) for r in rows),
              groups=[dict(owner=k[0], category=k[1], **v) for k, v in grouped.items()],
              processes=sorted(rows, key=lambda x: x.get('Pss_KiB', 0), reverse=True),
              unreadable_or_exited=unreadable, proc_meminfo_KiB=meminfo,
              original_resource_observer=dict(path=str(original_observer_path), collected=observer['collected'],
                  cgroup_usage_bytes=observer['cgroup_usage_bytes'], cgroup_stat=observer['cgroup_memory_stat_bytes'],
                  driver_XTT_single_pool_KiB=observer['driver_xtt_reports_KiB']['/sys/class/drm/card1/device/mem_info_xtt_used'],
                  cgroup_OOM=observer['cgroup_oom_control']),
              scope='Process attribution is by exact live driver trees and explicit native command markers. Mixed controller/worker heaps remain mixed. Driver XTT and OS cache are separate, not added to PSS as a guaranteed disjoint total.',
              training_mutations=0, signals=0, cache_release_calls=0, model_calls=0)
stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
path = AUDIT / ('host-memory-owner-inventory-' + stamp + '.json')
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     started=started, finished=result['finished'], process_count=len(rows),
                     PSS_GiB=result['process_PSS_total_KiB']/2**20,
                     groups=sorted([dict(owner=k[0],category=k[1],PSS_GiB=v['Pss_KiB']/2**20,processes=v['processes']) for k,v in grouped.items()],key=lambda x:x['PSS_GiB'],reverse=True),
                     top_processes=result['processes'][:12],
                     environment_outside_current_trees=[r for r in rows if r['owner']=='outside_current_training_trees' and ('environment' in r['category'] or r['exact_historical_session'])],
                     unreadable_count=len(unreadable), MemAvailable_GiB=meminfo['MemAvailable']/2**20,
                     driver_XTT_single_pool_GiB=result['original_resource_observer']['driver_XTT_single_pool_KiB']/2**20)))
