"""Read both current formal AppWorld runs with the unchanged owner log reader."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'


def resource_snapshot(include_pss):
    """Observe native proc/sysfs counters; do not call training or CUDA APIs."""
    def counters(path):
        result = {}
        for line in Path(path).read_text().splitlines():
            words = line.replace(':', '').split()
            if len(words) >= 2 and words[1].isdigit():
                result[words[0]] = int(words[1])
        return result

    manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
    memory = Path('/sys/fs/cgroup/memory')
    snapshot = dict(collected=time.time(), observer_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    proc_meminfo_KiB=counters('/proc/meminfo'),
                    cgroup_usage_bytes=int((memory / 'memory.usage_in_bytes').read_text()),
                    cgroup_limit_bytes=int((memory / 'memory.limit_in_bytes').read_text()),
                    cgroup_memory_stat_bytes=counters(memory / 'memory.stat'),
                    cgroup_oom_control=counters(memory / 'memory.oom_control'), jobs=[],
                    driver_xtt_reports_KiB={}, pss_sample=include_pss,
                    formal_training_mutations=0, GPU_or_model_calls=0)
    # These card interfaces report the same host pool. Preserve both readings;
    # never sum the eight aliases or add this counter to RSS as a disjoint total.
    for card in (1, 8):
        p = Path(f'/sys/class/drm/card{card}/device/mem_info_xtt_used')
        if p.exists():
            snapshot['driver_xtt_reports_KiB'][str(p)] = int(p.read_text().split()[0])
    ttm = Path('/sys/class/drm/ttm/memory_accounting/kernel/used_memory')
    if ttm.exists():
        snapshot['ttm_kernel_used_memory_KiB'] = int(ttm.read_text().split()[0])
    for job in manifest['jobs']:
        if job.get('task') not in ('TextCraft', 'AppWorld'):
            continue
        row = dict(task=job['task'], method=job['method'], version=job.get('execution_version', job.get('version')),
                   pid=job['pid'], birth=job['birth'], alive=False, processes=[], observed_process_count=0)
        try:
            root = psutil.Process(job['pid'])
            if root.create_time() != job['birth'] or root.status() == psutil.STATUS_ZOMBIE:
                snapshot['jobs'].append(row)
                continue
            row['alive'] = True
            total_pss = 0
            missing = []
            processes = []
            for process in [root, *root.children(recursive=True)]:
                try:
                    status = counters(f'/proc/{process.pid}/status')
                    current = dict(pid=process.pid, birth=process.create_time(), phase=process.name(),
                                   RSS_bytes=process.memory_info().rss,
                                   peak_RSS_KiB=status.get('VmHWM'),
                                   oom_score_adj=int(Path(f'/proc/{process.pid}/oom_score_adj').read_text()))
                    if include_pss:
                        pss = counters(f'/proc/{process.pid}/smaps_rollup').get('Pss')
                        current['PSS_KiB'] = pss
                        if pss is None:
                            missing.append(process.pid)
                        else:
                            total_pss += pss
                    processes.append(current)
                except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError, ProcessLookupError):
                    missing.append(process.pid)
            row['observed_process_count'] = len(processes)
            selected = sorted(processes, key=lambda p: p['RSS_bytes'], reverse=True)[:12]
            ids = {p['pid'] for p in selected}
            selected += [p for p in processes if p['pid'] not in ids and
                         ('WorkerDict' in p['phase'] or 'TaskRunner' in p['phase'])]
            row['processes'] = selected
            row['unreadable_or_exited_processes'] = missing
            if include_pss:
                row['observed_tree_PSS_KiB'] = total_pss
                row['PSS_complete'] = not missing
            # Shared fork pages make a sum of RSS unsuitable as physical usage.
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        snapshot['jobs'].append(row)
    return snapshot


if '--resources-watch' in sys.argv:
    log = AUDIT / 'native-host-RSS-PSS-XTT-timeseries-v1.jsonl'
    latest = AUDIT / 'latest-host-RSS-PSS-XTT-v1.json'
    last_pss = 0.0
    while True:
        started = time.monotonic()
        try:
            include_pss = started - last_pss >= 60
            sample = resource_snapshot(include_pss)
            if include_pss:
                last_pss = started
            sample['collection_seconds'] = time.monotonic() - started
            latest.write_text(json.dumps(sample, indent=2) + '\n')
            with log.open('a') as stream:
                stream.write(json.dumps(sample, separators=(',', ':')) + '\n')
        except Exception as exc:
            with (AUDIT / 'native-resource-observer-errors-v1.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(collected=time.time(), error_type=type(exc).__name__,
                                             error=str(exc), training_actions=0)) + '\n')
        time.sleep(max(0, 10 - (time.monotonic() - started)))

parser_path = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(parser_path.read_bytes()).hexdigest() == 'c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner = runpy.run_path(str(parser_path), run_name='current_appworld_original_log_reader')
manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
jobs = [j for j in manifest['jobs'] if j.get('task') == 'AppWorld' and j.get('method') in ('dt','ppo')]
snapshot = dict(collected=time.time(), jobs=[], parser=str(parser_path), formal_training_mutations=0,
                host_MemAvailable_GiB=psutil.virtual_memory().available/2**30)
for job in jobs:
    current = dict(job)
    processes = []
    alive = False
    try:
        process = psutil.Process(job['pid'])
        alive = process.create_time() == job['birth'] and process.status() != psutil.STATUS_ZOMBIE
        if alive:
            children = process.children(recursive=True)
            gcs = next((p for p in children if p.name() == 'gcs_server'), None)
            if gcs:
                log_dir = next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--log_dir='))
                current['ray_session'] = str(Path(log_dir).parent)
            processes = [dict(pid=p.pid,birth=p.create_time(),phase=p.name()) for p in children if 'WorkerDict' in p.name()]
    except psutil.NoSuchProcess:
        pass
    records, errors = owner['read_job_logs'](current)
    metrics, phases = owner['parse_logs'](records)
    checkpoint = Path(job['checkpoints'])
    marker = checkpoint / 'latest_checkpointed_iteration.txt'
    latest = None if not marker.exists() else int(marker.read_text().strip())
    completed_files = []
    if latest is not None:
        directory = checkpoint / f'global_step_{latest}'
        completed_files = [dict(path=str(p),bytes=p.stat().st_size) for p in directory.rglob('*') if p.is_file()]
    row = dict(task='AppWorld',method=job['method'],version=job['version'],pid=job['pid'],birth=job['birth'],
               alive=alive,workers=processes,ray_session=current.get('ray_session'),
               completed_iterations=max((m['step'] for m in metrics),default=0),total_iterations=200,
               metrics=metrics,phases=phases,errors=errors,checkpoint_path=str(checkpoint),checkpoint_interval=2,
               latest_checkpointed_iteration=latest,latest_completed_checkpoint_files=completed_files,
               source_receipt=job['source_receipt'])
    snapshot['jobs'].append(row)
(AUDIT / 'latest-native-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print(json.dumps({**snapshot,'jobs':[{k:v for k,v in row.items() if k not in ('metrics','phases','errors','latest_completed_checkpoint_files')}
                                   | dict(latest_metrics=row['metrics'][-1:],error_marker_count=len(row['errors']))
                                   for row in snapshot['jobs']]},indent=2))
