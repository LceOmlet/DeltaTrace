"""Project saved original receipts into the existing plot owner's schema.

Offline only. Original console parsing and plotting remain owned by
plot_training_progress.py; no training, environment or numerical code runs.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PHASE = HERE / 'formal-phase-1791363603.json'
METRIC = HERE / 'textcraft-step7-complete-original-1791363676.json'
OWNER = REPO / 'experiments/rl/plot_training_progress.py'


def binding(path):
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    phase = json.loads(PHASE.read_bytes())
    raw = json.loads(METRIC.read_bytes())
    assert raw['source_sha256'] == '5013ebc878b972a5f52817f7c7de7ae55ddae7f1d61609e943e1ab55bf59b993'
    spec = importlib.util.spec_from_file_location('original_training_plot_owner', OWNER)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    records = [dict(text=line, path=raw['taskrunner_source']['path'], line=number)
               for number, line in enumerate(raw['taskrunner_original_read_text'].splitlines(), 1)
               if re.search(r'\bstep:\d+ - ', line)]
    metrics, phases = owner.parse_logs(records)
    assert [m['step'] for m in metrics] == list(range(1, 8))
    # This displayed rate is derived from the saved official binary scores,
    # not DT nonzero requests or an environment evaluation implemented here.
    rollout = raw['rollout7']
    assert rollout['rows'] == rollout['score_count'] == 256
    assert rollout['score_nonbinary'] == 0 and rollout['invalid_score_count'] == 0
    metrics[-1]['values']['episode/success_rate'] = rollout['score_ones'] / rollout['rows']
    metrics[-1]['derived_display_metric_source'] = dict(
        key='episode/success_rate', path=rollout['path'], sha256=rollout['sha256'],
        score_ones=rollout['score_ones'], rows=rollout['rows'],
        scope='Display-only fraction of saved official score==1; not an original console metric or independent evaluation')
    jobs = []
    for observed in phase['jobs']:
        job = observed['job']
        textcraft = job['task'] == 'TextCraft'
        jobs.append(dict(task=job['task'], gpu=job['devices'][0], devices=job['devices'],
            run_dir=job['output'], budget=330 if textcraft else 200,
            pid_alive=True, exit_code=None, checkpoint_step=None,
            metrics=metrics if textcraft else [], phases=phases if textcraft else [],
            records=records if textcraft else [], errors=[],
            phase_label='第8轮原采样' if textcraft else '首轮原生采样；198/240条已收集',
            identity=dict(pid=job['pid'], birth=job['observed_process_created_unix'],
                          source_sha256=observed['source_sha256'])))
    # Representation-only decoding of the original physical mx-smi snapshot.
    gpus = {}; gpu = None
    for line in phase['physical_gpu'].splitlines():
        match = re.match(r'\|\s*\d+\s+MetaX\s+\S+\s*\|\s*(\d+)\s', line)
        if match:
            gpu = match[1]
        match = re.search(r'(\d+)/(\d+) MiB', line)
        if match and gpu is not None:
            gpus[gpu] = dict(used_gib=int(match[1]) / 1024, total_gib=int(match[2]) / 1024)
            gpu = None
    snapshot = dict(collected_at=phase['observed_unix'],
        started=min(j['job']['started_unix'] for j in phase['jobs']),
        source_commit='d5b879d7acd49a5e7ba7550a52cb53a0f993d0d5',
        source_label='TextCraft source5013ebc8 · AppWorld source24b9e671 · 资源17:00 / metric补采17:01',
        root='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922',
        jobs=jobs, memory_gib=phase['cgroup_usage_bytes'] / 2**30,
        memory_limit_gib=None, gpus=gpus, mx_smi=phase['physical_gpu'],
        projection_sources=[binding(PHASE), binding(METRIC), binding(OWNER), binding(Path(__file__))],
        scope='Offline display of original current-run logs and saved official scores; missing metrics remain missing')
    path = HERE / 'current-formal-plot-snapshot.json'
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(binding(path)))


if __name__ == '__main__':
    main()
