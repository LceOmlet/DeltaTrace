"""Read existing owner logs and render hourly plots; never import/change training.

The current owner uses VERL LocalLogger (three decimal places), not TensorBoard.
Read only the active manifest's Ray logs, retaining file/line provenance. Missing
metrics stay missing; pilot runs, formatted-action flags and tool_call_count are
not substituted for task performance. Remote collection uses only the stdlib.
"""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time


ANSI = re.compile(r'\x1b\[[0-9;]*m')
NUMBER = r'[-+]?(?:\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf)'
METRIC = re.compile(r'(?:^| - )([^ :]+):(' + NUMBER + r')(?= - |$)', re.I)


def parse_logs(records):
    """Decode the pinned owner's console format, keeping metric names intact."""
    metrics, phases = {}, []
    for row in records:
        line = ANSI.sub('', row['text']).strip()
        match = re.search(r'\bstep:(\d+) - ', line)
        if match:
            step = int(match[1])
            entry = metrics.setdefault(step, {'step': step, 'values': {}, 'sources': []})
            entry['values'].update({k: float(v) for k, v in METRIC.findall(line[match.end():])})
            entry['sources'].append({'path': row['path'], 'line': row['line']})
        if '[DT rollout] phase=' in line:
            fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
            phases.append({**fields, 'source': {'path': row['path'], 'line': row['line']}})
    return [metrics[k] for k in sorted(metrics)], phases


def collect(root):
    root = Path(root)
    manifest = json.loads((root / 'formal-training.json').read_text())
    out = {'collected_at': time.time(), 'started': manifest['started'],
           'source_commit': manifest['source_commit'], 'root': str(root), 'jobs': []}
    for job in manifest['jobs']:
        log_root = Path(job['ray_tmpdir']) / 'ray/session_latest/logs'
        records, errors = [], []
        # Direct worker output is authoritative; do not count its driver echo twice.
        for path in sorted(log_root.glob('worker-*.out'), key=lambda p: p.stat().st_mtime):
            with path.open(errors='replace') as stream:
                for number, line in enumerate(stream, 1):
                    if '[DT rollout] phase=' in line or re.search(r'\bstep:\d+ - ', line):
                        records.append({'path': str(path), 'line': number, 'text': line.rstrip()})
        for path in log_root.glob('worker-*.err'):
            with path.open(errors='replace') as stream:
                for number, line in enumerate(stream, 1):
                    if re.search(r'OutOfMemoryError|CUDA out of memory|Traceback \(most recent|non.finite', line, re.I):
                        errors.append({'path': str(path), 'line': number, 'text': line.strip()[:600]})
        metrics, phases = parse_logs(records)
        checkpoint = Path(job['checkpoint_dir']) / 'latest_checkpointed_iteration.txt'
        exit_file = Path(job['exit_file'])
        out['jobs'].append({'task': job['task'], 'gpu': job['gpu'],
                           'run_dir': job['run_dir'], 'budget': job['paper_budget']['iterations'],
                           'pid_alive': Path('/proc', str(job['pid'])).exists(),
                           'exit_code': exit_file.read_text().strip() if exit_file.exists() else None,
                           'checkpoint_step': int(checkpoint.read_text()) if checkpoint.exists() else None,
                           'metrics': metrics, 'phases': phases, 'records': records, 'errors': errors[-10:]})
    mem = Path('/sys/fs/cgroup/memory')
    out['memory_gib'] = int((mem / 'memory.usage_in_bytes').read_text()) / 2**30
    out['memory_limit_gib'] = int((mem / 'memory.limit_in_bytes').read_text()) / 2**30
    out['memory_stat'] = {k: int(v) for k, v in
                          (line.split() for line in (mem / 'memory.stat').read_text().splitlines())}
    smi = subprocess.run(['mx-smi'], capture_output=True, text=True, check=True, timeout=30).stdout
    out['mx_smi'] = smi
    # mx-smi prints the GPU number on the row before its physical MiB counter.
    gpu, usage = None, {}
    for line in smi.splitlines():
        match = re.match(r'\|\s*\d+\s+MetaX\s+\S+\s*\|\s*(\d+)\s', line)
        if match:
            gpu = int(match[1])
        match = re.search(r'(\d+)/(\d+) MiB', line)
        if match and gpu is not None:
            usage[str(gpu)] = {'used_gib': int(match[1]) / 1024, 'total_gib': int(match[2]) / 1024}
            gpu = None
    out['gpus'] = usage
    return out


def completed(job):
    return max((int(m['values']['training/global_step']) for m in job['metrics']
                if 'training/global_step' in m['values']), default=0)


def phase_label(job):
    if job['exit_code'] is not None:
        return '已退出，代码 ' + job['exit_code']
    if not job['phases']:
        return '尚无阶段记录'
    p = job['phases'][-1]
    names = {'generation_start': '生成中', 'generation_end': '生成已结束',
             'dt_rpc_start': 'DT 计算中', 'dt_rpc_end': 'DT 已结束；等待更新日志'}
    return names.get(p['phase'], p['phase']) + (' · 交互 ' + p['step'] if 'step' in p else '')


def render(snapshot, history, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#475569',
                         'text.color': '#0f172a', 'axes.edgecolor': '#cbd5e1',
                         'figure.facecolor': '#f8fafc', 'axes.facecolor': 'white'})
    colors = {'Sokoban': '#b45309', 'Webshop': '#2563eb', 'AppWorld': '#0f766e'}
    jobs = sorted(snapshot['jobs'], key=lambda j: ['Webshop', 'Sokoban', 'AppWorld'].index(j['task']))
    now = dt.datetime.fromtimestamp(snapshot['collected_at'], dt.timezone(dt.timedelta(hours=8)))
    stamp = now.strftime('%Y%m%d-%H%M%S')
    header = now.strftime('%Y-%m-%d %H:%M:%S UTC+8') + '  ·  ' + snapshot['source_commit'][:7]

    def setup(ax, title, x, y):
        ax.set_title(title, loc='left', pad=12)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=.17)

    def finish(ax):
        if ax.lines:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, '尚无记录', ha='center', va='center', transform=ax.transAxes, color='#64748b')

    def line(ax, xs, ys, label, color, **kw):
        # Non-finite / absent observations break curves instead of becoming zero.
        ys = [v if v is not None and math.isfinite(v) else float('nan') for v in ys]
        if any(math.isfinite(v) for v in ys):
            ax.plot(xs, ys, marker='.', label=label, color=color, linewidth=1.5, **kw)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('DeltaTrace · 正式训练进展', x=.065, y=.985, ha='left', fontsize=21, weight='bold')
    fig.text(.065, .951, header, color='#64748b')
    for n, job in enumerate(jobs):
        task = job['task']
        fig.text(.065 + n * .32, .908,
                 f"{task}  {completed(job)}/{job['budget']} 迭代\n{phase_label(job)}",
                 color=colors[task], fontsize=11, linespacing=1.8)
    ax = axes[0, 0]
    setup(ax, '完成迭代', '距本次正式启动 / 小时', '已记录训练迭代')
    for job in jobs:
        pairs = [(s, next((j for j in s['jobs'] if j['task'] == job['task']), None)) for s in history]
        pairs = [(s, j) for s, j in pairs if j is not None]
        line(ax, [(s['collected_at'] - s['started']) / 3600 for s, _ in pairs],
             [completed(j) for _, j in pairs], job['task'], colors[job['task']])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-.05, max(1, max(completed(j) for j in jobs) * 1.08)); finish(ax)
    setup(axes[0, 1], '生成吞吐', '已完成的生成批次序号', '实际输出 token/s')
    setup(axes[1, 0], '阶段耗时（各阶段独立编号）', '已完成的阶段调用 / 更新序号', '分钟')
    setup(axes[1, 1], '生成开始时的活跃轨迹', '生成批次序号', '轨迹数')
    for job in jobs:
        color, task = colors[job['task']], job['task']
        ends = [p for p in job['phases'] if p['phase'] == 'generation_end']
        line(axes[0, 1], range(1, len(ends) + 1), [float(p['tokens_per_second']) for p in ends], task, color)
        line(axes[1, 0], range(1, len(ends) + 1), [float(p['seconds']) / 60 for p in ends], task + ' 生成', color)
        dt_ends = [p for p in job['phases'] if p['phase'] == 'dt_rpc_end']
        line(axes[1, 0], range(1, len(dt_ends) + 1), [float(p['seconds']) / 60 for p in dt_ends], task + ' DT', color, linestyle='--')
        updates = [m for m in job['metrics'] if 'timing_s/update_actor' in m['values']]
        line(axes[1, 0], [m['step'] for m in updates], [m['values']['timing_s/update_actor'] / 60 for m in updates], task + ' PPO', color, linestyle=':')
        starts = [p for p in job['phases'] if p['phase'] == 'generation_start']
        line(axes[1, 1], range(1, len(starts) + 1), [int(p['active'].split('/')[0]) for p in starts], task, color)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)); finish(ax)
    hours = [(s['collected_at'] - s['started']) / 3600 for s in history]
    setup(axes[2, 0], '容器内存（定时快照）', '距本次正式启动 / 小时', 'GiB')
    line(axes[2, 0], hours, [s['memory_gib'] for s in history], '当前内存', '#475569')
    axes[2, 0].axhline(snapshot['memory_limit_gib'], color='#dc2626', ls='--', lw=1, label='容器上限')
    axes[2, 0].set_ylim(bottom=0); finish(axes[2, 0])
    setup(axes[2, 1], '物理显存（定时快照，非连续峰值）', '距本次正式启动 / 小时', 'GiB')
    for job in jobs:
        gpu = str(job['gpu'])
        line(axes[2, 1], hours, [s['gpus'].get(gpu, {}).get('used_gib') for s in history], job['task'], colors[job['task']])
    limits = [snapshot['gpus'][str(j['gpu'])]['total_gib'] for j in jobs if str(j['gpu']) in snapshot['gpus']]
    if limits:
        axes[2, 1].axhline(min(limits), color='#dc2626', ls='--', lw=1, label='物理容量')
    axes[2, 1].set_ylim(bottom=0); finish(axes[2, 1])
    for ax in (axes[0, 0], axes[2, 0], axes[2, 1]):
        ax.set_xlim(0, max(hours) * 1.05)
    fig.text(.065, .015, '仅当前正式运行 · 点为实际日志/快照 · 活跃轨迹减少不等于成功 · 尚未结束的阶段不伪造耗时/吞吐', color='#64748b', fontsize=9)
    fig.subplots_adjust(left=.085, right=.985, bottom=.065, top=.835, hspace=.52, wspace=.24)
    files = []
    for name in ('progress.png', f'progress-{stamp}.png'):
        fig.savefig(output / name, dpi=150); files.append(output / name)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle('DeltaTrace · 训练效果与更新', x=.065, y=.985, ha='left', fontsize=21, weight='bold')
    fig.text(.065, .945, header + '  ·  训练集与独立评估分别标注', color='#64748b')
    for column, job in enumerate(jobs):
        task, color = job['task'], colors[job['task']]
        for row, (key, title, unit) in enumerate([
                ('episode/success_rate', '成功率', '%'),
                ('episode/reward/mean', '平均轨迹回报', '原环境奖励单位'),
                ('actor/grad_norm', 'PPO 梯度范数', '范数')]):
            ax = axes[row, column]; setup(ax, task + ' · ' + title, '训练迭代', unit)
            metrics = job['metrics']
            scale = 100 if row == 0 else 1
            line(ax, [m['step'] for m in metrics],
                 [m['values'][key] * scale if key in m['values'] else None for m in metrics], '训练', color)
            if row == 0:
                val_keys = sorted({k for m in metrics for k in m['values'] if k.startswith('val/') and 'success_rate' in k})
                for val_key in val_keys:
                    vals = [m for m in metrics if val_key in m['values']]
                    line(ax, [m['step'] for m in vals], [m['values'][val_key] * 100 for m in vals], '评估: ' + val_key[4:], '#9333ea', linestyle='--')
                ax.set_ylim(-3, 103)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True)); finish(ax)
    fig.text(.065, .012, '缺失不填零，不平滑，不混入 pilot · 回报来自 episode/reward/mean · 梯度按原 console 三位小数显示', color='#64748b', fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, bottom=.08, top=.86, hspace=.6, wspace=.3)
    for name in ('effects.png', f'effects-{stamp}.png'):
        fig.savefig(output / name, dpi=150); files.append(output / name)
    plt.close(fig)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect', action='store_true', help='Remote stdlib-only JSON output')
    p.add_argument('--source-root', required=True)
    p.add_argument('--host', default='root@ssh.v5000-prod-gw.nhss.zhejianglab.com')
    p.add_argument('--port', default='32036')
    p.add_argument('--snapshot', type=Path, help='Render an already collected snapshot offline')
    p.add_argument('--output', type=Path, default=Path('research/temporary/rl_training_plots'))
    p.add_argument('--publish', action='store_true', help='Copy artifacts into remote receipts for existing backup')
    args = p.parse_args()
    if args.collect:
        print(json.dumps(collect(args.source_root)))
        return
    ssh = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=20', '-p', args.port, args.host]
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
    else:
        cmd = shlex.join(['/opt/conda/bin/python', '-', '--collect', '--source-root', args.source_root])
        raw = subprocess.run(ssh + [cmd], input=Path(__file__).read_bytes(), capture_output=True, check=True, timeout=120).stdout
        snapshot = json.loads(raw)
    run = Path(snapshot['jobs'][0]['run_dir']).parent.name + '-' + str(int(snapshot['started']))
    output = args.output / run; output.mkdir(parents=True, exist_ok=True)
    snap_path = output / f"snapshot-{int(snapshot['collected_at'])}.json"
    snap_path.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    latest = output / 'latest.json'; latest.write_bytes(snap_path.read_bytes())
    history = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(output.glob('snapshot-*.json'))]
    history = [s for s in history if s['started'] == snapshot['started'] and s['root'] == snapshot['root']]
    files = render(snapshot, history, output)
    csv_path = output / 'metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['task', 'iteration', 'metric', 'value'])
        for job in snapshot['jobs']:
            for m in job['metrics']:
                writer.writerows((job['task'], m['step'], k, v) for k, v in m['values'].items())
    files += [snap_path, latest, csv_path]
    if args.publish:
        remote = str(Path(args.source_root).as_posix()).rstrip('/') + '/receipts/training-plots/' + run
        subprocess.run(ssh + ['mkdir -p -- ' + shlex.quote(remote)], check=True, timeout=60)
        subprocess.run(['scp', '-q', '-oBatchMode=yes', '-oConnectTimeout=20', '-P', args.port,
                        *[str(path) for path in files], args.host + ':' + remote + '/'], check=True, timeout=120)
    print(json.dumps({'output': str(output.resolve()), 'files': [str(f.resolve()) for f in files],
                      'published': args.publish, 'jobs': [{'task': j['task'], 'iterations': completed(j),
                      'phase': phase_label(j), 'checkpoint_step': j['checkpoint_step']} for j in snapshot['jobs']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
