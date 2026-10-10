"""Plot only the current formal owner's native metrics; never touch training.

Reuses the project's existing console parser. Original lines, line numbers,
source hashes and actor warnings are retained beside the figure and CSV.
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'experiments/rl/plot_training_progress.py').is_file())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def collect_live_snapshot():
    transport = module('current_transport', HERE.parents[1] / 'stage_environment_entry.py')
    remote = r'''
import hashlib,json,psutil,re,time
from pathlib import Path
root=Path(ROOT)
formal=root/'runs/textcraft-formal-stable-20261009-v1'
p=psutil.Process(982372)
assert p.create_time()==1791553809.84
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
trainer=next(ray.glob('*-985585.out'))
trainer_bytes=trainer.read_bytes()
records=[]
for n,line in enumerate(trainer_bytes.decode(errors='replace').splitlines(),1):
 if re.search(r'\bstep:\d+ - ',line):
  records.append(dict(path=str(trainer),line=n,text=line))
warnings=[]
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 worker=psutil.Process(pid)
 assert worker.create_time()==birth
 for suffix in ['out','err']:
  path=next(ray.glob('*-'+str(pid)+'.'+suffix))
  for n,line in enumerate(path.read_text(errors='replace').splitlines(),1):
   if 'grad_norm is not finite' in line:
    warnings.append(dict(pid=pid,path=str(path),line=n,text=line))
source=formal/'source.json'
print(json.dumps(dict(collected_at=time.time(),formal_dir=str(formal),
 driver=dict(pid=p.pid,birth=p.create_time()),trainer_log=str(trainer),
 trainer_sha256=hashlib.sha256(trainer_bytes).hexdigest(),
 source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 records=records,actor_warnings=warnings,production_changes=0,model_calls=0)))
'''.replace('ROOT', repr(transport.ROOT), 1)
    command = ('source ' + transport.ENTRY + '/metax-entry.env.sh\n'
               'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + remote + '\nPY\n')
    result = subprocess.run(transport.SSH + ['bash', '-s'], input=command.encode(),
                            capture_output=True, timeout=50)
    result.check_returncode()
    return json.loads(result.stdout)


def render(snapshot):
    owner = module('native_log_plot_owner', REPO / 'experiments/rl/plot_training_progress.py')
    metrics, _ = owner.parse_logs(snapshot['records'])
    stamp = dt.datetime.fromtimestamp(snapshot['collected_at'], dt.timezone(dt.timedelta(hours=8)))
    output = HERE / 'direct-credit-records-20261009-v1' / ('native-curves-' + stamp.strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / 'original-native-metric-lines.json'
    raw_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    keys = ['episode/reward/mean', 'actor/entropy_loss', 'actor/grad_norm', 'actor/ppo_kl', 'actor/pg_loss']
    csv_path = output / 'reward-entropy-gradient.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['iteration', *keys, 'source_path', 'source_line'])
        for m in metrics:
            writer.writerow([m['step'], *[m['values'].get(k, '') for k in keys],
                             m['sources'][0]['path'], m['sources'][0]['line']])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#f8fafc',
                         'axes.facecolor': 'white', 'font.size': 11})
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True)
    fig.suptitle('TextCraft · 当前正式训练的奖励、熵与梯度', x=.10, y=.97,
                 ha='left', fontsize=18, weight='bold')
    fig.text(.10, .925, stamp.strftime('%Y-%m-%d %H:%M:%S UTC+8') +
             '  |  基础权重起训 · 当前运行独立绘图 · 无平滑', color='#64748b', fontsize=10)
    bad = [m['step'] for m in metrics if 'actor/grad_norm' in m['values']
           and not math.isfinite(m['values']['actor/grad_norm'])]
    for ax, key, title, color in zip(axes, keys[:3],
            ['平均轨迹回报  (episode/reward/mean)',
             '策略熵日志值  (actor/entropy_loss)',
             '原框架梯度范数日志  (actor/grad_norm)'], ['#2563eb', '#0f766e', '#7c3aed']):
        xs = [m['step'] for m in metrics]
        ys = [m['values'].get(key, float('nan')) for m in metrics]
        ys = [v if math.isfinite(v) else float('nan') for v in ys]
        ax.plot(xs, ys, marker='o', markersize=4, linewidth=1.6, color=color)
        ax.set_title(title, loc='left', fontsize=11, weight='bold')
        ax.grid(alpha=.18)
        for step in bad:
            ax.axvline(step, color='#dc2626', linestyle='--', linewidth=1.2)
        finite = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
        if finite:
            x, y = finite[-1]
            ax.annotate(f'{y:.3f}', (x, y), xytext=(6, 5), textcoords='offset points', color=color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(.6, max(xs) + .7)
        ax.set_ylim(bottom=0)
    for step in bad:
        axes[0].annotate(f'第{step}轮：范数 NaN，跳过一次异常 optimizer 更新',
                         xy=(step, .98), xycoords=('data', 'axes fraction'),
                         xytext=(-8, -3), textcoords='offset points', ha='right', va='top',
                         fontsize=10, color='#b91c1c')
        axes[2].annotate('NaN（断线；未填 0）', xy=(step, .15),
                         xycoords=('data', 'axes fraction'), xytext=(-8, 0),
                         textcoords='offset points', ha='right', color='#b91c1c', fontsize=10)
    axes[-1].set_xlabel('原框架已记录的完整训练迭代')
    fig.text(.10, .025, '来源：同一正式运行的原 VERL trainer 日志；控制台精度为 3 位小数。\n'
             '奖励是训练轨迹回报；梯度范数按原日志聚合值展示。缺失或非有限值保留为空。',
             fontsize=9, color='#64748b')
    fig.subplots_adjust(left=.10, right=.95, top=.865, bottom=.11, hspace=.38)
    figure_path = output / 'reward-entropy-gradient.png'
    fig.savefig(figure_path, dpi=155)
    plt.close(fig)
    summary = dict(collected_at=snapshot['collected_at'], iterations=len(metrics),
                   last_iteration=metrics[-1]['step'], nonfinite_grad_norm_iterations=bad,
                   actor_warning_count=len(snapshot['actor_warnings']) if 'actor_warnings' in snapshot else None,
                   latest={k:metrics[-1]['values'].get(k) for k in keys},
                   first={k:metrics[0]['values'].get(k) for k in keys},
                   source_sha256=snapshot['source_sha256'],
                   parser_path=str(REPO/'experiments/rl/plot_training_progress.py'),
                   parser_sha256=hashlib.sha256((REPO/'experiments/rl/plot_training_progress.py').read_bytes()).hexdigest(),
                   image=str(figure_path),csv=str(csv_path),original_lines=str(raw_path),
                   production_changes=0,model_calls=0)
    (output/'plot-receipt.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path,
                        help='Render a saved original-log snapshot without remote calls.')
    args = parser.parse_args()
    render(json.loads(args.snapshot.read_text(encoding='utf-8'))
           if args.snapshot else collect_live_snapshot())
