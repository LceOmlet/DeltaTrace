"""Plot original owner metrics without splicing the discarded resume branch."""
from pathlib import Path
import csv
import importlib.util
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'experiments/rl/plot_training_progress.py').exists())
spec = importlib.util.spec_from_file_location('owner_log_plot', ROOT / 'experiments/rl/plot_training_progress.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
snapshot = json.loads((HERE / 'native-metrics.json').read_text(encoding='utf-8'))
runs = {r['name']: parser.parse_logs(r['records'])[0] for r in snapshot['runs']}
branches = [
    ('Original 1–25 (historical)', [r for r in runs['pre-fix'] if r['step'] <= 25], '#596579', '-'),
    ('Original 26–37 (discarded)', [r for r in runs['pre-fix'] if r['step'] >= 25], '#acb3bf', '--'),
    ('Resumed 26–27 (historical)', [r for r in runs['pre-fix'] if r['step'] == 25] + runs['resumed-fix'], '#087e8b', '-'),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, layout='constrained')
for ax, key, label in zip(axes, ['episode/reward/mean', 'actor/entropy_loss', 'actor/grad_norm'],
                          ['Training reward (mean)', 'Actor entropy', 'Gradient norm']):
    for name, rows, color, ls in branches:
        ax.plot([r['step'] for r in rows], [r['values'].get(key, math.nan) for r in rows],
                label=name, color=color, linestyle=ls, marker='o', markersize=3, linewidth=1.6)
    ax.axvline(25, color='#bd9463', linewidth=1, alpha=.8)
    ax.set_ylabel(label)
    ax.grid(alpha=.16)
    ax.set_xlim(.5, 38)
axes[0].legend(loc='lower left', fontsize=9)
axes[0].set_title('TextCraft · native training metrics', loc='left', fontsize=16, pad=16)
for step in (14, 37):
    axes[2].plot(step, .95, marker='x', color='#c74040', transform=axes[2].get_xaxis_transform(), markersize=8)
    axes[2].annotate(f'NaN at {step}', (step, .95), xycoords=('data', 'axes fraction'),
                     xytext=(-35, -21), textcoords='offset points', color='#a62c2c', fontsize=9)
axes[2].set_xlabel('Training iteration')
fig.get_layout_engine().set(rect=(0, .05, 1, .95))
fig.text(.07, .013, 'Native console metrics, no smoothing. Reward is from training, not independent evaluation.\n'
         'Both jobs now start fresh; these curves are historical. Old 26–37 form a separate branch. NaNs are not filled.', fontsize=9, color='#596579')
fig.savefig(HERE / 'training-curves.png', dpi=170)
fig.savefig(HERE / 'training-curves.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.7), layout='constrained')
keys = [('timing_s/gen', 'Generation', '#3c648a'), ('timing_s/adv', 'DT / advantages', '#b96c39'),
        ('timing_s/update_actor', 'Actor update', '#6b8e48'), ('timing_s/old_log_prob', 'Old log-prob', '#8d639c'),
        ('timing_s/ref', 'Reference log-prob', '#898989')]
for key, label, color in keys:
    for branch, rows, _, ls in branches:
        ax.plot([r['step'] for r in rows], [r['values'].get(key, math.nan) / 60 for r in rows],
                color=color, linestyle=ls, alpha=.45 if 'discarded' in branch else 1,
                marker='o', markersize=3, label=label if branch.startswith('Original 1') else None)
ax.axvline(25, color='#bd9463', linewidth=1)
ax.set(xlabel='Training iteration', ylabel='Wall time (minutes)', xlim=(.5, 38))
ax.set_title('TextCraft · native phase times', loc='left', fontsize=15)
ax.grid(alpha=.16)
ax.legend(ncol=3, fontsize=9)
fig.get_layout_engine().set(rect=(0, .07, 1, .93))
fig.text(.07, .015, 'Old 26–37: dashed; resumed 26–27: solid. Generation includes prefill and engine overhead.\n'
         'Different iterations contain different trajectories; these are observed phase times, not a speedup benchmark.', fontsize=9, color='#596579')
fig.savefig(HERE / 'phase-times.png', dpi=170)
fig.savefig(HERE / 'phase-times.pdf')
plt.close(fig)
with (HERE / 'native-metrics.csv').open('w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'iteration', 'metric', 'value', 'source_path', 'source_line'])
    for name, rows in runs.items():
        for row in rows:
            src = row['sources'][0]
            for key, value in row['values'].items():
                w.writerow([name, row['step'], key, value, src['path'], src['line']])
print(json.dumps({'images': [str(HERE / 'training-curves.png'), str(HERE / 'phase-times.png')],
                  'records': {k: len(v) for k, v in runs.items()}}, ensure_ascii=False))
