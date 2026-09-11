"""Compact, vector recall/time inset from the verified manuscript records.

Each recall axis has one range shared by every method: the observed extrema
plus 8% of their span, rounded outward to 5 pp (span < 25) or 10 pp.
The input percentages and timing records are never rescaled in the exports.
Only the radar's geometric radius is normalized to its labeled axis range.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[1]
OUT = PAPER / 'results/figures'
W, H = 2.70, 3.96
TASKS = [
    ('niah_mq_q2', 'MQ-Q2'), ('niah_mq_q4', 'MQ-Q4'), ('niah_mq_q8', 'MQ-Q8'),
    ('niah_mv_v2', 'MV-V2'), ('niah_mv_v4', 'MV-V4'), ('niah_mv_v8', 'MV-V8'),
    ('vt_h2_c3', 'VT-H2'), ('vt_h4_c1', 'VT-H4'),
    ('vt_h6_c1', 'VT-H6†'), ('vt_h10_c1', 'VT-H10‡'), ('hotpotqa_long', 'HotpotQA')]
METHODS = ['DT', 'FT', 'Perturbation', 'REAGENT', 'CLP', 'IFR', 'AttnLRP']
STYLE = {
    'DT': ('#1769AA', 'o', '-', 1.45),
    'FT': ('#DB674D', 's', '--', 1.05),
    'Perturbation': ('#7E95A9', 'v', ':', .72),
    'REAGENT': ('#7C9B65', '^', '--', .72),
    'CLP': ('#74A6A3', 'P', '-.', .72),
    'IFR': ('#A08469', 'D', '-', .80),
    'AttnLRP': ('#9A90BC', 'X', '--', .80),
    'FT multi-hop': ('#E5A142', 'D', '--', 1.05),
    'IG': ('#B07AA1', 'v', ':', .72),
    'IG × Attention': ('#E6A14B', '^', '-.', .72),
}
TIME_NAMES = {'DeltaTrace': 'DT', 'FlashTrace': 'FT'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def load_records():
    recall_path = PAPER / 'results/all_methods.csv'
    time_path = ROOT / 'experiments/efficiency/curve_data.json'
    verification = json.loads((PAPER / 'results/verification.json').read_bytes())
    assert verification['status'] == 'passed'
    assert sha(recall_path) == verification['output_sha256']['all_methods.csv']
    with recall_path.open(encoding='utf8', newline='') as stream:
        rows = {(r['dataset'], r['method']): r for r in csv.DictReader(stream)}
    axes = []
    for task, label in TASKS:
        values = {m: 100 * float(rows[task, m]['Recovery']) for m in METHODS}
        low, high = min(values.values()), max(values.values())
        span = high - low
        step = 5 if span < 25 else 10
        lower = max(0, step * math.floor((low - .08 * span) / step))
        upper = min(100, step * math.ceil((high + .08 * span) / step))
        assert lower < upper and all(lower <= v <= upper for v in values.values())
        axes.append(dict(dataset=task, label=label, minimum=lower, maximum=upper,
                         budget=float(rows[task, 'DT']['RecoveryBudgetFraction']),
                         metric=rows[task, 'DT']['RecoveryMetric'], recall_percent=values))
    timing = json.loads(time_path.read_bytes())
    assert len(timing['local']) == 21 and len(timing['published']) == 35
    return axes, timing, {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                         for p in (recall_path, time_path)}


def draw(axes, timing):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'text.color': '#263442', 'axes.labelcolor': '#263442',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'deltatrace-intro-recall-time'})
    fig = plt.figure(figsize=(W, H), facecolor='white')
    labels = []

    def text(x, y, s, **kw):
        t = fig.text(x/W, y/H, s, fontsize=8, **kw)
        labels.append(t)
        return t

    radar = fig.add_axes([.18, 2.40/H, .65, 1.22/H], projection='polar')
    radar.set_theta_offset(np.pi/2)
    radar.set_theta_direction(-1)
    theta = np.linspace(0, 2*np.pi, len(axes), endpoint=False)
    radar.set_ylim(0, 1)
    radar.set_xticks(theta, [])
    radar.set_yticks([0, .25, .5, .75, 1], [])
    radar.grid(color='#D5DDE5', lw=.4)
    radar.spines['polar'].set_color('#BBC7D3')
    radar.spines['polar'].set_linewidth(.45)
    for method in reversed(METHODS):
        values = [(a['recall_percent'][method]-a['minimum'])/(a['maximum']-a['minimum'])
                  for a in axes]
        color, marker, line, lw = STYLE[method]
        radar.plot(np.r_[theta, theta[0]], np.r_[values, values[0]],
                   color=color, marker=marker, ms=2.2 if method=='DT' else 1.5,
                   lw=lw, ls=line, alpha=1 if method in ('DT','FT') else .82,
                   zorder=5 if method=='DT' else 3)
        if method == 'DT':
            radar.fill(np.r_[theta, theta[0]], np.r_[values, values[0]], color=color, alpha=.035)
    for angle, axis in zip(theta, axes):
        # Anchor the entire label outside the circle; center alignment would
        # put half of each side label back over the data and outer grid line.
        sx, sy = np.sin(angle), np.cos(angle)
        ha = 'center' if abs(sx)<.1 else 'left' if sx>0 else 'right'
        va = 'bottom' if sy>.7 else 'top' if sy<-.8 else 'center'
        label_radius = 1.24 if -.8 < sy < -.6 else 1.12
        t = radar.text(angle, label_radius, f"{axis['label']}\n{axis['minimum']}–{axis['maximum']}",
                       fontsize=8, ha=ha, va=va, linespacing=1.05, clip_on=False)
        labels.append(t)
    text(.02, 3.84, '(a) Recall (%)', weight='bold')

    ax = fig.add_axes([.20, 1.05/H, .75, .74/H])
    lengths = [10, 100, 500, 1000, 2000, 5000, 10000]
    groups = {}
    for row in timing['published'] + timing['local']:
        method = TIME_NAMES.get(row['method'], row['method'])
        groups.setdefault(method, []).append(row)
    oom_artists = []
    for method, rows in groups.items():
        color, marker, line, lw = STYLE[method]
        ok = {r['output_tokens']: r for r in rows if r['status']=='ok'}
        ys = [ok[n]['seconds'] if n in ok else np.nan for n in lengths]
        ax.plot(lengths, ys, color=color, marker=marker, ms=2.4 if method=='DT' else 1.8,
                lw=lw, ls=line, zorder=5 if method=='DT' else 3)
        if rows[0]['series'] == 'same_C550_measured':
            good = [ok[n] for n in lengths if n in ok]
            ax.errorbar([r['output_tokens'] for r in good], [r['seconds'] for r in good],
                        yerr=[[r['seconds']-r['min_seconds'] for r in good],
                              [r['max_seconds']-r['seconds'] for r in good]],
                        fmt='none', ecolor=color, elinewidth=.5, capsize=1)
        oom = [r['output_tokens'] for r in rows if r['status']=='oom']
        if oom:
            band = 1.055 if method=='FT' else 1.15
            oom_artists.extend(ax.plot(oom, [band]*len(oom), transform=ax.get_xaxis_transform(),
                    color=color, marker='x', ms=3, mew=.8, ls='none', clip_on=False))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(7, 15000); ax.set_ylim(.045, 40000)
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x>=1000 else str(int(x))))
    ax.yaxis.set_major_locator(FixedLocator([.1, 10, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '1k' if y==1000 else f'{y:g}'))
    ax.minorticks_off()
    ax.tick_params(length=2, width=.45, pad=2)
    ax.set_xlabel('Rollout length (tokens)', labelpad=1)
    ax.set_ylabel('Attribution time (s)', labelpad=2)
    ax.grid(color='#DDE3E9', lw=.4)
    for side in ('top','right'): ax.spines[side].set_visible(False)
    for side in ('left','bottom'):
        ax.spines[side].set_linewidth(.45); ax.spines[side].set_color('#9DA6AE')
    text(.34, 1.91, '(b)', weight='bold')
    text(.80, 1.91, 'Mixed hardware')

    order = ['DT','Perturbation','IFR','FT','REAGENT','AttnLRP','FT multi-hop','CLP','IG','IG × Attention']
    short = {'Perturbation':'Perturb.', 'FT multi-hop':'FT-mh', 'IG × Attention':'IG×Attn.'}
    handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], ls=STYLE[m][2],
                      lw=STYLE[m][3], ms=2.4) for m in order]
    handles.append(Line2D([], [], color='#737D88', marker='x', ls='none', ms=3, mew=.8))
    legend = fig.legend(handles, [short.get(m,m) for m in order]+['OOM'],
                        loc='lower center', bbox_to_anchor=(.51, .065), ncol=4,
                        fontsize=8, handlelength=.9, handletextpad=.25, columnspacing=.6,
                        borderpad=0, labelspacing=.28, frameon=False)
    labels.extend(legend.get_texts())
    labels.extend([ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()])
    text(W/2, .09, 'Budgets: 10%; †20%; ‡30%.', ha='center')
    fig._inset_labels = labels
    fig._inset_regions = (radar, ax, oom_artists)
    return fig


def main():
    axes, timing, sources = load_records()
    fig = draw(axes, timing)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = [t.get_text() for t in fig._inset_labels
               if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])
               or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
    collisions = []
    for i,a in enumerate(fig._inset_labels):
        ba = a.get_window_extent(renderer)
        for b in fig._inset_labels[i+1:]:
            bb = b.get_window_extent(renderer)
            if min(ba.x1,bb.x1)-max(ba.x0,bb.x0)>1 and min(ba.y1,bb.y1)-max(ba.y0,bb.y0)>1:
                collisions.append([a.get_text(), b.get_text()])
    radar, time_ax, oom_artists = fig._inset_regions
    center = radar.transData.transform((0,0))
    radius = np.linalg.norm(radar.transData.transform((0,1))-center)
    clearance = 2 * fig.dpi / 72  # two physical points beyond the outer ring
    plot_collisions = []
    radar_clearances = []
    for label in fig._inset_labels:
        box = label.get_window_extent(renderer)
        nearest = np.clip(center, [box.x0,box.y0], [box.x1,box.y1])
        gap = np.linalg.norm(nearest-center)-radius
        radar_clearances.append(gap*72/fig.dpi)
        if gap < clearance:
            plot_collisions.append([label.get_text(), 'radar circle + 2 pt clearance'])
        if box.overlaps(time_ax.bbox):
            plot_collisions.append([label.get_text(), 'time plotting area'])
        for artist in oom_artists:
            if box.overlaps(artist.get_window_extent(renderer)):
                plot_collisions.append([label.get_text(), 'OOM markers'])
    assert not (outside or collisions or plot_collisions), (outside,collisions,plot_collisions)
    OUT.mkdir(exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        metadata = {'CreationDate':None, 'ModDate':None} if suffix=='pdf' else {'Date':None} if suffix=='svg' else None
        fig.savefig(OUT/f'deltatrace-recall-time.{suffix}', dpi=400, metadata=metadata)
        if suffix == 'svg':
            svg_path = OUT / 'deltatrace-recall-time.svg'
            svg_text = svg_path.read_text(encoding='utf8')
            svg_path.write_text('\n'.join(line.rstrip() for line in svg_text.splitlines())+'\n',
                                encoding='utf8', newline='\n')
    payload = dict(figure_inches=[W,H], font_pt=8, axes=axes,
                   axis_range_policy='All seven methods share per-task observed extrema plus 8% span padding; round outwards to 5 pp when span <25, otherwise 10 pp; bound within 0..100.',
                   radar_radius='(raw Recall percentage - axis minimum) / (axis maximum - axis minimum)',
                   radar_methods=METHODS, timing_methods=list(STYLE),
                   successful_timing_points=sum(r['status']=='ok' for r in timing['local']+timing['published']),
                   oom_points=sum(r['status']=='oom' for r in timing['local']),
                   time_scope={k:timing[k] for k in ('new_curves_timing_scope','published_reference_scope')},
                   ft_recall_scope='FT_K3 for VT and HotpotQA; released FT recovery for NIAH.',
                   sources=sources, builder_sha256=sha(Path(__file__)),
                   outputs={f'deltatrace-recall-time.{s}':sha(OUT/f'deltatrace-recall-time.{s}') for s in ('pdf','svg','png')},
                   outside_labels=outside, label_collisions=collisions,
                   text_plot_collisions=plot_collisions, minimum_radar_text_clearance_pt=min(radar_clearances),
                   raw_values_modified=False)
    (HERE/'intro_teaser_manifest.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:payload[k] for k in ('figure_inches','outside_labels','label_collisions','successful_timing_points','oom_points')}))


if __name__=='__main__': main()
