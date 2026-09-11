"""Build manuscript tables and measured cost figures from verified local records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tables-only', action='store_true', help='Update quality tables without rebuilding unrelated figures.')
args = parser.parse_args()
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.4,
                     'axes.titlesize': 9.4, 'axes.labelsize': 8.4,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.hashsalt': 'deltatrace-results-20260910',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#AAB6BF', 'text.color': '#263442',
                     'axes.labelcolor': '#263442'})
DT, FT = '#328B7F', '#7D8798'
def read(name):
    return json.loads((DATA / (name + '.json')).read_bytes())
def write(name, text):
    (HERE / name).write_text(text.strip() + '\n', encoding='utf-8')
def save(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / (name + '.' + ext), dpi=300,
                    metadata={'CreationDate': None} if ext == 'pdf' else None)
    svg = OUT / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in
                            svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8')
    plt.close(fig)
def polish(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#DFE5E9', lw=.55)
    ax.tick_params(length=2.5, color='#AAB6BF')
def labelbars(ax, bars, fmt='.2f'):
    ax.bar_label(bars, fmt=lambda v: format(v, fmt), padding=3, fontsize=8)
def better(a, b, up=False, decimals=4):
    val = f'{a:.{decimals}f}'
    return r'\textbf{' + val + '}' if (a > b if up else a < b) else val

provenance = json.loads((HERE / 'sources.json').read_bytes())
for item in provenance['sources'].values():
    assert hashlib.sha256((HERE / item['fixture']).read_bytes()).hexdigest() == item['fixture_sha256']
rows = list(csv.DictReader((DATA / 'qwen3_full.csv').open(encoding='utf-8')))
assert len(rows) == 13 and sum(int(r['count']) for r in rows) == 1243
names = {'niah_mq_q2':'MQ-Q2', 'niah_mq_q4':'MQ-Q4', 'niah_mq_q8':'MQ-Q8',
         'niah_mv_v2':'MV-V2', 'niah_mv_v4':'MV-V4', 'niah_mv_v8':'MV-V8',
         'vt_h2_c3':'VT-H2-C3', 'vt_h4_c1':'VT-H4-C1', 'vt_h6_c1':'VT-H6-C1',
         'vt_h10_c1':'VT-H10-C1', 'hotpotqa_long':'HotpotQA',
         'math':'MATH', 'morehopqa':'MoreHopQA'}
baseline_rows=list(csv.DictReader((DATA/'published_baselines.csv').open()))
assert len(baseline_rows)==65
baseline_source=read('published_baseline_sources')
for source in baseline_source['source_csv_files']:
    assert hashlib.sha256((HERE/source['fixture']).read_bytes()).hexdigest()==source['sha256']
methods=['Perturbation','REAGENT','CLP','IFR','AttnLRP','FT','DT']
combined={}
for row in rows:
    ds=row['dataset'];combined[ds]={}
    for method in ('FT','DT'):
        combined[ds][method]={'RISE':float(row[method+'_RISE']),'MAS':float(row[method+'_MAS']),
            'Recovery10':float(row[method+'_Recall10']) if row[method+'_Recall10'] else None}
for row in baseline_rows:
    combined[row['dataset']][row['method']]={m:float(row[m]) if row[m] else None for m in ('RISE','MAS','Recovery10')}
assert all(set(values)==set(methods) for values in combined.values())
recovery_source = read('current_recovery_sources')
assert recovery_source['status'] == 'verified_current_recovery_import'
assert hashlib.sha256((HERE/recovery_source['output_csv']).read_bytes()).hexdigest() == recovery_source['output_sha256']
for item in recovery_source['files']:
    assert hashlib.sha256((HERE/item['fixture']).read_bytes()).hexdigest() == item['sha256']
for ds, entries in combined.items():
    for method, values in entries.items():
        values['Recovery'] = values.pop('Recovery10')
        available = values['Recovery'] is not None
        values.update(RecoveryBudgetFraction=.1 if available else None,
                      RecoveryMetric='released_token_recovery' if available else '',
                      RecoveryBudgetUnit='released_eligible_tokens' if available else '',
                      RecoveryTargetMode='released_protocol' if available else '',
                      RecoveryAttributionVariant=method if available else '',
                      RecoveryProtocol='table1-data-v1' if available else '')
current_recovery = list(csv.DictReader((DATA/'current_recovery.csv').open()))
assert len(current_recovery) == 35
for row in current_recovery:
    values = combined[row['dataset']][row['method']]
    assert values['Recovery'] is None
    for key in ('Recovery','RecoveryBudgetFraction'):
        values[key] = float(row[key])
    for key in ('RecoveryMetric','RecoveryBudgetUnit','RecoveryTargetMode','RecoveryProtocol'):
        values[key] = row[key]
    values['RecoveryAttributionVariant'] = row['attribution_variant']
vt_tasks = ['vt_h2_c3','vt_h4_c1','vt_h6_c1','vt_h10_c1']
vt_macro = {method:float(np.mean([combined[t][method]['Recovery'] for t in vt_tasks])) for method in methods}
lowest_counts={}
tables=[]
for metric,label,caption in [
    ('RISE','tab:full',r'RISE $\downarrow$ on the complete released Qwen3-8B tasks. FT is FlashTrace and DT is \dt{}. The five other baselines and FT use published aggregates \citep{pan2026flashtrace}; DT uses signed ranking. $n$ is the released task size. Bold marks the lowest observed mean across all seven methods.'),
    ('MAS','tab:mas',r'MAS $\downarrow$ on the same released tasks and seven methods as Table~\ref{tab:full}. DT uses its positive source contributions. Bold marks the lowest observed mean.'),
    ('Recovery','tab:recovery',r'Evidence recovery ($\%$, higher is better) at the listed token budgets. NIAH/VT report token Recall; HotpotQA reports supporting-fact Recall. VT macro averages its four rows. For VT/HotpotQA, FT uses K3 and $\dagger$ denotes amended AttnLRP (Appendix~\ref{app:evaluation}). Bold marks the highest observed mean.')]:
    recovery = metric == 'Recovery'
    table=[r'\begin{table}[t]',r'\centering\footnotesize' if recovery else r'\centering\small',
           r'\caption{'+caption+'}',r'\label{'+label+'}',
           r'\setlength{\tabcolsep}{2.2pt}' if recovery else r'\setlength{\tabcolsep}{3.0pt}',
           r'\begin{tabular}{@{}lrr rrrrrrr@{}}' if recovery else r'\begin{tabular}{@{}lr rrrrrrr@{}}',r'\toprule',
           (r'Task & $n$ & Budget & Perturbation & REAGENT & CLP & IFR & AttnLRP$^\dagger$ & FT & DT \\' if recovery else
            r'Task & $n$ & Perturbation & REAGENT & CLP & IFR & AttnLRP & FT & DT \\'),r'\midrule']
    best_count=0
    for i,row in enumerate(rows):
        ds=row['dataset']
        if recovery and combined[ds]['DT']['Recovery'] is None:continue
        if i in (6,10):table.append(r'\midrule')
        values=[combined[ds][m][metric] for m in methods]
        assert all(v is not None for v in values)
        best=(max if recovery else min)(values)
        best_count+=abs(values[-1]-best)<1e-12
        formatted=[]
        for val in values:
            s=f'{100*val:.2f}' if recovery else f'{val:.4f}'
            formatted.append(r'\textbf{'+s+'}' if abs(val-best)<1e-12 else s)
        lead=[names[ds],row['count']]
        if recovery:lead.append(f"{100*combined[ds]['DT']['RecoveryBudgetFraction']:.0f}"+r'\%')
        table.append(' & '.join(lead+formatted)+r' \\')
        if recovery and ds == 'vt_h10_c1':
            maximum = max(vt_macro.values())
            table.append(' & '.join(['VT macro','400','mixed']+[
                (r'\textbf{'+f'{100*vt_macro[m]:.2f}'+'}') if abs(vt_macro[m]-maximum)<1e-12 else f'{100*vt_macro[m]:.2f}'
                for m in methods])+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    tables += table+['']
    lowest_counts[metric]=int(best_count)
write('full_table.tex','\n'.join(tables))
combined_rows=[]
for row in rows:
    for method in methods:
        record={'dataset':row['dataset'],'released_task_count':row['count'],'method':method}
        record.update(combined[row['dataset']][method]);combined_rows.append(record)
with (HERE/'all_methods.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=combined_rows[0]);writer.writeheader();writer.writerows(combined_rows)
with (HERE/'recovery_summary.csv').open('w',newline='',encoding='utf-8') as f:
    macro_rows=[dict(method=m,scope='VT macro',metric='body_token_recall',
                    task_budgets='H2=10%;H4=10%;H6=20%;H10=30%',count=400,recall=vt_macro[m]) for m in methods]
    writer=csv.DictWriter(f,fieldnames=macro_rows[0]);writer.writeheader();writer.writerows(macro_rows)
current_summary=dict(recovery_tasks=11,recovery_method_task_cells=77,
                     metric_cells=259,added_current_recovery_cells=35,
                     vt_task_budgets=recovery_source['budgets'],
                     vt_macro_task_specific_budgets=vt_macro,
                     retrospective_budget_selection=True,
                     source_manifest_sha256=hashlib.sha256((DATA/'current_recovery_sources.json').read_bytes()).hexdigest())
if args.tables_only:
    summary=json.loads((HERE/'summary.json').read_bytes())
    summary.update(best_among_all_methods=lowest_counts,current_recovery=current_summary)
    write('summary.json',json.dumps(summary,indent=2))
    print(json.dumps(dict(status='quality_tables_updated',best_among_all_methods=lowest_counts,current_recovery=current_summary)))
    raise SystemExit(0)
dev=read('clean_development32')['models']
table=[r'\begin{table}[htbp]',r'\centering\small',
       r'\caption{Paired development results: the first eight MQ-Q2 and eight MoreHopQA examples per model. These 32 model--example pairs form a separate development set. FT uses one recursive hop for RISE/MAS and three for recovery. RISE is signed; MAS and recovery use positive scores.}',
       r'\label{tab:development}',r'\begin{tabular}{@{}lllrrr@{}}',r'\toprule',
       r'Model & Task & Method & RISE $\downarrow$ & MAS $\downarrow$ & Recovery@10\% $\uparrow$ \\',r'\midrule']
dev_summary=[]
for model,label in [('qwen3','Qwen3-8B'),('qwen35','Qwen3.5-9B')]:
    for ds in ('niah_mq_q2','morehopqa'):
        cases=[r for r in dev[model] if r['dataset']==ds]
        assert len(cases)==8
        for method in ('FT_K1','DT'):
            rise=np.mean([r[method]['rise'] for r in cases]); mas=np.mean([r[method]['mas'] for r in cases])
            recovery=np.mean([r['DT']['needle'] if method=='DT' else r['FT_K3_needle'] for r in cases])*100 if ds=='niah_mq_q2' else None
            rec='---' if recovery is None else f'{recovery:.2f}'
            table.append(f'{label if ds=="niah_mq_q2" and method=="FT_K1" else ""} & {names[ds] if method=="FT_K1" else ""} & '+('FlashTrace' if method=='FT_K1' else r'\dt{}')+f' & {rise:.4f} & {mas:.4f} & {rec} '+r'\\')
            dev_summary.append(dict(model=model,task=ds,method=method,rise=rise,mas=mas,recovery_percent=recovery))
    if model=='qwen3': table.append(r'\midrule')
table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
write('development_table.tex','\n'.join(table))

cost=read('qwen3_matched_cost16')
warm=read('qwen35_batch_cost16')
fig,axs=plt.subplots(2,2,figsize=(7.2,4.75))
fig.subplots_adjust(left=.085,right=.985,bottom=.095,top=.92,hspace=.69,wspace=.31)
for ax in axs.flat: polish(ax)
x=np.arange(2); width=.32
for j,metric in enumerate(('mean_seconds','maximum_peak_bytes')):
    ax=axs[0,j]
    for i,(key,label,color) in enumerate([('both_1','FlashTrace (1 hop)',FT),('finite','DeltaTrace',DT)]):
        vals=[cost['dataset_summary'][ds]['methods'][key][metric] for ds in ('niah_mq_q2','morehopqa')]
        if j: vals=np.array(vals)/1e9
        bars=ax.bar(x+(i-.5)*width,vals,width,color=color,label=label)
        labelbars(ax,bars)
    ax.set_xticks(x,['MQ-Q2 (8)','MoreHopQA (8)'])
    ax.set_ylim(0,26 if j else 1.12)
    ax.set_ylabel('Peak allocated memory (GB)' if j else 'Attribution latency (s / example)')
    ax.set_title(('(b)' if j else '(a)')+' Qwen3-8B, sample batch = 1',loc='left',pad=8,fontweight='bold')
axs[0,0].legend(loc='upper right',frameon=False,fontsize=7.7)
ax=axs[1,0]
ax.set_title('(c) Qwen3.5-9B, 16 examples',loc='left',pad=8,fontweight='bold')
groups=[('Native capture',['native_root_with_CPU_checkpoints','native_root_with_cuda_checkpoints'],'#779BA5'),
        ('Layer replay',['native_replay','native_replay'],'#B8CCD0'),
        ('Finite propagation',['finite_decoder','finite_decoder'],DT),
        ('Other API work',[None,None],'#DADFE4')]
bottom=np.zeros(2)
totals=np.array([warm['aggregate'][k]['mean_seconds_per_16'] for k in ('baseline','accelerated')])
for name,keys,color in groups:
    vals=np.array([warm['stage_mean_seconds_per_16'][k][keys[i]] if keys[i] else totals[i]-bottom[i] for i,k in enumerate(('baseline','accelerated'))])
    ax.bar(x,vals,.52,bottom=bottom,color=color,label=name);bottom+=vals
for i,total in enumerate(totals):ax.text(i,total+.35,f'{total:.2f}',ha='center',fontsize=8)
ax.set_xticks(x,['Serial B1\nCPU checkpoints','Real B2\nGPU checkpoints'])
ax.set_ylabel('Attribution time (s / 16 examples)');ax.set_ylim(0,21)
ax.legend(loc='upper right',ncol=2,frameon=False,fontsize=6.6,columnspacing=.6,handlelength=1)
ax=axs[1,1]
ax.set_title('(d) Qwen3.5-9B memory',loc='left',pad=8,fontweight='bold')
bars=ax.bar(x,[warm['aggregate'][k]['peak_allocated_bytes']/1e9 for k in ('baseline','accelerated')],.52,color=[FT,DT]);labelbars(ax,bars)
ax.set_xticks(x,['Serial B1\nCPU checkpoints','Real B2\nGPU checkpoints'])
ax.set_ylabel('Peak allocated memory (GB)');ax.set_ylim(0,29)
save(fig,'deltatrace-efficiency')

long=read('qwen3_tiling_cost')
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
fig.subplots_adjust(left=.085,right=.985,bottom=.24,top=.78,wspace=.31)
x=np.arange(3)
for j,ax in enumerate(axs):
    polish(ax)
    for i,(key,label,color) in enumerate([('dense','Dense finite propagation',FT),('finite','Tiled finite propagation',DT)]):
        vals=[r['modes'][key]['median_seconds' if j==0 else 'peak_bytes']/(1 if j==0 else 1e9) for r in long['cases']]
        bars=ax.bar(x+(i-.5)*.32,vals,.32,label=label,color=color);labelbars(ax,bars,'.2f' if j==0 else '.1f')
    ax.set_xticks(x,['MQ-Q2\n1,241 tokens','HotpotQA\n3,470 tokens','MATH\n3,762 tokens'])
    ax.set_ylabel('Attribution latency (s)' if j==0 else 'Peak allocated memory (GB)')
    ax.set_ylim(0,7.6 if j==0 else 45)
    ax.set_title(('(a) Complete attribution' if j==0 else '(b) Resident model included'),loc='left',fontweight='bold')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.54,.995),fontsize=8)
save(fig,'deltatrace-tiling-cost')

table=[r'\begin{table}[htbp]',r'\centering\small',
       r'\caption{Additional paired replay-retention measurements on 16 development examples. Each row compares the existing implementation with retained replay within one benchmark. Warm time is the mean complete pass over 16 examples across two interleaved rounds. Peak memory gives before/after allocations, including resident weights. All paired complete source vectors were identical. The baselines already include prior scheduling improvements.}',
       r'\label{tab:retention}',r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',
       r'Model & Sample batch & Before (s) & Retained (s) & Reduction & Peak GB (before/after) \\',r'\midrule']
retention=[]
for key,label in [('qwen3_retained16','Qwen3-8B'),('qwen35_retained16','Qwen3.5-9B')]:
    d=read(key); before=d['baseline_seconds']; after=d.get('candidate_seconds',d.get('retained_seconds'))
    peaks=d['warm_peak_bytes']; peak_before=peaks['baseline']/1e9; peak_after=peaks.get('candidate',peaks.get('accelerated'))/1e9
    assert all(r.get('all_six_complete_vectors_equal',r.get('all_complete_vectors_equal')) for r in d['rows'])
    table.append(f'{label} & {d["sample_batch"]} & {before:.3f} & {after:.3f} & {100*(1-after/before):.2f}'+r'\%'+f' & {peak_before:.2f} / {peak_after:.2f} '+r'\\')
    retention.append(dict(model=label,before=before,after=after,reduction=1-after/before))
table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
write('retention_table.tex','\n'.join(table))
gain=[100*(float(r['DT_Recall10'])-float(r['FT_Recall10'])) for r in rows if r['DT_Recall10']]
summary={'examples':1243,'tasks':13,'RISE_better_tasks':sum(float(r['DT_RISE'])<float(r['FT_RISE']) for r in rows),'MAS_better_tasks':sum(float(r['DT_MAS'])<float(r['FT_MAS']) for r in rows),'retrieval_recovery_gains_pp':gain,'retrieval_macro_gain_pp':float(np.mean(gain)), 'development':dev_summary,'retention':retention,'qwen35_batch_speedup':warm['aggregate']['speedup'], 'qwen3_cost_total_s':{m:sum(c['methods'][m]['median_seconds'] for c in cost['cases']) for m in ['finite','both_1','both_3']},'source_fixtures_verified':True,'best_among_all_methods':lowest_counts,'quality_methods':methods,'additional_baseline_source_files_verified':len(baseline_source['source_csv_files'])}
summary['current_recovery']=current_summary
write('summary.json',json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('development','retention')}))
