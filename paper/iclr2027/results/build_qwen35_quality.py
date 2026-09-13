"""Build the manuscript's Qwen3.5 quality table from frozen task means."""
import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=HERE/'data/qwen35_official_quality.csv'
ledger=json.loads((HERE/'data/qwen35_official_quality_sources.json').read_bytes())
assert hashlib.sha256(source.read_bytes()).hexdigest()==ledger['output_sha256']
assert hashlib.sha256((HERE/ledger['case_csv']).read_bytes()).hexdigest()==ledger['case_csv_sha256']
for item in ledger['files']:
    if 'fixture' in item:
        assert hashlib.sha256((HERE/item['fixture']).read_bytes()).hexdigest()==item['sha256']
with source.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
by={(r['dataset'],r['method']):r for r in rows}
tasks=[('niah_mq_q2','MQ-Q2',10,10),('niah_mq_q4','MQ-Q4',10,10),
       ('niah_mq_q8','MQ-Q8',10,10),('niah_mv_v2','MV-V2',10,10),
       ('niah_mv_v4','MV-V4',10,10),('niah_mv_v8','MV-V8',10,10),
       ('math','MATH',6,None),('morehopqa','MoreHopQA',6,None)]
lines=[r'\begin{table}[t]',r'\centering\small',
    r'\caption{Qwen3.5-9B quality with symmetric GDN propagation. NIAH uses ten new key/value assignments per task on fixed templates; MATH and MoreHopQA each use six fixed released examples. Recovery is token Recall at the listed eligible-token budget. FT denotes FlashTrace: K3 for Recall and K1 for RISE/MAS. DT uses signed RISE and positive-part MAS. Bold marks the better mean; dashes indicate undefined recovery.}',
    r'\label{tab:qwen35-quality}',r'\setlength{\tabcolsep}{4.0pt}',
    r'\begin{tabular}{@{}lrr rr rr rr@{}}',r'\toprule',
    r'Task & $n$ & Budget & \multicolumn{2}{c}{Recall (\%) $\uparrow$} & \multicolumn{2}{c}{RISE $\downarrow$} & \multicolumn{2}{c}{MAS $\downarrow$} \\',
    r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}',
    r' & & & FT & DT & FT & DT & FT & DT \\',r'\midrule']
better_counts=dict(recall=0,rise=0,mas=0)
for i,(task,name,count,budget) in enumerate(tasks):
    d=by[task,'DT'];f=by[task,'FT_K1']
    assert int(d['count'])==int(f['count'])==count
    if i==6:lines.append(r'\midrule')
    cells=[name,str(count),'---' if budget is None else str(budget)+r'\%']
    for metric in ('recall','rise','mas'):
        if metric=='recall' and budget is None:
            assert not d[metric] and not f[metric]
            cells+=['---','---'];continue
        control=by[task,'FT_K3'] if metric=='recall' else f
        dt,ft=float(d[metric]),float(control[metric])
        wins=dt>ft if metric=='recall' else dt<ft
        better_counts[metric]+=wins
        for value,is_better in [(ft,not wins and ft!=dt),(dt,wins)]:
            s=f'{100*value:.2f}' if metric=='recall' else f'{value:.4f}'
            cells.append(r'\textbf{'+s+'}' if is_better else s)
    lines.append(' & '.join(cells)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
assert better_counts==dict(recall=6,rise=6,mas=7)
(HERE/'qwen35_quality_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
receipt=dict(source='data/'+source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    attribution_profile='gdn-symmetric-v1',
    tasks=8,cases=sum(x[2] for x in tasks),quality_values=44,DT_better_tasks=better_counts,
    output='qwen35_quality_table.tex')
(HERE/'qwen35_quality_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
