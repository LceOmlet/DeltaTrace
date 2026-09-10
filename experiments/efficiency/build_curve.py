"""Verify released/recorded cells and render rollout scaling; no GPU required."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
from collections import defaultdict, Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGURES = HERE / 'figures'
DELIVERY = ROOT / 'output/pdf'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
METHODS = {'IG':'IG', 'attention_I_G':'IG × Attention', 'perturbation_all':'Perturbation',
    'perturbation_REAGENT':'REAGENT', 'ifr_all_positions':'IFR', 'perturbation_CLP':'CLP',
    'attnlrp':'AttnLRP', 'ifr_multi_hop_both':'FlashTrace'}
FILES = {'out-0':[10,100], 'out-2':[500], 'out-3':[1000], 'out-4':[2000], 'out-5':[5000]}


def collect():
    source = json.loads((HERE / 'upstream/source_manifest.json').read_bytes())
    protocol = json.loads((HERE/'protocol.json').read_bytes())
    assert sha((HERE/'upstream/run_time_curve.py').read_bytes()) == protocol['upstream_script_sha256']
    groups = defaultdict(list)
    failure_counts = Counter()
    for record in source['files']:
        path = HERE / 'upstream' / record['path']
        raw = path.read_bytes()
        assert sha(raw) == record['sha256'], path
        if path.suffix != '.jsonl':
            continue
        for line_index, line in enumerate(raw.decode().splitlines(), 1):
            row = json.loads(line)
            directory = path.parent.name
            method = row['attr_func']
            if row['status'] != 'ok':
                failure_counts[(directory, row['status'].split(':')[0])] += 1
                continue
            if (directory not in FILES or method not in METHODS or row['target_input_tokens'] != 10
                    or row['target_output_tokens'] not in FILES[directory]):
                continue
            assert row['time_sec'] > 0
            row['source'] = {'file':record['path'], 'line':line_index, 'sha256':record['sha256']}
            groups[(method, row['target_output_tokens'])].append(row)
    published = []
    for (method, length), rows in groups.items():
        counts = {len(r['peak_mem_by_device_gb']) for r in rows}
        assert len(counts) == 1
        times = [r['time_sec'] for r in rows]
        published.append(dict(method=METHODS[method], output_tokens=length, status='ok',
            seconds=statistics.mean(times), min_seconds=min(times), max_seconds=max(times),
            repeats=len(rows), devices=counts.pop(), peak_allocated_bytes=max(r['peak_mem_gb'] for r in rows)*1e9,
            formatted_prompt_tokens=rows[0]['actual_formatted_prompt_tokens'],
            generation_tokens=rows[0]['actual_generation_tokens'], sources=[r['source'] for r in rows]))
    local = []
    paths = sorted((HERE / 'results').glob('*_*/result.json'))
    if not paths:
        raise ValueError('No measured local cells; a DT curve cannot be inferred from published baselines.')
    for path in paths:
        row = json.loads(path.read_bytes())
        assert row['driver_sha256'] == sha((HERE/'benchmark.py').read_bytes()), path
        assert row['protocol_sha256'] == sha((HERE/'protocol.json').read_bytes()), path
        item = dict(method='DeltaTrace' if row['method']=='DT' else 'FlashTrace',
            output_tokens=row['output_tokens'], status=row['status'], source=str(path.relative_to(HERE)),
            sha256=sha(path.read_bytes()), devices=1,
            repeats=sum(not call['warmup'] for call in row['calls']))
        if row['status'] == 'ok':
            assert len(row['calls']) == 4 and len([c for c in row['calls'] if not c['warmup']]) == 3
            assert len({c['input_ids_sha256'] for c in row['calls']}) == 1
            item.update(seconds=row['median_seconds'], min_seconds=row['min_seconds'],
                max_seconds=row['max_seconds'], repeats=3, devices=1,
                peak_allocated_bytes=row['peak_allocated_bytes'], cold_seconds=row['calls'][0]['seconds'],
                input_ids_sha256=row['input']['input_ids_sha256'],
                formatted_prompt_tokens=row['input']['formatted_prompt_tokens'],
                generation_tokens=row['input']['generation_tokens'])
        local.append(item)
    local.sort(key=lambda row:(row['method'],row['output_tokens']))
    published.sort(key=lambda row:(row['method'],row['output_tokens']))
    assert len({(r['method'],r['output_tokens']) for r in local}) == len(local)
    expected = {(method,length) for method in ['DeltaTrace','FlashTrace'] for length in protocol['output_tokens']}
    assert {(r['method'],r['output_tokens']) for r in local} == expected, 'Incomplete length grid'
    assert all(r['status'] in ['ok','oom','timeout','error'] for r in local)
    for length in sorted({r['output_tokens'] for r in local}):
        pair = [r for r in local if r['output_tokens']==length and r['status']=='ok']
        if len(pair)==2:
            assert pair[0]['input_ids_sha256']==pair[1]['input_ids_sha256']
    plotted = [r for r in published if r['method']!='FlashTrace'] + local
    output = dict(published=published, measured=local, plotted=plotted,
        failures=[dict(directory=k[0],status=k[1],count=v) for k,v in failure_counts.items()],
        source_commit=source['commit'], distinct_hardware_sources=True)
    (HERE/'curve_data.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    fields=['cohort','method','output_tokens','status','seconds','min_seconds','max_seconds','repeats','devices','peak_allocated_bytes']
    with (HERE/'curve_data.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for cohort,rows in [('published',published),('measured',local)]:
            writer.writerows(dict(cohort=cohort,**r) for r in rows if cohort!='published' or r['method']!='FlashTrace')
    return output


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,
        'svg.fonttype':'none','svg.hashsalt':'deltatrace-rollout-scaling-v2'})
    palette={'DeltaTrace':'#c7543d','FlashTrace':'#078b75','IG':'#6887a9',
        'IG × Attention':'#a398bd','Perturbation':'#c4aa51','REAGENT':'#9e7757',
        'IFR':'#727dba','CLP':'#849168','AttnLRP':'#bf869e'}
    # Replace the old FT series with the newly measured FT series, then add DT.
    plotted=data['plotted']
    order=['DeltaTrace','FlashTrace','IG','IG × Attention','Perturbation','REAGENT','IFR','CLP','AttnLRP']
    fig,ax=plt.subplots(figsize=(6.7,4.65))
    fig.subplots_adjust(left=.15,right=.965,bottom=.145,top=.765)
    fig.text(.15,.958,'Attribution time vs. rollout length',fontsize=11,fontweight='bold')
    fig.text(.965,.958,'Qwen3-8B',fontsize=9,color='#59636b',ha='right')
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlim(8,16000);ax.set_ylim(.11,30000)
    ax.xaxis.set_major_locator(FixedLocator([10,100,500,1000,2000,5000,10000]))
    ax.xaxis.set_major_formatter(FixedFormatter(['10','100','500','1k','2k','5k','10k']))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([.2,1,10,60,600,3600,18000]))
    ax.yaxis.set_major_formatter(FixedFormatter(['0.2 s','1 s','10 s','1 min','10 min','1 h','5 h']))
    ax.yaxis.set_minor_locator(NullLocator())
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['bottom','left']:ax.spines[side].set_color('#aeb8be');ax.spines[side].set_linewidth(.8)
    ax.tick_params(length=3,color='#aeb8be')
    ax.grid(axis='y',color='#e4e8eb',linewidth=.65)
    ax.set_axisbelow(True)
    ax.set_xlabel('Rollout length (tokens)',labelpad=8)
    ax.set_ylabel('Attribution time',labelpad=10)
    for method in order:
        rows=sorted([r for r in plotted if r['method']==method and r['status']=='ok'],key=lambda r:r['output_tokens'])
        x=[r['output_tokens'] for r in rows];y=[r['seconds'] for r in rows]
        highlighted=method in ['DeltaTrace','FlashTrace']
        ax.plot(x,y,marker='o',markersize=4.2 if highlighted else 3.5,
            linewidth=2.3 if highlighted else 1.2,color=palette[method],
            alpha=1 if highlighted else .85,label=method,zorder=5 if highlighted else 2)
        if highlighted:
            ax.fill_between(x,[r['min_seconds'] for r in rows],[r['max_seconds'] for r in rows],
                alpha=.12,color=palette[method],linewidth=0)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.143,.917),ncol=3,
        frameon=False,fontsize=8.5,columnspacing=2.3,handlelength=2.1,handletextpad=.65,labelspacing=.6)
    dt_last=max([r for r in plotted if r['method']=='DeltaTrace' and r['status']=='ok'],key=lambda r:r['output_tokens'])
    ax.annotate(f'DT  {dt_last["seconds"]:.2f} s',(dt_last['output_tokens'],dt_last['seconds']),
        xytext=(-4,9),textcoords='offset points',ha='right',fontsize=8,color=palette['DeltaTrace'],fontweight='bold')
    ft_last=max([r for r in plotted if r['method']=='FlashTrace' and r['status']=='ok'],key=lambda r:r['output_tokens'])
    failures=[r for r in plotted if r['method']=='FlashTrace' and r['status']=='oom']
    if failures:
        failed_lengths=' / '.join(f'{r["output_tokens"]//1000}k' for r in sorted(failures,key=lambda r:r['output_tokens']))
        ax.annotate('FT\nOOM at '+failed_lengths,(ft_last['output_tokens'],ft_last['seconds']),
            xytext=(9,-6),textcoords='offset points',ha='left',va='top',fontsize=7.5,
            color=palette['FlashTrace'],linespacing=1.35)
    FIGURES.mkdir(parents=True,exist_ok=True)
    DELIVERY.mkdir(parents=True,exist_ok=True)
    for extension in ['pdf','svg','png']:
        metadata={'Creator':'DeltaTrace rollout scaling'}
        if extension=='pdf':metadata.update(CreationDate=None,ModDate=None)
        if extension=='svg':metadata['Date']=None
        fig.savefig(FIGURES/f'deltatrace-rollout-scaling.{extension}',dpi=320,facecolor='white',metadata=metadata)
        if extension=='svg':
            svg=FIGURES/'deltatrace-rollout-scaling.svg'
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    (DELIVERY/'deltatrace-rollout-scaling.pdf').write_bytes((FIGURES/'deltatrace-rollout-scaling.pdf').read_bytes())
    verification=dict(complete_length_grid=True,source_hashes_verified=True,paired_inputs_verified=True,
        published_FT_replaced_by_measured_FT=True,plotted_methods=order,
        mixed_hardware_overlay=True,provenance='DT/FT: one C550; remaining baselines: released 8/6-device logs.',
        protocol_sha256=sha((HERE/'protocol.json').read_bytes()),builder_sha256=sha(Path(__file__).read_bytes()),
        environment_receipt_sha256=sha((HERE/'results/environment_receipt.json').read_bytes()),
        data_sha256={name:sha((HERE/name).read_bytes()) for name in ['curve_data.csv','curve_data.json']},
        generated_files={f.name:sha(f.read_bytes()) for f in sorted(FIGURES.glob('deltatrace-rollout-scaling.*'))})
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(dict(plotted_methods=len(order),reused_baseline_points=sum(r['method']!='FlashTrace' for r in data['published']),
        measured_cells=len(data['measured']),measured_statuses=dict(Counter(r['status'] for r in data['measured'])))))


if __name__=='__main__':
    plot(collect())
