"""One rollout figure from original released rows and verified new API calls."""
from collections import defaultdict,Counter
from pathlib import Path
import csv,hashlib,io,json,statistics,zipfile
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
STUDY=HERE/'verified_memory_v4'
sha=lambda b:hashlib.sha256(b).hexdigest()
METHODS={'IG':'IG','attention_I_G':'IG × Attention','perturbation_all':'Perturbation',
 'perturbation_REAGENT':'REAGENT','ifr_all_positions':'IFR','perturbation_CLP':'CLP','attnlrp':'AttnLRP'}
LABELS={'deltatrace_streamed':'DeltaTrace','ifr_multi_hop_both':'FlashTrace','ifr_multi_hop':'FT multi-hop'}
LENGTHS=[10,100,500,1000,2000,5000,10000]
RUN='memory_v4_rollout'

def archive():
    path=STUDY/('raw/evidence_'+RUN+'.zip')
    receipt=json.loads((STUDY/(RUN+'_archive_verification.json')).read_bytes())
    assert sha(path.read_bytes())==receipt['sha256']
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('archive_manifest.json'))
        assert set(z.namelist())==set(manifest['files'])|{'archive_manifest.json'}
        files={}
        for name,row in manifest['files'].items():
            data=z.read(name);assert len(data)==row['bytes'] and sha(data)==row['sha256'],name
            files[name]=data
    return files,receipt

def collect():
    accepted=json.loads((STUDY/'memory_v4_confirmation_verification.json').read_bytes())
    assert accepted['acceptance_passed'] and accepted['independently_recomputed_all_statistics_equal_to_remote']
    protocol=json.loads((STUDY/(RUN+'_protocol.json')).read_bytes())
    protocol_sha=sha((STUDY/(RUN+'_protocol.json')).read_bytes())
    upstream=HERE/'upstream';source=json.loads((upstream/'source_manifest.json').read_bytes())
    assert sha((upstream/'source_manifest.json').read_bytes())==protocol['published_reference']['manifest_sha256']
    assert sha((upstream/'run_time_curve.py').read_bytes())==protocol['exp1_sha256']
    selected={'out-0':[10,100],'out-2':[500],'out-3':[1000],'out-4':[2000],'out-5':[5000]}
    groups=defaultdict(list);failures=Counter()
    for entry in source['files']:
        path=upstream/entry['path'];raw=path.read_bytes()
        assert len(raw)==entry['bytes'] and sha(raw)==entry['sha256']
        if path.suffix!='.jsonl':continue
        for line_number,line in enumerate(raw.decode().splitlines(),1):
            row=json.loads(line);method=row['attr_func'];n=row['target_output_tokens'];folder=path.parent.name
            if folder not in selected or method not in METHODS or row['target_input_tokens']!=10 or n not in selected[folder]:continue
            if row['status']!='ok':failures[(method,n,row['status'])]+=1;continue
            assert row['time_sec']>0
            row['source']={'file':entry['path'],'line':line_number,'sha256':entry['sha256']}
            groups[(method,n)].append(row)
    published=[]
    for (method,n),rows in groups.items():
        devices={len(r['peak_mem_by_device_gb']) for r in rows};assert len(devices)==1
        times=[r['time_sec'] for r in rows]
        published.append({'series':'released_reference','method':METHODS[method],'output_tokens':n,'status':'ok',
            'seconds':statistics.mean(times),'min_seconds':min(times),'max_seconds':max(times),'repeats':len(times),
            'devices':devices.pop(),'actual_formatted_prompt_tokens':rows[0]['actual_formatted_prompt_tokens'],
            'actual_generation_tokens':rows[0]['actual_generation_tokens'],'sources':[r['source'] for r in rows]})
    assert len(published)==35,len(published)
    files,receipt=archive();queue=json.loads(files[RUN+'/queue.json']);assert queue['status']=='complete'
    assert len(queue['jobs'])==21
    local=[];inputs={};environments=set();all_rows=[]
    for job in queue['jobs']:
        name=RUN+'/'+job['name']+'/results.json';data=json.loads(files[name]);method=job['method'];n=job['output_length']
        assert data['driver_sha256']==protocol['driver_sha256'] and data['protocol_sha256']==protocol_sha
        for helper,digest in protocol['common_loading_helpers'].items():assert sha(files[helper])==digest
        assert not data['model_loading_strategy']['model_forward_modified'] and not data['model_loading_strategy']['FT_implementation_modified']
        assert data['method']==method and data['generation_calls']==0 and len(data['cases'])==1
        case=data['cases'][0];ids=np.asarray(case['input_ids'],dtype=np.int64)
        assert sha(ids.tobytes())==case['input_sha256']
        assert len(ids)==case['lengths']['total_tokens'] and case['input_length']==10 and case['output_length']==n
        if n in inputs:assert inputs[n]==case['input_sha256']
        else:inputs[n]=case['input_sha256']
        env=data['environment'];environments.add(tuple(env[k] for k in ['torch','transformers','device','checkpoint','native_model_sha256','dtype']))
        rows=data['rows'];measured=[r for r in rows if r['phase']=='measured'];warm=[r for r in rows if r['phase']=='warm']
        all_rows.extend(dict(r,job=job['name']) for r in rows)
        complete=data['status']=='complete' and len(rows)==5 and all(r['status']=='ok' for r in rows)
        point={'series':'same_C550_measured','method':LABELS[method],'method_id':method,'output_tokens':n,
            'status':'ok' if complete else 'oom' if any(r['status']=='oom' for r in rows) else 'failed',
            'seconds':None,'devices':1,'source':name,'source_sha256':sha(files[name]),'input_sha256':case['input_sha256'],
            'actual_total_tokens':len(ids),'actual_formatted_prompt_tokens':case['lengths']['formatted_prompt_tokens'],
            'actual_generation_tokens':case['lengths']['generation_tokens'],'model_load_seconds':data['model_load_seconds'],
            'model_loading_strategy':data['model_loading_strategy'],
            'model_load_memory':data['model_load_memory'],'pre_call_initialization_memory':data['pre_call_initialization_memory'],
            'controller_setup_seconds':sum(x['seconds'] for x in data['initialization']),
            'first_geometry_call':warm[0] if warm else None,'second_warm_call':warm[1] if len(warm)>1 else None,
            'row_statuses':[r['status'] for r in rows],'error':data.get('error'),'returncode':job.get('returncode'),
            'timeout_seconds':job.get('timeout_seconds'),'audit':case.get('audit')}
        if complete:
            assert len(warm)==2 and len(measured)==3
            assert all(r['input_sha256']==case['input_sha256'] for r in rows)
            with np.load(io.BytesIO(files[RUN+'/'+job['name']+'/vectors.npz'])) as z:
                vector=z[str(n)];audit=case['audit'];assert sha(vector.tobytes())==audit['vector_sha256']
                assert list(vector.shape)==audit['vector_shape'] and bool(np.isfinite(vector).all())==audit['finite']
                if method=='deltatrace_streamed':
                    assert audit['finite'] and len(vector)==len(ids)
                    detail=case['details'];assert detail['deferred_validation']['all_passed']
                    meta=detail['output_materialization'];assert meta['fresh_mutable_containers_each_return'] and not meta['attribution_values_reused']
                    assert detail['deferred_validation']['predicates'] in [829,865]
                    assert len(detail['layer_checks'])==36 and detail['native_layer_replay_calls']==36
                    assert all(c['native_input_exact'] and c['native_output_exact'] for c in detail['native_layer_boundary_checks']['paired_batch'])
                    if len(ids)<=1024:
                        assert case['retained_vector_exact'] and case['retained_math_exact'] and np.array_equal(vector,z[str(n)+'_retained'])
                        for k in ['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']:assert detail[k]==case['retained_details'][k]
            times=[r['time_sec'] for r in measured]
            point.update(seconds=statistics.mean(times),min_seconds=min(times),max_seconds=max(times),repeats=3,times_seconds=times,
                peak_allocated_gb=max(data['model_load_memory']['peak_allocated_gb'],data['pre_call_initialization_memory']['peak_allocated_gb'],*(r['peak_allocated_gb'] for r in rows)),
                peak_reserved_gb=max(data['model_load_memory']['peak_reserved_gb'],data['pre_call_initialization_memory']['peak_reserved_gb'],*(r['peak_mem_reserved_gb'] for r in rows)),
                steady_peak_allocated_gb=max(r['peak_allocated_gb'] for r in measured),steady_peak_reserved_gb=max(r['peak_mem_reserved_gb'] for r in measured),
                output_vector_finite=case['audit']['finite'],retained_vector_exact=case.get('retained_vector_exact'))
        local.append(point)
    assert len(environments)==1 and len(inputs)==7
    return {'protocol_sha256':protocol_sha,'raw_archive':receipt,'new_curves_timing_scope':'Original author exp1 synchronized complete API timer. Mean of3 measured calls after2 separately recorded warm calls; fresh process per cell.',
        'published_reference_scope':'Original released rows on6/8 GPUs, nominal input10. Historical reference only; no cross-hardware speedup claims, interpolation, timing rescaling or inferred missing points.',
        'environment':dict(zip(['torch','transformers','device','checkpoint','native_model_sha256','dtype'],next(iter(environments)))),
        'local':sorted(local,key=lambda r:(r['method'],r['output_tokens'])),
        'published':sorted(published,key=lambda r:(r['method'],r['output_tokens'])),
        'published_failures':[{'method':k[0],'output_tokens':k[1],'status':k[2],'count':v} for k,v in failures.items()],
        'measured_all_rows_count':len(all_rows),'all_measured_rows':all_rows}

def draw(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter,LogLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.labelsize':12,'axes.titlesize':15,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'deltatrace-rollout-20260911'})
    fig,ax=plt.subplots(figsize=(9.3,5.0));fig.subplots_adjust(left=.095,right=.975,bottom=.19,top=.88)
    colors=['#b07aa1','#e6a14b','#7e95a9','#7c9b65','#a08469','#74a6a3','#9a90bc']
    for color,method in zip(colors,METHODS.values()):
        rows=sorted([r for r in data['published'] if r['method']==method],key=lambda r:r['output_tokens'])
        values={r['output_tokens']:r['seconds'] for r in rows}
        ax.plot(LENGTHS,[values.get(n,np.nan) for n in LENGTHS],color=color,marker='o',ms=4,lw=1.25,alpha=.8,label=method)
    styles={'FlashTrace':('#db674d','s','-',2.1),'FT multi-hop':('#e5a142','D','--',1.8),'DeltaTrace':('#1769aa','o','-',2.8)}
    for method,(color,marker,line,width) in styles.items():
        rows=sorted([r for r in data['local'] if r['method']==method],key=lambda r:r['output_tokens'])
        valid={r['output_tokens']:r for r in rows if r['status']=='ok'}
        # Missing/failed cells split lines; do not bridge an unmeasured interval.
        y=[valid[n]['seconds'] if n in valid else np.nan for n in LENGTHS]
        ax.plot(LENGTHS,y,color=color,marker=marker,ms=5.4,lw=width,ls=line,label=method,zorder=5)
        # OOM has no measured latency: mark the exact x coordinate outside the y axis.
        oom=[r['output_tokens'] for r in rows if r['status']=='oom']
        if oom:
            band={'FlashTrace':1.045,'FT multi-hop':1.10}[method]
            marks,=ax.plot(oom,[band]*len(oom),transform=ax.get_xaxis_transform(),
                color=color,marker=r'$\times$',ms=8.5,ls='none',clip_on=False,zorder=7)
            marks.set_gid('oom-'+method.replace(' ','-'))
        good=[r for r in rows if r['status']=='ok']
        if good:
            ax.errorbar([r['output_tokens'] for r in good],[r['seconds'] for r in good],
                yerr=np.asarray([[r['seconds']-r['min_seconds'] for r in good],[r['max_seconds']-r['seconds'] for r in good]]),fmt='none',ecolor=color,elinewidth=1,capsize=2,zorder=4)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(7,14500)
    ax.set_xlabel('Rollout length (tokens)');ax.set_ylabel('Attribution time (seconds)')
    ax.xaxis.set_major_locator(FixedLocator(LENGTHS));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{int(x):,}'))
    ax.yaxis.set_major_locator(LogLocator(base=10));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    ax.grid(True,which='major',color='#e1e6eb',lw=.7);ax.set_axisbelow(True)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_color('#9da6ae')
    handles,labels=ax.get_legend_handles_labels();order=[labels.index(x) for x in ['DeltaTrace','FlashTrace','FT multi-hop']]+list(range(7))
    fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='lower center',bbox_to_anchor=(.53,.009),ncol=5,frameon=False,fontsize=8.6,columnspacing=1.2,handlelength=2.1)
    target=HERE/'figures';target.mkdir(exist_ok=True)
    for suffix in ['png','svg','pdf']:
        metadata={'Date':None} if suffix=='svg' else {'CreationDate':None,'ModDate':None} if suffix=='pdf' else None
        fig.savefig(target/('deltatrace-rollout-scaling.'+suffix),dpi=220,facecolor='white',metadata=metadata)
    plt.close(fig)
    return {'numpy':np.__version__,'matplotlib':matplotlib.__version__,'figure_inches':[9.3,5.0],'png_dpi':220,'svg_hashsalt':'deltatrace-rollout-20260911',
        'title':False,'subtitle':False,'footnote':False,'oom_marker':'×','oom_marker_count':sum(r['status']=='oom' for r in data['local']),
        'oom_positions':'Exact rollout length, axes-coordinate band outside the latency axis; no time value assigned.'}

def main():
    data=collect();(HERE/'curve_data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',newline='\n')
    fields=['series','method','output_tokens','status','seconds','min_seconds','max_seconds','repeats','devices','actual_formatted_prompt_tokens','actual_generation_tokens','actual_total_tokens','peak_allocated_gb','peak_reserved_gb','source','source_sha256']
    with (HERE/'curve_data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(data['local']+data['published'])
    plotting_environment=draw(data)
    verified={'local_cells':len(data['local']),'successful_local_cells':sum(r['status']=='ok' for r in data['local']),'published_reference_cells':len(data['published']),
        'same_input_hash_across_all_three_local_methods':True,'all_raw_sources_and_vectors_verified':True,'excluded_local_rows':0,
        'figure':'figures/deltatrace-rollout-scaling.png','one_figure_in_three_formats':True,'protocol_sha256':data['protocol_sha256'],'plotting_environment':plotting_environment}
    (HERE/'verification.json').write_text(json.dumps(verified,indent=2)+'\n',newline='\n');print(json.dumps(verified))

if __name__=='__main__':main()
