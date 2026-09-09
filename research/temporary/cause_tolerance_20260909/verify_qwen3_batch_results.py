"""Verify batch provenance, complete calls, vector reuse and original curves.

No model execution or alternate metric implementation. The AUC reconstruction
and input hashes audit the already saved original scorer outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


def read(path):return json.loads(path.read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',type=Path,required=True)
    args=p.parse_args()
    ref_path=args.snapshots/'tmp/codex_clean_development16_20260909_v1/qwen3/results.json'
    ref=read(ref_path);references={f"{c['dataset']}_{c['index']}":c for c in ref['cases']}
    extra_path=HERE/'qwen3_batch_extra_inputs.json';extra=read(extra_path)
    references.update({f"{c['dataset']}_{c['index']}":c for c in extra['cases']})
    summaries={};reports={};vectors={}
    for name,driver in [('qwen3_batch_pilot','benchmark_qwen3_batch.py'),
                        ('qwen3_batch16','benchmark_qwen3_batch_v2.py'),
                        ('qwen3_batch16_guarded','benchmark_qwen3_batch_guarded.py'),
                        ('qwen3_batch_extra','benchmark_qwen3_batch_guarded.py')]:
        r=read(HERE/name/'results.json');reports[name]=r
        assert r['status']=='complete' and r['driver_sha256']==sha((HERE/driver).read_bytes())
        assert r['reference_sha256']==sha((extra_path if name.endswith('extra') else ref_path).read_bytes())
        assert r['vectors_sha256']==sha((HERE/name/'vectors.npz').read_bytes())
        for filename,digest in r['candidate_sources'].items():assert sha((HERE/filename).read_bytes())==digest,filename
        for filename,digest in r['baseline_sources']['files'].items():assert sha((HERE.parents[2]/filename).read_bytes())==digest,filename
        z=np.load(HERE/name/'vectors.npz',allow_pickle=False);vectors[name]=z
        calls=[c for c in r['calls'] if c['name']!='model_load']
        assert all(c['status']=='returned' for c in r['calls'])
        assert all(r[k]==0 for k in ('FT_calls','generation_calls','metric_calls'))
        for c in calls:
            d=c.get('details');v=d['deferred_validation'] if d else c['validation']
            assert v['all_passed'] and v['predicates']==828 and v['statistics']==36
            if d:
                assert c['sample_batch']==d['sample_batch']==2 and d['endpoint_batch']==4
                assert d['native_layer_replay_calls']==d['extra_native_fa_attention_calls']==36
                n=d['actual_root']['shape'][1];packed=np.full((4,n),151645,dtype=np.int64)
                for b,t in enumerate(d['prefix_receipts']):
                    row=references[t['case']];length=len(row['input_ids'])
                    assert t['input_sha256']==row['input_sha256'] and t['tail_target_count']==0
                    assert t['valid_length']==length and t['padded_length']==n
                    packed[2*b:2*b+2,:length]=row['input_ids']
                    packed[2*b,[row['user_positions'][j] for j in row['keep']]]=151645
                assert sha(packed.tobytes())==d['actual_root']['input_sha256']
            if c['name'].startswith(('r0_','r1_')):
                assert c['compiler_after']['unique_graphs']==c['compiler_before']['unique_graphs']
        for c in r['comparisons']:
            key=c['case'];i=next(i for i,g in enumerate(r['groups']) if key in g)
            a=z['r0_single/'+key];b=z[f'r0_batch/{i}/'+key]
            assert a.shape==b.shape==(len(references[key]['input_ids']),)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            assert np.array_equal(a,z['r1_single/'+key]) and np.array_equal(b,z[f'r1_batch/{i}/'+key])
            assert bool(np.array_equal(a,b))==c['full_vector_equal']
        for c in r['prefix_checks']:
            assert c['tail_target_count']==0
            if name!='qwen3_batch_pilot':assert c['same_shape_changed_tail_hidden_exact']
        total=lambda mode:sum(c['seconds'] for c in calls if c['name'].startswith(('r0_'+mode,'r1_'+mode)))/2
        assert total('single')==r['baseline_seconds_per_pass'] and total('batch')==r['batch_seconds_per_pass']
        summaries[name]={'n':sum(map(len,r['groups'])),'complete_DT_calls':len(calls),
            'sample_vectors':sum(c['sample_batch'] for c in calls),'same_shape_tail_probes':r.get('native_tail_probe_calls',0),
            'baseline_seconds':total('single'),'candidate_seconds':total('batch'),
            'reduction_fraction':r['warm_reduction_fraction'],'throughput_gate_passed':r['throughput_gate_passed'],
            'warm_peaks':{m:max(c['peak_allocated'] for c in calls if c['name'].startswith(('r0_'+m,'r1_'+m))) for m in ('single','batch')},
            'cold_calls':[{'name':c['name'],'seconds':c['seconds'],'peak':c['peak_allocated']} for c in calls if c['name'].startswith('warm_')],
            'max_vector_relative_l2':max(c['relative_l2'] for c in r['comparisons'])}
        summaries[name]['group_costs']=[]
        for i,group in enumerate(r['groups']):
            baseline=sum(c['seconds'] for c in calls if any(c['name']==f'r{j}_single/'+key for key in group for j in (0,1)))/2
            batch=sum(c['seconds'] for c in calls if any(c['name']==f'r{j}_batch/{i}' or c['name'].startswith(f'r{j}_batch/{i}/') for j in (0,1)))/2
            summaries[name]['group_costs'].append({'cases':group,'baseline_seconds':baseline,'candidate_seconds':batch,'reduction_fraction':1-batch/baseline})
    metric=read(HERE/'qwen3_batch16_metrics/results.json')
    assert metric['status']=='complete' and len(metric['cases'])==16
    assert metric['batch_result_sha256']==sha((HERE/'qwen3_batch16/results.json').read_bytes())
    assert metric['driver_sha256']==sha((HERE/'score_qwen3_batch_vectors.py').read_bytes())
    assert metric['native_metric_root_calls']==672 and len(metric['metric_calls'])==32
    assert metric['attribution_calls']==metric['FT_calls']==metric['generation_calls']==0
    metrics={};case_deltas=[]
    z=vectors['qwen3_batch16'];guard=vectors['qwen3_batch16_guarded']
    for c in metric['cases']:
        key=c['case'];row=references[key];i=next(i for i,g in enumerate(reports['qwen3_batch16']['groups']) if key in g)
        gi=next(i for i,g in enumerate(reports['qwen3_batch16_guarded']['groups']) if key in g)
        for mode in ('single','batch'):
            array=z['r0_single/'+key if mode=='single' else f'r0_batch/{i}/'+key]
            assert sha(array.tobytes())==c[mode+'_vector_sha256']
            m=c['metrics'][mode];curve=m['positive'];signed=array[row['user_positions']].astype(np.float32)
            assert m['rise']==curve['rise'] and m['mas']==curve['mas']
            proof=m['signed_reuse_proof'];step=proof['zero_step'];y=np.asarray(curve['normalized_model_response'])
            assert np.all(y[step:]==0) and len(curve['deleted_user_indices'][step])==proof['deletion_budget_at_zero']
            # Strictly positive ordered prefix is unaffected by clipping negatives.
            assert np.all(signed[curve['deleted_user_indices'][step]]>0)
            assert int((signed[row['keep']]>0).sum())==proof['positive_count']
            assert abs(float(np.trapezoid(y,dx=.05))-m['rise'])<1e-12
            for deleted,digest in zip(curve['deleted_user_indices'],curve['actual_input_hashes']):
                ids=np.array(row['input_ids'],dtype=np.int64);ids[[row['user_positions'][j] for j in deleted]]=151645
                assert sha(ids.tobytes())==digest
        assert np.array_equal(guard['r0_single/'+key],z['r0_single/'+key])
        use='single' if len(reports['qwen3_batch16_guarded']['groups'][gi])==1 else 'batch'
        old=z['r0_single/'+key if use=='single' else f'r0_batch/{i}/'+key]
        assert np.array_equal(guard[f'r0_batch/{gi}/'+key],old)
        metrics[key]={'single':c['metrics']['single'],'guarded':c['metrics'][use]}
        case_deltas.append({'case':key,**{k:(metrics[key]['guarded'][k]-metrics[key]['single'][k]) if metrics[key]['single'][k] is not None else None for k in ('rise','mas','needle')},
                            'guarded_score_reuse':'Complete vector equality to already scored '+use+' vector.'})
    means=[]
    for task in ('niah_mq_q2','morehopqa'):
        ms=[m for key,m in metrics.items() if key.startswith(task+'_')]
        means.append({'dataset':task,'n':len(ms),**{mode:{k:statistics.mean(m[mode][k] for m in ms) if ms[0][mode][k] is not None else None for k in ('rise','mas','needle')} for mode in ('single','guarded')}})
    assert reports['qwen3_batch16_guarded']['throughput_gate_passed']
    assert not reports['qwen3_batch_extra']['throughput_gate_passed']
    out={'status':'verified','runs':summaries,'original_metrics':{'native_score_forwards':672,'means':means,'case_deltas':case_deltas},
         'decision':'Keep batch implementation in temporary research. Original16 speedup transfers weakly to frozen VT/Hotpot six (below predeclared 3% cost gate); no additional expensive metrics and no production promotion.',
         'extra_task_RISE_MAS_status':'Not measured. Extra vectors and cost do not establish quality preservation.',
         'clean_or_native_sources_modified':False,'production_default_changed':False}
    (HERE/'qwen3_batch_summary.json').write_text(json.dumps(out,indent=2)+'\n',newline='\n')
    print(json.dumps({'status':'verified','runs':{k:{n:v for n,v in d.items() if n!='cold_calls'} for k,d in summaries.items()},'metric_means':means},indent=2))


if __name__=='__main__':main()
