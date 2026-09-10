"""Verify and summarize passive-capture optimization from preserved raw records."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from statistics import mean,pstdev
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sha=lambda b:hashlib.sha256(b).hexdigest()
    manifest=json.loads((a.raw/'raw_manifest.json').read_bytes())
    for name,r in manifest.items():
        b=(a.raw/name).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256'],name
    post=json.loads((a.raw/'postflight.json').read_bytes())
    assert post['status']=='verified' and len(post['files'])==68 and all(x['matches'] for x in post['files'])
    assert 'no process found' in (a.raw/'final_gpu_state.txt').read_text()
    accelerated=a.repo/'deltatrace/accelerated'
    local_manifest=json.loads((accelerated/'code_local_qwen35_sources.json').read_bytes())
    for name,digest in local_manifest['files'].items():
        b=(a.repo/name).read_bytes();assert sha(b)==digest
        assert (a.raw/'production_release'/name).read_bytes()==b,name
    old=(accelerated/'qwen35/qwen35_retained_controller.py').read_bytes()
    new=(accelerated/'qwen35/qwen35_code_local_controller.py').read_bytes()
    assert old.replace(b'from qwen35_retained_capture import',b'from qwen35_code_local_capture import')==new
    sys.path.insert(0,str(accelerated))
    from native_capture_events import check_runtime
    cleanup=check_runtime()
    sources={};runs={};cells=[];cost={'timed_full_calls':0,'timed_full_call_seconds':0.,
      'separate_audit_calls':0,'separate_audit_seconds':0.,'author_compatibility_calls':0,
      'author_compatibility_seconds':0.,'model_load_seconds':0.,'eager_initialization_seconds':0.}
    failed_imports=[]
    for file in sorted(a.raw.glob('*/results.json')):
        r=json.loads(file.read_bytes());name=file.parent.name;sources[name]=sha(file.read_bytes());runs[name]=r
        cost['model_load_seconds']+=r.get('model_load_seconds',0)
        cost['eager_initialization_seconds']+=sum(x['seconds'] for x in r.get('initialization',[]))+r.get('native_eager_initialization_seconds',0)
        if r['status']=='failed':
            assert name in ('qwen3_profile','qwen35_profile') and not r.get('rows')
            failed_imports.append(name);continue
        assert r['status']=='complete',name
        rows=r.get('rows',[])
        jsonl=file.parent/'time_curve_runs.jsonl'
        if jsonl.exists():assert [json.loads(x) for x in jsonl.read_text().splitlines()]==rows
        cost['timed_full_calls']+=len(rows);cost['timed_full_call_seconds']+=sum(x['time_sec'] for x in rows)
        for row in rows:assert row['status']=='ok' and 0<row['actual_total_tokens']<=1024
        vectors=np.load(file.parent/'vectors.npz')
        for c in r['cases']:
            if name=='qwen35_author_short':continue
            ids=np.asarray(c['input_ids'],dtype=np.int64)[None]
            assert sha(ids.tobytes())==c['input_sha256']
            pair=np.concatenate((ids.copy(),ids));pair[0,c['eligible_positions']]=ids[0,-1]
            audits=c.get('audits',{'profile':c.get('audit')})
            for mode,audit in audits.items():
                assert audit is not None and audit['finite']
                key=str(c['input_length'])+('_'+mode if 'audits' in c else '')
                v=vectors[key];assert sha(v.tobytes())==audit['vector_sha256'] and np.isfinite(v).all()
                assert len(audit['actual_calls'])==1
                assert audit['actual_calls'][0]['input_sha256']==sha(pair.tobytes())
                assert audit['actual_calls'][0]['shape']==list(pair.shape)
                cost['separate_audit_calls']+=1;cost['separate_audit_seconds']+=audit['separately_charged_seconds']
            if 'audits' in c:
                ref=vectors[str(c['input_length'])+'_retained']
                for mode in c['audits']:assert np.array_equal(ref,vectors[str(c['input_length'])+'_'+mode])
                assert all(c['candidate_vectors_equal'].values())
                if r['family']=='qwen35':
                    ref_detail=c['details']['retained']
                    for mode,detail in c['details'].items():
                        for key in ['layers','root_effect','seed_effect','signed_sum','relative_residual',
                          'norm_gate_rules','attention_pv_rules','finite_fla_by_layer','key_norm_by_layer',
                          'actual_head_input_shapes','selected_predictor_rows','target_logp0','target_logp1']:
                            assert detail[key]==ref_detail[key],(name,c['input_length'],mode,key)
            for mode in sorted({x.get('candidate_mode') for x in rows if x.get('candidate_mode')}):
                group=[x for x in rows if x['target_input_tokens']==c['input_length'] and x['candidate_mode']==mode and x['phase']=='measured']
                assert len(group)==3
                cells.append({'run':name,'family':r['family'],'target_input_tokens':c['input_length'],
                  'actual_total_tokens':c['lengths']['total_tokens'],'mode':mode,
                  'seconds_mean':mean(x['time_sec'] for x in group),'seconds_std':pstdev(x['time_sec'] for x in group),
                  'times_seconds':[x['time_sec'] for x in group],
                  'peak_allocated_gb':max(x['peak_allocated_gb'] for x in group)})
    author=runs['qwen35_author_short'];plan=json.loads((a.raw/'author_short_plan.json').read_bytes())
    assert len(author['cases'])==len(plan['cases'])==11 and author['all_vectors_exact']
    assert author['plan_sha256']==sha((a.raw/'author_short_plan.json').read_bytes())
    va=np.load(a.raw/'qwen35_author_short/vectors.npz')
    for c,planned in zip(author['cases'],plan['cases']):
        assert c['key']==planned['key'] and c['input_sha256']==planned['input_sha256'] and c['exact']
        ids=np.asarray(planned['input_ids'],dtype=np.int64)[None];assert sha(ids.tobytes())==planned['input_sha256']
        pair=np.concatenate((ids.copy(),ids));pair[0,[planned['user_positions'][i] for i in planned['keep']]]=ids[0,-1]
        assert len(c['calls'])==2
        for call in c['calls']:
            assert call['actual_root']['input_sha256']==sha(pair.tobytes())
            assert call['actual_root']['sample_batch']==1 and call['actual_root']['endpoint_batch']==2
            assert sha(va[c['key']+'/'+call['mode']].tobytes())==call['vector_sha256']
            cost['author_compatibility_calls']+=1;cost['author_compatibility_seconds']+=call['seconds_including_any_compilation']
        assert np.array_equal(va[c['key']+'/retained'],va[c['key']+'/local'])
        assert c['calls'][0]['details']['layers']==c['calls'][1]['details']['layers']
    production=[c for c in cells if c['run']=='qwen35_code_local_production'];comparisons=[]
    for n in [128,256,512,1024]:
        old=next(x for x in production if x['target_input_tokens']==n and x['mode']=='retained')
        new=next(x for x in production if x['target_input_tokens']==n and x['mode']=='local')
        comparisons.append({'target_input_tokens':n,'actual_total_tokens':new['actual_total_tokens'],
          'retained_seconds':old['seconds_mean'],'local_seconds':new['seconds_mean'],
          'latency_change_percent':100*(new['seconds_mean']/old['seconds_mean']-1),
          'retained_peak_allocated_gb':old['peak_allocated_gb'],'local_peak_allocated_gb':new['peak_allocated_gb']})
    assert cost['timed_full_calls']==90 and cost['separate_audit_calls']==23 and cost['author_compatibility_calls']==22
    summary={'status':'verified','raw_manifest_files':len(manifest),'source_reports':sources,
      'production_manifest_sha256':sha((accelerated/'code_local_qwen35_sources.json').read_bytes()),
      'controller_import_only_derivation_verified':True,'monitoring_cleanup_checks':cleanup,
      'postflight_files_verified':len(post['files']),'GPU_jobs_finished':True,
      'cells':cells,'production_comparisons':comparisons,'author_short_cases':[{'key':c['key'],'total_tokens':c['total_tokens'],'exact':c['exact']} for c in author['cases']],
      'production_synthetic_vectors_exact':4,'production_author_vectors_exact':11,
      'native_root_input_checks':True,'original_layer_diagnostics_exact':True,
      'failed_import_attempts_preserved':failed_imports,'recorded_cost':cost,
      'adoption':'Explicit Qwen3.5 code-local factory; short-input latency gains in first three bins. Largest bin +0.47% does not establish a gain. No Qwen3 adoption or input-dependent dispatch.',
      'limits':['CPython >=3.12; single-use observer contexts; existing sys.setprofile profilers are refused.',
        'Synthetic warmed timing: one warm call per mode/length, three interleaved measured calls per mode. Model load/constructor and shape compilation are separate.',
        'Author calls check the 11 original development examples <=1024 only, with their complete frozen targets. Five longer examples excluded by input-length constraint.',
        'Complete-vector identity establishes compatibility for tested inputs, not a proof for every model/input or a new quality improvement.',
        'Profiled calls and real-author compatibility calls include extra instrumentation or compilation and are not speed-acceptance measurements.',
        'First local-monitor candidate retained callback cycles and increased memory; production breaks the cycle. Both candidate versions and results remain archived.',
        'Qwen3 pilot mean did not improve. This experiment has not established that DT is faster than FT.']}
    (a.output/'code_local_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (a.output/'code_local_cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in cells[0] if k!='times_seconds'],extrasaction='ignore');w.writeheader();w.writerows(cells)
    print(json.dumps({'status':summary['status'],'production_comparisons':comparisons,'author_exact':11,'recorded_cost':cost},indent=2))


if __name__=='__main__':main()
