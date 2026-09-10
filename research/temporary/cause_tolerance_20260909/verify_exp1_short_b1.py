"""Check actual-input receipts, timing completeness, source hashes and serialization."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--postflight',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.summary.read_bytes())
    sha=lambda b:hashlib.sha256(b).hexdigest()
    manifest=json.loads((a.raw/'raw_manifest.json').read_bytes())
    for name,record in manifest.items():
        data=(a.raw/name).read_bytes()
        assert len(data)==record['bytes'] and sha(data)==record['sha256'],name
    postflight=json.loads(a.postflight.read_bytes())
    assert postflight['status']=='verified' and all(x['matches'] for x in postflight['files'])
    expected_methods=set(json.loads((a.raw/'benchmark_protocol.json').read_bytes())['methods'])
    drivers={sha((a.raw/name).read_bytes()) for name in ['benchmark.py','benchmark_v2.py']}
    protocols={sha(path.read_bytes()) for path in a.raw.glob('benchmark_protocol*.json')}
    excluded=set(d['selection']['excluded_attempts'])
    methods={};roots=0;audited_vectors=0;timed=0;failed=0
    for name,digest in d['raw_result_sha256'].items():
        path=a.raw/name;raw=path.read_bytes();assert sha(raw)==digest,name
        r=json.loads(raw)
        assert r['driver_sha256'] in drivers and r['protocol_sha256'] in protocols
        if path.parent.name in excluded:continue
        key=r['family'],r['method'];assert key not in methods,key
        methods[key]=r
        interrupted=path.parent.name in d['selection'].get('interrupted_attempts',{})
        assert interrupted or r['status'] in ['complete','failed'],(name,r['status'])
        assert r['generation_calls']==0 and r['sample_batch']==1
        if r['status']=='complete':
            expected_runs=24 if r['method'] in ['deltatrace_retained','ifr_multi_hop','ifr_multi_hop_both'] else 12
            assert len(r['rows'])==expected_runs,(name,len(r['rows']))
        jsonl=path.parent/'time_curve_runs.jsonl'
        if jsonl.exists():assert [json.loads(line) for line in jsonl.read_text().splitlines()]==r['rows']
        for row in r['rows']:
            timed+=1
            assert row['target_input_tokens'] in [128,256,512,1024] and row['target_output_tokens']==32
            assert row['actual_total_tokens']<=1024 and row['actual_generation_tokens']==33
            if row['status']=='ok':assert row['time_sec']>0 and row['peak_allocated_gb']>0
            else:assert row['time_sec'] is None;failed+=1
        file=path.parent/'vectors.npz'
        vectors=np.load(file) if file.exists() else None
        for case in r['cases']:
            ids=np.asarray(case['input_ids'],dtype=np.int64)[None]
            assert sha(ids.tobytes())==case['input_sha256']
            assert ids.shape[1]==case['lengths']['total_tokens']<=1024
            assert case['lengths']['formatted_prompt_tokens']+33==ids.shape[1]
            audit=case.get('audit')
            if audit is None:continue
            vector=vectors[str(case['input_length'])]
            assert sha(vector.tobytes())==audit['vector_sha256'];audited_vectors+=1
            assert all(x['shape'][-1]<=1024 for x in audit['actual_calls'])
            if r['method']=='deltatrace_retained':
                assert audit['finite'] and len(audit['actual_calls'])==1
                eos=int(ids[0,-1]);base=ids.copy();base[0,case['eligible_positions']]=eos
                pair=np.concatenate((base,ids))
                assert audit['actual_calls'][0]['input_sha256']==sha(pair.tobytes())
                assert audit['actual_calls'][0]['shape']==list(pair.shape)
            elif r['method'].startswith('ifr_'):
                assert len(audit['actual_calls'])==1
                assert audit['actual_calls'][0]['input_sha256']==case['input_sha256']
                assert audit['actual_calls'][0]['shape']==list(ids.shape)
            roots+=len(audit['actual_calls'])
    expected={(f,m) for f in ['qwen3','qwen35'] for m in expected_methods}
    missing={(f,'perturbation_REAGENT') for f in ['qwen3','qwen35']}
    assert set(methods)==expected-missing
    assert (a.raw/'user_cancellation.json').exists()
    assert len(d['selection']['not_started_after_cancellation'])==2
    for family in ['qwen3','qwen35']:
        for method in ['deltatrace_retained','ifr_multi_hop','ifr_multi_hop_both']:
            r=methods[family,method];assert r['status']=='complete'
            assert all(row['status']=='ok' for row in r['rows'])
    queue=json.loads((a.raw/'queue_v2.json').read_bytes());assert queue['status']=='complete'
    for previous,current in zip(queue['runs'],queue['runs'][1:]):
        assert previous['started']+previous['seconds']<=current['started'],(previous,current)
    recovery=json.loads((a.raw/'queue_v3.json').read_bytes())
    assert recovery['runs'][-1]['method']=='perturbation_CLP' and recovery['runs'][-1]['family']=='qwen35'
    for previous,current in zip([queue['runs'][-1],*recovery['runs'][:-1]],recovery['runs']):
        assert previous['started']+previous['seconds']<=current['started'],(previous,current)
    assert len(d['warm_comparisons'])==16
    interrupted=methods['qwen35','perturbation_CLP']
    assert timed==276+len(interrupted['rows']),timed
    assert len(excluded)==8
    assert sum(r['rows'] for r in d['excluded_attempts'])==120
    result={'status':'verified','source_reports':len(methods),'timed_rows':timed,'failed_rows':failed,
      'separately_audited_vectors':audited_vectors,'recorded_native_root_calls_in_audits':roots,
      'shared_requested_token_ids_verified_all_methods':True,
      'actual_root_inputs_verified_for':['deltatrace_retained','ifr_multi_hop','ifr_multi_hop_both','ifr_all_positions'],
      'other_method_input_scope':'Audit input_ids calls are retained when available; IG can use inputs_embeds, which this hook does not observe. Shared author-builder token IDs are verified for every method, not every perturbed/interpolated native input.',
      'physical_model_sequence_limit':1024,'fixed_target_plus_eos_tokens':33,
      'core_DT_FT_calls_successful':True,'serial_queue_nonoverlap_verified':True,
      'archived_raw_files_verified':len(manifest),'postflight_checkpoint_and_source_files_verified':len(postflight['files']),
      'postflight_sha256':sha(a.postflight.read_bytes()),
      'user_cancelled_remaining_baselines':True,
      'missing_methods_after_user_cancellation':sorted(missing),
      'interrupted_methods':d['selection']['interrupted_attempts'],
      'raw_NaN_placeholders':'Original FT/IG/LRP visual matrix convention; no raw values are altered.',
      'summary_sha256':sha(a.summary.read_bytes()),'excluded_attempts':d['excluded_attempts'],
      'limitations':'Verifies benchmark provenance and observed complete-call records, not mathematical attribution quality or an independent empty-cache cold start.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
