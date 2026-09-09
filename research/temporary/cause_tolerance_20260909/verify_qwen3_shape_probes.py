"""Verify completed shape profiles and original native varlen packing probes."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


def main():
    profile=json.loads((HERE/'qwen3_batch_shapes/results.json').read_bytes())
    assert profile['status']=='complete' and len(profile['calls'])==12
    assert profile['driver_sha256']==sha((HERE/'profile_qwen3_batch_shapes_v2.py').read_bytes())
    assert profile['vectors_sha256']==sha((HERE/'qwen3_batch_shapes/vectors.npz').read_bytes())
    z=np.load(HERE/'qwen3_batch_shapes/vectors.npz',allow_pickle=False)
    before=np.load(HERE/'qwen3_batch_extra/vectors.npz',allow_pickle=False)
    summaries=[]
    for i,group in enumerate(profile['groups']):
        for key in group:
            for batch in (1,2):
                a=z[f'warm/B{batch}/'+key];b=z[f'profile/B{batch}/'+key]
                assert np.array_equal(a,b)
                previous=before['r0_single/'+key if batch==1 else f'r0_batch/{i}/'+key]
                assert np.array_equal(a,previous)
        calls=[c for c in profile['calls'] if c['name'].startswith('profile/') and any(k in c['name'] for k in group)]
        assert len(calls)==3 and sorted(c['sample_batch'] for c in calls)==[1,1,2]
        out={'cases':group,'padding_linear_ratio':2*max(len(before['r0_single/'+key]) for key in group)/sum(len(before['r0_single/'+key]) for key in group)}
        for label,size in [('sum_B1',1),('B2',2)]:
            selected=[c for c in calls if c['sample_batch']==size]
            scopes={k:sum(c['matrix_scopes'][k]['device_us'] for c in selected)/1000 for k in ('native_root','native_replay_or_other','finite_projection')}
            # MetaX emits GPU annotation rows alongside CPU-attributed rows.
            # Use the GPU annotation exactly once; never sum these duplicates.
            ranges={}
            for name in ('DT_ROOT_CHECKPOINT','DT_REPLAY_AND_FINITE','ATTR_VENDOR_FA_FINITE_P1','ATTR_NATIVE_HALF_LINEAR'):
                vals=[]
                for c in selected:
                    records=[e for e in c['events'] if e['key']==name and e['total_cpu_us']==0 and e['self_device_us']>0]
                    assert len(records)==1
                    vals.append(records[0]['total_device_us']/1000)
                ranges[name]=sum(vals)
            clone=[e for c in selected for e in c['events'] if e['key']=='aten::clone' and e['total_cpu_us']>0]
            out[label]={'matrix_device_ms':scopes,'GPU_annotation_range_ms':ranges,'clone_calls':sum(e['count'] for e in clone),'clone_device_ms':sum(e['total_device_us'] for e in clone)/1000}
        summaries.append(out)
    for c in profile['calls']:
        assert c['status']=='returned' and c['validation']['all_passed'] and c['validation']['predicates']==828
        if c['name'].startswith('profile'):assert c['compiler_before']==c['compiler_after']
    packed=json.loads((HERE/'native_qwen3_packing/results.json').read_bytes())
    assert packed['status']=='complete' and packed['root_calls']==len(packed['calls'])==28
    assert packed['driver_sha256']==sha((HERE/'probe_native_qwen3_packing.py').read_bytes())
    assert packed['public_LSE_calls']==1 and packed['public_LSE_contract']['testing_numel']==0 and packed['public_LSE_contract']['output_exact']
    assert all(packed[k]==0 for k in ('DT_calls','FT_calls','metric_calls','generation_calls'))
    assert len(packed['isolation_checks'])==4 and all(c['hidden_and_target_exact'] for c in packed['isolation_checks'])
    assert packed['target_logprobs_sha256']==sha((HERE/'native_qwen3_packing/target_logprobs.npz').read_bytes())
    values=np.load(HERE/'native_qwen3_packing/target_logprobs.npz',allow_pickle=False);costs=[]
    for i,group in enumerate(profile['groups']):
        row={'cases':group,'modes':{}}
        for mode in ('single','dense','packed'):
            calls=[c for c in packed['calls'] if c['name'].startswith((f'r0/{i}/{mode}',f'r1/{i}/{mode}'))]
            assert len(calls)==(4 if mode=='single' else 2)
            row['modes'][mode]={'native_root_seconds':sum(c['seconds'] for c in calls)/2,'peak_allocated':max(c['peak_allocated'] for c in calls)}
        for key in group:
            for phase in ('warm','r0','r1'):
                assert np.array_equal(values[f'{phase}/{i}/dense/'+key],values[f'{phase}/{i}/packed/'+key])
        row['dense_packed_target_logprobs_exact']=True
        costs.append(row)
    out={'status':'verified','profile':summaries,'native_packing':costs,'native_packing_LSE':packed['public_LSE_contract'],
         'source_native_FA_adapter_sha256':packed['native_FA_adapter_sha256'],
         'all_profile_vectors_equal_prior_batch_run':True,'packing_cross_sample_isolation_passed':True,
         'limitations':['Profile annotations include queue gaps and overlap with child kernels; they are diagnostics, not additive cost fractions.',
                        'Native packing measurements exclude input construction and include no DT finite propagation. They do not prove complete DT speed or quality.',
                        'The frozen finite FA C tensor contract remains dense [B,H,N,128]. Native varlen feasibility does not make that existing finite kernel varlen-compatible.'],
         'next_efficiency_candidate':'Retain native replay tensors instead of clone; keep original root/checkpoint compact copies. Test exact complete vectors and full warm cost before expansion.'}
    (HERE/'qwen3_shape_probes_summary.json').write_text(json.dumps(out,indent=2)+'\n',newline='\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
