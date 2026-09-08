"""Independent CPU arithmetic audit, never an attribution/model runtime backend."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_FT_native_content_20260908_v2';sha=lambda b:hashlib.sha256(b).hexdigest()
start=time.perf_counter()
def extract(directory):
    with zipfile.ZipFile(directory/'review_bundle.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            f=(directory/name).resolve();assert f.is_relative_to(directory.resolve());f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(name))
    raw=(directory/'results.json').read_bytes();r=json.loads(raw)
    for name,digest in r['protocol']['files_sha256'].items():
        source=(directory/name).read_bytes();assert sha(source)==digest;ast.parse(source)
    assert json.loads((directory/'protocol.json').read_bytes())==r['protocol']
    for a in r['artifacts']:
        if a['download']:assert sha((directory/a['file']).read_bytes())==a['sha256'] and (directory/a['file']).stat().st_size==a['bytes']
    return raw,r
failed_raw,failed=extract(A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_FT_native_content_20260908_v1')
assert failed['status']=='failed' and not failed['calls'] and failed['parameter_loads']==0 and failed['meta_model_constructions']==0
raw,r=extract(D);assert r['status']=='both_native_content_adjoint_FT_boundaries_executed'
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
for key in ['full_model_loads','root_forwards','decoder_replays','generation_calls','quality_queries','whole_FT_attributions']:assert r[key]==0
assert r['parameter_loads']==5 and r['meta_model_constructions']==1 and len(r['calls'])==15
expected=['load_parameter_'+n for n in ['3.self_attn.o_proj.weight','3.input_layernorm.weight','0.linear_attn.out_proj.weight','0.linear_attn.norm.weight','0.linear_attn.conv1d.weight']]
expected+=['layer3_'+n for n in ['public_FA_expanded_heads_forward','public_FA_value_backward','FT_projection_and_author_proximity_with_diagnostics','extra_fixed_CPU_review_projection']]
expected+=['layer0_'+n for n in ['public_FLA_normalized_forward','public_FLA_value_backward','public_linear_conv_forward','public_linear_conv_backward','FT_projection_and_author_proximity_with_diagnostics','extra_fixed_CPU_review_projection']]
assert [x['kind'] for x in r['calls']]==expected
def metric(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'relative_L2':float(np.linalg.norm(b-a))/na if na else None,'max_abs':float(np.abs(b-a).max()),'cosine':float(a@b)/(na*nb) if na*nb else None}
def bf16(x):
    x=np.ascontiguousarray(x,dtype=np.float32);bits=x.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
lengths=[605,368];summaries={}
for i in [3,0]:
    a=dict(np.load(D/f'layer{i}_content_review.npz',allow_pickle=False));row=r['layers'][str(i)]
    assert row['status']=='complete' and all(np.isfinite(x).all() for x in a.values())
    assert np.count_nonzero(a['components'][1,368:])==0 and np.count_nonzero(a['token_scores'][1,368:])==0
    b,t,h,d=a['components'].shape;reference=np.zeros((b,t,h,d),dtype=np.float64)
    if i==3:
        assert any('FlashAttn' in x['node'] for x in row['native_node_events'])
        # Saved public LSE from the genuine original GQA call. Only32 sink rows
        # at once; no global attention matrix and no runtime substitution.
        for bi,n in enumerate(lengths):
            sink=np.flatnonzero(a['sink_weights'][bi,:n]);key_positions=np.arange(n)
            for hi in range(h):
                k=a['key'][bi,hi//4,:n].astype(np.float64)
                for begin in range(0,len(sink),32):
                    rows=sink[begin:begin+32];q=a['query'][bi,hi,rows].astype(np.float64)
                    logits=(q@k.T)*row['fixed_CPU_reference']['FA_scale']
                    probs=np.exp(logits-a['LSE'][bi,hi,rows,None].astype(np.float64))
                    probs*=key_positions[None,:]<=rows[:,None]
                    reference[bi,:n,hi]+=probs.T@a['seed_native'][bi,rows,hi].astype(np.float64)
        source=a['value'];gradient=a['value_gradient']
    else:
        assert any('GatedDelta' in x['node'] for x in row['native_node_events'])
        assert any('CausalConv' in x['node'] for x in row['native_node_events'])
        # Independent fixed-route GDN recurrence adjoint. State[H,K,V] only.
        scale=row['fixed_CPU_reference']['GDN_scale']
        for bi,n in enumerate(lengths):
            state=np.zeros((h,a['q'].shape[-1],d),dtype=np.float64)
            for ti in reversed(range(n)):
                q=a['q'][bi,ti].astype(np.float64);k=a['k'][bi,ti].astype(np.float64)
                beta=a['beta'][bi,ti].astype(np.float64)
                state+=scale*q[:,:,None]*a['seed_native'][bi,ti,None,:].reshape(h,1,d).astype(np.float64)
                kd=(k[:,:,None]*state).sum(1)
                reference[bi,ti]=beta[:,None]*kd
                state=(state-beta[:,None,None]*k[:,:,None]*kd[:,None,:])*np.exp(a['raw_g'][bi,ti].astype(np.float64))[:,None,None]
        w=a['conv_weights'].astype(np.float64);width=w.shape[-1]
        dc=np.zeros_like(a['source_gradient'],dtype=np.float64)
        for bi,n in enumerate(lengths):
            for offset in range(width):
                shift=width-1-offset
                dc[bi,:n-shift]+=a['pre_value_seed'][bi,shift:n].astype(np.float64)*w[:,:,offset]
        source=a['source'];gradient=a['source_gradient']
    s={'native_value_gradient_vs_CPU':metric(reference,a['value_gradient']),
        'value_gradient_per_sample':[metric(reference[bi,:n],a['value_gradient'][bi,:n]) for bi,n in enumerate(lengths)],
        'head_content_reconstruction':metric(a['head_native_sum'],a['components'].astype(np.float64).sum(1)),
        'head_content_per_sample':[metric(a['head_native_sum'][bi],a['components'][bi].astype(np.float64).sum(0)) for bi in range(b)],
        'individual_head_relative_L2':[[metric(a['head_native_sum'][bi,hi],a['components'][bi,:,hi].astype(np.float64).sum(0))['relative_L2'] for hi in range(h)] for bi in range(b)],
        'projected_reconstruction':metric(a['native_projected_sum'],a['projected_head_sums'].sum(1)),
        'padding_components_nonzero':int(np.count_nonzero(a['components'][1,368:])),
        'normalization_sum':(a['token_scores'].astype(np.float64).sum(1)+a['residual_score']).tolist(),
        'native_node_events':row['native_node_events']}
    assert np.array_equal(source*gradient,a['components'])
    assert abs(s['head_content_reconstruction']['relative_L2']-row['head_content_reconstruction']['relative_L2'])<1e-10
    assert abs(s['projected_reconstruction']['relative_L2']-row['projected_reconstruction']['relative_L2'])<1e-10
    if i==0:s['native_conv_source_gradient_vs_CPU']=metric(dc,gradient)
    for name in ['auxiliary_core_vs_saved_native','ungated_core_vs_native_gated_output','frozen_output_gain_vs_native_gated_output','official_raw_weight_RMS_vs_native','frozen_SiLU_vs_native_fused_value','conv_content_sum_vs_postconv','local_token_proximity_summary']:
        if name in row:s[name]=row[name]
    selected=a['components'][:,row['fixed_CPU_reference']['source_rows']][:,:,row['fixed_CPU_reference']['heads']]
    exact=np.einsum('brhd,hdc->brhc',selected.astype(np.float64),a['selected_out_weights'].astype(np.float64),optimize=True)
    rounded=np.einsum('brhd,hdc->brhc',bf16(selected).astype(np.float64),a['selected_out_weights'].astype(np.float64),optimize=True)
    s['fixed_projection_vs_FP64_unrounded_components']=metric(exact,a['selected_projection'])
    s['fixed_projection_vs_FP64_native_BF16_operands']=metric(rounded,a['selected_projection'])
    summaries[str(i)]=s
    print(json.dumps({'layer':i,'value_gradient_CPU':s['native_value_gradient_vs_CPU'],'head_reconstruction':s['head_content_reconstruction'],'projected_reconstruction':s['projected_reconstruction']}),flush=True)
observed=sum(x['seconds'] for x in r['calls'])
s={'status':'two_native_FT_content_boundaries_executed_and_CPU_audited_whole_FT_pending',
    'raw_sha256':sha(raw),'protocol_sha256':sha((D/'protocol.json').read_bytes()),'source_sha256':r['protocol']['files_sha256'],
    'previous_failed_start':{'raw_sha256':sha(failed_raw),'error':failed['error'],'GPU_calls':0,'parameter_loads':0,'job_seconds_before_bundle':failed['job_seconds_before_bundle']},
    'target_positions':r['target_positions'],'baseline':r['protocol']['baseline'],'layers':summaries,'budget':r['protocol']['budget'],
    'calls':r['calls'],'job_seconds_before_bundle':r['job_seconds_before_bundle'],'timed_call_seconds':observed,
    'unsegmented_job_seconds':r['job_seconds_before_bundle']-observed,'peak_stage_bytes':max(x['peak_bytes'] for x in r['calls']),
    'source_tree':r['sources_before'],'loaded_parameters':r['loaded_parameters'],'versions':r['versions'],'artifacts':r['artifacts'],
    'CPU_audit_seconds':time.perf_counter()-start,
    'limits':['Boundary screen only: no complete32-layer FT, no0-3-hop aggregate or original quality metrics.',
        'Corrected native-content FT is explicitly different from unchanged e81b3be; retain both identities.',
        'CPU reference is arithmetic validation only; runtime uses actual public FA/FLA/causal-conv kernels and their native backward.',
        'FA CPU reference uses original public GQA LSE whereas auxiliary expands KV; numerical layout differences are reported.',
        'A faithful content decomposition is not proof of attribution usefulness, DeltaTrace victory or conditional deletion signs.',
        'Engineering cold calls include default native compilation and discarded gradients, not matched warm whole-method performance; ZIP/SFTP and prior root costs are separate.']}
(A/'qwen35_FT_native_content_summary_20260908.json').write_text(json.dumps(s,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
print(json.dumps({'status':s['status'],'job_seconds_before_bundle':s['job_seconds_before_bundle'],'timed_seconds':observed,'CPU_audit_seconds':s['CPU_audit_seconds']}))
