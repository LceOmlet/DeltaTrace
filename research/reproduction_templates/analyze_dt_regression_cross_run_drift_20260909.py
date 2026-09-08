"""Retained-evidence CPU comparison; no model or framework execution."""
import ast,hashlib,json,math
from pathlib import Path
import numpy as np

A=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
DIRS={'s0':'codex_dt_fixed_eight_case_regression_20260909_s0_v1',
      'standalone':'codex_dt_MH0_pristine_FT0_comparison_20260909_v1',
      'stability':'codex_dt_layer0_fla_endpoint_average_stability_cost_20260909_v1'}
DATA={k:{'dir':A/'snapshot/tmp'/v} for k,v in DIRS.items()}
for d in DATA.values():
    d.update(r=read(d['dir']/'results.json'),p=read(d['dir']/'protocol.json'),z=np.load(d['dir']/'vectors.npz'))

def difference(old,new):
    old=np.asarray(old,dtype=np.float64);new=np.asarray(new,dtype=np.float64)
    assert old.shape==new.shape and np.isfinite(old).all() and np.isfinite(new).all()
    delta=new-old;den=np.linalg.norm(old.ravel())
    return {'shape':list(old.shape),'exact_equal':bool(np.array_equal(old,new)),
        'relative_L2_to_old':float(np.linalg.norm(delta.ravel())/den) if den else None,
        'max_absolute':float(np.max(np.abs(delta),initial=0)),
        'changed_entries':int(np.count_nonzero(delta)),
        'old_sum':float(old.sum()),'new_sum':float(new.sum()),'sum_delta':float(delta.sum()),
        'old_positive':float(old[old>0].sum()),'new_positive':float(new[new>0].sum()),
        'old_negative':float(old[old<0].sum()),'new_negative':float(new[new<0].sum())}

def source_comparison(oldname):
    op=DATA[oldname]['p'];np_=DATA['s0']['p']
    fields=('checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256',
        'installed_FA_interface_sha256','native_stage_source_sha256','runtime_source_sha256',
        'dependency_overlays','expected_weight_stats','cache_paths','cache_hashes',
        'finite_FA_library','finite_FA_library_sha256','compiler_cache','boundary_compiler_cache',
        'official_package_blob_sha1','FT_commit')
    out={key:op.get(key)==np_.get(key) for key in fields if key in op and key in np_}
    oldf=op['files_sha256'];newf=np_['files_sha256']
    out['nonstudy_source_identity']={k:oldf[k]==newf[k] for k in oldf.keys()&newf.keys() if k!='study.py'}
    assert all(v for k,v in out.items() if k!='nonstudy_source_identity')
    assert all(out['nonstudy_source_identity'].values())
    out['study_different']=oldf['study.py']!=newf['study.py']
    return out

def dt_run(d,key,method):
    return next(x for x in d['r']['runs'] if x['case']==key and x['method']==method)

def compare(oldname,key,oldmethod,newmethod):
    old=DATA[oldname];new=DATA['s0'];oc=old['r']['cases'][key];nc=new['r']['cases'][key]
    oi=old['r']['input_freeze_before_model_load'][key];ni=new['r']['input_freeze_before_model_load'][key]
    identity={k:oi[k]==ni[k] for k in ('input','input_ids','target_ids','baseline_sha256')}
    identity['gold']=oc['gold']==nc['gold']
    identity['mapping']={k:oc['mapping'][k]==nc['mapping'][k] for k in ('source_record_sha256','original_sink_span','new_sink_span','new_thinking_span','gold')}
    assert all(v for k,v in identity.items() if k!='mapping') and all(identity['mapping'].values())
    out={'old_run':oldname,'case':key,'old_method':oldmethod,'new_method':newmethod,'identity':identity,'vectors':{}}
    for suffix in ('full','evaluated'):
        out['vectors'][suffix]=difference(old['z'][key+'_'+oldmethod+'_'+suffix],new['z'][key+'_'+newmethod+'_'+suffix])
    if oldmethod in oc['curves'] and newmethod in nc['curves']:
        ov=oc['curves'][oldmethod];nv=nc['curves'][newmethod]
        out['metric_comparison']={'old':ov['return_metrics'],'new':nv['return_metrics'],
            'new_minus_old':[b-a for a,b in zip(ov['return_metrics'],nv['return_metrics'])],
            'old_needle':ov['needle'],'new_needle':nv['needle'],
            'density_drift':difference(ov['density'],nv['density'])}
        n=math.ceil(.1*len(oc['input']['keep']));oset=set(ov['sorted_keep'][:n]);nset=set(nv['sorted_keep'][:n])
        out['top10percent']={'k':n,'intersection':len(oset&nset),'old_only':sorted(oset-nset),'new_only':sorted(nset-oset),
            'note':'Order was the actual scorer order, not the author torch.topk tie-break. Needle audit separately bounds all tie choices.'}
        gold=set(oc['gold'])&set(oc['input']['keep'])
        out['top10percent']['old_only_gold']=sorted((oset-nset)&gold)
        out['top10percent']['new_only_gold']=sorted((nset-oset)&gold)
        oldbyhash={rr['input_sha256']:(i,ov['scores'][i]) for i,rr in enumerate(ov['input_receipts'])}
        matching=[];nonmatching=[]
        for i,rr in enumerate(nv['input_receipts']):
            if rr['input_sha256'] in oldbyhash:
                j,value=oldbyhash[rr['input_sha256']]
                matching.append({'old_step':j,'new_step':i,'old_score':value,'new_score':nv['scores'][i],'new_minus_old':nv['scores'][i]-value,'input_sha256':rr['input_sha256']})
            else:nonmatching.append(i)
        out['actual_scorer_same_input_only']={'matches':matching,'new_steps_without_same_old_input':nonmatching,
            'note':'Unmatched curve points use different deletion sets and are not same-input numerical drift.'}
    if not oldmethod.startswith('FT'):
        od=dt_run(old,key,oldmethod)['details'];nd=dt_run(new,key,newmethod)['details']
        out['DT_actual_B2_root_logp']={k:difference(od[k],nd[k]) for k in ('target_logp0','target_logp1')}
        out['DT_root_and_seed']={k:{'old':od[k],'new':nd[k],'new_minus_old':nd[k]-od[k]} for k in ('root_effect','seed_effect','signed_sum','compiled_seed_logprob_effect','compiled_seed_logprob_effect_minus_root')}
        out['same_rules_and_head']={k:od[k]==nd[k] for k in ('norm_gate_rules','finite_fla_by_layer','select_output_rows','selected_predictor_rows','actual_head_input_shapes')}
        assert all(out['same_rules_and_head'].values())
        layers={}
        for layer in range(32):
            o=od['layers'][str(layer)];n=nd['layers'][str(layer)]
            layers[str(layer)]={k:{'old':o[k],'new':n[k],'new_minus_old':n[k]-o[k]} for k in ('root_output_effect','replay_output_effect','replay_relative_L2','input_effect')}
        out['layer_scalar_ledgers']=layers
        out['layer_ledger_limit']='Contractions depend on each run upstream coefficients and endpoint differences. Scalars cannot locate the first hidden-state drift or identify a faulty operator.'
    return out

out={'status':'retained_evidence_comparison_complete','sources':{},'comparisons':[],
    'scope':'CPU retained arrays and source receipts only; no new native/model/FT/scorer calls. Drift is measured, not assumed acceptable BF16 or proven harmless.'}
for name,d in DATA.items():out['sources'][name]={k:sha(d['dir']/k) for k in ('results.json','protocol.json','vectors.npz','terminal_receipt.json')}
out['source_identity']={k:source_comparison(k) for k in ('standalone','stability')}
for oldname,key,om,nm in [('standalone','morehopqa_0','DT','DT_candidate'),
    ('standalone','morehopqa_0','FT0','FT0'),('standalone','morehopqa_0','FT3','FT3'),
    ('stability','morehopqa_0','control','DT_control'),('stability','morehopqa_0','candidate','DT_candidate'),
    ('stability','niah_mq_q2_0','control','DT_control'),('stability','niah_mq_q2_0','candidate','DT_candidate')]:
    out['comparisons'].append(compare(oldname,key,om,nm))
out['within_s0_C_S_root_equality']={}
for key in ('niah_mq_q2_0','morehopqa_0'):
    c=dt_run(DATA['s0'],key,'DT_control')['details'];s=dt_run(DATA['s0'],key,'DT_candidate')['details']
    out['within_s0_C_S_root_equality'][key]={k:difference(c[k],s[k]) for k in ('target_logp0','target_logp1')}
old=DATA['stability'];other=DATA['standalone'];key='morehopqa_0'
out['prior_stability_vs_standalone_candidate']={
    'full_vector':difference(old['z'][key+'_candidate_full'],other['z'][key+'_DT_full']),
    'root_logp':{k:difference(dt_run(old,key,'candidate')['details'][k],dt_run(other,key,'DT')['details'][k]) for k in ('target_logp0','target_logp1')}}
out['static_initialization_calls']={}
for name,d in DATA.items():
    src=(d['dir']/'study.py').read_text();tree=ast.parse(src)
    relevant=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            f=ast.unparse(node.func)
            if f in ('torch.manual_seed','Qwen3_5ForConditionalGeneration.from_pretrained','model.set_attn_implementation','model.eval'):
                relevant.append({'line':node.lineno,'expression':ast.unparse(node)})
    out['static_initialization_calls'][name]={'study_sha256':sha(d['dir']/'study.py'),'calls':sorted(relevant,key=lambda x:x['line'])}
out['findings']=[
    'Actual IDs, targets, eligible sets, remapped gold, cache records and all common non-study sources agree. Weight identity evidence is path/size/mtime plus configuration and loading receipts, not full safetensor byte hashes.',
    'Both C/S roots are identical within s0; cross-process drift is present in native B2 target logprobs before the finite rule is applied. FT vectors and some exact same B1 deletion-input scores also drift.',
    'All retained per-layer within-run replay_relative_L2 values are zero; this validates replay of that run, not equality across processes. Layer scalar contractions cannot locate the first changing hidden tensor.',
    'The scripts use the same seed73, original full-model BF16/eager load, empty loading-info assertion, eval/frozen weights, NI0 eager initialization and official FA backend setter. Normal DT roots explicitly pass use_cache=False. No explicit stale-cache reuse or omitted initialization is visible in these sources.',
    'Operation order and prior shapes/warmups differ. The retained data do not establish that these differences caused the drift, identify nondeterminism versus kernel/initialization issues, or prove the measured drift acceptable.',
    'Use same-run paired quality comparisons. Do not replace scorer points from historical curves or promote across-run changes into method gains.'
]
out['execution_context']={}
for name,d in DATA.items():
    out['execution_context'][name]={'DT_order':[{'case':x['case'],'method':x['method'],'phase':x.get('phase')} for x in d['r']['runs']],
        'calls':d['r']['calls'],'backend_schedule':d['p'].get('backend_schedule'),
        'note':'All study sources use the original NI0 eager initialization before FA DT. Different order/warmup/compiled shapes are execution-context differences, not proven causes. Native caches and hidden tensors were not retained by these normal runs.'}
out['analyzer_sha256']=sha(Path(__file__))
target=A/'dt_fixed_regression_s0_cross_run_drift_20260909.json'
target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'path':str(target),'sha256':sha(target),'comparisons':[{
    'old':x['old_run'],'case':x['case'],'method':x['new_method'],
    'eval':x['vectors']['evaluated'],'metric':x.get('metric_comparison'),
    'root':x.get('DT_actual_B2_root_logp'),'sameinput':x.get('actual_scorer_same_input_only')
    } for x in out['comparisons']]},allow_nan=False))
