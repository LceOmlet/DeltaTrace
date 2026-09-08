"""Actual 34-boundary NI conditional ledger on frozen current-control bins.

One normal symmetric-layer0 DT, four original scorer calls, passive captures.
No new finite rule, extra operator, alternate forward, or metric curve.
"""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
    TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'native_eager_diagnostics':0,'DT_calls_entered':0,'DT_calls':0,
   'scoring_forwards_entered':0,'scoring_forwards_returned':0,'FT_calls':0,'generation_calls':0,
   'extra_operator_calls':0,'calls':[],'points':{}}
started=time.perf_counter();handles=[];capture=None;observer=None;finite_fa=None;finite_fla=None
BOUNDARIES=tuple(str(i) for i in range(33))+('norm',)


def save():
    q=A/'results.partial';q.write_text(json.dumps(r,indent=2));q.replace(A/'results.json')


def sources():
    out={}
    for name,want in p['files_sha256'].items():out[name]=sha((A/name).read_bytes());assert out[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        out['FT/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        out['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes());assert out['native/'+name]==want,name
    return out


def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();item={'kind':kind,'status':'entered'};r['calls'].append(item)
    try:
        out=fn();torch.cuda.synchronize();item['status']='returned';return out
    finally:item['seconds']=time.perf_counter()-tick


def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().to('cpu',copy=True)
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
    return x


def nbytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(nbytes(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(nbytes(v) for v in x)
    return 0


def dot(m,x):
    assert m.shape==x.shape,(m.shape,x.shape)
    return float((m.double()*x.double()).sum())

def token_dot(m,x):
    assert m.shape==x.shape and m.ndim==3
    return (m.double()*x.double()).sum(-1)

def signed_stats(x):
    assert bool(torch.isfinite(x).all())
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum())}


class CountFinite:
    def __init__(self,op):self.op=op;self.entered=0;self.returned=0
    def __call__(self,*args,**kw):
        self.entered+=1;out=self.op(*args,**kw);self.returned+=1;return out


class Observer:
    def __init__(self):self.boundaries={};self.coarse_scoring={};self.current=None
    def boundary(self,name,m,x):
        assert name in BOUNDARIES and name not in self.boundaries
        self.boundaries[name]={'m':cpu(m),'paired':cpu(x)}


class ScoringCapture:
    """Only passive module boundary hooks; no interior operator diagnostics."""
    def __init__(self,model,observer):
        self.observer=observer;self.handles=[]
        layers=model.model.language_model.layers;norm=model.model.language_model.norm
        for name in BOUNDARIES[:-2]:
            def boundary(module,args,kw,name=name):
                x=args[0] if args else kw['hidden_states']
                dst=self.observer.coarse_scoring.setdefault(str(self.observer.current),{})
                assert name not in dst;dst[name]=cpu(x)
            self.handles.append(layers[int(name)].register_forward_pre_hook(boundary,with_kwargs=True))
        def finalnorm(module,args,out):
            dst=self.observer.coarse_scoring.setdefault(str(self.observer.current),{})
            assert '32' not in dst and 'norm' not in dst
            dst['32']=cpu(args[0]);dst['norm']=cpu(out)
        self.handles.append(norm.register_forward_hook(finalnorm))
    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]


try:
    def timeout(*args):raise TimeoutError('Frozen NI34boundary observation budget expired.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import numpy as np
    import torch,flash_attn
    torch.set_num_threads(4)
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from flash_attn import flash_attn_func,flash_attn_varlen_func
    from transformers.integrations.flash_attention import flash_attention_forward
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from flashtrace.improved import keep_token_indices
    from llm_attr_eval import LLMAttributionEvaluator
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    assert p['capture_steps']==[0,1,10,20] and tuple(p['boundaries'])==BOUNDARIES
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'];assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available;verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    data={k:Path(v).read_bytes() for k,v in p['cache_paths'].items()}
    for k,raw in data.items():assert sha(raw)==p['cache_hashes'][k]
    raw=Path(p['reference_results_path']).read_bytes();assert sha(raw)==p['reference_results_sha256'];prior=json.loads(raw)
    assert prior['status']=='all8FA_clean_secant_4DT84score_pilot_complete'
    frozen=prior['cases']['niah_mq_q2_1']['curves']['control']
    historical=prior['runs'][3];assert historical['case']=='niah_mq_q2_1' and historical['method']=='control'
    raw=Path(p['reference_vectors_path']).read_bytes();assert sha(raw)==p['reference_vectors_sha256']
    with np.load(p['reference_vectors_path'],allow_pickle=False) as archive:
        reference=archive['niah_mq_q2_1_control_full'].copy()
        reference_evaluated=archive['niah_mq_q2_1_control_evaluated'].copy()
    def prepare(dataset,index):
        rec=json.loads(data[dataset].decode().splitlines()[index]);tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        keep=keep_token_indices([rec['prompt'][a:b] for a,b in tok['offset_mapping']]);target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long);base=ids.clone();base[keep]=tokenizer.eos_token_id
        inf={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),'target_length':len(target),'total_length':len(ids),'keep':keep}
        return rec,ids,base,torch.tensor(target),inf
    control=prepare('niah_mq_q2',0);rec,ids,base,target,info=prepare('niah_mq_q2',1)
    assert info==prior['cases']['niah_mq_q2_1']['input'];r['input']=info
    assert sha(base.numpy().tobytes())==prior['cases']['niah_mq_q2_1']['baseline_sha256']
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    layers=model.model.language_model.layers;assert len(layers)==32
    for layer in layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule and layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_order_native_eager_B1_diagnostic',lambda:model(input_ids=control[1][None].to('cuda'),
        attention_mask=torch.ones_like(control[1][None],device='cuda'),use_cache=False))
    r['native_eager_diagnostics']=1;del warm,control;model.set_attn_implementation('flash_attention_2')
    finite_fa=CountFinite(VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']))
    finite_fla=CountFinite(make_compiled_finite_pullback(reuse_scalar_products=False))
    runner=Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,norm_gate_rules={0:'symmetric'});observer=Observer()
    pairs=torch.stack((base,ids)).to('cuda');mask=torch.ones_like(pairs)
    selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':info['prompt_length']}],[list(range(len(target)))],len(ids),'cuda')
    r['status']='one_normal_current_DT_with_34_passive_boundaries';r['DT_calls_entered']+=1;save()
    signed,details=timed('actual_DT_with_passive_boundary_diagnostics',lambda:runner.attribute(pairs,mask,selection,select_output_rows=True,observer=observer))
    r['DT_calls']+=1;r['DT_details']=details;assert details['norm_gate_rules']=={'0':'symmetric'}
    assert finite_fa.entered==finite_fa.returned==8 and finite_fla.entered==finite_fla.returned==24
    kinds=[q['kind'] for q in details['calls']]
    assert sum(k.startswith('native_replay_') for k in kinds)==32 and sum(k.startswith('finite_decoder_') for k in kinds)==32
    assert sum(k.startswith('public_FA_LSE_') for k in kinds)==8
    assert set(observer.boundaries)==set(BOUNDARIES)
    signed=signed[0];evaluated=signed[:info['prompt_length']].float();assert bool(torch.isfinite(signed).all())
    r['reference_vector_drift_report_only']={'relative_L2':float((signed-torch.from_numpy(reference)).norm()/torch.from_numpy(reference).norm()),
        'max_absolute':float((signed-torch.from_numpy(reference)).abs().max()),'bitwise_equal':bool(torch.equal(signed,torch.from_numpy(reference))),
        'policy':'No historical2percent admission guard; current actual coefficients explain frozen most-recent current-control deletion sets.'}
    vectors={'current_full':signed.numpy(),'current_evaluated':evaluated.numpy()}
    r['evaluated_vector_drift_report_only']={'relative_L2':float((evaluated.double()-torch.from_numpy(reference_evaluated).double()).norm()/torch.from_numpy(reference_evaluated).double().norm()),
        'max_absolute':float((evaluated.double()-torch.from_numpy(reference_evaluated).double()).abs().max())}
    r['historical_boundary_coefficient_drift']={'available':False,'reason':'The frozen wholepilot contains input attribution vectors but no34boundary coefficient tensors; do not substitute older coefficients.'}
    r['B2_boundary_contractions']={name:dot(value['m'],value['paired'][1::2].double()-value['paired'][0::2].double()) for name,value in observer.boundaries.items()}
    r['B2_actual_root_effect']=details['root_effect'];r['prior_current_root_effect']=historical['details']['root_effect']
    r['root_effect_minus_reference']=details['root_effect']-historical['details']['root_effect']
    r['seed_effect_minus_reference']=details['seed_effect']-historical['details']['seed_effect']
    r['boundary_coefficient_metadata']={name:{'shape':list(x['m'].shape),'dtype':str(x['m'].dtype),'bytes':nbytes(x['m']),
        'sha256':sha(x['m'].contiguous().view(torch.uint8).numpy().tobytes())} for name,x in observer.boundaries.items()}
    np.savez_compressed(A/'vectors.npz',**vectors)
    del pairs,mask,selection;gc.collect();evaluator=LLMAttributionEvaluator(model,tokenizer);capture=ScoringCapture(model,observer)
    def before_score(module,args,kw):
        assert r['scoring_forwards_entered']<4;r['scoring_forwards_entered']+=1
        x=kw['input_ids'].detach().cpu();receipt=frozen['input_receipts'][observer.current]
        assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
        assert sha(x.numpy().tobytes())==receipt['input_sha256'] and (x[0]!=ids).nonzero().flatten().tolist()==receipt['deleted_positions']
    handles=[model.register_forward_pre_hook(before_score,with_kwargs=True)]
    for step in p['capture_steps']:
        observer.current=step;receipt=frozen['input_receipts'][step];prompt=ids[:info['prompt_length']].clone();prompt[receipt['deleted_positions']]=tokenizer.eos_token_id
        point={'input_receipt':receipt};r['points'][str(step)]=point;r['status']='four_frozen_original_scores_step'+str(step);save()
        with torch.no_grad():lp=timed('original_frozen_scorer_'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(prompt[None].to('cuda'),target[None].to('cuda')))
        r['scoring_forwards_returned']+=1;point['actual_native_score']=float(lp.sum().cpu());point['actual_native_logprob_dtype']=str(lp.dtype)
        point['prior_actual_native_score']=frozen['scores'][step];point['native_score_minus_prior']=point['actual_native_score']-point['prior_actual_native_score']
        assert set(observer.coarse_scoring[str(step)])==set(BOUNDARIES)
        if step:
            effect=r['points']['0']['actual_native_score']-point['actual_native_score'];deleted=receipt['deleted_positions']
            pred= float(evaluated[deleted].double().sum());point['complete_input']={'actual_effect':effect,'evaluated_prediction':pred,
                'full_signed_prediction':float(signed[deleted].sum()),'prediction_minus_actual':pred-effect}
            boundary_vectors={name:token_dot(observer.boundaries[name]['m'],observer.coarse_scoring['0'][name].double()-observer.coarse_scoring[str(step)][name].double()) for name in BOUNDARIES}
            contractions={name:float(x.sum()) for name,x in boundary_vectors.items()}
            terms={left+'_to_'+right:boundary_vectors[left]-boundary_vectors[right] for left,right in zip(BOUNDARIES,BOUNDARIES[1:])}
            regions={name:float(x.sum()) for name,x in terms.items()}
            regions['norm_to_actual_score']=contractions['norm']-effect
            regions['evaluated_input_map']=pred-contractions['0']
            point['coarse']={'sign_convention':'prediction_minus_actual','boundary_contractions':contractions,'regions':regions,
                'prediction_minus_actual':pred-effect,'telescoping_error':sum(regions.values())-(pred-effect),
                'head_scope':'norm_to_actual_score includes output head/seed,DT FP32 versus native BF16 scoring and B1/B2 convention differences; no pure operator causality claimed.'}
            assert abs(point['coarse']['telescoping_error'])<1e-7
            keep=set(info['keep']);deleted_set=set(deleted);P=info['prompt_length'];L=info['total_length']
            groups={'deleted':sorted(deleted_set),'kept':sorted(keep-deleted_set),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,L))}
            assert sum(map(len,groups.values()))==L
            point['coordinate_signed_groups']={}
            for name,x in {**{'boundary_'+k:v for k,v in boundary_vectors.items()},**terms}.items():
                vectors[str(step)+'_'+name]=x.numpy()
                group_stats=signed_stats(x)
                grouped={g:{'count':len(ix),**signed_stats(x[:,ix])} for g,ix in groups.items()}
                assert abs(sum(v['net'] for v in grouped.values())-group_stats['net'])<1e-7
                point['coordinate_signed_groups'][name]={'total':group_stats,'groups':grouped}
            r['group_scope']='Hidden-coordinate/token contraction groups are allocations in actual captured coordinates,not independent original-token causal effects; positive/negative terms may cancel across boundaries.'
        else:
            point['B1_clean_vs_B2_input_boundary_relative_L2']={name:float((observer.coarse_scoring['0'][name].double()-observer.boundaries[name]['paired'][1::2].double()).norm()/observer.boundaries[name]['paired'][1::2].double().norm().clamp_min(1e-30)) for name in BOUNDARIES}
        if step==20:
            point['B1_allEOS_vs_B2_EOS_boundary_relative_L2']={name:float((observer.coarse_scoring['20'][name].double()-observer.boundaries[name]['paired'][0::2].double()).norm()/observer.boundaries[name]['paired'][0::2].double().norm().clamp_min(1e-30)) for name in BOUNDARIES}
        np.savez_compressed(A/'vectors.npz',**vectors)
        del lp;save()
    capture.close();capture=None
    for h in handles:h.remove()
    handles=[];assert r['DT_calls']==1 and r['scoring_forwards_entered']==r['scoring_forwards_returned']==4
    r['sources_after']=sources();r['weight_stats_after']=stats();assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'] and sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['status']='NI_current_34boundaries_1DT4score_observation_complete'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    if capture is not None:capture.close()
    sys.setprofile(None);signal.alarm(0)
    for h in handles:h.remove()
    r['finite_callback_counts']={name:{'entered':op.entered if op is not None else 0,'returned':op.returned if op is not None else 0} for name,op in [('finite_FA',finite_fa),('finite_FLA',finite_fla)]}
    r['partial_native_count_policy']='Unreturned DT has no completed replay ledger; partial native work is unknown,not zero. The entered/returned attribution callbacks and scorer counts remain recorded.'
    if observer is not None:
        artifact=A/'NI1_current_34boundaries_actual_private.pt'
        try:
            stored={'boundaries':observer.boundaries,'scoring':observer.coarse_scoring,'input':r.get('input'),'points':r['points'],'schema':'34actual production coefficients and paired B2 inputs plus4original B1 hidden inputs;no model weights'}
            torch.save(stored,artifact);digest=hashlib.sha256()
            with artifact.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
            r['private_artifact']={'file':artifact.name,'sha256':digest.hexdigest(),'bytes':artifact.stat().st_size,'tensor_CPU_bytes':nbytes(stored)}
        except Exception:r['private_artifact_error']=traceback.format_exc();r['status']='failed'
    r['job_seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_calls':r['DT_calls'],'scoring_forwards':r['scoring_forwards_returned'],'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)
