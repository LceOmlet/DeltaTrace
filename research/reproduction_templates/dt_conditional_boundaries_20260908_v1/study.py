"""Observe actual coefficients and original scoring boundaries; no method change."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'DT_calls':0,'FT_calls':0,'generation_calls':0,
   'scoring_forwards':0,'cases':{},'calls':[]};started=time.perf_counter();handles=[];vectors={}
def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(A/'results.json')
def sources():
    out={}
    for name,want in p['files_sha256'].items():out[name]=sha((A/name).read_bytes());assert out[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        out['FT/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        out['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes());assert out['native/'+name]==want,name
    return out
def timed(kind,fn):
    torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':kind,'seconds':time.perf_counter()-t});return out
def timeout(*args):raise TimeoutError('Frozen conditional-boundary budget expired.')

class BoundaryObserver:
    def __init__(self):self.coeff={};self.endpoint_effect={};self.clean={};self.points={};self.current=-1;self.logp={}
    def boundary(self,name,m,x):
        self.coeff[name]=m.detach().to('cpu',copy=True)
        self.endpoint_effect[name]=float((self.coeff[name].double()*(x[1::2].double()-x[0::2].double())).sum())
    def activation(self,name,x):
        if self.current not in p['capture_steps']:return
        value=x.detach().cpu()
        if self.current==0:self.clean[name]=value.clone()
        else:
            delta=self.clean[name].double()-value.double()
            self.points.setdefault(str(self.current),{})[name]=float((self.coeff[name].double()*delta).sum())
    def original_logits(self,logits,selection):
        if self.current not in p['capture_steps']:return
        # FP32 diagnostic of this SAME real B1 output; original scorer is unchanged.
        z=logits[selection.samples,selection.positions].float()
        lp=z.log_softmax(-1)
        self.logp[str(self.current)]=float(lp.gather(-1,selection.labels[:,None]).double().sum())

try:
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import numpy as np
    import torch,flash_attn
    torch.set_num_threads(4)
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens,faithfulness_test_skip_tokens
    from llm_attr_eval import LLMAttributionEvaluator
    from fixed_input_metric_view import FixedInputMetricView
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    prior_raw=Path(p['parent_results_path']).read_bytes();assert sha(prior_raw)==p['parent_results_sha256'];prior=json.loads(prior_raw)
    data={k:Path(v).read_bytes() for k,v in p['cache_paths'].items()}
    for k,raw in data.items():assert sha(raw)==p['cache_hashes'][k]
    def prepare(dataset,index):
        rec=json.loads(data[dataset].decode().splitlines()[index]);tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        keep=keep_token_indices([rec['prompt'][a:b] for a,b in tok['offset_mapping']])
        target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long);base=ids.clone();base[keep]=tokenizer.eos_token_id
        key=f'{dataset}_{index}';info={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),
            'target_length':len(target),'total_length':len(ids),'keep':keep}
        if key in p['cases']:
            expected=prior['cases'][key]['input']
            assert all(info[k]==expected[k] for k in info),key
        return key,rec,ids,base,torch.tensor(target),info
    control=prepare('niah_mq_q2',0);cases=[prepare(*x) for x in p['case_indices']]
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    for layer in model.model.language_model.layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_order_native_eager_B1_diagnostic',lambda:model(input_ids=control[2][None].to('cuda'),
        attention_mask=torch.ones_like(control[2][None],device='cuda'),use_cache=False))
    del warm,control;model.set_attn_implementation('flash_attention_2')
    runner=Qwen35DenseFiniteRunner(model,VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']),
        make_compiled_finite_pullback(reuse_scalar_products=False))
    evaluator=LLMAttributionEvaluator(model,tokenizer)
    for case_index,(key,rec,ids,base,target,info) in enumerate(cases):
        row={'input':info,'curve':{'input_receipts':[]}};r['cases'][key]=row
        pairs=torch.stack((base,ids)).to('cuda');mask=torch.ones_like(pairs)
        selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':info['prompt_length']}],
            [list(range(len(target)))],len(ids),'cuda')
        if case_index==0:
            r['status']='uninstrumented_control_'+key;save()
            reference,details=runner.attribute(pairs,mask,selection,select_output_rows=True);r['DT_calls']+=1
            row['uninstrumented_control']=details;vectors[key+'_control']=reference[0].numpy()
        r['status']='observe_actual_coefficients_'+key;save();observer=BoundaryObserver()
        signed,details=runner.attribute(pairs,mask,selection,select_output_rows=True,observer=observer);r['DT_calls']+=1
        signed=signed[0];vectors[key+'_DT_full']=signed.numpy();score=signed[:info['prompt_length']].float()
        vectors[key+'_DT_evaluated']=score.numpy();np.savez_compressed(A/'vectors.npz',**vectors)
        row['DT']=details;row['B2_endpoint_boundary_effects']=observer.endpoint_effect
        row['coefficient_CPU_bytes']=sum(x.numel()*x.element_size() for x in observer.coeff.values())
        if case_index==0:
            relative=float((signed-reference[0]).norm()/reference[0].norm())
            row['passive_observer_vector_relative_L2']=relative;row['passive_observer_vectors_equal']=bool(torch.equal(signed,reference[0]));save()
            assert relative<=p['observer_vector_relative_L2_ceiling'];del reference
        assert set(observer.coeff)==set([str(i) for i in range(33)]+['norm'])
        del pairs,mask
        curve=row['curve'];view=FixedInputMetricView(evaluator,rec['prompt'])
        def before(_module,args,kw):
            observer.current+=1;r['scoring_forwards']+=1
            x=kw['input_ids'].detach().cpu();assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
            changed=(x[0]!=ids).nonzero().flatten().tolist();assert set(changed)<=set(info['keep'])
            assert all(int(x[0,j])==tokenizer.eos_token_id for j in changed)
            curve['input_receipts'].append({'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed})
        def after(_module,args,output):observer.original_logits(output.logits,selection)
        handles=[model.register_forward_pre_hook(before,with_kwargs=True),model.register_forward_hook(after)]
        for i,layer in enumerate(model.model.language_model.layers):
            def activation_pre(_module,args,kw,i=i):observer.activation(str(i),args[0] if args else kw['hidden_states'])
            handles.append(layer.register_forward_pre_hook(activation_pre,with_kwargs=True))
        def norm_hook(_module,args,output):
            observer.activation('32',args[0]);observer.activation('norm',output)
        handles.append(model.model.language_model.norm.register_forward_hook(norm_hook))
        def observe(frame,event,arg):
            if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                v=frame.f_locals
                for k in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:curve[k]=np.asarray(v[k]).copy().tolist()
                curve['return_metrics']=[float(x) for x in arg];curve['attr_sum']=float(v['attr_sum'])
                curve['sorted_keep']=[int(x) for x in v['sorted_keep']]
        r['status']='original_curve_with_passive_boundaries_'+key;save();sys.setprofile(observe)
        try:
            with torch.no_grad():timed('original_curve_with_diagnostics_'+key,lambda:faithfulness_test_skip_tokens(view,score[None],rec['prompt'],rec['target'],
                keep_prompt_token_indices=info['keep'],user_prompt_indices=list(range(info['prompt_length'])),k=20))
        finally:
            sys.setprofile(None)
            for h in handles:h.remove()
            handles=[]
        assert observer.current==20 and curve['input_receipts'][-1]['deleted_positions']==info['keep']
        row['B1_FP32_target_logprob']=observer.logp;row['B1_boundary_contractions']=observer.points;row['decomposition']={}
        for step in p['capture_steps'][1:]:
            a=observer.points[str(step)];deleted=curve['input_receipts'][step]['deleted_positions'];pred=float(score[deleted].double().sum())
            actual=curve['scores'][0]-curve['scores'][step];actual32=observer.logp['0']-observer.logp[str(step)]
            e={str(i):a[str(i+1)]-a[str(i)] for i in range(32)}
            terms={'scoring_precision':actual-actual32,'head_and_logprob_seed':actual32-a['norm'],
                'final_norm':a['norm']-a['32'],'decoders':sum(e.values()),'input_map':a['0']-pred}
            row['decomposition'][str(step)]={'actual_native_effect':actual,'actual_FP32_effect':actual32,'DT_predicted_effect':pred,
                'actual_minus_predicted':actual-pred,'terms':terms,'decoder_errors':e,
                'FA_decoder_sum':sum(e[str(i)] for i,l in enumerate(model.model.language_model.layers) if l.block_type=='full_attention'),
                'GDN_decoder_sum':sum(e[str(i)] for i,l in enumerate(model.model.language_model.layers) if l.block_type=='linear_attention'),
                'telescoping_error':sum(terms.values())-(actual-pred)}
            assert abs(row['decomposition'][str(step)]['telescoping_error'])<1e-7
        save();del observer,signed,score,selection;gc.collect()
    assert r['DT_calls']==3 and r['scoring_forwards']==42
    r['sources_after']=sources();r['weight_stats_after']=stats()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='conditional_boundaries_complete'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0)
    for h in handles:h.remove()
    r['job_seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_calls':r['DT_calls'],'scoring_forwards':r['scoring_forwards'],
        'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)
