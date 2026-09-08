"""Four complete current-DT attributions, full/selected/selected/full head rows.

Only the real model's public logits_to_keep argument changes. Original finite
rules, model attention, vocabulary, targets and EOS input baseline are fixed.
"""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'DT_calls':0,'FT_calls':0,'generation_calls':0,
    'scoring_forwards_entered':0,'scoring_forwards_completed':0,'runs':{},'curves':{},'calls':[]}
start=time.perf_counter();handles=[];vectors={}

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
    torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':kind,'seconds':time.perf_counter()-t});return value

def timeout(*args):raise TimeoutError('Frozen four-attribution native-row comparison budget expired.')

try:
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import numpy as np
    import torch,flash_attn
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens,faithfulness_test_skip_tokens
    from llm_attr_eval import LLMAttributionEvaluator
    from fixed_input_metric_view import FixedInputMetricView
    from native_target_logit_rows import NativeTargetLogitRows
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    # Small CPU indexing test only: duplicate predictor positions across samples
    # must gather the correct endpoint/sample rows. This is not a model benchmark.
    tiny=PackedAnswerTargets([{'target_ids':torch.arange(4),'prompt_length':3},{'target_ids':torch.arange(5),'prompt_length':5}],
        [[0,2],[0,4]],10,'cpu');sel=NativeTargetLogitRows(tiny)
    sentinel=torch.arange(4*10*7).reshape(4,10,7)
    assert torch.equal(sel.pack_logits(sentinel[:,sel.rows,:]),tiny.pack_hidden(sentinel))
    r['CPU_indexing_check']={'two_samples':True,'repeated_predictor_positions':True,'passed':True,'model_calls':0}
    del tiny,sel,sentinel
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
    raw=Path(p['cache_path']).read_bytes();assert sha(raw)==p['cache_sha256'];record=json.loads(raw.decode().splitlines()[0])
    raw_contract=Path(p['input_contract_path']).read_bytes();assert sha(raw_contract)==p['input_contract_sha256'];contract=json.loads(raw_contract)
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    prompt=tokenizer(record['prompt'],add_special_tokens=False,return_offsets_mapping=True)
    labels=[record['prompt'][a:b] for a,b in prompt['offset_mapping']];keep=keep_token_indices(labels)
    assert keep==contract['current_keep'] and len(keep)==310
    target=tokenizer(record['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
    ids=torch.tensor(prompt['input_ids']+target,dtype=torch.long);assert len(ids)==588 and len(target)==248
    assert sha(ids.numpy().tobytes())==p['input_sha256']
    baseline=ids.clone();baseline[keep]=tokenizer.eos_token_id;assert (baseline!=ids).nonzero().flatten().tolist()==keep
    r['input']={'prompt_length':340,'target_length':248,'total_length':588,'keep':keep,'gold':contract['current_gold'],
        'clean_sha256':sha(ids.numpy().tobytes()),'baseline_sha256':sha(baseline.numpy().tobytes()),'sample_batch':1,'endpoint_batch':2}
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading_actual_model';save()
    model,info=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    model.eval().requires_grad_(False);r['model_loads']=1;assert not any(info.values())
    for layer in model.model.language_model.layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    clean=ids[None].to('cuda')
    with torch.no_grad():warm=timed('original_order_native_eager_B1_diagnostic',lambda:model(input_ids=clean,attention_mask=torch.ones_like(clean),use_cache=False))
    r['eager_diagnostic_shape']=list(warm.logits.shape);del warm,clean
    model.set_attn_implementation('flash_attention_2')
    pairs=torch.stack((baseline,ids)).to('cuda');mask=torch.ones_like(pairs)
    selection=PackedAnswerTargets([{'target_ids':torch.tensor(target),'prompt_length':340}],[list(range(248))],588,'cuda')
    runner=Qwen35DenseFiniteRunner(model,VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']),
        make_compiled_finite_pullback(reuse_scalar_products=False))
    for name,selected in p['run_order']:
        r['status']='attribution_'+name;save()
        wall=time.perf_counter();gc.collect();torch.cuda.synchronize();housekeeping=time.perf_counter()-wall
        before=torch.cuda.memory_allocated()
        signed,detail=runner.attribute(pairs,mask,selection,select_output_rows=selected);r['DT_calls']+=1
        assert signed.shape==(1,588) and torch.isfinite(signed).all()
        assert all(float(signed[0,j])==0 for j in range(588) if j not in set(keep))
        vectors[name]=signed[0].numpy();io_start=time.perf_counter();np.savez_compressed(A/'vectors.npz',**vectors)
        archive_seconds=time.perf_counter()-io_start;needle_start=time.perf_counter()
        needle=float(evaluate_attr_recovery_skip_tokens(signed[:,:340].float(),keep_prompt_token_indices=keep,
            gold_prompt_token_indices=contract['current_gold'],top_fraction=0.1))
        needle_seconds=time.perf_counter()-needle_start
        detail.update(housekeeping_seconds=housekeeping,allocated_before=before,
            cumulative_vector_archive_seconds=archive_seconds,CPU_needle_seconds=needle_seconds,
            caller_seconds_including_housekeeping_archive_and_needle=time.perf_counter()-wall,needle=needle)
        r['runs'][name]=detail;save();del signed
    evaluator=LLMAttributionEvaluator(model,tokenizer);view=FixedInputMetricView(evaluator,record['prompt'])
    assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
    for name,_ in p['run_order']:
        r['status']='original_metrics_'+name;row={'input_receipts':[]};r['curves'][name]=row;save()
        def pre(_module,args,kw):
            x=kw['input_ids'].detach().cpu();assert x.shape==(1,588) and bool(kw['attention_mask'].eq(1).all())
            changed=(x[0]!=ids).nonzero().flatten().tolist()
            assert set(changed)<=set(keep) and all(int(x[0,j])==tokenizer.eos_token_id for j in changed)
            if not row['input_receipts']:assert not changed
            row['input_receipts'].append({'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed});r['scoring_forwards_entered']+=1
        def post(*args):r['scoring_forwards_completed']+=1
        def observe(frame,event,arg):
            if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                values=frame.f_locals
                for key in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:row[key]=np.asarray(values[key]).copy().tolist()
                row['sorted_keep']=[int(j) for j in values['sorted_keep']];row['attr_sum']=float(values['attr_sum']);row['return_metrics']=[float(v) for v in arg]
        handles=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post)];sys.setprofile(observe)
        try:
            with torch.no_grad():metrics=timed('original_20_step_curve_'+name,lambda:faithfulness_test_skip_tokens(view,
                torch.as_tensor(vectors[name][:340],dtype=torch.float32)[None],record['prompt'],record['target'],
                keep_prompt_token_indices=keep,user_prompt_indices=list(range(340)),k=20))
        finally:
            sys.setprofile(None)
            for h in handles:h.remove()
            handles=[]
        assert row['return_metrics']==[float(v) for v in metrics] and len(row['input_receipts'])==21
        assert row['input_receipts'][-1]['deleted_positions']==keep
        assert all(np.isfinite(row[k]).all() for k in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']);save()
    assert r['DT_calls']==4 and r['scoring_forwards_entered']==r['scoring_forwards_completed']==84
    r['sources_after']=sources();r['weight_stats_after']=stats()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='native_logit_rows_complete_ABBA_NI0'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0)
    for h in handles:h.remove()
    r['job_seconds']=time.perf_counter()-start
    if (A/'vectors.npz').exists():r['vectors_sha256']=sha((A/'vectors.npz').read_bytes())
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_calls':r['DT_calls'],'score_forwards':r['scoring_forwards_completed'],
        'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)
