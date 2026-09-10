"""Check every frozen NI/MH development example with a total length <=1024.

One complete retained/local API call per case. These are compatibility calls,
including shape compilation, not extra warm latency or quality measurements.
"""
import argparse,hashlib,inspect,json,os,sys,time,traceback
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();A=args.root;B=Path('/tmp/codex_short_b1_efficiency_20260910_v1')
    R=A/'production_release';plan=json.loads((A/'author_short_plan.json').read_bytes())
    sha=lambda b:hashlib.sha256(b).hexdigest()
    env=json.loads((R/'environment.json').read_bytes())['qwen35']
    out=A/'qwen35_author_short';out.mkdir(exist_ok=False)
    report={'status':'imports','driver_sha256':sha(Path(__file__).read_bytes()),
      'plan_sha256':sha((A/'author_short_plan.json').read_bytes()),'cases':[],
      'generation_calls':0,'FT_calls':0,'metric_calls':0,'sample_batch':1,
      'scope':'Compatibility only. Complete formal attribute_batch B1, native checkpoints, finite propagation, CPU output and all diagnostics. Calls can include compilation; no warm speed claim.'}
    def save():(out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
          TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0',TRITON_CACHE_DIR=str(B/'triton_qwen35'),
          TORCHINDUCTOR_CACHE_DIR=str(B/'inductor_qwen35'))
        for folder in [B/'deps',R/'deltatrace/clean/qwen35',R/'deltatrace/accelerated',R/'experiments/official']:
            sys.path.insert(0,str(folder))
        for name,digest in plan['additional_source_files'].items():assert sha((R/name).read_bytes())==digest,name
        import torch,numpy as np
        from transformers import Qwen3_5ForConditionalGeneration
        from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
        from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
        from retained_qwen35 import make_retained_qwen35
        from code_local_qwen35 import make_code_local_qwen35
        from batching import attribute_batch
        from native_capture_events import check_runtime
        report['monitoring_runtime_check']=check_runtime()
        torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
        torch._dynamo.config.cache_size_limit=max(64,torch._dynamo.config.cache_size_limit)
        torch._dynamo.config.accumulated_cache_size_limit=max(256,torch._dynamo.config.accumulated_cache_size_limit)
        verify_native_sources(env['native_stage_source_sha256'])
        t=time.perf_counter();model=Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True)
        torch.cuda.synchronize();report['model_load_seconds']=time.perf_counter()-t
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        native={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        ids=torch.tensor(plan['cases'][0]['input_ids'],device=model.device)[None]
        t=time.perf_counter()
        with torch.no_grad():initial=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        del initial,ids;torch.cuda.synchronize();report['native_eager_initialization_seconds']=time.perf_counter()-t
        model.set_attn_implementation('flash_attention_2')
        fa=VendorFAFiniteP1BF16D256(str(R/'libdeltatrace_fa_finite_bf16_d256.so'),env['finite_library_sha256'])
        fla=make_compiled_finite_pullback(reuse_scalar_products=False)
        old,report['retained_sources']=make_retained_qwen35(R,model,fa,fla)
        new,report['local_sources']=make_code_local_qwen35(R,model,fa,fla)
        vectors={}
        for index,row in enumerate(plan['cases']):
            ids=torch.tensor(row['input_ids'],dtype=torch.long)[None]
            assert sha(ids.numpy().tobytes())==row['input_sha256'] and ids.shape[1]<=1024
            case={'key':row['key'],'row':row,'ids':ids,'prompt_len':row['prompt_length'],
              'gen_len':row['target_length'],'eval_target':ids[:,row['prompt_length']:],
              'eligible':[row['user_positions'][j] for j in row['keep']]}
            rec={'key':row['key'],'input_sha256':row['input_sha256'],'total_tokens':ids.shape[1],'calls':[]}
            report['cases'].append(rec)
            for mode in (['retained','local'] if index%2==0 else ['local','retained']):
                report['status']=row['key']+'_'+mode;save()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                signed,detail,root=attribute_batch(old if mode=='retained' else new,[case],int(ids[0,-1]),model.device)
                torch.cuda.synchronize();seconds=time.perf_counter()-t
                vector=signed[0].numpy();assert np.isfinite(vector).all()
                assert detail['norm_gate_rules']==detail['attention_pv_rules']=={}
                assert detail['finite_fla_by_layer']==detail['key_norm_by_layer']==[]
                for n,m in model.named_modules():assert native[n]==(type(m).forward,m.forward),n
                vectors[row['key']+'/'+mode]=vector
                rec['calls'].append({'mode':mode,'seconds_including_any_compilation':seconds,
                  'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'actual_root':root,
                  'vector_shape':list(vector.shape),'vector_sha256':sha(vector.tobytes()),'details':detail})
                np.savez(out/'vectors.npz',**vectors);save()
                del signed,vector,detail
            rec['exact']=bool(np.array_equal(vectors[row['key']+'/retained'],vectors[row['key']+'/local']))
            assert rec['exact'],row['key'];save();print(json.dumps({'case':row['key'],'exact':rec['exact']}),flush=True)
        report['status']='complete';report['all_vectors_exact']=all(x['exact'] for x in report['cases'])
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc()
    save();print(json.dumps({'status':report['status'],'error':report.get('error')}),flush=True)


if __name__=='__main__':main()
