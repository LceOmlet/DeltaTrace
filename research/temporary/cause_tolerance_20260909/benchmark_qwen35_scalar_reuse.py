"""Cost-first pilot of the existing finite FLA scalar-product reuse option.

Reuse inner-product duality to remove three vendor GEMMs. Ordinary model/FA/FLA
are unchanged, BF16 roundoff may differ. Full native DT calls on NI0/MH0,
sample B1/E2, GPU checkpoints and dynamic compiler in both modes. Metrics and
claims of quality acceptance are deferred until useful speed is demonstrated.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release','environment','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    env = json.loads(args.environment.read_bytes())['qwen35']
    references = [('/tmp/codex_clean_development16_20260909_v1/qwen35/results.json',
                   '04c59aaee006b49bb1c93d5ded737805b17570c3517aa045370bd38d37226cdb'),
                  ('/tmp/codex_clean_development16_20260909_mh_recovery_v1/qwen35/results.json',
                   '5999968354f16fe3760a4e2f9bfae9401524d366648cfe7f11fe563339719467')]
    all_rows = []
    for path,digest in references:
        raw = Path(path).read_bytes()
        assert sha(raw) == digest
        all_rows.extend(r for r in json.loads(raw)['cases'] if r['status']=='complete')
    rows = [next(r for r in all_rows if r['dataset']==task and r['index']==0)
            for task in ('niah_mq_q2','morehopqa')]
    manifest = json.loads((args.release/'deltatrace/clean/sources.json').read_bytes())
    for path,receipt in manifest['models']['qwen35']['files'].items():
        assert sha((args.release/path).read_bytes()) == receipt['sha256']
    accel = args.release/'deltatrace/accelerated/qwen35'
    for name,digest in {'controller.py':'e3ac00cc1fc3c238f4cbf1f05f5af020641a1e232508235db8bcddc9f8efefba',
                        'dynamic_finite.py':'b3d4947f60e73a99322fb80da224865918615aa109a9098b3e688c3f4a7a7020'}.items():
        assert sha((accel/name).read_bytes())==digest,name
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/deltatrace_clean_v1_inductor')
    for path in env.get('dependency_overlays',[]):sys.path.insert(0,path)
    sys.path.insert(0,str(args.release/'deltatrace/clean/qwen35'))
    sys.path.insert(0,str(accel))
    import numpy as np
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from controller import Qwen35DenseFiniteRunner
    from dynamic_finite import configure_dynamic_finite
    from qwen35_answer_finite import PackedAnswerTargets
    from finite_fla_gpu import mixed_coefficients,native_input_adjoints,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    torch.set_num_threads(4)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(64,torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit=max(256,torch._dynamo.config.accumulated_cache_size_limit)
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'loading','script_sha256':sha(Path(__file__).read_bytes()),
            'accelerated_sources':{n:sha((accel/n).read_bytes()) for n in ('controller.py','dynamic_finite.py')},
            'cases':[],'calls':[],'root_calls':[],'comparisons':[],
            'sample_batch':1,'endpoint_batch':2,'metrics_recomputed':False,'generation_calls':0,'FT_calls':0,
            'gate_for_quality_followup':'At least 3% aggregate warmed wall-time reduction in this two-case screen; no acceptance without subsequent quality evaluation.'}
    vectors={}

    def save():
        path=args.output/'results.partial'
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        path.replace(args.output/'results.json')

    def timed(name,fn):
        report['status']=name
        row={'name':name,'status':'entered'}
        report['calls'].append(row)
        save()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        try:
            result=fn();torch.cuda.synchronize();row['status']='returned';return result,row
        finally:
            row.update(seconds=time.perf_counter()-tick,peak_allocated=torch.cuda.max_memory_allocated(),
                       peak_reserved=torch.cuda.max_memory_reserved());save()

    try:
        tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True)
        model,_=timed('model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,
                    attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        identities={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        first=torch.tensor(rows[0]['input_ids'],device='cuda')[None]
        with torch.no_grad():initial,_=timed('native_initialization',lambda:model(input_ids=first,attention_mask=torch.ones_like(first),use_cache=False))
        del initial,first
        model.set_attn_implementation('flash_attention_2');verify_native_sources(env['native_stage_source_sha256'])
        fa=VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
        compiled=torch.compile(mixed_coefficients,fullgraph=True,dynamic=True,
                               options={'triton.cudagraphs':False,'max_autotune':False})
        def reuse(e,m,scale):
            return compiled(e,native_input_adjoints(e,m,scale),scale,reuse_scalar_products=True)
        runner=configure_dynamic_finite(Qwen35DenseFiniteRunner(model,fa,reuse,checkpoint_device='cuda'))
        original=runner.finite_fla
        for row in rows:
            case=f"{row['dataset']}_{row['index']}"
            report['cases'].append({k:row[k] for k in ('dataset','index','input_sha256','prompt_length','target_length')})
            cpu=torch.tensor(row['input_ids'],dtype=torch.long)
            assert sha(cpu.numpy().tobytes())==row['input_sha256']
            for repeat,mode in enumerate(('baseline','reuse','baseline','reuse','reuse','baseline')):
                runner.finite_fla=original if mode=='baseline' else reuse
                name=case+'_'+mode+('_cold' if repeat<2 else f'_measured{repeat}')
                root_calls=[]
                def observe(_module,a,kw):
                    root_calls.append({'shape':list(kw['input_ids'].shape),'sample_batch':1,'endpoint_batch':2,
                                       'input_sha256':sha(kw['input_ids'].cpu().numpy().tobytes())})
                handle=model.register_forward_pre_hook(observe,with_kwargs=True)
                def attribute():
                    pair=cpu.to('cuda')[None].repeat(2,1)
                    pair[0,[row['user_positions'][k] for k in row['keep']]]=tokenizer.eos_token_id
                    target=cpu[row['prompt_length']:]
                    selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':row['prompt_length']}],
                        [list(range(len(target)))],len(cpu),'cuda')
                    return runner.attribute(pair,torch.ones_like(pair),selection)
                try:(signed,detail),cost=timed(name,attribute)
                finally:handle.remove()
                assert len(root_calls)==1 and torch.isfinite(signed).all()
                report['root_calls'].append({'name':name,**root_calls[0]})
                cost.update(warm=repeat>=2,detail=detail)
                vectors[name]=signed[0].numpy()
                for n,m in model.named_modules():assert (type(m).forward,m.forward)==identities[n],n
                np.savez_compressed(args.output/'vectors.npz',**vectors);save()
                print(json.dumps({'name':name,'seconds':cost['seconds']}),flush=True)
            a=vectors[case+'_baseline_measured5'];b=vectors[case+'_reuse_measured4']
            report['comparisons'].append({'case':case,'relative_l2':float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30)),
                'max_absolute':float(np.abs(b-a).max()),'baseline_repeat_equal':bool(np.array_equal(a,vectors[case+'_baseline_measured2'])),
                'reuse_repeat_equal':bool(np.array_equal(b,vectors[case+'_reuse_measured3']))})
        total=lambda mode:sum(c['seconds'] for c in report['calls'] if c.get('warm') and '_'+mode+'_' in c['name'])
        report['mean_latency_reduction_fraction']=1-total('reuse')/total('baseline')
        report['quality_followup_warranted']=report['mean_latency_reduction_fraction']>=.03
        report['status']='complete';report['vectors_sha256']=sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()
