"""Profile complete current DT calls; no model, method or validation changes.

One warm and one instrumented call per original NI0/MH0. Timing under the
profiler is diagnostic only. Source-bound native FA/FLA and frozen arithmetic.
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['qwen3','qwen35'],required=True)
    for name in ('release','environment','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    env=json.loads(args.environment.read_bytes())[args.family]
    refs=[('/tmp/codex_clean_development16_20260909_v1/'+args.family+'/results.json',
           'dccfbf8d2f2ba33b0ff68c686031fb86b2ac76501164f63d91228be545984de6' if args.family=='qwen3' else
           '04c59aaee006b49bb1c93d5ded737805b17570c3517aa045370bd38d37226cdb')]
    if args.family=='qwen35':refs.append(('/tmp/codex_clean_development16_20260909_mh_recovery_v1/qwen35/results.json',
                                          '5999968354f16fe3760a4e2f9bfae9401524d366648cfe7f11fe563339719467'))
    allrows=[]
    for path,digest in refs:
        raw=Path(path).read_bytes();assert sha(raw)==digest
        allrows.extend(r for r in json.loads(raw)['cases'] if r['status']=='complete')
    rows=[next(r for r in allrows if r['dataset']==task and r['index']==0) for task in ('niah_mq_q2','morehopqa')]
    sources=json.loads((args.release/'deltatrace/clean/sources.json').read_bytes())
    for name,rec in sources['models'][args.family]['files'].items():assert sha((args.release/name).read_bytes())==rec['sha256']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/deltatrace_clean_v1_inductor')
    sys.path.insert(0,env['official_root'])
    for path in env.get('dependency_overlays',[]):sys.path.insert(0,path)
    sys.path.insert(0,str(args.release/'deltatrace/clean'/args.family))
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(64,torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit=max(256,torch._dynamo.config.accumulated_cache_size_limit)
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'loading','family':args.family,'script_sha256':sha(Path(__file__).read_bytes()),
            'reference_sha256':[s for _,s in refs],'sample_batch':1,'endpoint_batch':2,'calls':[],
            'quality_or_cost_acceptance':False,'metric_calls':0,'generation_calls':0,'FT_calls':0}
    vectors={}
    def save():
        p=args.output/'results.partial';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(args.output/'results.json')
    try:
        tick=time.perf_counter()
        if args.family=='qwen3':
            from exp.exp2 import run_exp as author
            model,tokenizer=author.load_model(env['checkpoint'],'cuda:0')
            from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
            from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant
            from vendor_fa_finite_runtime import VendorFAFiniteP1
            finite_fa=VendorFAFiniteP1(env['finite_library'],env['finite_library_sha256'])
        else:
            from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
            model=Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,
                attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True)
            tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True)
            from qwen35_answer_finite import PackedAnswerTargets
            from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
            from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
            sys.path.insert(0,str(args.release/'deltatrace/accelerated/qwen35'))
            from controller import Qwen35DenseFiniteRunner
            from dynamic_finite import configure_dynamic_finite
            verify_native_sources(env['native_stage_source_sha256'])
            finite_fa=VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
            runner=configure_dynamic_finite(Qwen35DenseFiniteRunner(model,finite_fa,
                    make_compiled_finite_pullback(False),checkpoint_device='cuda'))
        model.eval().requires_grad_(False);torch.cuda.synchronize();report['load_seconds']=time.perf_counter()-tick
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        identities={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        if args.family=='qwen35':
            ids=torch.tensor(rows[0]['input_ids'],device='cuda')[None]
            with torch.no_grad():out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            del out,ids
        model.set_attn_implementation('flash_attention_2')
        for row in rows:
            case=f"{row['dataset']}_{row['index']}"
            for mode in ('warm','profile'):
                report['status']=case+'_'+mode;save()
                def run():
                    ids=torch.tensor(row['input_ids'],device='cuda')[None];mask=torch.ones_like(ids);base=ids.clone()
                    base[0,[row['user_positions'][k] for k in row['keep']]]=tokenizer.eos_token_id
                    if args.family=='qwen3':
                        with torch.profiler.record_function('DT_ROOT_CHECKPOINT'):
                            before,after=capture_checkpoint_pair(model,base,ids,mask,row['prompt_length'])
                        with torch.profiler.record_function('DT_NATIVE_REPLAY_AND_FINITE'):
                            detail=propagate_paired_secant(model,before,after,pv_rule='content_P1',finite_attention=finite_fa)
                        return np.asarray(detail['signed_full_sequence'],dtype=np.float64)
                    target=ids[0,row['prompt_length']:].cpu()
                    selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':row['prompt_length']}],
                        [list(range(len(target)))],len(row['input_ids']),'cuda')
                    signed,detail=runner.attribute(torch.cat((base,ids)),torch.cat((mask,mask)),selection)
                    return signed[0].numpy()
                torch.cuda.synchronize();start=time.perf_counter()
                if mode=='profile':
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        vector=run();torch.cuda.synchronize()
                    events=[{'key':e.key,'count':e.count,'self_cpu_us':e.self_cpu_time_total,'total_cpu_us':e.cpu_time_total,
                             'self_device_us':e.self_device_time_total,'total_device_us':e.device_time_total} for e in prof.key_averages()]
                else:
                    vector=run();torch.cuda.synchronize();events=None
                vectors[case+'_'+mode]=vector
                report['calls'].append({'case':case,'mode':mode,'seconds':time.perf_counter()-start,
                    'input_sha256':row['input_sha256'],'events':events})
                for n,m in model.named_modules():assert (type(m).forward,m.forward)==identities[n],n
                np.savez_compressed(args.output/'vectors.npz',**vectors);save()
                print(json.dumps({'case':case,'mode':mode,'events':len(events) if events else None}),flush=True)
            a=vectors[case+'_warm'];b=vectors[case+'_profile'];assert np.isfinite(b).all()
            report['calls'][-1]['vector_equal_to_warm']=bool(np.array_equal(a,b))
        report['status']='complete';report['vectors_sha256']=sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()
