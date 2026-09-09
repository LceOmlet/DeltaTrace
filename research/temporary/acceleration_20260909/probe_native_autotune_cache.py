"""Test the installed vendor Triton's persistent-cache option on real inputs.

Only environment configuration changes. Native model/FA/FLA/Triton functions
are observed, never replaced. This measures cold-process reuse, not DT quality.
"""
import argparse, hashlib, inspect, json, os, sys, time, traceback
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    assert os.environ['TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS']=='1'
    assert Path(os.environ['TRITON_AUTOTUNE_CONFIG_PATH'])==args.cache
    raw=args.reference.read_bytes();assert sha(raw)==args.reference_sha256
    reference=json.loads(raw);assert reference['status']=='complete'
    env=json.loads(args.environment.read_bytes())['qwen35']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,env['official_root'])
    for overlay in env.get('dependency_overlays',[]):sys.path.insert(0,overlay)
    sys.path.insert(0,str(args.release/'deltatrace/clean/qwen35'))
    import numpy as np
    import torch
    import triton.runtime.autotuner as autotuner
    from transformers import Qwen3_5ForConditionalGeneration
    from qwen35_answer_finite import PackedAnswerTargets
    from native_target_logit_rows import NativeTargetLogitRows
    from finite_fla_gpu import verify_native_sources
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    verify_native_sources(env['native_stage_source_sha256'])
    autotuner_path=Path(inspect.getfile(autotuner))
    source=autotuner_path.read_bytes()
    assert b'TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS' in source and b'TRITON_AUTOTUNE_CONFIG_PATH' in source
    selected=[next(c for c in reference['cases'] if c['index']==index and c['dataset']=='morehopqa') for index in (5,1)]
    args.output.mkdir(exist_ok=False)
    cache_files=lambda:{str(f.relative_to(args.cache)):sha(f.read_bytes()) for f in sorted(args.cache.rglob('*.json'))}
    report={'status':'loading','source_sha256':sha(Path(__file__).read_bytes()),'autotuner_source_sha256':sha(source),
        'reference_sha256':sha(raw),'cases':[{k:c[k] for k in ('dataset','index','input_sha256')} for c in selected],
        'cache_before':cache_files(),'calls':[],'sample_batch':2,'endpoint_batch':4,
        'device':torch.cuda.get_device_name(),'torch_version':torch.__version__,
        'scope':'Native two-case first-process cache write / next-process cache reuse; no quality metric claim'}
    def save():(args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    def timed(name,fn):
        events=[];active={}
        def observe(frame,event,result):
            if frame.f_code.co_filename!=str(autotuner_path) or frame.f_code.co_name!='_bench':return
            if event=='call':
                active[frame]=(time.perf_counter(),frame.f_locals['self'].base_fn.__name__)
            elif event=='return' and frame in active:
                tick,kernel=active.pop(frame);events.append({'kernel':kernel,'seconds':time.perf_counter()-tick})
        assert sys.getprofile() is None
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        row={'name':name,'status':'entered'};report['calls'].append(row);save()
        sys.setprofile(observe)
        try:
            result=fn();torch.cuda.synchronize();row['status']='returned';return result
        finally:
            sys.setprofile(None)
            row.update(seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),autotune_benches=events)
            save()
    save()
    try:
        model=timed('load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        identity={name:(type(module).forward,module.forward) for name,module in model.named_modules()}
        length=max(len(c['input_ids']) for c in selected);eos=selected[0]['input_ids'][-1]
        ids=torch.full((4,length),eos,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids);targets=[];offsets=[]
        for j,c in enumerate(selected):
            actual=torch.tensor(c['input_ids'],dtype=torch.long,device='cuda');n=len(actual)
            assert sha(actual.cpu().numpy().tobytes())==c['input_sha256']
            ids[2*j:2*j+2,:n]=actual
            ids[2*j,[c['user_positions'][k] for k in c['keep']]]=eos
            mask[2*j:2*j+2,:n]=1
            targets.append({'target_ids':actual[c['prompt_length']:].cpu(),'prompt_length':c['prompt_length']})
            offsets.append(list(range(n-c['prompt_length'])))
        selection=PackedAnswerTargets(targets,offsets,length,'cuda');selector=NativeTargetLogitRows(selection)
        report['paired_input_sha256']=sha(ids.cpu().numpy().tobytes());report['mask_sha256']=sha(mask.cpu().numpy().tobytes())
        native_heads=[]
        handle=model.lm_head.register_forward_pre_hook(lambda _m,a:native_heads.append(list(a[0].shape)))
        report['native_target_logp']={}
        for name,implementation in [('eager_initialization','eager'),('FA_first','flash_attention_2'),('FA_repeat','flash_attention_2')]:
            model.set_attn_implementation(implementation)
            with torch.no_grad():
                out=timed(name,lambda:model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=selector.rows))
                packed=selector.pack_logits(out.logits)
                values=packed.float().log_softmax(-1).gather(-1,selection.labels.repeat_interleave(2)[:,None]).squeeze(-1)
                report['native_target_logp'][name]=values.cpu().tolist()
            del out,packed,values
            for module_name,module in model.named_modules():assert (type(module).forward,module.forward)==identity[module_name]
            save()
        handle.remove()
        report['native_head_input_shapes']=native_heads
        assert sha(autotuner_path.read_bytes())==report['autotuner_source_sha256']
        report['cache_after']=cache_files();report['status']='complete'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()
