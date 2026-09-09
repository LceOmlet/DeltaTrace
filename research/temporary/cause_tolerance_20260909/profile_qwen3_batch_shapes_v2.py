"""Profile the already measured VT/Hotpot sample pairs using unchanged DT.

Two affected/control groups, each two native B1 calls and one native B2 call,
one warmup and one profiled pass: 12 complete DT calls, 16 sample vectors.
No generation or metric/FT calls. Profiler costs are diagnostic, not throughput.
"""
import argparse
import collections
import gc
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
    for n in ('release','candidate','environment','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    previous=json.loads((args.candidate/'qwen3_batch_extra/results.json').read_bytes())
    assert previous['status']=='complete'
    assert sha(args.reference.read_bytes())==previous['reference_sha256']
    rows={f"{r['dataset']}_{r['index']}":r for r in json.loads(args.reference.read_bytes())['cases']}
    groups=[[rows[k] for k in g] for g in previous['groups'] if len(g)==2]
    assert len(groups)==2
    for n,d in previous['candidate_sources'].items():assert sha((args.candidate/n).read_bytes())==d
    env=json.loads(args.environment.read_bytes())['qwen3']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/deltatrace_clean_v1_inductor')
    for directory in (env['official_root'],str(args.release/'deltatrace/clean/qwen3'),str(args.release/'deltatrace/accelerated'),str(args.candidate)):
        sys.path.insert(0,directory)
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    from exp.exp2 import run_exp as author
    from deferred import make_deferred_qwen3
    from vendor_fa_finite_runtime import VendorFAFiniteP1
    from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
    propagate,receipt=make_deferred_qwen3(args.release)
    from qwen3_batch_pair import capture_batch,propagate_batch
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(128,torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit=max(512,torch._dynamo.config.accumulated_cache_size_limit)
    args.output.mkdir(exist_ok=False)
    report={'status':'loading','driver_sha256':sha(Path(__file__).read_bytes()),'baseline_sources':receipt,
            'candidate_sources':previous['candidate_sources'],'reference_sha256':sha(args.reference.read_bytes()),
            'previous_result_sha256':sha((args.candidate/'qwen3_batch_extra/results.json').read_bytes()),
            'groups':[[f"{r['dataset']}_{r['index']}" for r in g] for g in groups],
            'calls':[],'FT_calls':0,'metric_calls':0,'generation_calls':0,'acceptance_from_profile':False}
    vectors={}
    def save():
        tmp=args.output/'results.partial';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(args.output/'results.json')
    try:
        tick=time.perf_counter();model,tokenizer=author.load_model(env['checkpoint'],'cuda:0')
        model.eval().requires_grad_(False);model.set_attn_implementation('flash_attention_2')
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        finite=VendorFAFiniteP1(env['finite_library'],env['finite_library_sha256'])
        identities={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        report['model_load_seconds']=time.perf_counter()-tick
        for mode in ('warm','profile'):
            for i,group in enumerate(groups):
                for subset in ([group[0]],[group[1]],group):
                    label=mode+'/'+','.join(f"{r['dataset']}_{r['index']}" for r in subset)
                    record={'name':label,'sample_batch':len(subset),'status':'entered','compiler_before':dict(counters['stats'])}
                    report['calls'].append(record);report['status']=label;save();root_peak=[]
                    def run():
                        with torch.profiler.record_function('DT_ROOT_CHECKPOINT'):
                            if len(subset)==2:master=capture_batch(model,subset,tokenizer.eos_token_id)
                            else:
                                r=subset[0];ids=torch.tensor(r['input_ids'],device='cuda')[None];base=ids.clone()
                                base[0,[r['user_positions'][j] for j in r['keep']]]=tokenizer.eos_token_id
                                before,after=capture_checkpoint_pair(model,base,ids,torch.ones_like(ids),r['prompt_length'])
                            root_peak.append(torch.cuda.max_memory_allocated())
                        with torch.profiler.record_function('DT_REPLAY_AND_FINITE'):
                            if len(subset)==2:values,detail=propagate_batch(model,master,finite)
                            else:
                                detail=propagate(model,before,after,pv_rule='content_P1',finite_attention=finite)
                                values=[torch.tensor(detail['signed_full_sequence'],dtype=torch.float64)]
                        record['validation']=detail['deferred_validation']
                        return [v.numpy() for v in values]
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                    if mode=='profile':
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                            values=run();torch.cuda.synchronize()
                        record['events']=[{'key':e.key,'count':e.count,'shapes':e.input_shapes,'self_cpu_us':e.self_cpu_time_total,
                                          'total_cpu_us':e.cpu_time_total,'self_device_us':e.self_device_time_total,'total_device_us':e.device_time_total}
                                         for e in prof.key_averages(group_by_input_shape=True)]
                        # Ancestor ranges separate finite projection GEMMs from
                        # native forward/replay; kernel sums need not be additive.
                        classified=collections.defaultdict(lambda:{'count':0,'device_us':0.,'cpu_us':0.})
                        for e in prof.events():
                            if e.name not in ('aten::mm','aten::addmm','aten::bmm'):continue
                            parent=e.cpu_parent;names=[]
                            while parent is not None:names.append(parent.name);parent=parent.cpu_parent
                            scope='finite_projection' if 'ATTR_NATIVE_HALF_LINEAR' in names else 'native_root' if 'DT_ROOT_CHECKPOINT' in names else 'native_replay_or_other'
                            v=classified[scope];v['count']+=1;v['device_us']+=e.device_time_total;v['cpu_us']+=e.cpu_time_total
                        record['matrix_scopes']=dict(classified)
                    else:values=run();torch.cuda.synchronize()
                    record.update(status='returned',seconds=time.perf_counter()-tick,peak_allocated=max(root_peak+[torch.cuda.max_memory_allocated()]),compiler_after=dict(counters['stats']))
                    for r,v in zip(subset,values):
                        key=f"{r['dataset']}_{r['index']}";name=f'{mode}/B{len(subset)}/'+key;vectors[name]=v
                        if mode=='profile':assert np.array_equal(v,vectors[f'warm/B{len(subset)}/'+key]),name
                    for n,m in model.named_modules():assert (type(m).forward,m.forward)==identities[n]
                    np.savez_compressed(args.output/'vectors.npz',**vectors);save();gc.collect()
        report['status']='complete';report['vectors_sha256']=sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()
