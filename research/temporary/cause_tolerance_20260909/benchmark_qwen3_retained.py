"""Whole-call test of retaining native replay tensors instead of copying.

Pilot NI0/MH0 + frozen VT1/Hotpot3; 24 complete paired-endpoint DT calls.
Original native checkpoint copies (including compact target logits) remain.
One warmup and two reversed-order paired rounds. Stop if any vector changes;
there is no changed-vector scorer or silent framework fallback in this test.
"""
import argparse
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
    for n in ('release','environment','reference','extra','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','development16','extra6'],default='pilot')
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    assert sha(args.reference.read_bytes())=='dccfbf8d2f2ba33b0ff68c686031fb86b2ac76501164f63d91228be545984de6'
    assert sha(args.extra.read_bytes())=='28c513b607d30d7275063649cc08699971cc3cece76002dad5e2caf7ba0f547e'
    original=json.loads(args.reference.read_bytes())['cases'];extra=json.loads(args.extra.read_bytes())['cases']
    if args.phase=='pilot':rows=[r for r in original if r['index']==0]+[r for r in extra if (r['dataset'],r['index']) in [('vt_h2_c3',1),('hotpotqa_long',3)]]
    else:rows=original if args.phase=='development16' else extra
    assert len(rows)=={'pilot':4,'development16':16,'extra6':6}[args.phase]
    env=json.loads(args.environment.read_bytes())['qwen3']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/deltatrace_clean_v1_inductor')
    for d in (env['official_root'],str(args.release/'deltatrace/clean/qwen3'),str(args.release/'deltatrace/accelerated')):sys.path.insert(0,d)
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    from exp.exp2 import run_exp as author
    from deferred import make_deferred_qwen3
    baseline,receipt=make_deferred_qwen3(args.release)
    from qwen3_retained_pair import propagate_paired_secant as candidate
    from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
    from vendor_fa_finite_runtime import VendorFAFiniteP1
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(128,torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit=max(512,torch._dynamo.config.accumulated_cache_size_limit)
    args.output.mkdir(exist_ok=False)
    report={'status':'loading','phase':args.phase,'script_sha256':sha(Path(__file__).read_bytes()),'baseline_sources':receipt,
        'candidate_sources':{n:sha((Path(__file__).parent/n).read_bytes()) for n in ('qwen3_retained_replay.py','qwen3_retained_pair.py')},
        'cases':[{k:r[k] for k in ('dataset','index','input_sha256','prompt_length','target_length')} for r in rows],
        'calls':[],'comparisons':[],'FT_calls':0,'generation_calls':0,'metric_calls':0,
        'gate':'Require exact complete vectors; >=3% aggregate warm reduction before expansion or adoption.'}
    vectors={}
    def save():
        p=args.output/'results.partial';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(args.output/'results.json')
    try:
        tick=time.perf_counter();model,tokenizer=author.load_model(env['checkpoint'],'cuda:0')
        model.eval().requires_grad_(False);model.set_attn_implementation('flash_attention_2')
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        report['model_load_seconds']=time.perf_counter()-tick;identities={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        finite=VendorFAFiniteP1(env['finite_library'],env['finite_library_sha256'])
        for phase in ('warm','r0','r1'):
            for row in (rows if phase!='r1' else reversed(rows)):
                key=f"{row['dataset']}_{row['index']}"
                for mode in (('baseline','candidate') if phase!='r1' else ('candidate','baseline')):
                    name=phase+'/'+mode+'/'+key;report['status']=name
                    c={'name':name,'status':'entered','compiler_before':dict(counters['stats'])};report['calls'].append(c);save()
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                    ids=torch.tensor(row['input_ids'],device='cuda')[None]
                    assert sha(ids.cpu().numpy().tobytes())==row['input_sha256']
                    base=ids.clone();base[0,[row['user_positions'][j] for j in row['keep']]]=tokenizer.eos_token_id
                    before,after=capture_checkpoint_pair(model,base,ids,torch.ones_like(ids),row['prompt_length'])
                    root_peak=torch.cuda.max_memory_allocated()
                    result=(baseline if mode=='baseline' else candidate)(model,before,after,pv_rule='content_P1',finite_attention=finite)
                    torch.cuda.synchronize();c.update(status='returned',seconds=time.perf_counter()-tick,
                        peak_allocated=max(root_peak,torch.cuda.max_memory_allocated()),validation=result['deferred_validation'],
                        native_layer_replay_calls=result['native_layer_replay_calls'],public_FA_calls=result['extra_native_fa_attention_calls'],compiler_after=dict(counters['stats']))
                    vector=np.asarray(result['signed_full_sequence'],dtype=np.float64);vectors[name]=vector
                    np.savez_compressed(args.output/'vectors.npz',**vectors)
                    # The optimization promises identical stored operands and arithmetic;
                    # changed vectors falsify it regardless of favorable task scores.
                    if phase!='warm' or mode=='candidate':assert np.array_equal(vector,vectors['warm/baseline/'+key]),name
                    for n,m in model.named_modules():assert (type(m).forward,m.forward)==identities[n]
                    del before,after,result,ids,base,vector;gc.collect();save()
        for row in rows:
            key=f"{row['dataset']}_{row['index']}";report['comparisons'].append({'case':key,'all_six_complete_vectors_equal':True})
        total=lambda mode:sum(c['seconds'] for c in report['calls'] if c['name'].startswith(('r0/'+mode,'r1/'+mode)))/2
        report.update(baseline_seconds_per_pass=total('baseline'),candidate_seconds_per_pass=total('candidate'),warm_reduction_fraction=1-total('candidate')/total('baseline'))
        report['efficiency_gate_passed']=report['warm_reduction_fraction']>=.03
        report['vectors_sha256']=sha((args.output/'vectors.npz').read_bytes());report['status']='complete'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()
