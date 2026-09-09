"""Cost-first real Qwen3 sample batching, preserving author prefixes and targets.

Pilot: first length-sorted pair within each original NI8/MH8 task. Development:
all eight task-local pairs. One warmup per mode/group, two interleaved measured
rounds with reversed mode order. No generation or FT calls. Original metrics are
a subsequent gate if batch vectors change and worthwhile throughput is observed.
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
    for n in ('release','environment','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reference-sha256',required=True)
    p.add_argument('--phase',choices=['pilot','development16','extra'],default='pilot')
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    assert sha(args.reference.read_bytes())==args.reference_sha256
    reference=json.loads(args.reference.read_bytes());assert reference['family']=='qwen3'
    assert reference.get('status')=='complete' or reference.get('preparation_status')=='frozen'
    from qwen3_padding_groups import group_by_padding
    rows=reference['cases'];groups=[]
    for task in dict.fromkeys(r['dataset'] for r in rows):
        selected=sorted([r for r in rows if r['dataset']==task],key=lambda r:(len(r['input_ids']),r['index']))
        if args.phase!='extra':assert len(selected)==8
        if args.phase=='pilot':selected=selected[:2]
        groups.extend(group_by_padding(selected,2,lambda r:len(r['input_ids'])))
    assert all(1<=len(g)<=2 for g in groups)
    env=json.loads(args.environment.read_bytes())['qwen3']
    clean=json.loads((args.release/'deltatrace/clean/sources.json').read_bytes())
    for name,r in clean['models']['qwen3']['files'].items():assert sha((args.release/name).read_bytes())==r['sha256']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/deltatrace_clean_v1_inductor')
    sys.path.insert(0,env['official_root'])
    sys.path.insert(0,str(args.release/'deltatrace/clean/qwen3'))
    sys.path.insert(0,str(args.release/'deltatrace/accelerated'))
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    from exp.exp2 import run_exp as author
    from deferred import make_deferred_qwen3
    from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
    from vendor_fa_finite_runtime import VendorFAFiniteP1
    propagate,source=make_deferred_qwen3(args.release)
    from qwen3_batch_pair import capture_batch,propagate_batch
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(128,torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit=max(512,torch._dynamo.config.accumulated_cache_size_limit)
    args.output.mkdir(exist_ok=False)
    report={'status':'loading','driver_sha256':sha(Path(__file__).read_bytes()),'reference_sha256':args.reference_sha256,
            'phase':args.phase,'baseline_sources':source,'candidate_sources':{n:sha((Path(__file__).parent/n).read_bytes()) for n in
                ('qwen3_batch_pair.py','qwen3_batch_capture.py','qwen3_batch_layout.py','qwen3_batch_finite.py','qwen3_padding_groups.py')},
            'groups':[[f"{c['dataset']}_{c['index']}" for c in g] for g in groups], 'max_padding_work_ratio':1.25,
            'calls':[],'prefix_checks':[],'comparisons':[],'generation_calls':0,'FT_calls':0,'metric_calls':0,
            'throughput_gate':'At least 3% aggregate warm reduction before any changed-vector metric follow-up; no quality acceptance from timing or L2 alone.'}
    vectors={};native={}
    def save():
        tmp=args.output/'results.partial';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(args.output/'results.json')
    def timed(name,fn):
        report['status']=name;save();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        row={'name':name,'status':'entered','compiler_before':dict(counters['stats'])};report['calls'].append(row)
        try:
            result=fn();torch.cuda.synchronize();row['status']='returned';return result,row
        finally:row.update(seconds=time.perf_counter()-tick,peak_allocated=torch.cuda.max_memory_allocated(),compiler_after=dict(counters['stats']));save()
    try:
        (model,tokenizer),_=timed('model_load',lambda:author.load_model(env['checkpoint'],'cuda:0'))
        model.eval().requires_grad_(False);model.set_attn_implementation('flash_attention_2')
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        identities={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        finite=VendorFAFiniteP1(env['finite_library'],env['finite_library_sha256'])
        def identity():
            for n,m in model.named_modules():assert (type(m).forward,m.forward)==identities[n]
        def key(c):return f"{c['dataset']}_{c['index']}"
        def single(row,label):
            name=label+'/'+key(row);stages={}
            def run():
                ids=torch.tensor(row['input_ids'],device='cuda')[None];base=ids.clone();base[0,[row['user_positions'][j] for j in row['keep']]]=tokenizer.eos_token_id
                before,after=capture_checkpoint_pair(model,base,ids,torch.ones_like(ids),row['prompt_length'])
                stages['root_peak']=torch.cuda.max_memory_allocated()
                if label.startswith('warm'):
                    native[key(row)]={'last':before['paired_checkpoint']['last'].cpu(),
                                      'logp':torch.stack((before['target_logprobs32'],after['target_logprobs32'])).cpu()}
                return propagate(model,before,after,pv_rule='content_P1',finite_attention=finite)
            result,call=timed(name,run);call['peak_allocated']=max(call['peak_allocated'],stages['root_peak']);call['sample_batch']=1
            call['validation']=result['deferred_validation'];vectors[name]=np.asarray(result['signed_full_sequence'],dtype=np.float64)
            identity();np.savez_compressed(args.output/'vectors.npz',**vectors);save();gc.collect()
        def batched(group,index,label):
            if len(group)==1:
                single(group[0],label+'/'+str(index));return
            name=label+'/'+str(index);stages={}
            def run():
                master=capture_batch(model,group,tokenizer.eos_token_id);stages['root_peak']=torch.cuda.max_memory_allocated()
                if label.startswith('warm'):
                    for b,row in enumerate(group):
                        n=len(row['input_ids']);a=native[key(row)]['last'].to('cuda');v=master['last'][2*b:2*b+2,:n]
                        select=master['target_samples']==b
                        lp=native[key(row)]['logp'].to('cuda');newlp=master['target_logprobs32'][:,select]
                        check={'case':key(row),'tail_tokens':master['length']-n,'last_hidden_equal':bool(torch.equal(a,v)),
                               'last_hidden_relative_l2':float((a.float()-v.float()).norm()/a.float().norm().clamp_min(1e-30)),
                               'target_logp_max_absolute':float((lp-newlp).abs().max()),'tail_target_count':0}
                        report['prefix_checks'].append(check)
                        del a,v,lp,newlp
                    # Same shape/dtype/native model, altered tails only. This
                    # separates causal padding correctness from normal native
                    # B2-vs-B4 GEMM/FA numerical changes in the comparison above.
                    tail_ids=torch.full((2*len(group),master['length']),tokenizer.eos_token_id,device='cuda',dtype=torch.long)
                    changed_tail=0
                    for b,row in enumerate(group):
                        n=len(row['input_ids']);tail_ids[2*b:2*b+2,:n]=torch.tensor(row['input_ids'],device='cuda')
                        tail_ids[2*b,[row['user_positions'][j] for j in row['keep']]]=tokenizer.eos_token_id
                        tail_ids[2*b:2*b+2,n:]=0;changed_tail+=2*(master['length']-n)
                    if changed_tail:
                        captured=[]
                        def norm_input(_module,args):captured.append(args[0].detach())
                        handle=model.model.norm.register_forward_pre_hook(norm_input)
                        try:
                            with torch.no_grad():tail_output=model(input_ids=tail_ids,attention_mask=torch.ones_like(tail_ids),use_cache=False)
                        finally:handle.remove()
                        assert len(captured)==1
                        for b,row in enumerate(group):
                            n=len(row['input_ids'])
                            same=bool(torch.equal(captured[0][2*b:2*b+2,:n],master['last'][2*b:2*b+2,:n]))
                            report['prefix_checks'][len(report['prefix_checks'])-len(group)+b]['same_shape_changed_tail_hidden_exact']=same
                            assert same,'Native causal prefix changed when only future tail tokens changed'
                        report['native_tail_probe_calls']=report.get('native_tail_probe_calls',0)+1
                        del tail_output,captured
                    del tail_ids
                return propagate_batch(model,master,finite)
            (signed,details),call=timed(name,run);call['sample_batch']=2;call['peak_allocated']=max(call['peak_allocated'],stages['root_peak'])
            call['details']=details
            for row,vector in zip(group,signed):vectors[name+'/'+key(row)]=vector.numpy()
            identity();np.savez_compressed(args.output/'vectors.npz',**vectors);save();gc.collect()
        for i,g in enumerate(groups):
            for row in g:single(row,'warm_single')
            batched(g,i,'warm_batch')
        native.clear()
        for repeat in range(2):
            for i in (range(len(groups)) if repeat==0 else reversed(range(len(groups)))):
                for mode in (('single','batch') if repeat==0 else ('batch','single')):
                    if mode=='single':
                        for row in groups[i]:single(row,f'r{repeat}_single')
                    else:batched(groups[i],i,f'r{repeat}_batch')
        for i,g in enumerate(groups):
            for row in g:
                case=key(row);a=vectors['r0_single/'+case];b=vectors[f'r0_batch/{i}/'+case]
                report['comparisons'].append({'case':case,'full_vector_equal':bool(np.array_equal(a,b)),
                    'relative_l2':float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30)),
                    'max_absolute':float(np.abs(b-a).max()),'single_repeat_equal':bool(np.array_equal(a,vectors['r1_single/'+case])),
                    'batch_repeat_equal':bool(np.array_equal(b,vectors[f'r1_batch/{i}/'+case]))})
        total=lambda mode:sum(c['seconds'] for c in report['calls'] if c['name'].startswith(('r0_'+mode,'r1_'+mode)))/2
        report['baseline_seconds_per_pass']=total('single');report['batch_seconds_per_pass']=total('batch')
        report['warm_reduction_fraction']=1-total('batch')/total('single')
        report['throughput_gate_passed']=report['warm_reduction_fraction']>=.03
        report['status']='complete';report['vectors_sha256']=sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()
