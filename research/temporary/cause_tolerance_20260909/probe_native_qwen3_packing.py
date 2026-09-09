"""Native Qwen3 packing feasibility; not a new attribution implementation.

Use public position_ids/cu-sequence metadata to run the installed model's
padding-free FA path. Same frozen VT/Hotpot pairs, baseline/input endpoints.
28 root forwards: warm + two interleaved measured rounds for B1/dense/packed,
then four same-shape cross-sample isolation checks. One public varlen LSE call.
No model/FA replacement, backward, attribution, generation, RISE or MAS calls.
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
    for n in ('release','candidate','environment','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
    previous=json.loads((args.candidate/'qwen3_batch_extra/results.json').read_bytes());assert previous['status']=='complete'
    assert sha(args.reference.read_bytes())==previous['reference_sha256']
    rows={f"{r['dataset']}_{r['index']}":r for r in json.loads(args.reference.read_bytes())['cases']}
    groups=[[rows[k] for k in g] for g in previous['groups'] if len(g)==2];assert len(groups)==2
    env=json.loads(args.environment.read_bytes())['qwen3']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,env['official_root'])
    import numpy as np
    import torch
    import transformers.modeling_flash_attention_utils as fa_utils
    from exp.exp2 import run_exp as author
    from flash_attn import flash_attn_func,flash_attn_varlen_func
    assert sha(Path(fa_utils.__file__).read_bytes())=='293fe81c6bd38aac8a3bc6ae086ff21b9695b349e97c9650b1fe61e42a36d710'
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(exist_ok=False)
    report={'status':'loading','driver_sha256':sha(Path(__file__).read_bytes()),'reference_sha256':sha(args.reference.read_bytes()),
        'native_FA_adapter_sha256':sha(Path(fa_utils.__file__).read_bytes()),'calls':[],'comparisons':[],'isolation_checks':[],
        'FA_metadata_checks':[],'root_calls':0,'public_LSE_calls':0,'DT_calls':0,'FT_calls':0,'metric_calls':0,'generation_calls':0,
        'scope':'Only installed native root model calls. No finite propagation, full DT speed or quality claim.'}
    saved={};warm={}
    def save():
        q=args.output/'results.partial';q.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');q.replace(args.output/'results.json')
    try:
        tick=time.perf_counter();model,tokenizer=author.load_model(env['checkpoint'],'cuda:0')
        model.eval().requires_grad_(False);model.set_attn_implementation('flash_attention_2')
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        report['load_seconds']=time.perf_counter()-tick;identities={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        def make(group,mode):
            lengths=[len(r['input_ids']) for r in group];total=sum(lengths)
            if mode=='packed':
                ids=torch.empty((2,total),device='cuda',dtype=torch.long);positions=torch.cat([torch.arange(n,device='cuda') for n in lengths])[None].expand(2,-1)
                offset=0;segments=[]
                for r,n in zip(group,lengths):
                    ids[:,offset:offset+n]=torch.tensor(r['input_ids'],device='cuda');ids[0,[offset+r['user_positions'][j] for j in r['keep']]]=tokenizer.eos_token_id
                    segments.append((0,offset,n,r));offset+=n
                cumulative=[0];offset=0
                for n in lengths*2:offset+=n;cumulative.append(offset)
                cu=torch.tensor(cumulative,device='cuda',dtype=torch.int32)
                kwargs={'position_ids':positions,'cu_seq_lens_q':cu,'cu_seq_lens_k':cu,'max_length_q':max(lengths),'max_length_k':max(lengths)}
            else:
                n=max(lengths);ids=torch.full((2*len(group),n),tokenizer.eos_token_id,device='cuda',dtype=torch.long);segments=[]
                for b,(r,length) in enumerate(zip(group,lengths)):
                    ids[2*b:2*b+2,:length]=torch.tensor(r['input_ids'],device='cuda');ids[2*b,[r['user_positions'][j] for j in r['keep']]]=tokenizer.eos_token_id
                    segments.append((2*b,0,length,r))
                kwargs={}
            return ids,kwargs,segments
        def forward(group,mode,label,mutate=None):
            ids,kwargs,segments=make(group,mode);captured=[];native=[];last={}
            if mutate is not None:
                side,offset,n,_=segments[mutate];ids[side:side+2,offset:offset+n]=0
            def hidden(_module,args):captured.append(args[0].detach())
            def observe(frame,event,result):
                if event!='return' or frame.f_code not in (flash_attn_func.__code__,flash_attn_varlen_func.__code__):return
                is_packed=frame.f_code is flash_attn_varlen_func.__code__;assert is_packed==(mode=='packed')
                native.append(is_packed)
                if is_packed:
                    for name,key in [('cu_seqlens_q','cu_seq_lens_q'),('cu_seqlens_k','cu_seq_lens_k')]:assert torch.equal(frame.f_locals[name],kwargs[key])
                    assert frame.f_locals['max_seqlen_q']==kwargs['max_length_q'] and frame.f_locals['max_seqlen_k']==kwargs['max_length_k']
                    assert frame.f_locals['causal'] and frame.f_locals['dropout_p']==0
                    if label=='warm/0/packed':
                        last['args']={n:frame.f_locals[n] for n in inspect.signature(flash_attn_varlen_func).parameters}
                        last['out']=result
            handle=model.model.norm.register_forward_pre_hook(hidden)
            # Observe only warm/isolated calls; keep profile callback cost out of measured timing.
            observing=label.startswith('warm') or mutate is not None
            assert sys.getprofile() is None
            if observing:sys.setprofile(observe)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            try:
                with torch.no_grad():output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,**kwargs)
                torch.cuda.synchronize()
            finally:
                elapsed=time.perf_counter()-tick;handle.remove()
                if observing:sys.setprofile(None)
            report['root_calls']+=1
            record={'name':label,'mode':mode,'sample_count':len(group),'input_shape':list(ids.shape),'seconds':elapsed,'peak_allocated':torch.cuda.max_memory_allocated()}
            report['calls'].append(record);assert len(captured)==1
            if observing:
                assert len(native)==36
                report['FA_metadata_checks'].append({'name':label,'public_calls':len(native),'varlen_calls':sum(native),'cumulative_lengths_exact':mode=='packed'})
            values={}
            for side,offset,n,r in segments:
                key=f"{r['dataset']}_{r['index']}";h=captured[0][side:side+2,offset:offset+n].cpu()
                logits=output.logits[side:side+2,offset+r['prompt_length']-1:offset+n-1]
                target=torch.tensor(r['input_ids'][r['prompt_length']:],device='cuda')
                lp=logits.float().log_softmax(-1).gather(-1,target[None,:,None].expand(2,-1,-1)).squeeze(-1).cpu()
                values[key]={'hidden':h,'logp':lp};saved[label+'/'+key]=lp.numpy()
            if last:
                last['args']['return_attn_probs']=True
                with torch.no_grad():aux=flash_attn_varlen_func(**last['args'])
                report['public_LSE_calls']+=1;assert len(aux)==3
                assert torch.equal(aux[0],last['out'])
                assert aux[2] is None or aux[2].numel()==0
                report['public_LSE_contract']={'shape':list(aux[1].shape),'dtype':str(aux[1].dtype),'testing_numel':0,'output_exact':True}
            for n,m in model.named_modules():assert (type(m).forward,m.forward)==identities[n]
            save();return values
        for phase in ('warm','r0','r1'):
            for i in (range(2) if phase!='r1' else reversed(range(2))):
                group=groups[i]
                for mode in (('single','dense','packed') if phase!='r1' else ('packed','dense','single')):
                    sets=[[r] for r in group] if mode=='single' else [group]
                    for j,subset in enumerate(sets):
                        label=f'{phase}/{i}/{mode}'+(f'/{j}' if mode=='single' else '')
                        value=forward(subset,mode,label)
                        if phase=='warm':
                            for key,v in value.items():warm[(i,mode,key)]=v
        for i,group in enumerate(groups):
            for r in group:
                key=f"{r['dataset']}_{r['index']}";a=warm[(i,'single',key)]
                for mode in ('dense','packed'):
                    b=warm[(i,mode,key)]
                    report['comparisons'].append({'group':i,'case':key,'mode':mode,'hidden_equal':bool(torch.equal(a['hidden'],b['hidden'])),
                        'hidden_relative_l2':float((a['hidden'].float()-b['hidden'].float()).norm()/a['hidden'].float().norm()),
                        'target_logp_max_absolute':float((a['logp']-b['logp']).abs().max())})
            for mutate in range(2):
                values=forward(group,'packed',f'isolate/{i}/{mutate}',mutate=mutate)
                r=group[1-mutate];key=f"{r['dataset']}_{r['index']}";a=warm[(i,'packed',key)];b=values[key]
                exact=torch.equal(a['hidden'],b['hidden']) and torch.equal(a['logp'],b['logp'])
                assert exact,'Native varlen packed call mixed independent samples'
                report['isolation_checks'].append({'group':i,'mutated_case':mutate,'unchanged_case':key,'hidden_and_target_exact':exact})
        np.savez_compressed(args.output/'target_logprobs.npz',**saved)
        report['target_logprobs_sha256']=sha((args.output/'target_logprobs.npz').read_bytes());report['status']='complete'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()


if __name__=='__main__':main()
