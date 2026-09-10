"""Compare target-row operations on the exact same original full-model logits."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time,traceback
T=Path(__file__).resolve().parent;P=json.loads((T/'target_diagnostic_protocol.json').read_bytes());R=Path(P['release_root']);B=Path('/tmp/codex_short_b1_efficiency_20260910_v1')
out=T/'target_diagnostic';out.mkdir(exist_ok=False);sha=lambda b:hashlib.sha256(b).hexdigest()
report={'status':'imports','cases':[],'driver_sha256':sha(Path(__file__).read_bytes()),'protocol_sha256':sha((T/'target_diagnostic_protocol.json').read_bytes())}
def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    assert report['driver_sha256']==P['driver_sha256']
    for n,h in P['runtime_files'].items():assert sha((R/n).read_bytes())==h,n
    for n,h in P['diagnostic_modules'].items():assert sha((T/n).read_bytes())==h,n
    env=json.loads((R/'environment.json').read_bytes())['qwen3']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',TRITON_CACHE_DIR=str(T/'triton_rollout_deltatrace_streamed'),TORCHINDUCTOR_CACHE_DIR=str(T/'inductor_rollout_deltatrace_streamed'))
    for folder in [B/'deps',Path(env['official_root']),R/'deltatrace/clean/qwen3',R/'deltatrace/accelerated']:sys.path.insert(0,str(folder))
    import torch,numpy as np
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=64;torch._dynamo.config.accumulated_cache_size_limit=256
    source=R/'official_exp1/run_time_curve.py';assert sha(source.read_bytes())==P['exp1_sha256']
    spec=importlib.util.spec_from_file_location('pinned_author',source);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    model,tokenizer=bench.load_model_balanced(env['checkpoint'],'cuda:0');model.eval().requires_grad_(False);model.set_attn_implementation('flash_attention_2')
    from memory_efficient_qwen3 import make_memory_efficient_qwen3
    verified,report['source_receipt']=make_memory_efficient_qwen3(R,model,P['candidate_finite_library']['path'],P['candidate_finite_library']['sha256'])
    from bounded_target_ops import target_logprobs,target_seed
    from compiled_logprob_seed import compiled_seed
    from qwen3_projection_cast_boundaries import seed_half
    def diff(a,b):
        assert a.shape==b.shape
        return {'shape':list(a.shape),'exact':bool(torch.equal(a,b)),'max_abs':float((a.float()-b.float()).abs().max()),'different':int((a!=b).sum())}
    vectors={}
    for n in [100,500]:
        record=next(x for x in json.loads((T/'rounding_repair/rollout/results.json').read_bytes())['cases'] if x['output_length']==n)
        ids=torch.tensor(record['input_ids'],device='cuda',dtype=torch.long)[None];pair=ids.repeat(2,1);pair[0,record['eligible_positions']]=tokenizer.eos_token_id
        case={'output_length':n,'input_sha256':record['input_sha256'],'pair_sha256':sha(pair.cpu().numpy().tobytes()),'comparisons':{}};report['cases'].append(case);save();start=time.perf_counter()
        with torch.no_grad():
            native=model(input_ids=pair,attention_mask=None,use_cache=False);z=native.logits[:,record['lengths']['formatted_prompt_tokens']-1:-1];target=pair[:,record['lengths']['formatted_prompt_tokens']:]
            new16,new32=target_logprobs(z,target)
            old16=z.log_softmax(-1).gather(2,target[:,:,None]).squeeze(-1)
            old32=z.float().log_softmax(-1).gather(2,target[:,:,None]).squeeze(-1)
            case['comparisons']['root_lp16']=diff(new16,old16);case['comparisons']['root_lp32']=diff(new32,old32)
            case['comparisons']['root_sum16']=diff(new16.sum(-1),old16.sum(-1));case['comparisons']['root_sum32_64']=diff(new32.double().sum(-1),old32.double().sum(-1));save()
            original,original_check=compiled_seed(z[0].float(),z[1].float(),target[1]);original=original.half()
            full,full_check=seed_half(z[0].float(),z[1].float(),target[1])
            chunk,chunk_check=target_seed(z[0],z[1],target[1])
            parts=[];checks=[]
            for j in range(0,target.shape[1],64):
                value,check=compiled_seed(z[0,j:j+64].float(),z[1,j:j+64].float(),target[1,j:j+64]);parts.append(value.half());checks.append(check)
            original_chunk=torch.cat(parts);del parts
            case['comparisons']['full_fused_half_vs_original']=diff(full,original)
            case['comparisons']['chunk_fused_half_vs_original']=diff(chunk,original)
            case['comparisons']['chunk_original_float_then_half_vs_original']=diff(original_chunk,original)
            case['checks_all_true']=bool(original_check&full_check&chunk_check&torch.stack(checks).all())
            # Save complete small log-prob vectors and exact differing seed entries,
            # including coordinates and both actual values, with whole-tensor hashes.
            for key,value in [('new16',new16),('old16',old16),('new32',new32),('old32',old32)]:vectors[str(n)+'_'+key]=value.cpu().numpy()
            for key,value in [('original',original),('full',full),('chunk',chunk),('original_chunk',original_chunk)]:
                case[key+'_seed_sha256']=sha(value.cpu().numpy().tobytes())
            mask=chunk!=original;vectors[str(n)+'_seed_difference_indices']=mask.nonzero().cpu().numpy();vectors[str(n)+'_seed_new_different']=chunk[mask].cpu().numpy();vectors[str(n)+'_seed_original_different']=original[mask].cpu().numpy()
            np.savez(out/'vectors.npz',**vectors);case['seconds']=time.perf_counter()-start;save()
            del ids,pair,native,z,target,new16,new32,old16,old32,original,full,chunk,original_chunk,mask
    report['status']='complete'
except BaseException:report['status']='failed';report['error']=traceback.format_exc()
finally:save();print(json.dumps(report),flush=True)
