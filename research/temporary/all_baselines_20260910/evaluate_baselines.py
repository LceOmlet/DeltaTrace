"""Run only remaining algorithms on exact saved DT/FT inputs; checkpoint per case."""
import argparse
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(path,value):
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('environment','preflight','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--methods',nargs='+',required=True)
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--indices',nargs='+',type=int)
    p.add_argument('--mlm',type=Path)
    a=p.parse_args();plan=json.loads((HERE/'protocol.json').read_bytes())
    assert set(a.methods)<=set(plan['new_methods']) and len(a.methods)==len(set(a.methods))
    assert set(a.datasets)<=set(plan['tasks']) and len(a.datasets)==len(set(a.datasets))
    assert sha(HERE/'PROTOCOL.md')==plan['spec_sha256'] and sha(HERE/'inputs.json')==plan['inputs_sha256']
    preflight=json.loads(a.preflight.read_bytes());assert preflight['status']=='passed' and preflight['case_count']==448
    assert preflight['inputs_sha256']==plan['inputs_sha256'] and preflight['source_identity_sha256']==plan['source_identity_sha256']
    env=json.loads(a.environment.read_bytes())['qwen3'];official=Path(env['official_root'])
    assert preflight['environment_sha256']==sha(a.environment)
    assert preflight['checkpoint_receipt_sha256']==env['checkpoint_receipt_sha256']==sha(Path(env['checkpoint_receipt']))
    for record in preflight['weight_files']:
        stat=(Path(env['checkpoint'])/record['name']).stat()
        assert stat.st_size==record['bytes'] and stat.st_mtime_ns==record['mtime_ns']
    for name,digest in plan['baseline_sources'].items():
        assert hashlib.sha256((official/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,name
    sources=json.loads((ROOT/'deltatrace/clean/sources.json').read_bytes())
    for family in sources['models'].values():
        for name,record in family['files'].items():assert sha(ROOT/name)==record['sha256']
    if 'REAGENT' in a.methods:
        assert a.mlm is not None
        for record in plan['mlm']['files']:
            path=a.mlm/record['name'];assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
    inputs=json.loads((HERE/'inputs.json').read_bytes())['cases']
    selected=[r for r in inputs if r['dataset'] in a.datasets and (a.indices is None or r['index'] in a.indices)]
    if a.indices is not None:assert len(selected)==len(a.indices)*len(a.datasets)
    identity=dict(protocol_sha256=sha(HERE/'protocol.json'),driver_sha256=sha(Path(__file__)),
        adapter_sha256=sha(HERE/'baseline_adapters.py'),preflight_sha256=sha(a.preflight),inputs_sha256=plan['inputs_sha256'])
    pending=[];resumed=0
    for item in selected:
        for method in a.methods:
            folder=a.output/method/item['dataset']/f'{item["index"]:03d}'
            if (folder/'results.json').exists():
                done=json.loads((folder/'results.json').read_bytes())
                assert done['status']=='complete' and done['identity']==identity and done['input_sha256']==item['input_sha256']
                assert done['vectors_sha256']==sha(folder/'vectors.npz');resumed+=1
            else:pending.append((item,method,folder))
    print(json.dumps(dict(status='selected',pending=len(pending),resumed=resumed,methods=a.methods,datasets=a.datasets)),flush=True)
    if not pending:return
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false');os.environ.setdefault('MACA_PATH','/opt/maca')
    sys.path.insert(0,str(official))
    import torch
    from exp.exp2 import run_exp as author
    import shared_utils
    from baseline_adapters import native_attributor,calculate,validate_target
    torch.set_num_threads(4);torch.manual_seed(plan['seed']);torch.backends.cuda.matmul.allow_tf32=False
    model,tokenizer=author.load_model(env['checkpoint'],'cuda:0');model.eval().requires_grad_(False)
    assert sha(Path(inspect.getfile(type(model))))==env['native_model_sha256']
    original_forwards={name:type(module).forward for name,module in model.named_modules()}
    original_bound_forwards={name:module.forward for name,module in model.named_modules()}
    modeling=importlib.import_module(type(model).__module__)
    original_attention=getattr(modeling,'ALL_ATTENTION_FUNCTIONS',None)
    original_eager=getattr(modeling,'eager_attention_forward',None)
    original_generate=model.generate
    def forbid_generation(*args,**kwargs):raise RuntimeError('Generating a new response is outside this frozen run')
    model.generate=forbid_generation
    tracers={m:native_attributor(m,model,tokenizer,mlm_path=a.mlm) for m in a.methods}
    def verify_model_restored():
        for name,module in model.named_modules():
            assert type(module).forward is original_forwards[name] and module.forward==original_bound_forwards[name]
        assert getattr(modeling,'eager_attention_forward',None) is original_eager
        # The native LRP context restores the same mapping entries in a new dict.
        # Restore the original mapping object as well, without changing entries.
        current=getattr(modeling,'ALL_ATTENTION_FUNCTIONS',None)
        if original_attention is not None:
            assert dict(current)==dict(original_attention)
            modeling.ALL_ATTENTION_FUNCTIONS=original_attention
    try:
        for item,method,folder in pending:
            verify_model_restored();tracer=tracers[method]
            tracer.target_response(item['prompt'],item['target'])
            ids=torch.cat((tracer.prompt_ids,tracer.generation_ids),dim=1)
            assert ids[0].tolist()==item['input_ids'] and list(tracer.user_prompt_indices)==item['user_positions']
            weights=validate_target(item['target_weights'],tracer.generation_tokens)
            assert ids.shape[1]==item['prompt_length']+item['target_length']
            folder.mkdir(parents=True,exist_ok=True)
            attempt=len(list(folder.glob('failed_attempt_*.json')))
            call=dict(dataset=item['dataset'],index=item['index'],method=method,input_sha256=item['input_sha256'],
                identity=identity,target_weights=weights,target_mode=item['target_mode'],prompt_length=item['prompt_length'],
                user_prompt_length=len(item['user_positions']),target_length=item['target_length'],status='entered',
                calls=dict(model_forwards=0,unperturbed_causal_prefixes=0,perturbed_prefixes=0),generation_calls=0,
                spacy_pipeline=shared_utils.nlp.pipe_names,seed=plan['seed'])
            torch.manual_seed(plan['seed']+1000*list(plan['tasks']).index(item['dataset'])+item['index'])
            audit_calls=[];original_compute={};expected_embed=None
            if method=='AttnLRP':
                with torch.no_grad():expected_embed=model.get_input_embeddings()(ids).detach()
            def observe_forward(_module,args,kwargs):
                call['calls']['model_forwards']+=1
                if method=='IFR':assert torch.equal(kwargs['input_ids'],ids)
                elif method=='AttnLRP':assert torch.equal(kwargs['inputs_embeds'].detach(),expected_embed)
            hook=model.register_forward_pre_hook(observe_forward,with_kwargs=True)
            if method in ('Perturbation','CLP','REAGENT'):
                for name in ('compute_logprob_response_given_prompt','compute_kl_response_given_prompt'):
                    original_compute[name]=getattr(tracer,name)
                    def observed(prefix,response,_compute=original_compute[name]):
                        n=int(prefix.shape[1]);m=int(response.shape[1])
                        assert prefix.shape[0]==response.shape[0]==1 and item['prompt_length']<=n<n+m<=ids.shape[1]
                        assert torch.equal(response,ids[:,n:n+m]),'Perturbation sink tokens differ from the fixed causal input'
                        diffs=torch.where(prefix[0]!=ids[0,:n])[0].tolist()
                        allowed=set(item['user_positions'])|set(range(item['prompt_length'],n))
                        assert set(diffs)<=allowed,'Perturbation altered wrapper tokens'
                        call['calls']['perturbed_prefixes' if diffs else 'unperturbed_causal_prefixes']+=1
                        audit_calls.append(dict(prefix_tokens=n,response_tokens=m,changed_positions=diffs))
                        return _compute(prefix,response)
                    setattr(tracer,name,observed)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            try:
                signed,native,raw=calculate(method,tracer,item)
                torch.cuda.synchronize();call['seconds']=time.perf_counter()-started
                call['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                count=len(item['user_positions']);assert signed.shape==native.shape==(count+item['target_length'],)
                assert np.isfinite(signed).all() and np.isfinite(native).all()
                assert tracer.prompt_ids.shape[1]==item['prompt_length'] and tracer.generation_ids.shape[1]==item['target_length']
                assert torch.cat((tracer.prompt_ids,tracer.generation_ids),dim=1)[0].tolist()==item['input_ids']
                verify_model_restored()
                if method in ('Perturbation','CLP','REAGENT'):
                    assert call['calls']['unperturbed_causal_prefixes']>0
                    call['causal_prefix_audit']=audit_calls
                vectors=dict(prompt_signed=signed[:count],prompt_native_positive_row_normalized=native[:count],**raw)
                temporary=folder/'vectors.partial.npz';np.savez_compressed(temporary,**vectors);temporary.replace(folder/'vectors.npz')
                call.update(status='complete',vectors_sha256=sha(folder/'vectors.npz'),complete_input_verified=True,model_restored=True)
                write_json(folder/'results.json',call)
                print(json.dumps(dict(status='complete',dataset=item['dataset'],index=item['index'],method=method,seconds=call['seconds'])),flush=True)
            except BaseException:
                call.update(status='failed',seconds=time.perf_counter()-started,error=traceback.format_exc())
                write_json(folder/f'failed_attempt_{attempt:03d}.json',call)
                raise
            finally:
                hook.remove()
                for name,compute in original_compute.items():setattr(tracer,name,compute)
            del raw,signed,native,vectors,ids,expected_embed
            gc.collect()
    finally:
        model.generate=original_generate

if __name__=='__main__':main()
