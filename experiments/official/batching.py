"""Input packing for the explicit, separately pinned Qwen3.5 acceleration path.

All attention, finite expressions and metrics remain in their existing modules.
"""
import hashlib
import importlib.util
import json


def make_accelerated_runner(root, model, finite_fa, finite_fla):
    manifest_path=root/'deltatrace/accelerated/sources.json'
    raw=manifest_path.read_bytes();manifest=json.loads(raw)
    assert hashlib.sha256((root/'deltatrace/clean/sources.json').read_bytes()).hexdigest()==manifest['base_clean_sources_sha256']
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    def load(name,filename):
        spec=importlib.util.spec_from_file_location(name,root/'deltatrace/accelerated/qwen35'/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    controller=load('dt_pinned_accelerated_controller','controller.py')
    dynamic=load('dt_pinned_dynamic_finite','dynamic_finite.py')
    runner=dynamic.configure_dynamic_finite(controller.Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,checkpoint_device='cuda'))
    return runner,{'manifest_sha256':hashlib.sha256(raw).hexdigest(),'files':manifest['files']}


def group_cases(cases, batch_size):
    # Input length only; no attribution, answer correctness or metric selection.
    order=sorted(cases,key=lambda c:(c['ids'].shape[1],c['row']['index']))
    return [order[i:i+batch_size] for i in range(0,len(order),batch_size)]


def attribute_batch(runner, cases, eos_token_id, device):
    import torch
    from qwen35_answer_finite import PackedAnswerTargets
    size=len(cases);length=max(c['ids'].shape[1] for c in cases)
    ids=torch.full((2*size,length),eos_token_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids);targets=[];offsets=[]
    for j,c in enumerate(cases):
        n=c['ids'].shape[1];ids[2*j:2*j+2,:n]=c['ids'].to(device)
        ids[2*j,c['eligible']]=eos_token_id;mask[2*j:2*j+2,:n]=1
        targets.append({'target_ids':c['eval_target'][0],'prompt_length':c['prompt_len']})
        offsets.append(list(range(c['gen_len'])))
    selection=PackedAnswerTargets(targets,offsets,length,device)
    roots=[]
    def observe(_module,args,kwargs):
        assert torch.equal(kwargs['input_ids'],ids) and torch.equal(kwargs['attention_mask'],mask)
        roots.append({'input_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            'mask_sha256':hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
            'sample_batch':size,'endpoint_batch':2*size})
    handle=runner.model.register_forward_pre_hook(observe,with_kwargs=True)
    try:signed,detail=runner.attribute(ids,mask,selection,select_output_rows=True,observer=None)
    finally:handle.remove()
    assert len(roots)==1 and bool(torch.isfinite(signed).all())
    for j,c in enumerate(cases):assert bool(signed[j,c['ids'].shape[1]:].eq(0).all())
    return [signed[j,:c['ids'].shape[1]].detach().cpu() for j,c in enumerate(cases)],detail,roots[0]
