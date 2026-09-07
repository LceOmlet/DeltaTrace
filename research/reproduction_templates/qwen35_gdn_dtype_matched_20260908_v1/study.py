"""Recover full vectors/gradient-limit evidence using saved native layer inputs.

Load only the exact original GDN class and its nine checkpoint tensors. No full
model load/forward, replacement model, generation or quality query is performed.
"""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
import ast,hashlib,inspect,io,json,sys,threading,time,traceback,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ['TRITON_CACHE_DIR']=p['compiler_cache'];os.environ['TORCHINDUCTOR_CACHE_DIR']=p['reuse_inductor_cache']
r={'status':'running','protocol':p,'full_model_loads':0,'full_model_forwards':0,'generation_calls':0,'quality_queries':0,
    'native_module_load_attempts':0,'finite_attempts':0,'native_forward_attempts':0,'native_backward_attempts':0,
    'calls':[],'artifacts':[]};start=time.perf_counter()
def save():
    path=HERE/'results.partial';path.write_text(json.dumps(r,indent=2));path.replace(HERE/'results.json')
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    parent=Path(p['parent_directory']);raw=(parent/'results.json').read_bytes();assert sha(raw)==p['parent_sha256']
    previous=json.loads(raw);assert previous['status']=='failed' and previous['finite_completed']==2
    tree=ast.parse((parent/'study.py').read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sources')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<unchanged-native-source-audit>','exec'))
    r['sources_before']=sources();assert r['sources_before']['sha256']==p['source_tree_sha256']
    cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{x.name:[x.stat().st_size,x.stat().st_mtime_ns] for x in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==previous['weight_stats_before']
    epfile=parent/'native_first_GDN_endpoints.pt';assert sha(epfile.read_bytes())==p['endpoints_sha256']
    import numpy as np
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.utils import ContextManagers
    from transformers.core_model_loading import _materialize_copy
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import causal_conv1d
    from causal_conv1d.causal_conv1d_interface import CausalConv1dFn
    from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback,resolve_native_gdn_forward
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    verify_native_sources(p['native_stage_source_sha256'])
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False)
    cfg._attn_implementation='flash_attention_2'
    cls=native.Qwen3_5ForConditionalGeneration
    r['meta_model_constructions']=1;r['native_module_load_attempts']+=1;save()
    with ContextManagers(cls.get_init_context(torch.bfloat16,False,False,False)):
        container=cls(cfg)
    plan=container._get_dtype_plan(torch.bfloat16);assert plan=={}
    module=container.model.language_model.layers[0].linear_attn
    desired_dtypes={k:v.dtype for k,v in module.state_dict().items()}
    assert set(desired_dtypes.values())=={torch.bfloat16}
    r['official_loading_policy']={'requested_dtype':'torch.bfloat16','dtype_plan':plan,
        'empty_parameter_dtypes':{k:str(v) for k,v in desired_dtypes.items()},
        'context':'Original model class get_init_context; metadata only, all parameters remain meta until nine GDN tensors are loaded.',
        'materialization':'Original Transformers _materialize_copy with the initialized parameter dtype.'}
    index=json.loads((cp/'model.safetensors.index.json').read_bytes())['weight_map'];prefix='model.language_model.layers.0.linear_attn.'
    names={name:shard for name,shard in index.items() if name.startswith(prefix)}
    assert len(names)==9 and {n.removeprefix(prefix) for n in names}==set(module.state_dict())
    state={};receipts={}
    for shard in sorted(set(names.values())):
        with safe_open(cp/shard,framework='pt',device='cpu') as sf:
            for name in names:
                if names[name]!=shard:continue
                raw_value=sf.get_tensor(name)
                value=_materialize_copy(sf.get_slice(name),device='cpu',dtype=desired_dtypes[name.removeprefix(prefix)])
                state[name.removeprefix(prefix)]=value
                receipts[name]={'shard':shard,'shape':list(value.shape),'dtype':str(value.dtype),
                    'tensor_bytes_sha256':sha(value.view(torch.uint8).numpy().tobytes()),
                    'checkpoint_dtype':str(raw_value.dtype),'checkpoint_tensor_sha256':sha(raw_value.view(torch.uint8).numpy().tobytes())}
    module.load_state_dict(state,strict=True,assign=True);module.eval().requires_grad_(False).to('cuda');del state,container
    assert 'forward' not in module.__dict__ and module.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    original_forward=type(module).forward
    r['weight_tensor_receipts']=receipts;r['captured_body']={'name':resolve_native_gdn_forward(type(module)).__qualname__,
        'line':resolve_native_gdn_forward(type(module)).__code__.co_firstlineno}
    saved=torch.load(epfile,map_location='cpu',weights_only=True)
    c={k:None if v is None else v.to('cuda') for k,v in saved['values'].items()}
    ep={k:v.to('cuda') for k,v in saved['endpoints'].items()};scale=saved['scale']
    assert c['mask'].sum(1).tolist()==[605,605,368,368]
    seed=c['output'][1::2].contiguous()*c['mask'][1::2,:,None]
    finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False)
    def persist(name,values):
        path=HERE/(name+'.npz');np.savez_compressed(path,**{k:v.detach().float().cpu().numpy() for k,v in values.items()})
        r['artifacts'].append({'file':path.name,'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size});save()
    def run_finite(label,cc,ee):
        assert r['finite_attempts']<2;r['finite_attempts']+=1;save()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();tick=time.perf_counter()
        with torch.no_grad():out,_=gdn_finite_pullback(module,cc,ee,seed,scale,finite_fla,False)
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick;assert torch.isfinite(out).all()
        r['calls'].append({'kind':label,'seconds':elapsed,'before_bytes':before,'peak_bytes':torch.cuda.max_memory_allocated()});save()
        return out
    finite=run_finite('saved_actual_endpoints',c,ep)
    persist('actual',{'finite':finite,'input0':c['input'][0::2],'input1':c['input'][1::2],
        'output0':c['output'][0::2],'output1':c['output'][1::2],'mask':c['mask'][1::2]});del finite
    # A graph-node hook observes execution on autograd workers. A main-thread
    # sys.setprofile hook is deliberately NOT used as a backward-call oracle.
    main_thread=threading.get_ident();node_events=[];handles=[]
    capture=NativeGDNCapture(module,'cpu');x=c['input'].detach().requires_grad_(True)
    r['native_forward_attempts']+=1;save();torch.cuda.synchronize();tick=time.perf_counter()
    with capture:y=module(x,attention_mask=c['mask'])
    visited=set();queue=[y.grad_fn];nodes=[]
    while queue:
        node=queue.pop()
        if node is None or node in visited:continue
        visited.add(node)
        if isinstance(node,CausalConv1dFn._backward_cls):nodes.append(node)
        queue.extend(n for n,_ in node.next_functions if n is not None)
    assert len(nodes)==1
    def observe(grad_inputs,grad_outputs):
        node_events.append({'thread_id':threading.get_ident(),'main_thread':main_thread,'returned_input_gradient':grad_inputs[0] is not None})
    handles.append(nodes[0].register_hook(observe))
    dy=torch.zeros_like(y);dy[1::2]=seed;r['native_backward_attempts']+=1;save()
    try:native_gradient,=torch.autograd.grad(y,x,dy)
    finally:
        for handle in handles:handle.remove()
    torch.cuda.synchronize();r['calls'].append({'kind':'native_GDN_forward_and_backward_with_capture','seconds':time.perf_counter()-tick})
    r['native_conv_backward_node_events']=node_events;r['native_replay_capture_calls']=capture.calls;save()
    assert len(node_events)==1 and node_events[0]['returned_input_gradient']
    assert capture.calls=={'module':1,'conv':1,'FLA':1,'stage':1}
    def relative(a,b):
        a=a.float().numpy().astype(np.float64);b=b.float().numpy().astype(np.float64)
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        norm=float(np.linalg.norm(a.ravel()));error=float(np.linalg.norm((b-a).ravel()))
        return {'relative_L2':error/norm if norm else None,'max_abs':float(np.abs(a-b).max()),'exact_equal':bool(np.array_equal(a,b))}
    r['native_replay_vs_real_root']={group:{k:relative(saved[group][k],v) for k,v in values.items() if v is not None}
        for group,values in [('values',capture.values),('endpoints',capture.endpoints)]}
    save()
    persist('native',{'gradient':native_gradient[1::2],'replay_output':y.detach()})
    del native_gradient,x,y,dy,visited,queue,nodes
    duplicate=lambda x:None if x is None else x[1::2].repeat_interleave(2,dim=0).to('cuda')
    equal_c={k:duplicate(v) for k,v in capture.values.items()};equal_ep={k:duplicate(v) for k,v in capture.endpoints.items()}
    equal=run_finite('equal_endpoints_limit',equal_c,equal_ep);persist('equal',{'finite':equal})
    r['compiler_counters']={k:dict(v) for k,v in torch._dynamo.utils.counters.items()}
    r['sources_after']=sources();assert r['sources_before']==r['sources_after']
    r['weight_stats_after']=stats();assert r['weight_stats_before']==r['weight_stats_after']
    assert type(module).forward is original_forward and 'forward' not in module.__dict__
    r['status']='dtype_matched_GDN_vectors_and_native_gradient_limit_recovered'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']+[x['file'] for x in r['artifacts']]:z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'finite_attempts':r['finite_attempts'],'native_backward_attempts':r['native_backward_attempts'],'error':r.get('error')}),flush=True)
