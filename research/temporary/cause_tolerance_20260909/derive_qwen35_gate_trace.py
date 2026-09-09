import ast,difflib,hashlib,json
from pathlib import Path

here=Path(__file__).resolve().parent
path=here/'trace_qwen35_partial_deletion.py'; source=path.read_text()
assert hashlib.sha256(path.read_bytes()).hexdigest()=='686d34ffa357469cc008e6645e7c932a2369978613ae065b825e06ddb58647ab'
replacements=[
("    from finite_fla_gpu import verify_native_sources\n", "    from finite_fla_gpu import verify_native_sources\n    from qwen35_decoder_finite import _linear_transpose\n    from transformers.integrations.flash_attention import flash_attention_forward\n"),
("    a = p.parse_args()\n", "    a = p.parse_args()\n    assert a.phase == 'pilot'\n"),
("                def final(_m, args, output):\n", '''                attention_indices = {id(layer.self_attn): i for i,layer in enumerate(layers) if layer.block_type=='full_attention'}
                for i,layer in enumerate(layers):
                    if layer.block_type=='full_attention':
                        def gate(_m,_args,output,i=i,layer=layer):
                            h=layer.self_attn.config.num_attention_heads; d=layer.self_attn.head_dim
                            retain(f'{i}/gate',output.view(2,n,h,2*d)[...,d:])
                        def product(_m,args,i=i,layer=layer):
                            retain(f'{i}/product',args[0].view(2,n,layer.self_attn.config.num_attention_heads,layer.self_attn.head_dim))
                        handles.append(layer.self_attn.q_proj.register_forward_hook(gate))
                        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(product))
                    else:
                        def norm_gate(_m,args,output,i=i,layer=layer):
                            h=layer.linear_attn.num_v_heads; d=layer.linear_attn.head_v_dim
                            assert len(args)==2 and args[0].numel()==args[1].numel()==output.numel()==2*n*h*d
                            retain(f'{i}/core',args[0].reshape(2,n,h,d)); retain(f'{i}/gate',args[1].reshape(2,n,h,d))
                            retain(f'{i}/product',output.reshape(2,n,h,d))
                        handles.append(layer.linear_attn.norm.register_forward_hook(norm_gate))
                def profile(frame,event,result):
                    if frame.f_code is flash_attention_forward.__code__ and event=='return' and result is not None:
                        module=frame.f_locals['module']; assert id(module) in attention_indices
                        retain(f'{attention_indices[id(module)]}/core',result[0])
                def final(_m, args, output):
'''),
("                    with torch.no_grad():\n                        out = model", "                    assert sys.getprofile() is None\n                    sys.setprofile(profile)\n                    with torch.no_grad():\n                        out = model"),
("                finally:\n                    for handle in handles: handle.remove()\n", "                finally:\n                    sys.setprofile(None)\n                    for handle in handles: handle.remove()\n"),
("                assert len(values) == 32 * 7 + 2\n", "                assert len(values) == 32 * 10 + 2\n"),
("                    for method, partial in partials.items():\n", '''                    module=layers[index].self_attn if layers[index].block_type=='full_attention' else layers[index].linear_attn
                    inner=terms['mixer']; mmid=terms['m_mixer_output']
                    if layers[index].block_type=='full_attention':
                        h=module.config.num_attention_heads; dim=module.head_dim
                        projection=_linear_transpose(mmid,module.o_proj.weight).reshape(selection.batch,n,h,dim)
                        core=c['attention_output']; gate=c['q_proj_output'].view(2,n,h,2*dim)[...,dim:]
                        product=c['o_proj_input'].reshape(2,n,h,dim)
                        mcore=inner['mcontent'].transpose(1,2); mraw=mcore; mgate=inner['mgate']
                    else:
                        h=module.num_v_heads; dim=module.head_v_dim
                        projection=inner['mnorm'].reshape(selection.batch,n,h,dim)
                        core=e['o']; gate=c['z']; product=c['norm_output'].reshape(2,n,h,dim)
                        mcore=inner['mo_native']; mraw=inner['mo_before_cast']; mgate=inner['mz']
                    for method, partial in partials.items():
'''),
("                        rec['methods'][method] = {'input_prediction': pin, 'output_prediction': pout, 'errors': errors}\n", '''                        pcore=dot(mcore,core,'core'); praw=dot(mraw,core,'core'); pgate=dot(mgate,gate,'gate')
                        pproduct=dot(projection,product,'product')
                        post={'output_projection':pproduct-mix,'output_gate_or_norm_gate':praw+pgate-pproduct,
                              'GDN_upstream_cast':pcore-praw,
                              'remaining_mixer':errors['mixer']-(pproduct-mix)-(praw+pgate-pproduct)-(pcore-praw)}
                        assert abs(sum(post.values())-errors['mixer'])<1e-7*max(1.,abs(errors['mixer']))
                        rec['methods'][method] = {'input_prediction':pin,'output_prediction':pout,'errors':errors,
                                                  'mixer_parts':post,'mixer_boundary_predictions':{'core_raw':praw,'core_native':pcore,'gate':pgate,'product':pproduct,'output':mix}}
''')]
derived=source
for old,new in replacements:
    assert derived.count(old)==1,old
    derived=derived.replace(old,new)
ast.parse(derived)
out=here/'trace_qwen35_output_gate.py';out.write_text(derived,newline='\n')
info={'source':'trace_qwen35_partial_deletion.py','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'derived_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
      'diff':''.join(difflib.unified_diff(source.splitlines(True),derived.splitlines(True))),
      'scope':'Read-only native norm/qproj/oproj hooks and existing official FA-interface observation; additional own linear-transpose diagnostic only. Original DT/FA/FLA unchanged.'}
(here/'qwen35_output_gate_derivation.json').write_text(json.dumps(info,indent=2)+'\n',newline='\n')
print(json.dumps({k:v for k,v in info.items() if k!='diff'}))
