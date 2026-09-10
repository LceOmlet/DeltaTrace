"""Source-pinned read-only split of the remaining mixer discrepancy."""
import ast,difflib,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((ROOT/'deltatrace/accelerated/deferred_sources.json').read_bytes())
controller_path='deltatrace/accelerated/qwen35/controller_deferred.py'
raw=(ROOT/controller_path).read_bytes();assert sha(raw)==manifest['files'][controller_path]
controller=raw.decode().replace('\r\n','\n')
before="            if focused:observer.decoder(i,d,c,e,m,new,terms)\n"
after="            if focused and is_fa:terms['position_embeddings']=kw['position_embeddings']\n"+before
assert controller.count(before)==1
traced=controller.replace(before,after);ast.parse(traced)
(HERE/'qwen35_internal_trace_controller.py').write_text(traced,newline='\n')
source_path=HERE/'trace_qwen35_output_gate.py';source=source_path.read_text()
assert sha(source_path.read_bytes())=='d0112904984ec4a25616293c7f84a771aef08e71dd38d252218acb3e57d4bfd6'
replacements=[
("import inspect\n","import inspect\nimport importlib\n"),
("    from qwen35_decoder_finite import _linear_transpose\n", "    from qwen35_decoder_finite import _linear_transpose, _partial_rotation_transpose\n    from qwen35_gdn_finite import _l2_pullback\n    from signed_secant_rules import rmsnorm_secant_pullback\n    from qwen35_internal_trace_controller import Qwen35DenseFiniteRunner as TraceRunner\n"),
("        runner, report['baseline_sources'] = make_deferred_qwen35(a.release, model, fa, None)\n", "        runner, report['baseline_sources'] = make_deferred_qwen35(a.release, model, fa, None)\n        trace_runner=TraceRunner.__new__(TraceRunner)\n        trace_runner.__dict__.update(runner.__dict__)\n        assert trace_runner.__dict__==runner.__dict__\n        derivation=json.loads((Path(__file__).parent/'qwen35_internal_trace_derivation.json').read_bytes())\n        assert sha((a.release/derivation['controller']['original_path']).read_bytes())==derivation['controller']['original_sha256']\n        assert sha((Path(__file__).parent/'qwen35_internal_trace_controller.py').read_bytes())==derivation['controller']['derived_sha256']\n        report['observer_derivation']=derivation\n"),
("                for i,layer in enumerate(layers):\n", "                active_gdn={}\n                chunk=importlib.import_module('fla.ops.gated_delta_rule.chunk')\n                stage_code=inspect.unwrap(chunk.chunk_gated_delta_rule_fwd).__code__\n                fla_codes={inspect.unwrap(layer.linear_attn.chunk_gated_delta_rule).__code__ for layer in layers if layer.block_type=='linear_attention'}\n                assert len(fla_codes)==1\n                for i,layer in enumerate(layers):\n"),
("                        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(product))\n", "                        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(product))\n                        for coordinate in ('q','k'):\n                            def norm_coordinates(_m,args,output,i=i,coordinate=coordinate):\n                                retain(f'{i}/raw_'+coordinate,args[0]);retain(f'{i}/norm_'+coordinate,output)\n                            handles.append(getattr(layer.self_attn,coordinate+'_norm').register_forward_hook(norm_coordinates))\n"),
("                        handles.append(layer.linear_attn.norm.register_forward_hook(norm_gate))\n", "                        handles.append(layer.linear_attn.norm.register_forward_hook(norm_gate))\n                        def gdn_enter(_m,_args,i=i):\n                            assert not active_gdn;active_gdn['index']=i\n                        def gdn_leave(_m,_args,_output,i=i):\n                            assert active_gdn.pop('index')==i\n                        handles.append(layer.linear_attn.register_forward_pre_hook(gdn_enter))\n                        handles.append(layer.linear_attn.register_forward_hook(gdn_leave))\n"),
("                def profile(frame,event,result):\n", "                def profile(frame,event,result):\n                    if active_gdn:\n                        i=active_gdn['index']\n                        if frame.f_code in fla_codes and event=='call':\n                            for coordinate in ('q','k'):retain(f'{i}/raw_'+coordinate,frame.f_locals[coordinate])\n                            retain(f'{i}/core_g',frame.f_locals['g'])\n                        if frame.f_code is stage_code and event=='return' and result is not None:\n                            for coordinate in ('q','k'):retain(f'{i}/norm_'+coordinate,frame.f_locals[coordinate])\n                            for coordinate in ('v','beta'):retain(f'{i}/core_'+coordinate,frame.f_locals[coordinate])\n                    if frame.f_code is flash_attention_forward.__code__ and event=='call':\n                        i=attention_indices[id(frame.f_locals['module'])]\n                        for coordinate,name in [('q','query'),('k','key'),('v','value')]:retain(f'{i}/core_'+coordinate,frame.f_locals[name])\n"),
("                assert len(values) == 32 * 10 + 2\n", "                assert len(values) == 32 * 17 + 2 and not active_gdn\n"),
("                    for method, partial in partials.items():\n", '''                    fa_block=layers[index].block_type=='full_attention'
                    coefficients=inner['coeff']
                    if fa_block:
                        groups=module.num_key_value_groups;kh=h//groups
                        mqcore=coefficients['dq'].float()
                        mkcore=coefficients['dk'].float().reshape(selection.batch,kh,groups,n,dim).sum(2)
                        mvcore=coefficients['dv'].float().reshape(selection.batch,kh,groups,n,dim).sum(2)
                        cos,sin=terms['position_embeddings']
                        mqnorm=_partial_rotation_transpose(mqcore,cos[1::2],sin[1::2]).transpose(1,2)
                        mknorm=_partial_rotation_transpose(mkcore,cos[1::2],sin[1::2]).transpose(1,2)
                        mqraw=rmsnorm_secant_pullback(c['q_norm_input'][0::2].float(),c['q_norm_input'][1::2].float(),1+module.q_norm.weight.float(),mqnorm,module.q_norm.eps)
                        mkraw=rmsnorm_secant_pullback(c['k_norm_input'][0::2].float(),c['k_norm_input'][1::2].float(),1+module.k_norm.weight.float(),mknorm,module.k_norm.eps)
                        query_gate=torch.cat((mqraw,mgate),dim=-1).reshape(selection.batch,n,h*2*dim)
                        reconstructed=(_linear_transpose(query_gate,module.q_proj.weight)
                                      +_linear_transpose(mkraw.reshape(selection.batch,n,kh*dim),module.k_proj.weight)
                                      +_linear_transpose(mvcore.transpose(1,2).reshape(selection.batch,n,kh*dim),module.v_proj.weight))
                        boundary_difference=terms['m_mixer_input']-reconstructed
                        reconstruction={'relative_L2':float(boundary_difference.norm()/terms['m_mixer_input'].norm()),
                                        'max_absolute':float(boundary_difference.abs().max())}
                    else:
                        mqnorm=coefficients['q'];mknorm=coefficients['k']
                        mqraw=_l2_pullback(c['raw_q'][0::2],c['raw_q'][1::2],mqnorm)
                        mkraw=_l2_pullback(c['raw_k'][0::2],c['raw_k'][1::2],mknorm)
                        repeat=module.num_v_heads//module.num_k_heads
                        assert torch.equal(mqraw.reshape(selection.batch,n,module.num_k_heads,repeat,dim).sum(3),inner['mq'])
                        assert torch.equal(mkraw.reshape(selection.batch,n,module.num_k_heads,repeat,dim).sum(3),inner['mk'])
                        reconstruction={'grouped_coefficients_exact':True}
                    rec['normalization_coefficient_reconstruction']=reconstruction
                    for method, partial in partials.items():
'''),
("                        rec['methods'][method] = {'input_prediction':pin", '''                        if fa_block:
                            pqraw=dot(mqraw,c['q_norm_input'],'raw_q');pkraw=dot(mkraw,c['k_norm_input'],'raw_k')
                            pqnorm=dot(mqnorm,c['q_norm_output'],'norm_q');pknorm=dot(mknorm,c['k_norm_output'],'norm_k')
                            pqcore=dot(mqcore,c['query'],'core_q');pkcore=dot(mkcore,c['key'],'core_k');pvcore=dot(mvcore,c['value'],'core_v')
                            pnative=dot(mcore.to(torch.bfloat16),core,'core')
                            internal={'Q_normalization':pqraw-pqnorm,'K_normalization':pkraw-pknorm,
                                      'rotary_rounding':pqnorm+pknorm-pqcore-pkcore,
                                      'finite_core':pqcore+pkcore+pvcore-pnative,'FA_upstream_cast':pnative-pcore,
                                      'compiled_boundary_reconstruction':dot(boundary_difference,c['input'],'input_norm_output')}
                        else:
                            pqraw=dot(mqraw,c['raw_q'],'raw_q');pkraw=dot(mkraw,c['raw_k'],'raw_k')
                            pqnorm=dot(mqnorm,e['q'],'norm_q');pknorm=dot(mknorm,e['k'],'norm_k')
                            pvcore=dot(coefficients['v'],e['v'],'core_v');pgcore=dot(coefficients['g'],e['raw_g'],'core_g');pbcore=dot(coefficients['beta'],e['beta'],'core_beta')
                            internal={'Q_normalization':pqraw-pqnorm,'K_normalization':pkraw-pknorm,
                                      'rotary_rounding':0.,'finite_core':pqnorm+pknorm+pvcore+pgcore+pbcore-pcore,
                                      'FA_upstream_cast':0.,'compiled_boundary_reconstruction':0.}
                        internal['projection_conv_and_other']=post['remaining_mixer']-sum(internal.values())
                        assert abs(sum(internal.values())-post['remaining_mixer'])<1e-7*max(1.,abs(post['remaining_mixer']))
                        rec['methods'][method] = {'input_prediction':pin'''),
("'mixer_parts':post,'mixer_boundary_predictions':", "'mixer_parts':post,'internal_parts':internal,'mixer_boundary_predictions':"),
("lambda: runner.attribute(pair, mask, selection, observer=obs)","lambda: trace_runner.attribute(pair, mask, selection, observer=obs)")]
derived=source
for before,after in replacements:
    assert derived.count(before)==1,before
    derived=derived.replace(before,after)
ast.parse(derived)
out=HERE/'trace_qwen35_internal_normalization.py';out.write_text(derived,newline='\n')
report={'controller':{'original_path':controller_path,'original_sha256':sha(raw),
                      'derived_sha256':sha((HERE/'qwen35_internal_trace_controller.py').read_bytes()),
                      'diff':''.join(difflib.unified_diff(controller.splitlines(True),traced.splitlines(True)))},
        'driver':{'source':source_path.name,'source_sha256':sha(source_path.read_bytes()),'derived_sha256':sha(out.read_bytes()),
                  'diff':''.join(difflib.unified_diff(source.splitlines(True),derived.splitlines(True)))},
        'scope':'Original native/model/FA/FLA code untouched. Own controller only exposes actual RoPE operands to observer. Read-only capture of normalized Q/K and actual core inputs. Existing finite normalization re-evaluation and linear transposes are diagnostic work, with compiled boundary discrepancy separate.'}
(HERE/'qwen35_internal_trace_derivation.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
print(json.dumps({k:v['derived_sha256'] for k,v in report.items() if isinstance(v,dict)}))
