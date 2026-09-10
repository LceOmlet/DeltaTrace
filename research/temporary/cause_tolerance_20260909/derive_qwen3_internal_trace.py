"""Expose actual Qwen3 finite normalization coefficients to a passive observer."""
import ast,difflib,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((ROOT/'deltatrace/accelerated/deferred_sources.json').read_bytes())
outputs={}
for name in ('finite','pair'):
    relative=f'deltatrace/accelerated/qwen3/qwen3_deferred_{name}.py'
    raw=(ROOT/relative).read_bytes();assert sha(raw)==manifest['files'][relative]
    source=raw.decode().replace('\r\n','\n')
    if name=='finite':
        replacements=[('validation=None):','validation=None,observer=None):'),
            ('        del m_final\n','        if observer is not None:\n            observer.final(m, m_final, after)\n        del m_final\n'),
            ('            m_mid = m\n','            m_layer_output = m if observer is not None else None\n            m_mid = m\n'),
            ('            if progress is not None:\n',"            if observer is not None:\n                observer.decoder(li,raw1,m_layer_output,m,m_mid,(mqa,mka,mva),(m_x_gate,m_x_up),{'raw_q':mqp,'raw_k':mkp,'norm_q':mqn,'norm_k':mkn,'dq':q_product,'dk':k_product,'dv':mvr,'u':mout})\n            if progress is not None:\n")]
    else:
        replacements=[('from qwen3_deferred_finite import','from qwen3_internal_finite import'),
            ('finite_activity=None):','finite_activity=None,observer=None):'),
            ('finite_activity=finite_activity,validation=validation)','finite_activity=finite_activity,validation=validation,observer=observer)')]
    derived=source
    for old,new in replacements:
        assert derived.count(old)==1,old;derived=derived.replace(old,new)
    ast.parse(derived)
    restored=derived
    for old,new in reversed(replacements):restored=restored.replace(new,old)
    assert restored==source
    output=HERE/f'qwen3_internal_{name}.py';output.write_text(derived,newline='\n')
    outputs[output.name]={'original_path':relative,'original_sha256':sha(raw),'derived_sha256':sha(output.read_bytes()),
                         'diff':''.join(difflib.unified_diff(source.splitlines(True),derived.splitlines(True)))}
(HERE/'qwen3_internal_trace_derivation.json').write_text(json.dumps(outputs,indent=2)+'\n',newline='\n')
source_path=HERE/'trace_qwen3_partial_deletion.py';source=source_path.read_text()
records={r['path']:r for r in json.loads((ROOT/'evidence/export_manifest.json').read_bytes())['artifacts']}
assert sha(source_path.read_bytes())==records[str(source_path.relative_to(ROOT)).replace('\\','/')]['public_sha256']
replacements=[
("    a = p.parse_args()\n","    a = p.parse_args()\n    assert a.phase == 'pilot'\n"),
("from qwen3_trace_pair import", "from qwen3_internal_pair import"),
("    from vendor_fa_finite_runtime import VendorFAFiniteP1\n", "    from vendor_fa_finite_runtime import VendorFAFiniteP1\n    from flash_attn import flash_attn_func\n"),
("'qwen3_trace_derivation.json'","'qwen3_internal_trace_derivation.json'"),
("                def final(_m,args,output):\n",'''                active_attention={}
                for i,layer in enumerate(layers):
                    for coordinate in ('q','k'):
                        def coordinates(_m,args,output,i=i,coordinate=coordinate):
                            retain(f'{i}/raw_'+coordinate,args[0]);retain(f'{i}/norm_'+coordinate,output)
                        handles.append(getattr(layer.self_attn,coordinate+'_norm').register_forward_hook(coordinates))
                    def enter(_m,_args,i=i):
                        assert not active_attention;active_attention['index']=i
                    def leave(_m,_args,_output,i=i):
                        assert active_attention.pop('index')==i
                    handles.append(layer.self_attn.register_forward_pre_hook(enter))
                    handles.append(layer.self_attn.register_forward_hook(leave))
                def profile(frame,event,result):
                    if frame.f_code is flash_attn_func.__code__:
                        i=active_attention['index']
                        if event=='call':
                            for coordinate in ('q','k','v'):retain(f'{i}/core_'+coordinate,frame.f_locals[coordinate])
                        elif event=='return':
                            assert isinstance(result,torch.Tensor);retain(f'{i}/core_output',result)
                def final(_m,args,output):
'''),
("                    with torch.no_grad():\n                        out = model", "                    assert sys.getprofile() is None\n                    sys.setprofile(profile)\n                    with torch.no_grad():\n                        out = model"),
("                finally:\n                    for h in handles: h.remove()\n", "                finally:\n                    sys.setprofile(None)\n                    for h in handles: h.remove()\n"),
("                assert len(values) == 36*7+2\n","                assert len(values) == 36*15+2 and not active_attention\n"),
("def decoder(self,index,raw,mout,minput,mmid,ma_parts,mn_parts):", "def decoder(self,index,raw,mout,minput,mmid,ma_parts,mn_parts,inner):"),
("                    for method,partial in partials.items():\n",'''                    groups=layers[index].self_attn.num_key_value_groups
                    b,t,kh,dim=raw['fa_k'].shape
                    dq=inner['dq'].transpose(1,2)
                    dk=inner['dk'].reshape(b,kh,groups,t,dim).sum(2).transpose(1,2)
                    dv=inner['dv'].reshape(b,kh,groups,t,dim).sum(2).transpose(1,2)
                    u=inner['u'].transpose(1,2)
                    for method,partial in partials.items():
'''),
("                        rec['methods'][method] = {'input_prediction':pin,'output_prediction':pout,'errors':errors}\n",'''                        pqraw=dot(inner['raw_q'],raw['qpre'],'raw_q');pkraw=dot(inner['raw_k'],raw['kpre'],'raw_k')
                        pqnorm=dot(inner['norm_q'],raw['qnorm'],'norm_q');pknorm=dot(inner['norm_k'],raw['knorm'],'norm_k')
                        pqcore=dot(dq,raw['fa_q'],'core_q');pkcore=dot(dk,raw['fa_k'],'core_k');pvcore=dot(dv,raw['fa_v'],'core_v')
                        poutput=dot(u,raw['fa_out'],'core_output');pnative=dot(u.to(torch.float16),raw['fa_out'],'core_output')
                        post={'output_projection':poutput-mix,'output_gate_or_norm_gate':0.,'GDN_upstream_cast':0.,
                              'remaining_mixer':errors['mixer']-(poutput-mix)}
                        internal={'Q_normalization':pqraw-pqnorm,'K_normalization':pkraw-pknorm,
                                  'rotary_rounding':pqnorm+pknorm-pqcore-pkcore,
                                  'finite_core':pqcore+pkcore+pvcore-pnative,'FA_upstream_cast':pnative-poutput,
                                  'compiled_boundary_reconstruction':0.}
                        internal['projection_conv_and_other']=post['remaining_mixer']-sum(internal.values())
                        assert abs(sum(post.values())-errors['mixer'])<1e-7*max(1.,abs(errors['mixer']))
                        assert abs(sum(internal.values())-post['remaining_mixer'])<1e-7*max(1.,abs(post['remaining_mixer']))
                        rec['methods'][method] = {'input_prediction':pin,'output_prediction':pout,'errors':errors,
                                                  'mixer_parts':post,'internal_parts':internal}
''')]
derived=source
for old,new in replacements:
    assert derived.count(old)==1,old;derived=derived.replace(old,new)
ast.parse(derived)
out=HERE/'trace_qwen3_internal_normalization.py';out.write_text(derived,newline='\n')
(HERE/'qwen3_internal_driver_derivation.json').write_text(json.dumps({'source':source_path.name,'source_sha256':sha(source_path.read_bytes()),
    'derived_sha256':sha(out.read_bytes()),'diff':''.join(difflib.unified_diff(source.splitlines(True),derived.splitlines(True)))},indent=2)+'\n',newline='\n')
print(json.dumps({'driver':sha(out.read_bytes()),**{k:v['derived_sha256'] for k,v in outputs.items()}}))
