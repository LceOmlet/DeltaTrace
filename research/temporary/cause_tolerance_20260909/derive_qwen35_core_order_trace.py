"""One uniform reversed core allocation, tested locally before a whole-method trial."""
import ast,difflib,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
path='deltatrace/accelerated/qwen35/controller_deferred.py';raw=(ROOT/path).read_bytes()
assert sha(raw)==json.loads((ROOT/'deltatrace/accelerated/deferred_sources.json').read_bytes())['files'][path]
source=raw.decode().replace('\r\n','\n')
old='            if focused:observer.decoder(i,d,c,e,m,new,terms)\n'
new="            if focused:terms['native_scale']=scale\n            if focused and is_fa:\n                terms['position_embeddings']=kw['position_embeddings']\n                terms['native_lse']=lse\n"+old
assert source.count(old)==1
controller=source.replace(old,new);ast.parse(controller)
(HERE/'qwen35_core_order_trace_controller.py').write_text(controller,newline='\n')
sp=HERE/'trace_qwen35_internal_normalization.py';s=sp.read_text()
assert sha(sp.read_bytes())=='f702c1c18fd6bc11ca01b330678d564577b4406d7c4286430b81e65770642984'
replacements=[
 ('qwen35_internal_trace_controller','qwen35_core_order_trace_controller'),
 ('qwen35_internal_trace_derivation.json','qwen35_core_order_trace_derivation.json'),
 ('from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256','from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256, RightPaddedLengths'),
 ("                    rec['normalization_coefficient_reconstruction']=reconstruction\n",'''                    if fa_block:
                        lse=terms['native_lse']
                        ops={'q0':c['query'][1::2],'q1':c['query'][0::2],'k0':c['key'][1::2],'k1':c['key'][0::2],
                             'v0':c['value'][1::2],'u':inner['mcontent'],'lse0':lse[1::2],'lse1':lse[0::2]}
                        alternative=fa(ops,module.scaling,RightPaddedLengths([n],n,'cuda'))
                        alt_q=alternative['dq'].float()
                        alt_k=alternative['dk'].float().reshape(selection.batch,kh,groups,n,dim).sum(2)
                        alt_v=alternative['dv'].float().reshape(selection.batch,kh,groups,n,dim).sum(2)
                    else:
                        permutation=torch.arange(2*selection.batch,device='cuda')^1
                        assert all(value.shape[0]==2*selection.batch for value in e.values())
                        reversed_endpoints={name:value.index_select(0,permutation) for name,value in e.items()}
                        alternative=runner.finite_fla(reversed_endpoints,inner['mo_native'],terms['native_scale'])
                        alt_q,alt_k,alt_v=[alternative[k] for k in ('q','k','v')]
                        del reversed_endpoints
                    assert all(torch.isfinite(v).all() for v in alternative.values())
                    rec['normalization_coefficient_reconstruction']=reconstruction
'''),
 ("                        internal['projection_conv_and_other']=post['remaining_mixer']-sum(internal.values())\n",'''                        if fa_block:
                            prediction_alt=dot(alt_q,c['query'],'core_q')+dot(alt_k,c['key'],'core_k')+dot(alt_v,c['value'],'core_v')
                            actual_core=pnative
                        else:
                            prediction_alt=(dot(alt_q,e['q'],'norm_q')+dot(alt_k,e['k'],'norm_k')+dot(alt_v,e['v'],'core_v')
                                            +dot(alternative['g'],e['raw_g'],'core_g')+dot(alternative['beta'],e['beta'],'core_beta'))
                            actual_core=pcore
                        local_order={'actual_effect':actual_core,'current_prediction':internal['finite_core']+actual_core,
                                     'reversed_prediction':prediction_alt,'current_error':internal['finite_core'],
                                     'reversed_error':prediction_alt-actual_core}
                        internal['projection_conv_and_other']=post['remaining_mixer']-sum(internal.values())
'''),
 ("'mixer_parts':post,'internal_parts':internal,", "'mixer_parts':post,'internal_parts':internal,'core_order_contrast':local_order,")]
derived=s
for old,new in replacements:
    if old.startswith('qwen35_'):assert derived.count(old)>=1
    else:assert derived.count(old)==1,old
    derived=derived.replace(old,new)
ast.parse(derived)
out=HERE/'trace_qwen35_core_order.py';out.write_text(derived,newline='\n')
report={'controller':{'original_path':path,'original_sha256':sha(raw),
                     'derived_sha256':sha((HERE/'qwen35_core_order_trace_controller.py').read_bytes()),
                     'diff':''.join(difflib.unified_diff(source.splitlines(True),controller.splitlines(True)))},
        'driver':{'source':sp.name,'source_sha256':sha(sp.read_bytes()),'derived_sha256':sha(out.read_bytes()),
                  'diff':''.join(difflib.unified_diff(s.splitlines(True),derived.splitlines(True)))},
        'scope':'Current upstream and actual partial core states held fixed. Same finite FA/FLA backends with reversed paired endpoints, one extra local core call per layer. Alternative coefficients are never fed into model or original propagation. No changed native implementation.'}
(HERE/'qwen35_core_order_trace_derivation.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
print(json.dumps({k:v['derived_sha256'] for k,v in report.items() if isinstance(v,dict)}))
