"""Prepare independent leading-example dimension for the verified finite FA op.

No GPU execution or batching claim from this source preparation.
"""
import ast
from pathlib import Path
A=Path(__file__).resolve().parent
s=(A/'qwen_signed_secant_vendor_fa.py').read_text()
assert s.count('from compiled_secant_boundaries import attention_layout,')==1
s=s.replace('from compiled_secant_boundaries import attention_layout,','from compiled_secant_boundaries import',1)
s='from compiled_secant_batched_layout import attention_layout\n'+s
s=s.replace("    device = next(model.parameters()).device", "    batch_size=before['last'].shape[0]\n    assert batch_size==len(before['prompt_len'])==len(before['actual_lengths'])\n    device = next(model.parameters()).device",1)
s=s.replace('.view(1, n,','.view(batch_size, n,').replace('.reshape(1, n,','.reshape(batch_size, n,').replace('.view(1,n,','.view(batch_size,n,')
old="        start = before['prompt_len'] - 1\n        m_final = torch.zeros_like(f(before['norm_out']))\n        m_final[0, start:-1] = native_half_linear(seed, model.lm_head.weight)"
new="""        m_final = torch.zeros_like(f(before['norm_out']))
        final_seed=native_half_linear(seed,model.lm_head.weight)
        offset=0
        for sample,(plen,length) in enumerate(zip(before['prompt_len'],before['actual_lengths'])):
            count=length-plen
            m_final[sample,plen-1:length-1]=final_seed[offset:offset+count]
            offset+=count
        assert offset==final_seed.shape[0]
        del final_seed"""
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('.sum(-1).flatten()', '.sum(-1)').replace('total = float(signed.sum())','total = signed.sum(-1).cpu()')
s=s.replace("'target_delta_score32_sum64': g_delta", "'target_delta_score32_sum64': g_delta.tolist()")
s=s.replace("'target_delta_score16': after['score16'] - before['score16']", "'target_delta_score16': (after['score16'] - before['score16']).tolist()")
s=s.replace("'signed_sum': total, 'unassigned_total': g_delta - total", "'signed_sum': total.tolist(), 'unassigned_total': (g_delta - total).tolist()")
ast.parse(s);(A/'qwen_signed_secant_batched_vendor_fa.py').write_text(s,encoding='utf-8')
source=(A/'qwen_signed_secant_batched_public_fa.py').read_text()
source=source.replace('from qwen_signed_secant_batched_pv import propagate_signed_secant','from qwen_signed_secant_batched_vendor_fa import propagate_signed_secant')
source=source.replace("def propagate_batch(model,before,after,pv_rule='content_P1',activity=None):", "def propagate_batch(model,before,after,pv_rule='content_P1',activity=None,finite_attention=None,finite_activity=None):")
source=source.replace("result=propagate_signed_secant(model,left,right,'rescale',pv_rule=pv_rule)","result=propagate_signed_secant(model,left,right,'rescale',pv_rule=pv_rule,finite_attention=finite_attention,finite_activity=finite_activity)")
source=source.replace('Same P1 arithmetic generalized over independent leading sample dimension; no model/native backward replacement.',
 'Content P1 generalized over independent leading sample dimension with the traceable FA finite extension; no global NxN attribution matrix or model/native backward replacement.')
source=source.replace('Explicit finite attribution P from actual QKV and documented auxiliary public FA LSE. Auxiliary output exactly checked, never replaces model output.',
 'No global P: finite FA tiles use actual QKV and public native LSE. Auxiliary output exactly checked, never replaces model output.')
ast.parse(source);(A/'qwen_signed_secant_batched_vendor_fa_public.py').write_text(source,encoding='utf-8')
print('Prepared true multi-example finite-FA source; actual batching validation remains pending.')
