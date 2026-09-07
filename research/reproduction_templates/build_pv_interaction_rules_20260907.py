"""Change only PV interaction allocation, not actual model endpoints or other rules."""
import ast,json,hashlib
from pathlib import Path
A=Path(__file__).resolve().parent
s=(A/'qwen_signed_secant_production_boundaries.py').read_text()
old="def propagate_signed_secant(model, before, after, variant='rescale', progress=None):"
assert s.count(old)==1;s=s.replace(old,"def propagate_signed_secant(model, before, after, variant='rescale', progress=None, pv_rule='symmetric'):")
s=s.replace("    assert variant == 'rescale'","    assert variant == 'rescale'\n    assert pv_rule in ['symmetric', 'content_P1', 'content_P0']",1)
old='''            mp = mout @ midpoint(vr0, vr1).transpose(-1, -2)
            mvr = midpoint(p0, p1).transpose(-1, -2) @ mout'''
new='''            if pv_rule == 'symmetric':
                mp = mout @ midpoint(vr0, vr1).transpose(-1, -2)
                mvr = midpoint(p0, p1).transpose(-1, -2) @ mout
            elif pv_rule == 'content_P1':
                mp = mout @ vr0.transpose(-1, -2)
                mvr = p1.transpose(-1, -2) @ mout
            else: # content_P0: reverse interaction allocation order.
                mp = mout @ vr1.transpose(-1, -2)
                mvr = p0.transpose(-1, -2) @ mout'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("return {'variant': variant, 'signed_full_sequence':", "return {'variant': variant, 'pv_rule': pv_rule, 'pv_rule_scope': 'Only PV interaction allocation differs: symmetric, deltaP*V0+P1*deltaV, or deltaP*V1+P0*deltaV. Mixed endpoints are algebraic attribution terms, never claimed as actual model counterfactual forwards. NativeFA probability approximation/global residual remain explicit.', 'signed_full_sequence':",1)
ast.parse(s);(A/'qwen_signed_secant_pv_rules.py').write_text(s,encoding='utf-8')
t=(A/'qwen_signed_secant_native_paired_boundaries.py').read_text()
t=t.replace('from qwen_signed_secant_production_boundaries import','from qwen_signed_secant_pv_rules import')
t=t.replace('def propagate_paired_secant(model,before,after,progress=None):',"def propagate_paired_secant(model,before,after,progress=None,pv_rule='symmetric'):")
t=t.replace("full_secant_pullback(model,left,right,'rescale',progress=progress)","full_secant_pullback(model,left,right,'rescale',progress=progress,pv_rule=pv_rule)")
ast.parse(t);(A/'qwen_signed_secant_native_paired_pv_rules.py').write_text(t,encoding='utf-8')
print('Prepared3discretePVrules. No continuous tuning parameter,2PVmatmuls per layer each; other source unchanged.')
