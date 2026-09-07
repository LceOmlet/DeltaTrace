"""Independent extracted-expression checks plus entire surrounding source coverage."""
import ast,copy,hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent
base_text=(A/'qwen_signed_secant_production_swiglu.py').read_text()
candidate_text=(A/'qwen_signed_secant_production_boundaries.py').read_text()
base=ast.parse(base_text);candidate=ast.parse(candidate_text)
helper=ast.parse((A/'compiled_secant_boundaries.py').read_text())
fn=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
old=fn(base,'propagate_signed_secant');new=fn(candidate,'propagate_signed_secant')
def expr(text):return ast.dump(ast.parse(text,mode='eval').body)
def assignment(tree,name):
    rows=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
    assert len(rows)==1,(name,len(rows));return rows[0].value
def check_assignment(tree,name,expected):assert ast.dump(assignment(tree,name))==expr(expected),name
def check_return(tree,expected):assert ast.dump(tree.body[-1].value)==expr(expected),tree.name
check_return(fn(helper,'midpoint_rule'),'(a+b)*0.5')
check_assignment(old,'mp','mout @ ((vr0+vr1)*0.5).transpose(-1,-2)')
check_assignment(new,'mp','mout @ midpoint(vr0,vr1).transpose(-1,-2)')
check_assignment(old,'mvr','((p0+p1)*0.5).transpose(-1,-2) @ mout')
check_assignment(new,'mvr','midpoint(p0,p1).transpose(-1,-2) @ mout')
check_assignment(old,'mq','mz @ ((kr0+kr1)*0.5)*scaling')
check_assignment(new,'q_product','mz @ midpoint(kr0,kr1)')
check_assignment(old,'mkr','mz.transpose(-1,-2) @ ((q0+q1)*0.5)*scaling')
check_assignment(new,'k_product','mz.transpose(-1,-2) @ midpoint(q0,q1)')
layout=fn(helper,'attention_layout_rule')
for name,expected in {'mq':'q_product*scaling','mkr':'k_product*scaling',
 'mk':'mkr.reshape(1,k_heads,groups,n,h).sum(2)','mv':'v_product.reshape(1,v_heads,groups,n,h).sum(2)',
 'cos':'cos.float().unsqueeze(1)','sin':'sin.float().unsqueeze(1)',
 'mqn':'rotation_transpose(mq,cos,sin).transpose(1,2)','mkn':'rotation_transpose(mk,cos,sin).transpose(1,2)'}.items():check_assignment(layout,name,expected)
check_return(layout,'(mqn,mkn,mv)')
check_assignment(old,'mk','mkr.reshape(1,k0.shape[1],groups,n,h).sum(2)')
check_assignment(old,'mv','mvr.reshape(1,v0.shape[1],groups,n,h).sum(2)')
calls=[n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='attention_layout']
assert len(calls)==1 and ast.dump(calls[0])==expr("attention_layout(q_product,k_product,mvr,before['cos'],before['sin'],groups,scaling,k0.shape[1],v0.shape[1],n,h)")
for name in ['mqn','mkn']:assert ast.dump(assignment(old,name))==ast.dump(assignment(layout,name))
for name in ['cos','sin']:check_assignment(old,name,f"f(before['{name}']).unsqueeze(1)")
# f on captured native tensors is a device-preserving FP32 conversion, matching
# float() on the original same-device cos/sin. Native capture/replay is unchanged.
check_return(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='rotation_transpose'),'m*cos-rotate_half(m*sin)')
rotate=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='rotate_half')
assert ast.unparse(rotate.body[0])=='first, second = x.chunk(2, dim=-1)'
check_return(rotate,'torch.cat((-second,first),dim=-1)')
rotation=fn(helper,'rotation_transpose');check_assignment(rotation,'value','m*sin')
assert ast.unparse(rotation.body[1])=='first, second = value.chunk(2, dim=-1)'
check_return(rotation,'m*cos-torch.cat((-second,first),dim=-1)')
attention_old=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='native_attention_operands')
attention_new=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='native_attention_operands')
check_assignment(attention_old,'raw','q @ kr.transpose(-1,-2)*layer.self_attn.scaling')
check_assignment(attention_new,'qk_unscaled','q @ kr.transpose(-1,-2)')
check_return(fn(helper,'scaled_probability_rule'),'probability_with_audit(qk*scaling,lse)')
assert any(isinstance(n,ast.Call) and ast.dump(n)==expr('compiled_probability(raw,lse)') for n in ast.walk(attention_old))
assert any(isinstance(n,ast.Call) and ast.dump(n)==expr('scaled_probability(qk_unscaled,lse,layer.self_attn.scaling)') for n in ast.walk(attention_new))
assert not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in ['score0','score1'] for n in ast.walk(new))
check_return(attention_old,'(q,k,v,kr,vr,prob,raw,z)');check_return(attention_new,'(q,k,v,kr,vr,prob,None,z)')
for count in ['two','three']:
    rule=fn(helper,'norm_residual_'+count+'_rule')
    check_assignment(rule,'upstream','first+second' if count=='two' else 'first+second+third')
    check_assignment(rule,'normalized','rmsnorm_secant_pullback(x0,x1,weight,upstream,eps)')
    check_return(rule,'residual+normalized')
check_assignment(old,'m_mlp_input','m_x_gate+m_x_up')
check_assignment(old,'m_norm',"norm_back(layer.post_attention_layernorm,a['mid'],b['mid'],m_mlp_input)")
check_assignment(old,'ma','mqa+mka+mva')
check_assignment(old,'m_in',"norm_back(layer.input_layernorm,a['x'],b['x'],ma)")
for expression in ["norm_residual_two(a['mid'],b['mid'],f(layer.post_attention_layernorm.weight),m_x_gate,m_x_up,m_mid,layer.post_attention_layernorm.variance_epsilon)",
 "norm_residual_three(a['x'],b['x'],f(layer.input_layernorm.weight),mqa,mka,mva,m_x_residual,layer.input_layernorm.variance_epsilon)"]:
    assert sum(isinstance(n,ast.Call) and ast.dump(n)==expr(expression) for n in ast.walk(new))==1
norm=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='norm_back')
check_assignment(norm,'answer','rmsnorm_secant_pullback(x0,x1,f(module.weight),m,module.variance_epsilon)')
# Exact source coverage: reversing the declared extractions must restore every
# original statement. Math above is checked independently of this manifest.
restored=candidate_text.split('\n',1)[1]
for change in reversed(json.loads((A/'secant_boundary_extraction_manifest_20260907.json').read_text())):
    assert restored.count(change['after'])==change['count'],(change['after'],restored.count(change['after']))
    restored=restored.replace(change['after'],change['before'])
assert restored==base_text
calls=[n for n in ast.walk(helper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='torch.compile']
assert len(calls)==5
for call in calls:assert {x.arg:ast.literal_eval(x.value) for x in call.keywords}=={'fullgraph':True,'dynamic':True,'backend':'inductor'}
assert not any(isinstance(n,ast.BinOp) and isinstance(n.op,ast.MatMult) for n in ast.walk(helper))
paired=(A/'qwen_signed_secant_native_paired_boundaries.py').read_text().replace('from qwen_signed_secant_production_boundaries import','from qwen_signed_secant_production_swiglu import')
assert paired==(A/'qwen_signed_secant_native_paired_swiglu.py').read_text()
out={'status':'verified_complete','boundary_groups':['midpoint4','scaled_probability_and_original_audit','attention_scale_GQA_RoPE','two_branch_RMSNorm_residual','three_branch_RMSNorm_residual'],
 'existing_fullgraph_compiled_helpers':5,'model_FA_GEMM_replacement':False,'all_other_source_restores_exactly':True,
 'candidate_sha256':hashlib.sha256((A/'qwen_signed_secant_production_boundaries.py').read_bytes()).hexdigest(),
 'scope':'Source expressions and original native/finite rule/checks preserved, no intended math change. Actual compiler rounding/cost/quality/native dispatch still require runtime evidence. Only unused raw scaled score slot explicitly None; no current diagnostic ledger claimed.'}
(A/'secant_boundary_source_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
