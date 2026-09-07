"""Independent AST and exact noncommutative polynomial checks for PV-only change."""
import ast, copy, hashlib, json
from fractions import Fraction as F
from pathlib import Path
A=Path(__file__).resolve().parent
def tree(name): return ast.parse((A/name).read_text(encoding='utf-8'))
def dump(node): return ast.dump(node,include_attributes=False)
old=tree('qwen_signed_secant_production_boundaries.py')
new=tree('qwen_signed_secant_pv_rules.py')
rules={'symmetric':('mout @ midpoint(vr0, vr1).transpose(-1, -2)','midpoint(p0, p1).transpose(-1, -2) @ mout'),
       'content_P1':('mout @ vr0.transpose(-1, -2)','p1.transpose(-1, -2) @ mout'),
       'content_P0':('mout @ vr1.transpose(-1, -2)','p0.transpose(-1, -2) @ mout')}
class Restore(ast.NodeTransformer):
    def __init__(self,rule): self.rule=rule;self.branches=0
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        if node.name=='propagate_signed_secant':
            assert node.args.args[-1].arg=='pv_rule'
            assert node.args.defaults[-1].value=='symmetric'
            node.args.args.pop();node.args.defaults.pop()
        return node
    def visit_Assert(self,node):
        if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='pv_rule':
            assert dump(node.test)==dump(ast.parse("pv_rule in ['symmetric', 'content_P1', 'content_P0']",mode='eval').body)
            return None
        return self.generic_visit(node)
    def visit_If(self,node):
        if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='pv_rule':
            assert dump(node.test)==dump(ast.parse("pv_rule == 'symmetric'",mode='eval').body)
            other=node.orelse[0]
            assert dump(other.test)==dump(ast.parse("pv_rule == 'content_P1'",mode='eval').body)
            selected={'symmetric':node.body,'content_P1':other.body,'content_P0':other.orelse}[self.rule]
            assert len(selected)==2
            for statement,target,expr in zip(selected,['mp','mvr'],rules[self.rule]):
                assert dump(statement)==dump(ast.parse(f'{target} = {expr}').body[0])
            assert sum(isinstance(x,ast.MatMult) for z in selected for x in ast.walk(z))==2
            self.branches+=1
            # After checking the selected branch, restore the old symmetric AST.
            return copy.deepcopy(node.body)
        return self.generic_visit(node)
    def visit_Dict(self,node):
        node=self.generic_visit(node)
        if any(isinstance(k,ast.Constant) and k.value=='pv_rule_scope' for k in node.keys):
            pairs=[(k,v) for k,v in zip(node.keys,node.values) if not (isinstance(k,ast.Constant) and k.value in ['pv_rule','pv_rule_scope'])]
            node.keys=[x[0] for x in pairs];node.values=[x[1] for x in pairs]
        return node
proof={}
for rule in rules:
    restore=Restore(rule);got=restore.visit(copy.deepcopy(new))
    assert restore.branches==1 and dump(got)==dump(old)
    proof[rule]={'only_two_PV_multiplier_assignments_differ':True,'PV_matmuls_per_layer':2,
                 'PV_midpoints_per_layer':2 if rule=='symmetric' else 0}
paired=(A/'qwen_signed_secant_native_paired_pv_rules.py').read_text()
paired=paired.replace('from qwen_signed_secant_pv_rules import','from qwen_signed_secant_production_boundaries import')
paired=paired.replace(",pv_rule='symmetric'",'').replace(',pv_rule=pv_rule','')
assert dump(ast.parse(paired))==dump(tree('qwen_signed_secant_native_paired_boundaries.py'))
# Ordered monomials P_i V_j; no commutation of matrix operands is assumed.
def add(*polys):
    out={}
    for poly in polys:
        for key,value in poly.items(): out[key]=out.get(key,F(0))+value
    return {k:v for k,v in out.items() if v}
def product(p,v): return {(i,j):a*b for i,a in p.items() for j,b in v.items()}
dp={0:F(-1),1:F(1)};dv=dp;mean={0:F(1,2),1:F(1,2)}
polynomials={
 'symmetric':add(product(dp,mean),product(mean,dv)),
 'content_P1':add(product(dp,{0:F(1)}),product({1:F(1)},dv)),
 'content_P0':add(product(dp,{1:F(1)}),product({0:F(1)},dv))}
for rule,poly in polynomials.items():
    assert poly=={(1,1):F(1),(0,0):F(-1)}
    proof[rule]['exact_ordered_PV_polynomial_identity']=True
summary={'status':'pass','rules':proof,'unchanged_native_capture_and_replay':True,
 'scope':'Source/formula verification, zero model calls. Exact real arithmetic identity does not erase native-LSE approximation or finite precision residuals, and is not benchmark evidence.',
 'sha256':{f:hashlib.sha256((A/f).read_bytes()).hexdigest() for f in ['qwen_signed_secant_production_boundaries.py','qwen_signed_secant_pv_rules.py','qwen_signed_secant_native_paired_pv_rules.py']}}
(A/'pv_interaction_source_summary_20260907.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
