"""Prepare an isolated original-owner dtype repair, never install or launch it."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'fla-seed-range-preserved/manifest.json').read_bytes())
owner=next(v for v in data['files'] if v['name']=='original-AppWorld-GDN.py')
raw=(HERE/'fla-seed-range-preserved'/owner['name']).read_bytes()
assert hashlib.sha256(raw).hexdigest()==owner['sha256']=='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'
source=raw.decode('utf8')
old="    native_mo=mo.to(e['o'].dtype).to(e['q'].dtype)"
new="""    native_mo=mo.to(e['o'].dtype)
    seed_exponent=None
    if e['q'].dtype==torch.float16:
        # Native FLA is linear in this cotangent. Use the smallest power of
        # two that avoids overflowing its actual FP16 representation, then
        # undo it on every returned coefficient. No attribution is clipped.
        magnitude=native_mo.float().abs().amax(dim=(1,3),keepdim=True)
        seed_exponent=torch.ceil(torch.log2(
            magnitude/torch.finfo(torch.float16).max).clamp_min(0)).to(torch.int32)
        native_mo=torch.ldexp(native_mo.float(),-seed_exponent)
    native_mo=native_mo.to(e['q'].dtype)
    def restore_seed_range(coeff,exponent):
        if exponent is None:return coeff
        return {name:torch.ldexp(value.float(),exponent if value.ndim==4 else exponent[...,0])
                for name,value in coeff.items()}"""
assert source.count(old)==1;candidate=source.replace(old,new)
old="            parts.append(fla_pullback(group,native_mo[:,cut:,start:start+fla_head_batch_size].contiguous(),scale))"
new="""            part=fla_pullback(group,native_mo[:,cut:,start:start+fla_head_batch_size].contiguous(),scale)
            exponent=None if seed_exponent is None else seed_exponent[:,:,start:start+fla_head_batch_size]
            parts.append(restore_seed_range(part,exponent))
            del part,exponent"""
assert candidate.count(old)==1;candidate=candidate.replace(old,new)
old="        coeff=fla_pullback(fla_endpoints,native_mo[:,cut:],scale)"
new=old+"\n        coeff=restore_seed_range(coeff,seed_exponent)"
assert candidate.count(old)==1;candidate=candidate.replace(old,new)
ast.parse(candidate)
folder=HERE/'native-fla-range-owner-prepared';folder.mkdir(exist_ok=True)
path=folder/'qwen35_gdn_finite.py';path.write_text(candidate,encoding='utf8',newline='\n')
(folder/'owner.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),
    fromfile=owner['path'],tofile='prepared-only/qwen35_gdn_finite.py')),encoding='utf8')
record=dict(status='prepared_only_unaccepted',base_owner={k:v for k,v in owner.items() if k!='source'},
    candidate=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    operator_diagnostic=str(HERE/'fla-seed-range-v2-result.json'),
    invariant='For fixed endpoints, every FLA coefficient is linear in the incoming cotangent. Per-head powers of two are shared across time and restored on every coefficient before existing norm/projection rules.',
    change='Only original GDN owner FP16 cotangent representation; same callback count and allocation rule. BF16 branch unchanged.',
    numerical_limit='Finite arithmetic still rounds; power-two check is diagnostic, not the required original FLA tolerance or collection quality acceptance.',
    production_modified=False,default_import_modified=False,QVA_modified=False,
    official_tolerance_verified=False,whole_DT_verified=False,extreme_credit_repaired=False)
(folder/'prepared.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
