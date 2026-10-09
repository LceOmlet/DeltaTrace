"""Prepare the algebraic FLA output-scale reorder on the two exact owners.

No active file is changed. Keep the accepted FP16 range representation: a
fixed attention scale alone cannot bound arbitrary finite cotangents.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
OUT = HERE/'fla-early-scale-review-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def transform(raw):
    source = raw.decode()
    edits = [
        ('def _norm_gate_finite_rule(o,z,m,weight,eps,norm_gate_rule):',
         'def _norm_gate_finite_rule(o,z,m,weight,eps,norm_gate_rule,mo_scale=1.0):'),
        ('weight,m*s1,eps)', 'weight,m*s1*mo_scale,eps)'),
        ('weight,m*s_mean,eps)', 'weight,m*s_mean*mo_scale,eps)'),
        ("    mo,mz=norm_gate(e['o'],c['z'],m,module.norm.weight,module.norm.eps,norm_gate_rule)",
         """    # FLA output is scale * qH; its state update does not depend on scale.
    # Fuse that scale into the RMS content cotangent before its low-precision
    # boundary, then use unit scale for both native adjoints and finite terms.
    # The SiLU gate cotangent remains unchanged. Preserve the BF16 route.
    fla_scale=scale
    if e['q'].dtype==torch.float16:
        mo,mz=norm_gate(e['o'],c['z'],m,module.norm.weight,module.norm.eps,norm_gate_rule,scale)
        fla_scale=1.0
    else:
        mo,mz=norm_gate(e['o'],c['z'],m,module.norm.weight,module.norm.eps,norm_gate_rule)"""),
        ('native_mo[:,cut:,start:start+fla_head_batch_size].contiguous(),scale)',
         'native_mo[:,cut:,start:start+fla_head_batch_size].contiguous(),fla_scale)'),
        ('coeff=fla_pullback(fla_endpoints,native_mo[:,cut:],scale)',
         'coeff=fla_pullback(fla_endpoints,native_mo[:,cut:],fla_scale)'),
        ("'mo_native':native_mo,'mz':mz,'coeff':coeff,",
         "'mo_native':native_mo,'mz':mz,'coeff':coeff,\n            'fla_scale':fla_scale,'norm_output_seed_scale':scale if fla_scale==1.0 else 1.0,"),
    ]
    candidate = source
    for old,new in edits:
        assert candidate.count(old) == 1, old
        candidate = candidate.replace(old,new)
    before,after = ast.parse(source),ast.parse(candidate)
    functions = lambda tree:{x.name:ast.dump(x) for x in tree.body if isinstance(x,ast.FunctionDef)}
    a,b = functions(before),functions(after)
    assert {k for k in a if a[k] != b[k]} == {'_norm_gate_finite_rule','gdn_finite_pullback'}
    public = lambda tree:next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='gdn_finite_pullback')
    assert ast.dump(public(before).args) == ast.dump(public(after).args)
    assert 'torch.ldexp' in candidate and 'restore_seed_range' in candidate
    return candidate.encode()


def main():
    bases = dict(textcraft=REPO/'deltatrace/clean/qwen35/qwen35_gdn_finite.py',
        appworld=OUT/'baseline-qwen35_gdn_finite.py')
    hashes = dict(textcraft='bc1a11d95255f8990f312696235c50da219a91d2d3f026155084f2f56273bb15',
        appworld='33b169b3fb660eb8ce57a6bda6ecb04c7f7235a029aa04038ff447b2faddf6fd')
    record = dict(version='fla-early-output-scale-20261009-v1',status='prepared_only',owners={},
        identity='For fixed captured endpoints: F_E(m, s) = F_E(s*m, 1). Native state reverse calls use s*m; all direct qH/qKV finite terms use the same s*m. Gate cotangent mz is unchanged.',
        scope='Algebraic numerical-range reorder, not an extreme-credit estimation repair.',
        range_guard='Accepted per-head power-of-two representation remains after the reorder; fixed s alone is not a universal FP16 overflow bound.',
        cost='One scalar fused into the existing RMS upstream product; no new model call, full tensor, reduction or FLA call. Existing range guard retained, not a promised speedup.',
        default_import_modified=False,production_modified=False,official_tolerance_verified=False)
    for task,path in bases.items():
        raw = path.read_bytes()
        assert sha(raw) == hashes[task]
        new = transform(raw)
        target = OUT/task/'qwen35_gdn_finite.py'
        target.parent.mkdir(exist_ok=False)
        target.write_bytes(new)
        (target.parent/'owner.patch').write_text(''.join(difflib.unified_diff(raw.decode().splitlines(True),new.decode().splitlines(True),
            fromfile=task+'-accepted-range-base',tofile=task+'-early-scale-candidate')),encoding='utf-8')
        record['owners'][task] = dict(base_path=str(path),base_sha256=sha(raw),candidate_path=str(target),
            candidate_sha256=sha(new),consume_captures=(task=='appworld'),public_signature_unchanged=True)
    (OUT/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
