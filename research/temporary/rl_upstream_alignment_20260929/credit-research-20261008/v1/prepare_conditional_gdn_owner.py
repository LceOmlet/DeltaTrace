"""Prepare a default-inert patch in the exact preserved GDN owner, not launch."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
OUT=HERE/'conditional-gdn-owner-prepared'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    original=(HERE/'native-fla-range-owner-prepared/qwen35_gdn_finite.py').read_bytes()
    assert sha(original)=='33b169b3fb660eb8ce57a6bda6ecb04c7f7235a029aa04038ff447b2faddf6fd'
    source=original.decode()
    old='conv_silu_pullback=None,input_shape=None,fla_coefficient_start=0,capture_start=0):'
    assert source.count(old)==1
    source=source.replace(old,old[:-2]+',conditional_context_pullback=None):',1)
    start=source.index('    fla_endpoints=slice_native_fla_endpoints(e,cut) if cut else e\n')
    stop=source.index('    mx=_linear_transpose(mqkv,_linear_weights(module.in_proj_qkv))\n',start)
    block=source[start:stop]
    # The default branch is the preserved owner verbatim, including capture
    # releases. Only the new memory-context seam has distinct semantics.
    branch='''    if conditional_context_pullback is not None:
        if diagnostics:
            raise ValueError('Conditional GDN context uses its own passive observer; full joint diagnostics are incompatible.')
        mqkv,mb,ma=conditional_context_pullback(module,c,e,native_mo,scale,cut=cut,
            seed_exponent=seed_exponent,restore_seed_range=restore_seed_range,
            head_batch=fla_head_batch_size,release_consumed=release_consumed)
        if release_consumed:del native_mo
    else:
'''+''.join('    '+line if line.strip() else line for line in block.splitlines(keepends=True))
    branch+='        if release_consumed:del mprojected\n'
    source=source[:start]+branch+source[stop:]
    old='    if release_consumed:del mprojected,mqkv\n'
    assert source.count(old)==1
    source=source.replace(old,'    if release_consumed:del mqkv\n',1)
    ast.parse(source)
    (OUT/'qwen35_gdn_finite.py').write_bytes(source.encode())
    (OUT/'gdn-owner.patch').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),
        source.splitlines(True),fromfile='verified-range-owner33b169b3',tofile='default-inert-conditional-GDN')))

    original_fla=(REPO/'deltatrace/clean/qwen35/finite_fla_gpu.py').read_bytes()
    assert sha(original_fla)=='f1555736d32974668676eff4e040a500f4c2b907e6ac1b298f01dc2ba87f7db6'
    fla=original_fla.decode()
    old='''    raw1=token('raw_g',1).float();distance=(raw1-g0).abs()
    denominator=torch.where(distance==0,torch.ones_like(distance),distance)
    ratio=torch.where(distance==0,torch.ones_like(distance),-torch.expm1(-distance)/denominator)
    dg=dalpha*torch.exp(torch.maximum(g0,raw1))*ratio
'''
    assert fla.count(old)==1
    helper='''def exp_secant_factors(g0, raw1):
    """Existing stable exponential secant, shared with the GDN owner seam."""
    distance=(raw1-g0).abs()
    denominator=torch.where(distance==0,torch.ones_like(distance),distance)
    ratio=torch.where(distance==0,torch.ones_like(distance),-torch.expm1(-distance)/denominator)
    return torch.exp(torch.maximum(g0,raw1)),ratio


'''
    fla=fla.replace('def mixed_coefficients(',helper+'def mixed_coefficients(',1)
    fla=fla.replace(old,"    raw1=token('raw_g',1).float()\n    exponential,ratio=exp_secant_factors(g0,raw1)\n    dg=dalpha*exponential*ratio\n",1)
    ast.parse(fla)
    (OUT/'finite_fla_gpu.py').write_bytes(fla.encode())
    (OUT/'fla-owner.patch').write_text(''.join(difflib.unified_diff(original_fla.decode().splitlines(True),
        fla.splitlines(True),fromfile='original-finite-FLA-f1555736',tofile='shared-original-exp-secant')))
    result=dict(status='prepared_only_not_deployed_or_verified',
        original_GDN_sha256=sha(original),original_FLA_sha256=sha(original_fla),
        candidate_GDN_sha256=sha(source.encode()),candidate_FLA_sha256=sha(fla.encode()),
        callback='conditional_gdn_context.conditional_gdn_context',
        default_path='Original owner branch and state release preserved; original exp secant factored without changing operations.',
        whole_DT_repair_accepted=False,production_modified=False,formal_restart=False)
    (OUT/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
