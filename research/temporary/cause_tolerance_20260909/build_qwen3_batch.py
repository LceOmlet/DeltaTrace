"""Derive batch-shaped DT plumbing from pinned own-DT sources, not model code.

Only checkpoint target packing, batch reshapes, seed scattering and endpoint
views change. Existing model/replay/public FA/vendor finite kernel are reused.
Right EOS tails are attended normally but receive no target seed. Causal real
prefix outputs must be checked against separate original native calls.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()


def derive(source,output,changes):
    path=ROOT/source;raw=path.read_bytes();original=raw.decode().replace('\r\n','\n');s=original
    for a,b in changes:
        assert s.count(a)==1,(output,a,s.count(a));s=s.replace(a,b)
    ast.parse(s);(HERE/output).write_text(s,newline='\n')
    return {'source':source,'source_sha256':sha(raw),'output':output,'output_sha256':sha((HERE/output).read_bytes()),
            'diff':''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=source,tofile=output))}


def main():
    clean_raw=(ROOT/'deltatrace/clean/sources.json').read_bytes()
    deferred=json.loads((ROOT/'deltatrace/accelerated/deferred_sources.json').read_bytes())
    assert sha(clean_raw)==deferred['base_clean_sources_sha256']
    for name,record in json.loads(clean_raw)['models']['qwen3']['files'].items():
        assert sha((ROOT/name).read_bytes())==record['sha256'],name
    for name,digest in deferred['files'].items():assert sha((ROOT/name).read_bytes())==digest,name
    rows=[]
    # Keep only the native checkpoint collector. No layer or attention forward
    # is copied or replaced; the collector attaches the existing native hooks.
    source='deltatrace/clean/qwen3/qwen_signed_secant_paired_public_fa.py'
    raw=(ROOT/source).read_bytes();s=raw.decode().replace('\r\n','\n')
    start=s.index('def capture_checkpoint_pair_raw(');end=s.index('\n\ndef capture_checkpoint_pair(',start)
    body=s[start:end];original=body
    replacements=[
        ('def capture_checkpoint_pair_raw(model,ids,mask,prompt_len):','def capture_checkpoint_batch_raw(model,ids,mask,target_samples,target_positions,target_ids):'),
        ('assert ids.shape[0]==mask.shape[0]==2','assert ids.shape==mask.shape and ids.shape[0]%2==0\n    batch=ids.shape[0]//2'),
        ("            logits=result.logits[:,prompt_len-1:-1];target=ids[:,prompt_len:]", "            paired_samples=2*target_samples[None,:]+torch.arange(2,device=ids.device)[:,None]\n            logits=result.logits[paired_samples,target_positions[None,:]]\n            target=target_ids[None,:].expand(2,-1)"),
        ("            cache['score16']=lp16.sum(-1).cpu().tolist();cache['score32_sum64']=lp32.double().sum(-1).cpu().tolist()", "            cache['score16']=lp16.sum(-1).cpu().tolist();cache['score32_sum64']=lp32.double().sum(-1).cpu().tolist()\n            cache['scores_by_sample']=[torch.zeros(batch,device=ids.device,dtype=torch.float64).index_add_(0,target_samples,v.double()).cpu().tolist() for v in lp32]"),
        ("            cache['prompt_len']=prompt_len;cache['length']=ids.shape[1]", "            cache['length']=ids.shape[1];cache['sample_batch']=batch\n            cache['target_samples']=target_samples;cache['target_positions']=target_positions")]
    for a,b in replacements:assert body.count(a)==1,a;body=body.replace(a,b)
    out='qwen3_batch_capture.py';text='"""Native root checkpoint collector with explicit packed target rows."""\nimport time\n\n'+body+'\n'
    ast.parse(text);(HERE/out).write_text(text,newline='\n')
    rows.append({'source':source,'source_sha256':sha(raw),'output':out,'output_sha256':sha((HERE/out).read_bytes()),
                 'scope':'Original collector only; other original functions imported/reused separately',
                 'diff':''.join(difflib.unified_diff(original.splitlines(True),body.splitlines(True),fromfile=source,tofile=out))})
    rows.append(derive('deltatrace/clean/qwen3/compiled_secant_boundaries.py','qwen3_batch_layout.py',[
        ('mkr.reshape(1,k_heads,groups,n,h)','mkr.reshape(mkr.shape[0],k_heads,groups,n,h)'),
        ('v_product.reshape(1,v_heads,groups,n,h)','v_product.reshape(v_product.shape[0],v_heads,groups,n,h)')]))
    rows.append(derive('deltatrace/accelerated/qwen3/qwen3_deferred_finite.py','qwen3_batch_finite.py',[
        ('from compiled_secant_boundaries import attention_layout, norm_residual_two, norm_residual_three','from compiled_secant_boundaries import norm_residual_two, norm_residual_three\nfrom qwen3_batch_layout import attention_layout'),
        ("    assert before['length'] == after['length'] and before['prompt_len'] == after['prompt_len']","    assert before['length']==after['length'] and before['sample_batch']==after['sample_batch']\n    batch=before['sample_batch']"),
        ("values['v'].view(1,n,-1,h)","values['v'].view(batch,n,-1,h)"),
        ("        start = before['prompt_len'] - 1\n        m_final = torch.zeros_like(f(before['norm_out']))\n        m_final[0, start:-1] = native_half_linear(seed, model.lm_head.weight)","        m_final = torch.zeros_like(f(before['norm_out']))\n        m_final[before['target_samples'],before['target_positions']] = native_half_linear(seed, model.lm_head.weight)"),
        ('m_concat.view(1,n,heads,h)','m_concat.view(batch,n,heads,h)'),
        ('mqp.reshape(1, n, -1)','mqp.reshape(batch, n, -1)'),
        ('mkp.reshape(1, n, -1)','mkp.reshape(batch, n, -1)'),
        ('mv.transpose(1, 2).reshape(1, n, -1)','mv.transpose(1, 2).reshape(batch, n, -1)'),
        ('signed = (m.double() * embedding_delta.double()).sum(-1).flatten()','signed = (m.double() * embedding_delta.double()).sum(-1)')]))
    (HERE/'qwen3_batch_derivation.json').write_text(json.dumps({'status':'derived_not_accepted','sources':rows,
        'native_model_or_attention_copied':False,'finite_arithmetic_change':'none; batch shapes and target-row layout only'},indent=2)+'\n',newline='\n')
    print('Derived batch checkpoint/layout/finite modules from pinned DT sources.')


if __name__=='__main__':main()
