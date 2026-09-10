"""CPU-only source invariance checks for the explicit root-retained Qwen3 API."""
import argparse,ast,hashlib,json
from pathlib import Path


def verify(root):
    root=Path(root);study=root/'research/temporary/qwen3_short_efficiency_20260910'
    sha=lambda b:hashlib.sha256(b).hexdigest()
    manifest_path=root/'deltatrace/accelerated/root_retained_qwen3_sources.json'
    manifest=json.loads(manifest_path.read_bytes())
    assert manifest['base_retained_manifest_sha256']==sha((root/'deltatrace/accelerated/retained_sources.json').read_bytes())
    for name,digest in manifest['files'].items():
        data=(root/name).read_bytes();assert sha(data)==digest,name
        if name.endswith('.py'):ast.parse(data)
    changes=json.loads((study/'root_production_derivation.json').read_bytes())
    for name,row in changes['files'].items():
        assert sha((root/row['source']).read_bytes())==row['source_sha256'],row['source']
        assert sha((root/name).read_bytes())==row['sha256'],name
    # Construct the only permitted edits to the original finite expression body.
    source=(root/'deltatrace/accelerated/qwen3/qwen3_deferred_finite.py').read_text()
    start=source.index('        reconstructed_q, reconstructed_k = apply_rotary_pos_emb(')
    end=source.index("        validation.equal(values['v']",start)
    expected=source[:start]+'''        validation.rope(values['qnorm'],values['knorm'],qs,ks,before['cos'],before['sin'],
            f'layer{layer.self_attn.layer_idx}.rope_q',f'layer{layer.self_attn.layer_idx}.rope_k')
'''+source[end:]
    assert expected==(study/'qwen3_rope_finite.py').read_text()
    expected=expected.replace('    torch.cuda.reset_peak_memory_stats()\n','    # Full-call controller resets the peak before root capture.\n')
    assert expected==(root/'deltatrace/accelerated/qwen3/qwen3_rope_finite.py').read_text()
    # Construct the sole preparation substitution in the original finite FA ABI.
    source=(root/'deltatrace/clean/qwen3/vendor_fa_finite_runtime.py').read_text()
    start=source.index('        values=[operands[name].detach()');end=source.index('        tau=torch.empty',start)
    expected=source[:start]+'''        for name in ['lse0','lse1']:
            value=operands[name]
            assert value.shape==(batch,heads,length) and value.dtype==torch.float32 and value.device==reference.device
        inputs=[operands[name] for name in names+['lse0','lse1']]
        values=compiled_inputs(*inputs)
        if getattr(self,'audit_inputs',False):
            eager=eager_inputs(*inputs)
            assert len(eager)==len(values)==10
            exact=[bool(torch.equal(a,b)) for a,b in zip(eager,values)]
            assert all(exact),exact
            if activity is not None:activity['compiled_input_buffers_exact']=exact
            del eager
'''+source[end:]
    expected=expected.replace('class VendorFAFiniteP1:', 'from compiled_fa_inputs import compiled_inputs,eager_inputs\n\nclass CompiledInputsFA:')
    assert expected==(root/'deltatrace/accelerated/qwen3/compiled_fa_runtime.py').read_text()
    expected=(study/'qwen3_root_rope.py').read_text().replace('import torch\n','import torch\nfrom native_capture_events import LocalCaptureEvents\n',1)
    expected=expected.replace('            sys.setprofile(self.observe_fa)','            self.local_events=LocalCaptureEvents([self.public_code],self.observe_fa,returns_only=True)\n            self.local_events.__enter__()')
    expected=expected.replace('        sys.setprofile(None)\n        for handle',"        events=getattr(self,'local_events',None)\n        if events is not None:events.__exit__(*args)\n        for handle")
    assert expected==(root/'deltatrace/accelerated/qwen3/qwen3_root_retained.py').read_text()
    for name in ['batched_validation_rope.py','compiled_fa_inputs.py']:
        assert (study/name).read_bytes()==(root/'deltatrace/accelerated/qwen3'/name).read_bytes()
    clean=json.loads((root/'deltatrace/clean/sources.json').read_bytes());count=0
    for model in clean['models'].values():
        for name,row in model['files'].items():assert sha((root/name).read_bytes())==row['sha256'];count+=1
    return {'runtime_files_verified':len(manifest['files']),'clean_files_verified':count,
        'finite_math_body_preserved':True,'finite_FA_ABI_and_all_buffer_checks_preserved':True,
        'production_capture_equals_tested_code_local_capture':True,'peak_reset_moved_to_complete_API_entry':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output',type=Path);args=parser.parse_args();result=verify(args.repo)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(result,indent=2))
