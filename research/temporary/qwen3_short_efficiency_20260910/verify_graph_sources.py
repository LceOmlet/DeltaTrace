"""CPU-only derivation checks for the native finite-program graph API."""
import argparse,ast,hashlib,json
from pathlib import Path
from verify_root_sources import verify as verify_root


def verify(root):
    root=Path(root);study=root/'research/temporary/qwen3_short_efficiency_20260910'
    sha=lambda b:hashlib.sha256(b).hexdigest();base=verify_root(root)
    raw=(root/'deltatrace/accelerated/graphed_qwen3_sources.json').read_bytes();manifest=json.loads(raw)
    assert manifest['base_root_retained_manifest_sha256']==sha((root/'deltatrace/accelerated/root_retained_qwen3_sources.json').read_bytes())
    for name,digest in manifest['files'].items():
        data=(root/name).read_bytes();assert sha(data)==digest,name;ast.parse(data)
    changes=json.loads((study/'graph_production_derivation.json').read_bytes())
    for name,row in changes['files'].items():
        assert sha((root/row['source']).read_bytes())==row['source_sha256'],row['source']
        assert sha((root/name).read_bytes())==row['sha256'],name
    # The seed's original compiled function remains identical. Only its host
    # Boolean read is deferred to the same strict result-validation boundary.
    source=(study/'qwen3_rope_finite.py').read_text()
    removed={"assert torch.equal(before['target'], after['target'])",
        "assert torch.equal(before['cos'], after['cos']) and torch.equal(before['sin'], after['sin'])",
        'torch.cuda.synchronize()','torch.cuda.reset_peak_memory_stats()',
        'assert torch.isfinite(signed).all()'}
    source=''.join(line for line in source.splitlines(keepends=True) if line.strip() not in removed)
    source=source.replace('        total = float(signed.sum())','        total = signed.sum()')
    source=source.replace("'signed_full_sequence': signed.cpu().tolist()","'signed_full_sequence': signed")
    source=source.replace('from compiled_logprob_seed import logprob_secant_seed','from compiled_logprob_seed import compiled_seed')
    source=source.replace("        seed = logprob_secant_seed(z0, z1, after['target'].to(device))",
        "        seed, seed_check = compiled_seed(z0, z1, after['target'].to(device))\n        validation.predicate(seed_check,'finite_logprob_seed')")
    assert source==(root/'deltatrace/accelerated/qwen3/qwen3_graph_tensor_finite.py').read_text()
    mapping={'qwen3_graph_storage_inputs.py':'qwen3_graph_storage_inputs.py',
        'qwen3_graph_validation_v3.py':'qwen3_graph_validation.py',
        'qwen3_graph_tensor_finite_v3.py':'qwen3_graph_tensor_finite.py',
        'qwen3_root_tensor_graph_v3.py':'qwen3_root_tensor_graph.py',
        'qwen3_finite_graph_v4.py':'qwen3_finite_graph.py'}
    for original,target in mapping.items():
        source=(study/original).read_text()
        for old,new in mapping.items():source=source.replace('from '+old[:-3]+' import','from '+new[:-3]+' import')
        if original=='qwen3_finite_graph_v4.py':
            source=source.replace('from qwen3_root_local import LocalNativeRootTape','from qwen3_root_retained import NativeRootTape as LocalNativeRootTape')
            source=source.replace('    def attribute(self,before_ids,after_ids,mask,prompt_len,*,mutation_audit=False):',
                "    def close(self):\n        assert not self.busy\n        self.program=None;self.signature=None\n\n    def attribute(self,before_ids,after_ids,mask,prompt_len,*,mutation_audit=False):")
        if original=='qwen3_root_tensor_graph_v3.py':
            start=source.index('class NativeRootTape:');end=source.index('class NativeRootAccess:',start)
            source=source[:start]+'from qwen3_root_retained import NativeRootTape\n\n\n'+source[end:]
        assert source==(root/'deltatrace/accelerated/qwen3'/target).read_text(),target
    finite=(root/'deltatrace/accelerated/qwen3/qwen3_finite_graph.py').read_text()
    resolver=(root/'deltatrace/accelerated/qwen3/qwen3_graph_validation.py').read_text()
    for original in ["assert torch.equal(before['target'],after['target'])",
        "assert torch.equal(before['cos'],after['cos']) and torch.equal(before['sin'],after['sin'])"]:assert original in finite
    assert 'assert bool(torch.isfinite(signed).all())' in resolver
    assert "if failures:raise ValueError('Graph DT validation failed before return: '" in resolver
    assert "total=float(result['signed_sum'])" in resolver and 'signed.cpu().tolist()' in resolver
    return {'base_root_verification':base,'graph_runtime_files_verified':len(manifest['files']),
        'finite_GPU_math_preserved':True,'original_compiled_seed_and_boolean_preserved':True,
        'original_native_model_and_FA_imports_preserved':True,'native_half_linear_and_finite_FA_library_unmodified':True,
        'moved_strict_checks_present_before_API_return':True,'complete_vector_and_scalar_CPU_conversion_preserved':True,
        'production_controller_matches_screened_v4_except_verified_capture_class_imports_and_close':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path)
    args=p.parse_args();result=verify(args.repo)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(result,indent=2))
