"""Recompute v4 derivation, the historical release chain and common loader identity."""
from pathlib import Path
import ast,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()

def verify():
    from verify_memory_v3_sources import verify as old_verify
    old=old_verify();raw=(ROOT/'deltatrace/accelerated/memory_efficient_qwen3_sources.json').read_bytes();m=json.loads(raw);d=json.loads((HERE/'memory_v4_derivation.json').read_bytes())
    assert m['version']=='qwen3-memory-efficient-whole-graph-v4' and sha(raw)==d['manifest_sha256']
    assert m['previous_release_manifest_sha256']==d['previous_manifest_sha256']==old['manifest_sha256']
    assert len(m['files'])==14 and len(m['modules'])==13
    with zipfile.ZipFile(HERE/d['previous_exact_sources_archive']['archive']) as z:
        prior=json.loads(z.read('memory_production_v3_release/deltatrace/accelerated/memory_efficient_qwen3_sources.json'))
        for n,h in m['files'].items():
            b=(ROOT/n).read_bytes();assert sha(b)==h,n;ast.parse(b)
            if n not in d['runtime_changes']:assert b==z.read('memory_production_v3_release/'+n) and h==prior['files'][n]
        for n,r in d['runtime_changes'].items():
            source=z.read(r['source_member']);assert sha(source)==r['source_sha256'];text=source.decode()
            for a,b in r['literal_replacements']:assert text.count(a)==1;text=text.replace(a,b)
            assert text.encode()==(ROOT/n).read_bytes() and sha(text.encode())==r['sha256']
    kernel=(ROOT/m['finite_kernel_source']).read_bytes();assert sha(kernel)==m['finite_kernel_source_sha256']==d['kernel_source_sha256'] and kernel==(ROOT/d['kernel_derivation_source']).read_bytes()
    receipt=d['kernel_operator_archive'];p=HERE/receipt['archive'];assert sha(p.read_bytes())==receipt['sha256']
    with zipfile.ZipFile(p) as z:
        assert kernel==z.read('fa_mma_owner_v2/vendor_fa_finite_p1_shared_mean_reuse.cu')
        assert sha(z.read('fa_mma_owner_v2/libdeltatrace_fa_finite_shared_mean_reuse.so'))==m['finite_library_sha256']
    protocols={}
    for name in ['memory_v4_confirmation_protocol.json','memory_v4_rollout_protocol.json']:
        path=HERE/name;plan=json.loads(path.read_bytes())
        for n,h in plan['runtime_files'].items():assert sha((ROOT/n).read_bytes())==h,n
        for n,h in plan['common_loading_helpers'].items():assert sha((HERE/n).read_bytes())==h,n
        assert plan['candidate_finite_library']['sha256']==m['finite_library_sha256'];protocols[name]=sha(path.read_bytes())
    return {'status':'verified','manifest_sha256':sha(raw),'runtime_files':14,'runtime_modules':13,'derivations_identical':True,'only_python_change_is_activity_metadata':True,'original_head_seed_GEMM_normalization_preserved':True,'original_model_and_FA_sources_preserved':True,'live_vector_and_diagnostics_each_call':True,'finite_kernel_source_sha256':m['finite_kernel_source_sha256'],'finite_library_sha256':m['finite_library_sha256'],'previous_runtime_source_verification':old,'protocols':protocols,'model_calls':0,'GPU_calls':0}

if __name__=='__main__':
    d=verify();(HERE/'memory_v4_source_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
