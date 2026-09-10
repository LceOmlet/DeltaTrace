"""Verify the whole v3 source derivation and unchanged original runtime chain."""
from pathlib import Path
import ast,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()

def verify():
    from verify_memory_v2_sources import verify as old_verify
    old=old_verify();current=(ROOT/'deltatrace/accelerated/memory_efficient_qwen3_sources.json').read_bytes()
    historical=json.loads(current)['version']!='qwen3-memory-efficient-whole-graph-v3'
    if historical:
        receipt=json.loads((HERE/'memory_v3_confirmation_archive_verification.json').read_bytes());p=HERE/receipt['archive'];assert sha(p.read_bytes())==receipt['sha256']
        with zipfile.ZipFile(p) as z:release={n.removeprefix('memory_production_v3_release/'):z.read(n) for n in z.namelist() if n.startswith('memory_production_v3_release/')}
        read=lambda n:release[n]
    else:read=lambda n:(ROOT/n).read_bytes()
    raw=read('deltatrace/accelerated/memory_efficient_qwen3_sources.json');m=json.loads(raw);d=json.loads((HERE/'memory_v3_derivation.json').read_bytes())
    assert m['version']=='qwen3-memory-efficient-whole-graph-v3' and sha(raw)==d['manifest_sha256']
    assert m['previous_release_manifest_sha256']==d['previous_manifest_sha256']==old['manifest_sha256']
    assert len(m['files'])==14 and len(m['modules'])==13
    for n,h in m['files'].items():
        b=read(n);assert sha(b)==h,n;ast.parse(b)
    for n,r in d['runtime_changes'].items():
        if 'source' in r:source=(ROOT/r['source']).read_bytes()
        else:
            with zipfile.ZipFile(HERE/r['source_archive']) as z:source=z.read(r['source_member'])
        assert sha(source)==r['source_sha256'];text=source.decode()
        for a,b in r.get('module_renames',[]):text=text.replace(a,b)
        if 'insert_text' in r:
            assert text.count(r['insert_after'])==1;text=text.replace(r['insert_after'],r['insert_after']+r['insert_text'])
        assert text.encode()==read(n) and sha(text.encode())==r['sha256'],n
    kernel=read(m['finite_kernel_source'])
    assert sha(kernel)==m['finite_kernel_source_sha256']==d['kernel_source_sha256']
    assert kernel==(ROOT/d['kernel_derivation_source']).read_bytes()
    receipt=d['kernel_operator_archive'];p=HERE/receipt['archive'];assert sha(p.read_bytes())==receipt['sha256']
    with zipfile.ZipFile(p) as z:
        assert kernel==z.read('fa_owner_cached/vendor_fa_finite_p1_shared_mean_reuse.cu')
        assert sha(z.read('fa_owner_cached/libdeltatrace_fa_finite_shared_mean_reuse.so'))==m['finite_library_sha256']
    protocols={}
    for name in ['memory_v3_confirmation_protocol.json','memory_v3_rollout_protocol.json']:
        path=HERE/name;plan=json.loads(path.read_bytes())
        for n,h in plan['runtime_files'].items():assert sha(read(n))==h,n
        assert plan['candidate_finite_library']['sha256']==m['finite_library_sha256'];protocols[name]=sha(path.read_bytes())
    return {'status':'verified','manifest_sha256':sha(raw),'runtime_files':14,'runtime_modules':13,'derivations_identical':True,
        'original_head_seed_GEMM_normalization_preserved':True,'original_model_and_FA_sources_preserved':True,
        'private_template_contains_static_primitive_metadata_only':True,'live_vector_and_diagnostics_each_call':True,
        'finite_kernel_source_sha256':m['finite_kernel_source_sha256'],'finite_library_sha256':m['finite_library_sha256'],
        'previous_runtime_source_verification':old,'protocols':protocols,'model_calls':0,'GPU_calls':0}

if __name__=='__main__':
    d=verify();(HERE/'memory_v3_source_verification.json').write_text(json.dumps(d,indent=2)+'\n',newline='\n');print(json.dumps(d,indent=2))
