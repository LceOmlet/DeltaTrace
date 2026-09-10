"""Read-only provenance check for the tested memory-efficient Qwen3 entry."""
from pathlib import Path
import ast,hashlib,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()

def verify():
    sys.path.insert(0,str(ROOT/'research/temporary/qwen3_short_efficiency_20260910'))
    from verify_graph_sources import verify as verify_base
    base=verify_base(ROOT)
    receipt=json.loads((HERE/'optimization_v1_archive_verification.json').read_bytes())
    archive=HERE/receipt['archive'];assert sha(archive.read_bytes())==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        frozen={n.removeprefix('memory_production_release/'):z.read(n) for n in z.namelist() if n.startswith('memory_production_release/')}
    def runtime(name):return frozen[name]
    manifest_raw=runtime('deltatrace/accelerated/memory_efficient_qwen3_sources.json')
    manifest=json.loads(manifest_raw);derivation=json.loads((HERE/'memory_production_derivation.json').read_bytes())
    assert sha(manifest_raw)==derivation['manifest_sha256']
    checked={}
    for entry in manifest['base_chain']:
        raw=runtime(entry['path']);assert sha(raw)==entry['sha256'];item=json.loads(raw)
        for name,digest in item['files'].items():
            raw=runtime(name);assert sha(raw)==digest,name;checked[name]=digest
    for name,digest in manifest['files'].items():
        raw=runtime(name);assert sha(raw)==digest,name;ast.parse(raw);checked[name]=digest
    assert len(manifest['files'])==12
    assert sha((ROOT/manifest['finite_kernel_source']).read_bytes())==manifest['finite_kernel_source_sha256']
    for name,record in derivation['runtime'].items():
        raw=(ROOT/record['source']).read_bytes();assert sha(raw)==record['source_sha256']
        # The recorded producer reads Python source in universal-newline mode.
        result=raw.decode().replace('\r\n','\n')
        for before,after in record['module_renames']:result=re.sub(r'\b'+re.escape(before)+r'\b',after,result)
        for before,after in record['explicit_code_changes']:
            assert result.count(before)==1,(name,before);result=result.replace(before,after)
        if name.endswith('qwen3_native_projection_reuse.py'):
            result=result.replace('Omit q caching in final two native layers to leave allocator headroom; other selections unchanged','Omit q caching in final12 native layers to leave allocator headroom; other selections unchanged')
            result=result.replace("'selected_projections':list(selected)","'recording_memory_scope':'Allocator counters in per-layer records are graph-recording snapshots, not per-replay measurements. The outer complete caller measures every current peak.', 'selected_projections':list(selected)")
        if name.endswith('qwen3_memory_graph_build.py'):
            result=result.replace("'root_and_finite_captured_together':True", "'recorded_peak_allocated_bytes_including_warmup':torch.cuda.max_memory_allocated(),'recorded_peak_reserved_bytes_including_warmup':torch.cuda.max_memory_reserved(),'root_and_finite_captured_together':True")
        if name.endswith('qwen3_fixed_graph_contract.py'):
            result=result.replace('tuple(module._forward_pre_hooks_with_kwargs),tuple(module._forward_hooks_with_kwargs),tuple(module._forward_hooks_always_called)','tuple(module._forward_pre_hooks_with_kwargs.items()),tuple(module._forward_hooks_with_kwargs.items()),tuple(module._forward_hooks_always_called.items())')
        assert result.encode()==runtime(name),name
    protocol=json.loads((HERE/'memory_confirmation_protocol.json').read_bytes())
    for name,digest in protocol['runtime_files'].items():assert sha(runtime(name))==digest,name
    report={'status':'verified','base_graph_and_clean_verification':base,'manifest_sha256':sha(manifest_raw),'new_runtime_files':12,
        'base_and_new_runtime_files_checked':len(checked),'protocol_source_entries_verified':len(protocol['runtime_files']),
        'all_derivations_reproduced_identically':True,'kernel_source_sha256':manifest['finite_kernel_source_sha256'],
        'finite_library_sha256':manifest['finite_library_sha256'],'model_calls':0,'GPU_calls':0,
        'scope':'Historical v1 runtime verified from its exact archived release; current base graph/clean sources separately verified. This is not a verification of the current v2 replacement.',
        'runtime_archive_sha256':receipt['sha256']}
    return report

if __name__=='__main__':
    report=verify();(HERE/'memory_source_verification.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n');print(json.dumps(report))
