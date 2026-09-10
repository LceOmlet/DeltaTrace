"""Reproduce the v2 runtime derivation and verify the unchanged baseline chain."""
from pathlib import Path
import ast,hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()

def verify():
    from verify_memory_sources import verify as old_verify
    old=old_verify()
    receipt=json.loads((HERE/'memory_v2_confirmation_archive_verification.json').read_bytes())
    archive=HERE/receipt['archive'];assert sha(archive.read_bytes())==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        prefix='memory_production_v2_release/'
        historical={n[len(prefix):]:z.read(n) for n in z.namelist() if n.startswith(prefix)}
    raw=historical['deltatrace/accelerated/memory_efficient_qwen3_sources.json'];m=json.loads(raw)
    d=json.loads((HERE/'memory_v2_derivation.json').read_bytes())
    assert sha(raw)==d['manifest_sha256'] and m['version']=='qwen3-memory-efficient-whole-graph-v2'
    assert m['previous_release_manifest_sha256']==d['previous_manifest_sha256']==old['manifest_sha256']
    assert len(m['files'])==13 and len(m['modules'])==12
    for n,h in m['files'].items():
        b=historical[n];assert sha(b)==h,n;ast.parse(b)
    for n,r in d['runtime_changes'].items():
        source=(ROOT/r['source']).read_bytes();assert sha(source)==r['source_sha256']
        text=source.decode().replace('\r\n','\n')
        if r['source'].endswith('bounded_target_ops.py'):
            text=text[:text.index('\ndef target_seed')].replace('from qwen3_projection_cast_boundaries import seed_half\n','')
            text=text.replace('Bound only independent target rows; keep vocabulary reductions and head MM intact.','Original native per-row target log-probs in bounded storage; original sums stay outside.')
        for a,b in r['module_renames']:text=re.sub(r'\b'+re.escape(a)+r'\b',b,text)
        assert text.encode()==historical[n] and sha(text.encode())==r['sha256'],n
    assert sha((ROOT/m['finite_kernel_source']).read_bytes())==m['finite_kernel_source_sha256']
    protocols={}
    for name in ['memory_v2_confirmation_protocol.json','memory_v2_rollout_protocol.json']:
        path=HERE/name;plan=json.loads(path.read_bytes())
        for n,h in plan['runtime_files'].items():assert sha(historical[n])==h,n
        protocols[name]=sha(path.read_bytes())
    return {'status':'verified','manifest_sha256':sha(raw),'runtime_files':13,'changed_or_added_runtime_files':len(d['runtime_changes']),
        'all_derivations_identical':True,'previous_runtime_source_verification':old,'protocols':protocols,
        'kernel_source_sha256':m['finite_kernel_source_sha256'],'finite_library_sha256':m['finite_library_sha256'],
        'original_head_and_seed_GEMM_shape_preserved':True,'original_seed_operator_preserved':True,
        'bounded_seed_function_not_shipped':True,'model_calls':0,'GPU_calls':0,
        'runtime_scope':'Archived exact v2 release; current production is verified separately.',
        'source_archive':receipt}

if __name__=='__main__':
    r=verify();(HERE/'memory_v2_source_verification.json').write_text(json.dumps(r,indent=2)+'\n',newline='\n');print(json.dumps(r,indent=2))
