"""Read-only source provenance check for the actual official FT hybrid variant."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
A=Path(__file__).resolve().parent;rev='e81b3be50a48dcfc652fbf1b530069b552736e66';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'published_flashtrace'/('qwen35-'+rev);D.mkdir(parents=True,exist_ok=True)
shared=A/'snapshot${PRIVATE_MOUNT_PATH}'
names=['flashtrace/qwen35.py','flashtrace/core.py','flashtrace/attribution.py','tests/test_qwen35_support.py','evaluations/qwen35_faithfulness_smoke.py','LICENSE']
def fetch(name):
    url=f'https://raw.githubusercontent.com/wbopan/flashtrace/{rev}/{name}'
    with urllib.request.urlopen(url,timeout=25) as response:raw=response.read()
    file=D/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(raw)
    local=shared/name
    return {'file':name,'url':url,'sha256':sha(raw),'bytes':len(raw),'shared_byte_identical':local.read_bytes()==raw if local.exists() else None}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(fetch,names))
assert all(x['shared_byte_identical'] for x in receipts if x['shared_byte_identical'] is not None)
r={'status':'official_commit_sources_match_shared_variant','commit':rev,'receipts':receipts,
    'model_identity':'Shared checkpoint is Qwen3.5-9B; inspected model directories also contain Qwen3-8B. No Qwen3.5-8B directory found in bounded search.',
    'comparison_required':'DeltaTrace versus this official FT hybrid variant on the SAME actual Qwen3.5 checkpoint, dtype, backend, input trajectory, target span and eligible tokens. No cross-model quality or cost victory.',
    'review_findings':[
        {'file':'flashtrace/core.py','lines':[428,446],'finding':'linearize_norm returns raw weight/rms for Qwen3_5RMSNorm, while the loaded native model uses(1+weight)/rms. Full-attention value path calls this helper.',
         'status':'Source-level concrete semantic mismatch against the pinned native model; magnitude/quality effect not yet measured.'},
        {'file':'flashtrace/qwen35.py','lines':[112,162],'finding':'Full-attention LayerAttnInput has no output sigmoid gate; GDN LayerAttnInput projects pre-output-norm core value mixtures and discards core_ref, with no gated output RMS/SiLU contribution in that triple.',
         'status':'Layer decomposition boundary omission visible in source. No claim about resulting metric degradation without a matched test.'},
        {'file':'flashtrace/qwen35.py','lines':[67,78,96,98,101,197],'finding':'Explicit B1 guard, GDN replay without attention_mask, identity probe expands value dimension to S and materializes per-head S-square mixing; full layers require eager/output_attentions.',
         'status':'Current interface does not meet this project default-FA/variable-batch/linear-memory requirements.'},
        {'file':'flashtrace/qwen35.py','lines':[23,44,101],'finding':'Capture wrapper delegates to the original kernel and the identity probe calls that same kernel. The V-linear recurrence identity is legitimate exact-real algebra conditional on q/k/g/beta; it is an auxiliary probe, not automatically a shadow implementation.',
         'status':'Do not falsely classify genuine native auxiliary work as shadow work. Charge its cost and verify actual default FLA execution/precision before use.'},
        {'file':'tests/test_qwen35_support.py','lines':[50,68],'finding':'Tiny7-token test reconstructs pre-norm recurrence core. It does not check native full mixer/decoder boundaries, default FA/FLA workload, real batched padding or official needle/RISE/MAS.',
         'status':'Useful narrow core test; insufficient evidence for full-model quality.'},
        {'file':'evaluations/qwen35_faithfulness_smoke.py','lines':[1,17,65,75],'finding':'Smoke script uses handcrafted samples and double argsort ranks without tie averaging, then compares across Qwen3.5-9B/Qwen3-8B. This is not the author original benchmark requested here.',
         'status':'Do not run as replacement quality benchmark or accept the PASS string as evidence.'}],
    'next':'Keep original FT hybrid source pinned as the reference. Correct/explicitly label native-structure mismatches for a credible same-model baseline; validate actual backend, norm/gate boundaries, target mapping and all paid costs before a small official-metric comparison. Preserve original and corrected variant identities.'}
(A/'qwen35_FT_variant_audit_20260908.json').write_text(json.dumps(r,indent=2),encoding='utf-8',newline='\n')
print(json.dumps({'status':r['status'],'commit':rev,'verified_files':len(receipts),'GPU_calls':0,'quality_queries':0}))
