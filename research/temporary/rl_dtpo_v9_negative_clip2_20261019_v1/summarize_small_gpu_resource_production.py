"""Select the first natural production window with the original Util reducer.

The original dmon/phase parser, native log reader, and checkpoint criterion
remain unchanged. Only the applied resource identity and window are selected.
"""
import hashlib
from pathlib import Path
import sys

owner = Path(__file__).with_name('summarize_root_resource_production.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '6ed68c31a011471d12ff544a6a06b60bce21035fbd9e583998f0a71e5b583e3b'
code = owner.read_text()
for old, new in (
    ('root-pageable-resource-override-completed.json', 'small-GPU-resident-resource-override-completed.json'),
    ('completed_runtime_root_storage_override', 'completed_runtime_bounded_original_storage_selection'),
    ("lower = max(r['override']['applied_unix'] for r in override['actual_workers'])",
     "lower = max(r['override']['installed_unix'] for r in override['actual_workers'])"),
    ("owner['completed'](job) >= 14", "owner['completed'](job) >= 22"),
    ("upper = snapshot['collected_at']", "upper = production['collected_unix']"),
    ("'root-resource-production-'", "'small-GPU-resource-production-'"),
    ("effective_flags=override['effective_root_flags']",
     "effective_flags=dict(pin_root_host=False,pin_replay_host=True,offload_replay_mixer='False only for paired rows<=8 and extent<=2048; otherwise True')"),
    ("if m['step'] >= 14", "if m['step'] >= 22"),
    ('passively_observed_steps=[17, 18]', 'passively_observed_steps=[]'),
):
    assert old in code, old
    code = code.replace(old, new)
anchor = "    assert len(override['actual_workers']) == 8"
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + '''
    production_path=RUN/'runtime-overrides/actual-worker-imports-small-GPU-resident-production-v1.json'
    production=json.loads(production_path.read_bytes())
    assert sorted(r['rank'] for r in production['results'])==list(range(8))
    for row in production['results']:
        detail=row['small_GPU_resident_production']
        counters=detail['counters']
        assert counters['small_calls']>0 and counters['offloaded_calls']>0
        assert counters['largest_small_extent']<=2048
        assert counters['small_peak_allocated']<55*2**30
        assert detail['current_flags']==dict(pin_root_host=False,pin_replay_host=True,offload_replay_mixer=True)
        assert detail['installed_attribute_name']=='choose_original_mixer_storage'
        assert detail['resource_mutations']==0 and detail['additional_DT_calls']==0''')
anchor = "    target = destination / 'production-window-receipt.json'"
assert code.count(anchor) == 1
code = code.replace(anchor, '''    report['production_counter_source']=dict(path=str(production_path),SHA256=hashlib.sha256(production_path.read_bytes()).hexdigest())
    report['production_counters']=[dict(rank=r['rank'],pid=r['pid'],birth=r['birth'],detail=r['small_GPU_resident_production']) for r in production['results']]
    report['window_scope']='After the last installation and before completion of the single read-only counter RPC; excludes prior validation/profile scopes. Strict adjacent eight-rank phase matching remains required; this is an observed production window, not a matched-input controlled throughput trial.'
    report['interpretation']='Native GPU device activity, not a fraction of peak FLOPS. Different production input/token amounts prevent attributing the entire timing/Util difference to a kernel speedup. Resource selection adds no model/DT/CUDA/collective calls. The one original metadata RPC is recorded separately from native DT timing; boundary and mixed observations remain separate.'
''' + anchor)
if '--through-latest-progress' in sys.argv:
    assert "upper = production['collected_unix']" in code
    code = code.replace("upper = production['collected_unix']", "upper = snapshot['collected_at']")
    code = code.replace('After the last installation and before completion of the single read-only counter RPC; excludes prior validation/profile scopes.',
                        'After the last installation through the current original log collection, including later natural production phases. The separate counter RPC confirms iteration22 only. Excludes prior validation/profile scopes.')
namespace = {'__name__': 'original_small_GPU_resource_production', '__file__': str(owner)}
exec(compile(code, str(owner), 'exec'), namespace)
namespace['main']()
