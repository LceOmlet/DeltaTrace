"""Select the accepted storage regime in the unchanged physical/phase owner."""
from pathlib import Path

owner = Path(__file__).with_name('summarize_small_gpu_resource_production.py')
code = owner.read_text()
for old, new in (
    ('small-GPU-resident-resource-override-completed.json', 'validated-16384-GPU-resource-override-completed.json'),
    ('completed_runtime_bounded_original_storage_selection', 'completed_runtime_verified_16384_original_storage_selection'),
    ('small-GPU-resident-production-v1', 'validated-16384-GPU-production-v1'),
    ('small_GPU_resident_production', 'validated_GPU_resource_production'),
    ("counters['small_calls']>0 and counters['offloaded_calls']>0", "counters['small_calls']>0"),
    ("counters['largest_small_extent']<=2048", "counters['largest_small_extent']<=16384"),
    ("        assert counters['small_peak_allocated']<55*2**30\n", ''),
    ('extent<=2048', 'extent<=16384'),
    ("owner['completed'](job) >= 22", "owner['completed'](job) >= 37"),
    ("if m['step'] >= 22", "if m['step'] >= 37"),
    ('small-GPU-resource-production-', 'validated-16384-GPU-production-'),
    ('The separate counter RPC confirms iteration22 only.', 'The separate counter RPC confirms its recorded native iteration only.'),
):
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())
