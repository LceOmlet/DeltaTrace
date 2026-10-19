"""Reuse the native JSON-metadata reader for replay capture byte counts."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('collect_existing_DT_timings.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'b841476e54813430a7991dae0db0125873cb61dd620179c6b52e5a088d2ce228'
code = owner.read_text()
for old, new in (
    ('native_DT_existing_timing', 'native_DT_capture_storage'),
    ('existing-DT-native-stage-timings.json', 'existing-DT-capture-storage-results.json'),
    ('results_dtpo_v9_negative2_existing_DT_timings_20261019_v4.json',
     'results_dtpo_v9_negative2_capture_storage_20261019_v4.json'),
):
    assert old in code
    code = code.replace(old, new)
anchor = 'if rows or scopes or errors:'
assert code.count(anchor) == 1
code = code.replace(anchor, '''report['capture_storage_coverage_complete']=bool(complete and rows and all(
    r['value']['replay_capture_storage']['complete_layer_capture_counts']
    and not r['value']['replay_capture_storage']['metadata_errors'] for r in rows)
    and all(s['value']['capture_class_methods_restored'] for s in scopes))
report['training_iterations_excluded_from_production_efficiency']=sorted({r['value']['training_iteration'] for r in rows+scopes})
report['maximum_extra_capture_storage_by_rank']={rank:max(
    (dict(extra_bytes=r['value']['replay_capture_storage']['conservative_extra_replay_storage_bytes'],
          paired_shape=r['value']['paired_shape'],minibatch=r['value']['minibatch'],
          training_iteration=r['value']['training_iteration'],path=r['path'],line=r['line'])
     for r in records),key=lambda x:x['extra_bytes']) for rank,records in by_rank.items()}
report['metadata_only']=True
report['destination_CPU_storage_only']=True
report['GPU_retention_source_storage_measured']=False
report['new_tensor_copies']=0
report['new_CUDA_calls']=0
report['new_collectives']=0
report['capacity_pass_claimed']=False
report['interpretation'].append('Storage metadata only, not tensor contents. The legacy extra_bytes fields sum destination CPU decoder/mixer storage; they do not bound the original GPU parent storage borrowed by a view. They are not total physical peak or unobserved input capacity. No numerical operator or resource flag changed.')
''' + anchor)
anchor = 'print(json.dumps(brief, ensure_ascii=False))'
assert code.count(anchor) == 1
code = code.replace(anchor, "brief['storage_coverage_complete']=report['capture_storage_coverage_complete']\n"
    "brief['maximum_extra_capture_storage_by_rank']=report['maximum_extra_capture_storage_by_rank']\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
