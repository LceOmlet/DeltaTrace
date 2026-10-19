"""Select the first natural production window with the original Util reducer.

The original dmon/phase parser, native log reader, and checkpoint criterion
remain unchanged. Only the applied resource identity and window are selected.
"""
import hashlib
from pathlib import Path
import sys

owner = Path(__file__).with_name('summarize_root_resource_production.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '6a2316e649ff144f6749b6725713a316b4a5090e12a7c00f1c7b53123e80ee9a'
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
    code = code.replace("upper = production['collected_unix']", '''upper = snapshot['collected_at']
    validation_window_selection=[]
    for report_name,job_name in (
            ('existing-DT-FA-capture-resource-pair-results.json','FA-capture-long-pair-job.json'),
            ('existing-DT-long-GPU-resident-resource-pair-results.json','long-GPU-resident-pair-job.json'),
            ('existing-DT-capture-storage-results.json','capture-storage-observer-job.json')):
        report_path=AUDIT/report_name
        job_path=AUDIT/job_name
        if report_path.exists():
            window_report=json.loads(report_path.read_bytes())
            scopes=window_report.get('restored_scopes',[])
            timing_rows=window_report.get('native_timing_records',[])
            if (len(scopes)==8 and all(s['value']['restored_original_methods'] for s in scopes)):
                lower=max(lower,max(s['value']['finished'] for s in scopes))
                validation_window_selection.append(dict(policy='after_all_validation_scopes_restored',source=str(report_path),lower=lower))
            elif timing_rows or scopes:
                cutoff=min(r['value']['started'] for r in timing_rows+scopes)
                upper=min(upper,cutoff)
                validation_window_selection.append(dict(policy='before_active_validation_readout',source=str(report_path),upper=upper))
        elif job_path.exists():
            validation_job=json.loads(job_path.read_bytes())
            upper=min(upper,validation_job['submitted'])
            validation_window_selection.append(dict(policy='before_pending_validation_submission',source=str(job_path),upper=upper))
    assert lower<upper,'No fresh clean physical window yet; do not substitute the validation samples' ''')
    code = code.replace("str(int(upper))", "str(int(snapshot['collected_at']))")
    code = code.replace("    report['production_counter_source']=", "    report['validation_window_selection']=validation_window_selection\n    report['production_counter_source']=")
    code = code.replace('After the last installation and before completion of the single read-only counter RPC; excludes prior validation/profile scopes.',
                        'Applied-storage production interval selected outside the new validation callback: before pending/active validation, or after all eight scopes restore. The separate counter RPC confirms iteration22 only. Exact lower/upper selection is retained.')
namespace = {'__name__': 'original_small_GPU_resource_production', '__file__': str(owner)}
exec(compile(code, str(owner), 'exec'), namespace)
namespace['main']()
