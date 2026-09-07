"""Four unchanged finite-operator calls on pinned real operands; no model/quality run."""
import gc
import hashlib
import json
import os
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

A = Path(__file__).resolve().parent
P = Path('${ARTIFACT_ROOT}/codex_fa_two_sweeps_operator_20260907_v3')
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert sha(P / 'protocol.json') == '66581ca4b5ef3bb2a22a58d4a9a9e202f99219a11d191e6d12d2695a9730221b'
p = json.loads((P / 'protocol.json').read_text())
for name, expected in p['sources'].items():
    assert sha(P / name) == expected
assert sha(p['capture_results']) == p['capture_results_sha256']
assert sha(p['build_result']) == p['build_result_sha256']
build = json.loads(Path(p['build_result']).read_text())
assert sha(p['library']) == build['library']['sha256']
sys.path.insert(0, str(P))
os.environ['MACA_PATH'] = '/opt/maca'
import numpy as np
import torch
from vendor_fa_finite_shared_mean_reuse_runtime import VendorFAFiniteP1SharedMeanReuse
from vendor_fa_finite_two_sweeps_runtime import VendorFAFiniteP1TwoSweeps

record = json.loads(Path(p['capture_results']).read_text())['captured_layers'][0]
assert record == p['actual_operands']
ops = {}
for name, row in record['operands'].items():
    if name not in ['q0', 'q1', 'k0', 'k1', 'v0', 'u', 'lse0', 'lse1']:
        continue
    file = Path(p['capture_results']).parent / row['file']
    assert sha(file) == row['sha256']
    ops[name] = torch.from_numpy(np.load(file, allow_pickle=False)).cuda()
compact = dict(ops)
for name in ['k0', 'k1', 'v0']:
    compact[name] = ops[name][:, ::record['groups']].contiguous()
    assert torch.equal(compact[name].repeat_interleave(record['groups'], dim=1), ops[name])
methods = {'shared_mean': VendorFAFiniteP1SharedMeanReuse(p['old_library'], p['old_library_sha256']),
           'two_sweeps': VendorFAFiniteP1TwoSweeps(p['library'], build['library']['sha256'])}
r = {'status': 'running', 'study_sha256': sha(__file__), 'model_calls': 0, 'quality_queries': 0,
     'whole_attributions': 0, 'new_compiles': 0, 'operator_calls': [],
     'protocol': {'operator_call_budget': 4, 'order': ['shared_mean_warm', 'two_sweeps_warm', 'two_sweeps_profile', 'shared_mean_profile'],
                  'profiler_scope': 'GPU kernel durations and wrapper; profiled wall time is not an ordinary latency measurement.',
                  'source_parent_protocol_sha256': sha(P / 'protocol.json'),
                  'shape': list(compact['q0'].shape), 'fixed_input': 'Previously captured NI0 layer35 actual operands',
                  'no_parameter_sweep': True, 'automatic_full_model_expansion': False,
                  'actual_launches': 'Three kernels for each implementation; old three quadratic sweeps, new two quadratic sweeps plus linear conversion.',
                  'legacy_metadata_correction': 'Parent pass_count/shared_memory_bytes are stale inherited fields. Execution/source hashes are retained; those fields are not used to describe the final two-sweep kernel.'}}


def save():
    f = A / 'results.partial'
    f.write_text(json.dumps(r, indent=2))
    f.replace(A / 'results.json')


save()
outputs = {}
try:
    for profiled, names in [(False, ['shared_mean', 'two_sweeps']), (True, ['two_sweeps', 'shared_mean'])]:
        for name in names:
            assert len(r['operator_calls']) < 4
            gc.collect()
            torch.cuda.synchronize()
            activity = {}
            start = time.perf_counter()
            if profiled:
                with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as profiler:
                    with torch.profiler.record_function('finite_operator_' + name):
                        out = methods[name](compact, record['scale'], activity)
                    torch.cuda.synchronize()
            else:
                out = methods[name](compact, record['scale'], activity)
                torch.cuda.synchronize()
            row = {'method': name, 'profiled': profiled, 'seconds_including_profiler_if_enabled': time.perf_counter() - start,
                   'activity': activity}
            r['operator_calls'].append(row)
            save()
            if profiled:
                path = A / (name + '_profile.json')
                profiler.export_chrome_trace(str(path))
                events = json.loads(path.read_text())['traceEvents']
                kernels = [e for e in events if e.get('cat') == 'kernel']
                assert kernels, 'Profiler produced no kernel evidence'
                row['profile'] = {'file': path.name, 'sha256': sha(path),
                                  'kernel_events': [{'name': e['name'], 'dur_us': e['dur'], 'ts': e['ts']} for e in kernels],
                                  'category_counts': dict(Counter(e.get('cat', '') for e in events))}
                outputs[name] = {k: v.cpu().numpy() for k, v in out.items()}
            del out
            save()
    checks = {}
    for key, original in outputs['shared_mean'].items():
        new = outputs['two_sweeps'][key]
        a, b = original.astype(np.float64), new.astype(np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        rms = float(np.sqrt(np.mean(a * a)))
        diff = b - a
        checks[key] = {'exact': bool(np.array_equal(original, new)),
                       'relative_L2': float(np.sqrt(np.mean(diff * diff))) / max(rms, 1e-300),
                       'max_abs_over_old_RMS': float(np.max(np.abs(diff))) / max(rms, 1e-300),
                       'sign_flips': int(np.count_nonzero(a * b < 0))}
    r['numerics'] = checks
    assert all(checks[k]['exact'] for k in ['tau', 'center', 'dk', 'dv'])
    assert checks['dq']['relative_L2'] <= .001 and checks['dq']['max_abs_over_old_RMS'] <= .01
    r['status'] = 'four_call_phase_profile_complete'
except Exception:
    r['status'] = 'phase_profile_failed'
    r['error'] = traceback.format_exc()
finally:
    save()
    with zipfile.ZipFile(A / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for file in [Path(__file__), A / 'results.json', *A.glob('*_profile.json')]:
            z.write(file, file.name)
    print(json.dumps({'status': r['status'], 'operator_calls': len(r['operator_calls'])}), flush=True)
