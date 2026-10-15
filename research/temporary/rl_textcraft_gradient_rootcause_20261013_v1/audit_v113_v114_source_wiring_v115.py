"""Expand local orchestration wrappers without executing their operational code.

Only wrapper string transformations run. Leaf bodies and IO/process preambles
are retained as text, compiled for syntax, and inspected for the named wiring.
This is deployment preparation, not a training/quality/root-cause experiment.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
real_compile, real_exec = compile, exec
dependencies = {}


def source_assignment(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    return (isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'source'
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == 'with_name')


def expand(code, filename, level=0):
    assert level < 20
    file = Path(filename)
    assert file.parent == HERE
    dependencies[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    tree = ast.parse(code, filename=str(file))
    indices = [i for i,n in enumerate(tree.body) if source_assignment(n)]
    if not indices:
        return code
    assert len(indices) == 1
    index = indices[0]
    before = ast.unparse(ast.Module(body=tree.body[:index], type_ignores=[]))
    suffix = ast.Module(body=tree.body[index:], type_ignores=[])
    result = []
    def capture_compile(text, path, mode, *args, **kwargs):
        assert isinstance(text, str) and mode == 'exec'
        return text, path
    def capture_exec(payload, *args, **kwargs):
        text, path = payload
        result.append(expand(text, path, level+1))
    namespace = dict(__file__=str(file), __name__='static_orchestration_expansion',
                     Path=Path, hashlib=hashlib, json=json, sys=sys,
                     compile=capture_compile, exec=capture_exec)
    # Native process/SSH/torch operations occur only in skipped preambles or
    # leaf bodies. Do not execute either, even when compilation is successful.
    real_exec(real_compile(suffix, str(file), 'exec'), namespace)
    assert len(result) == 1
    return before+'\n'+result[0]


names = [
    'prepare_PPO_pause_plan_v114.py', 'stop_PPO_for_same_session_probe_v114.py',
    'resume_PPO_after_same_session_probe_v114.py', 'queue_native_PPO_restore_v114.py',
    'record_native_PPO_restore_v114.py', 'launch_same_session_saved_update_feedback_v113.py',
    'observe_saved_update_natural_PG_v99.py',
    'wait_PPO_native_checkpoint_for_v113.py',
]
expanded = {}
for name in names:
    p = HERE/name
    text = expand(p.read_text(encoding='utf-8'), p)
    real_compile(text, name+'.review-only', 'exec')
    out = HERE/(name+'.v115-review-only.txt')
    out.write_text(text, encoding='utf-8')
    expanded[name] = dict(path=str(out), SHA256=hashlib.sha256(text.encode()).hexdigest(), text=text)

def compact(name):
    return ''.join(expanded[name]['text'].split())

plan = compact(names[0]);stop = compact(names[1]);resume = compact(names[2])
queue = compact(names[3]);record = compact(names[4]);launch = compact(names[5]);observer = compact(names[6])
assert 'assertstep>=30,' in plan and 'mmap=True' in plan
assert 'PPO-native-pause-for-same-session-v113-plan-v114.json' in plan
assert 'native-PPO-resume-after-same-session-v113-v114-prepared.json' in stop
assert 'live_protected' in stop and 'os.killpg' in stop
assert 'PPO-temporary-stop-for-same-session-v113-v114.json' in resume
assert "diag=OUT/'native-same-session-saved-update-feedback-v113'" in resume
assert 'previous_job_identity' in resume
assert 'native-PPO-resume-after-same-session-v113-v114-job.json' in queue
assert "versionin('v114',)" in queue and "sys.argv.append('v114')" in queue
assert "role['scheduler_last_epoch']==restore_step" in record
assert "role['optimizer_state_steps']==[float(4*restore_step)]" in record
assert 'resume_PPO_after_same_session_probe_v114.py' in record
assert 'native-PPO-restore-verification-v114.json' in record
assert 'native-same-session-saved-update-feedback-v113' in launch
assert "env['CUDA_VISIBLE_DEVICES']='2,3'" in launch
assert 'observe_saved_update_natural_PG_v99.py' in launch
assert 'defload_diagnostic_partition(' in observer and 'native_forward_inputs' in observer
assert 'native_buffers' in observer and 'capture_boundary_seed' in observer
waiter = compact(names[7])
assert '(2666718,1792028884.24)' in waiter and 'required_checkpoint=30' in waiter
assert 'verified-PPO-native-checkpoint-wait-v113-v114.json' in waiter
assert 'signals_sent=0' in waiter
manifest_path = HERE/'same-session-saved-update-feedback-v113-final-prepared.json'
manifest = json.loads(manifest_path.read_text())['report']
assert len(manifest['cases']) == 6
assert sum(bool(c.get('boundary_seed_source_case')) for c in manifest['cases']) == 5
assert manifest['planned_native_full_policy_derivatives'] == 1
assert manifest['planned_native_fixed_seed_derivatives'] == 5
assert hashlib.sha256((HERE/'observe_saved_update_natural_PG_v99.py').read_bytes()).hexdigest() == manifest['existing_observer_SHA256']
report = dict(unix=time.time(), scope=__doc__, status='static_source_wiring_compiled_not_operationally_executed',
    root_helpers=names, dependencies=dependencies,
    expanded={k:{i:v for i,v in value.items() if i!='text'} for k,value in expanded.items()},
    prepared_manifest=dict(path=str(manifest_path), SHA256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()),
    native_derivatives_planned=dict(actual_PG=1, fixed_same_session_seed=5),
    paused_PPO=False, GPU_submitted=False, model_calls=0, optimizer_calls=0,
    limits=['This checks local source transformations, named paths and dynamic restore wiring. It does not prove runtime checkpoint completeness or actual restored state.',
            'Operational preambles and leaf bodies were never executed; expanded review text is not a deployed replacement implementation.',
            'Actual checkpoint, PID/birth, occupancy, imports and native state readback remain required when the authorized workflow runs.'])
out = HERE/'v113-v114-source-wiring-v115.json'
out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(out.resolve()),status=report['status'],dependency_files=len(dependencies))))
