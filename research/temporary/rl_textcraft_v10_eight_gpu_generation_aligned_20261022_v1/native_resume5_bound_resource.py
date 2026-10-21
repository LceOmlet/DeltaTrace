"""Bind one explicit native continuation to the complete accepted DT resource source.

Reuse the existing preparation, readiness, process and session owners.  This
adapter changes their attempt identities and the demonstrated DT_ROOT mismatch;
it supplies no training, checkpoint, attribution or retry implementation.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).parent
PREFIX = 'resume5-bound-resource-v2'
BASES = {
    'run_native_resume5.py': 'f9891106b251731ead60b9f4e5037cd1fab31d302f4554010ff2077f92b4f371',
    'prepare_native_resume5.py': '3bf92b995e4472326fb4b48e56a8f7a1d63c420116633005dcbdb2f32973c35e',
    'submit_native_resume5.py': '9f92d3cbd62a07e5cded8b8bcf9425fd68f30b82ee1fed32f0f04d41cd00b2e2',
}


def source_text(name):
    data = (SOURCE / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == BASES[name], name
    return data.decode()


def replace_one(code, before, after):
    assert code.count(before) == 1, before
    return code.replace(before, after)


def rebind_attempt(code):
    # These are the terminal attempt's identities, never a process selection rule.
    for before, after in (
        ('composed-native-resume5-resource-owner.py', 'composed-native-bound-resource-owner.py'),
        ("'resume5-", "'" + PREFIX + '-'),
        ("'runtime-overrides/accepted-storage-resume5-v1'",
         "'runtime-overrides/accepted-storage-bound-resource-v2'"),
        ('2654663', '3608766'), ('1792605592.06', '1792614357.67'),
        ('2654698', '3608813'), ('1792605592.43', '1792614358.05'),
        ('attempt2-native-DT-host-allocation-failure', 'attempt3-native-DT-resource-source-guard-failure'),
        ('failed_exit1_DT_host_pinned_allocation', 'failed_exit1_before_DT_resource_runner_SHA_guard'),
        ('from run_native_resume5 import ', 'from native_resume5_bound_resource import '),
    ):
        code = code.replace(before, after)
    return code


# Expose the same original readiness/controller functions under this attempt's
# receipt names.  Exclude the base file's main block, so importing never starts it.
_base = rebind_attempt(source_text('run_native_resume5.py'))
_tree = ast.parse(_base)
assert isinstance(_tree.body[-1], ast.If)
assert ast.unparse(_tree.body[-1].test) == "__name__ == '__main__'"
exec(compile(ast.Module(body=_tree.body[:-1], type_ignores=[]),
             str(SOURCE / 'run_native_resume5.py'), 'exec'), globals())

OLD_DT = ROOT / 'candidates/textcraft-dt-reward-target-exponential-20261016-v5/deltatrace'
ACCEPTED_DT = ROOT / 'candidates/dt-native-replay-host-lifetime-20261017-v3/deltatrace'
RUNNER = 'clean/qwen35/qwen35_dense_finite_runner.py'
ACCEPTED_RUNNER_SHA = '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'


def bind_complete_accepted_resource(row):
    """Reuse the accepted source and options; do not relax the original SHA guard."""
    environment = row['effective_environment']
    assert environment['DT_ROOT'] == str(OLD_DT)
    assert sha(ACCEPTED_DT / RUNNER) == ACCEPTED_RUNNER_SHA
    original = json.loads((ROOT / 'runs/textcraft-dtpo-v5-exponential-verified-8gpu-200-20261020-v1/source.json').read_text())
    accepted_environment = original['effective_environment']
    assert accepted_environment['DT_ROOT'] == str(ACCEPTED_DT)
    assert accepted_environment['DT_NATIVE_HOST_CACHE_BATCH'] == '1'
    assert accepted_environment['DT_NATIVE_HOST_CACHE_REPLAY_LAYER'] == '0'
    environment['PYTHONPATH'] = environment['PYTHONPATH'].replace(str(OLD_DT), str(ACCEPTED_DT))
    environment.update(DT_ROOT=str(ACCEPTED_DT), DT_NATIVE_HOST_CACHE_BATCH='1',
                       DT_NATIVE_HOST_CACHE_REPLAY_LAYER='0')
    row['resource_source_binding_correction'] = dict(
        prior_DT_ROOT=str(OLD_DT), accepted_DT_ROOT=str(ACCEPTED_DT),
        accepted_runner_SHA256=ACCEPTED_RUNNER_SHA, original_SHA_guards_unchanged=True,
        source_difference='104 of 105 Python files identical; only existing host-cache lifecycle statements differ',
        numerical_method_changed=False, reference='resource-DT-root-complete-python-source-diff.json')
    row['orchestration_source_bindings'][str(Path(__file__))] = sha(__file__)


def action_source(action):
    code = rebind_attempt(source_text(action + '_native_resume5.py'))
    if action == 'prepare':
        assert "'candidates/textcraft-v10-eight-gpu-native-resume5-resource-20261022-v1'" in code
        code = replace_one(code, '    bindings = {}',
            '    from native_resume5_bound_resource import bind_complete_accepted_resource, OLD_DT, ACCEPTED_DT, RUNNER, ACCEPTED_RUNNER_SHA\n'
            '    bind_complete_accepted_resource(row)\n    bindings = {}')
        code = replace_one(code,
            "mapped = str(path).replace(str(PARENT), str(CANDIDATE))",
            "mapped = str(path).replace(str(PARENT), str(CANDIDATE)).replace(str(OLD_DT), str(ACCEPTED_DT))")
        code = replace_one(code,
            "expected = prepared['source_bindings'].get(mapped, digest)",
            "expected = ACCEPTED_RUNNER_SHA if mapped == str(ACCEPTED_DT / RUNNER) else prepared['source_bindings'].get(mapped, digest)")
        code = replace_one(code, "        else:\n            argv.append(argument)",
            "        elif key not in additions:\n            argv.append(argument)")
        code = replace_one(code,
            "'+ray_init.runtime_env.env_vars.DT_ACCEPTED_STORAGE_RECEIPTS': str(receipt_root)}",
            "'+ray_init.runtime_env.env_vars.DT_ACCEPTED_STORAGE_RECEIPTS': str(receipt_root),\n"
            "                 '+ray_init.runtime_env.env_vars.DT_NATIVE_HOST_CACHE_BATCH': '1',\n"
            "                 '+ray_init.runtime_env.env_vars.DT_NATIVE_HOST_CACHE_REPLAY_LAYER': '0'}")
        code = replace_one(code,
            "'ray_init.runtime_env.env_vars.DT_ACCEPTED_STORAGE_PROFILE': PROFILE}",
            "'ray_init.runtime_env.env_vars.DT_ACCEPTED_STORAGE_PROFILE': PROFILE,\n"
            "                'ray_init.runtime_env.env_vars.DT_NATIVE_HOST_CACHE_BATCH': '1',\n"
            "                'ray_init.runtime_env.env_vars.DT_NATIVE_HOST_CACHE_REPLAY_LAYER': '0'}")
    elif action == 'submit':
        code = replace_one(code,
            "    for base, names, prefix in ((AUDIT, audit_files, 'audit'), (RUN, run_files, 'run')):",
            "    audit_files += ['resume5-controller-job.json', 'resume5-controller.stdout', 'resume5-controller.stderr',\n"
            "                    'resume5-dt-prepared.json', 'resume5-campaign-prepared.json',\n"
            "                    'resume5-preparation-completed.json', 'resume5-CLI-verification.json',\n"
            "                    'source/run_native_resume5.py', 'source/prepare_native_resume5.py',\n"
            "                    'source/submit_native_resume5.py', 'source/composed-native-resume5-resource-owner.py']\n"
            "    for base, names, prefix in ((AUDIT, audit_files, 'audit'), (RUN, run_files, 'run')):")
        code = replace_one(code,
            "str(AUDIT / 'source/run_native_resume5.py')],",
            "str(AUDIT / 'source/native_resume5_bound_resource.py'), 'run'],")
        code = replace_one(code,
            "source=str(AUDIT / 'source/run_native_resume5.py')",
            "source=str(AUDIT / 'source/native_resume5_bound_resource.py')")
        code = replace_one(code,
            "source_SHA256=sha(AUDIT / 'source/run_native_resume5.py')",
            "source_SHA256=sha(AUDIT / 'source/native_resume5_bound_resource.py')")
    else:
        raise ValueError(action)
    compile(code, str(SOURCE / (action + '_native_resume5.py')), 'exec')
    return code


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'run':
        assert sha(AUDIT / 'source/composed-native-bound-resource-owner.py') == hashlib.sha256(controller_source().encode()).hexdigest()
        namespace = dict(__name__='original_native_bound_resource_process_owner', __file__=str(PROCESS_OWNER))
        exec(compile(controller_source(), str(PROCESS_OWNER), 'exec'), namespace)
        namespace['main'](audit=AUDIT, version=VERSION, readiness=ready)
    elif action == 'ready':
        row = ready()
        assert row['effective_environment']['DT_ROOT'] == str(ACCEPTED_DT)
        assert row['effective_environment']['DT_NATIVE_HOST_CACHE_BATCH'] == '1'
        assert row['effective_environment']['DT_NATIVE_HOST_CACHE_REPLAY_LAYER'] == '0'
        print(json.dumps(dict(status='ready_for_one_same_run_native_bound_resource_resume',
                             previous_pid=row['previous_job_identity']['pid'],
                             complete_prefix=row['resumed_completed_iteration'], budget=row['budget']['total_iterations'])))
    else:
        exec(compile(action_source(action), str(SOURCE / (action + '_native_resume5.py')), 'exec'),
             dict(__name__='__main__', __file__=str(SOURCE / (action + '_native_resume5.py'))))
