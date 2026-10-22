"""Compose the original f71 process owner for this single fresh native run."""
import json
from pathlib import Path
from prepare_campaign import ROOT, AUDIT, PRIOR_AUDIT, RUN, VERSION, STATUS, sha


def ready():
    row = json.loads((AUDIT / 'dt-prepared.json').read_text())
    assert row['fresh_base_model'] and row['resume_checkpoint'] is None
    assert row['budget']['total_iterations'] == 330
    assert row['options']['trainer.resume_mode'] == 'disable'
    assert row['options']['trainer.resume_from_path'] is None
    assert json.loads((AUDIT / 'formal-CLI-verification.json').read_text())['status'] == 'passed_original_Hydra_parser_and_native_VERL_cfg_only'
    quality = json.loads((PRIOR_AUDIT / 'native-SSM-dtype-complete-quality-review-v1.json').read_text())
    assert quality['deployment_decision'] == 'accept_original_native_SSM_dtype_interface_fix'
    assert quality['native_actual_DP8_complete_case_verified']
    installed = json.loads((PRIOR_AUDIT / 'original-owner-SSM-dtype-installed-v1/deployment.json').read_text())
    assert installed['status'] == 'original_owner_SSM_argument_bridge_installed'
    assert all(sha(p) == h for p, h in row['source_bindings'].items())
    assert all(sha(p) == h for p, h in row['required_installed_source_bindings'].items())
    assert json.loads(Path(row['verification_receipts']['original_v10_CPU']).read_text())['status'] == 'passed'
    assert json.loads(Path(row['verification_receipts']['original_v10_native']).read_text())['status'] == 'completed'
    return row


if __name__ == '__main__':
    owner = ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2/source/run_campaign.py'
    assert sha(owner) == 'f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'
    code = owner.read_text()
    changes = {
        "'three_native_CPU_configs_resolved_and_common_workload_equal'": repr(STATUS),
        "with (AUDIT / 'campaign-controller.lock').open('a') as lock:": "with Path('/tmp/deltatrace-native-textcraft-v10-SSM-fp32-20261022-v1.lock').open('a') as lock:",
        "cwd=row['entry']": "cwd=row['verl_root']",
        "path = ROOT / name": "path = AUDIT / name",
        "owner['collect'](str(ROOT), [row['output']])": "owner['collect'](str(AUDIT), [row['output']])",
        "assert owner['completed'](current) == 200": "assert owner['completed'](current) == 330",
        "set(range(1, 201))": "set(range(1, 331))",
        "assert current['checkpoint_step'] == 200": "assert current['checkpoint_step'] == 330",
        "final = checkpoint / 'global_step_200'": "final = checkpoint / 'global_step_330'",
        "status='completed_200_native_iterations'": "status='completed_330_native_iterations'",
        "completed_iterations=200, final_native_checkpoint": "completed_iterations=330, final_native_checkpoint",
        "'all_three_native_200_iteration_runs_completed'": "'requested_v10_SSM_fp32_eight_gpu_native_330_iteration_run_completed'",
    }
    for before, after in changes.items():
        assert code.count(before) == (2 if before == "owner['collect'](str(ROOT), [row['output']])" else 1), before
        code = code.replace(before, after)
    (AUDIT / 'source/composed-native-process-owner.py').write_text(code)
    namespace = dict(__name__='v10_SSM_fp32_native_process_owner', __file__=str(owner))
    exec(compile(code, str(owner), 'exec'), namespace)
    namespace['main'](audit=AUDIT, version=VERSION, readiness=ready)
