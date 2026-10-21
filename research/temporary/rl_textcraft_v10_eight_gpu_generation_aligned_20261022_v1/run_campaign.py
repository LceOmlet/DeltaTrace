"""Compose the existing native process owner for one fresh v10 DP8 run.

Prepared only: the generation review, resolved CLI and bound source receipts
must exist before this adapter can launch the original training command.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
VERSION = 'textcraft-dtpo-v10-target-logprob-8gpu-formal-20261022-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready():
    row = json.loads((AUDIT / 'dt-prepared.json').read_text())
    receipts = row['verification_receipts']
    assert json.loads(Path(receipts['original_v10_CPU']).read_text())['status'] == 'passed'
    assert json.loads(Path(receipts['original_v10_native']).read_text())['status'] == 'completed'
    assert json.loads((AUDIT / 'formal-CLI-verification.json').read_text())['status'] == 'passed_original_Hydra_parser_and_native_VERL_cfg_only'
    review = json.loads((AUDIT / 'generation-controls-ready.json').read_text())
    assert review['status'] == 'ready_for_formal_training_after_full_native_generation_review'
    assert row['fresh_base_model'] and row['resume_checkpoint'] is None
    assert row['budget']['total_iterations'] == 330
    assert row['options']['trainer.n_gpus_per_node'] == 8
    for path, digest in row['source_bindings'].items():
        assert sha(path) == digest, path
    for path, digest in review['deployed_control_sources'].items():
        assert sha(path) == digest, path


if __name__ == '__main__':
    owner = ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2/source/run_campaign.py'
    assert sha(owner) == 'f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'
    code = owner.read_text()
    replacements = {
        "'three_native_CPU_configs_resolved_and_common_workload_equal'": "'v10_eight_GPU_formal_configuration_resolved'",
        "with (AUDIT / 'campaign-controller.lock').open('a') as lock:":
            "with Path('/tmp/deltatrace-native-textcraft-v10-eight-gpu-generation-aligned-20261022-v1.lock').open('a') as lock:",
        "cwd=row['entry']": "cwd=row['verl_root']",
        "path = ROOT / name": "path = AUDIT / name",
        "owner['collect'](str(ROOT), [row['output']])": "owner['collect'](str(AUDIT), [row['output']])",
        "assert owner['completed'](current) == 200": "assert owner['completed'](current) == 330",
        "set(range(1, 201))": "set(range(1, 331))",
        "assert current['checkpoint_step'] == 200": "assert current['checkpoint_step'] == 330",
        "final = checkpoint / 'global_step_200'": "final = checkpoint / 'global_step_330'",
        "status='completed_200_native_iterations'": "status='completed_330_native_iterations'",
        "completed_iterations=200, final_native_checkpoint": "completed_iterations=330, final_native_checkpoint",
        "'all_three_native_200_iteration_runs_completed'": "'requested_v10_eight_gpu_native_330_iteration_run_completed'",
    }
    for before, after in replacements.items():
        expected = 2 if before == "owner['collect'](str(ROOT), [row['output']])" else 1
        assert code.count(before) == expected, (before, code.count(before))
        code = code.replace(before, after)
    (AUDIT / 'source/composed-native-process-owner.py').write_text(code)
    namespace = {'__name__': 'v10_eight_gpu_native_process_owner', '__file__': str(owner)}
    exec(compile(code, str(owner), 'exec'), namespace)
    namespace['main'](audit=AUDIT, version=VERSION, readiness=ready)
