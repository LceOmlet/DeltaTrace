"""Configure the existing native process owner for this independent DP2 run."""
import json
from prepare_campaign import ROOT,AUDIT,VERSION,sha


def ready():
    row=json.loads((AUDIT/'dt-prepared.json').read_bytes())
    verification=json.loads((AUDIT/'native-behavior-status.json').read_bytes())
    assert verification['status']=='completed'
    assert json.loads((AUDIT/'CPU-behavior-verification.json').read_bytes())['status']=='passed'
    assert json.loads((AUDIT/'formal-CLI-verification.json').read_bytes())['status']=='passed_original_Hydra_parser_and_native_VERL_cfg_only'
    assert json.loads((AUDIT/'owner-constructor-verification.json').read_bytes())['status']=='passed_original_environment_factory_construction_only'
    assert row['fresh_base_model'] and row['resume_checkpoint'] is None
    for path,digest in row['source_bindings'].items():
        assert sha(path)==digest,path


if __name__=='__main__':
    owner=ROOT/'receipts/two-task-eight-gpu-sequential-20261017-v2/source/run_campaign.py'
    assert sha(owner)=='f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'
    code=owner.read_text()
    replacements={
      "'three_native_CPU_configs_resolved_and_common_workload_equal'":"'v10_target_endpoint_logprob_DP2_formal_configuration_resolved'",
      "with (AUDIT / 'campaign-controller.lock').open('a') as lock:":"with Path('/tmp/deltatrace-native-textcraft-v10-target-logprob-20261021-v1.lock').open('a') as lock:",
      "cwd=row['entry']":"cwd=row['verl_root']",
      "for device in range(8):":"for device in (2,3):",
      "path = ROOT / name":"path = AUDIT / name",
      "owner['collect'](str(ROOT), [row['output']])":"owner['collect'](str(AUDIT), [row['output']])",
      "assert owner['completed'](current) == 200":"assert owner['completed'](current) == 330",
      "set(range(1, 201))":"set(range(1, 331))",
      "assert current['checkpoint_step'] == 200":"assert current['checkpoint_step'] == 330",
      "final = checkpoint / 'global_step_200'":"final = checkpoint / 'global_step_330'",
      "f'{kind}_world_size_8_rank_{rank}.pt'":"f'{kind}_world_size_2_rank_{rank}.pt'",
      "for role in roles for rank in range(8)":"for role in roles for rank in range(2)",
      "status='completed_200_native_iterations'":"status='completed_330_native_iterations'",
      "completed_iterations=200, final_native_checkpoint":"completed_iterations=330, final_native_checkpoint",
      "'all_three_native_200_iteration_runs_completed'":"'requested_v10_two_gpu_native_330_iteration_run_completed'",
    }
    for before,after in replacements.items():
        expected=2 if before=="owner['collect'](str(ROOT), [row['output']])" else 1
        assert code.count(before)==expected,(before,code.count(before))
        code=code.replace(before,after)
    (AUDIT/'source/composed-native-process-owner.py').write_text(code)
    namespace={'__name__':'v10_native_process_owner','__file__':str(owner)}
    exec(compile(code,str(owner),'exec'),namespace)
    namespace['main'](audit=AUDIT,version=VERSION,readiness=ready)
