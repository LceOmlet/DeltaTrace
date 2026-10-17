"""Rebind the pinned six-run preparation; no launch or training-loop replacement."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-native-dp-owner-20261018-v1'
PRIOR = ROOT / 'receipts/six-run-fresh-after-register-threads-20261017-v1'
DP_AUDIT = ROOT / 'receipts/textcraft-native-dp-owner-20261018-v1'
CANDIDATE = ROOT / 'candidates/textcraft-native-dp-owner-20261018-v1'


def main():
    assert not (AUDIT/'six-combination-manifest.json').exists()
    AUDIT.mkdir(parents=True, exist_ok=True)
    for name in ('native-resource-owner-prepared.json', 'native-resource-owner-deployed.json'):
        (AUDIT/name).write_bytes((PRIOR/name).read_bytes())
    source = ROOT/'receipts/six-run-fresh-after-efficiency-20261017-v1/source/prepare_fresh_queue.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '3cf1500344a22215986cb69ad2a951be86daf77b22e9fb2ef432116d78b33355'
    code = source.read_text().replace("'receipts/six-run-fresh-after-efficiency-20261017-v1'",
        "'receipts/six-run-fresh-native-dp-owner-20261018-v1'")
    code = code.replace("'-eight-gpu-sequential-20261017-v2'", "'-eight-gpu-sequential-20261018-v4'")
    code = code.replace("old_row['version'].removesuffix('-v1') + '-v2'",
        "old_row['version'].removesuffix('-20261017-v1') + '-20261018-v4'")
    code = code.replace('expandable_segments:False,pinned_use_cuda_host_register:True',
        'expandable_segments:False,pinned_use_cuda_host_register:True,pinned_num_register_threads:8')
    anchor = "            assert row['options']['trainer.resume_mode'] == 'disable'\n"
    assert code.count(anchor) == 1
    code = code.replace(anchor, '''            if task == 'TextCraft':
                candidate = ROOT/'candidates/textcraft-native-dp-owner-20261018-v1'
                previous_verl = row['effective_environment']['VERL_ROOT']
                row['effective_environment']['VERL_ROOT'] = str(candidate/'verl-owner')
                row['effective_environment']['PYTHONPATH'] = str(candidate/'entry')+':'+str(candidate/'verl-owner')+':'+row['effective_environment']['PYTHONPATH']
                bindings = {}
                for path, value in row['source_bindings'].items():
                    if path.startswith(previous_verl+'/'):
                        path = str(candidate/'verl-owner')+path[len(previous_verl):]
                    bindings[path] = value
                for name in ('textcraft_owner_rollout.py', 'owner_environment_configs.json'):
                    bindings.pop(str(Path(row['entry'])/name))
                    bindings[str(candidate/'entry'/name)] = sha(candidate/'entry'/name)
                bindings[str(candidate/'entry/textcraft_native_dp_owner.py')] = sha(candidate/'entry/textcraft_native_dp_owner.py')
                bindings[str(candidate/'verl-owner/verl/workers/fsdp_workers.py')] = sha(candidate/'verl-owner/verl/workers/fsdp_workers.py')
                row['source_bindings'] = bindings
                row['native_DP_episode_owner_receipt'] = str(ROOT/'receipts/textcraft-native-dp-owner-20261018-v1/native-placement-accepted.json')
''' + anchor)
    code = code.replace("resource_version='native-host-register-and-MRoPE-row-copy-20261017-v1'",
        "resource_version='native-host-register-threads8-MRoPE-and-native-DP-owner-20261018-v4'")
    # Original owner builds argv, resolves all six Hydra configs and checks
    # task workload equality. No numerical section is edited here.
    exec(compile(code, str(source), 'exec'), dict(__name__='__main__', __file__=str(source)))
    plan = json.loads((AUDIT/'six-combination-manifest.json').read_bytes())
    plan.update(native_placement_acceptance=str(DP_AUDIT/'native-placement-accepted.json'),
        status='prepared_not_started', efficiency_candidate_not_accepted_at_preparation=True,
        launch_requires_completed_native_placement_acceptance=True,
        tuning_history=str(ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v3/campaign-state.json'))
    (AUDIT/'six-combination-manifest.json').write_text(json.dumps(plan, indent=2)+'\n')


if __name__ == '__main__':
    main()
