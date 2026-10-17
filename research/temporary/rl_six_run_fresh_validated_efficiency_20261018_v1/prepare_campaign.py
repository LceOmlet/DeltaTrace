"""Rebind the original six-run preparation to selected, measured native resources."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1'
PRIOR = ROOT/'receipts/six-run-fresh-after-register-threads-20261017-v1'


def main():
    assert not (AUDIT/'six-combination-manifest.json').exists()
    selected = json.loads((AUDIT/'selected-native-resources.json').read_bytes())
    assert selected['status'] == 'selected_measured_native_resources_for_fresh_six'
    assert selected['rollout_TP'] == 1 and selected['native_DP_placement_candidate'] is False
    assert selected['method_changes'] == selected['sampling_workload_changes'] == 0
    for path, expected in selected['source_bindings'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    for name in ('native-resource-owner-prepared.json', 'native-resource-owner-deployed.json'):
        (AUDIT/name).write_bytes((PRIOR/name).read_bytes())
    source = ROOT/'receipts/six-run-fresh-after-efficiency-20261017-v1/source/prepare_fresh_queue.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '3cf1500344a22215986cb69ad2a951be86daf77b22e9fb2ef432116d78b33355'
    code = source.read_text().replace("'receipts/six-run-fresh-after-efficiency-20261017-v1'",
        "'receipts/six-run-fresh-validated-efficiency-20261018-v1'")
    code = code.replace("'-eight-gpu-sequential-20261017-v2'", "'-eight-gpu-sequential-20261018-v5'")
    code = code.replace("old_row['version'].removesuffix('-v1') + '-v2'",
        "old_row['version'].removesuffix('-20261017-v1') + '-20261018-v5'")
    code = code.replace('expandable_segments:False,pinned_use_cuda_host_register:True',
        'expandable_segments:False,pinned_use_cuda_host_register:True,pinned_num_register_threads:8')
    code = code.replace('native-host-register-and-MRoPE-row-copy-20261017-v1',
        'native-measured-threads8-MRoPE-TP1-20261018-v5')
    # Any additional selected engine configuration goes through the existing
    # native owner_command and Hydra override interface, equally for all three
    # algorithms in a task. Empty overrides retain the validated baseline.
    engine_options = selected['rollout_engine_options']
    if engine_options:
        anchor = "            assert row['options']['trainer.resume_mode'] == 'disable'\n"
        assert code.count(anchor) == 1
        code = code.replace(anchor,
            "            row['options'].update(" + repr(engine_options) + ")\n" + anchor)
        anchor = "            for section in ('data', 'env', 'actor_rollout_ref', 'critic', 'algorithm'):\n"
        assert code.count(anchor) == 1
        update = ""
        for key, value in engine_options.items():
            assert key.startswith('+actor_rollout_ref.rollout.engine_kwargs.vllm.')
            field = key.rsplit('.', 1)[1]
            update += "            prior['actor_rollout_ref']['rollout']['engine_kwargs']['vllm'][" + repr(field) + "] = " + repr(value) + "\n"
        code = code.replace(anchor, update + anchor)
    exec(compile(code, str(source), 'exec'), dict(__name__='__main__', __file__=str(source)))
    path = AUDIT/'six-combination-manifest.json'
    plan = json.loads(path.read_bytes())
    plan.update(selected_native_resources=str(AUDIT/'selected-native-resources.json'),
        execution_version='20261018-v5', DT_method_version='v6',
        prior_tuning_and_unaccepted_candidates_excluded=True,
        prepare_commit=os.environ['RL_CAMPAIGN_COMMIT'])
    path.write_text(json.dumps(plan, indent=2)+'\n')


if __name__ == '__main__':
    main()
