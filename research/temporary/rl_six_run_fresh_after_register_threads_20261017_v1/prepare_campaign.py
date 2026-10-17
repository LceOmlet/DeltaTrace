"""Bind the accepted six-run preparation to fresh v3 identities and native threads8."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
PRIOR = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1'
AUDIT = ROOT / 'receipts/six-run-fresh-after-register-threads-20261017-v1'


def main():
    assert not (AUDIT / 'six-combination-manifest.json').exists()
    proof = json.loads((PRIOR / 'register-threads-ABBA-summary-v1.json').read_bytes())
    assert len(proof['workers']) == 8
    for row in proof['workers']:
        assert row['raw_equal'] and row['max_abs'] == 0
        assert row['rng_unchanged'] and row['restored']
        assert row['scheduler_unchanged'] and row['optimizer_unchanged']
        assert row['trials']['B8_warm']['owner_seconds'] < min(
            row['trials'][a]['owner_seconds'] for a in ('A1_natural', 'A1_after'))
    for name in ('native-resource-owner-prepared.json', 'native-resource-owner-deployed.json'):
        # The accepted integer-copy owner is already deployed; only the launch
        # allocator configuration below is new. Preserve its original receipt.
        (AUDIT / name).write_bytes((PRIOR / name).read_bytes())
    source = PRIOR / 'source/prepare_fresh_queue.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '3cf1500344a22215986cb69ad2a951be86daf77b22e9fb2ef432116d78b33355'
    code = source.read_text().replace(
        "'receipts/six-run-fresh-after-efficiency-20261017-v1'", "'receipts/six-run-fresh-after-register-threads-20261017-v1'")
    code = code.replace("'-eight-gpu-sequential-20261017-v2'", "'-eight-gpu-sequential-20261017-v3'")
    code = code.replace("removesuffix('-v1') + '-v2'", "removesuffix('-v1') + '-v3'")
    code = code.replace('expandable_segments:False,pinned_use_cuda_host_register:True',
                        'expandable_segments:False,pinned_use_cuda_host_register:True,pinned_num_register_threads:8')
    code = code.replace('native-host-register-and-MRoPE-row-copy-20261017-v1',
                        'native-host-register-threads8-and-MRoPE-row-copy-20261017-v2')
    # This invokes the same owner command builder, six Hydra resolves, task
    # workload equality checks and original full-state checkpoint paths.
    exec(compile(code, str(source), 'exec'), dict(__name__='__main__', __file__=str(source)))
    accepted = dict(status='accepted_native_launch_resource_configuration_not_yet_imported',
        native_resource_owner_changes=0, training_method_changes=0,
        configuration='pinned_use_cuda_host_register:True,pinned_num_register_threads:8',
        actual_comparison=str(PRIOR / 'register-threads-ABBA-summary-v1.json'),
        source_commit=os.environ['RL_CAMPAIGN_COMMIT'],
        same_input_raw_DT_vectors_exact_all_eight_ranks=True,
        scope='Natural B4 per rank, paired EOS contexts574..623; no whole-training or32k speed claim',
        prior_v2_training_excluded=True, all_six_start_from_base_and_zero=True)
    (AUDIT / 'thread8-acceptance-v1.json').write_text(json.dumps(accepted, indent=2)+'\n')


if __name__ == '__main__':
    main()
