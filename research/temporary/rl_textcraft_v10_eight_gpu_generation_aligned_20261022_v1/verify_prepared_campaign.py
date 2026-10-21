"""Resolve the exact formal argv with native Hydra, without starting workers."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_campaign import ROOT, AUDIT, GEN, VERSION, MODE, sha, put


def main():
    row = json.loads((AUDIT / 'dt-prepared.json').read_text())
    assert not Path(row['run']).exists(), 'No duplicate training launch'
    environment = dict(os.environ)
    environment.update(row['effective_environment'])
    environment['CUDA_VISIBLE_DEVICES'] = ''
    result = subprocess.run(row['argv'] + ['--cfg', 'job'], cwd=row['verl_root'],
        env=environment, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=240)
    (AUDIT / 'formal-CLI.stdout.yaml').write_text(result.stdout)
    (AUDIT / 'formal-CLI.stderr.log').write_text(result.stderr)
    assert result.returncode == 0, 'Inspect native Hydra stderr; no training was started'
    assert MODE in result.stdout and VERSION in result.stdout
    assert 'n_gpus_per_node: 8' in result.stdout
    verification = dict(status='passed_original_Hydra_parser_and_native_VERL_cfg_only',
        verified_unix=time.time(), argv=row['argv'] + ['--cfg', 'job'],
        resolved_stdout=str(AUDIT / 'formal-CLI.stdout.yaml'),
        resolved_stdout_SHA256=sha(AUDIT / 'formal-CLI.stdout.yaml'),
        source_bindings=row['source_bindings'], model_calls=0, DT_calls=0,
        environment_calls=0, optimizer_updates=0)
    put(AUDIT / 'formal-CLI-verification.json', verification)

    # The only new RNG operation is the same identity conversion already used in
    # the completed native control. Sampling itself remains vLLM SamplingParams.
    entry = Path(row['entry'])
    owner_text = (entry / 'textcraft_owner_rollout.py').read_text()
    probe_text = (GEN / 'run_full_first_rollout.py').read_text()
    get_seed = lambda text: next(node for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == 'request_seed')
    assert ast.dump(get_seed(owner_text), include_attributes=False) == ast.dump(get_seed(probe_text), include_attributes=False)
    assert sha(entry / 'reward_readout.py') == '812ef4662771055d1045d78ed76cc8a1d2d8bd3ae607d1b6e9ec809a49ee51cd'
    assert sha(entry / 'dt_training_batch.py') == '006419cd4c186a44f4bbbc895994eff032e01136f5e70d9036915750266dcfc4'
    for path, digest in row['source_bindings'].items():
        assert sha(path) == digest, path
    prep = json.loads((AUDIT / 'generation-control-preparation.json').read_text())
    review = json.loads((AUDIT / 'full-native-generation-review.json').read_text())
    assert review['status'] == 'completed_full_native_zero_update_generation_comparison'
    ready = dict(status='ready_for_formal_training_after_full_native_generation_review',
        reviewed_unix=time.time(), deployed_control_sources=prep['changed_sources'],
        credit_method_unchanged=prep['unchanged_v10_credit_sources'],
        native_generation_evidence=str(GEN / 'full-first-DP2-matched-vs-DP8-coupled.json'),
        native_generation_evidence_SHA256=sha(GEN / 'full-first-DP2-matched-vs-DP8-coupled.json'),
        formal_configuration_verification=str(AUDIT / 'formal-CLI-verification.json'),
        scope='Full native first-batch protocol control plus source/CLI wiring; first natural formal iteration and worker imports remain pending',
        kernel_fix_claim=False, quality_improvement_claim=False, whole_model_tolerance_verdict=None,
        model_calls=0, DT_calls=0, environment_calls=0, optimizer_updates=0)
    put(AUDIT / 'generation-controls-ready.json', ready)
    row['effective_config_SHA256'] = verification['resolved_stdout_SHA256']
    row['interface_verification'] = str(AUDIT / 'formal-CLI-verification.json')
    row['interface_verification_SHA256'] = sha(AUDIT / 'formal-CLI-verification.json')
    row['orchestration_source_bindings'] = {str(p): sha(p) for p in (AUDIT / 'source').glob('*.py')}
    put(AUDIT / 'dt-prepared.json', row)
    put(AUDIT / 'campaign-prepared.json', dict(status='v10_eight_GPU_formal_configuration_resolved', order=['dt'], jobs=[row]))
    print(json.dumps(dict(verification=verification['status'], ready=ready, formal_training_started=False)))


if __name__ == '__main__':
    main()
