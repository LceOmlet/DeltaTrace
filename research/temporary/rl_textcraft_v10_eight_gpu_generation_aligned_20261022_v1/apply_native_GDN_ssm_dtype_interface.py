"""Prepare/apply the verified argument bridge in the original installed owner.

Default action only verifies source identity and the existing source diff.
Applying is permitted only after the current complete native quality pair,
its explicit evidence-based disposition and original session cleanup.
There is no model, sampler, kernel, DT or optimizer implementation here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import stat
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
MODEL_CONFIG = Path('/mnt/si0021787ci2/default/models/Qwen3.5-9B/config.json')
HELPER = AUDIT / 'native_GDN_ssm_dtype_interface_candidate.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    assert sha(HELPER) == '00191c47072ffc01c0670af6e56d7543e15affe06bd5faa098af6439eca116ca'
    bridge = runpy.run_path(str(HELPER))
    cpu = json.loads((AUDIT / 'model-declared-SSM-dtype-CPU-interface-verification.json').read_text())
    assert cpu['original_ignored_native_SSM_override_reproduced']
    assert cpu['default_auto_and_existing_global_FP32_unchanged']
    assert cpu['CUDA_initialized'] is False
    resolver = cpu['installation']['methods'][0]['original_shared_resolver']
    assert sha(resolver) == bridge['RESOLVER_SHA256']
    files = []
    for module_name, spec in bridge['SPECS'].items():
        original = Path('/opt/conda/lib/python3.12/site-packages/vllm_metax/models') / (module_name + '.py')
        candidate = AUDIT / spec['candidate']
        assert sha(original) == spec['original_SHA256']
        assert candidate.read_bytes() == bridge['build_source'](original.read_text(), module_name).encode()
        compile(candidate.read_text(), str(original), 'exec')
        assert os.access(original.parent, os.W_OK), str(original.parent)
        files.append(dict(original=str(original), original_SHA256=sha(original),
                          candidate=str(candidate), candidate_SHA256=sha(candidate),
                          changed_methods=[list(pair) for pair in spec['methods']]))
    assert sha(MODEL_CONFIG) == 'd0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05'
    declared = json.loads(MODEL_CONFIG.read_text())['text_config']['mamba_ssm_dtype']
    assert declared == 'float32'
    return dict(status='prepared_only_original_owner_SSM_argument_bridge', files=files,
                model_config=str(MODEL_CONFIG), model_config_SHA256=sha(MODEL_CONFIG),
                required_native_launch_option={'actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype': declared},
                original_shared_resolver_SHA256=bridge['RESOLVER_SHA256'],
                original_default_auto_behavior_preserved=True, native_kernel_implementations_changed=0,
                model_calls=0, DT_calls=0, optimizer_updates=0, formal_training_deployed=False,
                quality_disposition='Pending the complete original native pair; preparation is not acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    prepared = prepare()
    if not args.apply:
        target = AUDIT / 'original-owner-SSM-dtype-deployment-preparation-v1.json'
        assert not target.exists(), 'Reuse the saved preparation instead of repeating it'
        target.write_text(json.dumps(prepared, indent=2) + '\n')
        print(json.dumps(prepared))
        return

    job = json.loads((AUDIT / 'full-native-model-declared-SSM-dtype-job-v1.json').read_text())
    receipt = json.loads((Path(job['output']) / 'receipt.json').read_text())
    assert receipt['status'] == 'completed_native_full_first_rollout_comparison_cases'
    assert all(receipt['cases'][name]['status'] == 'completed_native_full_rollout' for name in ('coupled', 'matched'))
    assert receipt['DT_calls'] == receipt['optimizer_updates'] == 0
    released = json.loads((AUDIT / 'full-native-model-declared-SSM-dtype-release-v1.json').read_text())
    assert released['pid'] == job['pid'] and abs(released['birth'] - job['birth']) < 0.02
    assert released['release']['remaining_non_zombie'] == []
    quality = json.loads((AUDIT / 'native-SSM-dtype-complete-quality-review-v1.json').read_text())
    assert quality['deployment_decision'] == 'accept_original_native_SSM_dtype_interface_fix'
    assert quality['native_actual_cache_allocation_verified']
    assert quality['both_original_complete_cases_verified']
    occupancy = subprocess.run(['mx-smi'], capture_output=True, text=True, timeout=30)
    assert occupancy.returncode == 0 and 'no process found' in occupancy.stdout
    destination = AUDIT / 'original-owner-SSM-dtype-installed-v1'
    destination.mkdir(exist_ok=False)
    result = dict(prepared, status='applying_original_owner_SSM_argument_bridge', applied_unix=time.time(),
                  quality_disposition=quality, installed_files=[])
    target = destination / 'deployment.json'
    try:
        for entry in prepared['files']:
            original = Path(entry['original'])
            assert sha(original) == entry['original_SHA256']
            backup = destination / ('original-' + original.name)
            shutil.copy2(original, backup)
            assert sha(backup) == entry['original_SHA256']
            temporary = original.with_name(original.name + '.SSM-dtype-interface-tmp')
            assert not temporary.exists()
            temporary.write_bytes(Path(entry['candidate']).read_bytes())
            os.chmod(temporary, stat.S_IMODE(original.stat().st_mode))
            assert sha(temporary) == entry['candidate_SHA256']
            os.replace(temporary, original)
            assert sha(original) == entry['candidate_SHA256']
            result['installed_files'].append(dict(entry, backup=str(backup), actual_installed_SHA256=sha(original)))
        result['status'] = 'original_owner_SSM_argument_bridge_installed'
    except Exception as error:
        result.update(status='failed_preserved_exact_partial_install_state', error=repr(error))
        raise
    finally:
        target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
