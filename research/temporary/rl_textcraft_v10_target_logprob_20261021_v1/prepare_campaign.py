"""Patch the pinned v5 readout owner and reuse its native DP2 formal configuration."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/textcraft-v10-target-logprob-20261021-v1'
VERSION = 'textcraft-dtpo-v10-target-logprob-2gpu-formal-20261021-v1'
MODE = 'reward_target_logprob_exponential_v10'
CANDIDATE = ROOT/'candidates/textcraft-v10-target-logprob-20261021-v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    from omegaconf import OmegaConf
    old_audit = ROOT/'receipts/textcraft-v5-two-gpu-reproduction-20261021-v1'
    before = json.loads((old_audit/'dt-prepared.json').read_bytes())
    entry = CANDIDATE/'entry'
    entry.mkdir(parents=True, exist_ok=False)
    old_entry = Path(before['entry'])
    original = old_entry/'reward_readout.py'
    assert sha(original) == '1797faaaa89537f5369f82b18b613100f92344f2160f3ca68c5bddfb53009017'
    canonical = (AUDIT/'source/canonical_reward_readout.py').read_text()
    helper_start = canonical.index('    @staticmethod\n    def _restore_target_endpoint_ratios')
    helper = canonical[helper_start:]
    text = original.read_text()
    prefix, body = text.split('class DirectActionTargetReadout:', 1)
    body = replace_once(body,
        "        self.reward_target_exponential = os.environ.get('DT_CREDIT_MODE') == 'reward_target_exponential_v5'",
        "        self.target_token_logprob = os.environ.get('DT_CREDIT_MODE') == '"+MODE+"'\n"
        "        self.reward_target_exponential = os.environ.get('DT_CREDIT_MODE') in ('reward_target_exponential_v5', '"+MODE+"')")
    body = replace_once(body, "            report['finite_trace_calls'] += 1",
        "            if self.target_token_logprob:\n"
        "                self._restore_target_endpoint_ratios(batch, detail)\n"
        "            report['finite_trace_calls'] += 1")
    body = replace_once(body, "self_target_mask=item['target'][None, None, :],",
        "self_target_mask=None if self.target_token_logprob else item['target'][None, None, :],")
    body = replace_once(body, "        report['raw_advantage_groups'] = groups",
        "        report['raw_advantage_groups'] = groups\n"
        "        if self.target_token_logprob:\n"
        "            report.update(target_credit_rule='per_token_native_endpoint_logprob_difference_then_exponential',\n"
        "                          target_self_constant_override=False,\n"
        "                          target_reference='original_v5_prior_source_EOS; fixed_labels_and_factual_target_prefix',\n"
        "                          additional_model_forwards=0)")
    patched = prefix+'class DirectActionTargetReadout:'+body.rstrip()+'\n\n'+helper
    ast.parse(patched)
    (entry/'reward_readout.py').write_text(patched)
    for filename in ('textcraft_owner_rollout.py', 'dt_training_batch.py'):
        text = (old_entry/filename).read_text()
        if filename == 'textcraft_owner_rollout.py':
            text = replace_once(text, "'weighted_reward_target_finite_v4', 'reward_target_exponential_v5')",
                "'weighted_reward_target_finite_v4', 'reward_target_exponential_v5', '"+MODE+"')")
        else:
            text = replace_once(text, "os.environ.get('DT_CREDIT_MODE') == 'reward_target_exponential_v5' and any(",
                "os.environ.get('DT_CREDIT_MODE') in ('reward_target_exponential_v5', '"+MODE+"') and any(")
        ast.parse(text); (entry/filename).write_text(text)

    run = ROOT/'runs'/VERSION
    checkpoint = ROOT.parent.parent/'deltatrace_backups/textcraft-2gpu-20261021'/VERSION/'checkpoints'
    assert not run.exists() and not checkpoint.exists()
    row = copy.deepcopy(before)
    old_run = before['run']
    env = {k:v.replace(old_run,str(run)) for k,v in before['effective_environment'].items()}
    env.update(DT_CREDIT_MODE=MODE, CUDA_VISIBLE_DEVICES='2,3',
               PYTHONPATH=str(entry)+':'+env['PYTHONPATH'])
    options = {k:(v.replace(old_run,str(run)) if isinstance(v,str) else v)
               for k,v in before['options'].items()}
    options.update({'trainer.experiment_name':VERSION, 'trainer.default_local_dir':str(checkpoint),
                    '+ray_init.runtime_env.env_vars.DT_CREDIT_MODE':MODE})
    for key in list(options):
        if key.startswith('+ray_init.runtime_env.env_vars.'):
            variable = key.rsplit('.',1)[1]
            if variable in env:
                options[key] = env[variable]
    argv = [arg.replace(old_run,str(run)) for arg in before['argv']]
    for key,value in options.items():
        label = key+'='
        found = sum(a.startswith(label) for a in argv)
        if found:
            assert found == 1
            argv = [label+json.dumps(value) if a.startswith(label) else a for a in argv]
        else:
            assert key == '+ray_init.runtime_env.env_vars.DT_CREDIT_MODE'
            argv.append(label+json.dumps(value))
    bindings = dict(before['source_bindings'])
    for filename in ('reward_readout.py','textcraft_owner_rollout.py','dt_training_batch.py'):
        bindings.pop(str(old_entry/filename), None)
        bindings[str(entry/filename)] = sha(entry/filename)
    for path,digest in bindings.items():
        assert sha(path) == digest, path
    row.update(version=VERSION, run=str(run), output=str(run/'textcraft-dt'),
        checkpoints=str(checkpoint), options=options, argv=argv, effective_environment=env,
        physical_gpus=[2,3], devices=[2,3], entry=str(entry), source_bindings=bindings,
        status='prepared_only_v10_native_target_endpoint_logprob_credit',
        original_launch=str(old_audit/'dt-prepared.json'), original_launch_SHA256=sha(old_audit/'dt-prepared.json'),
        parent_method_source_commit=before['method_source_commit'], method_source_commit=None,
        execution_source_commit=None, local_patch_commit=None, training_algorithm_code_changes=3,
        method_specification='experiments/rl/PLAN.md', fresh_base_model=True, resume_checkpoint=None,
        source_reuse='Pinned original v5 VERL/DT/environment; only three entry owner files patched',
        actor_gradient_norm='first_returned_rank_local_preclip_original_v5_logging')
    config = OmegaConf.load(old_audit/'dt-effective-config.yaml')
    for key,value in options.items():
        OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    OmegaConf.save(config,AUDIT/'dt-effective-config.yaml')
    row['effective_config_SHA256'] = sha(AUDIT/'dt-effective-config.yaml')
    (AUDIT/'dt-prepared.json').write_text(json.dumps(row,indent=2)+'\n')
    (AUDIT/'campaign-prepared.json').write_text(json.dumps(dict(
        status='v10_target_endpoint_logprob_DP2_formal_configuration_resolved',order=['dt'],jobs=[row]),indent=2)+'\n')
    for name in ('formal-training.json','active-training.json','active-source.json'):
        (AUDIT/name).write_text(json.dumps(dict(jobs=[],source_commit='v10-source-SHA-bound-prepared'))+'\n')
    helper = ROOT/'receipts/textcraft-v5-credit-integration-20261020-v1/source/resume_at_native_checkpoint.py'
    assert sha(helper) == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
    shutil.copyfile(helper,AUDIT/'source/resume_at_native_checkpoint.py')
    changed = {k:dict(before=before['options'].get(k),after=v) for k,v in options.items()
               if v != before['options'].get(k)}
    (AUDIT/'configuration-differences.json').write_text(json.dumps(changed,indent=2)+'\n')
    print(json.dumps(dict(status='prepared_only',version=VERSION,physical_gpus=[2,3],budget=row['budget'],
        changed_options=list(changed), patched_owner_files={p.name:sha(p) for p in entry.glob('*.py')})))


if __name__ == '__main__':
    main()
