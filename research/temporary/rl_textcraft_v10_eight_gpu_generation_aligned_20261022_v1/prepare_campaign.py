"""Bind the approved v10 owners and a default-inert request seed control.

No model, environment, DT, or optimizer execution. Generation stays in the
original TextCraft/VERL/vLLM owners; unchanged Hydra arguments stay byte-exact.
"""
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
GEN = ROOT / 'receipts/textcraft-generation-numerical-20261021-v1'
CANDIDATE = ROOT / 'candidates/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
PARENT = ROOT / 'candidates/textcraft-v10-target-logprob-20261021-v1/entry'
VERL_PARENT = ROOT / 'candidates/textcraft-dt-v9-negative-clip2-20261019-v3/verl-owner'
VERSION = 'textcraft-dtpo-v10-target-logprob-8gpu-formal-20261022-v1'
RUN = ROOT / 'runs' / VERSION
CHECKPOINTS = ROOT.parent.parent / 'deltatrace_backups/textcraft-8gpu-20261022' / VERSION / 'checkpoints'
MODE = 'task_sample_turn_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def replace_file(path, content):
    # Break a source hardlink/symlink before changing the candidate owner.
    temporary = path.with_name(path.name + '.prepared-tmp')
    temporary.write_text(content)
    temporary.replace(path)


def main():
    assert not CANDIDATE.exists(), 'Inspect any existing candidate rather than replace it'
    assert not RUN.exists() and not CHECKPOINTS.exists(), 'Fresh base-model run only'
    assert sha(PARENT / 'reward_readout.py') == '812ef4662771055d1045d78ed76cc8a1d2d8bd3ae607d1b6e9ec809a49ee51cd'
    assert sha(PARENT / 'dt_training_batch.py') == '006419cd4c186a44f4bbbc895994eff032e01136f5e70d9036915750266dcfc4'
    assert sha(PARENT / 'textcraft_owner_rollout.py') == 'f90536adc58b4a84cd46cc700dfed5349ce472c1955fdfc6724a7d985e983e43'
    exact = json.loads((GEN / 'full-first-DP2-matched-vs-DP8-coupled.json').read_text())
    assert exact['counts']['raw_generated_token_sequence_equal'] == 256
    assert exact['counts']['native_environment_returns_equal'] == 256
    assert exact['same_raw_output_request_pairs'] == exact['same_actual_input_request_pairs'] == 4722
    assert exact['identical_prefix_chosen_logp_abs_difference_max'] == 0

    source = AUDIT / 'source'
    source.mkdir(parents=True, exist_ok=True)
    entry, verl = CANDIDATE / 'entry', CANDIDATE / 'verl-owner'
    entry.mkdir(parents=True)
    for child in PARENT.iterdir():
        if child.name != '__pycache__':
            (entry / child.name).symlink_to(child.resolve())
    shutil.copytree(VERL_PARENT, verl, symlinks=True, copy_function=os.link,
                    ignore=shutil.ignore_patterns('.git', '__pycache__'))

    # Reuse the exact pure identity-to-seed function exercised in the full rollout.
    probe = GEN / 'run_full_first_rollout.py'
    assert sha(probe) == 'b9f6020f3827f54c8fe59fbbcf661dd09f03769d877d8885ba475967179ddff6'
    probe_text = probe.read_text()
    seed_node = next(node for node in ast.parse(probe_text).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'request_seed')
    seed_code = ast.get_source_segment(probe_text, seed_node)
    owner_path = entry / 'textcraft_owner_rollout.py'
    before = owner_path.read_text()
    after = before.replace('import json\n', 'import json\nimport hashlib\n', 1)
    assert after.count('\nclass TextCraftOwner:') == 1
    after = after.replace('\nclass TextCraftOwner:', '\n' + seed_code + '\n\n\nclass TextCraftOwner:', 1)
    anchor = """                batch = collector.preprocess_batch(transport_input, dict(
                    raw_prompt_ids=prompt_token_ids, sampling_kwargs=[sampling]*len(active)))"""
    replacement = """                sampling_values = [sampling]*len(active)
                if is_train and os.environ.get('TEXTCRAFT_REQUEST_SEED_MODE') == 'task_sample_turn_v1':
                    # Use the native trainer clock, including its original checkpoint resume.
                    native_step = int(gen_batch.meta_info['global_steps'])
                    seed_base = int(owner.config.data.get('seed', 1)) + native_step - 1
                    sampling_values = [dict(sampling, seed=request_seed(
                        seed_base, h.item_id, h.transport_index % n, len(records[h.transport_index])))
                        for h in active]
                batch = collector.preprocess_batch(transport_input, dict(
                    raw_prompt_ids=prompt_token_ids, sampling_kwargs=sampling_values))"""
    assert after.count(anchor) == 1
    after = after.replace(anchor, replacement)
    compile(after, str(owner_path), 'exec')
    replace_file(owner_path, after)
    owner_diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                            fromfile=str(PARENT / owner_path.name), tofile=str(owner_path)))
    (source / 'request-seed-owner.diff').write_text(owner_diff)

    trainer_path = verl / 'verl/trainer/ppo/ray_trainer.py'
    trainer_parent = VERL_PARENT / 'verl/trainer/ppo/ray_trainer.py'
    trainer_before = trainer_path.read_text()
    assert sha(trainer_parent) == 'df13e6903df130a759c9f97817387feff833e84167baa717c6ac34b86949e970'
    anchor = '                        gen_batch_output = self.traj_collector.multi_turn_loop(\n'
    insertion = """                        if os.environ.get('TEXTCRAFT_REQUEST_SEED_MODE') == 'task_sample_turn_v1':
                            gen_batch.meta_info['global_steps'] = self.global_steps
"""
    assert trainer_before.count(anchor) == 1
    trainer_after = trainer_before.replace(anchor, insertion + anchor)
    compile(trainer_after, str(trainer_path), 'exec')
    replace_file(trainer_path, trainer_after)
    assert sha(trainer_parent) == 'df13e6903df130a759c9f97817387feff833e84167baa717c6ac34b86949e970'
    (source / 'native-trainer-clock.diff').write_text(''.join(difflib.unified_diff(
        trainer_before.splitlines(True), trainer_after.splitlines(True),
        fromfile=str(trainer_parent), tofile=str(trainer_path))))

    # The original AgentGym logger uses the supplied native global step. Give its
    # existing output paths this run's identity, without changing task settings.
    config_path = entry / 'owner_environment_configs.json'
    environment_config = json.loads(config_path.read_text())
    environment_config['TextCraft']['train']['actor_rollout_ref']['rollout']['rollout_log_dir'] = str(RUN / 'native-executer-logs/train')
    environment_config['TextCraft']['eval']['rollout']['rollout_log_dir'] = str(RUN / 'native-executer-logs/eval')
    replace_file(config_path, json.dumps(environment_config, indent=2) + '\n')

    parent_audit = ROOT / 'receipts/textcraft-v10-target-logprob-20261021-v1'
    row = copy.deepcopy(json.loads((parent_audit / 'dt-prepared.json').read_text()))
    old_run, old_checkpoints, old_verl = row['run'], row['checkpoints'], row['verl_root']
    row.update(version=VERSION, status='prepared_only_native_v10_DP8_with_request_identity_control',
        run=str(RUN), output=str(RUN / 'textcraft-dt'), checkpoints=str(CHECKPOINTS),
        entry=str(entry), verl_root=str(verl), devices='0,1,2,3,4,5,6,7', physical_gpus=list(range(8)),
        fresh_base_model=True, resume_checkpoint=None, training_algorithm_code_changes=False,
        method_source_commit='53b421c7a6d4bdb045806a49cd22f5e00dce7769',
        execution_source_commit='4545c55807e7d002287717b6a4254f9b083dd5fc',
        source_reuse=dict(v10_entry=str(PARENT), eight_GPU_VERL_owner=str(VERL_PARENT),
            DT_root=row['dt_root'], method='Original v10 readout812/dt_training_batch006 unchanged',
            generation='Native eight-way dispatch, original sampler and original vLLM; only explicit request seeds',
            trainer='Existing eight-GPU resource/global-norm owner plus default-inert native global-step metadata'),
        actor_gradient_norm='DTensor.full_tensor global pre-clip measured; separate raw local shard retained',
        generation_control=dict(mode=MODE, seed_base='original data.seed + native global_steps - 1',
            identity='task item_id, sample within original group, native assistant turn',
            submission='Original native eight-replica dispatch; unchanged max_num_seqs32',
            temperature_and_top_p_unchanged=True, seed_selection_by_score=False,
            claim='Comparison protocol control, not a kernel accuracy or quality improvement claim'))
    identity_changes = {
        'trainer.n_gpus_per_node': 8,
        'trainer.experiment_name': VERSION,
        'trainer.default_local_dir': str(CHECKPOINTS),
        'trainer.rollout_data_dir': str(RUN / 'textcraft-dt/rollouts'),
        'trainer.validation_data_dir': str(RUN / 'textcraft-dt/validation'),
        '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS': str(RUN / 'worker-runtime-overrides'),
    }
    for key, value in identity_changes.items():
        row['options'][key] = value
    flag = '+ray_init.runtime_env.env_vars.TEXTCRAFT_REQUEST_SEED_MODE'
    row['options'][flag] = MODE
    row['argv'] = [arg.split('=', 1)[0] + '=' + json.dumps(identity_changes[arg.split('=',1)[0]])
                   if '=' in arg and arg.split('=',1)[0] in identity_changes else arg for arg in row['argv']]
    row['argv'].append(flag + '=' + json.dumps(MODE))
    row['effective_environment']['CUDA_VISIBLE_DEVICES'] = row['devices']
    row['effective_environment']['VERL_ROOT'] = str(verl)
    row['effective_environment']['TEXTCRAFT_REQUEST_SEED_MODE'] = MODE
    row['effective_environment']['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(RUN / 'worker-runtime-overrides')
    paths = row['effective_environment']['PYTHONPATH'].split(':')
    row['effective_environment']['PYTHONPATH'] = ':'.join([str(entry), str(verl)] + [
        p for p in paths if p not in (str(PARENT), old_verl)])
    row['effective_environment']['PWD'] = str(verl)
    row['effective_environment'].pop('DT_NATIVE_HOST_MEMORY_LEASE', None)
    row['source_bindings'] = {
        (p.replace(old_verl, str(verl), 1) if p.startswith(old_verl + '/') else
         p.replace(str(PARENT), str(entry), 1) if p.startswith(str(PARENT) + '/') else p):
        sha(p.replace(old_verl, str(verl), 1) if p.startswith(old_verl + '/') else
            p.replace(str(PARENT), str(entry), 1) if p.startswith(str(PARENT) + '/') else p)
        for p in row['source_bindings']}
    row['verification_receipts'] = dict(
        original_v10_CPU=str(parent_audit / 'CPU-behavior-verification.json'),
        original_v10_native=str(parent_audit / 'native-behavior-status.json'),
        generation_full_native=str(GEN / 'full-first-DP2-matched-vs-DP8-coupled.json'),
        source_preparation=str(AUDIT / 'generation-control-preparation.json'))
    row['orchestration_source_bindings'] = {str(p): sha(p) for p in source.glob('*.py')}
    # The original recipe includes null total_training_steps and an untouched
    # Hydra dictionary literal. Do not reserialize those original CLI arguments.
    assert row['options']['trainer.total_epochs'] == 30 and row['options']['trainer.total_training_steps'] is None
    assert row['budget']['total_iterations'] == 330
    assert row['options']['actor_rollout_ref.actor.ppo_mini_batch_size'] == 64
    put(AUDIT / 'dt-prepared.json', row)
    put(AUDIT / 'campaign-prepared.json', dict(status='v10_eight_GPU_formal_configuration_resolved', order=['dt'], jobs=[row]))
    record = dict(status='prepared_only_not_deployed_or_training', prepared_unix=time.time(),
        candidate=str(CANDIDATE), run=str(RUN), native_seed_function_reuse=str(probe),
        native_seed_function_source_SHA256=sha(probe),
        changed_sources={str(owner_path): sha(owner_path), str(trainer_path): sha(trainer_path), str(config_path): sha(config_path)},
        unchanged_v10_credit_sources={str(entry / name): sha(entry / name) for name in ('reward_readout.py','dt_training_batch.py')},
        source_parent_unchanged=True, default_mode_unchanged=True,
        model_calls=0, DT_calls=0, optimizer_updates=0, environment_calls=0,
        original_config=str(PARENT / config_path.name), original_config_SHA256=sha(PARENT / config_path.name),
        config_changes='Only native AgentGym log destination identities',
        original_native_generation_evidence=str(GEN / 'full-first-DP2-matched-vs-DP8-coupled.json'))
    put(AUDIT / 'generation-control-preparation.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
