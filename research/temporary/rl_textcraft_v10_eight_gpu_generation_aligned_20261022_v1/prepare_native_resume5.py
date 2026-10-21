"""Prepare and resolve one explicit native continuation; no training is launched."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from run_native_resume5 import ROOT, AUDIT, RUN, native_checkpoint_files, controller_source, sha

CANDIDATE = ROOT / 'candidates/textcraft-v10-eight-gpu-native-resume5-resource-20261022-v1'
PARENT = ROOT / 'candidates/textcraft-v10-eight-gpu-generation-aligned-20261022-v2'
PROFILE = 'root_pageable_replay_pinned_resident16384_v1'


def put(name, value):
    (AUDIT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import psutil
    import torch
    previous = json.loads((RUN / 'source.json').read_text())
    assert previous['run'] == str(RUN) and previous['pid'] == 2654698
    assert (RUN / 'exit-code.txt').read_text().strip() == '1'
    for pid, birth in ((2654663, 1792605592.06), (2654698, 1792605592.43)):
        try:
            process = psutil.Process(pid)
            assert not (process.create_time() == birth and process.status() != psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            pass
    failure = json.loads((AUDIT / 'campaign-state.json').read_text())
    assert failure['status'] == 'needs_attention_no_failed_combination_skipped'
    checkpoint = Path(previous['checkpoints']) / 'global_step_5'
    files = native_checkpoint_files(checkpoint)
    assert len(files) == 25 and all(path.is_file() and path.stat().st_size for path in files)
    metadata = json.loads((AUDIT / 'native-checkpoint5-CPU-metadata-and-restore-owner.json').read_text())
    assert metadata['native_marker'].strip() == '5'
    assert all(row['optimizer_step_values'] == [20.0] for row in metadata['states'])
    assert sha(checkpoint / 'data.pt') == metadata['data_state_SHA256']
    epochs = []
    for row in metadata['states']:
        path = Path(row['extra_path'])
        assert sha(path) == row['extra_SHA256']
        extra = torch.load(path, map_location='cpu', weights_only=False)
        epochs.append(extra['lr_scheduler']['last_epoch'])
        assert set(extra['rng']) >= {'cpu', 'numpy', 'random', 'cuda'}
        assert extra['rollout_gen_random_states'].numel()
    assert set(epochs) == {5}
    assert metadata['data_state_scalars']['_num_yielded'] == 5
    assert not torch.cuda.is_initialized()
    put('resume5-native-checkpoint-state-verification.json', dict(
        status='verified_native_checkpoint5_prelaunch_metadata', verified_unix=time.time(),
        checkpoint=str(checkpoint), required_files={str(p.relative_to(checkpoint)):p.stat().st_size for p in files},
        optimizer_step_values=[20.0], scheduler_epochs=sorted(set(epochs)),
        native_data_batches_yielded=5, data_SHA256=metadata['data_state_SHA256'],
        RNG_and_rollout_RNG_present=True, actual_worker_restore_executed=False,
        model_calls=0, DT_calls=0, optimizer_calls=0))
    put('resume5-failure-precondition.json', dict(native_exit_code=1, completed_iterations=5,
        previous_source_SHA256=sha(RUN / 'source.json'), previous_launch_SHA256=sha(RUN / 'launch.json'),
        previous_campaign_state_SHA256=sha(AUDIT / 'campaign-state.json'),
        original_driver_pid=2654698, original_driver_birth=1792605592.43,
        native_checkpoint=str(checkpoint), prepared_unix=time.time()))

    prepared = json.loads((CANDIDATE / 'resource-source-prepared.json').read_text())
    row = copy.deepcopy(previous)
    for key in ('pid', 'birth', 'started_unix', 'observed_process_created_unix', 'source_receipt',
                'log', 'stdout', 'stderr', 'exit_file', 'active', 'alive', 'completed_iterations',
                'phase', 'progress_collected', 'ray_session'):
        row.pop(key, None)
    previous.update(status='failed_exit1_DT_host_pinned_allocation', active=False, alive=False,
                    terminal_native_exit_code=1, retained_complete_native_iterations=5)
    row.update(status='prepared_native_same_run_checkpoint5_continuation',
               entry=str(CANDIDATE / 'entry'), verl_root=str(CANDIDATE / 'verl-owner'),
               fresh_base_model=False, resume_checkpoint=str(checkpoint),
               resumed_completed_iteration=5, previous_job_identity=previous,
               original_fresh_base_model_start_preserved=True, same_logical_run=True,
               worker_initialization_code_changed=True,
               resource_initialization_patch_commit='98945557aa3008d9baaf14e1927fbbd21c0ddc48')
    row['effective_environment']['PYTHONPATH'] = row['effective_environment']['PYTHONPATH'].replace(str(PARENT), str(CANDIDATE))
    receipt_root = RUN / 'runtime-overrides/accepted-storage-resume5-v1'
    row['effective_environment'].update(DT_ACCEPTED_STORAGE_PROFILE=PROFILE,
                                       DT_ACCEPTED_STORAGE_RECEIPTS=str(receipt_root))
    updates = {'trainer.resume_mode': 'resume_path', 'trainer.resume_from_path': str(checkpoint)}
    additions = {'+ray_init.runtime_env.env_vars.DT_ACCEPTED_STORAGE_PROFILE': PROFILE,
                 '+ray_init.runtime_env.env_vars.DT_ACCEPTED_STORAGE_RECEIPTS': str(receipt_root)}
    seen = set()
    argv = []
    for argument in row['argv']:
        key = argument.split('=', 1)[0]
        if key in updates:
            argv.append(key + '=' + json.dumps(updates[key])); seen.add(key)
        else:
            argv.append(argument)
    assert seen == set(updates)
    assert not any(argument.split('=', 1)[0] in additions for argument in argv)
    row['argv'] = argv + [key + '=' + json.dumps(value) for key, value in additions.items()]
    row['options'].update(updates, **additions)
    bindings = {}
    for path, digest in previous['source_bindings'].items():
        mapped = str(path).replace(str(PARENT), str(CANDIDATE))
        expected = prepared['source_bindings'].get(mapped, digest)
        assert sha(mapped) == expected, mapped
        bindings[mapped] = expected
    bindings.update(prepared['source_bindings'])
    row['source_bindings'] = bindings
    row['runtime_resource_reuse'] = dict(mode=PROFILE, initialization='Before first DT on each new producer',
        accepted_owner_callbacks=str(AUDIT / 'resource-profile-callback-composition-verification.json'),
        actual_completed_actor_overrides_pending=True)
    row['changed_argv_keys'] = sorted(list(updates) + list(additions))
    put('resume5-dt-prepared.json', row)
    (AUDIT / 'source/composed-native-resume5-resource-owner.py').write_text(controller_source())

    # Reuse the original CLI/source verifier, with distinct prepared receipts and
    # the explicit same-run guard. Do not overwrite failed campaign metadata here.
    owner = AUDIT / 'source/verify_prepared_campaign.py'
    assert sha(owner) == 'b819767a4bea17a8a7763ac91b7db14363e2fa072075198e8cd4f488ecd0a122'
    code = owner.read_text()
    guard = "assert not Path(row['run']).exists(), 'No duplicate training launch'"
    assert code.count(guard) == 1
    code = code.replace(guard, "assert row['run'] == " + repr(str(RUN)) + " and Path(row['run']).is_dir()")
    for before, after in [('formal-CLI', 'resume5-CLI'), ('dt-prepared.json', 'resume5-dt-prepared.json'),
                          ('campaign-prepared.json', 'resume5-campaign-prepared.json'),
                          ('generation-controls-ready.json', 'resume5-generation-controls-ready.json')]:
        code = code.replace(before, after)
    exec(compile(code, str(owner), 'exec'), dict(__name__='__main__', __file__=str(owner)))
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(AUDIT / 'resume5-CLI.stdout.yaml')
    expected = {'trainer.resume_mode': 'resume_path', 'trainer.resume_from_path': str(checkpoint),
                'trainer.total_epochs': 30, 'trainer.total_training_steps': None,
                'trainer.n_gpus_per_node': 8, 'trainer.save_freq': 5, 'trainer.max_actor_ckpt_to_keep': 2,
                'data.train_batch_size': 32, 'env.rollout.n': 8,
                'actor_rollout_ref.actor.ppo_mini_batch_size': 64,
                'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 4,
                'actor_rollout_ref.actor.ppo_epochs': 1,
                'ray_init.runtime_env.env_vars.DT_ACCEPTED_STORAGE_PROFILE': PROFILE}
    assert all(OmegaConf.select(cfg, key) == value for key, value in expected.items())
    put('resume5-preparation-completed.json', dict(status='prepared_verified_same_run_native_resume5',
        prepared_unix=time.time(), expected_native_configuration=expected, candidate=str(CANDIDATE),
        checkpoint=str(checkpoint), original_epoch_budget=30, original_iteration_budget=330,
        model_calls=0, DT_calls=0, optimizer_calls=0, training_launches=0,
        controller_source_SHA256=sha(AUDIT / 'source/composed-native-resume5-resource-owner.py')))
    print(json.dumps(dict(status='prepared_verified_same_run_native_resume5',
                         checkpoint=str(checkpoint), budget=330, training_launches=0)))


if __name__ == '__main__':
    main()
