"""Verify the existing completed native checkpoint using a fresh trainer.

The original native resume creates a fresh StatefulDataLoader. Reusing an
active iterator in the earlier diagnostic was not the production resume path.
This check loads the already completed checkpoint; it repeats no DT or update.
"""
import json
import os
import time
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from verify_native_graph_capacity_checkpoint import ObservedWorker, hash_value

ROOT = Path(os.environ['DT_RUNTIME_ROOT'])
OUT = ROOT/'receipts/textcraft-native-graph-deployment-20261012-v1'


def main():
    report = json.loads((OUT/'integration-result.json').read_bytes())
    assert report['DT_seconds'] > 0 and report['update_seconds'] > 0
    cpu = json.loads((OUT/'dataloader-cpu-comparison.json').read_bytes())
    assert cpu['all_values_equal'] and cpu['native_checkpoint_equal']
    assert not cpu['same_instance_equal']
    original = dict(report)
    (OUT/'before-fresh-resume-integration-result.json').write_text(json.dumps(original, indent=2)+'\n')

    def record(phase, **values):
        report.update(phase=phase, **values, unix=time.time())
        (OUT/'integration-result.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
        print(json.dumps(dict(phase=phase, unix=time.time()), default=str), flush=True)

    cfg = OmegaConf.load(OUT/'integration-config.yaml')
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = str(OUT/'diagnostic-checkpoints/global_step_1')
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    trainer = object.__new__(RayPPOTrainer)
    trainer.config, trainer.tokenizer, trainer.processor, trainer.use_critic = cfg, tokenizer, None, False
    trainer._create_dataloader(None, None, None, None)
    assert trainer.total_training_steps == 330
    record('fresh_native_checkpoint_workers_initializing')
    ray.init(num_cpus=16, include_dashboard=False,
             runtime_env=OmegaConf.to_container(cfg.ray_init.runtime_env, resolve=True))
    try:
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                              RayClassWithInitArgs(ObservedWorker, cfg.actor_rollout_ref, 'actor_rollout'))
        trainer.actor_rollout_wg = group
        group.init_model()
        record('fresh_native_checkpoint_load')
        trainer._load_checkpoint()
        assert trainer.global_steps == 1
        restored = group.observe_state()
        for actual, saved in zip(restored, report['saved_worker_state']):
            assert actual['rank'] == saved['rank']
            actual['restoration_matches'] = {
                key: actual[key] == saved[key]
                for key in ('parameters', 'optimizer', 'scheduler', 'rng', 'rollout_rng')}
            assert all(actual['restoration_matches'].values()), actual['restoration_matches']
        report['restored_worker_state'] = restored
        expected = torch.load(OUT/'dataloader-unit-actual-batches.pt',
                              map_location='cpu', weights_only=False)['expected']
        actual = next(iter(trainer.train_dataloader))
        assert hash_value(actual) == hash_value(expected)
        report.update(dataloader_next_batch_exact=True,
                      checkpoint_restore_scope='Fresh original trainer, original _load_checkpoint, existing completed checkpoint; no repeated DT/update',
                      dataloader_diagnostic=str(OUT/'dataloader-cpu-comparison.json'))
        record('native_graph_post_update_and_fresh_restore')
        graph = json.loads((ROOT/'receipts/textcraft-restart-utilization-20261012-v1/graph-comparison/eager.json').read_bytes())
        prompts = []
        for item in graph['inputs']:
            row = torch.load(item['path'], map_location='cpu', weights_only=False)['rows'][0]
            prompts.append(row['input_ids'][:item['target_start']].tolist())
        report['graph_updated'] = group.verify_native_graph(prompts, 'updated-restored')
        record('passed', status='passed_native_graph_B8_32768_DT_update_checkpoint_roundtrip')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
