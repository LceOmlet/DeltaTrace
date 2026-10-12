"""Bounded integration checks through the recorded VERL/vLLM owners.

No rollout, loss, checkpoint, sampler, or DT propagation is implemented here.
The 32k storage case reuses the previously accepted capacity helper; it is
explicitly separate from task quality and the full formal training budget.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT = Path(os.environ['DT_RUNTIME_ROOT'])
OUT = ROOT / 'receipts/textcraft-native-graph-deployment-20261012-v1'
PREVIOUS = ROOT / 'runs/textcraft-dt-formal-eventwait-fresh-20261012-v2'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_value(value):
    digest = hashlib.sha256()
    def visit(x):
        if hasattr(x, 'to_local'):
            x = x.to_local()
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            digest.update(str((x.dtype, tuple(x.shape))).encode())
            digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=repr):
                visit(key); visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        else:
            digest.update(pickle.dumps(x))
    visit(value)
    return digest.hexdigest()


@ray.remote
class ObservedWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_dt_token_advantages(self, data):
        # Passive allocation ledger around the original native replay only.
        handles=[]
        ledger=[]
        sys.path.insert(0,str(Path(os.environ['DT_ROOT'])/'clean/qwen35'))
        from qwen35_decoder_finite import NativeDecoderCapture
        retain_owner=NativeDecoderCapture.retain
        transfer_verified=[False]
        def verify_gate_transfer(capture,name,value):
            retain_owner(capture,name,value)
            saved=capture.values.get(name)
            if name=='gate_output' and saved is not None and saved.device.type=='cpu' and not transfer_verified[0]:
                # The capture owner explicitly defers pinned D2H completion
                # until scope exit. This early diagnostic consumes it inside
                # the hook, so wait before reading the host operand.
                torch.cuda.current_stream().synchronize()
                for start in range(0,value.shape[1],2048):
                    restored=saved[:,start:start+2048].to(value.device,non_blocking=True)
                    assert torch.equal(value[:,start:start+2048],restored), 'Gate capture transfer changed values'
                transfer_verified[0]=True
                (OUT/f'gate-transfer-rank{self.rank}.json').write_text(json.dumps(dict(
                    unix=time.time(),shape=list(value.shape),dtype=str(value.dtype),
                    original_stride=list(value.stride()),captured_stride=list(saved.stride()),
                    original_device=str(value.device),capture_device=str(saved.device),
                    all_original_elements_exact=True,arithmetic_changed=False),indent=2)+'\n')
        NativeDecoderCapture.retain=verify_gate_transfer
        for name,module in self.actor_module_fsdp.named_modules():
            if name.endswith('.mlp.up_proj'):
                def observe(_module, args, name=name):
                    item=dict(unix=time.time(),name=name,input_shape=list(args[0].shape),
                              input_dtype=str(args[0].dtype),allocated=torch.cuda.memory_allocated(),
                              physical_free=torch.cuda.mem_get_info()[0])
                    ledger.append(item)
                    (OUT/f'DT-memory-rank{self.rank}.json').write_text(json.dumps(ledger,indent=2)+'\n')
                handles.append(module.register_forward_pre_hook(observe))
        try:
            return super().compute_dt_token_advantages(data)
        finally:
            for handle in handles:handle.remove()
            NativeDecoderCapture.retain=retain_owner

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def observe_state(self, remember=False, corrupt=False):
        parameters = {n: p for n, p in self.actor_module_fsdp.named_parameters() if p.requires_grad}
        result = dict(rank=self.rank, pid=os.getpid(),
                      parameters=hash_value(parameters),
                      optimizer=hash_value(self.actor_optimizer.state_dict()),
                      scheduler=hash_value(self.actor_lr_scheduler.state_dict()),
                      rng=hash_value(self.checkpoint_manager.get_rng_state()),
                      rollout_rng=hash_value(self.rollout_sharding_manager.gen_random_states),
                      optimizer_states=len(self.actor_optimizer.state),
                      scheduler_state=self.actor_lr_scheduler.state_dict(),
                      config=OmegaConf.to_container(self.config, resolve=True))
        if remember:
            self.saved_state_identity = {k: result[k] for k in ('parameters', 'optimizer', 'scheduler', 'rng', 'rollout_rng')}
        elif hasattr(self, 'saved_state_identity') and not corrupt:
            result['restoration_matches'] = {k: result[k] == v for k, v in self.saved_state_identity.items()}
            assert all(result['restoration_matches'].values()), result['restoration_matches']
        if corrupt:
            with torch.no_grad():
                for p in parameters.values():
                    (p.to_local() if hasattr(p, 'to_local') else p).add_(.01)
                for state in self.actor_optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            (value.to_local() if hasattr(value, 'to_local') else value).zero_()
            self.rollout_sharding_manager.gen_random_states=torch.zeros_like(self.rollout_sharding_manager.gen_random_states)
            self.actor_lr_scheduler.last_epoch += 7
            random.seed(981); np.random.seed(981); torch.manual_seed(981); torch.cuda.manual_seed(981)
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def verify_native_graph(self, prompts, label):
        harness = load_module('native_vllm_owner_tests', ROOT / 'receipts/upstream-alignment-20260929/verify_vllm_batching_owner.py')
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        with self.rollout_sharding_manager:
            engine = self.rollout.inference_engine
            ids = list(engine.llm_engine.list_loras())
            assert len(ids) == 1, ids
            adapter = LoRARequest(str(ids[0]), ids[0], '/simon-stub-path')
            runner = SimpleNamespace(llm=engine, get_inputs=lambda rows, **kw: [dict(prompt_token_ids=p) for p in rows])
            runner._final_steps_generate_w_logprobs = harness.owner('vllm015-conftest.py', '_final_steps_generate_w_logprobs', 'VllmRunner')
            for name in ('generate_w_logprobs', 'generate_greedy_logprobs'):
                setattr(runner, name, MethodType(harness.owner('vllm015-conftest.py', name, 'VllmRunner'), runner))
            sample = [prompts[i] for i in (0, 10, 20, 31)]
            individual = [runner.generate_greedy_logprobs([p], 64, 5, lora_request=adapter)[0] for p in sample]
            batch = runner.generate_greedy_logprobs(sample, 64, 5, lora_request=adapter)
            harness.compare(outputs_0_lst=individual, outputs_1_lst=batch,
                            name_0='native_individual', name_1='native_batch')
            times = []
            params = SamplingParams(temperature=1., top_p=1., max_tokens=128, min_tokens=128, ignore_eos=True, seed=0)
            for _ in range(3):
                start = time.perf_counter()
                outputs = engine.generate([dict(prompt_token_ids=p) for p in prompts], params,
                                          lora_request=adapter, use_tqdm=False)
                times.append(dict(seconds=time.perf_counter()-start,
                                  output_tokens=sum(len(x.token_ids) for y in outputs for x in y.outputs)))
            # Native sleep discards the prefix cache. The official end-to-end
            # sleep test starts from a cold cache; equalize that input state
            # here too, without changing the production cache configuration.
            assert engine.reset_prefix_cache()
            before = runner.generate_greedy_logprobs(sample, 64, 5, lora_request=adapter)
            engine.sleep(level=1)
            engine.wake_up(tags=['weights']); engine.wake_up(tags=['kv_cache'])
            after = runner.generate_greedy_logprobs(sample, 64, 5, lora_request=adapter)
            torch.save(dict(before=before, after=after, prompts=sample,
                            cache_scope='Both cold: explicit native reset before reference; native sleep resets before candidate'),
                       OUT/f'graph-{label}-sleep-rank{self.rank}.pt')
            harness.owner('vllm015-test-utils.py', 'check_outputs_equal')(
                outputs_0_lst=[x[:2] for x in before], outputs_1_lst=[x[:2] for x in after],
                name_0='native_before_sleep', name_1='native_after_sleep')
            vc = engine.llm_engine.vllm_config
            result = dict(rank=self.rank, label=label, timings=times,
                          batching='passed_original_check_logprobs_close',
                          sleep='passed_original_check_outputs_equal',
                          enforce_eager=vc.model_config.enforce_eager,
                          compilation_config=str(vc.compilation_config),
                          test_sources=harness.report['sources'])
            assert not result['enforce_eager']
        (OUT / f'graph-{label}-rank{self.rank}.json').write_text(json.dumps(result, indent=2)+'\n')
        return result


def main():
    report = dict(scope=__doc__, started_unix=time.time())
    def record(phase, **values):
        report.update(phase=phase, **values, unix=time.time())
        (OUT/'integration-result.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
        print(json.dumps(dict(phase=phase, unix=time.time(), **values), default=str), flush=True)
    launch = json.loads((PREVIOUS/'launch.json').read_bytes())
    cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key, value in launch['options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.rollout.enforce_eager = False
    cfg.actor_rollout_ref.rollout.free_cache_engine = False
    cfg.trainer.save_freq = 5
    cfg.trainer.default_local_dir = str(OUT/'diagnostic-checkpoints')
    cfg.ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS = str(OUT/'worker-runtime-overrides')
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    trainer = object.__new__(RayPPOTrainer)
    trainer.config, trainer.tokenizer, trainer.processor, trainer.use_critic = cfg, tokenizer, None, False
    trainer._create_dataloader(None, None, None, None)
    assert trainer.total_training_steps == 330
    (OUT/'integration-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    # Recover the exact compact native IDs and masks from the passive artifact.
    saved = sorted((PREVIOUS/'credit-records').glob('rank*/*.pt'))
    rows = torch.load(saved[0], map_location='cpu', weights_only=False)['rows']
    short = []
    for row in rows[:4]:
        prompt = row['prompt_length']; ids = row['input_ids']; positions = row['suffix_positions']
        short.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), responses=ids[prompt:],
                          policy_mask=row['policy_mask'][positions], target_mask=row['target_mask'][positions],
                          dt_direct_reward=row['reward'], traj_uid=row['traj_uid']))
    assert len(short) == 4 and len({x['input_ids'].numel()-x['responses'].numel() for x in short}) == 1
    from reward_readout import DirectActionTargetReadout
    owner = ROOT/'receipts/direct-target-memory-capacity-20261008-v1/check_capacity_lifetime.py'
    tree = ast.parse(owner.read_text())
    helper = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'capacity_row')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(owner), 'exec'), ns)
    def prepare(row, index):
        value = DirectActionTargetReadout._prepare_row(row, index)
        # The older storage helper expects this derived diagnostic length key.
        return value | {'full_context_length': value['selected'].numel()}
    expanded = [ns['capacity_row'](x, prepare, 32768) for x in short*2]
    tensors = {k: torch.stack([x[0][k] for x in expanded]) for k in
               ('input_ids','attention_mask','responses','policy_mask','target_mask')}
    tensors['position_ids'] = torch.arange(32768).unsqueeze(0).expand(8,-1).clone()
    tensors['loss_mask'] = torch.cat((torch.zeros((8,short[0]['input_ids'].numel()-short[0]['responses'].numel()), dtype=torch.bool), tensors['policy_mask']), dim=1)
    data = DataProto.from_dict(tensors=tensors, non_tensors={
        'dt_direct_reward': np.array([x[0]['dt_direct_reward'] for x in expanded]),
        'traj_uid': np.array([x[0]['traj_uid'] for x in expanded],dtype=object)})
    data.meta_info.update(temperature=1., multi_turn=True, global_token_num=[32768]*8,
                          dt_target_semantics='native_joint_action_target', eos_token_id=tokenizer.eos_token_id,
                          pad_token_id=tokenizer.pad_token_id)
    graph_inputs = torch.load(ROOT/'receipts/textcraft-restart-utilization-20261012-v1/graph-comparison/eager-outputs.pt',
                              map_location='cpu',weights_only=False)['prompts']
    # Same 32 inputs as the earlier paired native engine benchmark.
    graph_report = json.loads((ROOT/'receipts/textcraft-restart-utilization-20261012-v1/graph-comparison/eager.json').read_bytes())
    prompts = []
    for receipt in graph_report['inputs']:
        row = torch.load(receipt['path'],map_location='cpu',weights_only=False)['rows'][0]
        prompts.append(row['input_ids'][:receipt['target_start']].tolist())
    record('initializing_original_workers', capacity_sources=[x[1] for x in expanded])
    ray.init(num_cpus=16, include_dashboard=False, runtime_env=OmegaConf.to_container(cfg.ray_init.runtime_env, resolve=True))
    try:
        group = RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
                              RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor_rollout'))
        trainer.actor_rollout_wg = group
        group.init_model()
        record('native_graph_actor_coexistence')
        existing=[OUT/f'graph-initial-rank{rank}.json' for rank in range(2)]
        report['graph_initial'] = ([json.loads(p.read_bytes()) for p in existing]
                                   if all(p.exists() for p in existing)
                                   else group.verify_native_graph(prompts, 'initial'))
        record('native_B8_32768_DT_start')
        start=time.perf_counter(); values=group.compute_dt_token_advantages(data)
        report['DT_seconds']=time.perf_counter()-start
        data=data.union(values)
        for name in ('dt_token_advantages','dt_q_estimates','dt_v_estimates'):
            assert torch.isfinite(data.batch[name]).all(), name
        # The existing trainer function owns full-batch DT whitening and masks.
        data=compute_advantage(data,adv_estimator='deltatrace',gamma=cfg.algorithm.gamma,
                               lam=cfg.algorithm.lam,num_repeat=8,multi_turn=True)
        record('native_B8_32768_oldlogprob_start')
        start=time.perf_counter(); data=data.union(group.compute_log_prob(data));report['oldlogprob_seconds']=time.perf_counter()-start
        data.batch['ref_log_prob']=data.batch['old_log_probs'].clone()
        record('native_B8_32768_update_start')
        start=time.perf_counter(); metrics=group.update_actor(data).meta_info['metrics']
        report.update(update_seconds=time.perf_counter()-start, update_metrics=metrics)
        assert all(np.isfinite(np.asarray(x,dtype=float)).all() for k,x in metrics.items() if 'grad_norm' in k)
        # Config remains global minibatch64; B8 is capacity only, not a formal iteration.
        record('native_checkpoint_save_start')
        iterator=iter(trainer.train_dataloader);next(iterator)
        trainer.global_steps=1
        report['saved_worker_state']=group.observe_state(remember=True)
        trainer._save_checkpoint()
        expected_next=hash_value(next(iterator))
        group.observe_state(corrupt=True)
        record('native_checkpoint_restore_start')
        # Official resume constructs a fresh loader. A loader with a live,
        # prefetched iterator is not the original trainer's restart path.
        resumed = object.__new__(RayPPOTrainer)
        resumed.config, resumed.tokenizer, resumed.processor, resumed.use_critic = cfg, tokenizer, None, False
        resumed._create_dataloader(None, None, None, None)
        resumed.actor_rollout_wg = group
        resumed.global_steps = 0
        resumed._load_checkpoint()
        report['restored_worker_state']=group.observe_state()
        assert resumed.global_steps==1
        assert hash_value(next(iter(resumed.train_dataloader)))==expected_next
        report['dataloader_next_batch_exact']=True
        record('native_graph_post_update_and_restore')
        report['graph_updated']=group.verify_native_graph(prompts,'updated-restored')
        record('passed',status='passed_native_graph_B8_32768_DT_update_checkpoint_roundtrip')
    finally:
        ray.shutdown()


if __name__=='__main__':
    main()
