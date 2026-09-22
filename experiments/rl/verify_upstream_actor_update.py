"""Compare the pinned worker's real Qwen updates, never replace PPO.

Requires the recorded environment and official scripted task fixtures. The
nonuniform advantage is explicitly a test fixture, not a DT/task-performance
claim. Run each variant on a verified idle GPU; no installation or downloads.
"""
import argparse, hashlib, inspect, json, os, re, textwrap, traceback
from pathlib import Path
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from torch.distributed.fsdp._flat_param import FlatParamHandle

p = argparse.ArgumentParser()
p.add_argument('--orig-params', choices=['current', 'previous', 'upstream'], default='current')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--artifacts', type=Path, required=True)
p.add_argument('--reference', type=Path)
p.add_argument('--deterministic', action='store_true', help='Use public Torch deterministic mode for the numerical comparison')
p.add_argument('--long-input', action='store_true', help='Explicit synthetic 32k capacity fixture; not native task length')
args = p.parse_args()
result = {'scope': 'real Qwen checkpoint and upstream worker; scripted input and nonuniform advantage fixture, not task training',
          'orig_params': args.orig_params, 'context_cap': 32768, 'minibatch': 4,
          'synthetic_long_input': args.long_input, 'stages': [],
          'torch_deterministic': args.deterministic,
          'flash_attention_deterministic': os.getenv('FLASH_ATTENTION_DETERMINISTIC', '0')}
torch.manual_seed(2026)
if args.deterministic:
    torch.use_deterministic_algorithms(True)
cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
cfg.data.max_prompt_length = 32256
cfg.data.max_response_length = 512
c = cfg.actor_rollout_ref
c.model.path = os.environ['MODEL_PATH']
c.model.lora_rank = 1
c.model.lora_alpha = 2
c.model.trust_remote_code = True
c.actor.strategy = 'fsdp'
c.actor.ppo_mini_batch_size = 4
c.actor.ppo_micro_batch_size_per_gpu = 1
c.actor.ppo_max_token_len_per_gpu = 32768
c.actor.use_torch_compile = False
c.actor.entropy_coeff = 0.0
c.actor.clip_ratio_c = float('inf')
c.actor.optim.total_training_steps = 3
c.actor.fsdp_config.model_dtype = 'bfloat16'
c.actor.fsdp_config.optimizer_offload = True
c.rollout.name = 'hf'
c.rollout.n = 1
c.rollout.tensor_model_parallel_size = 1
c.rollout.log_prob_micro_batch_size_per_gpu = 1
c.rollout.micro_batch_size = 4
if args.orig_params != 'current':
    fn = ActorRolloutRefWorker._build_model_optimizer
    src = textwrap.dedent(inspect.getsource(fn))
    setting = 'False' if args.orig_params == 'upstream' else 'self._is_lora'
    src, count = re.subn(r'use_orig_params=[^,\n]+,', f'use_orig_params={setting},', src)
    assert count == 1
    ns = dict(fn.__globals__)
    exec(compile(src, '<diagnostic-use_orig_params>', 'exec'), ns)
    ActorRolloutRefWorker._build_model_optimizer = ns[fn.__name__]
writeback = FlatParamHandle._writeback_tensor
def diagnostic_writeback(self, tensor, *a, **kw):
    try:
        return writeback(self, tensor, *a, **kw)
    except Exception:
        result['writeback_failure'] = {'fqns': self.flat_param._fqns,
            'shape': None if tensor is None else list(tensor.shape)}
        print('WRITEBACK', result['writeback_failure'], flush=True)
        raise
FlatParamHandle._writeback_tensor = diagnostic_writeback
try:
    worker = ActorRolloutRefWorker(c, 'actor_rollout')
    worker.init_model()
    result['stages'].append('init')
    result['actual_root_use_orig_params'] = worker.actor_module_fsdp._use_orig_params
    result['ppo_source_sha256'] = hashlib.sha256(
        (Path(os.environ['VERL_ROOT'])/'verl/trainer/ppo/core_algos.py').read_bytes()).hexdigest()
    fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
    rows = fixture['tasks']['Sokoban']['rows'][:4]
    data = {}
    for name in ['input_ids','attention_mask','responses']:
        width = 512 if name == 'responses' else 32768
        pad = 0 if name == 'attention_mask' else worker.tokenizer.pad_token_id
        values = []
        for row in rows:
            v = row[name]
            values.append([pad]*(width-len(v))+v)
        data[name] = torch.tensor(values, device='cuda')
    if args.long_input:
        # A resource fixture only: replace left padding with a fixed ordinary
        # token. Preserve each task response and every response mask exactly.
        filler = worker.tokenizer.encode(' context', add_special_tokens=False)[0]
        data['input_ids'][:, :-512].masked_fill_(data['attention_mask'][:, :-512] == 0, filler)
        data['attention_mask'][:, :-512] = 1
    data['position_ids'] = (data['attention_mask'].cumsum(-1)-1).clamp_min(0)
    batch = DataProto(batch=TensorDict(data,batch_size=[4]), meta_info={'temperature':1.0,
        'global_token_num': data['attention_mask'].sum(-1).tolist()})
    result['nonpadding_lengths'] = data['attention_mask'].sum(-1).tolist()
    result['response_lengths'] = data['attention_mask'][:, -512:].sum(-1).tolist()
    out = worker.compute_log_prob(batch)
    batch.batch['old_log_probs'] = out.batch['old_log_probs'].cuda()
    result['stages'].append('old_log_probs')
    artifacts = {'old_log_probs': out.batch['old_log_probs'].cpu()}
    # Explicit test fixture exercises nonzero policy gradients; no fabricated DT.
    batch.batch['advantages'] = torch.linspace(-.2,.3,512,device='cuda').expand(4,-1).clone()
    trainable = {n:p.detach().cpu().clone() for n,p in worker.actor_module_fsdp.named_parameters() if p.requires_grad}
    artifacts['trainable_before'] = trainable
    artifacts['updates'] = []
    for step in range(2):
        output = worker.update_actor(batch)
        result['stages'].append('update_'+str(step))
        result.setdefault('metrics',[]).append(output.meta_info['metrics'])
        print('UPDATE', step, output.meta_info['metrics'], flush=True)
        norms = output.meta_info['metrics']['actor/grad_norm']
        assert all(torch.isfinite(torch.tensor(x)) and x > 0 for x in norms)
        artifacts['updates'].append({n:p.detach().cpu().clone()
            for n,p in worker.actor_module_fsdp.named_parameters() if p.requires_grad})
    result['changed_elements'] = sum(int((p.detach().cpu()!=trainable[n]).sum())
        for n,p in worker.actor_module_fsdp.named_parameters() if p.requires_grad)
    assert result['changed_elements'] > 0
    artifacts['trainable_after'] = {n:p.detach().cpu().clone()
        for n,p in worker.actor_module_fsdp.named_parameters() if p.requires_grad}
    args.artifacts.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, args.artifacts)
    if args.reference:
        reference = torch.load(args.reference, map_location='cpu', weights_only=True)
        # Same weights, seed, inputs, update code and resolved FSDP config: require
        # exact equality, not a tolerance selected after seeing the differences.
        torch.testing.assert_close(artifacts, reference, rtol=0, atol=0)
        result['reference_identical'] = True
    result['status'] = 'passed'
except Exception as e:
    result['status'] = 'failed'
    result['error'] = str(e)
    traceback.print_exc()
finally:
    args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps(result,default=str),flush=True)
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
if result['status'] != 'passed': raise SystemExit(1)
