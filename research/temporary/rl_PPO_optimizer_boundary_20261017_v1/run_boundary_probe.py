"""Observe native clipping-to-AdamW boundary for the failed offload replay.

Reuse the pinned real CriticWorker replay and native optimizer hooks. No copied
clip, optimizer, PPO or value-model computation; no production changes.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
base = ROOT / 'receipts/PPO-full-response-update-replay-20261017-v1/source/run_update_probe.py'
assert hashlib.sha256(base.read_bytes()).hexdigest() == '2117d945e21a1e9d4b0b665e611fa204aa7b298870b3a68d821afaf8b67432b5'
tree = ast.parse(base.read_text())
version = 'PPO-optimizer-boundary-20261017-v1'
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace('PPO-full-response-update-replay-20261017-v1', version)

seam = r"""
for replay_node in nodes:
    if not isinstance(replay_node, ast.ClassDef) or replay_node.name != 'ReplayCritic':
        continue
    method = next(n for n in replay_node.body if isinstance(n, ast.FunctionDef) and n.name == 'replay_update')
    install = ast.parse('''
boundary_records = []
dispatch_records = []
parameters = dict(self.critic_module.named_parameters())

def small_native_state():
    selected = {}
    for name, parameter in parameters.items():
        if not (name.endswith('norm.weight') or name.endswith('score.weight')):
            continue
        item = dict(parameter=fingerprint(parameter), parameter_prefix=local(parameter).reshape(-1)[:16].float().tolist())
        item['optimizer'] = {}
        for key, value in self.critic_optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                item['optimizer'][key] = dict(fingerprint=fingerprint(value), prefix=local(value).reshape(-1)[:16].float().tolist())
            else:
                item['optimizer'][key] = value
        selected[name] = item
    return selected

def before_native_optimizer(optimizer, args, kwargs):
    boundary_records.append(dict(
        postclip_gradients={name:fingerprint(parameter.grad) for name,parameter in parameters.items() if parameter.grad is not None},
        postclip_selected_gradients={name:local(parameter.grad).reshape(-1)[:16].float().tolist()
            for name,parameter in parameters.items() if parameter.grad is not None and (name.endswith('norm.weight') or name.endswith('score.weight'))},
        before=small_native_state()))

def after_native_optimizer(optimizer, args, kwargs):
    boundary_records[-1]['after'] = small_native_state()

pre_hook = self.critic_optimizer.register_step_pre_hook(before_native_optimizer)
post_hook = self.critic_optimizer.register_step_post_hook(after_native_optimizer)
layer = self.critic_module.model.language_model.layers[0]
checkpoint = layer._gradient_checkpointing_func
def observed_native_checkpoint(*args, **kwargs):
    dispatch_records.append(dict(hidden_shape=list(args[1].shape), mode=self._memory_mode,
                                 bound=8192, native_branch=args[1].shape[1]<=8192))
    return checkpoint(*args, **kwargs)
layer._gradient_checkpointing_func = observed_native_checkpoint
''').body
    method.body[0:0] = install
    final = next(n for n in method.body if isinstance(n, ast.Try))
    final.finalbody += ast.parse('''
pre_hook.remove()
post_hook.remove()
layer._gradient_checkpointing_func = checkpoint
write(AUDIT/f'{label}-optimizer-boundary-rank{self.rank}.json',
      dict(records=boundary_records, checkpoint_dispatch=dispatch_records,
           optimizer_class=type(self.critic_optimizer).__module__+'.'+type(self.critic_optimizer).__qualname__,
           scope='Native optimizer pre/post hooks and exact original checkpoint delegation; diagnostic synchronization, not performance acceptance'))
''').body
"""
for index, node in enumerate(tree.body):
    if isinstance(node, ast.Expr) and ast.unparse(node).startswith('exec(compile('):
        tree.body[index:index] = ast.parse(seam).body
        break
else:
    raise AssertionError('Pinned native class load not found')

for node in ast.walk(tree):
    if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and ast.unparse(node.target) == '(label, mode)':
        node.iter.elts.append(ast.Tuple(elts=[ast.Constant('native_plain_after_install'), ast.Constant('native_plain')], ctx=ast.Load()))
        break
else:
    raise AssertionError('Original replay contrast not found')

for node in tree.body:
    if isinstance(node, ast.If) and ast.unparse(node.test) == "'--preflight' in sys.argv":
        node.body.insert(0, ast.parse('ray.cloudpickle.dumps(ReplayCritic.__ray_metadata__.modified_class)').body[0])
exec(compile(ast.fix_missing_locations(tree), str(base), 'exec'), globals())
