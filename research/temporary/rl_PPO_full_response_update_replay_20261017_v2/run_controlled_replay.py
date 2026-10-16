"""Reuse native replay with test-only deterministic Conv1d and actual operands.

No PPO, optimizer or owner implementation changes. Original native update,
state restore and observers are reused from the pinned v1 diagnostic. The
Conv1d flag is for this numerical reference only, never production or timing.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
base = ROOT / 'receipts/PPO-full-response-update-replay-20261017-v1/source/run_update_probe.py'
assert hashlib.sha256(base.read_bytes()).hexdigest() == '2117d945e21a1e9d4b0b665e611fa204aa7b298870b3a68d821afaf8b67432b5'
tree = ast.parse(base.read_text())
old_version = 'PPO-full-response-update-replay-20261017-v1'
new_version = 'PPO-full-response-update-replay-20261017-v2'
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace(old_version, new_version)

seam = r'''
def _install_actual_conv_capture(worker):
    import inspect
    layer = worker.critic_module.model.language_model.layers[0].linear_attn
    function = layer.causal_conv1d_fn
    worker._actual_conv_binding = dict(
        causal_conv1d_fn=None if function is None else repr(function),
        Conv1d_class=type(layer.conv1d).__qualname__,
        Conv1d_owner=inspect.getfile(type(layer.conv1d)),
        test_only_cudnn_deterministic=torch.backends.cudnn.deterministic,
    )
    handles = []

    def capture_native_call(module, args, output):
        if not output.requires_grad:
            return
        record = dict(worker._actual_conv_binding)
        record.update(rank=worker.rank, input_shape=list(args[0].shape),
                      input_stride=list(args[0].stride()), input_dtype=str(args[0].dtype),
                      weight_dtype=str(module.weight.dtype), output_shape=list(output.shape),
                      output_dtype=str(output.dtype),
                      kernel_size=module.kernel_size, stride=module.stride,
                      padding=module.padding, dilation=module.dilation, groups=module.groups)
        tensors = dict(input=args[0].detach().cpu(), weight=local(module.weight).clone(),
                       bias=None if module.bias is None else local(module.bias).clone())

        def capture_native_output_gradient(gradient):
            tensors['output_gradient'] = gradient.detach().cpu()
            record['output_gradient_stride'] = list(gradient.stride())
            record['output_gradient_dtype'] = str(gradient.dtype)
            torch.save(tensors, AUDIT / f'actual-native-conv-operands-rank{worker.rank}.pt')
            write(AUDIT / f'actual-native-conv-operands-rank{worker.rank}.json', record)
            return gradient

        output.register_hook(capture_native_output_gradient)
        handles[0].remove()

    handles.append(layer.conv1d.register_forward_hook(capture_native_call))

for replay_node in nodes:
    if isinstance(replay_node, ast.ClassDef) and replay_node.name == 'ReplayCritic':
        initialize = next(n for n in replay_node.body if isinstance(n, ast.FunctionDef) and n.name == 'initialize_replay')
        initialize.body.insert(0, ast.parse('torch.backends.cudnn.deterministic = True').body[0])
        for i, statement in enumerate(initialize.body):
            if isinstance(statement, ast.Expr) and ast.unparse(statement).startswith('super().init_model()'):
                initialize.body.insert(i + 1, ast.parse('_install_actual_conv_capture(self)').body[0])
                break
        else:
            raise AssertionError('Native initialization seam not found')
        returned = next(n for n in initialize.body if isinstance(n, ast.Return))
        returned.value.keywords.append(ast.keyword(arg='actual_conv_binding', value=ast.parse('self._actual_conv_binding', mode='eval').body))
'''
for index, node in enumerate(tree.body):
    if isinstance(node, ast.Expr) and ast.unparse(node).startswith('exec(compile('):
        tree.body[index:index] = ast.parse(seam).body
        break
else:
    raise AssertionError('Pinned native replay class loading not found')
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'status' for x in node.targets):
        node.value.keywords.append(ast.keyword(arg='test_only_cudnn_deterministic', value=ast.Constant(True)))
exec(compile(ast.fix_missing_locations(tree), str(base), 'exec'), globals())
