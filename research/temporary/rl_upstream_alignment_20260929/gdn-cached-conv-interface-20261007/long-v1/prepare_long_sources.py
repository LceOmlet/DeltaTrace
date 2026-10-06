"""Prepare immutable long actual-case diagnostic sources; never import Torch.

Keep the already-completed short-case files unchanged. This applies narrow,
checked textual changes to the original diagnostic observer/driver only. The
owner forward, finite backward, public convolution API, and tolerance AST are
not implemented or replaced here. Root decides when to stage/run the files.
"""
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]


def source(relative, expected):
    path = AUDIT / relative
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise RuntimeError(f'Diagnostic base changed: {relative}: {digest}')
    return data.decode('utf-8').replace('\r\n', '\n'), dict(path=str(path), sha256=digest, bytes=len(data))


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Expected one original diagnostic seam: ' + old[:100])
    return text.replace(old, new, 1)


driver, driver_source = source('collect_appworld_cached_convolution.py',
    '9a7216766ac8de6decd5b4151504088217bf33f699df9d11611481b94b099786')
driver = replace_once(driver,
    "requests = sorted(payload['requests'], key=lambda row: row['context_tokens'])[:4]",
    "requests = sorted(payload['requests'], key=lambda row: row['context_tokens'])[84:88]")
driver = replace_once(driver, "selected.append(dict(source_step=request['source_step'],",
    "selected.append(dict(source_row_index=index, context_tokens=request['context_tokens'],\n"
    "                                 source_step=request['source_step'],")
driver = replace_once(driver, "selected=selected))", "selected=selected,\n"
    "                           selection=dict(order='context_tokens', start=84, stop=88)))")
driver = replace_once(driver, 'Use literal saved environment responses and their original complete returns.',
    'Use the original long-case context-sorted requests at offsets 84:88 per rank,\n'
    'their literal environment responses, and their original complete returns.')

observer, observer_source = source('observe_native_gdn0_conv_once.py',
    '707563f7af0d36382a7e5b8922d0bd8c2e1bb48b2b47f4123e75c3b49a5ba3dc')
observer = replace_once(observer,
    'maximum_single_input_bytes=128*1024*1024,maximum_total_snapshot_bytes=768*1024*1024,',
    'maximum_single_input_bytes=384*1024*1024,maximum_total_snapshot_bytes=3*1024*1024*1024,')
observer = replace_once(observer,
    'hooks=[];native_shape=None\n   def before(_module,args,kwargs):\n'
    "    nonlocal native_shape\n    value=args[0] if args else kwargs['hidden_states']\n"
    '    native_shape=tuple(value.shape)\n    return None\n',
    '''hooks=[];native_shape=None;pending_native={}
   def pending_retain(name,tensor):
    try:
     n=tensor.numel()*tensor.element_size()
     if n>record['maximum_single_input_bytes']:return
     pending_native['cpu'][name]=tensor.detach().to(device='cpu',copy=True)
     pending_native['metadata'][name]=dict(shape=list(tensor.shape),stride=list(tensor.stride()),
       dtype=str(tensor.dtype),device=str(tensor.device),bytes=n)
    except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
   def before(_module,args,kwargs):
    nonlocal native_shape
    value=args[0] if args else kwargs['hidden_states']
    native_shape=tuple(value.shape)
    pending_native.clear();pending_native.update(cpu={},metadata={},cached=False)
    try:
     if not record['native_calls']:
      cache=kwargs.get('cache_params') if 'cache_params' in kwargs else (args[1] if len(args)>1 else None)
      if cache is not None and cache.has_previous_state(_module.layer_idx):
       pending_native['cached']=True
       # Read the actual prior state before native update_conv_state mutates or replaces it.
       pending_retain('native_conv_state',cache.layers[_module.layer_idx].conv_states)
    except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
    return None
   def after_projection(_module,args,output):
    try:
     if pending_native.get('cached') and not record['native_calls']:
      # Keep the actual Linear output; its native transpose defines the second cat input.
      pending_retain('native_projection_output',output)
    except Exception as error:record.setdefault('observation_errors',[]).append(repr(error))
    return None
''')
observer = replace_once(observer,
    "if n>record['maximum_single_input_bytes'] or 6*n+2*small_bytes>record['maximum_total_snapshot_bytes']:",
    "if n>record['maximum_single_input_bytes'] or 7*n+2*small_bytes>record['maximum_total_snapshot_bytes']:")
observer = replace_once(observer,
    "retain('native_weight',a['weight']);retain('native_bias',a['bias'])",
    """retain('native_weight',a['weight']);retain('native_bias',a['bias'])
      for name in ('native_conv_state','native_projection_output'):
       if name in pending_native.get('cpu',{}):
        metadata=pending_native['metadata'][name]
        if record['tensor_bytes']+metadata['bytes']<=record['maximum_total_snapshot_bytes']:
         payload[name]=pending_native['cpu'][name]
         record['tensors'][name]=metadata;record['tensor_bytes']+=metadata['bytes']
      record['calls']['native']['actual_preconcat_inputs']=dict(
       conv_state='native_conv_state' if 'native_conv_state' in payload else None,
       projection_output='native_projection_output' if 'native_projection_output' in payload else None,
       projection_to_mixed_qkv='Original transpose(1,2); no output reconstruction',
       source='Original GDN pre-hook and original in_proj_qkv forward-hook',
       observer_hook_returns_none=True,extra_projection_forwards=0)
      pending_native.clear()""")
observer = replace_once(observer,
    'pre=module.register_forward_pre_hook(before,with_kwargs=True)\n',
    'pre=module.register_forward_pre_hook(before,with_kwargs=True)\n'
    '   projection_hook=module.in_proj_qkv.register_forward_hook(after_projection)\n')
observer = replace_once(observer,
    'module.causal_conv1d_fn=old_conv;pre.remove()',
    'module.causal_conv1d_fn=old_conv;pre.remove();projection_hook.remove();pending_native.clear()')

comparator, comparator_source = source(
    'gdn-cached-conv-interface-20261007/compare_actual_cached_conv.py',
    '0c00ce9fb166f5e10e910a8fc1195f68616ee16dd1125c249ecb4cf238b4d7f9')
comparator = replace_once(comparator,
    "default=HERE / 'test_causal_conv1d_v150.py'", "default=HERE.parent / 'test_causal_conv1d_v150.py'")
comparator = replace_once(comparator,
    "def channel_last(x):\n        return x.transpose(1, 2).contiguous().transpose(1, 2)\n",
    """def channel_last(x):
        return x.transpose(1, 2).contiguous().transpose(1, 2)

    def original_layout(name):
        cpu = tensors[name]
        metadata = record['tensors'][name]
        if list(cpu.shape) != metadata['shape'] or str(cpu.dtype) != metadata['dtype']:
            raise RuntimeError('Actual operand metadata differs: ' + name)
        # Restore the captured original layout, not the post-cat slice layout.
        value = torch.empty_strided(cpu.shape, metadata['stride'], device=device, dtype=cpu.dtype)
        value.copy_(cpu)
        return value
""")
comparator = replace_once(comparator,
    "left, suffix = x[..., :context], channel_last(x[..., context:])\n        initial = channel_last(left[..., -(width - 1):]).detach()",
    """left, suffix = x[..., :context], channel_last(x[..., context:])
        original_cat_inputs = None
        if prefix == 'native':
            try:
                if 'native_conv_state' in tensors and 'native_projection_output' in tensors:
                    original_left = original_layout('native_conv_state')
                    original_projection = original_layout('native_projection_output')
                    original_mixed = original_projection.transpose(1, 2)
                    joined = torch.cat([original_left, original_mixed], dim=-1)
                    reproduced = torch.equal(joined, x)
                    report['actual_preconcat_reproduction'] = dict(equal=reproduced,
                        original_state_stride=list(original_left.stride()),
                        original_projection_stride=list(original_projection.stride()),
                        original_mixed_stride=list(original_mixed.stride()),
                        joined_stride=list(joined.stride()), recorded_cat_stride=record['tensors']['native_x']['stride'])
                    del joined
                    if reproduced:
                        original_cat_inputs = (original_left, original_mixed)
                        left, suffix = original_cat_inputs
                    else:
                        report['native_concat_benchmark_unavailable'] = 'Actual original inputs do not reproduce saved native_x; no substitute used'
                else:
                    report['native_concat_benchmark_unavailable'] = 'No actual state/projection operands; kernel-only comparison'
            except (RuntimeError, KeyError) as error:
                report['native_concat_benchmark_unavailable'] = repr(error)
        initial = channel_last(left[..., -(width - 1):]).detach()""")
comparator = replace_once(comparator,
    "def concat_call():\n                return conv(torch.cat([left, suffix_x], dim=-1), weight, bias, activation=activation)",
    """def concat_call():
                actual_state, actual_projection_view = original_cat_inputs
                return conv(torch.cat([actual_state, actual_projection_view], dim=-1), weight, bias, activation=activation)""")
comparator = replace_once(comparator,
    "if not backward:\n                operations.append(('native_concat_then_kernel', concat_call))",
    "if not backward and original_cat_inputs is not None:\n                operations.append(('native_actual_concat_then_kernel', concat_call))")
comparator = replace_once(comparator,
    "# The saved projection values are materialized channel-last once,\n            # before timing. Existing cat remains inside the native call timer.",
    "# Original native state/projection layouts are restored before timing.\n            # A concat timer is emitted only when those actual inputs reproduce x.")

files = {
    'collect_appworld_cached_convolution.py': driver,
    'observe_native_gdn0_conv_once.py': observer,
    'compare_actual_cached_conv.py': comparator,
}
outputs = []
for name, text in files.items():
    ast.parse(text, filename=name)
    if name.startswith('observe_'):
        syntax = ast.parse(text)
        script = next(n.value.value for n in syntax.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'SCRIPT' for t in n.targets))
        ast.parse(script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    path = HERE / name
    data = text.encode('utf-8')
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError('Do not overwrite prepared diagnostic source: ' + name)
    path.write_bytes(data)
    outputs.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
manifest = dict(status='prepared_only_CPU_AST_verified_not_staged_or_run',
    bases=[driver_source, observer_source, comparator_source], outputs=outputs,
    selection=dict(real_request_order='context_tokens', per_rank_start=84, per_rank_stop=88),
    snapshot_limits=dict(maximum_single_input_bytes=384*1024**2,
                         maximum_total_snapshot_bytes=3*1024**3),
    scope='One native cached conv and its one existing finite VJP; actual pre-concat state/projection data and layouts; no model/owner/output replacements',
    operations=dict(gpu_calls=0, remote_calls=0, checkpoint_loads=0, installs=0, production_edits=0))
(HERE / 'prepared-sources.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(manifest))
