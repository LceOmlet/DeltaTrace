"""Bounded observation of native prefix variation, without a numerical repair.

Observe original layer outputs, then the selected original FA/FLA call. Invoke
the pinned reference and its unchanged forward assertions on real operands.
Cross-length whole-layer differences are observations, not new tolerances.
"""
import ast
import hashlib
import importlib.util
import inspect
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def first_changed_layer(observations):
    """Return the first observed unequal layer, without an error threshold."""
    return next((row['layer'] for row in observations if not row['equal']), None)


def synchronize_observation_layer(model, selected, layer_count):
    """Use the original process group so every FSDP rank schedules both reads."""
    value = layer_count if selected is None else selected
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        mesh = getattr(model.lm_head.weight, 'device_mesh', None)
        if mesh is not None and mesh.ndim > 1:
            mesh = mesh['fsdp']
        group = mesh.get_group() if mesh is not None else None
        agreed = torch.tensor(value, device=model.execution_device, dtype=torch.long)
        torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN, group=group)
        value = int(agreed.item())
    return None if value == layer_count else value


def save_operands(directory, label, tensors, *, layer, scope):
    """Persist the exact observed operands; never modify a model computation."""
    if directory is None:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = directory/f'prefix-operator-layer{layer}-{label}-rank{rank}.pt'
    if path.exists():
        raise FileExistsError(f'Preserve the existing original operand receipt: {path}')
    values = {name: value.detach().cpu() for name, value in tensors.items()}
    torch.save(dict(layer=layer, case=label, scope=scope, tensors=values), path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                tensors={name: dict(dtype=str(value.dtype), shape=list(value.shape))
                         for name, value in values.items()})


@torch.no_grad()
def check_first_attention(cases, prefix, official, save, *, layer_index=3,
                          query_window_tokens=None, operand_output_dir=None):
    """Invoke exact pinned FA reference functions and output assertion."""
    assert hashlib.sha256(official.read_bytes()).hexdigest() == 'a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9'
    tree=ast.parse(official.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('attention_ref','construct_local_mask')]
    namespace=dict(torch=torch,F=F,math=math,rearrange=rearrange,repeat=repeat)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),
                 str(official),'exec'),namespace)
    test=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
              and n.name=='test_flash_attn_output')
    assertions=[n for n in test.body if isinstance(n,ast.Assert)
                and 'out_ref' in ast.unparse(n)]
    original=compile(ast.fix_missing_locations(ast.Module(body=assertions,type_ignores=[])),
                     str(official),'exec')
    checks=[]
    for label,case in cases:
        observed=case['attention']
        arguments=observed.dense_arguments
        assert observed.calls['native_dense']==1 and observed.calls['native_varlen']==0
        query_start = max(0, prefix-query_window_tokens) if query_window_tokens is not None else 0
        q=observed.values['dense_q'][:,query_start:prefix].to('cuda')
        k,v=[observed.values['dense_'+n][:,:prefix].to('cuda') for n in ('k','v')]
        assert arguments['causal'] and arguments['dropout_p']==0
        assert arguments['softmax_scale'] in (None,q.shape[-1]**-.5)
        tick=time.perf_counter()
        out_ref,_=namespace['attention_ref'](q,k,v,causal=True)
        out_pt,_=namespace['attention_ref'](q,k,v,causal=True,upcast=False,reorder_ops=True)
        out=observed.values['attention_output'][:,query_start:prefix].to('cuda')
        row=dict(case=label,query_shape=list(q.shape),key_shape=list(k.shape),
                 layer=layer_index, actual_query_start=query_start,
                 dtype=str(q.dtype),native_arguments=arguments,
                 native_max_error=float((out-out_ref).abs().max()),
                 original_low_precision_max_error=float((out_pt-out_ref).abs().max()),
                 native_calls=observed.calls,
                 original_assertions=[ast.unparse(n) for n in assertions])
        row['saved_actual_operands'] = save_operands(operand_output_dir, label,
            dict(q=q, k=k, v=v, native_output=out), layer=layer_index,
            scope='Actual query window and its complete causal prefix KV from the original dense call; unchanged FA reference and assertion.')
        try:
            exec(original,dict(out=out,out_ref=out_ref,out_pt=out_pt))
            row['original_fa_output_assertion']='passed'
        except AssertionError as error:
            row['original_fa_output_assertion']='failed'
            row['failure']=str(error)
        torch.cuda.synchronize()
        row['reference_and_check_seconds']=time.perf_counter()-tick
        checks.append(row)
        save('actual_first_attention_owner_checked',first_attention_checks=checks,
             official_fa_source=dict(path=str(official),sha256=hashlib.sha256(official.read_bytes()).hexdigest()),
             reference_tf32=torch.backends.cuda.matmul.allow_tf32)
        del q,k,v,out,out_ref,out_pt
    return checks


@torch.no_grad()
def diagnose(runner, ids, prefix, save, *, comparison_ids=None, matched_rows=None,
             prepared_prefix_fields=None, cache_tensors=None, gdn_layer_index=0,
             operand_output_dir=None):
    from accelerated.qwen35.qwen35_code_local_capture import NativeGDNCapture
    from accelerated.qwen35.qwen35_code_local_capture import NativeDenseAttentionCapture
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    import fla.utils as owner
    # The owning comparison can become warning-only in CI. Keep its original
    # strict assertion behavior, as the existing official test entry does.
    assert not owner.FLA_CI_ENV

    # The probe is below receipts/owner-b8-dispatch-20260930/<attempt>.
    official = Path(__file__).parents[2] / 'training-setup/official-kernel-tests/test_gated_delta_v041.py'
    assert hashlib.sha256(official.read_bytes()).hexdigest() == '35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
    spec = importlib.util.spec_from_file_location('pinned_prefix_fla_reference', official)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    test = next(n for n in ast.parse(official.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'test_chunk')
    assertions = [n for n in test.body if isinstance(n, ast.Expr)
                  and isinstance(n.value, ast.Call)
                  and isinstance(n.value.func, ast.Name)
                  and n.value.func.id == 'assert_close'
                  and n.value.args[0].value in ('o', 'ht')]
    original_assertions = compile(ast.fix_missing_locations(ast.Module(
        body=assertions, type_ignores=[])), str(official), 'exec')

    class FirstOperator(NativeGDNCapture):
        def event(self, frame, kind, value):
            super().event(frame, kind, value)
            if self.codes.get(frame.f_code) == 'stage' and kind == 'return' and value is not None:
                self.operator_o = frame.f_locals['o'].detach().cpu()
                self.operator_u = frame.f_locals['u'].detach().cpu()
                self.final_state = frame.f_locals['final_state'].detach().cpu()

    def first(value):
        return value[0] if isinstance(value, (tuple, list)) else value

    layers = runner.model.model.language_model.layers
    attention_only=os.environ.get('DT_PREFIX_ATTENTION_DIAGNOSTIC')=='1'
    requested_layer = os.environ.get('DT_PREFIX_COMPONENT_LAYER')
    observed_layer_index = None if requested_layer is None else int(requested_layer)
    auto_layer = gdn_layer_index is None and observed_layer_index is None
    bounded_operator = auto_layer or observed_layer_index is not None
    attention_layer_index = 3
    if observed_layer_index is not None:
        if not 0 <= observed_layer_index < len(layers):
            raise ValueError('Recorded observation layer is outside the original model')
        # Every rank receives the same explicit diagnostic choice. Reuse the
        # recorded first-difference evidence instead of repeating the two
        # automatic model probes or selecting another rank's earlier layer.
        attention_only = not hasattr(layers[observed_layer_index], 'linear_attn')
        if attention_only:
            attention_layer_index = observed_layer_index
        else:
            gdn_layer_index = observed_layer_index
    if auto_layer:
        if comparison_ids is None or not matched_rows:
            raise ValueError('Automatic layer diagnosis needs the observed source and original consumer rows')

        def output_probe(input_ids, row_indices):
            values, handles = {}, []
            def record(index, _module, _args, value):
                values[index] = first(value)[row_indices, max(0, prefix-64):prefix].detach().cpu()
            try:
                for i, layer in enumerate(layers):
                    handles.append(layer.register_forward_hook(
                        lambda m,a,o,i=i: record(i,m,a,o)))
                result = runner.forward_prefix(input_ids)
                del result
                torch.cuda.synchronize()
                return values
            finally:
                for handle in handles:
                    handle.remove()

        save('actual_lease_layer_probe_start', observation_window_tokens=min(64,prefix),
             matched_rows=matched_rows,
             scope='Two unchanged B4 native forwards; compare only the recorded target row at the last 64 prefix positions. Equality locates variation and is not a numerical acceptance threshold.')
        long_outputs = output_probe(ids, [row[0] for row in matched_rows])
        short_outputs = output_probe(comparison_ids, [row[1] for row in matched_rows])
        variation = [dict(layer=i, equal=bool(torch.equal(long_outputs[i],short_outputs[i])),
            dtype=str(long_outputs[i].dtype), shape=list(long_outputs[i].shape),
            max_absolute_difference=float((long_outputs[i].float()-short_outputs[i].float()).abs().max()))
            for i in range(len(layers))]
        del long_outputs, short_outputs
        own_selected = first_changed_layer(variation)
        selected = synchronize_observation_layer(runner.model, own_selected, len(layers))
        save('actual_lease_first_layer_observed', layer_output_variation=variation,
             first_observed_unequal_layer=own_selected,
             shared_observation_layer=selected,
             scope='First unequal layer within the recorded target row/window; no claim about every earlier prefix position or whole-DT tolerance.')
        if selected is None:
            save('component_diagnostic_no_observed_variation',
                 numerical_scope='No unequal layer in this observation window; no FA/FLA or whole-DT acceptance is claimed.')
            return []
        attention_only = not hasattr(layers[selected], 'linear_attn')
        if attention_only:
            attention_layer_index = selected
        else:
            gdn_layer_index = selected
    sources = [official, Path(inspect.getsourcefile(owner.assert_close)),
               Path(inspect.getsourcefile(chunk_gated_delta_rule_fwd_h)),
               Path(inspect.getsourcefile(NativeGDNCapture)), Path(__file__)]
    save('component_diagnostic_start', scope=__doc__,
         explicit_observation_layer=observed_layer_index,
         observation_window_tokens=min(64,prefix) if bounded_operator else prefix,
         gdn_layer_index=None if attention_only else gdn_layer_index,
         attention_layer_index=attention_layer_index if attention_only else None,
         owner_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         owner_fla_ci_environment=owner.FLA_CI_ENV,
         model_parameter_scope='Original actor init from unchanged model.path and native LoRA init, not a restored formal optimizer or policy checkpoint.',
         original_forward_assertions=[] if attention_only else [ast.unparse(n) for n in assertions])

    def capture(input_ids, *, retain_cache_fields=False):
        values = {}
        handles = []
        operator = None if attention_only else FirstOperator(layers[gdn_layer_index].linear_attn, device='cpu')
        entered = [False]
        attention=None
        attention_entered=[False]
        if attention_only:
            from flash_attn import flash_attn_func,flash_attn_varlen_func
            from transformers.integrations.flash_attention import flash_attention_forward
            attention=NativeDenseAttentionCapture(layers[attention_layer_index].self_attn,flash_attention_forward,
                flash_attn_varlen_func,flash_attn_func,destination='cpu',
                retained_names=('dense_q','dense_k','dense_v','attention_output'))

        def output(name, _module, _args, value):
            # Keep complete operands for the existing first-four-layer probe;
            # later layers need only the last 64 prefix positions to locate
            # the earliest variation without retaining all activations.
            layer_index = int(name.split('.')[0][5:])
            begin = (max(0, prefix-64) if bounded_operator or (layer_index >= 4 and
                      not gdn_layer_index-2 <= layer_index <= gdn_layer_index) else 0)
            values[name] = first(value)[:, begin:prefix].detach().cpu()

        def projection_reference(module, args, value):
            # Observe the owner's actual base Linear call. This FP32 read is
            # diagnostic only, not a replacement, correction or new tolerance.
            weight = module.weight.detach()
            if hasattr(weight, 'to_local'):
                weight = weight.to_local()
            actual = first(value)[:, max(0, prefix-64):prefix].detach()
            hidden = args[0][:, max(0, prefix-64):prefix].detach()
            if tuple(weight.shape) != (actual.shape[-1], hidden.shape[-1]):
                values['base_qkv_reference_unavailable'] = 'Owner weight is not gathered at this hook; no extra gather performed'
                return
            previous = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                bias = module.bias
                ref = F.linear(hidden.float(), weight.float(),
                               None if bias is None else bias.detach().float())
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
            values['base_qkv_reference_input'] = hidden.cpu()
            values['base_qkv_reference_output'] = ref.cpu()
            values['base_qkv_actual_output'] = actual.cpu()

        def begin(_module, args, kwargs):
            hidden = args[0] if args else kwargs['hidden_states']
            values[f'layer{gdn_layer_index}.input'] = hidden[:, :prefix].detach().cpu()
            operator.__enter__()
            entered[0] = True

        def end(_module, _args, _value):
            operator.__exit__(None, None, None)
            entered[0] = False

        def begin_attention(_module,_args,_value):
            attention.__enter__()
            attention_entered[0]=True

        def end_attention(_module,_args,_value):
            attention.__exit__(None,None,None)
            attention_entered[0]=False

        try:
            if not attention_only:
                base_projection = layers[gdn_layer_index].linear_attn.in_proj_qkv
                base_projection = getattr(base_projection, 'base_layer', base_projection)
                handles.append(base_projection.register_forward_hook(projection_reference))
            if operator is not None:
                handles.append(layers[gdn_layer_index].linear_attn.register_forward_pre_hook(begin, with_kwargs=True))
                handles.append(layers[gdn_layer_index].linear_attn.register_forward_hook(end))
            if attention is not None:
                # Enter before the selected layer begins, so the observer installs
                # its normal hooks without mutating the executing attention.
                handles.append(layers[attention_layer_index-1].register_forward_hook(begin_attention))
                handles.append(layers[attention_layer_index].register_forward_hook(end_attention))
            observed_layers = layers if prepared_prefix_fields is not None else layers[:4]
            for i, layer in enumerate(observed_layers):
                points = [('input_norm', layer.input_layernorm), ('output', layer),
                          ('mlp', layer.mlp)]
                if hasattr(layer, 'linear_attn'):
                    points += [('qkv', layer.linear_attn.in_proj_qkv),
                               ('gdn', layer.linear_attn),
                               ('out_projection', layer.linear_attn.out_proj)]
                else:
                    points += [('q_projection', layer.self_attn.q_proj),
                               ('k_projection', layer.self_attn.k_proj),
                               ('v_projection', layer.self_attn.v_proj),
                               ('attention', layer.self_attn)]
                for label, module in points:
                    handles.append(module.register_forward_hook(
                        lambda m, a, o, name=f'layer{i}.{label}': output(name, m, a, o)))
            tick = time.perf_counter()
            result = runner.forward_prefix(input_ids)
            stored_state = (None if attention_only else
                            result.past_key_values.layers[gdn_layer_index].recurrent_states.detach().cpu())
            cache_fields = cache_tensors(result.past_key_values) if retain_cache_fields else None
            del result
            torch.cuda.synchronize()
            return dict(values=values, operator=operator, attention=attention, stored_state=stored_state,
                        cache_fields=cache_fields,
                        seconds=time.perf_counter()-tick)
        finally:
            if entered[0]:
                operator.__exit__(None, None, None)
            if attention_entered[0]:
                attention.__exit__(None,None,None)
            for handle in handles:
                handle.remove()

    long = capture(ids)
    save('long_native_components_captured', native_forward_seconds=long['seconds'])
    short = capture(ids[:, :prefix] if comparison_ids is None else comparison_ids,
                    retain_cache_fields=prepared_prefix_fields is not None)
    differences = []
    for name, expected in short['values'].items():
        if not isinstance(expected, torch.Tensor) or not isinstance(long['values'].get(name), torch.Tensor):
            continue
        actual = long['values'][name]
        if matched_rows is not None:
            actual = actual[[row[0] for row in matched_rows]]
            expected = expected[[row[1] for row in matched_rows]]
        differences.append(dict(component=name, dtype=str(actual.dtype), shape=list(actual.shape),
            equal=bool(torch.equal(expected, actual)),
            max_absolute_difference=float((expected.float()-actual.float()).abs().max()),
            owner_error_ratio=float(owner.get_err_ratio(expected, actual))))
    save('native_component_variation_observed', shorter_native_forward_seconds=short['seconds'],
         component_variation=differences, matched_rows=matched_rows,
         base_projection_fp32_observations=[dict(case=label,
             dtype=str(case['values']['base_qkv_actual_output'].dtype),
             max_absolute_error=float((case['values']['base_qkv_actual_output'].float()-
                                       case['values']['base_qkv_reference_output']).abs().max()),
             scope='Actual base Linear on the last 64 prefix positions; diagnostic FP32 reference with TF32 disabled, no acceptance threshold')
             for label,case in [('long_readout',long),('direct_short',short)]
             if 'base_qkv_reference_output' in case['values']])

    if prepared_prefix_fields is not None:
        cache_variation=[]
        for key,expected in short['cache_fields'].items():
            actual=prepared_prefix_fields[key]
            rows=[row[1] for row in matched_rows]
            actual,expected=actual[rows],expected[rows]
            cache_variation.append(dict(layer=key[0],field=key[1],
                dtype=str(actual.dtype),reference_dtype=str(expected.dtype),shape=list(actual.shape),
                equal=bool(torch.equal(actual,expected)),
                max_absolute_difference=float((actual.float()-expected.float()).abs().max()),
                owner_error_ratio=float(owner.get_err_ratio(expected,actual))))
        save('actual_prefix_cache_variation_observed',cache_variation=cache_variation)

    if not attention_only and matched_rows is not None:
        operand_variation=[]
        for name in ('raw_q','raw_k','v','beta','raw_g','k','w','g','stage_u','stage_final_state'):
            def value(case):
                operator=case['operator']
                if name=='stage_u':return operator.operator_u[:, :prefix]
                if name=='stage_final_state':return operator.final_state
                if name in ('raw_q','raw_k'):return operator.values[name][:, :prefix]
                return operator.endpoints[name][:, :prefix]
            # A full final state consumes more tokens in the long case; only
            # compare actual prefix operands, not unlike final states.
            if name=='stage_final_state':continue
            actual=value(long)[[row[0] for row in matched_rows]]
            expected=value(short)[[row[1] for row in matched_rows]]
            operand_variation.append(dict(operand=name,dtype=str(actual.dtype),shape=list(actual.shape),
                equal=bool(torch.equal(actual,expected)),
                max_absolute_difference=float((actual.float()-expected.float()).abs().max()),
                owner_error_ratio=float(owner.get_err_ratio(expected,actual))))
        save('first_fla_prefix_operand_variation_observed',first_fla_operand_variation=operand_variation)

    if attention_only:
        checks=check_first_attention([('long_prefix',long),('direct_short',short)],prefix,
            official.with_name('test_flash_attn_v263.py'),save,
            layer_index=attention_layer_index,
            query_window_tokens=64 if bounded_operator else None,
            operand_output_dir=operand_output_dir)
        save('first_attention_diagnostic_complete',first_attention_checks=checks,
             peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
             physical_free_bytes=torch.cuda.mem_get_info()[0],
             numerical_scope='Original FA output assertion on the selected actual attention call; no gradient, whole-DT or Cache numerical acceptance is claimed.')
        return checks

    checks=[]
    for label, case in [('long_readout', long), ('direct_short', short)]:
        captured=case['operator']
        values, endpoints=captured.values, captured.endpoints
        operands={name: values['raw_'+name][:, :prefix].to('cuda') for name in ('q', 'k')}
        operands.update(v=endpoints['v'][:, :prefix].to('cuda'),
                        beta=endpoints['beta'][:, :prefix].to('cuda'),
                        g=endpoints['raw_g'][:, :prefix].to('cuda'))
        tick=time.perf_counter()
        ref, ref_ht=reference.recurrent_gated_delta_rule_ref(
            q=F.normalize(operands['q'], p=2, dim=-1),
            k=F.normalize(operands['k'], p=2, dim=-1), v=operands['v'],
            beta=operands['beta'], g=operands['g'], scale=captured.scale,
            output_final_state=True, initial_state=None)
        tri=captured.operator_o[:, :prefix].to('cuda')
        if label == 'long_readout':
            h, v_new, tri_ht=chunk_gated_delta_rule_fwd_h(
                k=endpoints['k'][:, :prefix].to('cuda').contiguous(),
                w=endpoints['w'][:, :prefix].to('cuda').contiguous(),
                u=captured.operator_u[:, :prefix].to('cuda').contiguous(),
                g=endpoints['g'][:, :prefix].to('cuda').contiguous(),
                initial_state=None, output_final_state=True)
            del h, v_new
        else:
            tri_ht=captured.final_state.to('cuda')
        row=dict(case=label, tokens=prefix, local_batch=ids.shape[0],
                 layer=gdn_layer_index,
                 actual_operand_dtypes={k: str(v.dtype) for k,v in operands.items()},
                 native_output_dtype=str(tri.dtype), native_state_dtype=str(tri_ht.dtype),
                 model_boundary_output_dtype=str(endpoints['o'].dtype),
                 model_boundary_output_error_ratio=float(owner.get_err_ratio(
                     ref,endpoints['o'][:, :prefix].to('cuda'))),
                 output_error_ratio=float(owner.get_err_ratio(ref,tri)),
                 state_error_ratio=float(owner.get_err_ratio(ref_ht,tri_ht)))
        recorded = {**operands, 'native_output':tri, 'native_prefix_final_state':tri_ht,
                    'model_boundary_output':endpoints['o'][:,:prefix]}
        recorded.update({'stage_'+name:endpoints[name][:,:prefix] for name in ('k','w','g')})
        recorded['u'] = captured.operator_u[:,:prefix]
        recorded['native_stored_cache_state'] = case['stored_state']
        if prepared_prefix_fields is not None:
            for name in ('conv_states','recurrent_states'):
                recorded['prepared_prefix_'+name] = prepared_prefix_fields[gdn_layer_index,name]
        row['saved_actual_operands'] = save_operands(operand_output_dir, label, recorded,
            layer=gdn_layer_index,
            scope='Actual native FLA operands and prefix state; full-length stored cache consumes this case input, while native_prefix_final_state consumes exactly the checked prefix. Unchanged FLA reference and assertions.')
        if label == 'direct_short':
            row['native_cache_state_dtype']=str(case['stored_state'].dtype)
            row['native_cache_state_error_ratio']=float(owner.get_err_ratio(
                ref_ht,case['stored_state'].to('cuda')))
        try:
            exec(original_assertions, dict(assert_close=owner.assert_close,
                 ref=ref, tri=tri, ref_ht=ref_ht, tri_ht=tri_ht))
            row['original_fla_forward_assertions']='passed'
        except AssertionError as error:
            row['original_fla_forward_assertions']='failed'
            row['failure']=str(error)
        torch.cuda.synchronize()
        row['reference_and_check_seconds']=time.perf_counter()-tick
        checks.append(row)
        save('actual_operand_owner_reference_checked', first_operator_checks=checks)
        del operands, ref, ref_ht, tri, tri_ht
    import psutil
    save('component_diagnostic_complete', first_operator_checks=checks,
         peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
         physical_free_bytes=torch.cuda.mem_get_info()[0],
         pss_bytes=psutil.Process().memory_full_info().pss,
         numerical_scope='Only the first native FLA output/state use the pinned FLA assertions. Layer/Cache variation has no invented whole-model threshold. No DT/PPO or production acceleration is modified.')
    return checks
