"""Bounded observation of native prefix variation, without a numerical repair.

Observe the first four original layers and the first original FLA call. Invoke
the pinned FLA reference and its unchanged forward assertions on real operands.
Cross-length whole-layer differences are observations, not new tolerances.
"""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import time

import torch
import torch.nn.functional as F


@torch.no_grad()
def diagnose(runner, ids, prefix, save):
    from accelerated.qwen35.qwen35_code_local_capture import NativeGDNCapture
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    import fla.utils as owner

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
    sources = [official, Path(inspect.getsourcefile(owner.assert_close)),
               Path(inspect.getsourcefile(chunk_gated_delta_rule_fwd_h)),
               Path(inspect.getsourcefile(NativeGDNCapture)), Path(__file__)]
    save('component_diagnostic_start', scope=__doc__,
         owner_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         owner_fla_ci_environment=owner.FLA_CI_ENV,
         model_parameter_scope='Original actor init from unchanged model.path and native LoRA init, not a restored formal optimizer or policy checkpoint.',
         original_forward_assertions=[ast.unparse(n) for n in assertions])

    def capture(input_ids):
        values = {}
        handles = []
        operator = FirstOperator(layers[0].linear_attn, device='cpu')
        entered = [False]

        def output(name, _module, _args, value):
            values[name] = first(value)[:, :prefix].detach().cpu()

        def begin(_module, args, kwargs):
            hidden = args[0] if args else kwargs['hidden_states']
            values['layer0.input'] = hidden[:, :prefix].detach().cpu()
            operator.__enter__()
            entered[0] = True

        def end(_module, _args, _value):
            operator.__exit__(None, None, None)
            entered[0] = False

        try:
            handles.append(layers[0].linear_attn.register_forward_pre_hook(begin, with_kwargs=True))
            handles.append(layers[0].linear_attn.register_forward_hook(end))
            for i, layer in enumerate(layers[:4]):
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
            stored_state = result.past_key_values.layers[0].recurrent_states.detach().cpu()
            del result
            torch.cuda.synchronize()
            return dict(values=values, operator=operator, stored_state=stored_state,
                        seconds=time.perf_counter()-tick)
        finally:
            if entered[0]:
                operator.__exit__(None, None, None)
            for handle in handles:
                handle.remove()

    long = capture(ids)
    save('long_native_components_captured', native_forward_seconds=long['seconds'])
    short = capture(ids[:, :prefix])
    differences = []
    for name, expected in short['values'].items():
        actual = long['values'][name]
        differences.append(dict(component=name, dtype=str(actual.dtype), shape=list(actual.shape),
            equal=bool(torch.equal(expected, actual)),
            max_absolute_difference=float((expected.float()-actual.float()).abs().max()),
            owner_error_ratio=float(owner.get_err_ratio(expected, actual))))
    save('native_component_variation_observed', shorter_native_forward_seconds=short['seconds'],
         component_variation=differences)

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
                 actual_operand_dtypes={k: str(v.dtype) for k,v in operands.items()},
                 native_output_dtype=str(tri.dtype), native_state_dtype=str(tri_ht.dtype),
                 model_boundary_output_dtype=str(endpoints['o'].dtype),
                 model_boundary_output_error_ratio=float(owner.get_err_ratio(
                     ref,endpoints['o'][:, :prefix].to('cuda'))),
                 output_error_ratio=float(owner.get_err_ratio(ref,tri)),
                 state_error_ratio=float(owner.get_err_ratio(ref_ht,tri_ht)))
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
