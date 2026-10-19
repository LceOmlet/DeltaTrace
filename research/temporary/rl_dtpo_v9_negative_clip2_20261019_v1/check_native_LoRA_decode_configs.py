"""Bounded original-vLLM LoRA operator check; never installs a production config.

Use the pinned owner's test data, reference functions and dtype-specific assert.
Only rank 0 runs numerical kernels. Production model/DT/optimizer calls are zero.
"""
from pathlib import Path
import hashlib


def run_check(actor):
    import ast
    import dataclasses
    import json
    import time
    import torch
    import triton.testing
    import vllm.lora.ops.torch_ops as torch_ops
    import vllm.lora.ops.triton_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
    from vllm.lora.ops.triton_ops import utils

    if actor.rank != 0:
        return dict(status='not_selected', rank=actor.rank, numerical_calls=0)
    started = time.time()
    root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    audit = root / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
    source = audit / 'source'
    report = dict(status='running', started=started, rank=0,
                  actor_scheduler_epoch=actor.actor_lr_scheduler.last_epoch,
                  model_calls=0, DT_calls=0, optimizer_updates=0,
                  production_config_changed=False, cases=[])
    engine = actor.rollout.inference_engine.llm_engine.engine_core.engine_core
    runner = engine.model_executor.driver_worker.worker.model_runner
    signatures = {}
    for name, module in runner.model.named_modules():
        aa = getattr(module, 'lora_a_stacked', ())
        bb = getattr(module, 'lora_b_stacked', ())
        if isinstance(aa, torch.Tensor):
            aa = (aa,)
        if isinstance(bb, torch.Tensor):
            bb = (bb,)
        if not aa or not bb:
            continue
        sa = [list(x.shape) for x in aa]
        sb = [list(x.shape) for x in bb]
        key = json.dumps([sa, sb])
        signatures.setdefault(key, dict(module=name, A=sa, B=sb, count=0))['count'] += 1
    report['loaded_LoRA_shapes'] = list(signatures.values())
    selected = []
    for row in signatures.values():
        if len(row['A']) not in (1, 2) or len(row['A']) != len(row['B']):
            continue
        if len({tuple(x[-2:]) for x in row['A']}) != 1:
            continue
        if len({tuple(x[-2:]) for x in row['B']}) != 1:
            continue
        k = row['A'][0][-1]
        n = row['B'][0][-2]
        ns = len(row['A'])
        if (k, n, ns) in ((4096, 12288, 2), (12288, 4096, 1)):
            selected.append((row['module'], k, n, ns))
    report['selected_actual_shapes'] = selected
    if not selected:
        report.update(status='actual_representative_shapes_not_found', finished=time.time())
        return report

    # Extract unchanged test owners, avoiding their global pointer-cache clears.
    ns = dict(torch=torch, dataclass=dataclasses.dataclass, torch_ops=torch_ops)
    owners = []
    for name, expected, names in (
        ('upstream-vllm-v0.15.0/lora_utils.py',
         '9234e5d2faa2355b45c5087f7a13fcba2069deab59e638434dd77b53dc518467',
         {'PunicaTensors', 'assert_close', 'generate_data_for_nslices'}),
        ('upstream-vllm-v0.15.0/test_punica_ops.py',
         '97559608d7a6aea9008629172947caf107425148312da2700ecd4b469c38ca0f',
         {'sgmv_shrink_for_nslices', 'sgmv_expand_for_nslices'}),
    ):
        path = source / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected
        body = [node for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert len(body) == len(names)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)
        owners.append(dict(path=str(path), SHA256=actual))
    report['official_test_owners'] = owners
    report['official_test_version'] = 'v0.15.0'
    report['official_reference_url'] = 'https://github.com/vllm-project/vllm/blob/v0.15.0/tests/lora/test_punica_ops.py'
    report['official_assertion_url'] = 'https://github.com/vllm-project/vllm/blob/v0.15.0/tests/lora/utils.py#L95-L101'
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    caches = [(cache, dict(cache)) for cache in (utils._LORA_A_PTR_DICT, utils._LORA_B_PTR_DICT)]
    borrowed_configs = []
    held = []
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for module, hidden_in, hidden_out, nslices in selected:
                for op in ('shrink', 'expand'):
                    hidden = hidden_in if op == 'shrink' else hidden_out
                    data = ns['generate_data_for_nslices'](
                        32, hidden, 1, 8, 1, nslices, torch.bfloat16, op, 'cuda')
                    # The loaded Punica add_lora_linear uses an FP32 shrink buffer.
                    if op == 'expand':
                        data.inputs_tensor = data.inputs_tensor.float()
                    meta = LoRAKernelMeta.make(max_loras=1, max_num_tokens=32, device='cuda')
                    meta.prepare_tensors(data.token_lora_mapping)
                    held.extend((data, meta))
                    max_len, tokens = data.meta()
                    reference_args = (data.b_seq_start_loc, data.seq_len_tensor,
                                      data.prompt_lora_mapping, 32, max_len, tokens)
                    if op == 'shrink':
                        ns['sgmv_shrink_for_nslices'](
                            nslices, data.inputs_tensor, data.lora_weights,
                            data.ref_out_tensor, *reference_args, 1.0)
                        config = utils.get_lora_op_configs(
                            'shrink', max_loras=meta.active_lora_ids.size(0), batch=32,
                            hidden_size=hidden, rank=8, num_slices=nslices)
                        def call():
                            ops.lora_shrink(data.inputs_tensor, data.lora_weights,
                                            data.our_out_tensor,
                                            *meta.meta_args(token_nums=32), 1.0)
                        field, candidate = 'split_k', 16
                    else:
                        ns['sgmv_expand_for_nslices'](
                            nslices, hidden, data.inputs_tensor, data.lora_weights,
                            data.ref_out_tensor, *reference_args, add_inputs=True)
                        config = utils.get_lora_op_configs(
                            op_type='expand', max_loras=meta.active_lora_ids.size(0), batch=32,
                            hidden_size=hidden, rank=8, num_slices=nslices, add_inputs=True)
                        def call():
                            ops.lora_expand(data.inputs_tensor, data.lora_weights,
                                            data.our_out_tensor,
                                            *meta.meta_args(token_nums=32),
                                            offset_start=0, add_inputs=True)
                        field, candidate = 'block_m', 32
                    original = dict(config)
                    borrowed_configs.append((config, field, config[field]))
                    initial_output = data.our_out_tensor.clone()
                    checks = []
                    for label, value in (('original', original[field]),
                                         ('candidate', candidate),
                                         ('warm_original', original[field])):
                        config[field] = value
                        data.our_out_tensor.copy_(initial_output)
                        call()
                        ns['assert_close'](data.our_out_tensor, data.ref_out_tensor)
                        max_abs = (data.our_out_tensor.float() - data.ref_out_tensor.float()).abs().max().item()
                        # Original Triton timing owner, on a side stream as required.
                        ms = triton.testing.do_bench_cudagraph(call, rep=5, return_mode='median')
                        checks.append(dict(label=label, config=dict(config), ms=ms,
                                           official_assert_passed=True, max_abs=max_abs))
                    config[field] = original[field]
                    assert config == original
                    report['cases'].append(dict(module=module, op=op,
                        input_dtype=str(data.inputs_tensor.dtype), output_dtype=str(data.our_out_tensor.dtype),
                        input_shape=list(data.inputs_tensor.shape),
                        weight_shapes=[list(w.shape) for w in data.lora_weights],
                        atol=0.01 if op == 'shrink' else 0.06,
                        rtol=0.01 if op == 'shrink' else 0.06, checks=checks))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        report['status'] = 'completed_bounded_original_operator_check'
    except Exception as exc:
        report.update(status='failed_bounded_check_not_deployed', error=repr(exc))
    finally:
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for config, field, value in borrowed_configs:
            config[field] = value
        for cache, before in caches:
            for key in list(cache):
                if key not in before:
                    del cache[key]
            assert all(cache.get(key) is value for key, value in before.items())
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        report.update(finished=time.time(), production_configs_restored=True,
                      existing_pointer_cache_objects_preserved=True,
                      RNG_restored=True, production_efficiency_window=False,
                      scope='Bounded original LoRA operators only. Exclude this callback and whole training iteration from production efficiency. No production CUDA graphs recaptured or changed.')
        (audit / 'native-LoRA-decode-config-check-results.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    owner = Path(__file__).with_name('inspect_native_workers.py')
    assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
    code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1', 'bounded-native-LoRA-decode-config-check-v1')
    anchor = "exec(compile(code,str(owner),'exec'),globals())"
    payload = '''    import runpy
    module=runpy.run_path(%r,run_name='bounded_numerical_module')
    result['native_LoRA_decode_config_check']=module['run_check'](actor)
    return result''' % str(Path(__file__).resolve())
    assert code.count(anchor) == 1
    code = code.replace(anchor, "assert code.count('    return result')==1\ncode=code.replace('    return result'," + repr(payload) + ")\n" + anchor)
    exec(compile(code, str(owner), 'exec'), globals())
