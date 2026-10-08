"""Instrument the existing layer diagnostic, without replacing its observers.

The observed native-reference drift justifies a lifecycle/observer isolation
test. Keep every frozen B4/UID, but use its already fixed first query only:
this tests preservation of model behavior, not single-delete quality on a
new smaller development set. Existing whole-collection metrics stay primary.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROBE = '''                    def native_probe(name):
                        query_list = [entry['queries'][0] if row < batch_spec['actual_rows'] else None
                                      for row, entry in enumerate(entries)]
                        width_probe = max(row['selected'].numel() for row in rows)
                        selected_probe = PackedAnswerTargets([row['case'] for row in rows],
                            [row['target_offsets'] for row in rows], width_probe, 'cuda')
                        selector_probe = NativeTargetLogitRows(selected_probe)
                        ids_probe = []
                        for row, query in zip(rows, query_list):
                            deleted_probe = row['selected'].clone()
                            if query is not None:
                                assert int(deleted_probe[query['packed_slot']]) == query['token_id']
                                deleted_probe[query['packed_slot']] = self.tokenizer.eos_token_id
                            ids_probe.extend((deleted_probe, row['selected']))
                        packed_probe = pad_2d_list_to_length([v.tolist() for v in ids_probe],
                            self.tokenizer.eos_token_id, max_length=width_probe).to('cuda')
                        def flags():
                            return dict(matmul_precision=torch.get_float32_matmul_precision(),
                                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                actor_training=self.actor_module_fsdp.training,
                                layer_training=[layer.training for layer in text.layers],
                                attention=text.config._attn_implementation,
                                native_fla_fp16=producer.native_fla_fp16,
                                native_conv_initial_states=runner.native_conv_initial_states,
                                native_FLA_bindings=[dict(layer=i, function=m.chunk_gated_delta_rule.__qualname__,
                                    unwrapped=inspect.unwrap(m.chunk_gated_delta_rule).__qualname__)
                                    for i, layer in enumerate(text.layers)
                                    if (m := getattr(layer, 'linear_attn', None)) is not None])
                        observation = dict(name=name, before_flags=flags(),
                            input_sha256=__import__('hashlib').sha256(packed_probe.cpu().numpy().tobytes()).hexdigest(),
                            paired_shape=list(packed_probe.shape))
                        save('native_lifecycle_probe_begin', batch=batch_index, probe=name)
                        tick_probe = time.perf_counter()
                        precision_probe = nullcontext()
                        if producer.native_fla_fp16:
                            from accelerated.qwen35.native_fla_precision import native_fla_fp16
                            precision_probe = native_fla_fp16(self.actor_module_fsdp)
                        text.set_attn_implementation('flash_attention_2')
                        with torch.no_grad(), precision_probe, globals_['_native_conv_initial_states_scope'](text.layers, runner.native_conv_initial_states):
                            output_probe = runner.model.forward_root(input_ids=packed_probe,
                                attention_mask=torch.ones_like(packed_probe), use_cache=False,
                                logits_to_keep=selector_probe.rows)
                            logits_probe = selector_probe.pack_logits(output_probe.logits)
                            values_probe = selected_target_log_probs(logits_probe, selected_probe)
                            factual_probe = selected_probe.sample_sums(values_probe[1::2].double()).cpu()
                            deleted_probe = selected_probe.sample_sums(values_probe[0::2].double()).cpu()
                        observation.update(after_flags=flags(), seconds=time.perf_counter()-tick_probe,
                            factual_logp=factual_probe.tolist(), deleted_logp=deleted_probe.tolist())
                        batch.setdefault('lifecycle_probes', []).append(observation)
                        record['operations']['native_forward'] += 1
                        del packed_probe, output_probe, logits_probe, values_probe
                        save('native_lifecycle_probe_complete', batch=batch_index, probe=name)
                    native_probe('before_DT_repeat_1')
                    native_probe('before_DT_repeat_2')
'''


def main():
    original_path = HERE/'inspect_layer_collection.py'
    original = original_path.read_text(encoding='utf-8')
    text = original
    marker = "                    save('DT_begin', batch=batch_index, original_readout=True, observer_argument=False)"
    assert text.count(marker) == 1
    text = text.replace(marker, PROBE+marker)
    marker = "                    assert set(bank) == set(range(33)) and captured"
    assert text.count(marker) == 1
    text = text.replace(marker, "                    native_probe('after_DT_without_observer_hooks')\n"+marker)
    marker = "                    rounds = max(spec['batches'][i]['native_paired_forwards'] for i in (pair_index, pair_index+1))"
    assert text.count(marker) == 1
    text = text.replace(marker, "                    rounds = 1  # Fixed first query per existing B4; preservation test, not replacement quality sample.")
    marker = "                    handles.clear()\n                    bank.clear()"
    assert text.count(marker) == 1
    text = text.replace(marker, "                    handles.clear()\n                    native_probe('after_original_observer_hooks_removed')\n                    bank.clear()")
    marker = "            record = dict(scope=__doc__, task=task, rank=self.rank, pid=os.getpid(),"
    assert text.count(marker) == 1
    text = text.replace(marker, "            record = dict(scope='Original DT and layer-observer lifecycle isolation on all frozen B4 groups; not a replacement quality sample.', task=task, rank=self.rank, pid=os.getpid(),")
    text = text.replace('class LayerCollectionWorker(', 'class LifecycleCollectionWorker(').replace('return LayerCollectionWorker', 'return LifecycleCollectionWorker')
    ast.parse(text)
    output = HERE/'inspect_native_dt_lifecycle.py'
    output.write_text(text, encoding='utf-8', newline='\n')
    diff = HERE/'native-dt-lifecycle-observer.diff'
    diff.write_text(''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
        fromfile='existing-inspect_layer_collection.py', tofile='isolated-inspect_native_dt_lifecycle.py')), encoding='utf-8', newline='\n')
    receipt = dict(scope=__doc__, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original=dict(path=str(original_path), sha256=hashlib.sha256(original_path.read_bytes()).hexdigest()),
        generated=dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
        diff=dict(path=str(diff), sha256=hashlib.sha256(diff.read_bytes()).hexdigest()),
        production_patches=0, optimizer=0, rollout=0, candidate=False,
        expected_per_rank=dict(DT_B4=6, native_B8=30, fixed_B4_groups=6))
    (HERE/'native-dt-lifecycle-preparation.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
