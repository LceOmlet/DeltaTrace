"""Check the current native FLA cache path against the pinned forward assertion.

Uses saved real first-rollout IDs, native Qwen/cache, existing passive captures,
and the original FLA FP32 reference/assert_close. No DT rule or PPO is changed.
This diagnoses layer-zero cache arithmetic, not whole-network credit accuracy.
"""
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import fla.utils
from verify_author_rollout import configuration
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from deltatrace_rollout import DeltaTraceRolloutProducer
from accelerated.qwen35.native_fla_precision import native_fla_fp16
from inspect_event_numerics import native_prefix_diagnostic
from verify_official_kernel_tolerances import load


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    root = Path(os.environ['DT_RUNTIME_ROOT'])
    rec = root/'receipts/rollout-major-cost'
    output = rec/'linear-return-cached-owner.json'
    started = time.perf_counter()
    report = dict(scope=__doc__, checks=[])
    def save(phase):
        report.update(phase=phase, seconds=time.perf_counter()-started)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(phase=phase, seconds=report['seconds'])), flush=True)
    save('native_actor_init')
    worker = ActorRolloutRefWorker(configuration(1024).actor_rollout_ref, 'actor')
    worker.init_model()
    producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp.eval(),
        eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=worker.tokenizer.pad_token_id)
    runtime = producer.runner
    model = runtime.model.model.language_model
    original_attention = model.config._attn_implementation
    payload = json.loads((rec/'linear-return-first-minimum.json').read_text())
    artifact = torch.load(rec/'linear-return-boundary-audit.pt', map_location='cpu', weights_only=False)
    pair = artifact['pair'].cuda()
    cases = [dict(target_ids=torch.tensor(s['selected_input_ids'][-1:]),
                  prompt_length=len(s['selected_input_ids'])-1) for s in payload['samples']]
    labels = producer.readout.alphabet.label_ids(worker.tokenizer)
    assert labels == payload['outcome_token_ids']
    del artifact
    try:
        model.set_attn_implementation('flash_attention_2')
        with native_fla_fp16(producer.actor):
            save('native_full_and_cache_capture')
            observed = native_prefix_diagnostic(runtime, pair[1::2], pair[0::2],
                cases, labels, producer, output)
        # That historical capture helper also reports native BF16 head scores.
        # They are not the current FP32 categorical readout and are not compared.
        observed.pop('full_logp', None)
        observed.pop('cached_logp', None)
        report['native_hidden_boundaries'] = observed
        runtime.model.release_owner_params()
        save('pinned_fla_reference')
        saved = torch.load(output.with_suffix('.pt'), map_location='cpu', weights_only=True)
        cut = saved['cut']
        full, cached = saved['full_gdn'], saved['cached_gdn']
        owner = load('pinned_fla_test', root/'receipts/training-setup/official-kernel-tests/test_gated_delta_v041.py')
        assert not fla.utils.FLA_CI_ENV
        report['native_operand_dtypes'] = {k:str(v.dtype) for k,v in cached['endpoints'].items() if isinstance(v,torch.Tensor)}
        def reference(part, initial=None):
            v, e = part['values'], part['endpoints']
            return owner.recurrent_gated_delta_rule_ref(
                q=F.normalize(v['raw_q'][:2].cuda(), p=2, dim=-1),
                k=F.normalize(v['raw_k'][:2].cuda(), p=2, dim=-1),
                v=e['v'][:2].cuda(), beta=e['beta'][:2].cuda(),
                g=e['raw_g'][:2].cuda(), scale=part['scale'], initial_state=initial)[0]
        full_ref = reference(full)
        local_ref = reference(cached, cached['endpoints']['h'][:2,0].cuda().float())
        for name, ref, actual in [
            ('native_full', full_ref, full['endpoints']['o'][:2].cuda()),
            ('native_cached_vs_full_reference', full_ref[:,cut:], cached['endpoints']['o'][:2].cuda()),
            ('native_cached_given_native_state', local_ref, cached['endpoints']['o'][:2].cuda())]:
            row = dict(name=name, threshold=.005,
                rms_ratio=float(fla.utils.get_err_ratio(ref,actual)),
                max_abs=float(fla.utils.get_abs_err(ref,actual)))
            try:
                fla.utils.assert_close(name, ref, actual, .005)
                row['status'] = 'passed'
            except AssertionError as exc:
                row.update(status='failed', error=str(exc))
            report['checks'].append(row)
            print(json.dumps(row), flush=True)
        save('completed')
    except BaseException as exc:
        report['error'] = repr(exc)
        save('failed')
        raise
    finally:
        model.set_attn_implementation(original_attention)
        runtime.model.release_owner_params()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
