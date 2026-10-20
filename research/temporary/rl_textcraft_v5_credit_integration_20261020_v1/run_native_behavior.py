"""Reuse the original real-payload v5 DT/u verification on all eight ranks."""
from prepare_campaign import ROOT, AUDIT

owner = ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source/run_native_behavior_v5.py'
code = owner.read_text().replace("AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'",
    "AUDIT=ROOT/'receipts/textcraft-v5-credit-integration-20261020-v1'")
code = code.replace('Native B4/rank credit transport and exact-32768 capacity, zero optimizer.',
    'Native eight-rank real-payload DT/u integration only; zero optimizer.')
code = code.replace("AUDIT/'prepared-launch.json'", "AUDIT/'dt-prepared.json'")
code = code.replace('total_training_steps=330', 'total_training_steps=200')
code = code.replace('physical_gpus=[2,3]', 'physical_gpus=list(range(8))')
code = code.replace('num_cpus=8,num_gpus=2', 'num_cpus=16,num_gpus=8')
code = code.replace("_temp_dir='/tmp/ray-tc-exponential-v5'", "_temp_dir='/tmp/ray-v5-integration-20261020'")
code = code.replace('RayResourcePool([2]', 'RayResourcePool([8]')
# Select only the original two native parser payloads. No synthetic capacity
# padding or duplicate model/optimizer quality experiment is required here.
start = code.index('    small=base.requests(')
end = code.index('    ray.init(', start)
code = code[:start] + code[end:]
start = code.index("        reused=AUDIT/")
end = code.index('        for capacity in ', start)
code = code[:start] + code[end:]
start = code.index("        if '--multi-only' in sys.argv:", code.index("state.update(status='completed'"))
end = code.index("        write(AUDIT/'native-behavior-status.json',state)", start)
code = code[:start] + code[end:]
# The old two-rank oracle overwrote original event values with RPC padding
# copies. Mark copies using the native pad result, so expected values come
# from the original returned row even when duplicate DT evaluations round
# differently. Neither padding nor model/credit calculations are replaced.
anchor = '    returned={}\n'
assert code.count(anchor) == 1
code = code.replace(anchor, '''    returned={}
    import verl.protocol as protocol
    import numpy as np
    native_pad = protocol.pad_dataproto_to_divisor
    def mark_padding(data, divisor):
        original_count = len(data)
        result, count = native_pad(data, divisor)
        result.non_tensor_batch['verification_padding_copy'] = np.arange(len(result)) >= original_count
        return result, count
''')
anchor = "            for i,identity in enumerate(requests.non_tensor_batch['traj_uid']):\n"
assert code.count(anchor) == 1
code = code.replace(anchor, '''            torch.save(dict(
                identities=requests.non_tensor_batch['traj_uid'].tolist(),
                padding_copy=requests.non_tensor_batch['verification_padding_copy'].tolist(),
                request_ids=requests.batch['input_ids'].cpu(),
                coefficients=requests.batch['dt_direct_reward'].cpu(),
                QVA={key:result.batch[key].cpu() for key in CREDIT_KEYS}),
                AUDIT/(stage+'-native-return.pt'))
            for i,identity in enumerate(requests.non_tensor_batch['traj_uid']):
                if requests.non_tensor_batch['verification_padding_copy'][i]:
                    continue
''')
anchor = '    output=compute_direct_target_credit(data,ObservedNativeGroup(),eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)'
assert code.count(anchor) == 1
code = code.replace(anchor, '''    protocol.pad_dataproto_to_divisor = mark_padding
    try:
        output=compute_direct_target_credit(data,ObservedNativeGroup(),eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    finally:
        protocol.pad_dataproto_to_divisor = native_pad
    torch.save(dict(QVA={key:output.batch[key].cpu() for key in CREDIT_KEYS},
        original_event_returns=returned), AUDIT/(stage+'-driver-return.pt'))''')
exec(compile(code, str(owner), 'exec'), globals())
