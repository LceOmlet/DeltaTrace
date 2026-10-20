"""Reuse the original real-payload v5 DT/u verification on all eight ranks."""
from prepare_campaign import ROOT, AUDIT

owner = ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source/run_native_behavior_v5.py'
code = owner.read_text().replace("AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'",
    "AUDIT=ROOT/'receipts/textcraft-v5-credit-integration-20261020-v1'")
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
exec(compile(code, str(owner), 'exec'), globals())
