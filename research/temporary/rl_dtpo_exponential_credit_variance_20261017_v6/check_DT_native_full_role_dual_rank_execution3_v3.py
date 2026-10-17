"""Use the original DT verification worker with both full actor/rollout ranks.

The same-input resource comparison has completed. This execution measures the
tested resource path with the original full worker role and B4/32768 per rank;
it adds no training, generation, optimizer or checkpoint implementation.
"""
import ast
import hashlib
from pathlib import Path
import time

directory = Path(__file__).resolve().parent
legacy = directory / 'check_DT_replay_host_lifetime_v3.py'
sensor = directory.parent / 'execution2/source/check_DT_replay_host_lifetime_execution2_v3.py'
assert hashlib.sha256(legacy.read_bytes()).hexdigest() == '8794f5234857e56ff781796d03dcee2872c4d8c31641df2231b0c6d168873e09'
assert hashlib.sha256(sensor.read_bytes()).hexdigest() == 'b255042e12cb4bed0232d2da846f56bcd94dc5dc81a756cc03538536617d81e0'
helper, = [n for n in ast.parse(sensor.read_bytes()).body
           if isinstance(n, ast.FunctionDef) and n.name == 'physical_host_snapshot']
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(sensor), 'exec'), globals())
code = legacy.read_text()

physical = """            before_physical = {
                'collected': time.time(),
                'MemAvailable_KiB': int(next(x.split()[1] for x in
                    Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))),
                'XTT_single_pool_KiB': int(Path(
                    '/sys/class/drm/card1/device/mem_info_xtt_used').read_text()),
            }
"""
stages = """            for stage, flag in [('baseline-warmup-32768', '0'),
                                ('baseline-warm-32768', '0'),
                                ('candidate-warm-32768', '1')]:
"""
finished = """        baseline = torch.load(AUDIT / 'baseline-warm-32768-vectors-rank0.pt',
                              map_location='cpu', weights_only=False)
        candidate = torch.load(AUDIT / 'candidate-warm-32768-vectors-rank0.pt',
                               map_location='cpu', weights_only=False)
        assert len(baseline) == len(candidate)
        state.update(status='completed', finished=time.time(),
                     native_DT_vectors_equal=[torch.equal(a, b) for a, b in zip(baseline, candidate)],
                     scope='Single native B4 worker; two-rank/four-resident capacity remains unproven')
"""
finish_dual = """        reports = [json.loads((AUDIT / f'candidate-full-role-dual-32768-native-resource-rank{rank}.json').read_bytes())
                   for rank in (0, 1)]
        assert len({r['pid'] for r in reports}) == 2
        assert all(r['native_worker_role'] == 'actor_rollout' and r['native_rollout_created'] for r in reports)
        state.update(status='completed_full_role_dual_rank_B4_exact32768_capacity', finished=time.time(),
                     full_two_rank_capacity_verified=True, full_two_rank_actor_rollout_capacity_verified=True,
                     four_resident_safety_verified=False,
                     scope='Both native actor/rollout ranks, B4/32768 each, while other three native jobs remain resident; AppWorld environment/controller footprint and unrestricted concurrent heavy phases are not covered')
"""
replacements = [
    (physical, '            before_physical = physical_host_snapshot()\n'),
    ("AUDIT = ROOT / 'receipts/dt-native-replay-host-lifetime-20261017-v3'",
     "AUDIT = ROOT / 'receipts/dt-native-replay-host-lifetime-20261017-v3/execution3-full-role-dual-rank'"),
    ("env['CUDA_VISIBLE_DEVICES'] = '6'", "env['CUDA_VISIBLE_DEVICES'] = '6,7'"),
    ('harness.base.requests(tokenizer, capacity=True).chunk(2)[0]', 'harness.base.requests(tokenizer, capacity=True)'),
    ('assert len(data) == 4', 'assert len(data) == 8'),
    ("data.batch['attention_mask'].sum(-1).tolist() == [32768] * 4", "data.batch['attention_mask'].sum(-1).tolist() == [32768] * 8"),
    ('physical_gpus=[6], world_size=1', 'physical_gpus=[6, 7], world_size=2'),
    ('num_cpus=8, num_gpus=1', 'num_cpus=16, num_gpus=2'),
    ("_temp_dir='/tmp/ray-DT-host-lifetime-v3'", "_temp_dir='/tmp/ray-DT-host-lifetime-v3-execution3-full-role-dual-rank'"),
    ('RayResourcePool([1], use_gpu=True', 'RayResourcePool([2], use_gpu=True'),
    ("RayClassWithInitArgs(ResourceWorker, config.actor_rollout_ref, 'actor')", "RayClassWithInitArgs(ResourceWorker, config.actor_rollout_ref, 'actor_rollout')"),
    (stages, "            for stage, flag in [('candidate-full-role-dual-32768', '1')]:\n"),
    ('PSS_bytes=psutil.Process().memory_full_info().pss,',
     "native_worker_role=self.role, native_rollout_created=hasattr(self, 'rollout'),\n                PSS_bytes=psutil.Process().memory_full_info().pss,"),
    (finished, finish_dual),
    ('original_actor_verification_method_reused=True',
     "original_actor_verification_method_reused=True, intended_worker_role='actor_rollout', intended_world_size=2, physical_sample=physical_host_snapshot()"),
]
for old, new in replacements:
    assert code.count(old) == 1, old
    code = code.replace(old, new, 1)
EXECUTED_SOURCE_SHA256 = hashlib.sha256(code.encode()).hexdigest()
exec(compile(code, str(legacy), 'exec'), globals())
