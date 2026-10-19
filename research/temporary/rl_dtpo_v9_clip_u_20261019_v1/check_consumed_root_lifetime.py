"""Compose the accepted native B4/32768 resource comparison for the v9 owner.

CPU --preflight is safe during training. A real comparison requires the
recorded checkpoint resource window and an empty physical GPU0. No optimizer
or rollout generation is invoked; both DT functions come from exact owners.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dt-native-consumed-root-lifetime-20261019-v1'
owner = ROOT / 'receipts/dt-native-replay-host-lifetime-20261017-v3/source/check_DT_replay_host_lifetime_v3.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '8794f5234857e56ff781796d03dcee2872c4d8c31641df2231b0c6d168873e09'
sensor_owner = AUDIT / 'source/original_physical_readout_owner.py'
assert hashlib.sha256(sensor_owner.read_bytes()).hexdigest() == 'b255042e12cb4bed0232d2da846f56bcd94dc5dc81a756cc03538536617d81e0'
sensor, = [node for node in ast.parse(sensor_owner.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == 'physical_host_snapshot']
exec(compile(ast.Module(body=[sensor], type_ignores=[]), str(sensor_owner), 'exec'), globals())

code = owner.read_text()
replacements = [
    ('on GPU6.', 'on GPU0.'),
    ('dt-native-replay-host-lifetime-20261017-v3', 'dt-native-consumed-root-lifetime-20261019-v1'),
    ("HARNESS = ROOT / 'receipts/textcraft-reward-target-exponential-20261016-v5/source/run_native_behavior_v5.py'",
     "HARNESS = ROOT / 'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1/source/run_native_behavior_v9.py'"),
    ("HARNESS_SHA = 'ae4ba91b1978d572c17a5ab4589c0d2804be1eefb8d59d1de308f17d22815541'",
     "HARNESS_SHA = '9b36c6bf627c9a9ad1c84414117d22e928ae2f6db02972588fbd66c7db360abd'"),
    ("ROOT / 'runs/appworld-dtpo-reward-target-exponential-variance-formal-20261017-v6/source.json'",
     "ROOT / 'runs/textcraft-dtpo-v9-clip-u-8gpu-200-20261019-v2/source.json'"),
    ("ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6/AppWorld_DTPO-effective-config.yaml'",
     "ROOT / 'receipts/textcraft-eight-gpu-v9-clip-u-20261019-v2/dt-effective-config.yaml'"),
    ("source['budget']['iterations']", "source['budget']['total_iterations']"),
    ("env['CUDA_VISIBLE_DEVICES'] = '6'", "env['CUDA_VISIBLE_DEVICES'] = '0'"),
    ("physical_gpus=[6]", "physical_gpus=[0]"),
    ("_temp_dir='/tmp/ray-DT-host-lifetime-v3'", "_temp_dir='/tmp/ray-DT-consumed-root-lifetime-v9'"),
    ("('candidate-warm-32768', '1')", "('candidate-warm-32768', '0')"),
    ("'HF_HUB_OFFLINE', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')}",
     "'HF_HUB_OFFLINE', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'PYTORCH_CUDA_ALLOC_CONF')}"),
    ("Single native B4 worker; two-rank/four-resident capacity remains unproven",
     "Single native B4 worker; eight-rank formal training capacity is not established by this comparison"),
]
for old, new in replacements:
    assert old in code, old
    code = code.replace(old, new)

old = '''            before_physical = {
                'collected': time.time(),
                'MemAvailable_KiB': int(next(x.split()[1] for x in
                    Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))),
                'XTT_single_pool_KiB': int(Path(
                    '/sys/class/drm/card1/device/mem_info_xtt_used').read_text()),
            }
'''
assert code.count(old) == 1
code = code.replace(old, '            before_physical = physical_host_snapshot()\n')

anchor = "    env['DT_ROOT'] = candidate\n"
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + "    env['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(AUDIT/'worker-runtime-overrides')\n")

anchor = "        native_trace = reward_readout.trace_token_attribution\n"
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + '''        import ast,importlib,types,sys
        assert os.environ.get('DT_CREDIT_MODE')=='reward_target_exponential_clipped_v9'
        assert 'pinned_use_cuda_host_register:False' in os.environ['PYTORCH_CUDA_ALLOC_CONF']
        assert os.environ['DT_NATIVE_HOST_CACHE_REPLAY_LAYER']=='0'
        # The original producer establishes these import paths lazily. This
        # resource comparison selects the class before that producer exists.
        native_root=Path(os.environ['DT_ROOT']).resolve()
        native_environment=json.loads(Path(os.environ.get('DT_ENVIRONMENT_JSON',native_root/'environment.json')).read_bytes())['qwen35']
        official=Path(os.environ.get('DT_OFFICIAL_ROOT') or native_environment['official_root'])
        sys.path[:0]=[str(native_root),str(official),str(native_root/'clean/qwen35')]
        sys.path.append(native_environment['ft_extension_root'])
        os.environ.update(native_environment.get('runtime_environment',{}),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        native_module = importlib.import_module('qwen35_dense_finite_runner')
        original_path = ROOT / 'candidates/dt-native-replay-host-lifetime-20261017-v3/deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'
        candidate_path = ROOT / 'candidates/dt-native-consumed-root-lifetime-20261019-v1/deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
        assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == '9f13c898c9faec72f57d77a81966315f25950c251e4381598b8f5442e287c5cf'
        original_class, = [n for n in ast.parse(original_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Qwen35DenseFiniteRunner']
        original_function, = [n for n in original_class.body if isinstance(n,ast.FunctionDef) and n.name=='attribute']
        namespace = {}
        exec(compile(ast.Module(body=[original_function],type_ignores=[]),str(original_path),'exec'),native_module.__dict__,namespace)
        candidate_function = native_module.Qwen35DenseFiniteRunner.attribute
        selected_function = candidate_function if stage.startswith('candidate') else namespace['attribute']
        selected_path = candidate_path if stage.startswith('candidate') else original_path
        native_module.Qwen35DenseFiniteRunner.attribute = selected_function
        if hasattr(self,'_deltatrace_producer'):
            runner = self._deltatrace_producer.runner
            runner.attribute = types.MethodType(selected_function,runner)
''')

anchor = "                native_RPC_and_verification_seconds=time.monotonic()-started,\n"
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + "                selected_attribute_source=str(selected_path), selected_attribute_source_SHA256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),\n")
anchor = "            reward_readout.trace_token_attribution = native_trace\n"
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + '''            native_module.Qwen35DenseFiniteRunner.attribute = candidate_function
            if hasattr(self,'_deltatrace_producer'):
                runner=self._deltatrace_producer.runner
                runner.attribute=types.MethodType(candidate_function,runner)
''')

anchor = "    lock = open(AUDIT / 'resource-comparison-single-execution.lock', 'a')\n"
assert code.count(anchor) == 1
code = code.replace(anchor, '''    # No test may steal a training card or start before the native checkpoint window.
    window=json.loads((AUDIT/'native-checkpoint-resource-window.json').read_bytes())
    assert window['status']=='native_checkpoint_verified_formal_session_released_for_bounded_resource_check'
    assert window['checkpoint_step']>=5 and not window['remaining_non_zombie']
    physical=subprocess.check_output(['mx-smi'],text=True)
    import re
    assert not re.search(r'\\|\\s*0\\s+\\d+\\s+',physical.split('| Process:')[-1]), 'Physical GPU0 is occupied'
''' + anchor)

# Record the actual composed source; preparation itself is not native acceptance.
EXECUTED_SOURCE_SHA256 = hashlib.sha256(code.encode()).hexdigest()
ast.parse(code)
exec(compile(code, str(owner), 'exec'), globals())
