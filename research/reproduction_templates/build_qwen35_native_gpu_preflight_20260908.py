"""Freeze one actual Qwen3.5 load and at most four default-backend forwards."""
import ast
import base64
import hashlib
import json
import shlex
import zlib
from pathlib import Path
A = Path(__file__).resolve().parent
R = A.parent / 'DeltaTrace'
sha = lambda b: hashlib.sha256(b).hexdigest()
study = (A / 'qwen35_native_gpu_preflight_20260908.py').read_bytes(); ast.parse(study)
runtime = (R / 'research/runtime/official_fixed_text_inputs.py').read_bytes(); ast.parse(runtime)
schedule_runtime = (R / 'research/runtime/fla_maca_chunk_schedule.py').read_bytes(); ast.parse(schedule_runtime)
tokenizer = json.loads((A / 'snapshot${ARTIFACT_ROOT}/codex_qwen35_tokenizer_identity_20260908_v3/results.json').read_text())
weights = json.loads((A / 'qwen35_weight_identity_results_20260908.json').read_text())
study_mapping = ast.parse((A / 'qwen35_maca_mapping_probe_20260908.py').read_text())
# Load literal wheel provenance from the already reviewed import probe without
# executing its environment mutation or import test.
root = '${PRIVATE_MOUNT_PATH}'
p = {'study_sha256': sha(study), 'input_runtime_sha256': sha(runtime), 'schedule_runtime_sha256': sha(schedule_runtime),
    'checkpoint': root + '/models/Qwen3.5-9B',
    'verified_shard_sizes': {k: v['bytes'] for k, v in weights['shards'].items()},
    'weight_identity_receipt_sha256': sha((A / 'qwen35_weight_identity_results_20260908.json').read_bytes()),
    'checkpoint_config_tokenizer_sha256': {k: v['sha256'] for k, v in tokenizer['files'].items()},
    'native_model_sha256': 'cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba',
    'isolated_site': '${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/lib/python3.12/site-packages',
    'fla_mapped_utils_sha256': '2971d3805da9da36e1000e82c0abddb491d65eec27c393e1431921281f09b2a6',
    'fla_scheduled_chunk_sha256': 'e4a81e5f69991d6681fa1ad43c16f135da26032786f71428e305859677bcaaba',
    'isolated_environment': '${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env',
    'compiler_cache': '${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v2/triton_cache',
    'wheels': {root + '/wheels/tf513_full/transformers-5.13.0-py3-none-any.whl': '8adbc1d20bd5463cd6876b2eb7cb31971e1065788e7dc6bc12bab597a7c504b7',
               root + '/wheels/veomni/flash_linear_attention-0.4.1-py3-none-any.whl': 'd18bdfe9d1f4b424676444eac9d50fb8433b70e5d4e0e0878b20bcbcdbea57ce',
               root + '/wheels/veomni/fla_core-0.4.1-py3-none-any.whl': '93c6afe4c80fc7bc705fa8aeea6a46d2cf2d77383f9619a41863c7114c801bab'},
    'author_root': '${FLASHTRACE_ROOT}',
    'author_source_sha256': {'llm_attr.py': 'a141e5c346681dc4c29cd62ee07561e93882d4f56b7fd95788743ee8ef403979',
        'shared_utils.py': '0d3e68a6656728369ad2f99300859b38d584e7c7676352441a92187ead7cba67'},
    'selection': [['niah_mq_q2', 0], ['morehopqa', 1]],
    'cache_sha256': tokenizer['protocol']['official_cache_sha256'],
    'call_selections': [[0], [1], [0, 1]],
    'maximum_model_loads': 1, 'maximum_root_forward_attempts': 3,
    'family_model_load_ceiling_after_import_and_schedule_repair': 3,
    'family_forward_attempt_ceiling': 4,
    'prior_failed_attempt': {'directory': '${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v1',
        'raw_sha256': sha((A / 'snapshot${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v1/results.json').read_bytes()),
        'actual_model_loads': 1, 'actual_forwards': 0,
        'cause': 'Unrelated wordfreq top-level dependency of author module missing. Compile unchanged hash-verified text-preparation class AST before loading model; record second load, no forward budget reset.'},
    'prior_FLA_resource_failure': {'directory': '${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v2',
        'raw_sha256': '921ab98a0bb7b9c9f47a3026aedd11239c6f2284e6d5e1bb76d0fba92944ff84',
        'model_loads': 1, 'forward_attempts': 1, 'complete_forwards': 0,
        'shared_memory_required': 76288, 'hardware_limit': 65536,
        'change': 'One original FLA autotune stage list becomes[1]; kernel bodies/model/backward remain unchanged. Three remaining forwards, no budget reset.'},
    'relative_L2_limit': 0.01, 'max_abs_logprob_limit': 0.25,
    'prefix_relative_L2_limit': 0.02,
    'additional_CPU_reference_budget': 'One original Transformers recurrent reference call on first actual layer first64 tokens, CPU only; explicitly diagnostic, not a model fallback.',
    'precision_scope': 'BF16 native model default FA/FLA. Batch differences are reported in full, not required bitwise equal. Limits are engineering screens, not official precision guarantees.',
    'stop': 'Stop at first failure, preserve failed profile/source/inputs metadata. No Torch fallback or replacement attention. No additional forwards outside this four-call budget.',
    'workload': 'Actual complete author fixed-target inputs. Two B1 calls, one right-padded B2 call. First NI and B2 profiled including compilation/autotune. Standard compiler cache reused, no activation/result cache. No steady speed comparison.',
    'scope': 'Native backend preflight only; no attribution, quality scan, new generation, FT winner or gold remapping claim. Same author target_response and tokenizer PAD=EOS policy. Existing full weight identity is retained; current load checks size and before/after stat, not a new20GB full rehash.'}
protocol = json.dumps(p, indent=2).encode()
(A / 'qwen35_native_gpu_preflight_protocol_20260908.json').write_bytes(protocol)
files = {'study.py': study, 'protocol.json': protocol, 'official_fixed_text_inputs.py': runtime, 'fla_maca_chunk_schedule.py': schedule_runtime}
packed = base64.b64encode(zlib.compress(json.dumps({n: base64.b64encode(b).decode() for n, b in files.items()}).encode())).decode()
python = '${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader = ('import base64,pathlib,subprocess,json,zlib; d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_native_gpu_preflight_20260908_v3"); '
    'd.mkdir(exist_ok=False); files=json.loads(zlib.decompress(base64.b64decode(' + repr(packed) + '))); '
    '[(d/n).write_bytes(base64.b64decode(b)) for n,b in files.items()]; '
    'f=(d/"driver.log").open("w"); j=subprocess.Popen([' + repr(python) + ',str(d/"study.py")],'
    'stdout=f,stderr=subprocess.STDOUT,start_new_session=True); (d/"pid").write_text(str(j.pid)); '
    'print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A / 'launch_qwen35_native_gpu_preflight_20260908.json').write_text(json.dumps({'cmd': python + ' -c ' + shlex.quote(loader), 'timeout': 10}))
print(json.dumps({'study_sha256': sha(study), 'protocol_sha256': sha(protocol), 'maximum_forwards': 3, 'total_family_forward_attempts':4}))
