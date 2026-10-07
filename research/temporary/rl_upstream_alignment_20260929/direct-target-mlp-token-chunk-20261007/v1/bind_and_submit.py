"""Bind actual token-chunk comparison receipts through the existing submit owner.

This file is prepared for review only. When explicitly invoked it requires the
original diagnostic to have ended and both original rank comparisons to have
completed, then reuses the frozen binder's preservation/submit transport.
No new numerical tolerance or bitwise-equality requirement is introduced.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
OWNER = AUDIT / 'direct-target-gpu-lifetime-20261007/v1/bind_and_submit.py'
OWNER_SHA256 = '8642c1c5b7e00e097a088835c44fc8361603da7f03cee27ef2c5176e6c55ff1b'
DIAGNOSTIC_SHA256 = '6e57a053baea9eb4913a177532d2c56f890eee353389ac6a4ff3e0922faf937a'


def owner_module():
    assert hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA256
    spec = importlib.util.spec_from_file_location('frozen_lifetime_binding_owner', OWNER)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


def replace_once(code, before, after):
    assert code.count(before) == 1, before
    return code.replace(before, after, 1)


def composed_code():
    owner = owner_module()
    code = owner.CODE.replace('direct-target-gpu-lifetime-20261007-v1',
                              'direct-target-mlp-token-chunk-20261007-v1')
    code = replace_once(code, "job=read(O/'job.json')",
        "job=read(O/'job.json')\nassert job['pid']==1946042,'Unexpected diagnostic owner'")
    start = code.index('records=[]\n')
    end = code.index("source=read(B/'source-template.json')", start)
    code = code[:start] + r'''records=[]
input_path=O/'actual-direct-target-inputs.json'
input_sha=sha(input_path)
assert input_sha=='f98f93a77c18d2fb8a62dace02ec0dbf61136764ccb4e061c1f576d7dc75048c'
for rank in range(2):
 path=O/f'rank{rank}.json';x=read(path)
 assert x['rank']==rank and x['phase']=='mlp_token_chunk_real_input_complete'
 assert x['all_comparisons_passed'] is True and x['events']
 assert x['observed_multi_chunk_calls']>0 and x['observed_nondivisible_calls']>0
 assert x['operations']['original_B4_attribute_calls_per_rank']==1
 assert x['report']['finite_trace_calls']==1 and x['report']['minibatch_size']==4
 actual_input=next(item for item in x['inputs']['ranks'] if item['rank']==rank)
 assert actual_input['rows']==4 and x['inputs']['input']['sha256']==input_sha
 assert all(e['comparison_passed'] is True for e in x['events'])
 assert all(e['shape']==e['reference_shape'] and e['dtype']==e['reference_dtype']
            for e in x['events'])
 assert x['owners']['chunk_adapter']['sha256']=='1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234'
 records.append(dict(**binding(path),rank=rank,
  actual_mlp_replay_shapes=sorted({tuple(e['input_shapes'][0]) for e in x['events']}),
  actual_context_lengths=x['report']['actual_context_lengths'],
  causal_context_lengths=x['report'].get('causal_context_lengths'),
  input_preparation=actual_input,compared_calls=len(x['events']),
  all_comparisons_passed=x['all_comparisons_passed'],
  all_bitwise_equal=x['all_bitwise_equal'],
  observed_multi_chunk_calls=x['observed_multi_chunk_calls'],
  observed_nondivisible_calls=x['observed_nondivisible_calls'],
  assertion=x['assertion'],events=x['events'],
  first_comparison_including_compilation=x['events'][0],
  seconds=x['seconds'],runtime_override=x['runtime_override'],
  resources=x['resources'],memory_scope=x['memory_scope'],cost_scope=x['cost_scope']))
''' + code[end:]
    code = replace_once(code,
        "assert source['dt_source_sha256']['clean/qwen35/qwen35_dense_finite_runner.py']=='ba639b2876827cc250f4806af278065181ef78d28c8b22da97377c34b9e168d7'",
        "assert source['dt_source_sha256']['clean/qwen35/qwen35_dense_finite_runner.py']=='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'\n"
        "assert source['dt_source_sha256']['clean/qwen35/qwen35_decoder_finite.py']=='1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234'\n"
        "assert source['dt_source_sha256']['clean/qwen35/qwen35_answer_finite.py']=='1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e'\n"
        "assert source['entry_sha256']['reward_readout.py']=='7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27'\n"
        "assert source['actual_CPU_imports']['verl.workers.actor.dp_actor']['sha256']=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'\n"
        "assert source['resume_mode']=='disable' and source['checkpoint_restore_requested'] is False")
    code = replace_once(code,
        "assert sha(O/'diagnose_direct_target.py')=='2829969c633630605ea4aae78f64a7ee29efce0d4a6a4b063c63b197f044ddfe'",
        "assert sha(O/'diagnose_direct_target.py')=='" + DIAGNOSTIC_SHA256 + "'")
    code = replace_once(code,
        "verification=dict(observed_unix=time.time(),status='actual_same_capture_GDN_lifetime_regression_passed',",
        "verification=dict(observed_unix=time.time(),status='actual_same_capture_MLP_token_chunk_dtype_assertions_passed',")
    code = replace_once(code,
        "scope='One real B4 per rank through original worker/producer/readout. 24 same-capture GDN outputs per rank bitwise equal; original torch.testing.assert_close defaults also passed. Existing FA/FLA kernels/assertions and PPO unchanged. Actual replay widths below32k; not full episode joint-target32k capacity, formal peak, speed comparison or training-health proof.')",
        "scope='One saved real B4 per rank through original worker/producer/readout; nonempty same-capture MLP comparisons used unmodified torch.testing.assert_close dtype defaults. Multi-chunk and nondivisible real widths were observed. Bitwise equality and max-absolute differences are recorded, not acceptance gates. Original and candidate times include compilation and comparison; candidate retains the reference output and diagnostic prefix reuse is disabled. Not official whole-chain FA/FLA/PPO certification, full joint-target32k capacity, standalone physical peak, formal throughput or training-health proof.')")
    code = code.replace('before-lifetime-verification.json', 'before-mlp-token-chunk-verification.json')
    code = code.replace("['gpu_lifetime_verification']", "['mlp_token_chunk_verification']")
    code = replace_once(code,
        "source['mlp_token_chunk_verification']=dict(**binding(v),scope=verification['scope'])",
        "source['mlp_token_chunk_verification']=dict(**binding(v),scope=verification['scope'])\n"
        "source['runtime_verification_status']='Actual same-capture MLP token-chunk dtype assertions passed; formal health and full joint-target32k capacity pending'\n"
        "source['formal_training_health_verified']=False\n"
        "source['joint_target32k_capacity_verified']=False")
    ast.parse(code.replace('@ROOT@', repr(owner.stage.ROOT)))
    return code


def transport_owner_namespace():
    """Reuse the frozen binder's exact output preservation and submit command."""
    owner = owner_module()
    text = OWNER.read_text(encoding='utf-8')
    tree = ast.parse(text)
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Compare)
                 and isinstance(node.test.left, ast.Name)
                 and node.test.left.id == '__name__')
    body = ast.Module(body=guard.body, type_ignores=[])
    for node in ast.walk(body):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('direct-target-gpu-lifetime-20261007-v1',
                                            'direct-target-mlp-token-chunk-20261007-v1')
    ast.fix_missing_locations(body)
    namespace = dict(vars(owner), HERE=HERE, CODE=composed_code(), __file__=__file__)
    return body, namespace


if __name__ == '__main__':
    transport, namespace = transport_owner_namespace()
    exec(compile(transport, str(OWNER), 'exec'), namespace)
