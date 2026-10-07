set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

from pathlib import Path
import json,hashlib,time,psutil
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');C=R/'candidates/direct-target-mlp-token-chunk-20261007-v1';B=C/'appworld'
O=R/'receipts/direct-target-mlp-token-chunk-20261007-v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
binding=lambda p:dict(path=str(p),sha256=sha(p))
job=read(O/'job.json')
assert job['pid']==1946042,'Unexpected diagnostic owner'
try:
 p=psutil.Process(job['pid'])
 assert p.create_time()!=job['birth'] or p.status()==psutil.STATUS_ZOMBIE,'Original diagnostic still active'
except psutil.NoSuchProcess:pass
records=[]
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
source=read(B/'source-template.json');prepared=read(B/'prepared.json')
assert source['dt_source_sha256']['clean/qwen35/qwen35_dense_finite_runner.py']=='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'
assert source['dt_source_sha256']['clean/qwen35/qwen35_decoder_finite.py']=='1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234'
assert source['dt_source_sha256']['clean/qwen35/qwen35_answer_finite.py']=='1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e'
assert source['entry_sha256']['reward_readout.py']=='7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27'
assert source['actual_CPU_imports']['verl.workers.actor.dp_actor']['sha256']=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert source['resume_mode']=='disable' and source['checkpoint_restore_requested'] is False
assert source['dt_source_sha256']['clean/qwen35/qwen35_gdn_finite.py']=='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'
assert sha(O/'diagnose_direct_target.py')=='6e57a053baea9eb4913a177532d2c56f890eee353389ac6a4ff3e0922faf937a'
v=O/'verification.json';assert not v.exists(),'Preserve previous binding attempts'
verification=dict(observed_unix=time.time(),status='actual_same_capture_MLP_token_chunk_dtype_assertions_passed',
 records=records,diagnostic=binding(O/'diagnose_direct_target.py'),
 input=binding(O/'actual-direct-target-inputs.json'),job=binding(O/'job.json'),
 scope='One saved real B4 per rank through original worker/producer/readout; nonempty same-capture MLP comparisons used unmodified torch.testing.assert_close dtype defaults. Multi-chunk and nondivisible real widths were observed. Bitwise equality and max-absolute differences are recorded, not acceptance gates. Original and candidate times include compilation and comparison; candidate retains the reference output and diagnostic prefix reuse is disabled. Not official whole-chain FA/FLA/PPO certification, full joint-target32k capacity, standalone physical peak, formal throughput or training-health proof.')
v.write_text(json.dumps(verification,indent=2)+'\n')
assert not (B/'prepared-before-mlp-token-chunk-verification.json').exists()
(B/'source-before-mlp-token-chunk-verification.json').write_bytes((B/'source-template.json').read_bytes())
(B/'prepared-before-mlp-token-chunk-verification.json').write_bytes((B/'prepared.json').read_bytes())
source['mlp_token_chunk_verification']=dict(**binding(v),scope=verification['scope'])
source['runtime_verification_status']='Actual same-capture MLP token-chunk dtype assertions passed; formal health and full joint-target32k capacity pending'
source['formal_training_health_verified']=False
source['joint_target32k_capacity_verified']=False
source['source_bindings'][str(v)]=sha(v)
for item in records:source['source_bindings'][item['path']]=item['sha256']
source['source_bindings'][str(O/'diagnose_direct_target.py')]=sha(O/'diagnose_direct_target.py')
(B/'source-template.json').write_text(json.dumps(source,indent=2)+'\n')
prepared['source_template']['sha256']=sha(B/'source-template.json')
prepared['source_bindings']=source['source_bindings']
prepared['mlp_token_chunk_verification']=source['mlp_token_chunk_verification']
(B/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
print(json.dumps(dict(verification=binding(v),prepared=binding(B/'prepared.json'),source=binding(B/'source-template.json'),scope=verification['scope'])))

PY
"$VENV_PYTHON" /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1/setup/submit_prepared_direct_targets.py --runtime-root /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922 --candidate-root /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1 --tasks AppWorld --execute --receipt-dir /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-mlp-token-chunk-20261007-v1-submit
