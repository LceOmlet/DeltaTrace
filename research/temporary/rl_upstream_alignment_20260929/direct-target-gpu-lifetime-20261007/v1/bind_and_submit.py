"""Bind actual lifetime regression receipts, then call the existing submit owner."""
from pathlib import Path
import ast
import importlib.util
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('existing_transport', AUDIT / 'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

CODE = r'''
from pathlib import Path
import json,hashlib,time,psutil
R=Path(@ROOT@);C=R/'candidates/direct-target-gpu-lifetime-20261007-v1';B=C/'appworld'
O=R/'receipts/direct-target-gpu-lifetime-20261007-v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
binding=lambda p:dict(path=str(p),sha256=sha(p))
job=read(O/'job.json')
try:
 p=psutil.Process(job['pid'])
 assert p.create_time()!=job['birth'] or p.status()==psutil.STATUS_ZOMBIE,'Original diagnostic still active'
except psutil.NoSuchProcess:pass
records=[]
for rank in range(2):
 path=O/f'rank{rank}.json';x=read(path)
 assert x['phase']=='gpu_lifetime_real_input_complete'
 assert x['all_bitwise_equal'] and len(x['events'])==24
 assert all(e['bitwise_equal'] and e['max_abs_difference']==0 for e in x['events'])
 assert all(e['remaining_endpoints']==[] for e in x['events'])
 records.append(dict(**binding(path),rank=rank,
  actual_gdn_replay_shapes=sorted({tuple(e['shape']) for e in x['events']}),
  actual_context_lengths=x['report']['actual_context_lengths'],
  causal_context_lengths=x['report']['causal_context_lengths'],
  compared_calls=len(x['events']),all_bitwise_equal=True,seconds=x['seconds']))
source=read(B/'source-template.json');prepared=read(B/'prepared.json')
assert source['dt_source_sha256']['clean/qwen35/qwen35_dense_finite_runner.py']=='ba639b2876827cc250f4806af278065181ef78d28c8b22da97377c34b9e168d7'
assert source['dt_source_sha256']['clean/qwen35/qwen35_gdn_finite.py']=='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'
assert sha(O/'diagnose_direct_target.py')=='2829969c633630605ea4aae78f64a7ee29efce0d4a6a4b063c63b197f044ddfe'
v=O/'verification.json';assert not v.exists(),'Preserve previous binding attempts'
verification=dict(observed_unix=time.time(),status='actual_same_capture_GDN_lifetime_regression_passed',
 records=records,diagnostic=binding(O/'diagnose_direct_target.py'),
 input=binding(O/'actual-direct-target-inputs.json'),job=binding(O/'job.json'),
 scope='One real B4 per rank through original worker/producer/readout. 24 same-capture GDN outputs per rank bitwise equal; original torch.testing.assert_close defaults also passed. Existing FA/FLA kernels/assertions and PPO unchanged. Actual replay widths below32k; not full episode joint-target32k capacity, formal peak, speed comparison or training-health proof.')
v.write_text(json.dumps(verification,indent=2)+'\n')
assert not (B/'prepared-before-lifetime-verification.json').exists()
(B/'source-before-lifetime-verification.json').write_bytes((B/'source-template.json').read_bytes())
(B/'prepared-before-lifetime-verification.json').write_bytes((B/'prepared.json').read_bytes())
source['gpu_lifetime_verification']=dict(**binding(v),scope=verification['scope'])
source['source_bindings'][str(v)]=sha(v)
for item in records:source['source_bindings'][item['path']]=item['sha256']
source['source_bindings'][str(O/'diagnose_direct_target.py')]=sha(O/'diagnose_direct_target.py')
(B/'source-template.json').write_text(json.dumps(source,indent=2)+'\n')
prepared['source_template']['sha256']=sha(B/'source-template.json')
prepared['source_bindings']=source['source_bindings']
prepared['gpu_lifetime_verification']=source['gpu_lifetime_verification']
(B/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
print(json.dumps(dict(verification=binding(v),prepared=binding(B/'prepared.json'),source=binding(B/'source-template.json'),scope=verification['scope'])))
'''

if __name__ == '__main__':
    code = CODE.replace('@ROOT@', repr(stage.ROOT))
    ast.parse(code)
    remote = stage.ROOT + '/candidates/direct-target-gpu-lifetime-20261007-v1'
    script = ('set -eu\nsource ' + stage.ENTRY + '/metax-entry.env.sh\n'
              '"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
              '"$VENV_PYTHON" ' + remote + '/setup/submit_prepared_direct_targets.py '
              '--runtime-root ' + stage.ROOT + ' --candidate-root ' + remote
              + ' --tasks AppWorld --execute --receipt-dir ' + stage.ROOT
              + '/receipts/direct-target-gpu-lifetime-20261007-v1-submit\n')
    (HERE / 'bind-and-submit-command.sh').write_text(script)
    result = subprocess.run(stage.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=120)
    (HERE / 'formal-submit.stdout.jsonl').write_bytes(result.stdout)
    (HERE / 'formal-submit.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
