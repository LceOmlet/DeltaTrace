"""Index measured root-reuse evidence and keep requirement sources explicit."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import REPO, ENTRY, SSH


def receipt(path):
    path=REPO/path
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


paths=('experiments/rl/results_native_root_tape_20261004.json',
       'experiments/rl/results_native_root_tape_capacity_20261004.json')
measurements=[]
for filename in paths:
    raw=json.loads((REPO/filename).read_bytes())
    assert raw['completed'] and raw['driver']['status']=='exited'
    is_capacity='capacity' in filename
    baseline='capacity_disabled_warm' if is_capacity else 'shared_warm'
    enabled='capacity_root_warm' if is_capacity else 'root_tape_warm'
    rows=[]
    for rank in (0,1):
        result=raw['records'][f'rank{rank}.json']['value']
        before,after=[result['reports'][label] for label in (baseline,enabled)]
        rows.append(dict(rank=rank,baseline=baseline,enabled=enabled,
            baseline_wall_seconds=before['total_wall_seconds'],enabled_wall_seconds=after['total_wall_seconds'],
            reduction_fraction=1-after['total_wall_seconds']/before['total_wall_seconds'],
            baseline_peak_torch_allocated_bytes=before['peak_torch_allocated_bytes'],
            enabled_peak_torch_allocated_bytes=after['peak_torch_allocated_bytes'],
            source_scope=('Original synthetic capacity helper, response512, query313, total32768, distinct IDs; no task/actor-update claim'
                          if is_capacity else 'Same real sorted rows40-43, original immutable 88-request bank, actual factual lengths7410-7586; no whole-88 speed claim'),
            raw_value_observations=result['raw_value_observations'],
            original_capture_reuse=after['root_tape_observations']))
    measurements.append(dict(receipt=receipt(filename),remote_root=raw['remote_root'],
        diagnostic_commit=raw['records']['prepared.json']['value']['diagnostic_commit'],
        owner=raw['records']['prepared.json']['value']['native_root_tape_candidate'],rows=rows))

roots=[measurement['remote_root'] for measurement in measurements]
script=r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,json,pathlib,xml.etree.ElementTree as ET
result=[]
for name in @ROOTS@:
 root=pathlib.Path(name);path=root/'cpu-tests.xml'
 result.append(dict(remote_root=name,cpu_test_xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  suites=[node.attrib for node in ET.parse(path).getroot().iter('testsuite')]))
print(json.dumps(result))
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOTS@',repr(roots))
read=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,timeout=45)
if read.returncode:
    print(read.stdout.decode('utf8','replace'));raise SystemExit(read.returncode)

official=REPO/'research/temporary/rl_upstream_alignment_20260929/recipe-sources/verl-agent-20bd331'
def official_source(relative):
    path=official/relative
    return dict(commit='20bd331bdbc9026a5668e11362178e10ab7400c8',
                path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
value=dict(formal_deployment=False,numerical_acceptance=False,
    requirements_origin={
        'token_QVA':'User accepted PLAN; no value model, span credit, observation routing or alternate PPO',
        'fixed_resources':'User rank8/alpha16/per-card microbatch4; current two-GPU diagnostic owner retained',
        'real_behavior_and_tolerance':'User latest instruction: fixtures cannot replace actual official behavior/numerical checks',
        'head':dict(source=official_source('tests/kernels/test_linear_cross_entropy.py'),
                    assertion_scope='Original output head forward1e-4/1e-4 and backward1e-2/1e-4; not generic whole-PPO tolerance',
                    last_accepted_receipt=receipt('experiments/rl/results_actor_b8.json')),
        'padding':dict(source=official_source('tests/models/test_transformer.py'),
                       assertion_scope='Original masked-mean logprob atol1e-2/rtol1e-5; not per-token allclose'),
        'gradient':dict(source=official_source('tests/models/test_transformers_ulysses.py'),
                        assertion_scope='Original gradient atol1e-2/rtol1e-5; original sequence-parallel scope remains recorded'),
        'DT':'Original FA/FLA references and assertion statements on actual changed operands; no whole-DT threshold or correction',
    },measurements=measurements,actual_cpu_tests=json.loads(read.stdout),
    not_proven=['Whole formal iteration speed','Complete candidate numerical acceptance',
                'Candidate alongside sleeping/waking vLLM and actor update',
                'Environment/task behavior from the synthetic capacity fixture'],
    allocator_measurement_scope='Torch peak allocation is recorded. cuda.mem_get_info values labeled physical_free in older raw reports are not accepted as mx-smi physical peak measurements; raw receipts stay unchanged.')
path=REPO/'experiments/rl/results_native_root_tape_assessment_20261004.json'
with path.open('x',encoding='utf8') as stream:
    json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(local=str(path),cpu_tests=value['actual_cpu_tests'],
    reductions=[[row['reduction_fraction'] for row in m['rows']] for m in measurements]),indent=2))
