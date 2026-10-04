"""Read completed storage observations and record their exact measurement scope."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, SSH


raw = REPO/'experiments/rl/results_native_root_capture_inventory_20261004.json'
value = json.loads(raw.read_bytes())
assert value['completed'] and value['driver']['status']=='exited'
out = value['remote_root']
script = r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET
out=pathlib.Path(@OUT@)
tree=ET.parse(out/'cpu-tests.xml')
p=out/'verify_native_prefix_artifacts.py'
print(json.dumps(dict(cpu_tests=[n.attrib for n in tree.getroot().iter('testsuite')],
 cpu_log=(out/'cpu-tests.log').read_text(errors='replace')[-4000:],
 cpu_xml_sha256=hashlib.sha256((out/'cpu-tests.xml').read_bytes()).hexdigest(),
 actual_snapshot_callback_source=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
  function_source=p.read_text().split('def tensors(cache):',1)[1].split('@ray.remote',1)[0]),
 physical_after=subprocess.check_output(['mx-smi'],text=True))))
PY
'''.replace('@ENTRY@',ENTRY).replace('@OUT@',repr(out))
result=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT,timeout=45)
if result.returncode:
    print(result.stdout.decode('utf8','replace'))
    raise SystemExit(result.returncode)
remote=json.loads(result.stdout)
rows=[]
for rank in (0,1):
    inv=value['records'][f'root-capture-inventory-rank{rank}.json']['value']
    groups={}
    for row in inv['rows']:
        for group in row['storage_groups']:
            kind=('decoder' if any(f.startswith('decoder.') for f in group['fields'])
                  else row['block_type'])
            groups[kind]=groups.get(kind,0)+group['storage_bytes']
    rows.append(dict(rank=rank,unique_storage_bytes_by_capture=groups,
        total_per_layer_unique_storage_bytes=inv['sum_per_layer_unique_storage_bytes'],
        complete_layers=inv['complete_layers'],capture_metadata_errors=sum(
            len(row['diagnostic_errors']) for row in inv['rows']),
        actual_runner_gdn_cut=inv['original_runner_gdn_coefficient_start'],
        original_forward_calls_added=inv['original_forward_calls_added'],
        hooks_remaining=inv['hooks_remaining']))
prepared_sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (AUDIT/'native_root_capture_inventory_factory.py',
              AUDIT/'diagnose_native_prefix_leases.py',
              AUDIT/'run_native_prefix_reuse_workload.py')}
assessment=dict(raw_receipt=dict(path=str(raw),sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
    executed_diagnostic_commit='31f16be',formal_deployment=False,rows=rows,
    measurement_scope='Same recorded B4 rows40-43, full88-row bank, LoRA8/16, per-card4, max32768. Per-layer borrowed storage sums, not simultaneous physical peak or 32k capacity.',
    observed_diagnostic_defect=dict(
        affected='Old factory external-cache callback in executed31f16be',
        behavior='Original diagnostic tensors(cache) calls detach().cpu() for all cache layers; inventory invoked it at each root layer exit.',
        consequences='Instrumented root3.94s is not a root-reuse speed result. No external GPU storage aliases were identified from CPU snapshots. Captured native GPU tensor shape/dtype/storage metadata remain measured.',
        correction='Prepared source omits external callback/mapping entirely; no cache traversal substitute. 27 local structure/source tests pass; this corrected source was not rerun on GPU.',
        unchanged='Formal workers, DT equations, FA/FLA assertions, PPO, model configuration and raw executed receipts untouched.'),
    prepared_source_files=prepared_sources,remote=remote)
path=REPO/'experiments/rl/results_native_root_capture_inventory_assessment_20261004.json'
assert not path.exists()
path.write_text(json.dumps(assessment,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(local=str(path),rows=rows,cpu_tests=remote['cpu_tests']),indent=2))
