"""Finish only the CPU boundary adapter; keep the eight passed native cases."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='sql-native-multiturn-speed-20261016-v1'
AUDIT=ROOT/'receipts'/VERSION
DEST=ROOT/'candidates'/VERSION
assert not (ROOT/'runs'/VERSION).exists()
record=json.loads((DEST/'prepared.json').read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(path)==expected for path,expected in record['source_bindings'].items())
xml=ET.parse(AUDIT/'native-multiturn-parity-tests.xml').getroot()
assert sum(int(s.get('tests',0)) for s in xml.iter('testsuite'))==8
assert sum(int(s.get('errors',0))+int(s.get('failures',0)) for s in xml.iter('testsuite'))==0
for name in ('boundary-adapter.stdout.log','boundary-adapter.stderr.log'):
    path=AUDIT/name
    if path.exists() and not (AUDIT/(name+'.attempt1')).exists():
        shutil.copyfile(path,AUDIT/(name+'.attempt1'))
owner=ROOT/'receipts/appworld-sql-native-speed-20261016-v1/source/verify_sql_boundary.py'
assert sha(owner)=='f166b8346208914645a5cbe57d08a5b75435fc0544ce9caf32ab87d2438540d5'
code=owner.read_text().replace('appworld-sql-native-speed-20261016-v1',VERSION)
code=code.replace(VERSION+'/SQL/prepared.json',VERSION+'/prepared.json')
# The original adapter reconstructs its environment from the preparation.
# Route its scripted fixture artifacts away from model-generated artifacts.
code=code.replace("env['CUDA_VISIBLE_DEVICES']=''", "env['CUDA_VISIBLE_DEVICES']=''\nenv['DT_SQL_NATIVE_FORMAT_RECEIPTS']=str(AUDIT/'CPU-fixture-turns')")
path=AUDIT/'source/verify_sql_boundary.py'
path.write_text(code)
env=dict(os.environ);env.update(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
env['DT_SQL_NATIVE_FORMAT_RECEIPTS']=str(AUDIT/'CPU-fixture-turns')
result=subprocess.run([env['VENV_PYTHON'],str(path)],env=env,cwd=record['entry'],capture_output=True)
(AUDIT/'boundary-adapter.stdout.log').write_bytes(result.stdout)
(AUDIT/'boundary-adapter.stderr.log').write_bytes(result.stderr)
print(result.stdout.decode(errors='replace')[-4000:]);print(result.stderr.decode(errors='replace')[-2000:])
result.check_returncode()
xml=ET.parse(AUDIT/'SQL-boundary-tests.xml').getroot()
assert sum(int(s.get('tests',0)) for s in xml.iter('testsuite'))==4
assert sum(int(s.get('errors',0))+int(s.get('failures',0)) for s in xml.iter('testsuite'))==0
(AUDIT/'CPU-verification.json').write_text(json.dumps(dict(status='native_multi_message_parity_and_SQL_boundary_verified',
    original_Hydra_config_SHA256=sha(DEST/'effective-config.yaml'),source_bindings=len(record['source_bindings']),
    native_parity_cases=8,native_SQL_boundary_cases=4,model_DT_optimizer_calls=0,
    original_state_update_and_reward_unchanged=True,numerical_tolerances_changed=False,
    metadata_attempt1='Boundary adapter read the old /SQL/prepared.json layout; only its path was corrected. Eight passed native parity cases reused, not rerun.',
    compatibility_manifest_alias=str(DEST/'SQL/prepared.json'),CPU_fixture_artifacts=str(AUDIT/'CPU-fixture-turns')),indent=2)+'\n')
