"""Read the exact bounded operand-test handle, result and physical resources."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_transport',HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
launch = json.loads((HERE/'block-dtype-launch.json').read_bytes())
script = fr'''source {transport.ENTRY}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PYCODE'
import json,time,psutil,subprocess
from pathlib import Path
out=Path('{transport.ROOT}/receipts/current-extreme-official-block-dtype-20261008-v1')
pid={launch['pid']};birth={launch['birth']}
same=psutil.pid_exists(pid) and psutil.Process(pid).create_time()==birth
phase=json.loads((out/'phase.json').read_bytes()) if (out/'phase.json').exists() else None
result=json.loads((out/'official-result.json').read_bytes()) if (out/'official-result.json').exists() else None
hold=Path('{transport.ROOT}/receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt')
record=dict(unix=time.time(),pid=pid,birth=birth,same_birth=same,
 phase=phase,result=result,physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout,
 current_PSS_bytes=psutil.Process(pid).memory_full_info().pss if same else None,
 textcraft_same_birth=psutil.pid_exists(2833207) and psutil.Process(2833207).create_time()==1791370325.16,
 textcraft_release_present=[(hold/('rank'+str(i)+'-release-update')).exists() for i in (0,1)],
 log_tail=(out/'driver.log').read_text(errors='replace')[-3500:])
print(json.dumps(record))
PYCODE
'''
run = subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,check=True)
record = json.loads(run.stdout)
(HERE/('block-dtype-observation-'+str(int(record['unix']))+'.json')).write_text(
    json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(unix=record['unix'],same_birth=record['same_birth'],
    phase=(record['phase'] or {}).get('phase'),current_PSS_bytes=record['current_PSS_bytes'],
    official_result_status=(record['result'] or {}).get('status'),
    cases=[dict(dtype=c['dtype'],status=c['status'],checks=c['checks']) for c in (record['result'] or {}).get('cases',[])],
    log_tail=record['log_tail'],textcraft_same_birth=record['textcraft_same_birth'],
    textcraft_release_present=record['textcraft_release_present']),ensure_ascii=False))
