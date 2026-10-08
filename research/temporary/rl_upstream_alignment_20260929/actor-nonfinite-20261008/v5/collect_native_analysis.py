"""Run the read-only CPU tensor inspection against this one diagnostic."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
worker=HERE/'analyze_native_first_iteration.py'
code=worker.read_text()
out=transport.ROOT+'/receipts/textcraft-native-first-iteration-nonfinite-20261008-v5'
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 DT_ACTOR_INCIDENT_OUT='+out+' "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=120)
(HERE/'CPU-analysis.stderr.txt').write_bytes(result.stderr)
if result.returncode:print(result.stderr.decode(errors='replace'))
result.check_returncode()
value=json.loads(result.stdout)
(HERE/'native-input-gradient-analysis.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(CUDA_initialized=value['CUDA_initialized'],
    ranks=[dict(rank=r['rank'],status=r.get('status'),steps=[dict(index=s['index'],
        nonfinite_gradients=len(s['nonfinite_raw_gradients']),
        policy_KL=[m['native_policy_outputs'][2] for m in s['microbatches']],
        nonfinite_output_gradients=[dict(index=m['index'],category=name,quantity=k,count=v['nonfinite'])
            for m in s['microbatches'] for name,group in m['categories'].items()
            for k,v in group.items() if isinstance(v,dict) and v.get('nonfinite')]) for s in r.get('steps',[])]) for r in value['ranks']])))
