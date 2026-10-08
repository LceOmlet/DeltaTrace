"""Read installed owner formulas/config only; no model, Torch or GPU import."""
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ENTRY, SSH

BODY = r'''
import ast, hashlib, json, time
from pathlib import Path
import psutil
site=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/lib/python3.12/site-packages')
model=Path('/mnt/si0021787ci2/default/models/Qwen3.5-9B/config.json')
specs={
 'transformers/models/qwen3_5/modeling_qwen3_5.py': ['Qwen3_5GatedDeltaNet'],
 'fla/ops/common/chunk_delta_h.py': ['chunk_gated_delta_rule_bwd_dhu', 'chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64'],
 'fla/ops/common/chunk_o.py': ['chunk_fwd_o', 'chunk_bwd_dv_local'],
 'fla/ops/gated_delta_rule/chunk.py': ['chunk_gated_delta_rule_fwd', 'chunk_gated_delta_rule_bwd'],
}
result={'unix':time.time(), 'scope':'Installed source/config read only; no imports of model/Torch/FLA and no operator execution', 'owners':{}}
for relative,names in specs.items():
 p=site/relative
 data=p.read_bytes()
 text=data.decode('utf-8')
 tree=ast.parse(text)
 sections={}
 for node in tree.body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
   sections[node.name]={'line':node.lineno, 'end_line':node.end_lineno,
                       'source':ast.get_source_segment(text,node)}
 result['owners'][relative]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),
                            'sections':sections,'requested_missing':sorted(set(names)-set(sections))}
data=model.read_bytes()
config=json.loads(data)
result['model_config']={'path':str(model),'sha256':hashlib.sha256(data).hexdigest(),
                        'text_config':config.get('text_config',config)}
result['formal_processes']={}
for task,pid,birth in [('textcraft',2833207,1791370325.16),('appworld',2786671,1791369896.67)]:
 try:
  process=psutil.Process(pid)
  result['formal_processes'][task]={'pid':pid,'birth':process.create_time(),
      'same_birth':process.create_time()==birth,'status':process.status()}
 except psutil.NoSuchProcess:
  result['formal_processes'][task]={'pid':pid,'missing':True}
hold=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt')
result['text_update_releases']=[(hold/f'rank{rank}-release-update').exists() for rank in (0,1)]
print(json.dumps(result,ensure_ascii=False))
'''

if __name__ == '__main__':
    script = ('source '+ENTRY+'/metax-entry.env.sh\n'
              'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+BODY+'\nPY\n')
    run = subprocess.run(SSH+['bash', '-s'], input=script.encode(), capture_output=True, timeout=45)
    (HERE/'gdn-owner-readonly.stderr.txt').write_bytes(run.stderr)
    run.check_returncode()
    result = json.loads(run.stdout)
    (HERE/'gdn-owner-readonly.json').write_bytes(run.stdout)
    print(json.dumps({
        'scope':result['scope'],
        'owners':{k:{'sha256':v['sha256'],'sections':list(v['sections']),
                     'requested_missing':v['requested_missing']} for k,v in result['owners'].items()},
        'text_config':result['model_config']['text_config'],
        'formal_processes':result['formal_processes'],
        'text_update_releases':result['text_update_releases'],
    },ensure_ascii=False,indent=2))
