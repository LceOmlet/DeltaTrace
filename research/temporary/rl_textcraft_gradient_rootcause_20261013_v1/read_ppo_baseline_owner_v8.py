"""Read the fixed native PPO/critic capability and current physical occupancy.

This prepares the user-requested PPO control. It does not initialize a model,
modify a training source or configuration, or execute any GPU work.
"""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
VERL=Path(os.environ['VERL_ROOT'])
report={'unix':time.time(),'scope':__doc__,'GPU_model_calls':0,'VERL_ROOT':str(VERL),
    'physical_mx_smi':subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=30).stdout,
    'sources':{},'native_objects':{}}
for relative in ['verl/trainer/config/ppo_trainer.yaml','verl/workers/config/critic.yaml',
                 'verl/trainer/main_ppo.py','verl/workers/fsdp_workers.py','verl/trainer/ppo/ray_trainer.py']:
    path=VERL/relative
    if path.is_file():
        data=path.read_bytes()
        report['sources'][relative]={'path':str(path),'SHA256':hashlib.sha256(data).hexdigest(),
            'text':data.decode() if path.suffix=='.yaml' else None}
for module,name in [('verl.workers.fsdp_workers','CriticWorker'),('verl.workers.critic.dp_critic','DataParallelPPOCritic')]:
    try:
        obj=getattr(importlib.import_module(module),name)
        path=Path(inspect.getsourcefile(obj))
        report['native_objects'][name]={'path':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':inspect.getsource(obj)}
    except Exception as exc:report['native_objects'][name]={'error':repr(exc)}
from transformers import AutoConfig
launch=json.loads((ROOT/'runs/textcraft-grpo-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
model_path=launch['options']['actor_rollout_ref.model.path']
config=AutoConfig.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
report['model_config_class']={'module':type(config).__module__,'name':type(config).__name__,'model_type':config.model_type}
from transformers.models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
report['original_token_classification_mapping']=MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.get(config.model_type)
report['PPO_model_path']=model_path
path=OUT/'PPO-baseline-owner-v8.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
