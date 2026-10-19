"""Build CPU-only inputs with the existing native exact-32768 fixture owner."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from verl import DataProto

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
source = ROOT / 'receipts/textcraft-weighted-reward-targets-20261015-v4/source/run_native_behavior_v4.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'd354653d2b59eefe138b27d4c6ea48593e579450b521640c69881e92710d47bf'
text = source.read_text()
functions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == 'requests']
assert len(functions) == 1
namespace = dict(torch=torch, copy=copy, np=np, DataProto=DataProto,
                 AUDIT=ROOT / 'receipts/textcraft-weighted-reward-targets-20261015-v4')
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True,trust_remote_code=True)
inputs = {key:namespace['requests'](tokenizer,capacity)[:4] for key,capacity in (('small',False),('capacity',True))}
assert all(len(data) == 4 for data in inputs.values())
assert inputs['capacity'].batch['input_ids'].shape == (4,32768)
target = AUDIT / 'root-pageable-capacity-inputs.pt'
assert not target.exists()
torch.save(inputs,target)
receipt = dict(status='prepared_CPU_only_original_capacity_inputs',
               owner=str(source),owner_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
               function_SHA256=hashlib.sha256(ast.get_source_segment(text,functions[0]).encode()).hexdigest(),
               inputs=str(target),inputs_SHA256=hashlib.sha256(target.read_bytes()).hexdigest(),
               shapes={key:list(data.batch['input_ids'].shape) for key,data in inputs.items()},
               GPU_calls=0,model_calls=0,optimizer_updates=0,
               scope='Original artificial B4/exact-32768 storage fixture. Not a task-quality or throughput sample.')
(AUDIT / 'root-pageable-capacity-inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
