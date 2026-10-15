"""Read actual native optimizer construction/state ownership on CPU only."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'


def excerpts(path, words, radius=3, function=None):
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()
    start, end = 0, len(lines)
    if function:
        functions = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function]
        assert len(functions) == 1
        start, end = functions[0].lineno-1, functions[0].end_lineno
    selected = set()
    for n in range(start, end):
        if any(w in lines[n] for w in words):
            selected.update(range(max(start, n-radius), min(end, n+radius+1)))
    return dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                lines=[dict(line=n+1, text=lines[n]) for n in sorted(selected)])


worker = excerpts(inspect.getsourcefile(ActorRolloutRefWorker), ['optim'],
                  function='_build_model_optimizer')
assert worker['SHA256'] == '9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3'
manager = excerpts(inspect.getsourcefile(FSDPCheckpointManager),
                   ['optimizer.state_dict', 'optimizer.load_state_dict', 'optim_state_dict'])
assert manager['SHA256'] == 'f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01'
adam_module = inspect.getmodule(torch.optim.AdamW).__name__
import importlib
adam_impl = importlib.import_module('torch.optim.adam')
adam = excerpts(inspect.getsourcefile(adam_impl),
                ['bias_correction1 =', 'bias_correction2 =', 'bias_correction2_sqrt =',
                 'step_size =', 'denom =', 'param.addcdiv_'],
                function='_single_tensor_adam')
report = dict(unix=time.time(), scope=__doc__, worker=worker, checkpoint_manager=manager,
    AdamW_module=adam_module, native_Adam_formula_source=adam,
    model_calls=0, backward_calls=0, optimizer_steps=0, cuda_initialized=torch.cuda.is_initialized(),
    limits=['This source audit verifies construction and saved-state ownership; it does not execute an optimizer or change a training job.',
            'The FP64 moment-ratio statistic in v109 excludes native update rounding and weight decay; it is not an exact reconstructed CUDA parameter step.'])
assert not report['cuda_initialized']
dest = OUT/'native-optimizer-constructor-source-v110.json'
dest.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest), report=report)))
