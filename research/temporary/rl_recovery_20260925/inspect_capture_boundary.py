"""Exercise existing passive captures on one original Qwen GDN layer."""
import inspect
import json
import os
from pathlib import Path
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from qwen35_gdn_finite import NativeGDNCapture
from accelerated.qwen35.qwen35_code_local_capture import NativeGDNCapture as LocalCapture

torch.set_num_threads(4)
torch.manual_seed(2026)
config = AutoConfig.from_pretrained('/mnt/si0021787ci2/default/models/Qwen3.5-9B', local_files_only=True)
module = Qwen3_5GatedDeltaNet(config.text_config, layer_idx=0).cuda().bfloat16().eval()
inputs = torch.randn(8, 128, config.text_config.hidden_size, device='cuda', dtype=torch.bfloat16)
rows = []
with torch.no_grad():
    for capture_type in (NativeGDNCapture, LocalCapture):
        capture = capture_type(module, device='cuda')
        with capture:
            module(inputs, attention_mask=None)
        row = dict(capture=capture_type.__module__, calls=capture.calls,
                   event_file=inspect.getfile(capture.event),
                   endpoint_names=sorted(capture.endpoints))
        rows.append(row)
        print(json.dumps(row), flush=True)
path = Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-major-cost/native-capture-boundary.json'
path.write_text(json.dumps(dict(scope=__doc__, cases=rows), indent=2)+'\n')
