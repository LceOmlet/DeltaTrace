"""Compare saved installed FSDP lifecycle source with its PyTorch 2.8 owner."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
INSTALLED = ROOT / "native-FSDP2-owner-source-20261011-v1/torch/distributed/fsdp/_fully_shard"
SAVED = ROOT / "pytorch-v2.8.0-FSDP-owner-source-20261011-v1"
SAVED.mkdir(exist_ok=True)
rows = []
for name in ("_fsdp_collectives.py", "_fsdp_param.py", "_fsdp_param_group.py", "_fsdp_state.py", "_fully_shard.py"):
    url = "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/distributed/fsdp/_fully_shard/" + name
    p = SAVED / name
    if not p.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        ast.parse(data.decode("utf-8"))
        p.write_bytes(data)
    official = p.read_bytes()
    native = (INSTALLED / name).read_bytes()
    a, b = official.decode("utf-8"), native.decode("utf-8")
    same_ast = ast.dump(ast.parse(a), include_attributes=False) == ast.dump(ast.parse(b), include_attributes=False)
    diff = "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), fromfile=url, tofile=str(INSTALLED / name)))
    (SAVED / (name + ".installed.diff")).write_text(diff, encoding="utf-8")
    rows.append(dict(name=name, official_url=url,
                     official_sha256=hashlib.sha256(official).hexdigest(),
                     installed_sha256=hashlib.sha256(native).hexdigest(),
                     byte_equal=official == native, AST_equal=same_ast,
                     diff_lines=len(diff.splitlines())))
report = dict(observed_unix=time.time(), installed_torch="2.8.0+metax3.5.3.9", official_base_tag="v2.8.0", files=rows,
              scope="Saved installed Python lifecycle files only; does not validate MetaX C++ kernels, MCCL, allocator, or execution ordering.",
              production_changes=0, model_calls=0, CUDA_initialized=False)
p = ROOT / "native-FSDP-Python-owner-comparison-20261011-v1.json"
p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
