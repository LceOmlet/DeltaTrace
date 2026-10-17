"""Prepare one default-inert integer-copy change in the pinned native owner."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'
ORIGINAL = Path('/opt/conda/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py')
CANDIDATE = ROOT / 'candidates/native-mrope-row-copy-20261017-v1/vllm/v1/worker/gpu_model_runner.py'

original = ORIGINAL.read_text()
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == 'e8a73e1ccbfff37b880cd47ba9c115db572aef3c9d4d9eee50b18b729117f9bb'
before = '''            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )'''
after = '''            if getattr(self, "_native_mrope_row_copy", False):
                # Preserve the native strides and asynchronous input-prep
                # lifetime. Each row uses the original pinned Tensor.copy_.
                for position_axis in range(self.mrope_positions.gpu.shape[0]):
                    self.mrope_positions.gpu[position_axis, :total_num_scheduled_tokens].copy_(
                        self.mrope_positions.cpu[position_axis, :total_num_scheduled_tokens],
                        non_blocking=True,
                    )
            else:
                self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                    self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                    non_blocking=True,
                )'''
assert original.count(before) == 1
candidate = original.replace(before, after)
compile(candidate, str(CANDIDATE), 'exec')


class DisabledBranch(ast.NodeTransformer):
    def visit_If(self, node):
        if ast.unparse(node.test) == "getattr(self, '_native_mrope_row_copy', False)":
            return node.orelse
        return self.generic_visit(node)


disabled = DisabledBranch().visit(copy.deepcopy(ast.parse(candidate)))
assert ast.dump(disabled, include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
assert not CANDIDATE.exists()
CANDIDATE.parent.mkdir(parents=True)
CANDIDATE.write_text(candidate)
patch = AUDIT / 'native-MRoPE-row-copy-v1.patch'
patch.write_text(''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
    fromfile=str(ORIGINAL), tofile=str(CANDIDATE))))
record = dict(status='prepared_only_minimal_native_owner_copy_candidate_not_deployed',
    original=str(ORIGINAL), original_SHA256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
    candidate=str(CANDIDATE), candidate_SHA256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
    disabled_branch_whole_module_AST_equal_original=True,
    patch=str(patch), patch_SHA256=hashlib.sha256(patch.read_bytes()).hexdigest(),
    change='Only MRoPE int64 H2D transfer; shape,stride,address,original synchronize_input_prep and all model/sampler/update code retained',
    system_owner_modified=False, model_calls=0, optimizer_updates=0)
(AUDIT / 'native-MRoPE-row-copy-v1-prepared.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
