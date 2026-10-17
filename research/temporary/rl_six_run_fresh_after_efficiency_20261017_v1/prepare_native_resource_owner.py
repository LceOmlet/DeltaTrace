"""Prepare the verified int64 copy change in the native vLLM owner only."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1'
OWNER = Path('/opt/conda/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert sha(OWNER) == 'e8a73e1ccbfff37b880cd47ba9c115db572aef3c9d4d9eee50b18b729117f9bb'
    evidence = ROOT / 'receipts/six-run-native-utilization-20261017-v1'
    for rank in (0, 6):
        observed = json.loads((evidence / f'native-MRoPE-row-copy-ABA-20261017-v1-rank{rank}.json').read_bytes())
        assert observed['status'] == 'bounded_natural_input_preparation_observation_completed'
        assert observed['original_callable_restored']
    original = OWNER.read_text()
    before = '''            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )'''
    after = '''            # Keep the native padded strides and input-prep event lifetime.
            # Contiguous pinned rows avoid a temporary for the strided 2D view.
            for position_axis in range(self.mrope_positions.gpu.shape[0]):
                self.mrope_positions.gpu[position_axis, :total_num_scheduled_tokens].copy_(
                    self.mrope_positions.cpu[position_axis, :total_num_scheduled_tokens],
                    non_blocking=True,
                )'''
    assert original.count(before) == 1
    changed = original.replace(before, after)
    compile(changed, str(OWNER), 'exec')
    # Replacing exactly this one copy node reconstructs the whole native AST.
    tree = ast.parse(changed)
    target = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_prepare_inputs')
    branch = next(n for n in ast.walk(target) if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.uses_mrope')
    assert len(branch.body) == 1 and isinstance(branch.body[0], ast.For)
    branch.body = ast.parse(before.strip()).body
    assert ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
    candidate = ROOT / 'candidates/native-mrope-row-copy-20261017-v2/gpu_model_runner.py'
    candidate.parent.mkdir(parents=True, exist_ok=False)
    candidate.write_text(changed)
    (AUDIT / 'gpu_model_runner.before.py').write_bytes(OWNER.read_bytes())
    (AUDIT / 'native-MRoPE-owner.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True), fromfile=str(OWNER), tofile=str(candidate))))
    record = dict(status='prepared_not_deployed', owner_path=str(OWNER), original_SHA256=sha(OWNER),
        candidate=str(candidate), candidate_SHA256=sha(candidate),
        native_whole_AST_unchanged_except_integer_copy=True,
        exact_integer_copy_evidence=str(evidence / 'native-MRoPE-integer-copy-20261017-v1-completed.json'),
        natural_ABA_evidence=str(evidence / 'native-MRoPE-row-copy-ABA-v1-summary.json'),
        native_allocator_evidence=str(evidence / 'native-pinned-host-register-20261017-v1-completed.json'),
        decoding_gain_scope='Matched natural request-count intervals about 3-4 percent; not the main training speedup',
        model_or_optimizer_test_calls=0)
    (AUDIT / 'native-resource-owner-prepared.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
