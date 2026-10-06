"""Prepare only a default-inert target-coordinate owner extension.

No model, target labels, finite seed, reward or training coefficient is changed.
The candidate is outside the default imports and is not numerical acceptance.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SOURCE = REPO / 'deltatrace/clean/qwen35/qwen35_answer_finite.py'
EXPECTED = '9819cf34333d2ce67224cd49d404d21192040311001429b5c97982d76c727526'


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    # Preserve the existing source bytes except the new owner method. This is
    # a versioned patch of the target owner, not a second target implementation.
    original = raw.decode('utf8')
    newline = '\r\n' if '\r\n' in original else '\n'
    needle = newline + newline + 'def _answer_seed_rule'
    assert original.count(needle) == 1
    method = '''
    def suffix_rows(self,starts,lengths):
        """Rebase exact predictor identities for row-specific cached suffixes.

        Full lengths come from the original case prompt_length+target_ids.
        A cache boundary may only precede the already selected predictor.
        Every target, label and endpoint identity is retained unchanged.
        """
        starts=tuple(starts);lengths=tuple(lengths)
        if (len(starts)!=self.batch or len(lengths)!=self.batch or not all(
                type(start) is int and type(length) is int
                and 0<=start<length<=self.length
                for start,length in zip(starts,lengths))):
            raise ValueError('Row boundaries and original case lengths disagree.')
        offsets=torch.tensor(starts,device=self.positions.device,dtype=torch.long)
        ends=torch.tensor(lengths,device=self.positions.device,dtype=torch.long)
        if not bool(((self.positions>=offsets[self.samples])
                     &(self.positions<ends[self.samples])).all()):
            raise ValueError('Every original target predictor must lie in its cached suffix.')
        selected=copy.copy(self)
        selected.length=max(length-start for start,length in zip(starts,lengths))
        selected.positions=self.positions-offsets[self.samples]
        selected.paired_positions=selected.positions.repeat_interleave(2)
        return selected
'''
    candidate = original.replace(needle, method.replace('\n', newline) + needle)
    ast.parse(candidate)
    baseline_ast = ast.parse(original)
    candidate_ast = ast.parse(candidate)
    owner = next(node for node in candidate_ast.body
                 if isinstance(node, ast.ClassDef) and node.name == 'PackedAnswerTargets')
    owner.body = [node for node in owner.body if not (
        isinstance(node, ast.FunctionDef) and node.name == 'suffix_rows')]
    assert ast.dump(candidate_ast, include_attributes=False) == ast.dump(
        baseline_ast, include_attributes=False)
    out = HERE / 'native-representation-candidate'
    (out / 'baseline').mkdir(parents=True, exist_ok=True)
    (out / 'candidate').mkdir(exist_ok=True)
    old = out / 'baseline' / SOURCE.name
    new = out / 'candidate' / SOURCE.name
    old.write_bytes(raw)
    new.write_bytes(candidate.encode('utf8'))
    patch = ''.join(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True),
        fromfile='baseline/' + SOURCE.name, tofile='candidate/' + SOURCE.name))
    patch_path = out / (SOURCE.name + '.patch')
    patch_path.write_bytes(patch.encode('utf8'))
    receipt = dict(status='prepared_only_unaccepted_not_deployed',
        source=dict(path=str(SOURCE), sha256=EXPECTED),
        candidate=dict(path=str(new), sha256=hashlib.sha256(new.read_bytes()).hexdigest()),
        patch=dict(path=str(patch_path), sha256=hashlib.sha256(patch_path.read_bytes()).hexdigest()),
        unchanged_original_owner_AST=True,
        scope='One target-coordinate method only; default original owner AST unchanged. No tensor arithmetic, GPU, model, FA/FLA, target seed, PPO or performance validation.',
        production_changes=0)
    (out / 'target-row-positions-prepared.json').write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
