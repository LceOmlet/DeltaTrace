"""Read-only inventory against the pinned archive; not an approval of each patch."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
audit = root / 'receipts/upstream-alignment-20260929'
original = audit / 'official'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate', type=Path, default=root / 'candidates/official-verl-20bd331-distributed-dt')
parser.add_argument('--output', type=Path, default=audit / 'owner-source-inventory-current.json')
args = parser.parse_args()
candidate = args.candidate


def functions(source):
    output = {}

    def visit(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if not isinstance(child, ast.ClassDef):
                    output[name] = ast.dump(child)
                visit(child, name + '.')
            else:
                visit(child, prefix)
    visit(ast.parse(source))
    return output


result = dict(scope=__doc__, source=str(original), candidate=str(candidate), changed_files=[])
for source in sorted(original.rglob('*.py')):
    relative = source.relative_to(original)
    current = candidate / relative
    if not current.is_file():
        result.setdefault('missing_files', []).append(str(relative))
        continue
    before, after = source.read_bytes(), current.read_bytes()
    if before == after:
        continue
    entry = dict(path=str(relative), original_sha256=hashlib.sha256(before).hexdigest(),
                 candidate_sha256=hashlib.sha256(after).hexdigest())
    parsed = {}
    for side, source_bytes in (('original', before), ('candidate', after)):
        try:
            parsed[side] = functions(source_bytes)
        except SyntaxError as exc:
            entry[side + '_parse_error'] = str(exc)
    if len(parsed) == 2:
        left, right = parsed['original'], parsed['candidate']
        entry['added_functions'] = sorted(right.keys() - left.keys())
        entry['removed_functions'] = sorted(left.keys() - right.keys())
        entry['changed_functions'] = sorted(key for key in left.keys() & right.keys() if left[key] != right[key])
    result['changed_files'].append(entry)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
