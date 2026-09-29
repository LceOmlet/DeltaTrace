"""Read-only source identity audit of the extracted native environment."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--expected', type=Path, required=True)
p.add_argument('--owner-root', type=Path, required=True)
p.add_argument('--project-root', type=Path, required=True)
p.add_argument('--pristine-project-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
expected = json.loads(a.expected.read_text())
observed = {name: hashlib.sha256((a.owner_root / name).read_bytes()).hexdigest() for name in expected}
assert observed == expected, 'Environment owner file differs from pinned source'
relative = 'agent_system/environments/env_manager.py'
def renderer(path):
    cls = next(n for n in ast.parse(path.read_text()).body if getattr(n, 'name', '') == 'AppWorldEnvironmentManager')
    return ast.dump(next(n for n in cls.body if getattr(n, 'name', '') == 'build_text_obs'))
assert renderer(a.project_root / relative) == renderer(a.pristine_project_root / relative)
result = dict(loop_file_sha256=observed, project_renderer_ast_matches_pristine=True, passed=True)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(passed=True, owner_files=len(observed), project_renderer_ast_matches_pristine=True)))
