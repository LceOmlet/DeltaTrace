"""Remove the superseded optional renderer override from the installed owner."""
import ast
import hashlib
import json
import os
from pathlib import Path
from patch_verl_agent2 import restore_appworld_history_limit

root=Path(__file__).resolve().parent
path=Path(os.environ['VERL_ROOT'])/'agent_system/environments/env_manager.py'
original=root/'official/agent_system/environments/env_manager.py'
before=path.read_text()
after=restore_appworld_history_limit(before)
def renderer(text):
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='AppWorldEnvironmentManager')
    return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='build_text_obs'))
assert renderer(after)==renderer(original.read_text())
(root/'env-manager-before-history-restore.py').write_text(before)
path.write_text(after)
receipt=dict(path=str(path),renderer_ast_equals_pinned_owner=True,
    before_sha256=hashlib.sha256(before.encode()).hexdigest(),
    after_sha256=hashlib.sha256(after.encode()).hexdigest())
(root/'appworld-history-restored.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
