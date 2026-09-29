"""Exercise the actual patched owner's exit block, without launching workers."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from patch_verl_environment_entry import patch_main


@pytest.mark.parametrize('factory', [False, True])
@pytest.mark.parametrize('failure', [None, 'init', 'fit', 'eval_close'])
def test_original_cleanup_runs_on_success_and_failure(factory, failure):
    owner_path = Path(os.environ['VERL_ROOT']) / 'verl/trainer/main_ppo.py'
    source = patch_main(owner_path.read_text())
    assert patch_main(source) == source
    compile(source, str(owner_path), 'exec')
    tree = ast.parse(source)
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
                 and any(isinstance(i.optional_vars, ast.Name)
                         and i.optional_vars.id == 'environment_cleanup' for i in node.items))
    imports = ast.parse('from contextlib import ExitStack').body
    calls = []

    def operation(name):
        def run():
            calls.append(name)
            if name == failure:
                raise RuntimeError(name)
        return run

    namespace = dict(config=SimpleNamespace(env={'factory': factory}),
                     trainer=SimpleNamespace(init_workers=operation('init'), fit=operation('fit')),
                     envs=SimpleNamespace(close=operation('train_close')),
                     val_envs=SimpleNamespace(close=operation('eval_close')))
    code = compile(ast.Module(body=imports+[block], type_ignores=[]), str(owner_path), 'exec')
    should_raise = failure in ('init', 'fit') or (factory and failure == 'eval_close')
    if should_raise:
        with pytest.raises(RuntimeError, match=failure):
            exec(code, namespace)
    else:
        exec(code, namespace)
    expected = ['init'] if failure == 'init' else ['init', 'fit']
    if factory:
        expected += ['eval_close', 'train_close']
    assert calls == expected
