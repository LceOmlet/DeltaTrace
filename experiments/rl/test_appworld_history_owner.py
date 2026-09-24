"""Exercise the real author renderer/memory; no task-success claim."""
import ast
from pathlib import Path

from omegaconf import OmegaConf
from agent_system.environments import env_manager as owner
from agent_system.memory.memory import SimpleMemory
from patch_verl_agent2 import patch_appworld_history_limit


def classes():
    source = Path(owner.__file__).read_text()
    patched = patch_appworld_history_limit(source)
    assert patch_appworld_history_limit(patched) == patched
    # Recover the original renderer expression if the installed owner is patched.
    original = source.replace(
        '                history_char_limit = self.config.env.get("appworld_history_char_limit", 10000)\n'
        '                if history_char_limit is not None and len(action_history) > history_char_limit:\n'
        '                    action_history = "... " + action_history[-history_char_limit:]',
        '                if len(action_history) > 10000:\n'
        '                    action_history = "... " + action_history[-10000:]')
    result = []
    for text in (original, patched):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                    and n.name == 'AppWorldEnvironmentManager')
        namespace = dict(vars(owner))
        exec(compile(ast.Module(body=[node], type_ignores=[]), owner.__file__, 'exec'), namespace)
        result.append(namespace['AppWorldEnvironmentManager'])
    return result


def make(cls, *, full=False):
    manager = object.__new__(cls)
    manager.config = OmegaConf.create({'env': {'history_length': 40 if full else 2}})
    if full:
        manager.config.env.appworld_history_char_limit = None
    manager.supervisors = [dict(first_name='Fixture', last_name='Supervisor', email='fixture@example.test', phone_number='0')]
    manager.tasks = ['Read-only history boundary fixture.']
    manager.memory = SimpleMemory()
    manager.memory.reset(batch_size=1)
    for step in range(4):
        obs = f'BEGIN_RECORD_{step} ' + ('content ' * 2000) + f' END_RECORD_{step}'
        manager.memory.store({'action': [f'print({step})'], 'text_obs': [obs]})
    return manager


def test_default_matches_actual_author_renderer():
    original, patched = classes()
    a, b = make(original), make(patched)
    for init in (True, False):
        assert a.build_text_obs(['current'], init=init) == b.build_text_obs(['current'], init=init)


def test_approved_history_preserves_all_executed_actions_and_observations():
    _, patched = classes()
    manager = make(patched, full=True)
    text = manager.build_text_obs(['current'])[0]
    assert len(text) > 64000
    for step in range(4):
        assert f'print({step})' in text
        assert f'BEGIN_RECORD_{step}' in text and f'END_RECORD_{step}' in text
    assert len(manager.memory[0]) == 4
