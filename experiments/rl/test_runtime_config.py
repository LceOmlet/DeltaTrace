"""Check the launcher CPU limit reaches the pinned trainer's actual ray.init."""
import re
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir


def test_launcher_cpu_limit_reaches_owner_ray_init(monkeypatch):
    from verl.trainer import main_ppo as owner

    launcher = Path(__file__).with_name('run_verl_agent.sh').read_text()
    key = re.search(r'([+\w.]+)="\$\{DT_RAY_NUM_CPUS:-null\}"', launcher).group(1)
    received = {}
    monkeypatch.setattr(owner.ray, 'is_initialized', lambda: False)
    monkeypatch.setattr(owner.ray, 'init', lambda **kwargs: received.update(kwargs))
    monkeypatch.setattr(owner.ray, 'get', lambda value: value)
    monkeypatch.setattr(owner, 'TaskRunner', SimpleNamespace(remote=lambda: SimpleNamespace(
        run=SimpleNamespace(remote=lambda config: None))))
    with initialize_config_dir(version_base=None, config_dir=str(Path(owner.__file__).parent/'config')):
        config = compose(config_name='ppo_trainer', overrides=[key+'=8'])
    owner.run_ppo(config)
    assert received['num_cpus'] == 8
