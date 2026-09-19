"""Apply compatibility-only patches required by the pinned verl-agent2 tree.

The upstream Sokoban package has a nested package with the same name as its
parent.  Eagerly importing both modules creates a Python 3.12 import cycle
(and can segfault before a traceback).  Princeton's legacy Gym environment
also needs its checker disabled, and Sokoban must not import Transformers in a
Ray worker. These edits do not change environment dynamics or trainer math.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CONTENT = '''# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0.

from .projection import sokoban_projection


def build_sokoban_envs(*args, **kwargs):
    """Lazy import to avoid the upstream nested-package import cycle."""
    from .envs import build_sokoban_envs as _build_sokoban_envs
    return _build_sokoban_envs(*args, **kwargs)
'''

WEBSHOP_FILE = "agent_system/environments/env_package/webshop/envs.py"
WEBSHOP_OLD = "        env_kwargs['seed'] = seed\n        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)"
WEBSHOP_NEW = "        env_kwargs = dict(env_kwargs)\n        env_kwargs['seed'] = seed\n        # Princeton's legacy Gym environment has no action_space.\n        env_kwargs.setdefault('disable_env_checker', True)\n        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)"
SOKOBAN_ENV_FILE = "agent_system/environments/env_package/sokoban/sokoban/env.py"
SOKOBAN_ENV_OLD = "from agent_system.environments.env_package.sokoban.sokoban.base import BaseDiscreteActionEnv"
SOKOBAN_ENV_NEW = "from .base import BaseDiscreteActionEnv"
SOKOBAN_BASE_FILE = "agent_system/environments/env_package/sokoban/sokoban/base.py"
SOKOBAN_BASE_OLD = "from transformers import AutoTokenizer\nimport torch"
SOKOBAN_BASE_NEW = "from typing import TYPE_CHECKING\n\nimport torch\n\nif TYPE_CHECKING:\n    from transformers import AutoTokenizer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verl_root", type=Path)
    args = parser.parse_args()
    target = args.verl_root / "agent_system/environments/env_package/sokoban/__init__.py"
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.read_text() != CONTENT:
        target.write_text(CONTENT)
        print(f"patched {target}")
    else:
        print(f"already patched {target}")
    webshop = args.verl_root / WEBSHOP_FILE
    text = webshop.read_text()
    if WEBSHOP_NEW not in text:
        if WEBSHOP_OLD not in text:
            raise RuntimeError(f"cannot find WebShop compatibility anchor in {webshop}")
        webshop.write_text(text.replace(WEBSHOP_OLD, WEBSHOP_NEW, 1))
        print(f"patched {webshop}")
    else:
        print(f"already patched {webshop}")
    sokoban_env = args.verl_root / SOKOBAN_ENV_FILE
    text = sokoban_env.read_text()
    if SOKOBAN_ENV_NEW not in text:
        if SOKOBAN_ENV_OLD not in text:
            raise RuntimeError(f"cannot find Sokoban import anchor in {sokoban_env}")
        sokoban_env.write_text(text.replace(SOKOBAN_ENV_OLD, SOKOBAN_ENV_NEW, 1))
        print(f"patched {sokoban_env}")
    else:
        print(f"already patched {sokoban_env}")
    sokoban_base = args.verl_root / SOKOBAN_BASE_FILE
    text = sokoban_base.read_text()
    if SOKOBAN_BASE_NEW not in text:
        if SOKOBAN_BASE_OLD not in text:
            raise RuntimeError(f"cannot find Sokoban base import anchor in {sokoban_base}")
        text = text.replace(SOKOBAN_BASE_OLD, SOKOBAN_BASE_NEW, 1)
        text = text.replace("tokenizer: AutoTokenizer,", "tokenizer: \"AutoTokenizer\",", 1)
        sokoban_base.write_text(text)
        print(f"patched {sokoban_base}")
    else:
        print(f"already patched {sokoban_base}")


if __name__ == "__main__":
    main()
