"""Native AgentGym TextCraft client and prompt renderer in VERL's collector.

Only owner dataset/client/trajectory modules are loaded, avoiding the other
project's trainer, engine and optimizer. The original handler owns messages,
masks, token assembly and truncation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


def owner_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_textcraft_tokenizer(tokenizer):
    # The author's example uses this official Qwen2.5 task template. Keep that
    # conversation format while the requested Qwen3.5 tokenizer supplies IDs.
    import json
    source = json.loads(Path(__file__).with_name('textcraft_qwen_template.json').read_text())
    tokenizer.chat_template = source['chat_template']


def make_textcraft_environments(configuration, tokenizer):
    from textcraft_owner_rollout import TextCraftOwner
    configure_textcraft_tokenizer(tokenizer)
    return TextCraftOwner(configuration, tokenizer, False), TextCraftOwner(configuration, tokenizer, True)
