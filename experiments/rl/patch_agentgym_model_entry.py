"""Request the native RolloutHandler's list[int] API under Transformers 4/5.

Only the chat-template return type changes. The owner's prompt formatter,
history, parser, masks, reward, length handling and rollout loop stay intact.
"""
import argparse
from pathlib import Path


def patch(source: str) -> str:
    old = 'return tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True)'
    new = old[:-1] + ', return_dict=False)'
    if new in source:
        return source
    if source.count(old) != 1:
        raise ValueError('Expected the pinned native generation prompt call once')
    return source.replace(old, new, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner_root', type=Path)
    args = parser.parse_args()
    path = args.owner_root / 'AgentGym-RL/verl/workers/rollout/schemas.py'
    path.write_text(patch(path.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
