"""Minimal fix in LOOP's existing Qwen3 model client for tokenizer-owned IDs.

Qwen3.5 uses the existing reasoning-capable Qwen3 client and its unchanged
message/token/HTTP logic. Its three legacy token IDs come from the checkpoint
tokenizer; its generation-header probe includes a user query as that template
requires. No new client is introduced.
"""
from pathlib import Path
import argparse
import re


def patch(text: str) -> str:
    # Transformers 5 defaults to BatchEncoding. The owner's methods consume
    # list[int]; request that official API return type explicitly on both 4/5.
    text, count = re.subn(r'(?m)^(\s*)tokenize=True,\n(?!\s*return_dict=False,)',
                         r'\1tokenize=True,\n\1return_dict=False,\n', text)
    if count not in (0, 3) or (count == 0 and text.count('return_dict=False,') != 3):
        raise ValueError('Expected three pinned chat-template tokenization calls')
    probe = '        dummy_msgs = [{"role": "system", "content": "dummy"}]'
    text = text.replace(probe, '        dummy_msgs = [{"role": "user", "content": "dummy"}]', 1)
    if 'SpecialToken(id=self._tokenizer.convert_tokens_to_ids(' in text:
        return text
    loader = '        self._tokenizer = AutoTokenizer.from_pretrained(base_model_path)\n'
    anchor = '        self._special_tokens = {\n'
    if text.count(loader) != 1 or text.count(anchor) != 1:
        raise ValueError('Expected pinned Qwen3 tokenizer/ID initialization once')
    text = text.replace(loader, '', 1).replace(anchor, loader + '\n' + anchor, 1)
    for number, token in [(151644, '<|im_start|>'), (151645, '<|im_end|>'), (151643, '<|endoftext|>')]:
        old = f'SpecialToken(id={number}, content="{token}")'
        new = f'SpecialToken(id=self._tokenizer.convert_tokens_to_ids("{token}"), content="{token}")'
        if text.count(old) != 1:
            raise ValueError(f'Expected original special token entry {token}')
        text = text.replace(old, new, 1)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner_root', type=Path)
    args = parser.parse_args()
    path = args.owner_root / 'phi_agents/rl/llm/qwen_3.py'
    path.write_text(patch(path.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
