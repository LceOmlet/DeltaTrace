"""Extend the pinned actor's existing shared-padding seam to the right.

The owner still computes log-probs, entropy, loss and updates. Only columns
masked out for every row leave its model input; output columns are restored.
The existing VERL_TRIM_SHARED_PADDING flag controls both sides.
"""
from pathlib import Path

from patch_verl_environment_entry import replace_once


def patch(source):
    marker = '        response_right_pad = 0\n'
    if marker in source:
        return source
    source = replace_once(source,
        '        response_left_pad = 0\n',
        '        response_left_pad = 0\n' + marker)
    anchor = ('                    position_ids = position_ids[..., left_trim:]\n'
              '                extra_args = {}\n')
    source = replace_once(source, anchor,
        '                    position_ids = position_ids[..., left_trim:]\n'
        '                    columns = torch.arange(input_ids.shape[-1], device=attention_mask.device)\n'
        '                    right_end = int((attention_mask * columns).max().item()) + 1\n'
        '                    # Retain a response position even for an empty action mask.\n'
        '                    right_end = max(right_end, input_ids.shape[-1] - response_length + 1)\n'
        '                    if getattr(getattr(self.actor_module, "config", None), "model_type", None) in ("qwen3_5", "qwen3_5_text"):\n'
        '                        right_end = min(input_ids.shape[-1], (right_end + 63) // 64 * 64)\n'
        '                    response_right_pad = input_ids.shape[-1] - right_end\n'
        '                    if response_right_pad:\n'
        '                        response_length -= response_right_pad\n'
        '                        response_labels = response_labels[:, :response_length]\n'
        '                        input_ids = input_ids[:, :right_end]\n'
        '                        attention_mask = attention_mask[:, :right_end]\n'
        '                        position_ids = position_ids[..., :right_end]\n'
        '                extra_args = {}\n')
    return replace_once(source, '            if response_left_pad:\n',
        '            if response_right_pad:\n'
        '                log_probs = torch.nn.functional.pad(log_probs, (0, response_right_pad))\n'
        '                if entropy is not None:\n'
        '                    entropy = torch.nn.functional.pad(entropy, (0, response_right_pad))\n'
        '            if response_left_pad:\n')


if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1]) / 'verl/workers/actor/dp_actor.py'
    result = patch(path.read_text())
    compile(result, str(path), 'exec')
    path.write_text(result, newline='\n')
