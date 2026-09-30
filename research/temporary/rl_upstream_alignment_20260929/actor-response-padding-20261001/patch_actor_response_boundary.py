"""Restore native response columns after trimming shared model-input padding."""
from pathlib import Path
from patch_verl_environment_entry import replace_once


def patch(source):
    # Replace the previous shape-preserving bound: it kept the whole global
    # response width even when most of those columns were shared padding.
    previous = ('                    # A native trajectory may left-pad the response too.\n'
                '                    # Retain every response column and its preceding logit.\n'
                '                    left_trim = min(left_trim, input_ids.shape[-1] - response_length - 1)\n')
    source = source.replace(previous, '')
    source = replace_once(source,
        '        response_length = micro_batch["responses"].size(-1)\n',
        '        response_length = micro_batch["responses"].size(-1)\n'
        '        response_labels = micro_batch["responses"]\n'
        '        response_left_pad = 0\n')
    anchor = '                    left_trim = int(attention_mask.long().argmax(-1).min().item())\n'
    source = replace_once(source, anchor, anchor +
        '                    # Keep the predecessor of the first real token.\n'
        '                    left_trim = max(0, left_trim - 1)\n')
    source = replace_once(source,
        '                    input_ids = input_ids[:, left_trim:]\n',
        '                    # Preserve the caller\'s response layout on return.\n'
        '                    response_left_pad = max(0, left_trim - (input_ids.shape[-1] - response_length - 1))\n'
        '                    response_length -= response_left_pad\n'
        '                    response_labels = response_labels[:, response_left_pad:]\n'
        '                    input_ids = input_ids[:, left_trim:]\n')
    source = replace_once(source,
        'logprobs_from_logits(logits, micro_batch["responses"][:, :head_response_length])',
        'logprobs_from_logits(logits, response_labels[:, :head_response_length])')
    return replace_once(source, '            return entropy, log_probs\n',
        '            if response_left_pad:\n'
        '                log_probs = torch.nn.functional.pad(log_probs, (response_left_pad, 0))\n'
        '                if entropy is not None:\n'
        '                    entropy = torch.nn.functional.pad(entropy, (response_left_pad, 0))\n'
        '            return entropy, log_probs\n')


if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1])/'verl/workers/actor/dp_actor.py'
    source = patch(path.read_text())
    compile(source, str(path), 'exec')
    path.write_text(source, newline='\n')
