"""Keep native response columns when trimming shared prompt padding."""
from pathlib import Path
from patch_verl_environment_entry import replace_once


def patch(source):
    anchor = '                    left_trim = int(attention_mask.long().argmax(-1).min().item())\n'
    return replace_once(source, anchor, anchor +
        '                    # A native trajectory may left-pad the response too.\n'
        '                    # Retain every response column and its preceding logit.\n'
        '                    left_trim = min(left_trim, input_ids.shape[-1] - response_length - 1)\n')


if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1])/'verl/workers/actor/dp_actor.py'
    source = patch(path.read_text())
    compile(source, str(path), 'exec')
    path.write_text(source, newline='\n')
