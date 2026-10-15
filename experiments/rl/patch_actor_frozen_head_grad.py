"""Do not compute a discarded vocabulary gradient in the pinned VERL head.

The existing custom autograd owner already returns no vocabulary gradient when
that input is frozen. Pass that same condition into its compiled chunk helper.
The forward, hidden gradient, trainable head path and PPO objective stay owned
by the original implementation.
"""
from pathlib import Path


def patch_frozen_vocab_gradient(source):
    replacements = [
        (
            '    temperature: float = 1.0,\n) -> Tuple[torch.FloatTensor, torch.FloatTensor]:\n'
            '    logits = (hidden_states @ vocab_weights.t()) / temperature\n',
            '    temperature: float = 1.0,\n    compute_vocab_grad: bool = True,\n'
            ') -> Tuple[torch.FloatTensor, torch.FloatTensor]:\n'
            '    logits = (hidden_states @ vocab_weights.t()) / temperature\n',
        ),
        (
            '    dvocab_weights = (dlogits.t() @ hidden_states)\n',
            '    dvocab_weights = (dlogits.t() @ hidden_states) if compute_vocab_grad else None\n',
        ),
        (
            '                temperature=temperature,\n            )\n\n'
            '            if hidden_states.requires_grad:\n',
            '                temperature=temperature,\n'
            '                compute_vocab_grad=vocab_weights.requires_grad,\n'
            '            )\n\n            if hidden_states.requires_grad:\n',
        ),
    ]
    for old, new in replacements:
        if new in source:
            continue
        if source.count(old) != 1:
            raise ValueError('Pinned VERL frozen-head gradient anchor changed: '+old)
        source = source.replace(old, new, 1)
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner_root', type=Path)
    args = parser.parse_args()
    path = args.owner_root/'verl/utils/experimental/torch_functional.py'
    result = patch_frozen_vocab_gradient(path.read_text())
    compile(result, str(path), 'exec')
    path.write_text(result, newline='\n')
