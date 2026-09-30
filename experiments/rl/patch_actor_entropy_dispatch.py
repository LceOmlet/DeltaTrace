"""Use the pinned actor's configured entropy function on its dense path too."""
from pathlib import Path


def patch(source):
    old = '                        entropy = verl_F.entropy_from_logits(logits)'
    new = '                        entropy = self.compute_entropy_from_logits(logits)'
    if old not in source and source.count(new) == 1:
        return source
    if source.count(old) != 1:
        raise RuntimeError('Cannot find unique pinned dense entropy dispatch')
    return source.replace(old, new, 1)


if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1])/'verl/workers/actor/dp_actor.py'
    source = patch(path.read_text())
    compile(source, str(path), 'exec')
    path.write_text(source, newline='\n')
