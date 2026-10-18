"""Change only scalar logging around the pinned native optimizer return."""
import ast
import hashlib

ACTOR_OLD = '                data = {"actor/grad_norm": grad_norm.detach().item()}\n'
ACTOR_NEW = '''                # full_tensor resolves the native DTensor NormPartial; no norm is reimplemented.
                grad_norm_local = grad_norm.detach().item()
                grad_norm_global = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                data = {"actor/grad_norm": grad_norm_global.detach().item(),
                        "actor/grad_norm_local": grad_norm_local}
'''
FORMAT_OLD = '            output.append(f"{k}:{v:.3f}")'
FORMAT_NEW = '''            output.append(f"{k}:{v:.9g}" if k in ("actor/grad_norm", "actor/grad_norm_local") else f"{k}:{v:.3f}")'''


def patch_actor(text):
    assert text.count(ACTOR_OLD) == 1
    original = ast.parse(text)
    result = text.replace('from torch import nn\n', 'from torch import nn\nfrom torch.distributed.tensor import DTensor\n', 1)
    result = result.replace(ACTOR_OLD, ACTOR_NEW, 1)
    modified = ast.parse(result)
    for name in ('_optimizer_step', '_forward_micro_batch', 'compute_log_prob'):
        before = next(n for n in ast.walk(original) if isinstance(n, ast.FunctionDef) and n.name == name)
        after = next(n for n in ast.walk(modified) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(before) == ast.dump(after), name
    # Reversibility verifies every other byte, including backward/clip/optimizer.
    assert result.replace(ACTOR_NEW, ACTOR_OLD, 1).replace('from torch.distributed.tensor import DTensor\n', '', 1) == text
    return result


def patch_formatter(text):
    assert text.count(FORMAT_OLD) == 1
    result = text.replace(FORMAT_OLD, FORMAT_NEW, 1)
    ast.parse(result)
    return result
