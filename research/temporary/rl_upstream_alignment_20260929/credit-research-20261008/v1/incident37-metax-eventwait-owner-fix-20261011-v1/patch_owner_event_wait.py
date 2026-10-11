"""Candidate: change one lookup in the installed MetaX FSDP owner function.

The installed owner body and class are retained. Its CUDA-event cleanup calls
torch.cuda.current_stream instead of the broken generic accelerator bridge.
No function body is copied, no global torch API is patched, and no installed
package file is modified. Callers must record this explicit runtime override.
"""
import hashlib
import marshal
from pathlib import Path

OWNER_SHA = '2798a87b2631091df738afaff6dfcead38ce0cc421585b4331ecaaefb4e784f5'


def apply():
    import torch
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as owner

    assert torch.__version__ == '2.8.0+metax3.5.3.9'
    owner_path = Path(owner.__file__)
    assert hashlib.sha256(owner_path.read_bytes()).hexdigest() == OWNER_SHA
    function = owner.FSDPParamGroup.finalize_backward
    original = function.__code__
    assert original.co_names.count('accelerator') == 1
    # Exactly the owner's existing line:
    # torch.accelerator.current_stream().wait_event(event)
    # -> torch.cuda.current_stream().wait_event(event).
    # Names retain their original bytecode indices; all instructions,
    # constants, branches, event objects and owner state transitions stay put.
    names = tuple('cuda' if name == 'accelerator' else name for name in original.co_names)
    patched = original.replace(co_names=names)
    assert patched.co_code == original.co_code
    assert patched.co_consts == original.co_consts
    function.__code__ = patched
    receipt = dict(owner_path=str(owner_path), owner_file_SHA256=OWNER_SHA,
        function='FSDPParamGroup.finalize_backward',
        original_code_SHA256=hashlib.sha256(marshal.dumps(original)).hexdigest(),
        patched_code_SHA256=hashlib.sha256(marshal.dumps(patched)).hexdigest(),
        original_names=list(original.co_names), patched_names=list(patched.co_names),
        instruction_bytes_unchanged=True, constants_unchanged=True,
        copied_function_bodies=0, installed_files_modified=0,
        numerical_or_algorithm_changes=0)

    def restore():
        assert function.__code__ is patched
        function.__code__ = original

    return receipt, restore
