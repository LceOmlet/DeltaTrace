"""Source-pinned Qwen3 root retention with grouped exact diagnostics."""
import hashlib
import json
from pathlib import Path
import sys


def make_root_retained_qwen3(root,model,library,library_sha256):
    """Return a fresh-per-call B1/E2 controller and the verified source receipt."""
    root=Path(root).resolve()
    from retained import make_retained_qwen3
    _,base=make_retained_qwen3(root)
    raw=(root/'deltatrace/accelerated/root_retained_qwen3_sources.json').read_bytes()
    manifest=json.loads(raw)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    assert base['manifest_sha256']==manifest['base_retained_manifest_sha256']
    for name,digest in manifest['files'].items():assert sha((root/name).read_bytes())==digest,name
    from native_capture_events import check_runtime
    from qwen3_root_retained import NativeRootTape,propagate_root_tape
    from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
    from compiled_fa_runtime import CompiledInputsFA
    for name in manifest['modules']:
        expected=(root/'deltatrace/accelerated/qwen3'/(name+'.py')).resolve()
        assert Path(sys.modules[name].__file__).resolve()==expected,name
    assert Path(sys.modules['native_capture_events'].__file__).resolve()==root/'deltatrace/accelerated/native_capture_events.py'
    assert Path(__file__).resolve()==root/'deltatrace/accelerated/root_retained_qwen3.py'
    runtime=check_runtime()
    finite=CompiledInputsFA(library,library_sha256)

    class RootRetainedQwen3:
        def attribute(self,before_ids,after_ids,mask,prompt_len,*,mutation_audit=False):
            import torch
            assert not model.training and after_ids.shape[0]==1
            # The complete API owns this counter, including root operand storage.
            torch.cuda.reset_peak_memory_stats()
            tape=NativeRootTape(model,mutation_audit=mutation_audit)
            finite.audit_inputs=mutation_audit
            try:
                with tape:
                    before,after=capture_checkpoint_pair(model,before_ids,after_ids,mask,prompt_len)
                return propagate_root_tape(model,before,after,tape,pv_rule='content_P1',finite_attention=finite)
            finally:
                tape.clear()
                finite.audit_inputs=False

    return RootRetainedQwen3(),{'manifest_sha256':sha(raw),'files':manifest['files'],
        'base_retained':base,'family':'qwen3','capture_runtime':runtime,
        'finite_library_sha256':library_sha256,
        'scope':'One actual B2 root and per-call retained native operands; no decoder replay or cross-call tensor cache.'}
