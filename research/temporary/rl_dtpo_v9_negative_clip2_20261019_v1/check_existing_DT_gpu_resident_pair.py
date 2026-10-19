"""Compare the owner's GPU-resident replay mode on one actual small B4.

Compose the existing same-input resource comparison. Training keeps its
baseline result, original flags and RNG. No optimizer or rollout call is added.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('check_existing_DT_pageable_pair.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'e9a63d4f72b05d567c2d36255fca2fa5d8fa447d22cdcbb114dc314ce8995ec7'
code = owner.read_text()
for old, new in (
    ('_existing_DT_pageable_pair_scope_installed', '_existing_DT_gpu_resident_pair_scope_installed'),
    ('native_DT_existing_pageable_pair', 'native_DT_existing_gpu_resident_pair'),
    ('existing-DT-pageable-pair', 'existing-DT-gpu-resident-pair'),
    ('optional_same_input_pageable_pair', 'optional_same_input_gpu_resident_pair'),
    ('runner.pin_root_host,runner.pin_replay_host', 'runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer'),
    ('candidate_flags=[False,False]', 'candidate_flags=[False,True,False]'),
    ('assert flags==(True,True)', 'assert flags==(False,True,True)'),
    ('runner.pin_root_host=False', 'runner.offload_replay_mixer=False'),
    ('                        runner.pin_replay_host=False\n', ''),
    ('int(count[0]>=3)', 'validation_calls[0]'),
):
    assert old in code, old
    code = code.replace(old, new)
code = code.replace('                        candidate=native_attribute',
                    '                        validation_calls[0]+=1\n                        candidate=native_attribute')
anchor = "exec(compile(code, str(owner), 'exec'), globals())"
assert code.count(anchor) == 1
guard = '''                pair_allowed=False
                if count[0]==2:
                    import torch
                    observed_ids=native_args[0] if native_args else native_kwargs['paired_ids']
                    small=(observed_ids.shape[1]<=3072 and returned[1]['peak_allocated']<=55*2**30
                        and (runner.pin_root_host,runner.pin_replay_host,runner.offload_replay_mixer)==(False,True,True))
                    eligible=torch.tensor(int(small),device=observed_ids.device,dtype=torch.int32)
                    torch.distributed.all_reduce(eligible,op=torch.distributed.ReduceOp.MIN)
                    pair_allowed=bool(eligible.item())
                pair_info=None'''
extra = ("assert code.count('            count=[0]')==1\n"
         "code=code.replace('            count=[0]','            count=[0]\\n            validation_calls=[0]')\n"
         "assert code.count('                pair_info=None')==1\n"
         "code=code.replace('                pair_info=None'," + repr(guard) + ")\n"
         "code=code.replace('                if count[0]==2:', '                if count[0]==2 and pair_allowed:')\n")
# Restrict the pair branch only: the eligibility branch must still execute.
extra = extra.replace("code=code.replace('                if count[0]==2:', '                if count[0]==2 and pair_allowed:')",
    "code=code.replace('                if count[0]==2:\\n                    import torch\\n                    flags=', '                if count[0]==2 and pair_allowed:\\n                    import torch\\n                    flags=')")
code = code.replace(anchor, extra + anchor)
exec(compile(code, str(owner), 'exec'), globals())
