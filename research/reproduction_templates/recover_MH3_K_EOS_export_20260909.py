"""Recover a missing EOS array from existing actual tensors; zero model/GPU work.

The original numeric export reused K_raw_x0 for B1 step0, overwriting the
EOS export only. The experiment's computations used separate tensors and
are unchanged. Preserve original artifacts and record this separate recovery.
"""
import os,time,json,hashlib
from pathlib import Path
os.environ['MACA_PATH']='/opt/maca'
started=time.perf_counter();D=Path('/tmp/codex_dt_MH3_native_K_boundaries_20260909_v1')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
result=json.loads((D/'results.json').read_bytes());private=result['cases']['morehopqa_3']['private_artifact'];path=D/private['file']
assert sha(path)==private['sha256']
import torch,numpy as np
torch.set_num_threads(4)
k=torch.load(path,map_location='cpu',mmap=True,weights_only=True)['K'];B,T,H,F=k['mk'].shape
x=k['raw_k'][0:1].float().reshape(B,T,H,-1,F);assert torch.equal(x,x[:,:,:,:1].expand_as(x))
target=D/'recovered_EOS_K.npz';assert not target.exists();np.savez_compressed(target,x0=x[:,:,:,0].numpy())
assert not torch.cuda.is_initialized()
receipt={'status':'EOS_export_recovered_from_unchanged_actual_private_tensor','results_sha256':sha(D/'results.json'),
    'private_sha256':private['sha256'],'script_sha256':sha(Path(__file__)),'output_sha256':sha(target),
    'output_bytes':target.stat().st_size,'model_calls':0,'GPU_calls':0,'seconds':time.perf_counter()-started}
(D/'EOS_export_recovery_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
