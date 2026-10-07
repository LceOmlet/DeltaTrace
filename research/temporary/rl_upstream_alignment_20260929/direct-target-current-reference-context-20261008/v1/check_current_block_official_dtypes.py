"""Feed the actual first single-intervention GDN block to the original dtype test.

Only select/concatenate saved factual operands and call the existing diagnostic.
The original reference, finite implementation, thresholds and assertions own
all numerical computations. No model, full DT, reward or optimizer is invoked.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/current-extreme-official-block-dtype-20261008-v1'
SOURCE = ROOT/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
PARENT = ROOT/'receipts/direct-target-existing-pv-rule-20261008-v1-gdn-v2'
HARNESS = ROOT/'receipts/upstream-alignment-20260929/verify_saved_fla_dtypes.py'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(part)
    return h.hexdigest()


assert sha(SOURCE) == '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source = json.loads(SOURCE.read_bytes())
launch = json.loads((PARENT/'launch.json').read_bytes())
os.environ.update(source['environment'])
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ.pop('MACA_VISIBLE_DEVICES',None)
candidate = launch['memory_candidate']
dt_root = Path(candidate['candidate_dt_root'])
env = json.loads(Path(candidate['candidate_environment']).read_bytes())['qwen35']
sys.path[:0] = [str(HARNESS.parent),str(dt_root),
    os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],str(dt_root/'clean/qwen35'),
    *source['pythonpath'].split(':'),env['ft_extension_root']]
import psutil
import torch

OUT.mkdir(exist_ok=True)
assert not (OUT/'prepared-operands.pt').exists()
record = dict(pid=os.getpid(),birth=psutil.Process().create_time(),phase='prepare_actual_block',
    unix=time.time(),source_sha256=sha(SOURCE),harness=dict(path=str(HARNESS),sha256=sha(HARNESS)),
    source_artifacts=[],scope=__doc__,model=0,full_DT=0,optimizer=0,rollout=0)


def save():
    record.update(unix=time.time(),pss_bytes=psutil.Process().memory_full_info().pss)
    (OUT/'phase.json').write_text(json.dumps(record,indent=2)+'\n')


save()
native = json.loads((PARENT/'results/rank0.json').read_bytes())['gdn_subops']['30']
pieces = []
upstreams = []
for artifact in native['operand_artifacts']:
    path = Path(artifact['path'])
    assert sha(path) == artifact['sha256']
    saved = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    offset = saved['actual_single_time_start']-saved['time_start']
    assert offset >= 0 and offset % 64 == 0
    ep = saved['endpoints']
    part = {}
    for name in ('q','k','v','beta','raw_g'):
        factual = ep[name][1:2,offset:offset+64].clone()
        actual = saved['actual_single']['g' if name=='raw_g' else name][1:2,:64]
        assert torch.equal(factual,actual) and factual.shape[1] == 64
        part[name] = factual.repeat_interleave(2,0)
    part['h'] = ep['h'][1:2,offset//64:offset//64+1].clone().repeat_interleave(2,0)
    pieces.append(part)
    upstreams.append(saved['do'][:,offset:offset+64].clone())
    if len(pieces)==1:
        scale = saved['scale']
        record['absolute_block_start'] = saved['actual_single_time_start']
    assert scale == saved['scale']
    record['source_artifacts'].append(dict(**artifact,head_start=saved['head_start'],
                                          actual_suffix_offset=offset))
    save()
    del ep,saved,part
combined = {name:torch.cat([piece[name] for piece in pieces],dim=2)
            for name in pieces[0]}
upstream = torch.cat(upstreams,dim=2)
prepared = OUT/'prepared-operands.pt'
torch.save(dict(endpoints=combined,upstream=upstream,scale=scale),prepared)
assert not torch.cuda.is_initialized()
record.update(phase='original_official_dtype_harness',prepared=dict(path=str(prepared),
    sha256=sha(prepared)),shape=list(combined['q'].shape),upstream_shape=list(upstream.shape),
    initial_state_nonzero=bool(combined['h'].count_nonzero()),cuda_initialized_before_original=False)
save()
sys.argv = [str(HARNESS),'--operands',str(prepared),'--sources',
            str(ROOT/'receipts/training-setup/official-kernel-tests'),
            '--output',str(OUT/'official-result.json')]
try:
    runpy.run_path(str(HARNESS),run_name='__main__')
except BaseException:
    import traceback
    record.update(phase='failed',traceback=traceback.format_exc())
    save()
    raise
record.update(phase='complete',torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
save()
