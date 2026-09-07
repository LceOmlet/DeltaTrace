"""Classify the already recorded finite-P1 kernels; no model calls."""
import collections
import hashlib
import json
from pathlib import Path
A = Path(__file__).resolve().parent
folder = A/'snapshot${ARTIFACT_ROOT}/codex_finite_FA_FT_cost16_20260907_v1'
raw = (folder/'results.json').read_bytes()
receipt = json.loads((A/'finite_FA_FT_cost16_summary_20260907.json').read_text())
assert hashlib.sha256(raw).hexdigest() == receipt['raw_sha256']
r = json.loads(raw)
profile = r['records'][0]['profiles']['finite']
path = folder/profile['trace']
assert hashlib.sha256(path.read_bytes()).hexdigest() == profile['sha256']
events = json.loads(path.read_text())['traceEvents']
scopes = [e for e in events if e.get('name') == 'ATTR_NATIVE_HALF_LINEAR'
          and e.get('ph') == 'X' and e.get('cat') == 'user_annotation']
assert len(scopes) == 253
first_manual = min(e['ts'] for e in scopes)
cpu = {e.get('args', {}).get('External id'):e for e in events
       if e.get('cat') == 'cpu_op' and e.get('args', {}).get('External id') is not None}
groups = collections.defaultdict(lambda: {'calls':0, 'kernel_seconds':0.0})
unassigned = []
for e in events:
    if e.get('cat') != 'kernel': continue
    name = e['name']; parent = cpu.get(e.get('args', {}).get('External id'))
    if 'gemm' in name.lower():
        if parent is None:
            key = 'GEMM_without_CPU_correlation'; unassigned.append(name)
        elif any(s['ts'] <= parent['ts'] and parent['ts']+parent.get('dur',0) <= s['ts']+s['dur']+1e-3 for s in scopes):
            key = 'finite_propagation_native_half_GEMM'
        elif parent['ts'] < first_manual:
            key = 'actual_root_model_GEMM_including_head'
        else:
            key = 'actual_decoder_replay_GEMM'
    elif 'flash_fwd_kernel' in name:
        key = 'native_FA_forward_including_auxiliary'
    elif 'deltatrace' in name.lower() or 'finite' in name.lower():
        key = 'finite_FA_extension'
    else:
        key = 'other_kernels'
    groups[key]['calls'] += 1
    groups[key]['kernel_seconds'] += e['dur']/1e6
total = sum(x['kernel_seconds'] for x in groups.values())
for value in groups.values(): value['fraction_of_summed_kernel_duration'] = value['kernel_seconds']/total
out = {'status':'analyzed_existing_trace', 'root_forwards':0, 'VJPs':0,
    'profile_sha256':profile['sha256'], 'parent_raw_sha256':receipt['raw_sha256'],
    'scope':'One already recorded original NI0 finite-P1 profile, not representative timing or wall-time decomposition. Summed GPU kernel durations cannot be directly subtracted from latency.',
    'partition':dict(groups), 'total_summed_kernel_seconds':total,
    'unassigned_gemm_names':list(set(unassigned)),
    'next_candidate':'Use unmodified torch.utils.checkpoint.create_selective_checkpoint_contexts per native MLP region, saving only actual native mm outputs. Reuse those exact same-endpoint activations during existing replay, validate MLP inputs/weights/boundaries, free each consumed region. Keep model/FA and finite-P1 math unchanged. Fixed per-layer regions prevent reverse-layer FIFO mixups.',
    'not_a_claim':'Cache speed/memory and numerical equivalence are not established until the two-example bounded screen. No new counterfactual or backward is represented by a cached result.'}
(A/'existing_finite_profile_reuse_analysis_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
