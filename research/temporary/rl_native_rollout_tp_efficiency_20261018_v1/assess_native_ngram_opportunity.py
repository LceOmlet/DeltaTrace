"""CPU opportunity count using the installed official ngram proposer and saved IDs."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from vllm.v1.spec_decode import ngram_proposer

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/native-rollout-TP-efficiency-20261018-v1'
source = ROOT/'receipts/textcraft-native-dp-owner-20261018-v1/native-benchmark/0-A-global.pt'
packet = torch.load(source, map_location='cpu', weights_only=False)
tokens = forwards = saved = spans = 0
started = time.time()
for row in range(32):
    artifact = packet.non_tensor_batch['dt_direct_target_artifact'][row]
    ids = np.asarray(artifact['prompt_ids'] + artifact['response_ids'], dtype=np.int32)
    mask = artifact['policy_mask']
    offset = len(artifact['prompt_ids'])
    starts = [i for i, value in enumerate(mask) if value and (i == 0 or not mask[i-1])]
    for first in starts:
        end = first + 1
        while end < len(mask) and mask[end]:
            end += 1
        position, stop = offset + first, offset + end
        tokens += stop-position
        spans += 1
        while position < stop:
            forwards += 1
            position += 1
            if position >= stop:
                continue
            draft = ngram_proposer._find_longest_matched_ngram_and_propose_tokens(
                origin_tokens=ids[:position], min_ngram=5, max_ngram=5,
                max_model_len=32768, k=5)
            accepted = 0
            for value in draft:
                if position+accepted >= stop or int(value) != int(ids[position+accepted]):
                    break
                accepted += 1
            saved += accepted
            position += accepted
owner = Path(ngram_proposer.__file__)
result = dict(status='CPU_recorded_token_opportunity_not_engine_acceptance_or_speed',
    original_native_data=str(source), original_native_data_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
    actual_native_rows=32, actual_assistant_spans=spans, actual_policy_tokens=tokens,
    optimistic_target_forward_calls=forwards, matching_draft_tokens=saved,
    optimistic_forward_call_ratio=tokens/forwards, seconds=time.time()-started,
    original_proposer_file=str(owner), original_proposer_SHA256=hashlib.sha256(owner.read_bytes()).hexdigest(),
    policy_sampling_or_optimizer_changes=0, GPU_or_model_forward_calls=0,
    scope='Recorded continuation match only; excludes engine acceptance, verification overhead and wall time')
(AUDIT/'ngram-recorded-opportunity-CPU-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
