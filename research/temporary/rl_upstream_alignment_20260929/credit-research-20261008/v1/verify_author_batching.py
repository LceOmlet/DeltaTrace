"""Exact callback/metric replay using saved original native scores, CPU only.

This checks the batching seam on one measured artifact, repeated in four slots.
It is not four independent examples or evidence of collection attribution
quality, model forward parity, official kernel tolerance, or a repaired method.
"""
import hashlib
import json
from pathlib import Path
import time


def verify(output,official_root):
    import torch
    from transformers import AutoTokenizer
    import ft_ifr_improve
    import inspect_author_collection as collection

    started=time.perf_counter()
    root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    old_path=root/'receipts/direct-target-action-author-curve-20261007-v1/results/rank0.json'
    old=json.loads(old_path.read_bytes())
    native_path=root/'receipts/direct-target-prefix-runtime-20261007-v1/appworld-first-dt/rank1-readout-native-batch-6.pt'
    assert hashlib.sha256(native_path.read_bytes()).hexdigest()==old['native_sha256']
    assert hashlib.sha256((Path(official_root)/'ft_ifr_improve.py').read_bytes()).hexdigest()=='583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1'
    native=torch.load(native_path,map_location='cpu',weights_only=False)
    row=next(r for r in native['rows'] if r['batch_row']==old['geometry']['candidate']['row'])
    assert str(row['traj_uid'])==old['geometry']['candidate']['traj_uid']
    positions=row['prompt_length']+row['prior'][row['suffix_positions']].nonzero().flatten()
    assert positions.tolist()==old['source_positions']
    signed=native['native_signed'][row['batch_row'],positions].float()
    tokenizer=AutoTokenizer.from_pretrained('/mnt/si0021787ci2/default/models/Qwen3.5-9B',local_files_only=True)
    calls=[];views=['signed_RISE','positive_MAS']
    def replay(ids,phase):
        step=int(phase.removeprefix('author_point_'));scores=[]
        for pair,value in enumerate(ids):
            point=old['views'][views[pair%2]]['score_points'][step]
            expected=row['selected'].clone()
            expected[point['changed_input_positions']]=tokenizer.eos_token_id
            assert torch.equal(value,expected)
            scores.append(point['logp'])
        calls.append(step)
        return torch.tensor(scores,dtype=torch.float64,device='cpu')
    actual=collection.evaluate_author_curves_batched(ft_ifr_improve.faithfulness_test_skip_tokens,
        [row]*4,tokenizer,[signed]*4,[positions.tolist()]*4,replay)
    differences={f'{slot}/{views[view]}':max(abs(a-b) for a,b in zip(result['author_return'],old['views'][views[view]]['author_return']))
                 for (slot,view),result in actual.items()}
    assert calls==list(range(21)) and all(v==0 for v in differences.values())
    assert not torch.cuda.is_initialized()
    receipt={'scope':__doc__,'seconds':time.perf_counter()-started,'callback_requests':168,'native_score_replay_batches':21,
      'model_calls':0,'DT_calls':0,'gradient_calls':0,'optimizer_steps':0,'cuda_initialized':False,
      'source_artifact':{'path':str(old_path),'sha256':hashlib.sha256(old_path.read_bytes()).hexdigest()},
      'native_artifact':{'path':str(native_path),'sha256':old['native_sha256']},
      'metric_return_max_absolute_differences':differences,'callback_IDs_exact':True}
    Path(output).write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
