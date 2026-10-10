"""Inspect real native head tensors using the pinned VERL reference functions.

One saved B4 head; original chunk width512. No model, DT, loss, optimizer,
new tolerance, or parameter update. FP32 owner assertions retain their own
scope; BF16 actual-output comparisons are descriptive, not a new tolerance.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time

import psutil
import torch

HEAD_SHA = 'e285c3353bddbffae38df346b44014ee5038b606531874ddbeb10ee1241f77de'
TEST_SHA = '2748279e66dc23678d4d9aeac81f4b0cde342f2c6f15a46679f3fed600136f1c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from verl.utils.experimental import torch_functional as head
    import verl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(inspect.getsourcefile(head))
    assert sha(source) == HEAD_SHA
    test_path = Path(verl.__file__).resolve().parents[1]/'tests/kernels/test_linear_cross_entropy.py'
    assert sha(test_path) == TEST_SHA
    spec = importlib.util.spec_from_file_location('native_owner_head_reference', test_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    payload = torch.load(args.capture, map_location='cpu', weights_only=False)
    values = payload['tensors']
    metadata = payload['metadata']
    h = values['hidden_states'].flatten(0, 1)
    ids = values['input_ids'].flatten()
    old_logp = values['token_log_probs'].flatten()
    old_entropy = values['entropy'].flatten()
    weight = values['vocab_weights'].to('cuda')
    weight32 = weight.float()
    assert h.shape[0] == ids.numel() == old_logp.numel() == old_entropy.numel()
    assert metadata['autocast_dtype'] == 'torch.bfloat16'
    result = dict(started_unix=time.time(), scope=__doc__, capture=str(args.capture),
        capture_sha256=sha(args.capture), metadata=metadata,
        sources={str(source):sha(source),str(test_path):sha(test_path),__file__:sha(__file__)},
        shapes={k:list(v.shape) for k,v in values.items()},
        dtypes={k:str(v.dtype) for k,v in values.items()}, chunks=[],
        original_chunk_width=512, model_calls=0, DT_calls=0, optimizer_steps=0,
        official_forward_tolerance=dict(atol=1e-4, rtol=1e-4),
        official_reference_temperature=1.0, complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result.update(unix=time.time(), peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            pss_bytes=psutil.Process().memory_full_info().pss)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=True)+'\n')

    eager = inspect.unwrap(head._fused_linear_for_ppo_fwd)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for start in range(0, len(h), 512):
                end = min(start+512, len(h))
                hidden = h[start:end].to('cuda')
                labels = ids[start:end].to('cuda')
                point = dict(start=start, end=end)
                result['chunks'].append(point)
                with torch.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=metadata['autocast_enabled']):
                    actual = head._fused_linear_for_ppo_fwd(hidden, weight, labels, metadata['temperature'])
                    plain = eager(hidden, weight, labels, metadata['temperature'])
                torch.cuda.synchronize()
                for name, index, saved in [('log_prob',0,old_logp),('entropy',1,old_entropy)]:
                    a=actual[index].cpu(); b=plain[index].cpu(); original=saved[start:end]
                    point[name]=dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                        native_recompute_equal=torch.equal(a,original),
                        eager_equal=torch.equal(a,b),
                        max_abs_native_vs_saved=float((a-original).abs().max()),
                        max_abs_compiled_vs_eager=float((a-b).abs().max()))
                # The same owner's test explicitly converts H/W to FP32.
                # Import its reference functions and use its original asserts.
                hidden32=hidden.float().unsqueeze(0)
                labels2=labels.unsqueeze(0)
                torch_values=reference.run_torch_entropy(hidden32,weight32,labels2)
                verl_values=reference.run_verl_original_entropy(hidden32,weight32,labels2)
                fused_values=reference.run_verl_torch_fused_entropy(hidden32,weight32,labels2)
                point['official_FP32_max_abs']={}
                for name,index in [('log_prob',0),('entropy',1)]:
                    torch.testing.assert_close(torch_values[index],verl_values[index],atol=1e-4,rtol=1e-4)
                    torch.testing.assert_close(torch_values[index],fused_values[index],atol=1e-4,rtol=1e-4)
                    torch.testing.assert_close(verl_values[index],fused_values[index],atol=1e-4,rtol=1e-4)
                    point['official_FP32_max_abs'][name]=dict(
                        native_vs_torch=float((torch_values[index]-fused_values[index]).abs().max()),
                        verl_vs_torch=float((torch_values[index]-verl_values[index]).abs().max()))
                point['official_FP32_assertions_passed']=True
                save()
                del hidden,labels,hidden32,labels2,actual,plain,torch_values,verl_values,fused_values
        result.update(complete=True, seconds=time.perf_counter()-started)
    except BaseException:
        import traceback
        result.update(complete=False, traceback=traceback.format_exc(), seconds=time.perf_counter()-started)
        raise
    finally:
        save()
    print(json.dumps(dict(output=str(args.output), chunks=len(result['chunks']),
                          complete=result['complete'],seconds=result['seconds'])))


if __name__ == '__main__':
    main()
