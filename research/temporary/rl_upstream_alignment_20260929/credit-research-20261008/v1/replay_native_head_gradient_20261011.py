"""Call the actual VERL head with captured H/W/IDs and upstream gradients.

No backbone, DT, optimizer, policy-loss replacement or numerical acceptance
threshold. Replay the captured finite logp gradient; obtain entropy's gradient
by the existing VERL agg_loss with the actual native mask/coefficient. Check
the next head output after the native head backward, without FSDP or offload.
This does not reproduce the missing original failed H/W or FSDP stream state.
"""
import hashlib
import inspect
import json
from pathlib import Path
import resource
import time

import psutil
import torch


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-native-head-gradient-20261011-v1'
CAPTURE = ROOT/'receipts/textcraft-native-incident37-native-head-live-capture-20261011-v1/result'
SOURCE = ROOT/'runs/textcraft-formal-stable-20261009-v1/source.json'
HEAD_SHA = 'e285c3353bddbffae38df346b44014ee5038b606531874ddbeb10ee1241f77de'
CORE_SHA = 'fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def differences(a, b):
    a = a.detach().float().cpu(); b = b.detach().float().cpu()
    return dict(exact=torch.equal(a, b), nonfinite=int((~torch.isfinite(a-b)).sum()),
                max_abs=float((a-b).abs().max()), mean_abs=float((a-b).abs().mean()))


def main():
    from omegaconf import OmegaConf
    from verl.utils.experimental import torch_functional as head
    from verl.trainer.ppo import core_algos
    head_path = Path(inspect.getsourcefile(head)); core_path = Path(inspect.getsourcefile(core_algos))
    assert sha(head_path) == HEAD_SHA and sha(core_path) == CORE_SHA
    config_path = ROOT/'receipts/textcraft-native-incident37-runtime-metadata-20261011-v1/result/initialization-config.yaml'
    cfg = OmegaConf.load(config_path)
    alignment = json.loads((ROOT/'receipts/textcraft-native-label-alignment-20261011-v1/result.json').read_bytes())
    assert alignment['complete']
    actor = cfg.actor_rollout_ref.actor
    assert actor.entropy_coeff == .001 and actor.ppo_micro_batch_size_per_gpu == 4
    assert actor.ppo_mini_batch_size == 64
    accumulation = (actor.ppo_mini_batch_size//2)//actor.ppo_micro_batch_size_per_gpu
    assert accumulation == 8
    OUT.mkdir(exist_ok=True)
    report = dict(started_unix=time.time(), pid=psutil.Process().pid,
        birth=psutil.Process().create_time(), scope=__doc__, complete=False,
        sources={str(head_path):sha(head_path), str(core_path):sha(core_path),
                 str(config_path):sha(config_path), __file__:sha(__file__)},
        ranks=[], head_calls=0, model_calls=0, DT_calls=0, optimizer_calls=0,
        official_tolerance_tests=0, production_changes=0)

    def save(phase):
        report.update(phase=phase, unix=time.time(),
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        (OUT/'result.json').write_text(json.dumps(report, indent=2)+'\n')

    try:
        for rank in [0, 1]:
            capture_path = CAPTURE/f'rank{rank}-native-head-live-microbatch1.pt'
            loss_path = CAPTURE/f'rank{rank}-loss-inputs-microbatch1.pt'
            capture = torch.load(capture_path, map_location='cpu', weights_only=False)
            loss = torch.load(loss_path, map_location='cpu', weights_only=False)
            values = capture['tensors']; metadata = capture['metadata']
            h = values['hidden_states'].to('cuda').requires_grad_(True)
            w = values['vocab_weights'].to('cuda')
            ids = values['input_ids'].to('cuda')
            aligned = alignment['ranks'][rank]
            assert sha(capture_path) == aligned['capture']['sha256']
            width = h.shape[1]; offset = aligned['exact_unique_response_offset']
            visible = width-offset-1
            assert torch.equal(values['token_log_probs'][:, offset:-1], loss['log_prob'][:, :visible])
            assert bool((loss['log_prob_gradient'][:, visible:] == 0).all())
            grad_logp = torch.zeros((4, width), dtype=torch.float32, device='cuda')
            grad_logp[:, offset:-1] = loss['log_prob_gradient'][:, :visible].to('cuda')
            e = torch.zeros_like(loss['log_prob'], requires_grad=True)
            entropy_loss = core_algos.agg_loss(loss_mat=e, loss_mask=loss['response_mask'],
                                              loss_agg_mode=actor.loss_agg_mode)
            entropy_gradient, = torch.autograd.grad(-entropy_loss*actor.entropy_coeff/accumulation, e)
            assert bool((entropy_gradient[:, visible:] == 0).all())
            grad_entropy = torch.zeros_like(grad_logp)
            grad_entropy[:, offset:-1] = entropy_gradient[:, :visible].to('cuda')
            point = dict(rank=rank, capture=dict(path=str(capture_path),sha256=sha(capture_path)),
                loss_input=dict(path=str(loss_path),sha256=sha(loss_path)), response_offset=offset,
                entropy_coeff=actor.entropy_coeff, loss_agg_mode=actor.loss_agg_mode,
                gradient_accumulation=accumulation, head_shape=list(h.shape),
                input_dtypes=[str(x.dtype) for x in [h,w,ids]],
                input_requires_grad=[x.requires_grad for x in [h,w,ids]])
            report['ranks'].append(point); save('native_head_forward_begin')
            native = head.FusedLinearForPPO()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=metadata['autocast_enabled']):
                output = native(h, w, ids, metadata['temperature']); report['head_calls'] += 1
            point['before_backward_vs_saved'] = [differences(x,values[key]) for x,key in
                                                   zip(output,['token_log_probs','entropy'])]
            save('native_head_backward_begin')
            torch.autograd.backward(output, (grad_logp, grad_entropy))
            point['hidden_gradient_finite'] = bool(torch.isfinite(h.grad).all())
            point['H_unchanged_after_backward'] = torch.equal(h.detach().cpu(),values['hidden_states'])
            point['W_unchanged_after_backward'] = torch.equal(w.cpu(),values['vocab_weights'])
            point['IDs_unchanged_after_backward'] = torch.equal(ids.cpu(),values['input_ids'])
            save('native_head_after_backward_forward_begin')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                enabled=metadata['autocast_enabled']):
                after = native(h.detach(),w,ids,metadata['temperature']); report['head_calls'] += 1
            point['after_vs_before'] = [differences(x,y) for x,y in zip(after,output)]
            vectors_path=OUT/f'rank{rank}-head-replay-outputs.pt'
            torch.save(dict(before=[x.detach().cpu() for x in output],
                            after=[x.detach().cpu() for x in after]),vectors_path)
            point['vectors']=dict(path=str(vectors_path),bytes=vectors_path.stat().st_size,sha256=sha(vectors_path))
            save('rank_completed')
            del capture,loss,values,h,w,ids,output,after,grad_logp,grad_entropy,e,entropy_gradient,entropy_loss
        report['complete']=True
    except BaseException:
        import traceback
        report['traceback']=traceback.format_exc()
        raise
    finally:
        save('complete' if report['complete'] else 'failed')
    print(json.dumps(dict(complete=True,output=str(OUT/'result.json'))))


if __name__ == '__main__':
    main()
