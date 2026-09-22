"""Read saved owner runs and diagnose PPO effects using the original PPO function.

This is a CPU evidence reader, not a second objective or a training hook.
Direction/ratio/branch measurements have no newly invented pass threshold.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
from verl.trainer.ppo.core_algos import compute_policy_loss

from compare_short_owner_parity import tensors


def direction(actual, reference):
    x, y = torch.cat(tensors(actual)), torch.cat(tensors(reference))
    nx, ny = x.norm(), y.norm()
    both = (x != 0) & (y != 0)
    return dict(elements=x.numel(), actual_norm=float(nx), reference_norm=float(ny),
                cosine=float(torch.dot(x, y)/(nx*ny)) if nx > 0 and ny > 0 else None,
                relative_l2=float((x-y).norm()/ny) if ny > 0 else None,
                max_abs=float((x-y).abs().max()),
                norm_ratio=float(nx/ny) if ny > 0 else None,
                sign_disagreements_on_joint_nonzero=int(((x*y < 0) & both).sum()),
                jointly_nonzero=int(both.sum()))


def policy_observables(state, values, clip):
    mask = state['response_mask'].bool()
    forwards = torch.cat(values['policy_forward_log_probs'])
    assert forwards.shape[0] == 2*mask.shape[0], 'Expected the recorded two updates'
    outputs = []
    for log_probs in forwards.split(mask.shape[0]):
        log_probs = log_probs.detach().clone().requires_grad_(True)
        captured = {}
        def read_owner_return(frame, event, arg):
            if event == 'return' and frame.f_code is compute_policy_loss.__code__:
                for key in ('ratio', 'pg_losses1', 'pg_losses2'):
                    captured[key] = frame.f_locals[key].detach().clone()
        previous = sys.getprofile()
        try:
            sys.setprofile(read_owner_return)
            loss, clipfrac, kl, lower = compute_policy_loss(
                old_log_prob=values['old_log_probs'], log_prob=log_probs,
                advantages=state['dt_token_advantages'], response_mask=mask,
                cliprange=clip['clip_ratio'], cliprange_low=clip['clip_ratio_low'],
                cliprange_high=clip['clip_ratio_high'], clip_ratio_c=float('inf'),
                loss_agg_mode=clip['loss_agg_mode'])
        finally:
            sys.setprofile(previous)
        derivative, = torch.autograd.grad(loss, log_probs)
        # Read the branch from the owner's two already-computed alternatives.
        clipped = captured['pg_losses2'] > captured['pg_losses1']
        outputs.append(dict(ratio=captured['ratio'][mask], clipped=clipped[mask],
                            derivative=derivative[mask], loss=float(loss.detach()),
                            clipfrac=float(clipfrac.detach()), kl=float(kl.detach()), lower=float(lower.detach())))
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    state = torch.load(args.artifact, map_location='cpu', weights_only=True)
    receipt = json.loads(args.artifact.with_suffix('.json').read_text())
    # Older saved verifier receipts used the pinned config values below.
    # New runs also record those values directly from their composed config.
    clip = receipt.get('policy_loss_config', dict(clip_ratio=0.2, clip_ratio_low=0.2,
        clip_ratio_high=0.2, clip_ratio_c='inf', loss_agg_mode='token-mean'))
    assert str(clip['clip_ratio_c']) == 'inf'
    source = Path(inspect.getfile(compute_policy_loss))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt['ppo_core_sha256']
    result = dict(scope=__doc__, artifact=str(args.artifact),
                  artifact_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
                  owner_source=str(source), owner_sha256=receipt['ppo_core_sha256'],
                  policy_loss_config=clip, input_tokens=receipt['input_tokens'],
                  effective_input_tokens=receipt['effective_input_tokens'],
                  active_tokens=int(state['response_mask'].sum()), comparisons={})
    pairs = [(label, state, state[label]) for label in
             ('paired_owner', 'paired_owner_math', 'repeat_installed') if label in state]
    if 'paired_owner_math' in state:
        pairs.append(('owner_fa_vs_math', state['paired_owner'], state['paired_owner_math']))
    if 'paired_untrimmed_head' in state:
        pairs.append(('optimized_vs_untrimmed_head', state, state['paired_untrimmed_head']))
        pairs.append(('untrimmed_head_vs_owner', state['paired_untrimmed_head'], state['paired_owner']))
    for label, candidate, ref in pairs:
        actual = policy_observables(state, candidate, clip)
        expected = policy_observables(state, ref, clip)
        steps = []
        for i, (x, y) in enumerate(zip(actual, expected)):
            dx = {k:v.double() - (state['before'] if i == 0 else candidate[f'after_{i-1}'])[k].double()
                  for k,v in candidate[f'after_{i}'].items()}
            dy = {k:v.double() - (state['before'] if i == 0 else ref[f'after_{i-1}'])[k].double()
                  for k,v in ref[f'after_{i}'].items()}
            steps.append(dict(update=i,
                ratio_error=direction(x['ratio'], y['ratio']),
                actual_ratio_range=[float(x['ratio'].min()), float(x['ratio'].max())],
                reference_ratio_range=[float(y['ratio'].min()), float(y['ratio'].max())],
                ratio_abs_error_mean=float((x['ratio']-y['ratio']).abs().mean()),
                ratio_abs_error_p95=float(torch.quantile((x['ratio']-y['ratio']).abs().double(), 0.95)),
                clipped_branch_disagreements=int((x['clipped'] != y['clipped']).sum()),
                actual_clipped_tokens=int(x['clipped'].sum()),
                reference_clipped_tokens=int(y['clipped'].sum()),
                owner_loss_derivative=direction(x['derivative'], y['derivative']),
                raw_gradient=direction(candidate['raw_gradients'][i], ref['raw_gradients'][i]),
                incremental_parameter_update=direction(dx, dy),
                actual_owner_metrics={k:x[k] for k in ('loss','clipfrac','kl','lower')},
                reference_owner_metrics={k:y[k] for k in ('loss','clipfrac','kl','lower')}))
        result['comparisons'][label] = steps
    result['status'] = 'measured; no training-reliability certification or new tolerance gate'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name:[dict(update=s['update'], ratio_max_abs=s['ratio_error']['max_abs'],
        clipping_disagreements=s['clipped_branch_disagreements'], gradient_cosine=s['raw_gradient']['cosine'],
        update_cosine=s['incremental_parameter_update']['cosine']) for s in steps]
        for name,steps in result['comparisons'].items()}))


if __name__ == '__main__':
    main()
