"""Replay saved native operands with original PyTorch/FLA operations.

This is a numerical diagnostic, not a training implementation or a whole-model
acceptance test. It does not load the full model or change the training runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
import fla.utils
from verify_official_kernel_tolerances import load


def pair_delta(x):
    return (x[0::2].float() - x[1::2].float()).abs().flatten(1).amax(1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operands', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(8)
    saved = torch.load(args.operands, map_location='cpu', weights_only=True)
    values, endpoints = saved['values'], saved['endpoints']
    # The diagnostic capture is the untrained actor's layer 2; its LoRA B is
    # initialized to zero. Read only this original base projection's weight.
    key = 'model.language_model.layers.2.linear_attn.in_proj_a.weight'
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())
    with safe_open(args.model / index['weight_map'][key], framework='pt', device='cpu') as f:
        weight = f.get_tensor(key)
    x = values['input']
    with torch.no_grad():
        reference = F.linear(x.double(), weight.double())
        replay = F.linear(x.cuda(), weight.cuda()).cpu()
        torch.backends.cuda.matmul.allow_tf32 = False
        fp32 = F.linear(x.cuda().float(), weight.cuda().float()).cpu()
    result = dict(scope=__doc__, operands=str(args.operands), weight=key,
                  input_shape=list(x.shape), input_pair_delta=pair_delta(x),
                  quantities={}, unequal_coordinates=[])
    for name, actual in [('captured_bf16', values['a']), ('replayed_bf16', replay), ('fp32', fp32)]:
        error = actual.double() - reference
        result['quantities'][name] = dict(pair_delta=pair_delta(actual),
            max_abs_vs_cpu_fp64=float(error.abs().max()),
            rms_vs_cpu_fp64=float(error.square().mean().sqrt()),
            max_abs_vs_captured=float((actual.float() - values['a'].float()).abs().max()))
    for pair, token, channel in torch.nonzero(values['a'][0::2] != values['a'][1::2]).tolist():
        result['unequal_coordinates'].append(dict(pair=pair, token=token, channel=channel,
            captured=values['a'][2*pair:2*pair+2, token, channel].float().tolist(),
            reference=reference[2*pair:2*pair+2, token, channel].tolist(),
            reference_rounded_bf16=reference[2*pair:2*pair+2, token, channel].bfloat16().float().tolist()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)

    # Call the pinned owner's reference and its own assertion, unchanged.
    source = args.sources / 'test_gated_delta_v041.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
    assert not fla.utils.FLA_CI_ENV
    owner = load('official_gdn_reference', source)
    with torch.no_grad():
        expected, _ = owner.recurrent_gated_delta_rule_ref(
            q=F.normalize(values['raw_q'].cuda(), p=2, dim=-1),
            k=F.normalize(values['raw_k'].cuda(), p=2, dim=-1),
            v=endpoints['v'].cuda(), beta=endpoints['beta'].cuda(),
            g=endpoints['raw_g'].cuda(), scale=saved['scale'])
        actual = endpoints['o'].cuda()
        check = dict(scope='Saved native FLA output on its actual operands, forward only.',
                     rms_ratio=float(fla.utils.get_err_ratio(expected, actual)),
                     max_abs=float(fla.utils.get_abs_err(expected, actual)), threshold=0.005)
        try:
            fla.utils.assert_close('captured native GDN o', expected, actual, 0.005)
            check['status'] = 'passed'
        except AssertionError as exc:
            check.update(status='failed', error=str(exc))
    result.update(fla_forward=check, seconds=time.perf_counter()-started,
                  status='completed_operand_diagnostic_not_whole_model_acceptance')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
