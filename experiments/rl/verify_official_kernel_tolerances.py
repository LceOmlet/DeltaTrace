"""Call pinned FA/FLA tests unchanged; do not implement another tolerance.

The selected BF16 shapes exercise full/variable-length FA and Qwen's GDN head
dimension. This records operator checks, not a whole-PPO error certificate.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import logging
import time
import traceback
from pathlib import Path

import torch
import fla.utils


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    expected = {
        'test_flash_attn_v263.py': 'a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9',
        'test_gated_delta_v041.py': '35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813',
        'fla_utils_v041.py': '03e3e41a62741d45913a2d2d35de630aaba13dcbb0fb1e8c1f7904446a688cf1',
    }
    for name, sha in expected.items():
        assert hashlib.sha256((args.sources/name).read_bytes()).hexdigest() == sha
    # Keep the installed owner's actual comparison function, checking it is
    # the pinned function. Never enable its CI warning-only escape hatch.
    assert not fla.utils.FLA_CI_ENV
    original = ast.parse((args.sources/'fla_utils_v041.py').read_text())
    for name in ('get_abs_err', 'get_err_ratio', 'assert_close'):
        ref = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        active = ast.parse(inspect.getsource(getattr(fla.utils, name))).body[0]
        assert ast.dump(active) == ast.dump(ref), name
    fa = load('official_fa_v263_test', args.sources/'test_flash_attn_v263.py')
    fla_test = load('official_fla_v041_test', args.sources/'test_gated_delta_v041.py')
    result = dict(scope=__doc__, source_sha256=expected, cases=[],
                  source_manifest=json.loads((args.sources/'sources.json').read_text()),
                  fla_ci_warning_exemption=False)
    for length in (128, 447):
        fa_args = dict(seqlen_q=length, seqlen_k=length, d=256, dropout_p=0.0,
                       causal=True, local=False, alibi=False, deterministic=True,
                       mha_type='gqa', dtype=torch.bfloat16, kvpacked=False, softcap=0.0)
        fla_args = dict(B=1, T=length, H=32, D=128, scale=128**-0.5,
                        gate_logit_normalizer=1.0, mask_p=0.0,
                        use_qk_l2norm_in_kernel=True, dtype=torch.bfloat16)
        for fn, params in ((fa.test_flash_attn_output, fa_args),
                           (fa.test_flash_attn_varlen_output, fa_args),
                           (fla_test.test_chunk, fla_args)):
            entry = dict(function=fn.__name__, parameters={k:str(v) if isinstance(v, torch.dtype) else v
                                                         for k,v in params.items()})
            start = time.perf_counter()
            print('[official test]', entry, flush=True)
            try:
                fn(**params)
                torch.cuda.synchronize()
                entry['status'] = 'passed'
            except Exception as exc:
                entry.update(status='failed', error=str(exc), traceback=traceback.format_exc())
                traceback.print_exc()
            entry['seconds'] = time.perf_counter()-start
            result['cases'].append(entry)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    result['status'] = 'passed' if all(x['status'] == 'passed' for x in result['cases']) else 'failed'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
