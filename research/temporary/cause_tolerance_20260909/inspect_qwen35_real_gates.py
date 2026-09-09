"""Passive all-layer gate diagnostics on frozen author development16 inputs.

One unchanged DT call per case. Existing runner observer only reads its native
endpoint tensors and finite coefficients; it never changes coefficients or
model functions. Window survival is a diagnostic condition, not a benchmark
or a reconstructed model output. Metrics are not rerun for this observer.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'environment', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    base = Path('/tmp/codex_clean_development16_20260909_v1')
    refs = [(base/'qwen35/results.json', '04c59aaee006b49bb1c93d5ded737805b17570c3517aa045370bd38d37226cdb'),
            (Path('/tmp/codex_clean_development16_20260909_mh_recovery_v1/qwen35/results.json'),
             '5999968354f16fe3760a4e2f9bfae9401524d366648cfe7f11fe563339719467')]
    rows = []
    for path, digest in refs:
        assert sha(path.read_bytes()) == digest
        rows.extend(r for r in json.loads(path.read_bytes())['cases'] if r['status'] == 'complete')
    assert len(rows) == 16 and len({(r['dataset'],r['index']) for r in rows}) == 16
    rows.sort(key=lambda r:(r['dataset'], r['index']))
    env = json.loads(args.environment.read_bytes())['qwen35']
    sources = json.loads((args.release/'deltatrace/clean/sources.json').read_bytes())
    for name, record in sources['models']['qwen35']['files'].items():
        assert sha((args.release/name).read_bytes()) == record['sha256'], name
    os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/deltatrace_clean_v1_inductor')
    for path in env.get('dependency_overlays', []):
        sys.path.insert(0, path)
    sys.path.insert(0, str(args.release/'deltatrace/clean/qwen35'))
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from qwen35_clean_runner import make_qwen35_clean_runner
    from qwen35_answer_finite import PackedAnswerTargets
    from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    torch.set_num_threads(4)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit = max(256, torch._dynamo.config.accumulated_cache_size_limit)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'loading', 'script_sha256': sha(Path(__file__).read_bytes()),
              'references': [digest for _, digest in refs], 'cases': [], 'root_calls': 0,
              'generation_calls': 0, 'metric_calls': 0, 'sample_batch': 1, 'endpoint_batch': 2,
              'scope': 'passive unchanged finite propagation; not a repair or quality run',
              'lags': [1,2,4,8,16,32,64,128],
              'window_event': 'actual log survival >= -1 and baseline <= actual - log(10)'}
    arrays = {}

    def save():
        path = args.output/'results.partial'
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        path.replace(args.output/'results.json')

    try:
        tick = time.perf_counter()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'], dtype=torch.bfloat16,
                    attn_implementation='eager', device_map={'':'cuda:0'}, local_files_only=True)
        model.eval().requires_grad_(False)
        report['model_load_seconds'] = time.perf_counter()-tick
        assert sha(Path(inspect.getfile(type(model))).read_bytes()) == env['native_model_sha256']
        identities = {n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        tokenizer = AutoTokenizer.from_pretrained(env['checkpoint'], local_files_only=True)
        first = torch.tensor(rows[0]['input_ids'],device='cuda')[None]
        with torch.no_grad():
            initial = model(input_ids=first,attention_mask=torch.ones_like(first),use_cache=False)
        del initial, first
        report['root_calls'] += 1
        model.set_attn_implementation('flash_attention_2')
        verify_native_sources(env['native_stage_source_sha256'])
        runner = make_qwen35_clean_runner(model, VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256']),
                                         make_compiled_finite_pullback(reuse_scalar_products=False))
        layers = model.model.language_model.layers

        class Observer:
            def __init__(self, case, row):
                self.case = case
                self.positions = torch.tensor([row['user_positions'][k] for k in row['keep']], device='cuda')
                self.layers = []

            def wants_decoder(self, index):
                return layers[index].block_type == 'linear_attention'

            def boundary(self, *_args):
                pass

            def decoder(self, index, decoder, values, e, before, after, terms):
                coeff = terms['mixer']['coeff']
                terms_by_branch = {}
                rec = {'layer': index, 'branch_mass': {}, 'windows': {}}
                for name in ('q','k','v','beta','g'):
                    key = 'raw_g' if name == 'g' else name
                    delta = e[key][1::2].float()-e[key][0::2].float()
                    effect = coeff[name]*delta
                    token = effect.sum(-1) if effect.ndim == 4 else effect
                    terms_by_branch[name] = token
                    active = token[0,self.positions]
                    rec['branch_mass'][name] = {'signed_sum': float(active.double().sum()),
                        'positive_mass': float(active.double().clamp_min(0).sum()),
                        'negative_mass': float(active.double().clamp_max(0).sum())}
                    arrays[f'{self.case}_layer{index}_{name}_token'] = token[0].sum(-1).cpu().numpy()
                g0,g1 = e['raw_g'][0].float(),e['raw_g'][1].float()
                rec['raw_g_quantiles'] = {
                    label: torch.quantile(g[self.positions].flatten(), torch.tensor([0.,.1,.5,.9,1.],device='cuda')).cpu().tolist()
                    for label,g in (('baseline',g0),('actual',g1))}
                # Prefix sums are used only to classify short survival windows.
                # FP64 prevents cancellation from the long negative global prefix.
                c0 = torch.cat((torch.zeros_like(g0[:1],dtype=torch.float64),g0.double().cumsum(0)))
                c1 = torch.cat((torch.zeros_like(g1[:1],dtype=torch.float64),g1.double().cumsum(0)))
                gate_weight = terms_by_branch['g'][0].double().abs()
                for lag in report['lags']:
                    end = self.positions[self.positions >= lag-1]+1
                    a,b = c0[end]-c0[end-lag],c1[end]-c1[end-lag]
                    surviving = b >= -1
                    suppressed = surviving & ((a-b) <= -2.302585092994046)
                    enhanced = (a >= -1) & ((b-a) <= -2.302585092994046)
                    weight = gate_weight[end-1]
                    rec['windows'][str(lag)] = {'count': int(a.numel()),
                        'actual_surviving': int(surviving.sum()), 'baseline_suppressed': int(suppressed.sum()),
                        'actual_suppressed_reverse': int(enhanced.sum()),
                        'DT_gate_abs_total': float(weight.sum()),
                        'DT_gate_abs_baseline_suppressed': float(weight[suppressed].sum())}
                self.layers.append(rec)

        for row in rows:
            case = f"{row['dataset']}_{row['index']}"
            report['status'] = case
            save()
            ids = torch.tensor(row['input_ids'],device='cuda')[None]
            assert sha(ids.cpu().numpy().tobytes()) == row['input_sha256']
            pair = ids.repeat(2,1)
            pair[0,[row['user_positions'][k] for k in row['keep']]] = tokenizer.eos_token_id
            target = ids[0,row['prompt_length']:]
            selection = PackedAnswerTargets([{'target_ids':target.cpu(),'prompt_length':row['prompt_length']}],
                                           [list(range(len(target)))],len(row['input_ids']),'cuda')
            observer = Observer(case,row)
            vector, detail = runner.attribute(pair,torch.ones_like(pair),selection,observer=observer)
            report['root_calls'] += 1
            assert len(observer.layers) == 24
            assert sorted(r['layer'] for r in observer.layers) == [i for i,l in enumerate(layers) if l.block_type=='linear_attention']
            arrays[case+'_signed'] = vector[0].numpy()
            report['cases'].append({'case':case,'input_sha256':row['input_sha256'], 'layers':observer.layers,
                'root_effect':detail['root_effect'],'signed_sum':detail['signed_sum'],
                'diagnostic_attribution_seconds':detail['complete_attribution_seconds_with_diagnostics'],
                'peak_allocated':detail['peak_allocated']})
            for n,m in model.named_modules():
                assert (type(m).forward,m.forward) == identities[n], n
            np.savez_compressed(args.output/'vectors.npz', **arrays)
            save()
            print(json.dumps({'case':case,'layers':len(observer.layers),'seconds':detail['complete_attribution_seconds_with_diagnostics']}),flush=True)
        report['status'] = 'complete'
        report['vectors_sha256'] = sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
