"""Account for fixed-set partial-deletion mismatch through unchanged native DT."""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'environment', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--phase', choices=['pilot', 'development16', 'remaining12'], default='pilot')
    a = p.parse_args()
    refs = [('/tmp/codex_clean_development16_20260909_v1/qwen35/results.json',
             '04c59aaee006b49bb1c93d5ded737805b17570c3517aa045370bd38d37226cdb'),
            ('/tmp/codex_clean_development16_20260909_mh_recovery_v1/qwen35/results.json',
             '5999968354f16fe3760a4e2f9bfae9401524d366648cfe7f11fe563339719467')]
    rows = []
    for path, digest in refs:
        raw = Path(path).read_bytes(); assert sha(raw) == digest
        rows.extend(r for r in json.loads(raw)['cases'] if r['status'] == 'complete')
    rows = sorted([r for r in rows if a.phase == 'development16' or ((r['index'] < 2) == (a.phase == 'pilot'))], key=lambda r: (r['dataset'], r['index']))
    assert len(rows) == {'pilot':4, 'development16':16, 'remaining12':12}[a.phase]
    env = json.loads(a.environment.read_bytes())['qwen35']
    os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0')
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/deltatrace_clean_v1_inductor')
    for d in env.get('dependency_overlays', []): sys.path.insert(0, d)
    for d in ('deltatrace/clean/qwen35', 'deltatrace/accelerated'): sys.path.insert(0, str(a.release / d))
    import numpy as np
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from deferred import make_deferred_qwen35
    from qwen35_answer_finite import PackedAnswerTargets
    from native_target_logit_rows import NativeTargetLogitRows
    from finite_fla_gpu import verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    torch.set_num_threads(4); torch.manual_seed(73); torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = max(128, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_cache_size_limit = max(512, torch._dynamo.config.accumulated_cache_size_limit)
    a.output.mkdir(exist_ok=False)
    report = {'status': 'loading', 'script_sha256': sha(Path(__file__).read_bytes()), 'phase': a.phase,
              'references': refs, 'cases': [], 'calls': [], 'generation_calls': 0, 'FT_calls': 0, 'metric_calls': 0}
    vectors = {}

    def save():
        q = a.output / 'results.partial'; q.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); q.replace(a.output / 'results.json')

    def timed(name, fn):
        report['status'] = name; save(); torch.cuda.synchronize(); tick = time.perf_counter()
        value = fn(); torch.cuda.synchronize()
        report['calls'].append({'name': name, 'seconds': time.perf_counter() - tick}); save(); return value

    try:
        model = timed('load', lambda: Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'], dtype=torch.bfloat16,
                      attn_implementation='eager', device_map={'': 'cuda:0'}, local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes()) == env['native_model_sha256']
        identity = {n: (type(m).forward, m.forward) for n, m in model.named_modules()}
        with torch.no_grad():
            initial = torch.tensor(rows[0]['input_ids'], device='cuda')[None]
            timed('native_initialization', lambda: model(input_ids=initial, attention_mask=torch.ones_like(initial), use_cache=False))
            del initial
        model.set_attn_implementation('flash_attention_2'); verify_native_sources(env['native_stage_source_sha256'])
        fa = VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256'])
        runner, report['baseline_sources'] = make_deferred_qwen35(a.release, model, fa, None)
        layers = model.model.language_model.layers
        norm = model.model.language_model.norm
        for row in rows:
            key = f"{row['dataset']}_{row['index']}"; n = len(row['input_ids'])
            ids = torch.tensor(row['input_ids'], device='cuda')[None]
            assert sha(ids.cpu().numpy().tobytes()) == row['input_sha256']
            pair = ids.repeat(2, 1); pair[0, [row['user_positions'][j] for j in row['keep']]] = int(ids[0, -1])
            mask = torch.ones_like(pair)
            selection = PackedAnswerTargets([{'target_ids': ids[0, row['prompt_length']:].cpu(), 'prompt_length': row['prompt_length']}],
                                            [list(range(row['target_length']))], n, 'cuda')
            selector = NativeTargetLogitRows(selection)
            plain, plain_detail = timed(key + '/plain', lambda: runner.attribute(pair, mask, selection))
            vectors[key + '/plain'] = plain.numpy()
            partials = {}; partial_logprobs = {}; deleted_sets = {}; input_hashes = {}

            def snapshot(method):
                modified = pair.clone(); deleted = row['metrics'][method]['deleted_user_indices'][2]
                assert len(deleted) == len(row['metrics']['DT']['deleted_user_indices'][2])
                modified[1, [row['user_positions'][j] for j in deleted]] = int(ids[0, -1])
                digest = sha(modified[1].cpu().numpy().tobytes())
                assert digest == row['metrics'][method]['actual_input_hashes'][2]
                deleted_sets[method] = deleted; input_hashes[method] = digest
                values = {}; handles = []

                def retain(name, tensor):
                    assert name not in values
                    values[name] = tensor[1:2].detach().to('cpu', copy=True)

                for i, layer in enumerate(layers):
                    for name, module in [('input_norm', layer.input_layernorm), ('post_norm', layer.post_attention_layernorm)]:
                        def hook(_m, args, output, i=i, name=name):
                            retain(f'{i}/{name}_input', args[0]); retain(f'{i}/{name}_output', output)
                        handles.append(module.register_forward_hook(hook))
                    for name, module in [('mixer', layer.self_attn if layer.block_type == 'full_attention' else layer.linear_attn), ('mlp', layer.mlp), ('out', layer)]:
                        def hook(_m, _args, output, i=i, name=name):
                            retain(f'{i}/{name}', output[0] if isinstance(output, tuple) else output)
                        handles.append(module.register_forward_hook(hook))
                def final(_m, args, output):
                    retain('final_norm_input', args[0]); retain('final_norm_output', output)
                handles.append(norm.register_forward_hook(final))
                try:
                    with torch.no_grad():
                        out = model(input_ids=modified, attention_mask=mask, use_cache=False, logits_to_keep=selector.rows)
                        z = selector.pack_logits(out.logits)
                        lp = z.float().log_softmax(-1).gather(-1, selection.labels.repeat_interleave(2)[:, None]).squeeze(-1)
                        assert lp[0::2].cpu().tolist() == plain_detail['target_logp0']
                        partial_logprobs[method] = lp[1::2].cpu().tolist()
                        del out, z, lp
                finally:
                    for handle in handles: handle.remove()
                assert len(values) == 32 * 7 + 2
                return values

            for method in ('DT', 'FT_K1'): partials[method] = timed(key + '/partial/' + method, lambda method=method: snapshot(method))

            class Observer:
                def __init__(self): self.boundaries = {}; self.layers = []
                def wants_decoder(self, index): return True
                def dot(self, coefficient, original, partial):
                    return float((coefficient.double() * (original[1::2].double() - partial.to('cuda').double())).sum())
                def boundary(self, label, m, values):
                    name = 'final_norm_output' if label == 'norm' else 'final_norm_input' if label == '32' else label + '/input_norm_input'
                    self.boundaries[label] = {method: self.dot(m, values, d[name]) for method, d in partials.items()}
                def decoder(self, index, d, c, e, mout, minput, terms):
                    rec = {'layer': index, 'mixer': layers[index].block_type, 'methods': {}}
                    for method, partial in partials.items():
                        def dot(m, actual, name): return self.dot(m, actual, partial[f'{index}/' + name])
                        mmid = terms['m_mixer_output']; ma = terms['m_mixer_input']; mn = terms['m_mlp_norm_output']
                        pin = dot(minput, d['input_norm_input'], 'input_norm_input')
                        residual1 = dot(mmid, d['input_norm_input'], 'input_norm_input')
                        norm1 = dot(ma, d['input_norm_output'], 'input_norm_output')
                        mix = dot(mmid, c['output'], 'mixer')
                        mid = dot(mmid, d['post_norm_input'], 'post_norm_input')
                        residual2 = dot(mout, d['post_norm_input'], 'post_norm_input')
                        norm2 = dot(mn, d['post_norm_output'], 'post_norm_output')
                        mlp = dot(mout, d['mlp_output'], 'mlp')
                        pout = dot(mout, d['output'], 'out')
                        errors = {'input_norm': pin - residual1 - norm1, 'mixer': norm1 - mix,
                                  'residual1_rounding': residual1 + mix - mid, 'post_norm': mid - residual2 - norm2,
                                  'MLP': norm2 - mlp, 'residual2_rounding': residual2 + mlp - pout}
                        assert abs(sum(errors.values()) - (pin - pout)) < 1e-7 * max(1., abs(pin), abs(pout))
                        rec['methods'][method] = {'input_prediction': pin, 'output_prediction': pout, 'errors': errors}
                    self.layers.append(rec)

            obs = Observer()
            observed, detail = timed(key + '/observed', lambda: runner.attribute(pair, mask, selection, observer=obs))
            assert torch.equal(plain, observed)
            assert detail['target_logp0'] == plain_detail['target_logp0'] and detail['target_logp1'] == plain_detail['target_logp1']
            vectors[key + '/observed'] = observed.numpy()
            record = {'case': key, 'input_sha256': row['input_sha256'], 'deleted_sets': deleted_sets,
                      'partial_input_hashes': input_hashes, 'plain_observed_vector_equal': True,
                      'full_target_logprobs': plain_detail['target_logp1'], 'partial_target_logprobs': partial_logprobs,
                      'boundaries': obs.boundaries, 'layers': obs.layers, 'methods': {}}
            for method in ('DT', 'FT_K1'):
                actual = sum(float(x) - float(y) for x, y in zip(plain_detail['target_logp1'], partial_logprobs[method]))
                mass = float(plain[0, [row['user_positions'][j] for j in deleted_sets[method]]].sum())
                groups = {'head_seed': obs.boundaries['norm'][method] - actual,
                          'final_norm': obs.boundaries['32'][method] - obs.boundaries['norm'][method]}
                for layer in obs.layers:
                    replay_gap = layer['methods'][method]['output_prediction'] - obs.boundaries[str(layer['layer'] + 1)][method]
                    layer['methods'][method]['native_replay_boundary_gap'] = replay_gap
                    groups['native_replay_boundary'] = groups.get('native_replay_boundary', 0.) + replay_gap
                    for name, value in layer['methods'][method]['errors'].items():
                        family = layer['mixer'] if name == 'mixer' else name
                        groups[family] = groups.get(family, 0.) + value
                closing = sum(groups.values()) - (mass - actual)
                assert abs(obs.boundaries['0'][method] - mass) < 1e-7 * max(1., abs(mass))
                assert abs(closing) < 1e-6 * max(1., abs(mass), abs(actual)), closing
                record['methods'][method] = {'allocated_mass': mass, 'actual_drop': actual, 'prediction_error': mass - actual,
                                             'signed_family_errors': groups, 'telescope_closure': closing}
            report['cases'].append(record)
            for name, module in model.named_modules(): assert (type(module).forward, module.forward) == identity[name]
            np.savez_compressed(a.output / 'vectors.npz', **vectors); save()
            del partials, plain, observed, obs, detail, pair, ids, mask; gc.collect()
        report['status'] = 'complete'; report['vectors_sha256'] = sha((a.output / 'vectors.npz').read_bytes())
    except Exception:
        report['status'] = 'failed'; report['error'] = traceback.format_exc(); raise
    finally: save()


if __name__ == '__main__':
    main()
