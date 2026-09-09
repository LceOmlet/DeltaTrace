"""Audit a historical cross-model confound; do not change methods or rescore models.

The cross-view arithmetic keeps each original deletion path fixed. In particular,
signed densities on old clipped-score masks are NOT a fresh signed-order MAS run.
All original metrics are verified separately before this explanatory calculation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2]
E = R / 'evidence'
sources = {}
def read(name):
    p = E / name
    sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_bytes())

def auc(x):
    x = np.asarray(x, dtype=np.float64)
    return float((x.sum() - (x[0] + x[-1]) / 2) / (len(x) - 1))

def evaluate(response, density):
    penalty = np.abs(response - density)
    corrected = np.clip(response + penalty, 0, 1)
    span = corrected.max() - corrected.min()
    corrected = ((corrected - corrected.min()) / span if span else np.linspace(1, 0, len(corrected)))
    return {'RISE': auc(response), 'MAS': auc(corrected), 'alignment_AUC': auc(penalty),
            'raw_augmented_AUC': auc(response + penalty)}

def density_on_masks(w, keep, masks):
    total = w[keep].sum(dtype=np.float64)
    assert total > 0
    assert set(masks[-1]) == set(keep) and masks[0] == []
    return np.array([1 - w[np.asarray(mask, dtype=int)].sum(dtype=np.float64) / total for mask in masks])

def certify_signed_curve_reuse(w, keep, masks, raw, response):
    """A zero prefix-minimum response is absorbing in the unchanged MAS rule.

    Require STRICT signed-score separation at every reused prefix boundary,
    so neither Torch's tie ordering nor unknown negative-tail masks are assumed.
    Later raw scores are unknown; only their prefix-minimum response is known.
    """
    zeros = np.flatnonzero(response == 0)
    if not len(zeros):
        return {'certified': False, 'reason':'No zero normalized response in saved path'}
    stop = int(zeros[0]); universe = set(keep); prefix = []
    for step in range(1, stop+1):
        selected = set(masks[step]); remainder = universe-selected
        gap = float(w[list(selected)].min()-w[list(remainder)].max()) if remainder else None
        prefix.append({'step':step,'strict_score_gap':gap})
        if remainder and gap <= 0:
            return {'certified':False,'reason':'Signed-order prefix is different or tie-ambiguous before zero response',
                    'first_uncertified_step':step,'first_zero_response_step':stop,'prefix_checks':prefix}
    order = keep[np.argsort(-w[keep],kind='stable')]
    groups = np.array_split(order,20); signed_masks=[[]]; cumulative=[]
    for group in groups:
        cumulative += group.tolist(); signed_masks.append(list(cumulative))
    assert all(set(signed_masks[j])==set(masks[j]) for j in range(stop+1))
    signed_density = density_on_masks(w,keep,signed_masks)
    return {'certified':True,'first_zero_response_step':stop,'prefix_checks':prefix,
            'original_RISE_MAS_reconstructed':evaluate(response,signed_density),
            'signed_sorted_density':signed_density.tolist(),
            'scope':'All needed real-score prefixes have unique original signed-order masks. Thereafter original prefix-min response stays zero for every possible raw score. Endpoints unchanged. No raw tail score is invented; float64 arithmetic approximates original FP32 reductions.'}

def analyze(key, w, keep, masks, curve, original_view, metrics, ft, **metadata):
    keep = np.asarray(keep, dtype=int)
    assert len(set(keep)) == len(keep) and np.isfinite(w).all()
    assert len(masks) == len(curve) == 21
    seen = set()
    sizes = []
    for mask in masks:
        s = set(mask)
        assert seen <= s <= set(keep)
        sizes.append(len(s) - len(seen))
        seen = s
    assert sizes[1:] == [len(keep)//20 + int(i < len(keep)%20) for i in range(20)]
    raw = (np.asarray(curve, dtype=np.float64) - curve[-1]) / abs(curve[0] - curve[-1])
    response = np.minimum.accumulate(np.clip(raw, 0, 1))
    signed_density = density_on_masks(w, keep, masks)
    positive_density = density_on_masks(w.clip(min=0), keep, masks)
    view = {'signed': evaluate(response, signed_density), 'positive': evaluate(response, positive_density)}
    assert abs(view[original_view]['RISE'] - metrics['RISE']) < 1e-12
    error = abs(view[original_view]['MAS'] - metrics['MAS'])
    assert error < 2e-6, (key, error)
    x = w[keep]
    pos = float(x[x > 0].sum()); neg = float(-x[x < 0].sum()); net = pos - neg
    assert net > 0
    first = masks[1]
    return {'case': key, 'original_view': original_view, 'original_metrics': metrics,
        'fixed_FT': ft, 'original_metric_reconstruction_error': error,
        'mass': {'positive': pos, 'negative_magnitude': neg, 'net': net,
                 'negative_over_net': neg/net, 'absolute_over_net': (pos+neg)/net,
                 'negative_fraction_absolute': neg/(pos+neg), 'negative_tokens': int((x<0).sum()),
                 'eligible_tokens': len(keep)},
        'fixed_original_masks': {'signed_density': signed_density.tolist(), 'positive_density': positive_density.tolist(),
            'raw_response': raw.tolist(), 'monotone_response': response.tolist(),
            'signed': view['signed'], 'positive': view['positive'],
            'signed_minus_positive_MAS': view['signed']['MAS'] - view['positive']['MAS'],
            'signed_minus_positive_alignment_AUC': view['signed']['alignment_AUC'] - view['positive']['alignment_AUC'],
            'first_signed_allocated': float(w[first].sum()/net),
            'first_positive_allocated': float(w[first].clip(min=0).sum()/pos),
            'first_actual_deleted': float(1-raw[1]),
            'minimum_signed_density': float(signed_density.min()), 'minimum_raw_response': float(raw.min())},
        'signed_original_order_reuse':certify_signed_curve_reuse(w,keep,masks,raw,response) if original_view=='positive' else None,
        **metadata}

old = read('vendor_fa_development16_numeric_20260907.json')
cost = read('finite_FA_FT_cost16_numeric_20260907.json')
cost_by_key = {(r['dataset'],r['idx']):r for r in cost['records']}
rows = []
for row in old['records']:
    cr = cost_by_key[(row['dataset'],row['idx'])]
    run = next(r for r in cr['runs'] if r['mode']=='finite' and r['repeat']==1 and not r['warmup'])
    signed = np.asarray(run['signed_full_sequence'], dtype=np.float64)
    w = signed[row['user_positions']].astype(np.float32).astype(np.float64)
    score = np.asarray(row['scores']['finite'], dtype=np.float32)
    assert np.array_equal(w.clip(min=0).astype(np.float32), score)
    assert run['score'] == row['scores']['finite'] and cr['input_ids_sha256'] == row['input_ids_sha256']
    qm = row['metrics']['finite']
    assert cr['methods']['finite']['quality_metrics'] == qm
    assert cr['methods']['finite']['quality_source_sha256'] == old['raw_sha256']
    ft = cr['methods']['both_0']['quality_metrics']
    rows.append(analyze(row['dataset']+'_'+str(row['idx']), w, row['keep_local_indices'],
        row['evaluation_masks']['finite'], qm['raw_curve'], 'positive',
        {'RISE':qm['rise'],'MAS':qm['mas'],'needle':qm['recovery']},
        {'method':'historical_both0','RISE':ft['rise'],'MAS':ft['mas']},
        model='Qwen3-8B', input_sha256=row['input_ids_sha256'],
        provenance='Exact stored positive vector and quality-parent identity; original signed vector retained.'))

names = ['dt_GDN1_K_whole_pilot_summary_20260909.json'] + [f'dt_GDN1_K_remaining_summary_20260909_s{i}.json' for i in range(3)]
for name in names:
    summary = read(name)
    assert summary['status'] == 'independent_GDN1_K_whole_pilot_audit_passed'
    job = 'dt_GDN1_K_whole_pilot_20260909_v1' if 'whole_pilot' in name else f'dt_GDN1_K_remaining_20260909_s{name.split("_s")[-1].split(".")[0]}_v1'
    p = E / job / 'vectors.npz'
    sources[str(p.relative_to(E))] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sources[str(p.relative_to(E))] == summary['vectors_sha256']
    with np.load(p, allow_pickle=False) as vectors:
        for key, case in summary['cases'].items():
            m = case['methods']['control']
            w = vectors[key+'_control_evaluated'].astype(np.float64)
            keep = sorted(m['sorted_keep'])
            masks = [v['deleted_positions'] for v in m['input_receipts']]
            assert np.isclose(w[keep].sum(),m['eligible_signed']['net'])
            ft = case['historical_FT_reference_only']['curves']
            rows.append(analyze(key, w, keep, masks, m['scores'], 'signed',
                {'RISE':m['RISE'],'MAS':m['MAS'],'needle':None if m['needle'] is None else m['needle']['reported']},
                ft, model='Qwen3.5-9B_current_C', input_sha256=m['input_receipts'][0]['input_sha256'],
                provenance='Current unchanged C from latest complete fixed8; candidate excluded.'))

groups = {}
for model in sorted(set(r['model'] for r in rows)):
    for dataset in ['niah_mq_q2','morehopqa']:
        for scope in ['fixed0to3','all_available']:
            rs = [r for r in rows if r['model']==model and r['case'].startswith(dataset) and (scope=='all_available' or int(r['case'].split('_')[-1])<4)]
            groups[model+'/'+dataset+'/'+scope] = {'n':len(rs),
                'original_RISE':float(np.mean([r['original_metrics']['RISE'] for r in rs])),
                'original_MAS':float(np.mean([r['original_metrics']['MAS'] for r in rs])),
                'negative_over_net_mean':float(np.mean([r['mass']['negative_over_net'] for r in rs])),
                'fixed_path_signed_density_MAS_mean':float(np.mean([r['fixed_original_masks']['signed']['MAS'] for r in rs])),
                'fixed_path_positive_density_MAS_mean':float(np.mean([r['fixed_original_masks']['positive']['MAS'] for r in rs])),
                'view_gap_mean':float(np.mean([r['fixed_original_masks']['signed_minus_positive_MAS'] for r in rs])),
                'signed_view_larger_MAS_count':sum(r['fixed_original_masks']['signed_minus_positive_MAS']>0 for r in rs)}
            certified = [r for r in rs if r['signed_original_order_reuse'] and r['signed_original_order_reuse']['certified']]
            groups[model+'/'+dataset+'/'+scope]['signed_order_reuse_certified_count'] = len(certified)
            if len(certified)==len(rs):
                groups[model+'/'+dataset+'/'+scope]['certified_signed_original_MAS_mean'] = float(np.mean([r['signed_original_order_reuse']['original_RISE_MAS_reconstructed']['MAS'] for r in rs]))
                groups[model+'/'+dataset+'/'+scope]['certified_signed_MAS_better_than_historical_both0_count'] = sum(r['signed_original_order_reuse']['original_RISE_MAS_reconstructed']['MAS']<r['fixed_FT']['MAS'] for r in rs)
out = {'status':'historical_cross_model_score_contract_confounded_verified',
    'model_calls':0, 'new_method_candidates':0, 'runtime_changed':False, 'FT_changed':False,
    'sources_sha256':sources,'cases':rows,'groups':groups,
    'limits':[
        'Original Qwen3 quality used clipped positive scores; Qwen3.5 current quality used signed scores. The historical claim is not evidence of signed DT winning on Qwen3.',
        'Counterfactual density calculations KEEP ORIGINAL MASKS and real responses. They isolate score-view arithmetic, not a newly run original metric with changed sorting.',
        'No clipped view is promoted. Certified signed-order reuse is reported only where every needed actual prefix mask is uniquely determined by signed-score strict separation before the original minimum response reaches zero. Other paths remain unavailable.',
        'Model, tokenizer, prompt wrapper, FA precision and FT versions differ. This audit does not identify an architecture effect or prove no architecture effect.',
        'All examples are already-used development examples. Current Qwen3.5 C includes two layer0 changes; historical raw and repaired stages are not pooled.']}
path = E / 'cross_model_MAS_score_contract_20260909.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({'status':out['status'],'cases':len(rows),'groups':groups},ensure_ascii=False))
