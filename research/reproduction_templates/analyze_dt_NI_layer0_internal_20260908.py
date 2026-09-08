"""Independent CPU audit of actual current NI layer0 replay/finite contractions."""
import hashlib, json, time, zipfile
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parent
D = A / 'snapshot${ARTIFACT_ROOT}/codex_dt_NI_layer0_internal_20260908_v1'
started = time.perf_counter()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
p, r = read(D/'protocol.json'), read(D/'results.json')
assert p == r['protocol'] == read(A/'dt_NI_layer0_internal_protocol_20260908.json')
assert r['status'] == 'NI_current_layer0_5replay1finite_internal_measurement_complete'
receipt = read(D/'terminal_receipt.json')
for name, v in receipt['files'].items():
    assert sha(D/name) == v['sha256'] and (D/name).stat().st_size == v['bytes'], name
for name, want in p['files_sha256'].items():
    assert sha(D/name) == want, name
with zipfile.ZipFile(D/'review_bundle.zip') as archive:
    for name in archive.namelist():
        assert archive.read(name) == (D/name).read_bytes(), name
assert r['sources_before'] == r['sources_after']
assert r['weight_stats_before'] == r['weight_stats_after'] == p['expected_weight_stats']
source = A/'snapshot'/p['source_results_path'].lstrip('/')
assert sha(source) == p['source_results_sha256']
prior = read(source)
assert r['input'] == p['input'] == prior['input']
assert r['model_loads'] == r['finite_layer_entered'] == r['finite_layer_returned'] == 1
assert r['layer_replays_entered'] == r['layer_replays_returned'] == 5
assert r['finite_FLA_counts'] == {'entered': 1, 'returned': 1}
assert all(r[k] == 0 for k in ['whole_model_forwards','DT_calls','scorer_calls','FA_calls','FT_calls','generation_calls'])
assert p['layer'] == 0 and p['norm_gate_rule'] == r['current_norm_gate_rule'] == 'symmetric'
assert all(c['status'] == 'returned' for c in r['calls'])
for point, row in r['points'].items():
    assert all(v == 1 for v in row['decoder_calls'].values())
    assert row['mixer_calls'] == {'module': 1, 'conv': 1, 'FLA': 1, 'stage': 1}
    assert not row['initial_cache']['has_previous_state'] and row['mask'] is None
    assert row['initial_cache']['provided'] == (point != 'B2')

def close(x, y):
    assert np.max(np.abs(np.asarray(x)-np.asarray(y)), initial=0) < 1e-7

def stats(x):
    return dict(net=float(x.sum()), positive=float(x.clip(min=0).sum()),
                negative=float(x.clip(max=0).sum()), absolute=float(np.abs(x).sum()))

z = np.load(D/'vectors.npz', allow_pickle=False)
T, P = r['input']['total_length'], r['input']['prompt_length']
keep = set(r['input']['keep'])
out = dict(status='independent_NI_current_layer0_internal_audit_passed',
    analyzer_sha256=sha(Path(__file__)), protocol_sha256=sha(D/'protocol.json'),
    results_sha256=sha(D/'results.json'), vectors_sha256=sha(D/'vectors.npz'),
    seconds=r['seconds'], counts={k:r[k] for k in ['model_loads','layer_replays_entered',
        'layer_replays_returned','finite_layer_entered','finite_layer_returned','finite_FLA_counts',
        'whole_model_forwards','DT_calls','scorer_calls','FA_calls','FT_calls','generation_calls']},
    frozen_budget=p['budget'], m0_replay_drift=r['m0_replay_drift_report_only'],
    native_output_replay_drift={k:v['output_replay_drift'] for k,v in r['points'].items()}, points={},
    private_artifact=r['private_artifact'],
    proof_scope='Public signed contractions, source/weight receipts, frozen masks, and per-token closure audited. The multi-GB private actual tensors were not locally rehashed or independently recomputed. This is a replay diagnostic, not a new production method or metric evaluation.')
for step in ['1','10','20','B2']:
    row = r['conditional_ledgers'][step]
    ledger = row['replayed_16term_ledger']
    assert ledger['sign_convention'] == 'prediction_minus_actual' and len(ledger['terms']) == 16
    deleted = keep if step == 'B2' else set(prior['points'][step]['input_receipt']['deleted_positions'])
    assert deleted <= keep
    groups = dict(deleted=sorted(deleted), kept=sorted(keep-deleted),
                  other_prompt=sorted(set(range(P))-keep), response=list(range(P,T)))
    arrays = {name:z[step+'_'+name] for name in row['coordinate_groups']}
    for name, x in arrays.items():
        assert x.shape == (1,T) and np.isfinite(x).all()
        saved = row['coordinate_groups'][name]
        for k in ['net','positive','negative']: close(stats(x)[k], saved['total'][k])
        for group, indices in groups.items():
            assert saved['groups'][group]['count'] == len(indices)
            for k in ['net','positive','negative']: close(stats(x[0,indices])[k], saved['groups'][group][k])
    for name, value in ledger['terms'].items(): close(arrays[name].sum(), value)
    for name, values in row['transfer_terms'].items():
        for k in values: close(stats(arrays[name])[k], values[k])
    measured = arrays['saved_boundary_error']
    reconstruction = sum(arrays[k] for k in ledger['terms']) + sum(arrays[k] for k in row['transfer_terms'])
    close(reconstruction, measured)
    close(sum(ledger['terms'].values()), ledger['input_contraction']-ledger['output_contraction'])
    close(ledger['prediction_minus_actual'], sum(ledger['terms'].values()))
    close(ledger['actual_minus_predicted'], -ledger['prediction_minus_actual'])
    original = (prior['B2_boundary_contractions']['0']-prior['B2_boundary_contractions']['1']) if step == 'B2' else prior['points'][step]['coarse']['regions']['0_to_1']
    close(measured.sum(), original)
    terms = ledger['terms']
    out['points'][step] = dict(saved_boundary_error=stats(measured), replayed_16terms=ledger,
        transfer_terms=row['transfer_terms'], transfer_net=sum(v['net'] for v in row['transfer_terms'].values()),
        token_closure_max_absolute=float(np.max(np.abs(reconstruction-measured))),
        positive_terms_ranked=sorted([(k,v) for k,v in terms.items() if v>0], key=lambda kv:-kv[1]),
        negative_terms_ranked=sorted([(k,v) for k,v in terms.items() if v<0], key=lambda kv:kv[1]),
        coordinate_groups=row['coordinate_groups'])
out['interpretation'] = (
    'Current midpoint FLA boundary is the largest measured mismatch (+21.6463 of layer0 +23.1135); '
    'replay transfer is only +0.02096. This prioritizes a finite-FLA internal diagnosis, not a proven recurrence bug or guaranteed cure. '
    'Early FLA -1.2776 compensates MLP +2.2608: candidate validation must retain all signed terms and full-input effects. '
    'Current symmetric normgate is secondary at midpoint (+1.4528), MLP +1.7817. '
    'Near-zero allEOS/B2 totals contain large opposite token contributions; endpoint conservation does not establish conditional or tokenwise accuracy.')
out['audit_seconds'] = time.perf_counter()-started
target = A/'dt_NI_layer0_internal_summary_20260908.json'
target.write_text(json.dumps(out, indent=2, allow_nan=False))
print(json.dumps(dict(output=str(target), sha256=sha(target), seconds=out['audit_seconds'])))
