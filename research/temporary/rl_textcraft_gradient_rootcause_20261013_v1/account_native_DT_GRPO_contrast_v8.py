"""Account already recorded native DT/GRPO gradients; no new training calls."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path('F:/DeltaTraceBackups/plots/textcraft-dt-formal-native-graph-fresh-20261012-v1-1791793609/latest.json')
snapshot = json.loads(SNAPSHOT.read_bytes())
dt_path = HERE / 'collect_temporal_reports_v1.json'
dt = json.loads(dt_path.read_bytes())['report']['metrics']
grpo = next(j for j in snapshot['jobs'] if j['method'] == 'grpo')['metrics']
windows = []
for lower, upper in [(1, 10), (11, 20), (31, 40), (61, 70), (71, 80), (81, 85)]:
    row = {'requested_iterations': [lower, upper]}
    for label, records in [('DTPO', dt), ('GRPO', grpo)]:
        selected = [r for r in records if lower <= r['step'] <= upper]
        row[label] = {'available_iterations': [r['step'] for r in selected], 'count': len(selected)}
        if selected:
            row[label].update(
                medians={k: statistics.median(r['values'][k] for r in selected)
                         for k in ['actor/grad_norm', 'episode/reward/mean', 'actor/entropy_loss',
                                   'critic/advantages/max', 'critic/advantages/min']},
                grad_norm_range=[min(r['values']['actor/grad_norm'] for r in selected),
                                 max(r['values']['actor/grad_norm'] for r in selected)],
                original_sources=[{'iteration': r['step'], 'sources': r['sources']} for r in selected])
    windows.append(row)

pair_path = HERE / 'replay-status-3.json'
pair = json.loads(pair_path.read_bytes())
ranks = [pair[f'dt-components-rank{rank}.json'] for rank in [0, 1]]
assert ranks[0]['gradient_statistics'] == ranks[1]['gradient_statistics']
audit = []
for rank, data in enumerate(ranks):
    a = data['passes']['dt_pg']['microbatch_losses']
    b = data['passes']['grpo_pg']['microbatch_losses']
    fields = ['weighted_entropy', 'weighted_kl', 'ppo_kl']
    checks = {k: all(x[k] == y[k] for x, y in zip(a, b)) for k in fields}
    assert len(a) == len(b) == 8 and all(checks.values())
    audit.append({'rank': rank, 'native_microbatches_compared': len(a),
                  'same_forward_checks': checks,
                  'clip_fraction_compared': False,
                  'clip_reason': 'The native clipping branch also depends on the advantage sign.'})
stats = ranks[0]['gradient_statistics']
norms = stats['norms']
dot = stats['inner_products']['dt_pg:grpo_pg']
report = {
    'observed_local': datetime.now().astimezone().isoformat(),
    'version': 'native-DT-GRPO-recorded-contrast-v8',
    'scope': __doc__, 'new_GPU_model_calls': 0, 'new_GRPO_runs': 0,
    'new_optimizer_scheduler_calls': 0, 'native_log_windows': windows,
    'existing_same_weights_inputs_native_loss_comparison': {
        'input': 'Saved iteration81 first global64, original checkpoint80 and saved local LoRA/RNG',
        'native_owners': ranks[0]['sources'], 'effective_config': ranks[0]['effective_config'],
        'advantage_provenance': ranks[0]['coefficient_provenance'],
        'native_GRPO_entry': 'Original verl.trainer.ppo.ray_trainer.compute_advantage; global UID grouping.',
        'native_norm_scope': stats['norm_scope'], 'norms': norms,
        'DT_over_GRPO_PG_norm': norms['dt_pg'] / norms['grpo_pg'],
        'DT_GRPO_PG_inner_product': dot,
        'DT_GRPO_PG_cosine': dot / (norms['dt_pg'] * norms['grpo_pg']),
        'DT_minus_GRPO_PG_norm': math.sqrt(norms['dt_pg']**2 + norms['grpo_pg']**2 - 2*dot),
        'existing_forward_point_audit': audit,
    },
    'conclusions': [
        'In recorded iterations1-10 DTPO median0.035 vs GRPO0.0295 is a modest magnitude difference with wider DTPO dispersion, not an orders-of-magnitude initial difference.',
        'On the existing controlled late input, changing only native advantage construction changes both PG magnitude and direction. Uniform learning-rate scaling cannot reproduce that direction change.',
        'The same-state diagnostic is not a fresh-training counterfactual, nor a complete explanation of the85-iteration growth.',
        'Native original actor/core/GRPO are called; this file only reads prior evidence and computes reporting statistics.',
    ],
    'limits': [
        'The GRPO source snapshot ends at65. Later missing data are unavailable, never zero.',
        'Logged preclip mean norms and diagnostic FP64 complete sharded norms have different aggregation scopes.',
        'The2.208 native grouped PG belongs to a different recorded forward point; do not pair it with this0.3296 GRPO value.',
        'The withdrawn0.958/0.162 normalizer comparison remains withdrawn.',
    ],
    'rootcause_resolved': False, 'accepted_repair': None, 'deployed_repair': None,
    'sources': [{'path': str(p.resolve()), 'SHA256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in [SNAPSHOT, dt_path, pair_path, Path(__file__)]],
}
destination = HERE / 'native-DT-GRPO-recorded-contrast-v8.json'
destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'receipt': str(destination.resolve()),
                  'paired_PG_norms': norms, 'cosine': report['existing_same_weights_inputs_native_loss_comparison']['DT_GRPO_PG_cosine']}))
