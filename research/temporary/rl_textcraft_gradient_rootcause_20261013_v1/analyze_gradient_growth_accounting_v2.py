"""Analyze saved gradient growth without new training or causal claims from fits."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PLOTS = Path('F:/DeltaTraceBackups/plots/gradient-rootcause-20261014')
PLOTS.mkdir(parents=True, exist_ok=True)
temporal_path = HERE/'temporal-native-analysis-v1.json'
temporal = json.loads(temporal_path.read_bytes())
rows = temporal['rows']
x = np.array([r['step'] for r in rows], dtype=float)
y = np.array([r['metrics']['actor/grad_norm'] for r in rows], dtype=float)
assert len(rows) == 85 and np.array_equal(x, np.arange(1, 86))

def exponential_fit(xx, yy):
    # Profile out amplitude, then minimize the remaining one-dimensional SSE.
    # The coarse bracket is only numerical optimization, not a training test.
    def objective(b):
        z = np.exp(b*(xx-xx[-1]))
        amplitude = np.dot(z, yy)/np.dot(z, z)
        return np.square(amplitude*z-yy).sum()
    grid = np.linspace(0, 1, 1001)
    index = int(np.argmin([objective(b) for b in grid]))
    lo, hi = grid[max(0, index-1)], grid[min(len(grid)-1, index+1)]
    ratio = (math.sqrt(5)-1)/2
    for _ in range(100):
        left, right = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
        if objective(left) > objective(right):
            lo = left
        else:
            hi = right
    b = (lo+hi)/2
    z = np.exp(b*(xx-xx[-1]))
    amplitude = np.dot(z, yy)/np.dot(z, z)
    return np.array([amplitude, b, xx[-1]])

def fit(xx, yy, model):
    if model == 'exponential':
        return exponential_fit(xx, yy)
    return np.polyfit(xx, yy, 1 if model == 'linear' else 2)

def predict(xx, parameters, model):
    if model == 'exponential':
        a, b, endpoint = parameters
        return a*np.exp(b*(xx-endpoint))
    return np.polyval(parameters, xx)

fits = []
for end in (50, 70, 85):
    for model in ('linear', 'quadratic', 'exponential'):
        parameters = fit(x[:end], y[:end], model)
        fitted = predict(x[:end], parameters, model)
        row = {
            'model': model, 'fit_iterations': [1, end],
            'objective': 'unweighted least squares on original norm scale',
            'parameters': parameters.tolist(),
            'fit_RMSE': float(np.sqrt(np.mean((fitted-y[:end])**2))),
            'fit_nonpositive_prediction_count': int((fitted <= 0).sum()),
        }
        if model == 'quadratic':
            row['constant_second_derivative'] = float(2*parameters[0])
        elif model == 'exponential':
            row['descriptive_doubling_iterations'] = float(math.log(2)/parameters[1])
        if end < 85:
            stop = min(end+15, 85)
            held = predict(x[end:stop], parameters, model)
            row.update(
                next15_iterations=[end+1, stop],
                next15_RMSE=float(np.sqrt(np.mean((held-y[end:stop])**2))),
                next15_actual_median=float(np.median(y[end:stop])),
                next15_predicted_median=float(np.median(held)),
            )
        fits.append(row)

blocks = []
for start in range(0, 85, 10):
    part = y[start:start+10]
    blocks.append({
        'iterations': [start+1, start+len(part)], 'n': len(part),
        'center': float(np.mean(x[start:start+10])),
        'mean': float(np.mean(part)), 'median': float(np.median(part)),
        'minimum': float(part.min()), 'maximum': float(part.max()),
    })

baseline_path = HERE/'replay-status-3.json'
baseline = json.loads(baseline_path.read_bytes())['dt-components-rank0.json']
stats = baseline['gradient_statistics']
labels = ('dt_pg', 'weighted_entropy', 'weighted_kl')
gram = stats['inner_products']
total_sq = sum(gram[f'{a}:{b}']*(1 if a == b else 2)
               for i, a in enumerate(labels) for b in labels[i:])
pg = stats['norms']['dt_pg']
aux_bound = stats['norms']['weighted_entropy']+stats['norms']['weighted_kl']
candidate_path = HERE/'replay-status-5.json'
candidate = json.loads(candidate_path.read_bytes())['dt-global-token-mean-v2-components-rank0.json']
correct_pg = candidate['gradient_statistics']['norms']['dt_pg']
fixed_path = HERE/'fixed-history-analysis-v1.json'
fixed = json.loads(fixed_path.read_bytes())
account_path = HERE/'account_saved_denominator_norm_v1.json'
account = json.loads(account_path.read_bytes())['report']
stop_path = HERE/'stop_premature_temporal_candidate_v1.json'
stop = json.loads(stop_path.read_bytes())['report']

report = {
    'observed_local': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'scope': 'Only the persistent, accelerating DTPO gradient-norm growth; saved evidence only.',
    'new_GPU_model_calls': 0, 'new_optimizer_or_schedule_steps': 0,
    'new_training_runs': 0, 'GRPO_reruns': 0,
    'native_metric_scope': 'Original actor pre-clip norm returns, reduced by native metric owner over four optimizer minibatches and two ranks; console rounds to three decimals. Not a sum of squared norms, and not a cumulative norm across iterations.',
    'native_metric_source_owner': {
        'actor': 'owners/verl/workers/actor/dp_actor.py:297,506',
        'FSDP2': 'native-clip-owner-v2.json',
        'metric_reduce': 'owners/verl/utils/metric/utils.py:23',
        'trainer': 'owners/verl/trainer/ppo/ray_trainer.py:1285',
    },
    'temporal_observations': {
        'complete_iterations': 85, 'nonoverlapping_blocks': blocks,
        'mean_ratio_80to85_vs_1to10': temporal['windows']['80-85']['gradient_mean']/temporal['windows']['1-10']['gradient_mean'],
        'median_ratio_80to85_vs_1to10': temporal['windows']['80-85']['gradient_median']/temporal['windows']['1-10']['gradient_median'],
        'note': 'Both center and variability grow. Unequal window means are not a fitted exponent; block81-85 has only five observations.',
    },
    'descriptive_fits': fits,
    'trend_conclusion': 'The data support an increasing level and stronger late growth, but do not establish a fixed quadratic or exponential law. A quadratic on1-50 has quadratic coefficient2.477e-5; on1-85 it is1.9835e-4. Early fits underpredict late observations. The global exponential has slightly lower original-scale RMSE, but neither fit identifies a causal mechanism or justifies two-day trial training.',
    'fit_limits': [
        'These are rounded native aggregate observations; no exact temporal functional law is claimed.',
        'Batches differ and observations are temporally dependent. No independent-error confidence interval or causal percentage is invented.',
        'Original-scale least squares is influenced by the larger late observations; medians/min/max are separately retained.',
        'The same offline finite data can constrain mechanisms without either fitting a universal function or constructing an exact gradient Gram matrix.',
    ],
    'saved_late_component_budget': {
        'scope': 'Existing checkpoint80/iteration81 first global64 minibatch; FP64 statistics on native gradient shards; not native console mean of four minibatches.',
        'norms': {label: stats['norms'][label] for label in labels},
        'total_norm_reconstructed_with_cross_terms': total_sq**.5,
        'auxiliary_norm_triangle_upper_bound': aux_bound,
        'auxiliary_bound_relative_to_PG': aux_bound/pg,
        'conclusion': 'Policy gradient dominates this measured minibatch; entropy/KL cannot explain its magnitude. No historical component time series exists to promote this to all85 iterations.',
        'sources': baseline['sources'],
    },
    'denominator_accounting': {
        'exact_weight_identity_at_fixed_batch': 'h_j=grad(microbatch token mean); legacy g=(1/16)sum(h_j); correct g=sum((N_j/N)*h_j).',
        'consequence': 'Uniformly shortening all N_j leaves both sets of relative microbatch weights unchanged. Shortening alone does not imply parameter-gradient amplification.',
        'parameter_norm_identity': '||sum(w_j*h_j)||^2=sum_j(w_j^2*||h_j||^2)+2*sum_{j<k}(w_j*w_k*<h_j,h_k>).',
        'scope_of_identity': 'Algebra fixes weighting at given inputs/parameters; it does not supply unknown gradient directions, explain their temporal change, or demand exact Gram data as the only useful analysis.',
        'paired_late_PG_norms': {'legacy': pg, 'corrected': correct_pg, 'ratio': pg/correct_pg},
        'difference_norm_triangle_bounds': [abs(pg-correct_pg), pg+correct_pg],
        'same_point_cross_inner_product_available': False,
        'saved_old_policy_logprob_coefficient_L2_ratios_by_global64': [m['coefficient_L2_ratio'] for m in account['global64_minibatches']],
        'whole256_whitened_coefficient_second_moment': account['whole256_whitened_second_moment'],
        'conclusion': 'A substantial one-batch weighting defect is confirmed. Its temporal amplification and sufficiency for the accelerating anomaly have not been established. Do not divide native temporal growth by the paired late norm ratio.',
    },
    'fixed_input_parameter_location': {
        'scope': 'Existing fixed late input/advantage at checkpoints5/40/80, each recomputed old-policy point. Counterevidence only, not historical on-policy batches.',
        'observations': [{k:r[k] for k in ('checkpoint','FP64_shard_gradient_norm','A_squared_norm_fraction','top_layers_squared_norm_fraction')} for r in fixed['rows']],
        'interpretation': 'In this observation LoRA B gradients dominate the squared norm; at checkpoint80 A accounts for only0.655%. Do not confuse growing parameter/Adam magnitudes with the current gradient norm.',
    },
    'causal_claim_withdrawn': 'B4 denominator bias plus trajectory shortening and increasing target density is already the root cause of temporal acceleration.',
    'rootcause_resolved': False, 'accepted_repair': None, 'deployed_repair': None,
    'candidate_withdrawn': {'status': stop['status'], 'pid': stop['pid'], 'birth': stop['birth'], 'live_survivors': stop['live_survivors'], 'formal_GRPO_preserved': stop['formal_grpo_preserved']},
    'remaining_evidence_gap': 'Native85 scalar norms plus nonzero-only raw credits do not specify historical complete256 whitened inputs or historical component gradients. Use existing raw records/code/checkpoint constraints before deciding what new targeted information is worth obtaining; no new long-run candidate is authorized by this analysis.',
}
report['provenance'] = [{'path': str(p.resolve()), 'SHA256': hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in (temporal_path, baseline_path, candidate_path, fixed_path, account_path, stop_path, Path(__file__))]

csv_path = PLOTS/'native-gradient-growth-85.csv'
with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['iteration','native_grad_norm','native_pg_loss','native_entropy_loss','native_kl_loss','native_pg_clipfrac','original_sources'])
    writer.writeheader()
    for r in rows:
        m = r['metrics']
        writer.writerow({'iteration':r['step'], 'native_grad_norm':m['actor/grad_norm'],
            'native_pg_loss':m['actor/pg_loss'], 'native_entropy_loss':m['actor/entropy_loss'],
            'native_kl_loss':m['actor/kl_loss'], 'native_pg_clipfrac':m['actor/pg_clipfrac'],
            'original_sources':json.dumps(r['source_paths'],ensure_ascii=False)})

fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), constrained_layout=True)
for ax in axes:
    ax.plot(x, y, color='#738399', alpha=.65, lw=1.2, marker='.', ms=4, label='Native DTPO norm')
    ax.plot([r['center'] for r in blocks], [r['median'] for r in blocks], color='#2364aa', marker='o', lw=2, label='Block median (10; last 5)')
    ax.set_xlabel('Completed training iteration')
    ax.set_ylabel('Native actor/grad_norm')
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_title('Saved 85 iterations: level and variability increase')
axes[0].set_ylim(y.min()*.8, y.max()*1.08)
axes[1].set_yscale('log')
axes[1].set_title('Earlier fits do not explain the late growth')
for model, color, label in [('quadratic','#df8b27','Quadratic fitted on 1-70'),('exponential','#348064','Exponential fitted on 1-70')]:
    parameters = fit(x[:70], y[:70], model)
    yy = predict(x, parameters, model)
    axes[1].plot(x[:70], yy[:70], color=color, lw=1.5, label=label)
    axes[1].plot(x[69:], yy[69:], color=color, lw=1.5, ls='--')
axes[1].axvspan(70.5,85.5,color='#e7ebef',alpha=.55,zorder=-2)
axes[0].legend(frameon=False,fontsize=8,loc='upper left')
axes[1].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('DTPO accelerating gradient growth — descriptive evidence, not a causal fit',fontsize=12)
png_path = PLOTS/'native-gradient-growth-accounting-v2.png'
fig.savefig(png_path, dpi=180)
plt.close(fig)
report['artifacts'] = {'plot':str(png_path.resolve()),'CSV':str(csv_path.resolve())}
dest = HERE/'gradient-growth-accounting-v2.json'
dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(PLOTS/dest.name).write_bytes(dest.read_bytes())
print(json.dumps({'report':str(dest.resolve()),'artifacts':report['artifacts'], 'component_budget':report['saved_late_component_budget']['norms'], 'rootcause_resolved':False}))
