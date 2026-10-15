"""Account actual historical clipped-gradient energy from native Adam state.

Only existing CPU-read native optimizer statistics are used. The second-moment
recurrence is a reporting identity, not a new optimizer implementation. Native
FP32 update rounding, clipping and unequal EMA filters remain explicit limits.
"""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
source = HERE / 'read_temporal_adapter_states_v2.json'
original = json.loads(source.read_bytes())['report']
rows = []
for checkpoint in original['checkpoints']:
    values = defaultdict(lambda: dict(second=0., first_squared=0.))
    steps = set()
    betas = set()
    for rank in checkpoint['ranks']:
        assert rank['optimizer_state_count'] == 496
        for group in rank['optimizer_groups']:
            betas.add(tuple(group['betas']))
        for name, state in rank['moments_by_module'].items():
            values[name]['second'] += state['exp_avg_sq_sum']
            values[name]['first_squared'] += state['exp_avg_sumsq']
            steps.add(state['optimizer_step'])
    assert steps == {float(4 * checkpoint['step'])} and len(betas) == 1
    beta1, beta2 = next(iter(betas))
    step = int(next(iter(steps)))
    rows.append(dict(iteration=checkpoint['step'], optimizer_step=step,
        beta1=beta1, beta2=beta2, values=dict(values),
        moment_norm=math.sqrt(math.fsum(v['first_squared'] for v in values.values())),
        second_moment_trace=math.fsum(v['second'] for v in values.values())))
windows = []
for left, right in zip(rows, rows[1:]):
    assert (left['beta1'], left['beta2']) == (right['beta1'], right['beta2'])
    assert set(left['values']) == set(right['values'])
    delta = right['optimizer_step'] - left['optimizer_step']
    decay = right['beta2'] ** delta
    weight_sum = 1 - decay
    per_parameter = {name: (right['values'][name]['second'] - decay * left['values'][name]['second']) / weight_sum
                     for name in left['values']}
    total = math.fsum(per_parameter.values())
    assert total > 0
    layers = defaultdict(float)
    modules = defaultdict(float)
    sides = defaultdict(float)
    for name, energy in per_parameter.items():
        match = re.search(r'\.layers\.(\d+)\.', name)
        layers[int(match[1]) if match else -1] += energy
        modules[name.split('.lora_')[0]] += energy
        sides['A' if '.lora_A.' in name else 'B'] += energy
    def leading(values, count):
        return [dict(name=name, weighted_gradient_energy=value, fraction=value / total)
                for name, value in sorted(values.items(), key=lambda item: item[1], reverse=True)[:count]]
    windows.append(dict(iterations=[left['iteration'] + 1, right['iteration']],
        optimizer_steps=[left['optimizer_step'] + 1, right['optimizer_step']],
        native_beta2=right['beta2'], weighted_clipped_actor_gradient_energy=total,
        weighted_clipped_actor_gradient_RMS=math.sqrt(total),
        ending_beta1_momentum_norm=right['moment_norm'],
        ending_momentum_norm_over_window_RMS=right['moment_norm'] / math.sqrt(total),
        ratio_scope='Different beta1/beta2 windows; descriptive only, never an exact mean/variance decomposition.',
        lower0_7_energy_fraction=math.fsum(v for k, v in layers.items() if 0 <= k < 8) / total,
        LoRA_A_energy_fraction=sides['A'] / total,
        top_layers=leading(layers, 8), top_modules=leading(modules, 8),
        negative_parameter_energy_estimates=[dict(name=k, energy=v) for k, v in per_parameter.items() if v < 0]))
report = dict(recorded_at=datetime.now().astimezone().isoformat(), scope=__doc__,
    source=dict(path=str(source), SHA256=hashlib.sha256(source.read_bytes()).hexdigest()),
    checkpoints=[{k: v for k, v in row.items() if k != 'values'} for row in rows],
    windows=windows,
    reporting_identity='sum_p(v_b[p]-beta2**(b-a)*v_a[p])/(1-beta2**(b-a)) is the normalized beta2-weighted energy of gradients actually entering native Adam in that update window, up to original native FP32 EMA rounding.',
    model_calls=0, backward_calls=0, optimizer_calls=0, rootcause_repair_verified=False,
    limits=['These are actual post-clipping joint actor gradients, not the preclip console mean norm or policy-gradient-only components. Clipping can conceal preclip growth.',
        'Native original Adam beta1=.9 and beta2=.999 have different filters. Ending momentum norm/window RMS is not a variance estimate or proof of pure noise.',
        'Module ownership follows the recorded native optimizer/state_dict ordering and matched local shapes. No new optimizer ownership mapping is fabricated.',
        'Stored native EMA states were computed with finite precision. Any small negative scalar differences remain reported without invented tolerances or clamping.',
        'These historical windows constrain where actual gradient energy grew; they do not alone identify the update mechanism that caused the growth or accept a repair.'])
path = HERE / 'native-Adam-gradient-energy-windows-v104.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(path.resolve()), windows=[{k:v for k,v in w.items() if k not in ('top_layers','top_modules','negative_parameter_energy_estimates')} for w in windows], late_top_modules=windows[-1]['top_modules'][:4]), ensure_ascii=False))
