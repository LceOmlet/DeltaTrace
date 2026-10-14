"""Read saved native adjoints; separate weight growth from input changes.

No model, training formula, remote execution, or new derivative is evaluated.
Norm quotients locate the observed direction; they are not operator norms.
"""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def read(name):
    return json.loads((HERE / name).read_bytes())

old = read('native-gradient-ratio-chain-summary-v23.json')
new = read('matched-weight-interaction-summary-v28.json')
old_status = read('matched-history-ratio-chain-v23-status-final.json')['report']
new_status = read('matched-history-late-weights-v28-status-final.json')['report']
branch_status = read('matched-last-decoder-branches-v30-status-final.json')['report']
assert not branch_status['alive']
assert branch_status['state']['phase'] == 'complete_native_gradient_chain_localization_not_repair'
for status in (old_status, new_status, branch_status):
    for chain in status['chain_reports']:
        r = chain['value']
        assert r['trainable_SHA256_before'] == r['trainable_SHA256_after']
        assert r['scheduler_before'] == r['scheduler_after']

def active_forward_rms(status, case, boundary):
    result = []
    for chain in status['chain_reports']:
        r = chain['value']
        if r['iteration'] != case:
            continue
        for entry in r['records']:
            if entry['boundary'].endswith(boundary) and entry.get('input_adjoint_norm', 0) > 0:
                q = entry['input_forward_RMS_at_nonzero_adjoint_quantiles']
                result.append(dict(rank=r['rank'], micro=entry['micro'], quantiles=q))
    return dict(records=result, median_of_record_medians=statistics.median(r['quantiles'][2] for r in result),
                minimum_record_min=min(r['quantiles'][0] for r in result))

cases = []
for case in ('early', 'late'):
    a = next(r for r in old['cases'] if r['case'] == case)
    b = next(r for r in new['cases'] if r['case'] == case)
    rows = []
    for layer in range(32):
        name = f'model.layers.{layer}'
        ai, ao = a['boundaries'][name]['input'], a['boundaries'][name]['output']
        bi, bo = b['boundaries'][name]['input'], b['boundaries'][name]['output']
        before = active_forward_rms(old_status, case, name)
        after = active_forward_rms(new_status, case, name)
        rows.append(dict(layer=layer, checkpoint5_input_adjoint=ai, checkpoint5_output_adjoint=ao,
                         checkpoint80_input_adjoint=bi, checkpoint80_output_adjoint=bo,
                         weight_input_ratio=bi / ai, weight_output_ratio=bo / ao,
                         relative_directional_backward_gain=(bi / ai) / (bo / ao),
                         checkpoint5_input_RMS=before, checkpoint80_input_RMS=after,
                         median_RMS5_over_RMS80=before['median_of_record_medians'] / after['median_of_record_medians']))
    G5, G80 = a['native_global_sharded_parameter_norm'], b['native_parameter_gradient_norm']
    H5, H80 = a['native_global_final_hidden_adjoint_norm'], b['final_hidden_adjoint_norm']
    cases.append(dict(fixed_input=case, parameter_gradient_weight_ratio=G80 / G5,
                      final_hidden_adjoint_weight_ratio=H80 / H5,
                      G_over_H_weight_ratio=(G80 / H80) / (G5 / H5), layers=rows,
                      upper_29_to_31_relative_directional_gain=math.prod(rows[l]['relative_directional_backward_gain'] for l in (29, 30, 31))))

branches = []
for case in ('early', 'late'):
    chains = [r['value'] for r in branch_status['chain_reports'] if r['value']['iteration'] == case]
    modules = defaultdict(lambda: defaultdict(float))
    boundaries = defaultdict(lambda: defaultdict(float))
    for chain in chains:
        for r in chain['native_decoder_branch_backward']:
            name = r['name'].split('base_model.model.')[-1]
            for side in ('input', 'output'):
                for value in r['native_module_grad_' + side]:
                    assert value['nonfinite_positions'] == 0
                    modules[name][side] += value['norm'] ** 2
        for r in chain['records']:
            name = r['boundary'].split('base_model.model.')[-1]
            for side in ('input', 'output'):
                if side + '_adjoint_norm' in r:
                    boundaries[name][side] += r[side + '_adjoint_norm'] ** 2
    modules = {n: {s: math.sqrt(v) / 2 for s, v in d.items()} for n, d in modules.items()}
    boundaries = {n: {s: math.sqrt(v) / 2 for s, v in d.items()} for n, d in boundaries.items()}
    g = boundaries['model.layers.31']['output']
    m = modules['model.layers.31.post_attention_layernorm']['input']
    mid = modules['model.layers.31.self_attn']['output']
    attn = modules['model.layers.31.input_layernorm']['input']
    total = boundaries['model.layers.31']['input']
    G = next(r for r in branch_status['state']['observations'] if r['iteration'] == case)['gradient_statistics']['norms']['native_target']
    branches.append(dict(case=case, checkpoint=80, G=G, modules=modules, boundaries=boundaries,
                         MLP_residual_cosine_from_norm_identity=(mid * mid - g * g - m * m) / (2 * g * m),
                         attention_residual_cosine_from_norm_identity=(total * total - mid * mid - attn * attn) / (2 * mid * attn)))

now = datetime.now().astimezone().isoformat()
report = dict(observed_local=now, version='saved-fixed-input-growth-accounting-v32', status='CPU_existing_records_only',
              fixed_input_weight_comparisons=cases, completed_native_branch_v30=branches,
              v30_job=branch_status['job'], v30_actual_bindings=branch_status['state']['bindings'],
              user_scope=dict(whitening_88percent_negative_credit_to_shortening='User-supplied behavior causal chain; record separately, do not use it to explain gradient growth.',
                              common_reward_baseline_question='Potential gradient mechanism: examine direct existing evidence, distinguish from the separate shortening chain.'),
              conclusions=[
                  'With fixed inputs and fixed loss coefficients, checkpoint80 versus5 increases native target parameter-gradient norms22.26x on the early carrier and3.338x on the late carrier.',
                  'On the fixed late input, layer31/30/29 relative directional backward gains multiply to about8.09. This localizes part of a weight-induced propagation change without any changed input, whitening or denominator.',
                  'The fixed early input has a different profile: layers29/30 amplify but31 attenuates the relative ratio. Input/weight interaction prevents a universal last-layer explanation.',
                  'Existing upper-decoder input RMS medians change by only a few percent between checkpoints on both inputs. This is not evidence of a near-zero normalization denominator; medians are not exact directional Jacobians.',
                  'v30 completed without training updates. Its branch cancellation contrast holds weights fixed and changes inputs, so it must not be promoted to the cause of historical weight growth.',
                  'The already measured same-reward-group gradient and dominant target gradient motivate inspecting shared reward components; neither establishes the accelerating85-step mechanism alone.'],
              limits=[
                  'Observed adjoint norm quotients depend on the gradient direction and combine many token positions. They are not operator norms or independent causal shares.',
                  'Between-checkpoint measurements are separate native runs with retained low-precision variation. Exact bitwise equivalence is not claimed.',
                  'v30 residual cosines are derived from separately measured norms, not directly measured dot products; BF16 addition and reduction error remain.',
                  'The15 saved factual pairs are bounded carriers, not the full historical optimizer process or evidence that fresh correct-denominator training fails.',
                  'A nonzero gradient on sampled equal-reward groups is not by itself a defect: the finite sample does not establish constant population reward.'],
              sources={n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in [
                  'native-gradient-ratio-chain-summary-v23.json', 'matched-weight-interaction-summary-v28.json',
                  'matched-history-ratio-chain-v23-status-final.json', 'matched-history-late-weights-v28-status-final.json',
                  'matched-last-decoder-branches-v30-status-final.json']},
              new_model_calls=0, optimizer_updates=0, rootcause_resolved=False, accepted_repair=None, deployed_repair=None)
path = HERE / 'saved-fixed-input-growth-accounting-v32.json'
path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
candidate = HERE / 'candidates/native-QV-decomposition-v31/candidate-status.json'
candidate.write_text(json.dumps(dict(version='native-QV-decomposition-v31', status='prepared_only_unsubmitted_held',
                                    remote_destination='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1/native-DT-QV-gradient-components-v31',
                                    GPU_submission=False, accepted_repair=None, deployed_repair=None,
                                    reason='Held after user scope clarification. Latest common-reward question is being assessed against existing direct gradient evidence before any further measurement.'), indent=2) + '\n', encoding='utf-8')
entry = dict(observed_local=now, record=str(path), status=report['status'],
             fixed_input_ratios=[{k: v for k, v in c.items() if k != 'layers'} for c in cases],
             v30_status='completed_exited_no_training_updates', v30_job=report['v30_job'],
             user_scope=report['user_scope'], goal_status='active', rootcause_resolved=False,
             accepted_repair=None, deployed_repair=None, v31_status='prepared_only_unsubmitted_held')
for name in ('current_runtime.json', 'results_textcraft_gradient_rootcause_20261013_v1.json'):
    dest = ROOT / 'experiments/rl' / name
    value = json.loads(dest.read_bytes())
    dest.write_text(json.dumps({'20261014_fixed_input_growth_v32': entry, **value}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
dest = ROOT / 'experiments/rl/RUNTIME_RECORD.md'
dest.write_text(f'''## {now} 最新用户范围与已有固定输入反向数据v32

用户给出的“白化使88%信用为负、抑制相应token概率、推理链缩短”单独记录，不拿它解释grad norm陡增。随后用户询问不用组/期望优势而直接使用奖励相关优势的可能性；继续用已有同奖励组和target原生梯度证据检验，不把奖励相关本身当错误，不以单点大范数宣称时间根因。

本次仅本机CPU读v23/v28/v30旧记录，无新GPU/模型调用。固定输入/共同系数、只换checkpoint5→80，early G倍率{cases[0]['parameter_gradient_weight_ratio']:.6f}、late {cases[1]['parameter_gradient_weight_ratio']:.6f}；GH倍率{cases[0]['G_over_H_weight_ratio']:.6f}/{cases[1]['G_over_H_weight_ratio']:.6f}。固定late输入第31/30/29层方向范数增益倍率乘积{cases[1]['upper_29_to_31_relative_directional_gain']:.6f}，early profile不同；这定位权重相关传播变化，不是算子bug或独立因果份额。原上部输入RMS中位数只差数个百分点，未支持近零分母猜想。

v30 PID396544/1791967426.42已完成退出，实际worker400246/1791967451.9、401872/1791967467.25；源9eeac925/harnessc30a03cf、原owner/checkpoint80及scheduler/trainable前后保持。两输入G0.254238/1.102199，与v28小幅数值变化保留。最后31层MLP残差夹角由范数恒等式推导约−0.8704/+0.1610，非直接内积；它是固定权重不同输入的抵消对照，不能冒充训练时间增长原因。v31仅CPU prepared-only，未提交GPU，已移出默认launcher。

旧DTPO及分母候选保持停止，GRPO不重做，小时检查PAUSED；原PPO对照保持。goal active，尚无accepted/deployed repair。完整记录{path}。

''' + dest.read_text(encoding='utf-8'), encoding='utf-8')
dest = ROOT / 'experiments/rl/REMOTE_ENVIRONMENT.md'
dest.write_text(f'''## {now} v30完成退出；v32仅CPU读取旧记录

396544/1791967426.42已完成，无本诊断GPU占用；新设备操作仍须重查mx-smi。原PPO2/3、GRPO4/5、环境及缓存保持。v31 prepared-only未提交，默认launcher不接受。小时检查PAUSED、goal active，详细范围/版本见RUNTIME_RECORD顶部。

''' + dest.read_text(encoding='utf-8'), encoding='utf-8')
print(json.dumps(dict(record=str(path), fixed_input_ratios=entry['fixed_input_ratios'], v30_completed=True)))
