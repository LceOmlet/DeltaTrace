"""Append source-bound findings to the existing diagnosis and runtime record."""
import hashlib
import json
from pathlib import Path
import time

from analyze_textcraft_native_readout import source
from analyze_textcraft_matched_layout import LOCAL


REPO = Path(__file__).resolve().parents[3]


if __name__ == '__main__':
    report_path = REPO/'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report = json.loads(report_path.read_bytes())
    detail_path = LOCAL/'head-seed-cpu-analysis.json'
    detail = json.loads(detail_path.read_bytes())
    review_path = LOCAL/'independent-head-seed-review.json'
    review = json.loads(review_path.read_bytes())
    boundary_path = LOCAL/'existing-v6-decoder-boundary-review.json'
    boundary = json.loads(boundary_path.read_bytes())
    unique = detail['unique_probe_balanced']
    assert unique['binary_sign_checks']['head_native_same_nonzero_sign']==42
    assert unique['binary_sign_checks']['dt_native_opposite_sign']==21
    additions = [detail_path, review_path, boundary_path,
        *[REPO/item['path'] for item in boundary['sources'].values()],
        LOCAL/'head-seed-cpu-observation.json',
        LOCAL/'native-matched-layout-logits-map.json', LOCAL/'head-seed-cpu.stdout.txt',
        LOCAL/'head-seed-cpu.stdout.attempt-bare-env.txt',
        LOCAL/'owner-contract/head-seed-source-contract.json',
        *[LOCAL/'owner-contract'/name for name in ('compiled_logprob_seed.py',
            'compiled_finite_rules.py','signed_secant_rules.py')],
        *[Path(__file__).with_name(name) for name in ('prepare_textcraft_matched_logits.py',
            'observe_saved_textcraft_head_seed_cpu.py','run_saved_textcraft_head_seed_cpu.py',
            'analyze_saved_textcraft_head_seed_cpu.py',Path(__file__).name)]]
    seen = {item['path']: item for item in report['sources']}
    for path in additions:
        item = source(path)
        item['path'] = path.relative_to(REPO).as_posix()
        if item['path'] not in seen:
            report['sources'].append(item)
            seen[item['path']] = item
        else:
            seen[item['path']].update(item)
    report['updated_unix'] = time.time()
    report['status'] = 'loss_imbalance_and_token_quality_gap_output_head_excluded_as_sign_source_no_training_repair'
    report['head_seed_decomposition'] = dict(
        scope='Offline original-owner CPU FP32 seed on saved actual native logits, not actual GPU seed capture or a tolerance gate.',
        source=source(detail_path), independent_review=source(review_path),
        execution=detail['recomputation'], coverage=detail['coverage'],
        binary_sign_checks=unique['binary_sign_checks'],
        head_vs_native=unique['comparisons']['joint_head_on_actual_single_logits_vs_native'],
        errors=unique['errors'], error_MSE_identity=unique['error_MSE_identity'],
        initialization_repair='First CPU import omitted existing MetaX environment variables and failed in backend discovery; rerun reused the original recorded process environment. No package or cache changes.',
        no_candidate_deployed='Original equal-endpoint seed was inspected offline only; it was not substituted into DT credits.')
    report['existing_selected_decoder_storage_evidence'] = dict(source=source(boundary_path),
        scope=boundary['scope'], selection=boundary['selection'],
        terms=boundary['terms'],
        limitation='A separate selected v6 single-token intervention. Do not combine it with v4 values or identify it as the layer decomposition of the current42 joint probes.')
    findings = [
        'New offline original-owner recomputation: applying the joint categorical seed to actual single-token logit differences gives42/42 matching native signs and correlation0.985639. DT still has21/42 opposite signs. Top conditional curvature alone cannot explain these disagreements.',
        'Head conditional mismatch MAE0.004105/RMS0.007005; remaining propagation/allocation/projection/storage residual MAE0.015630/RMS0.023300. Their cross term is retained; no percentage causal attribution or numerical tolerance verdict is invented.',
        'Original GRPO group centering/std and its task-gradient scale are no longer inherited merely by keeping the same clipped loss when DT advantages replace them. This describes a changed estimator, not an instruction to add normalization to the fixed method.',
        'The next necessary localization is within the original pre-head finite propagation/allocation using real existing native operand boundaries, with numerical storage effects kept separate. A top-only seed replacement would not remove the observed sign errors.']
    report['findings'].extend(item for item in findings if item not in report['findings'])
    report['interpretation']['next_owner_check'] = 'Localize original pre-head finite propagation/allocation and native storage effects. Existing decoder3 evidence is selected-case evidence, not the layer attribution of this42-probe result. Keep Q/V/A, masks, official PPO, LoRA and batch settings unchanged.'
    for item in report['sources']:
        path = REPO/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], item['path']
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = REPO/'experiments/rl/RUNTIME_RECORD.md'
    old = runtime.read_bytes()
    heading = '## 2026-10-06 输出层拆分：反号来源在head之前'
    entry = '''## 2026-10-06 输出层拆分：反号来源在head之前

仅重用已保存的matched v1原FP32类别logits，正式c9 seed_with_checks及依赖三模块SHA
与原Git对象/实际import一致。CPU PID1147956，2.722秒、maxRSS655851520字节，CUDA/
distributed未初始化，模型/decoder有限传播/backward/optimizer/scheduler均0。
首个CPU导入因未继承MetaX环境变量失败；已复用原记录PID1856052/birth1791197944.7的
完整环境后完成，失败日志另存，没有重装、清缓存或更改训练环境。

52 transport观察先逐项计算再显式聚合42唯一probe（21成功首次response）：joint
categorical seed乘实际single Δlogits与原生单删42/42同号，相关0.985639；最终DT仍21/42
反号。head条件失配MAE0.004105/RMS0.007005，余下传播/分解/投影/存储残差
MAE0.015630/RMS0.023300；交叉项保留，不能按MSE做因果百分比分账。CPU原log-prob
与已存原生差最大1.043e-7，仅描述，未设官方之外的容差或纠偏。

新增证据排除仅靠换输出seed修反号的方向；未部署equal_endpoint或缩放信用，仍须在
原pre-head边界定位残差。独立review逐值复核原raw/map及52 dot/42聚合，来源在
results_textcraft_learning_degradation_20261006.json。PPO/core/optimizer、Q/V/A、
PLAN、LoRA8/16和B4不变，TextCraft正式保持停止，未称已修好质量。

'''.replace('\n','\r\n').encode('utf-8')
    title = b'# '+ '当前运行版本与修复记录'.encode('utf-8')
    assert old.startswith(title)
    end = old.index(b'\n')+1
    if heading.encode() not in old:
        runtime.write_bytes(old[:end]+b'\r\n'+entry+old[end:])
    print(json.dumps(dict(report=source(report_path), runtime=source(runtime), source_count=len(report['sources'])),indent=2))
