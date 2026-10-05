"""Record actual isolated execution and token-quality localization, not a repair."""
import hashlib
import json
from pathlib import Path
import time

from analyze_textcraft_native_readout import source
from stage_textcraft_conditional_boundaries import LOCAL


REPO=Path(__file__).resolve().parents[3]


if __name__=='__main__':
    report_path=REPO/'experiments/rl/results_textcraft_learning_degradation_20261006.json'
    report=json.loads(report_path.read_bytes())
    analysis=json.loads((LOCAL/'conditional-boundary-analysis.json').read_bytes())
    execution=json.loads((LOCAL/'completed.json').read_bytes())
    job=json.loads((LOCAL/'job.json').read_bytes())
    owner=json.loads((LOCAL/'native-owner-inspection.json').read_bytes())
    independent=json.loads((LOCAL/'independent-analysis-review.json').read_bytes())
    unique=analysis['unique_probe_balanced_boundary_analysis']
    assert analysis['coverage']['unique_probe_identities']==42
    assert unique['C0_vs_actual_full_signed']['difference']['max']==0
    assert execution['finite_trace_calls']==14 and execution['optimizer_steps']==execution['backward_calls']==0
    assert not owner['cuda_initialized']
    assert owner['actual_recipe_callback_binding']['globals_resolved_from_serialized_registered_body']
    selected=[unique['boundary_indices'][index] for index in (32,15,0)]
    assert [row['opposite_sign_count'] for row in selected]==[1,0,22]
    additions=[*sorted(LOCAL.glob('*.json')),LOCAL/'effective-config.yaml',LOCAL/'diagnostic.log',
        LOCAL/'physical-before-start.txt',LOCAL/'prepare.stdout.txt',LOCAL/'launch.stdout.txt',
        LOCAL/'native-owner-inspection-failure.stdout.txt',
        LOCAL/'native-owner-inspection-failure-closure.stdout.txt']
    failed=LOCAL.parent/'v1'
    additions.extend([*sorted(failed.glob('*.json')),failed/'effective-config.yaml',failed/'diagnostic.log',
        failed/'physical-before-start.txt',failed/'prepare.stdout.txt',failed/'launch.stdout.txt',
        *sorted((failed/'submitted-source').glob('*.py'))])
    additions.extend(Path(__file__).with_name(name) for name in (
        'verify_textcraft_conditional_boundaries.py','observe_textcraft_conditional_boundaries.py',
        'stage_textcraft_conditional_boundaries.py','observe_textcraft_conditional_boundary_job.py',
        'inspect_textcraft_boundary_resources.py','analyze_textcraft_conditional_boundaries.py',Path(__file__).name))
    seen={item['path']:item for item in report['sources']}
    for path in additions:
        item=source(path); item['path']=path.relative_to(REPO).as_posix()
        if item['path'] not in seen:
            report['sources'].append(item); seen[item['path']]=item
        else:
            seen[item['path']].update(item)
    report['updated_unix']=time.time()
    report['status']='task_gradient_imbalance_and_prehead_token_quality_loss_localized_no_production_repair'
    report['conditional_boundaries']=dict(
        scope='Actual full-span finite coefficients contracted with matched original single-EOS roots; diagnostic, not new credit or official numerical tolerance gate.',
        analysis=source(LOCAL/'conditional-boundary-analysis.json'),
        independent_review=source(LOCAL/'independent-analysis-review.json'),
        compact=source(LOCAL/'conditional-boundary-compact-summary.json'),
        coverage=analysis['coverage'],calls=analysis['calls'],
        execution=dict(pid=job['pid'],pid_birth=job['pid_birth'],devices=job['devices'],
            started_unix=job['started_unix'],completed_unix=execution['completed_unix'],
            wall_seconds=execution['completed_unix']-job['started_unix'],
            finite_calls=14,single_root_calls=28,backward_calls=0,optimizer_steps=0,scheduler_steps=0),
        actual_sources=analysis['actual_imported_sources'],actual_options=analysis['actual_owner_options'],
        dtype=analysis['dtype'],boundary_32_15_0=selected,
        C0_vs_actual_credit=unique['C0_vs_actual_full_signed'],
        actual_credit_vs_native=unique['actual_full_signed_vs_native'],
        native_vs_prior=unique['native_vs_prior_matched_root'],
        largest_absolute_residual_growth_layers=sorted(unique['decoder_increments'],
            key=lambda row:row['absolute_residual_change_input_minus_output']['mean'],reverse=True)[:5],
        duplicate_ranges=analysis['maximum_replica_ranges'],
        diagnostic_binding_repair=dict(failed_v1=source(failed/'failure-status.json'),
            v2_source_review=source(LOCAL/'independent-source-review.json'),
            cpu_binding=owner['actual_recipe_callback_binding'],
            issue='V1 replaced module globals, but the original Ray/VERL recipe executes a different function globals dictionary. It completed root-only calls and failed the adapter-count assertion. V2 resolves the actual official Ray method/VERL func closures, binds their exact globals, asserts caller identities, and restores the slot. No numerical owner or training code changed.',
            cpu_failures='Intermediate CPU checks failed before GPU at metadata/closure resolution. Both logs are retained. No package, cache, dependency or model changes.',
            limitation='V2 fixes the isolated observation only. It does not repair the learning signal.'),
        limitations=analysis['limitations'])
    findings=[
        'Actual matched full-trace input boundary C0 agrees with the exported signed DT credit on all52 transport observations (max difference4.34e-19). This particular DT export/probe indexing is closed; it does not validate every PPO input in historical training.',
        'On42 unique fixed random probes from21 successful first responses, C32/native correlation0.98627; C15 has42/42 matching native signs and mean absolute contraction0.015733. C0/native correlation0.04956 with22/42 opposite mean signs, native absolute mass49.7409%, and mean absolute contraction0.00181808. One duplicated identity crosses zero; full transport dispersion is retained.',
        'The broad conditional accuracy loss occurs while propagating toward lower decoder boundaries. Mean residual growth is largest atlayer6; larger absolute increments atlayer15 do not identify it as the harmful layer (it removes three opposite signs). These contractions do not separate nonlinear joint-allocation error from native storage/backend error or establish a kernel tolerance failure.',
        'Keeping the original PPO clipping/objective does not correct inaccurate signs or restore the changed task-gradient scale. The already measured complete-minibatch loss imbalance is consistent with persistent entropy-driven drift, while historical causal attribution and a production repair remain unverified.']
    report['findings'].extend(item for item in findings if item not in report['findings'])
    report['findings']=[item.replace(
        'Original clipping constrains the ratio in the PG branch.',
        'Original clipping clips the PG surrogate contribution; it is not a hard bound on policy ratios.')
        for item in report['findings']]
    report['interpretation']['next_owner_check']='Use the localized lower decoder operands to separate actual native dtype/storage effects from full-span conditional finite approximation, with original FA/FLA references and thresholds unchanged. Do not patch PPO, normalize or rescale credits, tune entropy, or resume TextCraft to conceal the weak/misdirected task signal.'
    report['interpretation']['what_changed']='In the measured complete-minibatch comparison the same original actor, masks, token loss reduction and regularizers are used; the advantage source changes. This establishes estimator/loss-component differences, not a full historical only-DT controlled experiment.'
    for item in report['sources']:
        assert hashlib.sha256((REPO/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    runtime=REPO/'experiments/rl/RUNTIME_RECORD.md'; raw=runtime.read_bytes()
    heading='## 2026-10-06 真实条件边界：低层传播削弱并错向token信号'
    entry='''## 2026-10-06 真实条件边界：低层传播削弱并错向token信号

隔离诊断v2 PID1472838/birth1791233251.16已完成退出，GPU4/5、185.171秒；原7个
B4/rank、原checkpoint25、LoRA8/16、原c7fc runner/0ad producer及原目标/布局/切点。
实际14次完整joint finite、28次single原生root，backward/optimizer/scheduler均0。
每组仅短暂留所选slot的实际FP32系数CPU副本（每rank最高约1.05GB），用原_token_effect
收缩原已有BF16 CPU roots并只保存标量；不是CPU模拟有限传播或官方GPU容差验收。

52 transport保留后聚合42既定probe/21成功首次response：C0与实际导出DT信用最大差
4.34e-19；C32与原生single d相关0.98627。C15仍42/42同号、平均|C|0.015733；
C0平均|C|0.001818、22/42反号、相关0.04956。重复槽位完整保留，其中一身份跨零。
误差沿低层传播扩大；不能由最大增量就指认单层bug，不能据此宣布FA/FLA核超差。
下一步针对已定位操作数区分原生dtype存储效应和联合有限分解的条件近似。

v1隔离诊断仅执行42原生root后被计数断言截住（finite0），失败资料及提交源冻结。
根因是public module dict与实际执行recipe globals不同；v2沿官方Ray method/VERL
func闭包解析真实字典，CPU roundtrip/实际caller双身份已核对。中间两次CPU失败日志
保留，无GPU提交。修的是诊断接线，PPO/core/QVA/PLAN/训练参数/正式代码均未改。
原始ranks、独立逐槽复核及来源汇总到results_textcraft_learning_degradation_20261006.json。
TextCraft仍停止，未声称已修复质量，不新增第四组或恢复训练。

'''.encode('utf-8')
    if heading.encode('utf-8') not in raw:
        title,tail=raw.split(b'\n',1); runtime.write_bytes(title+b'\n\n'+entry+tail)
    print(json.dumps(dict(report=source(report_path),runtime_record=source(runtime),
        recorded_sources=len(report['sources']),status=report['status']),ensure_ascii=False))
