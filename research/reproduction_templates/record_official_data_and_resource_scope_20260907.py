"""Record user scope corrections and verified stop; no model or evaluator calls."""
import ast
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path
A = Path(__file__).resolve().parent
R = A.parent/'DeltaTrace'
folder = A/'snapshot${ARTIFACT_ROOT}/codex_local_qwen8b_morehop_generation_20260907_v1'
with zipfile.ZipFile(folder/'interrupted_review_bundle.zip') as z:
    for name in z.namelist():
        assert '/' not in name and '\\' not in name and name not in ('.', '..')
        (folder/name).write_bytes(z.read(name))
receipt = json.loads((folder/'scope_exclusion_summary.json').read_text())
raw = (folder/'results.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == receipt['raw_sha256']
r = json.loads(raw)
assert receipt['process_verified_terminal'] and not receipt['new_cache_authorized_for_evaluation']
assert len(r['records']) == receipt['completed_saved_records'] == 44
assert len(r['batches']) == receipt['completed_saved_batches'] == 11
assert r['attribution_calls'] == r['quality_evaluation_calls'] == 0
assert r['native_generation_calls'] == 12
assert sum(x['root_forwards'] for x in r['batches']) == r['native_root_forwards'] == 1750
assert sum(x['decoder_calls'] for x in r['batches']) == r['native_decoder_layer_calls'] == 63000
for b in r['batches']:
    assert b['physical_batch_size'] == 4
    assert sum(b['actual_public_FA_returns'].values()) == b['decoder_calls']
receipt.update(completed_batch_generation_seconds=sum(b['seconds'] for b in r['batches']),
    completed_batch_peak_bytes=max(b['peak_allocated_bytes'] for b in r['batches']),
    local_raw_hash_and_saved_totals_independently_checked=True,
    official_benchmark_records_added=0, external_API_calls=0,
    abandoned_source_sha256=hashlib.sha256((folder/'study.py').read_bytes()).hexdigest(),
    abandoned_protocol_sha256=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest())
summary_name = 'local_generation_excluded_by_official_data_scope_20260907.json'
(A/summary_name).write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')

policy = {
    'status': 'active_user_scope',
    'official_data_required': True,
    'primary_data': 'Hash-verified author Table1 processed caches with original prompt, fixed target, sink/thinking spans, token eligibility and original metrics. No locally regenerated replacement cohort.',
    'official_sampling': 'Only the author original generation/filter process may supply supplementary data, distinctly labeled from released Table1 reproduction. The author script model defaults are not themselves proof of historical API model use. Existing processed caches need no generator/judge API.',
    'paper_result_reuse': 'Use identical author cached targets and protocol for comparison. Published aggregate values are comparable only with matching model/configuration, task cohort and aggregation; old8-case development means are not full-table reproduction.',
    'historical_exposure': 'NIq2 all100 and MoreHopQA all95 have prior-use evidence. Reuse is allowed for development/reproduction; never relabel those rows as pristine independent confirmation.',
    'local_generation_branch': 'Stopped by explicit user scope; saved44 outputs excluded from all official method comparisons. No attribution or quality score was computed on them.',
    'resource_priority': 'Use existing evidence to choose edits; no default exhaustive reruns, parameter grids or dataset expansion to seek significance.',
    'next_change': 'Preserve finite-P1 attribution math and use native implementation/reuse to reduce repeated work. Diagnose existing traces before starting another GPU job.',
    'initial_optimization_screen': {
        'cases': [['niah_mq_q2', 2], ['morehopqa', 5]],
        'reason': 'Existing development cases only: longer NI input and largest observed P1/FTseq cost ratio. Diagnostic selection, not representative quality confirmation.',
        'methods_first': ['unchanged_P1', 'single_optimized_P1'],
        'warm_calls_per_case_method': 1, 'measured_calls_per_case_method': 2,
        'maximum_initial_fresh_attributions': 12,
        'fresh_original_quality_curves': 0,
        'stop_if': 'Numerical/sign allocation validity fails, a quadratic buffer appears, memory violates the accepted bound, or measured cost does not give a credible reason to continue. Small noisy changes are inconclusive, not a win.',
        'optional_followup': 'Only after a promising initial screen, at most6 same-scope seq-only originalFT attributions and2 actual ordinary input-backward memory references under a frozen protocol. Independent confirmation and larger B4 evaluation are separate stages, never automatically launched.',
        'quality_policy': 'Compare full new signed/projected vectors to pinned original parent. Exact vector/source/input matches permit traceable old-curve reuse; changed vectors cannot inherit old metrics. Do not evaluate more curves until the implementation passes the screen and a bounded followup is recorded.'
    },
    'completion': 'Development quality advantage established; independent quality, conditional direction and full efficiency goals remain unproved.'
}
policy_name = 'official_data_resource_policy_20260907.json'
(A/policy_name).write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')
name = '官方数据优先与小预算验证_20260907.md'
text = '''# 官方数据优先与小预算验证

用户要求复用论文测量所用的数据。正式比较继续使用作者发布、哈希核验一致的处理后缓存，保留原prompt、target、答案/思考跨度、可归因范围及原RISE/MAS实现。现有缓存不依赖生成/评判API；补充数据只有在使用官方生成筛选流程时才进入相应分支，并与原表格复现区别标记。本地Qwen3-8B重新生成不替代论文原始轨迹。

## 已经足够支持下一决策的结论

|问题|已有证据|结论边界|
|---|---|---|
|开发质量|原NI/MH各8例：P1 needle 77.52%，最强FT59.33%；MH RISE 0.05839 vs0.11013，MAS 0.11710 vs0.19472；RISE/MAS越低越好|原开发优势成立，不是完整论文表格或独立确认|
|公平成本|原16例同次计时，FT只取seq后分数完全相同；P1/FTseq配对中位比1.377，16例均慢|当前单样本效率尚未达标，足以决定先改重复计算|
|显存与批处理|已有四例真实B4；有限传播峰值21.766GB，同批原生反向31.291GB|小规模真实批处理已验证，不能外推任意长度；无需仅为再次确认这点重跑|
|符号|开发诊断384个token，201个的真实有限效应随干预条件反号；NI强负分原输入删除同号30/64|P1是明确规则下的带符号有限分配，不能宣称其负分普遍表示删除有益|
|独立性|原NIq2-100和MH95均有历史使用记录|可继续用于复现/开发，不重新称独立确认|

## 已停止的偏离分支

本地8B生成曾启动，在用户明确要求官方数据后向已核对命令行的本次PID154937发送SIGTERM，并随后核实进程已终止。保存44条输出、11个完整B4批次，第12批被中止；归因和原质量评测均0次。输出只保留为排除分支的审计记录，没有并入任何正式缓存，也没有把缺失的中止批次成本填为0。完整已保存批次计数为1,750次原生根前向、63,000次层前向；公开FA调用观察与层计数一致。原始结果SHA见配套排除回执。

## 资源约束与下一步

先分析已有记录，保留强P1方法，针对实际重复计算做一个改动。首轮仅原开发NI2和MH5两个已用样本，旧/新P1各一次预热、两次实测，最多12次归因、0条新删除曲线。数值、符号分配、内存或耗时不支持继续时立即停止该候选，不铺开全量。仅初筛有明确依据继续时，才补至多6次公平FT入口计时及2次同目标普通反向显存参照；这也只是小样本工程验证。

完整质量曲线和更广范围确认留到实现值得验证之后。旧向量完全一致时复用可追溯旧曲线；向量改变不能直接继承质量结果。不会为了跨零显著性、补齐所有表格或试遍规则而自动扩量。实验需要保留失败证据，但不需要把已足够支持决策的每个候选都跑完。
'''
(A/name).write_text(text, encoding='utf-8')

state_path = A/'research_state_20260906.json'
old_bytes = state_path.read_bytes()
state = json.loads(old_bytes)
assert isinstance(state, dict)
(A/'state_recovery_20260907'/f'pre_official_data_resource_scope_{time.time_ns()}.json').write_bytes(old_bytes)
state.update(current_running_experiment=None, latest_report=name,
    latest_execution_status='local_generation_stopped_and_excluded;official_processed_data_only;bounded_screening',
    official_data_resource_policy=policy_name,
    local_generation_exclusion=summary_name,
    next_stage_status='analyze_existing_profiles_then_one_small_screen;no_new_GPU_job_started',
    next_stage=policy['next_change'],
    last_goal_turn_classification={'classification':'progress',
        'evidence':['verified stop of disallowed local-generation branch and preserved all saved costs/output hashes',
                    'official-cache requirement and small-screen execution limits recorded in active configuration'],
        'goal_complete':False})
tmp = state_path.with_suffix('.partial')
tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
tmp.replace(state_path)
cfg_path = R/'configs/pv_content_P1_development.json'
cfg = json.loads(cfg_path.read_text())
cfg['data_and_resource_scope'] = policy
cfg['next'] = policy['next_change']
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

note = f'当前执行约束：[官方数据与小预算验证]({name})。使用作者处理后原缓存；本地生成已停止并排除。现有证据足以先改P1重复计算，初筛限两个原开发样本、最多12次归因和0条新删除曲线；有依据再扩大，不默认重跑全量。'
for path in [A/'README.md', A/'研究目标_带符号高效归因_20260906.md', R/'README.md']:
    old = path.read_text(encoding='utf-8')
    insert = note if path.parent == A else note.replace(f']({name})', f'](docs/history/{name})')
    i = old.index('\n')
    if insert not in old:
        path.write_text(old[:i+1]+'\n'+insert+'\n'+old[i+1:], encoding='utf-8')
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = []
tree = ast.parse((A/'bootstrap_deltatrace_repository.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export')
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'curated_export', 'exec'))
export(summary_name, 'evidence/'+summary_name, redact=True)
export(policy_name, 'configs/'+policy_name, redact=True)
export(name, 'docs/history/'+name, redact=True)
for src in ['local_qwen8b_morehop_generation_20260907.py',
            'local_qwen8b_morehop_generation_protocol_20260907.json',
            'freeze_local_qwen8b_morehop_generation_20260907.py']:
    export(src, 'research/abandoned_local_generation/'+src, redact=True)
export(Path(__file__).name, 'research/reproduction_templates/'+Path(__file__).name, redact=True)
path = R/'evidence/export_manifest.json'
old = json.loads(path.read_text())
entries = {e['path']:e for e in old['artifacts']}
entries.update({e['path']:e for e in manifest})
old['artifacts'] = list(entries.values())
path.write_text(json.dumps(old, indent=2), encoding='utf-8')
print(json.dumps({'status':'verified_stopped_and_excluded', 'saved_records':44,
    'new_official_benchmark_rows':0, 'new_GPU_jobs_started':0, 'exported_artifacts':len(manifest)}))
