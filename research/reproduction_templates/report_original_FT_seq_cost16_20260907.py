"""Correct efficiency interpretation using score-identical original FT seq view."""
import ast,hashlib,json,re,time
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest();manifest=[]
s=json.loads((A/'original_FT_seq_cost16_summary_20260907.json').read_text());assert s['status']=='verified_complete' and s['seq_vector_matches_full_runner_cases']==16
assert all(v['exact_projected_score_match'] for r in s['cases'] for v in r['methods'].values())
name='原FT_seq入口公平性修正_20260907.md'
text='''# 原FT seq入口的公平性修正

结论：P1目前没有达到与有效FT计算相当的单样本效率。此前相对作者总入口的配对耗时比0.928只能描述该完整API；不能当作算法效率胜出。补充同次实测后，相对作者原函数组成的seq-only入口，P1在16/16例均更慢，配对中位耗时比1.377。

## 偏差从哪里来

原run_attribution总入口调用get_all_token_attrs，同时计算seq、row、rec三个视图。当前原指标比较实际使用seq；row/rec也计入了之前both家族的耗时。legacy对照此前直接使用原累计向量，其成本范围也应与both总入口区别说明。

更公平的FT1入口直接调用作者LLMIFRAttributionBoth.calculate_ifr_multi_hop_both，再调用作者原normalize_sum_to_one(value.attribution_matrix)，最后使用既有prompt侧求和。该表达式就是原get_all_token_attrs中的seq计算；注意力、归因规则、模型前向和归一化均执行原函数。无需自行实现另一套attention或归因近似。

额外NI0 Python调用记录确认：总入口确实调用get_all_token_attrs及compute_CAGE_token_attr；seq-only调用原归一化而不执行这两个无关步骤。16例、每例4次，两种FT入口的完整seq向量全部相同；48个预定样本—方法向量也与原曲线来源完全对应。没有新质量查询，也没有修改指标、gold或P1分数。

## 同次完整成本

原16条开发样本，每种方法一次预热加三次轮转实测；下表为三次中位数。P1仍为真实默认FA端点加已验证有限传播；FT为作者原类和原函数。每次重新准备并核对完整输入，所有捕获、传播、结果回传均计时。

|原样本|P1|FT总入口|FT原seq-only|P1/seq-only|
|---|---:|---:|---:|---:|
'''
for r in s['cases']:
    v=r['methods'];text+=f"|{r['dataset']} {r['idx']}|{v['finite']['median_seconds']:.3f}s|{v['both_1']['median_seconds']:.3f}s|{v['seq_1']['median_seconds']:.3f}s|{r['finite_to_seq_only_ratio']:.3f}|\n"
text+=f'''
FT仅取seq相对总入口的配对耗时比中位数{s['paired_median_seq_to_full_runner_ratio']:.4f}，约降低30.4%。同次P1相对总入口仍为{s['paired_median_finite_to_full_runner_ratio']:.4f}，但相对有效seq-only为{s['paired_median_finite_to_seq_only_ratio']:.4f}，约慢37.7%。所有16例均慢，最大约48.5%；不是只有一条异常值导致的结论。

本轮194次完整归因（64次P1、130次FT，含两个独立计费的调用观察）、194次根前向、2,304次真实层重放和2,304次额外原FA、2,304次有限算子；0次VJP和0次原质量查询。归因计时合计{s['sum_attribution_timer_seconds']:.3f}s，完整作业{s['job_elapsed_seconds']:.3f}s。紧凑JSON已实际接入本轮冻结study，完整数据仍保留；编码的独立字节核验见cost_artifact_serialization_summary_20260907.json。原模型计算未由保存格式改变。

## 修正后的下一步

保持P1的归因公式与开发质量候选。它当前每例的根前向处理两个端点，随后又对这两个端点逐层重放；因此每层实际执行四条端点轨迹，另有获取公开FA元数据的调用。这里还有可以审查的重复计算。下一步优先评估原生激活缓存/现成重算机制，减少昂贵矩阵运算的重复执行；仍以同批普通原生输入反向为显存参照，实际测量，不能靠平方级缓存换速度或把缓存值当成新反事实。

同时将真实B4核验扩展到完整原16条，以这个seq-only入口作为FT成本参照，并补齐0—3跳的同范围高效入口；重新检查原指标和逐样本退步。此前四例B4的吞吐收益值得利用，但不能跨作业套用倍率，直接宣布已抵消当前37.7%差距，也不能以批吞吐替代单样本延迟。

原开发质量优势仍成立：本轮评分向量未变；它仍不是独立确认。NIAH q2-100与Morehop95均有历史使用记录，新独立NIAH来源尚需核对冻结；新MoreHopQA原来源/作者采样脚本已部署并预检，仍缺作者所用生成/评判API配置，实际调用0。

符号也未被成本修正解决：P1负分不等于原输入删除方向，开发诊断中的NI强负分同号仅30/64。目标保持进行中，不因方法已实现、质量开发门槛通过或某个接口更慢而宣告完成。

证据：original_FT_seq_cost16_summary_20260907.json、original_FT_seq_cost16_numeric_20260907.json、冻结协议/study和独立核验脚本。原结果SHA256：{s['raw_sha256']}。
'''
(A/name).write_text(text,encoding='utf-8')
def keep(path,value):
    nl='\r\n' if b'\r\n' in path.read_bytes() else '\n';path.write_bytes(value.replace('\r\n','\n').replace('\n',nl).encode('utf-8'))
note=f'最新公平性修正：[原FT seq入口]({name})。用作者原函数省去无关row/rec视图后，16例完整FT分数相同、耗时约降30.4%；P1相对该入口反而慢37.7%（配对中位数，16/16均慢）。此前0.928仅为相对作者总入口，不构成效率胜出。下一步保留P1，减少原生重放并验证完整B4；独立质量与符号目标继续。'
for rel in ['README.md','研究目标_带符号高效归因_20260906.md']:
    path=A/rel;old=path.read_text();i=old.index('\n')
    if note not in old:keep(path,old[:i+1]+'\n'+note+'\n'+old[i+1:])
path=A/'无梯度带符号归因_连续研究记录_20260906.md'
if note not in path.read_text():keep(path,path.read_text()+'\n\n'+note+'\n')
path=R/'README.md';old=path.read_text();i=old.index('\n');publicnote=note.replace(f']({name})',f'](docs/history/{name})')
if publicnote not in old:keep(path,old[:i+1]+'\n'+publicnote+'\n'+old[i+1:])
path=R/'configs/pv_content_P1_development.json';cfg=json.loads(path.read_text())
cfg['seq_only_original_FT_cost']={'raw_sha256':s['raw_sha256'],'B1_development_only':True,'same_full_seq_vector_cases':16,'paired_median_ratio_P1_to_seq_only':s['paired_median_finite_to_seq_only_ratio'],'P1_slower_cases':16,'P1_efficiency_goal_met':False,'original_full_runner_not_an_algorithmic_speed_reference':True}
cfg['next']='Preserve P1 math. Audit native activation caching / existing recomputation mechanisms to reduce repeated matrix operations within the ordinary native-backward memory budget. Validate full16 trueB4 and efficient original FT0-3 seq-only controls with original curves, then freeze unused original confirmation sources; exact MoreHopQA API configuration remains pending.'
keep(path,json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
path=A/'research_state_20260906.json';raw=path.read_bytes();state=json.loads(raw);assert isinstance(state,dict)
(A/'state_recovery_20260907'/f'pre_seq_fairness_report_{time.time_ns()}.json').write_bytes(raw)
state.update(current_running_experiment=None,latest_report=name,latest_priority_note=name,latest_execution_status='original_FT_seq_fairness_verified;P1_single_example_efficiency_not_met;goal_active',
    next_stage_status='reduce_native_replay_and_validate_full_B4_against_seq_only_FT;independent_sources_pending',next_stage=cfg['next'],
    original_FT_seq_cost16={'status':'verified_complete','summary':'original_FT_seq_cost16_summary_20260907.json','raw_sha256':s['raw_sha256']},
    future_checkpoint_serialization='stdlib_json_compact_preserving_entire_payload;already_used_in_seq_cost16',
    last_goal_turn_classification={'classification':'progress','evidence':['proved original FT unused-view timing bias and measured score-identical seq-only control','P1 1.377x valid single-example FT cost now defines optimization target','conditional sign failures and exhausted original caches verified','authorized original sampler deployed; compact record serialization implemented'],'goal_complete':False})
tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
tree=ast.parse((A/'bootstrap_deltatrace_repository.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'curated_export','exec'))
for file in ['build_original_FT_seq_cost16_20260907.py','original_FT_seq_cost16_20260907.py','original_FT_seq_cost16_protocol_20260907.json','verify_original_FT_seq_cost16_20260907.py','report_original_FT_seq_cost16_20260907.py']:export(file,'research/reproduction_templates/'+file,redact=True)
for file in ['original_FT_seq_cost16_summary_20260907.json','original_FT_seq_cost16_numeric_20260907.json']:export(file,'evidence/'+file,redact=True)
export(name,'docs/history/'+name,redact=True)
path=R/'evidence/export_manifest.json';old=json.loads(path.read_text());entries={x['path']:x for x in old['artifacts']};entries.update({x['path']:x for x in manifest});old['artifacts']=list(entries.values());path.write_text(json.dumps(old,indent=2))
print('Published score-identical original FT seq-only control and corrected efficiency conclusion.')
