"""Publish completed conditional sign evidence without changing the candidate."""
import ast, hashlib, json, re, time
from pathlib import Path
A=Path(__file__).resolve().parent; R=A.parent/'DeltaTrace'
sha=lambda b:hashlib.sha256(b).hexdigest(); manifest=[]
s=json.loads((A/'finite_P1_conditional_sign_summary_20260907.json').read_text())
assert s['status']=='verified_complete' and s['selected_tokens']==384
assert sum(c['conditional_sign_reversals'] for c in s['cases'])==201
assert all(v['maximum_absolute_repeat_score_change']==0 and v['token_values_repeat_exact'] for c in s['cases'] for v in c['baseline_checks'].values())
name='P1有限分配_条件符号诊断_20260907.md'
report='''# P1有限分配的条件符号诊断

结论：当前P1仍有原开发集的质量优势，但它的负分不能直接解释成“从原输入删掉该token会提高目标分数”。本轮未改候选、排序或指标，未独立确认；为避免把有限分配守恒误称删除方向，补做真实模型的条件有限效应诊断。

## 定义和执行

固定原16条开发样本（NI0—7、MH0—7）、原生成轨迹、可归因token集合E、EOS替换和实际默认FA模型。G32为响应token的FP32 log-softmax值用FP64求和。对每个样本固定选择P1最大的8个正分、最小的8个负分，以及其余位置中按种子730907均匀抽取的8个token；不读取gold或删除曲线。

参照为s({j}|A)=G(I_A)−G(I_{A∪{j}})。clean条件A为空；eos条件A=E去掉j。因此前者是原输入单独删除，后者是其余可归因内容均被替换时单独恢复。两者都是原模型真实前向，未把混合端点传播算子充作模型反事实。响应、不可归因上下文和原生模型实现均保留。

每个背景使用真实B4前向；基线与干预按同一batch lane配对，6组干预共享实际基线，结束时重测基线。16例全部基线重复的逐token分数相同、漂移为0；某些样本的不同lane仍有差异，原值完整保存，没有修成相同值。这证明本次基线复用的稳定性，不是对所有扰动数值误差的普遍上界。

## 结果

下表每行64个token，是特意选择的开发诊断，不是全体token的准确率。正负均与P1原分配比较；“条件反号”比较两个真实有限效应。

|任务 / 选择|clean同号|eos同号|两个条件的真实效应反号|
|---|---:|---:|---:|
'''
for ds,groups in s['groups'].items():
    for kind,g in groups.items():
        report+=f"|{ds} / {kind}|{g['clean']['same_sign_count']}/64|{g['eos']['same_sign_count']}/64|{g['sign_changes_between_conditions']}/64|\n"
report+='''
总共201/384（52.34%）个token的真实效应随这两个条件反号。因此，一个不标明条件的普遍删除方向本身就不成立；本次现象不是置信区间跨零导致的判定问题。不能据此断言全部反号都超过混合精度误差上界，因为尚无这样的逐干预误差界。

强正分与clean方向的同号率为NI81.25%、MH82.81%，同号位置覆盖约94.71%、95.08%的绝对真实效应量；强负分在NI的clean同号仅46.88%，MH为71.88%。强负分在eos条件更常同号（78.13%、75.00%），仍不等于逐token保证。少数大效应会主导效应量指标，不能用它掩盖反号计数。

这不否定既有needle、RISE/MAS排序结果，也不证明当前符号目标完成。有限分配表达的是指定传播规则下的两端点贡献分配；其符号与条件有限效应是不同的可检验对象。

## 成本与下一决策

实际执行257次B4原生前向、1,028条轨迹、0次VJP、0次归因传播、0次原质量评测查询。完整作业135.689秒，前向及目标评分计时合计69.781秒，最大记录峰值19.721GB；总作业时间包括检查点读取和产物写出，分段计时不含返回CPU后的序列化。NI0 profile确认36次真实默认FA前向，无有限扩展或反向内核。

下一步保持P1质量候选，补齐同次FT0—3完整成本，冻结独立确认的质量、成本及符号主张。若需要输出可验证的删除方向，应在预先固定的少量位置测实际条件效应，并明确标记未测位置；这只是待评估方案，尚未实现，也不能以局部符号接口代替整项目的质量与成本目标。不得把两种背景的结果事后挑选成每个token都正确的符号。

新的MoreHopQA来源及作者采样脚本已准备，仍缺原生成/评判API配置；本诊断不消耗新确认样本。研究目标保持进行中。

证据：finite_P1_conditional_sign_summary_20260907.json、finite_P1_conditional_sign_numeric_20260907.json，以及固定协议、实际study和独立核验脚本。原始结果SHA256：f0785a39a95bc7eff0a6a8bed5a655f269edc54c7690f4492b5aabd4b6d3ac7b。
'''
(A/name).write_text(report,encoding='utf-8')
def keep(path,value):
    nl='\r\n' if b'\r\n' in path.read_bytes() else '\n'
    path.write_bytes(value.replace('\r\n','\n').replace('\n',nl).encode('utf-8'))
note=f'最新符号核验：[P1条件符号诊断]({name})。384个所选开发token中201个在两种真实干预条件下反号；NI强负分与原输入删除方向仅30/64同号。P1排序优势不等于可靠逐token删除方向；候选不变，独立确认与当前FT完整成本继续。'
for rel in ['README.md','研究目标_带符号高效归因_20260906.md']:
    path=A/rel; old=path.read_text(); i=old.index('\n')
    if note not in old: keep(path,old[:i+1]+'\n'+note+'\n'+old[i+1:])
path=A/'无梯度带符号归因_连续研究记录_20260906.md'
if note not in path.read_text():keep(path,path.read_text()+'\n\n'+note+'\n')
path=R/'README.md';old=path.read_text();i=old.index('\n')
publicnote=f'最新符号核验：[P1条件有限效应](docs/history/{name})。384个开发token中201个随干预条件反号；不能把P1负分直接当作原输入删除方向。原质量候选保持，符号与独立确认尚未完成。'
if publicnote not in old:keep(path,old[:i+1]+'\n'+publicnote+'\n'+old[i+1:])
path=R/'configs/pv_content_P1_development.json';cfg=json.loads(path.read_text())
cfg['conditional_sign_diagnostic']={'raw_sha256':s['raw_sha256'],'development_only':True,'selected_tokens':384,'effects':768,'context_sign_reversals':201,'NI_strong_negative_clean_same_sign':[30,64],'universal_token_deletion_sign_validated':False,'candidate_changed':False}
keep(path,json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
path=A/'research_state_20260906.json';raw=path.read_bytes();state=json.loads(raw);assert isinstance(state,dict)
(A/'state_recovery_20260907'/f'pre_sign_report_{time.time_ns()}.json').write_bytes(raw)
state.update(current_running_experiment=None,latest_report=name,latest_execution_status='conditional_sign_diagnostic_verified;goal_active',next_stage_status='fresh_FT_complete_cost_pending;original_API_configuration_pending',conditional_sign_diagnostic={'status':'verified_complete','summary':'finite_P1_conditional_sign_summary_20260907.json','raw_sha256':s['raw_sha256'],'negative_deletion_direction_claim_supported':False},last_goal_turn_classification={'classification':'progress','evidence':['verified and published complete conditional sign diagnostic; preserved failure of unconditional negative deletion interpretation'],'goal_complete':False})
tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
tree=ast.parse((A/'bootstrap_deltatrace_repository.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'curated_export','exec'))
for file in ['build_finite_P1_conditional_sign_review_20260907.py','finite_P1_conditional_sign_review_20260907.py','finite_P1_conditional_sign_review_protocol_20260907.json','verify_finite_P1_conditional_sign_review_20260907.py','report_conditional_sign_20260907.py']:
    export(file,'research/reproduction_templates/'+file,redact=True)
for file in ['finite_P1_conditional_sign_summary_20260907.json','finite_P1_conditional_sign_numeric_20260907.json']:
    export(file,'evidence/'+file,redact=True)
export(name,'docs/history/'+name,redact=True)
path=R/'evidence/export_manifest.json';old=json.loads(path.read_text());entries={x['path']:x for x in old['artifacts']};entries.update({x['path']:x for x in manifest});old['artifacts']=list(entries.values());path.write_text(json.dumps(old,indent=2))
print('Published conditional sign diagnostic:',len(manifest),'artifacts; candidate unchanged.')
