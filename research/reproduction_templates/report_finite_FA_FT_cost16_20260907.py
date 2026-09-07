"""Connect fresh full costs, actual quality reuse and confirmation-source status."""
import ast,hashlib,json,re,statistics,time
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest();manifest=[]
s=json.loads((A/'finite_FA_FT_cost16_summary_20260907.json').read_text());assert s['status']=='verified_complete'
exposure=json.loads((A/'original_cache_exposure_summary_20260907.json').read_text())
assert all(not r['not_proven_exposed_by_these_sources'] for r in exposure['datasets'].values())
name='FA有限P1_原FT同次完整成本_20260907.md'
text='''# FA有限P1与原FlashTrace的同次完整成本

本轮保持P1公式、真实模型FA和有限扩展不变，补齐全部原FT both0—3和legacy0—3各自独立计算的成本。使用原NI0—7、MH0—7开发样本；同一模型、FP16、固定响应及原可归因范围。FT运行作者原实现与eager配置，P1运行真实默认FA及明确标识的有限传播扩展；没有把FT换成自写近似实现。

后续已完成[原FT seq入口公平性修正](原FT_seq入口公平性修正_20260907.md)：本表both总入口包含未使用的row/rec视图。用作者原函数仅取相同seq后，FT更快，P1相对有效入口慢37.7%。下方0.928仅描述原完整API，不能当作算法效率胜出；保留原实测与预算上界作为该入口范围的记录。

每例9个方法各1次预热、3次轮转顺序实测；NI0另做P1和FTboth1完整profile。计时包含每次实际输入准备、端点前向、重放/传播、分数投影和CPU结果回传，不共享归因缓存。预热、profile和库初始化另列，不混入稳定耗时中位数。记录实际root输入，逐次核对FT为原输入、P1为真实EOS/原输入双端点。EOS取自检查点tokenizer配置和词表，源哈希匹配。

## 同次单样本延迟

这里是每个样本B1延迟；P1根调用的物理B2是两个端点，不是两个样本。真实B4吞吐证据另见先前批处理报告，不能互换。

|原样本|P1中位耗时|原FTboth1中位耗时|P1/FT1|P1峰值|FT1峰值|
|---|---:|---:|---:|---:|---:|
'''
for r in s['cases']:
    f=r['methods']['finite'];b=r['methods']['both_1']
    text+=f"|{r['dataset']} {r['idx']}|{f['median_seconds']:.3f}s|{b['median_seconds']:.3f}s|{r['finite_to_both1_median_ratio']:.3f}|{f['peak_bytes']/1e9:.3f}GB|{b['peak_bytes']/1e9:.3f}GB|\n"
text+=f"\n16例配对耗时比中位数{s['finite_to_both1_paired_median_ratio']:.4f}。仍慢于FTboth1的样本：{s['finite_slower_than_both1_cases']}。保留全部3次实测和冷启动，不删异常高值来制造优势。峰值含模型权重，GB按十进制。该比较不能证明每条输入、任意模型或任意设备均不退化。\n"
text+=f"\n首例冷启动计时：P1为{s['warmup_seconds_by_method']['finite']['first_case']:.3f}s，FTboth1为{s['warmup_seconds_by_method']['both_1']['first_case']:.3f}s。16例P1预热累计{s['warmup_seconds_by_method']['finite']['sum']:.3f}s，包含实际首次编译/准备，不计入上述稳定运行中位数，但仍属于本轮真实成本。\n"
text+='''
## 原质量与相同延迟预算

本轮新增质量查询为0。先前完整原曲线已经独立核验；只有本轮预定repeat1的完整投影向量与原记录完全一致，才允许关联原指标。该条件用于证明缓存可复用，不是要求FA优化必须逐位相同；有不同就需要新原曲线。原质量使用作者完整固定响应含EOS的评分，作者seq_attr视图也是对全部生成行聚合；P1账本/符号另用同一数学目标的G32，不冒充原低精度指标。

|数据集|方法|均值耗时|原恢复率|原RISE|原MAS|
|---|---|---:|---:|---:|---:|
'''
def metric(v):return '未确认' if v is None else f'{v:.6f}'
for ds,g in s['dataset_summary'].items():
    for method,v in g['methods'].items():
        q=v['quality'];text+=f"|{ds}|{method}|{v['mean_seconds']:.3f}s|{metric(q['recovery']['mean']) if ds=='niah_mq_q2' else '不适用'}|{metric(q['rise']['mean'])}|{metric(q['mas']['mean'])}|\n"
exact=sum(v['projected_score_exactly_matches_quality_parent'] for r in s['cases'] for v in r['methods'].values())
text+=f'\n原指标复用条件满足{exact}/144个样本—方法组合；原曲线和来源哈希在摘要中逐项保存。这是开发质量的同次成本补充，不是新独立确认。恢复率越高越好，RISE/MAS越低越好；不把RISE/MAS当作负向符号正确率。\n'
figure='finite_FA_FT_cost16_quality_cost_20260907.png'
if (A/'figures'/figure).exists():
    figure_manifest=json.loads((A/'figures/finite_FA_FT_cost16_figure_manifest_20260907.json').read_text())
    assert figure_manifest['summary_sha256']==sha((A/'finite_FA_FT_cost16_summary_20260907.json').read_bytes())
    text+=f'\n![开发质量与同次完整成本]({(A/"figures"/figure).as_posix()})\n'
text+='\n还按每例P1实际中位延迟筛出预算内的FT变体，再分别取质量最优者，得到严格的事后预算对照上界。它知道每例评测结果，所以不是可部署的FT选择器，不能隐瞒这一额外信息。逐例预算、所选变体、配对差及未获得可比较指标的样本全部保存。\n'
for ds,g in s['dataset_summary'].items():
    text+=f"\n{ds}："
    for m,v in g['budget_matched_oracle_deltas'].items():
        if v['count']:text+=f"{m}的P1减预算内最优FT均值差={metric(v['mean'])}（{v['count']}例）；"
    text+='\n'
text+=f'''
实际合计578次独立归因：65次P1（含profile）、513次FT；根前向578次，P1根端点轨迹130条，FT轨迹513条；额外原层重放2,340次（4,680条端点层轨迹）、额外原FA调用2,340次、有限算子2,340次（每次3个内核）；0次原生VJP、0次原质量查询。库初始化{s['library_initialization_seconds']:.3f}s；全部归因计时合计{s['sum_all_measured_scope_timer_seconds']:.3f}s；完整作业{s['job_elapsed_seconds']:.3f}s，含重复写审查产物、加载/核验检查点等，不能把这个总时间当作生产归因延迟。首次profile中实际内核与逐次调用数均单独核验。

## 独立确认与连续决策

新的MoreHopQA采样脚本、首256条未重叠来源和协议已部署，实际服务器预检验证源哈希，尚缺qwen3-235b-a22b-2507与deepseek-v3-1-terminus的API base及安全凭据加载配置。实际API调用0；不私自替换原生成或评判模型。

身份审查补充确认：NIAH q2第8—15条也出现在历史VJP实验记录中，prompt/target哈希与作者缓存匹配。虽然历史目录名带100ex，现存manifest实际只有28条，不能按目录名当作100条完成；但与后续经核验的16—31和32—99结果合并后，q2的100条均有使用证据。Morehop95条同样全部已用。这项审查只证明历史使用，不采信旧VJP质量或实现，不把它们重新归类为新确认数据。

下一步保持P1这一质量候选，优先验证真实B4在全部16条开发输入上的成本与原指标，并为新确认集固定批处理安排。NIAH独立来源需要从作者其他原任务/原流程核对并冻结，不能继续使用q2历史缓存冒充独立确认。随后使用新来源做当前候选与全部原FT对照的独立比较，不根据确认结果调参。

负向符号仍有实质边界：刚完成的条件诊断中NI强负分与原输入删除同号仅30/64，384个所选token中201个随两种背景反号。当前P1分配不等于普遍删除方向；成本改善不能替代符号和独立质量验证。研究目标未完成。

证据：finite_FA_FT_cost16_summary_20260907.json、finite_FA_FT_cost16_numeric_20260907.json、原冻结study/协议/核验脚本、original_morehop_sampler_preflight_20260907.json和original_cache_exposure_summary_20260907.json。原成本结果SHA256：{s['raw_sha256']}。
'''
io=json.loads((A/'cost_artifact_serialization_summary_20260907.json').read_text());assert io['status']=='verified_complete' and io['source_sha256']==s['raw_sha256']
text+='''
## 实验记录的额外开销

本轮完整作业40.88分钟，而归因分段计时合计7.24分钟；不能把两者混为模型延迟。为了定位可消除的测试开销，作业结束后对真实最终结果做一次同机标准JSON写入比较，0次模型/GPU调用：缩进版108.42MB、编码加原子写入2.641s；紧凑版30.72MB、0.442s。两种解码后的全部数据完全一致，原结果文件哈希保持不变，并在本地独立重建两种编码的完整字节哈希。

单次最终记录的序列化约快5.98倍，证明后续实验可以直接复用标准json的紧凑编码减少保存开销；未实测每次历史保存，不能将全部33.64分钟差额都归因于JSON。下一份批处理实验应将这一写法在冻结源码前接入，同时保留所有数组、调用与曲线；本轮已冻结study和原结果不追溯改写。这是审查记录开销优化，不是归因算子加速。
'''
(A/name).write_text(text,encoding='utf-8')
def keep(path,value):
    nl='\r\n' if b'\r\n' in path.read_bytes() else '\n';path.write_bytes(value.replace('\r\n','\n').replace('\n',nl).encode('utf-8'))
note=f"最新成本核验：[FA有限P1与原FT同次成本]({name})。16例9方法各自完整重跑；P1/FTboth1配对耗时比中位数{s['finite_to_both1_paired_median_ratio']:.3f}，仍慢的样本全部保留。原质量仅在完整分数相同后关联，非独立确认；NIq2-100与MH95均已历史使用，新MH预检通过但缺原API配置。"
for rel in ['README.md','研究目标_带符号高效归因_20260906.md']:
    path=A/rel;old=path.read_text();i=old.index('\n')
    if note not in old:keep(path,old[:i+1]+'\n'+note+'\n'+old[i+1:])
path=A/'无梯度带符号归因_连续研究记录_20260906.md'
if note not in path.read_text():keep(path,path.read_text()+'\n\n'+note+'\n')
path=R/'README.md';old=path.read_text();i=old.index('\n');publicnote=note.replace(f']({name})',f'](docs/history/{name})')
if publicnote not in old:keep(path,old[:i+1]+'\n'+publicnote+'\n'+old[i+1:])
path=R/'configs/pv_content_P1_development.json';cfg=json.loads(path.read_text())
cfg['fresh_original_FT_cost']={'raw_sha256':s['raw_sha256'],'B1_development_only':True,'paired_median_ratio_to_both1':s['finite_to_both1_paired_median_ratio'],'slower_cases':s['finite_slower_than_both1_cases'],'exact_quality_reuse_combinations':exact,'independent_quality_confirmed':False}
cfg['next']='Validate all16original development B4 batches and original curves under fixed complete costs; audit and freeze unused original NIAH source. New MoreHopQA original sampler deployed and preflight verified, exact API configuration pending. Do not reuse NIq2-100 or MH95 as independent.'
keep(path,json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
path=A/'research_state_20260906.json';raw=path.read_bytes();state=json.loads(raw);assert isinstance(state,dict)
(A/'state_recovery_20260907'/f'pre_fresh_cost_report_{time.time_ns()}.json').write_bytes(raw)
state.update(current_running_experiment=None,latest_execution_status='fresh_FT_cost16_verified;original_cache_exposure_audited;goal_active',latest_report=name,
    next_stage_status='full_development_B4_quality_and_cost_pending;new_independent_NI_source_pending;original_MH_API_configuration_pending',
    next_stage=cfg['next'],finite_FA_FT_cost16={'status':'verified_complete','summary':'finite_FA_FT_cost16_summary_20260907.json','raw_sha256':s['raw_sha256']},
    all_original_primary_cache_exposure={'niah_mq_q2':100,'morehopqa':95,'summary':'original_cache_exposure_summary_20260907.json'},
    last_goal_turn_classification={'classification':'progress','evidence':['fresh isolated FT0-3/P1 complete costs verified','conditional sign failures preserved and pushed','new original sampler deployed and preflight verified','NI8-15 historical exposure proved by matching identity'],'goal_complete':False})
state['historically_used_confirmation_not_pristine']['niah_mq_q2']=sorted(set(state['historically_used_confirmation_not_pristine']['niah_mq_q2'])|set(range(8,16)))
tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
tree=ast.parse((A/'bootstrap_deltatrace_repository.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'curated_export','exec'))
for file in ['build_finite_FA_FT_cost16_20260907.py','finite_FA_FT_cost16_20260907.py','finite_FA_FT_cost16_protocol_20260907.json','verify_finite_FA_FT_cost16_20260907.py','audit_original_cache_exposure_20260907.py','report_finite_FA_FT_cost16_20260907.py','plot_finite_FA_FT_cost16_20260907.py','probe_cost_artifact_serialization_20260907.py','verify_cost_artifact_serialization_20260907.py']:
    export(file,'research/reproduction_templates/'+file,redact=True)
for file in ['finite_FA_FT_cost16_summary_20260907.json','finite_FA_FT_cost16_numeric_20260907.json','original_cache_exposure_summary_20260907.json','original_morehop_sampler_preflight_20260907.json','cost_artifact_serialization_summary_20260907.json']:
    export(file,'evidence/'+file,redact=True)
export(name,'docs/history/'+name,redact=True)
if (A/'figures'/figure).exists():
    for file in [figure,figure.replace('.png','.svg'),'finite_FA_FT_cost16_figure_manifest_20260907.json']:export('figures/'+file,'figures/'+file)
path=R/'evidence/export_manifest.json';old=json.loads(path.read_text());entries={x['path']:x for x in old['artifacts']};entries.update({x['path']:x for x in manifest});old['artifacts']=list(entries.values());path.write_text(json.dumps(old,indent=2))
print('Published full fresh cost evidence:',name)
