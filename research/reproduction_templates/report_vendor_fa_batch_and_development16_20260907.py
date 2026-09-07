"""Record completed original development and actual minibatch evidence."""
import json,time
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace'
s=json.loads((A/'vendor_fa_development16_summary_20260907.json').read_text())
b=json.loads((A/'vendor_fa_batch_summary_20260907.json').read_text())
e=json.loads((A/'batch_endpoint_diagnostic_summary_20260907.json').read_text())
assert all(x['status']=='verified_complete' for x in [s,b,e])
name='FA有限传播_完整16条与真实批处理_20260907.md'
text='''# FA 有限传播：完整16条与真实 mini-batch

此前完整16条开发测试采用每次一个原样本。归因中的物理B2是该样本的EOS与原输入两个端点，原曲线评分B1。随后单独完成四个原样本的真实B2/B4归因，并以B4运行原评分器；两轮证据分别保存，不能将后者反写成前者已批处理。

## 四个原样本的真实批量归因

选择原NI0、NI3、NI6、MH1，总长度分别为601、586、607、367。B2一次两个不同样本，B4一次四个不同样本；对应真实模型端点批维分别为4和8。完整固定响应之后右补EOS，因果前向、每个样本的有效响应种子及归因位置保持，padding贡献为零。

每种模式一次预热、三次交错实测。以处理全部四个样本的总耗时取中位数，包含实际输入准备、捕获、层重放、额外原FA调用、有限传播、转换与返回；冷编译/预热独立保留。

|归因样本批量|完成四条耗时|平均每条|相对逐条吞吐|完整峰值显存|
|---|---:|---:|---:|---:|
'''
for mode,label in [('single','B1'),('batch2','B2'),('batch4','B4')]:
    c=b['costs'][mode]
    text+=f"|{label}|{c['median_seconds_for_all_four']:.4f}s|{c['seconds_per_example']:.4f}s|{c.get('throughput_speedup_vs_four_singles',1):.3f}×|{c['peak_bytes']/1e9:.3f}GB|\n"
text+='''
GB按十进制，包含模型权重。B4吞吐提高43.0%，总耗时下降30.1%，并不是四倍加速。本轮没有同批量普通反向参照，因此不能据此声称B4显存低于普通B4反向。B4端点长度仅到607，不能外推长rollout或任意batch。

B4真实profile捕获108次默认FA前向和108个明确命名的有限传播内核，模型forward/FA没有替换。所有B2/B4样本独立向量、padding零值、实际端点批维、源文件和算子缓冲契约均核验。原16条已完成的B1扩展移除了全局attention N×N中间矩阵；这里没有重新引入它们。

## 原曲线批处理与质量

评分采用原基准的实际eager后端，归因仍是默认FA。原函数只在两处真实评分调用位置暂停、收取实际原生返回的token logprob后恢复；反向还原AST证明删除逻辑、RISE/MAS及求和未改。没有占位分数或另写指标。按精确长度分桶，本轮每个原样本的四种方法曲线恰好成B4；这是删除状态批处理，不是四条异长原样本在评分阶段强行padding。

16条原曲线各21点，共336条实际评分轨迹、84次B4物理前向，完整评分耗时21.51s。全部删除分组、排序、density、原指标和needle恢复独立重算，最大指标重建差9.1e−8。FT分数取固定历史原对照，曲线本轮重算；FT归因本轮没有重跑。

三个needle恢复率在B1/B2/B4下均相同，分别42.50%、47.37%、45.95%。B4对比B1：四例最大RISE绝对变化3.03e−6，MAS绝对变化5.21e−5；B2有一例删除排序改变，RISE下降0.00269、MAS下降0.00524。这些不作为新的科学质量改进。B4完整归因向量相对L2差为0.316%—0.571%；四例反号位置为2、1、3、1个，原绝对贡献量占比最高0.00261%。符号原样保留，未分配量逐例保存。

首次独立核验在“各方法端点必须逐位相同”的检查失败，原脚本和失败收据保留。追加诊断直接调用未改动的原eager评分器，不经过本项目调度/归因：同一MH1输入复制四行，clean分数依次−118.375、−118.1875、−118.3125、−118.375；删除端点为−216.625、−216.625、−216.75、−216.5。两次重复的全部token值相同，且逐项复现原批测端点。故该差异可在现有原生评分后端独立复现，不能归咎于新增调度器；尚未定位具体底层算术原因。修订核验使用每条曲线真实端点重建原指标，保留全部行间差异，没有统一端点、改曲线或修改精度。后续很小的FT质量差不能越过此数值影响直接解释为算法优势。

## 完整16条原开发集结果

这轮是NI0—7、MH0—7，显式P1与FA有限P1都重新归因，并各自重跑原删除曲线；FT0—3的both/legacy对照引用核验过的同数据历史结果。

|原指标|显式P1|FA有限P1|各指标最强FT历史对照|
|---|---:|---:|---:|
'''
for dataset,key,label in [('niah_mq_q2','recovery','NI needle↑'),('niah_mq_q2','rise','NI RISE↓'),('niah_mq_q2','mas','NI MAS↓'),('morehopqa','rise','MH RISE↓'),('morehopqa','mas','MH MAS↓')]:
    m=s['means'][dataset];values=[v[key] for k,v in m.items() if k.startswith('flashtrace')];best=max(values) if key=='recovery' else min(values)
    text+=f"|{label}|{m['dense'][key]:.6f}|{m['finite'][key]:.6f}|{best:.6f}|\n"
text+='''
原开发联合门槛通过。FA扩展保持原P1质量，不能把微小浮点指标变化说成新的方法增强。完整耗时相对同作业显式P1的配对比值中位数0.888：15/16更快，一例慢2.53%；峰值减少0—1.142GB，所有峰值低于同作业普通输入反向参照至少1.453GB。普通参照为B1原生FA输入反向并使用原生logits_to_keep输出头，归因为B2端点完整输出头，两者浮点分数差已记录；这是显存参照，不是FT耗时比较。新路径最大向量相对差0.0634%，最大相对未分配量0.1967%。

完整16条预算818次根前向、16次输入VJP、130次有限传播、4680次层重放和4680次额外原FA调用、2340次有限算子。批测预算113次根前向、0VJP、29次有限传播、1044次层重放和1044次额外原FA调用、1044次有限算子。追加端点诊断另计16次原生前向、40条轨迹、0归因。物理调用与端点/评分轨迹没有混算。

## 接续决定与范围

保持已验证的content-P1和真实默认模型FA。短输入下一轮测试可用B4归因与等长B4评分，记录实际batch并采用所有方法相同调度；长输入先保守控制batch，待同批普通反向参照及原长输入验证后扩大。完整16条是开发结果，四例是批处理兼容/资源核验；都不代替独立质量确认、长输入验证或单token因果符号证明。历史对称无退步附加门槛失败仍保留，用户接受的needle小幅取舍未改。研究目标仍未完成。

证据：`vendor_fa_development16_summary_20260907.json`、`vendor_fa_batch_summary_20260907.json`、`batch_endpoint_diagnostic_summary_20260907.json`及对应原始数值、协议、核验脚本；所有哈希由仓库manifest记录。
'''
(A/name).write_text(text,encoding='utf-8')
def write_keep(path,value):
    nl='\r\n' if b'\r\n' in path.read_bytes() else '\n'
    path.write_bytes(value.replace('\r\n','\n').replace('\n',nl).encode('utf-8'))
note=f'最新核验：[完整16条与真实批处理]({name})。完整16条原开发门槛通过；真实B2/B4归因吞吐提高25.9%/43.0%，B4完整峰值21.70GB，三个needle恢复不变；原曲线B4、原生端点差异单独核验。研究目标未完成。'
for rel in ['README.md','研究目标_带符号高效归因_20260906.md']:
    path=A/rel;old=path.read_text();idx=old.index('\n');write_keep(path,old[:idx+1]+'\n'+note+'\n'+old[idx+1:])
path=A/'无梯度带符号归因_连续研究记录_20260906.md';write_keep(path,path.read_text()+'\n\n'+note+'\n')
path=R/'README.md';old=path.read_text()
old=old.replace('**FA 框架有限扩展已完成三个原样本整网核验；完整16条开发扩展运行中。**','**FA 框架有限扩展已完成原16条整网开发核验；四个原样本的真实 B2/B4 归因也已完成。**')
old=old.replace('|FA 双端点有限传播加速|原三例整网耗时下降8.55%—20.13%；完整16条扩展运行中|[整网首验](docs/history/FA有限传播整网接入_原三例首验_20260907.md)|',f'|FA 双端点有限传播加速|原16条配对耗时中位数下降11.2%，原开发门槛通过|[完整核验](docs/history/{name})|\n|真实多样本归因|四例B4吞吐1.430×，完整峰值21.70GB，needle恢复不变|[批处理核验](docs/history/{name})|')
start=old.index('当前研究配置为');end=old.index('\n## 运行核心代码',start)
old=old[:start]+f'''当前研究配置为 [`content_P1`](configs/pv_content_P1_development.json)。FA有限版本在原16条开发集的NI恢复率77.52%（最强FT历史对照59.33%），NI/MH RISE为0.05736/0.05839（最强FT分别0.06193/0.11013），MAS也较低。RISE/MAS越低越好。微小浮点变化不算新的算法提升；原对称无退步附加门槛仍记录失败，用户已接受该取舍。仍需独立质量和原长输入确认。

新有限路径已移除全局attention N×N中间矩阵，实际默认模型FA保持不变。可追溯有限扩展在 `research/prototypes/qwen_signed_secant_paired_vendor_fa.py`，多样本入口在 `research/prototypes/qwen_signed_secant_batched_vendor_fa_public.py`；明确使用固定厂商FA框架的独立库，未宣称任意设备/FA版本即插即用。

原16条采用单样本归因、B1评分。后续四例测试才是真实B2/B4归因和B4原评分，不能混称。评分的原生行间端点差异已直接复现，原曲线保留各自真实值；[完整报告](docs/history/{name})说明数值、内存、吞吐与证据边界。短输入下一轮使用B4需保持全方法相同调度，长输入batch仍需实际资源确认。
''' +old[end:]
old=old.replace('将 `core` 和 `research/runtime` 加入 `PYTHONPATH`，使用实际配置好的 Qwen3-8B 模型和完整目标 token 轨迹：','以下是保留的显式有限传播对照入口。将 `core` 和 `research/runtime` 加入 `PYTHONPATH`，使用实际配置好的 Qwen3-8B 模型和完整目标 token 轨迹：')
old=old.replace('当前 B4 数值/性能核验限于报告中的三个原样本。',f'原调度器的三例核验保留；新增四例的归因及完整原曲线批处理见[最新报告](docs/history/{name})。')
write_keep(path,old)
for rel in ['docs/fa_two_endpoint_extension.md','docs/backend_contract.md']:
    path=R/rel;write_keep(path,f'最新状态：[完整16条与真实批处理](history/{name})已完成独立核验。下方未接入/运行中状态属于此前阶段记录。\n\n'+path.read_text())
path=R/'configs/pv_content_P1_development.json';cfg=json.loads(path.read_text())
cfg.update(core_entry='qwen_signed_secant_paired_vendor_fa.propagate_paired_secant',backend='actual_installed_native_FA_endpoints_with_traceable_vendor_FA_finite_P1',FA_tiled_finite_extension_completed=True,
    FA_extension_scope='MetaX C550 pinned separate FA2.5.3 source framework library; actual installed model FA2.6.3 unchanged; original16 development and four short B2/B4 examples verified; no arbitrary platform/length claim.',
    finite_FA_development_raw_sha256=s['raw_sha256'],batch_raw_sha256=b['raw_sha256'],
    actual_example_batching={'validated':[1,2,4],'original_case_lengths':[601,586,607,367],'B4_peak_bytes':b['costs']['batch4']['peak_bytes'],'B4_same_batch_backward_memory_comparison_done':False,'candidate_batch_size_for_next_short_input_tests':4},
    next='Keep finite P1, test original longer inputs and same-batch ordinary backward memory, then independent original benchmark quality with matched batching. No bitwise precision gate; preserve native endpoint numerical effects and all failed guards.')
write_keep(path,json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
path=A/'research_state_20260906.json';raw=path.read_bytes();state=json.loads(raw);assert isinstance(state,dict)
(A/'state_recovery_20260907'/f'pre_batch_complete_{time.time_ns()}.json').write_bytes(raw)
state.update(latest_priority_note=name,current_running_experiment=None,queued_experiments=[],
    next_stage_status='full16_and_true_B2_B4_verified;direct_original_endpoint_diagnostic_verified',
    vendor_fa_development16={'status':'verified_complete','summary':'vendor_fa_development16_summary_20260907.json','raw_sha256':s['raw_sha256']},
    vendor_fa_true_batch={'status':'verified_complete','summary':'vendor_fa_batch_summary_20260907.json','raw_sha256':b['raw_sha256'],'original_cases':4,'example_batches':[1,2,4],'evaluation_batch':4},
    batch_endpoint_diagnostic={'status':'verified_complete','summary':'batch_endpoint_diagnostic_summary_20260907.json','raw_sha256':e['raw_sha256']},
    next_stage='Original long-input and same-batch backward memory checks, then independent quality with matched batching. Signed causal interpretation still unconfirmed. Preserve default native FA and traceable finite extension; goal not complete.')
t=path.with_suffix('.partial');t.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');assert isinstance(json.loads(t.read_text()),dict);t.replace(path)
print(name)
