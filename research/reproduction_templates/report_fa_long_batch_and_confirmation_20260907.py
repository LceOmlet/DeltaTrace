"""Record long-input and matched-batch cost evidence plus new-cohort readiness."""
import json,time
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace'
s=json.loads((A/'vendor_fa_original_long_cost_summary_20260907.json').read_text())
b=json.loads((A/'vendor_fa_batch_memory_summary_20260907.json').read_text())
c=json.loads((A/'original_morehop_confirmation_source_summary_20260907.json').read_text())
assert s['status']==b['status']=='verified_complete'
name='FA有限传播_原长输入与同批反向参照_20260907.md'
text='''# FA有限传播：原长输入与同批反向参照

上一阶段完成真实B2/B4归因和原曲线批处理。本轮补齐两项资源证据：FlashTrace原发布缓存的长输入，以及归因样本batch完全相同的普通原生FA输入反向。没有改动已冻结P1有限传播、厂商扩展、真实模型FA或精度设置。仍未完成独立质量与符号确认。

## 原长输入

按原缓存长度元数据事先选定NI2控制、HotpotQA最长上下文样本22、MATH最长完整轨迹样本92。完整输入与固定响应hash匹配之前不执行模型的1,243条容量清单，未截断、扩写或人为padding成“长基准”。每种方法一次预热加三次交错实测，另对MATH有限路径做完整profile；所有预热/profile成本单独计费。

|原样本 / 完整长度 / 响应长度|显式P1完整耗时|FA有限P1完整耗时|显式峰值|FA有限峰值|普通输入反向峰值|
|---|---:|---:|---:|---:|---:|
'''
for row in s['cases']:
    a=row['modes']['dense'];f=row['modes']['finite'];o=row['ordinary_backward']
    text+=f"|{row['dataset']} {row['idx']} / {row['N']} / {row['response_length']}|{a['median_seconds']:.3f}s|{f['median_seconds']:.3f}s|{a['peak_bytes']/1e9:.3f}GB|{f['peak_bytes']/1e9:.3f}GB|{o['peak_allocated_bytes']/1e9:.3f}GB|\n"
text+='''
GB按十进制，包含权重和捕获/重放/传播。两条长输入相对同作业显式P1耗时下降43.3%和44.4%，峰值减少10.121GB和7.119GB。这里的加速对照是原强P1显式实现，不是新测的FT时间。

MATH有限传播内部峰值29.998GB，端点捕获峰值30.060GB；现在整体峰值发生在捕获阶段。下一步若继续优化显存，应审查端点输出头和缓存，不能继续假定attention显式矩阵仍是当前峰值来源。根模型仍走默认FA；长MATH实际profile记录108次默认FA前向、108个明确命名的有限内核，0个带3762×3762输入的操作；15个有限入口缓冲形状与已验证源码均核验。该证据覆盖当前原长输入，不能外推任意未来长度/设备的分配器行为。

三例有限向量相对同作业显式P1的L2差分别0.0446%、0.1344%、0.2315%，符号改变3、29、0个；Hotpot改变位置占原绝对贡献量约0.00277%。保留默认混合精度，未因非逐位相同改回FP32。最大相对未分配量为MATH的1.056%（同例显式1.953%），如实保留，不能当作逐token误差界、守恒已精确实现或因果符号证明。

本轮没有重新跑长输入删除曲线，因此不宣称其解释质量已经通过。MATH旧失败记录也未改为成功：本轮是不同阶段的P1资源/数值复测，不能抹去旧对称方法57.48GB及残差失败。

## 同批量普通原生反向

仍用NI0、NI3、NI6、MH1四条原短样本。分别测4个B1、两个B2组合和一个B4组合；每个组合对有限归因与普通反向交错做一次预热加三次实测。普通反向对每个样本的原固定响应分别计算G32，再对总分执行一次真实输入VJP，参数冻结；没有训练参数梯度、基线缓存或额外损失。

普通参照复用模型原生logits_to_keep，只计算有效响应位置的并集，每行用切片视图取有效logits，未为抬高参照而物化额外logits副本。异长样本只在完整响应之后右补EOS，padding无目标分数且实际输入梯度为零。

|样本批量 / 组|FA有限归因峰值|同批普通输入反向峰值|峰值差（归因−反向）|
|---|---:|---:|---:|
'''
for row in b['groups']:
    a=row['costs']['finite'];o=row['costs']['ordinary']
    text+=f"|B{row['example_batch']} / {row['indices']}|{a['peak_bytes']/1e9:.3f}GB|{o['peak_bytes']/1e9:.3f}GB|{row['finite_peak_minus_same_batch_backward']/1e9:.3f}GB|\n"
text+='''
所有组合均满足当前显存要求；B4减少9.525GB。B4实际profile中，普通参照有36次默认FA前向和真实FA反向内核，无有限扩展；候选有108次默认FA前向及108个有限内核，无原生反向。这些都是实际执行证据，未把自行编写的有限传播称为官方backward。

B4归因约1.51s，同批普通反向约0.51s，约三倍耗时；普通反向不是FT，不能用这个参照替代与FT的完整成本比较。归因是2B双端点、完整输出头，参照是B单端点、原生选定输出头；相同数学目标下的FP16批维/头部布局分数差全部保存，没有强制修成相同数值。本结果限于测试过的组合，不能保证任意batch和长输入都满足显存门槛。

## 独立确认的接续

原MorehopQA固定缓存95条已被历史研究全部使用。用户现已明确允许从作者发布的1,118条原始MoreHopQA数据中，按作者原采样流程补充未用样本，且必须区别于原表格复现。

来源是[FlashTrace作者的table1-data-v1发布包](https://github.com/wbopan/flashtrace/releases/tag/table1-data-v1)。原来源文件与SHA256SUMS一致；实际服务器的attribution_datasets.py、sample_and_filter.py与作者固定提交075e7e4规范化逐字节一致。直接执行原数据类的类定义，证明95条旧缓存prompt和metadata与原记录对应，随后排除相同ID、完整上下文、基础问题或格式化prompt的记录：共排除295条，剩余823条，保持原始顺序与记录不变。

已经冻结首256条来源候选、目标64条接受样本、至多1,536次API尝试（含原重试规则）的采样协议。实际采样将执行未修改的作者main，保留原生成模型qwen3-235b-a22b-2507、原评判模型deepseek-v3-1-terminus及格式/span/正确性过滤。旁路记录真实API调用、返回内容、可用usage与部分接受缓存，密钥不进入命令行或日志。尚未生成新轨迹，实际API调用为0；服务器未配置这两个模型的API，已向用户请求base URL与安全的凭据加载位置。

方法实现保持冻结；新缓存完整生成并经独立核验后，才可运行预冻候选/FT0—3比较。相关问题需按原上下文/基础问题分组报告不确定性，不能把64条相关问题假称64个独立来源。采样本身不代表独立质量确认；符号主张和相同实际预算的FT比较仍须完成。

本轮资源作业合计86次根前向、32次原生输入VJP、54次有限传播、1,944次层重放与1,944次额外原FA调用、1,512次有限算子。没有质量查询、没有生成或judge调用。

证据：vendor_fa_original_long_cost_summary_20260907.json、vendor_fa_batch_memory_summary_20260907.json、original_morehop_confirmation_source_summary_20260907.json及对应原数值、协议、源码和核验脚本。研究目标继续，未达完成条件。
'''
(A/name).write_text(text,encoding='utf-8')
def write_keep(path,value):
    nl='\r\n' if b'\r\n' in path.read_bytes() else '\n';path.write_bytes(value.replace('\r\n','\n').replace('\n',nl).encode('utf-8'))
note=f'最新核验：[原长输入与同批反向参照]({name})。原3470/3762长度的完整有限归因耗时下降43.3%/44.4%，峰值减少10.12/7.12GB；真实B4归因21.77GB低于同批普通反向31.29GB。新MoreHopQA原来源已排除历史重叠，采样协议准备完成、尚缺原API配置；独立质量与符号目标未完成。'
for rel in ['README.md','研究目标_带符号高效归因_20260906.md']:
    path=A/rel;old=path.read_text();i=old.index('\n');write_keep(path,old[:i+1]+'\n'+note+'\n'+old[i+1:])
path=A/'无梯度带符号归因_连续研究记录_20260906.md';write_keep(path,path.read_text()+'\n\n'+note+'\n')
path=R/'README.md';old=path.read_text();i=old.index('\n');write_keep(path,old[:i+1]+f'\n最新资源核验：[原长输入与同批反向参照](docs/history/{name})。原长输入归因耗时下降43%—44%，真实B4显存低于同批普通FA反向；不等于长输入质量确认。新增原MoreHopQA来源与采样协议已准备，尚未生成新确认缓存。\n'+old[i+1:])
path=R/'configs/pv_content_P1_development.json';cfg=json.loads(path.read_text())
cfg['actual_example_batching'].update(B4_same_batch_backward_memory_comparison_done=True,B4_memory_reference_raw_sha256=b['raw_sha256'],
    B4_same_job_attribution_peak_bytes=21766003712,B4_same_job_ordinary_backward_peak_bytes=31291300864)
cfg.update(original_long_resource_raw_sha256=s['raw_sha256'],original_long_resource_verified_cases=[['niah_mq_q2',2,1241],['hotpotqa_long',22,3470],['math',92,3762]],
    original_long_quality_verified=False,next='Freeze original FT comparison, sign-validation and efficiency claims; prepare authorized new unused MoreHopQA through exact original sampling once API configuration is available. Long-input and same-batch memory checks passed for documented cases; independent quality/sign work remains.')
write_keep(path,json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
path=A/'research_state_20260906.json';raw=path.read_bytes();state=json.loads(raw);assert isinstance(state,dict)
(A/'state_recovery_20260907'/f'pre_long_and_batch_memory_{time.time_ns()}.json').write_bytes(raw)
state.update(latest_execution_status='original_long_and_same_batch_memory_verified;new_original_morehop_source_prepared;goal_active',latest_report=name,latest_priority_note=name,
    current_running_experiment=None,queued_experiments=[],next_research_question='Can the fixed finite-P1 quality advantage and explicitly scoped signs survive independent original-source samples under frozen complete budgets?',
    next_stage_status='new_original_morehop_sampling_needs_API_configuration;independent_quality_and_sign_protocol_pending',
    next_stage='Preserve fixed finite-P1. Finish pre-attribution FT0-3 quality/cost/sign protocol and independent-cache validation; run authorized original MoreHopQA sampling when exact generator/judge API is configured. Other original benchmarks remain required; do not call reused95-case results independent.',
    vendor_fa_original_long_cost={'status':'verified_complete','summary':'vendor_fa_original_long_cost_summary_20260907.json','raw_sha256':s['raw_sha256'],'new_quality_evaluated':False},
    vendor_fa_batch_memory={'status':'verified_complete','summary':'vendor_fa_batch_memory_summary_20260907.json','raw_sha256':b['raw_sha256']},
    original_morehop_confirmation={'status':'source_and_sampling_protocol_prepared_API_configuration_missing','source_summary':'original_morehop_confirmation_source_summary_20260907.json',
        'protocol':'original_morehop_sampler_protocol_20260907.json','user_authorized_original_resampling':True,'eligible_original_rows':823,'actual_API_calls':0,'new_fixed_cache_generated':False},
    last_goal_turn_classification={'classification':'progress','evidence':['verified original long resource results','verified same-batch ordinary-backward references','authorized original-source overlap removal and frozen sampling protocol'],'goal_complete':False})
t=path.with_suffix('.partial');t.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');assert isinstance(json.loads(t.read_text()),dict);t.replace(path)
print(name)
