"""Render all prespecified Recall results and their limits."""
import argparse
import hashlib
import json
from pathlib import Path

TASKS=['vt_h2_c3','vt_h4_c1','vt_h6_c1','vt_h10_c1','hotpotqa_long']
LABELS={'vt_h2_c3':'VT H2-C3','vt_h4_c1':'VT H4-C1','vt_h6_c1':'VT H6-C1','vt_h10_c1':'VT H10-C1','hotpotqa_long':'HotpotQA'}
PRIMARY={'VT_raw':'VT · raw tokens','VT_density':'VT · sentence mean','HotpotQA_raw':'HotpotQA · raw tokens','HotpotQA_density':'HotpotQA · sentence mean'}


def percent(x):return f'{100*x:.2f}%'
def points(x):return f'{100*x:+.2f}'
def bounds(x):return f'[{100*x[0]:+.2f}, {100*x[1]:+.2f}]'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    r=json.loads(a.analysis.read_bytes());assert r['status']=='verified_validation' and r['case_count']==80
    out=a.analysis.parent
    lines=['# 80 例保留验证：解释目标修复与 Recall 的适用范围','',
        '使用运行前冻结的任务设置：VT 只解释生成的最终答案，HotpotQA 解释完整回答。两种方法在每个任务内使用相同原始模型输入、非停止目标 token、正文候选集和 `ceil(0.10*N_source)` token 预算；DT 使用原 EOS 参照与原 content-P1 规则，FT K3 live 执行。',
        '', '这是任务特定的解释目标预设。它没有通过或替代此前失败的统一目标门槛。第二组 80 例在选择该预设时封存，均已完成后才进行结果分析；属于已审计基准的内部保留验证。', '',
        '## 四项预定主比较','',
        'VT 为四个任务等权均值。区间是 10,000 次任务内配对 bootstrap 的 Bonferroni 调整区间：每项覆盖率 98.75%，四项名义联合覆盖率 95%。单个任务和其他预算只作次要分析。', '',
        '| 范围及排序 | DT Recall | FT K3 Recall | DT−FT（百分点） | 调整后区间 | 判定 |',
        '| --- | ---: | ---: | ---: | --- | --- |']
    verdict={'advantage':'确认优势','disadvantage':'确认劣势','ceiling_parity':'共同达到预算上限，持平','inconclusive':'不确定'}
    for key in PRIMARY:
        v=r['primary_comparisons'][key]
        lines.append(f"| {PRIMARY[key]} | {percent(v['methods']['DT_target'])} | {percent(v['methods']['FT_K3'])} | {points(v['mean_difference'])} | {bounds(v['adjusted_ci9875'])} | {verdict[v['adjusted_verdict']]} |")
    lines+=['','原始 token 排序的优势只说明该视图下的差异。句聚合为两种方法共同提供，必须同时查看其结果；不能把共同预算上限持平解释成 DT 超过 FT。','',
        '## 全部五个任务','',
        '| 任务（各 16 例） | 原始 DT | 原始 FT K3 | 句均值 DT | 句均值 FT K3 | 10% 预算 Recall 上限 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for t in TASKS:
        raw=r['task_comparisons'][t]['raw']['0.1'];den=r['task_comparisons'][t]['density']['0.1']
        lines.append(f"| {LABELS[t]} | {percent(raw['methods']['DT_target'])} | {percent(raw['methods']['FT_K3'])} | {percent(den['methods']['DT_target'])} | {percent(den['methods']['FT_K3'])} | {percent(den['mean_ceiling'])} |")
    lines+=['','## 可论证的修复与限制','',
        '- 多链 VT 的完整响应包含未被提问的其他链，而 gold 只覆盖被问到的链。开发对照证明：仅限制最终答案种子不足，移除已生成推理后才恢复到双方 100% 的句聚合 Recall。这改变了解释目标，不能称为模型能力提高。',
        '- 正文候选过滤与句聚合是独立检索步骤。两者都同等给予 FT；句聚合仍严格支付 token 预算，原始 signed DT 向量单独保存。',
        '- HotpotQA 保留完整回答是开发集选择。桥接上下文是可能的解释，尚未用独立机制消融证明。其优势范围以本表为准。',
        '- 所有先前失败保留：正文 reference 的两项完整任务负结果、首轮 80 例验证、目标/参照/聚合/逐层对称开发实验。第二组验证后不再在这 80 例上选择其他候选。',
        '- 单个 VT 任务与 5%/20% 预算的区间仅作次要分析；不把其中的正值改称预定主检验成功。', '',
        '## 预算敏感性（次要）','',
        '| 范围及排序 | 5% DT−FT | 10% DT−FT | 20% DT−FT |','| --- | ---: | ---: | ---: |']
    for g in ['VT','HotpotQA']:
        for view in ['raw','density']:
            vals=[points(r['groups'][g][view][str(f)]['mean_difference']) for f in [.05,.1,.2]]
            lines.append('| '+PRIMARY[g+'_'+view]+' | '+' | '.join(vals)+' |')
    lines+=['','## 复核记录','',
        '80 例输入、原 prompt、gold、答案子串、完整参照和实际 DT/FT 目标权重全部核对；全部 Recall 曲线从保存向量重算。运行前已冻结代码、计划、样本与目标策略，27 项正式方法依赖未改写。', '',
        f"模型操作总时间 {r['completed_gpu_operation_seconds']:.2f} 秒，包含加载、额外完整响应控制与两种 FT 配置；逐操作时间和归因数值残差保留在分析文件中。", '',
        '[逐例结果](cases.csv) · [完整分析和执行成本](analysis.json) · [预定设计](../TARGET_SCOPE_VALIDATION.md) · [冻结目标策略](../scope_choice.json) · [先前开发结果](../answer_development/RESULTS.md)', '',
        f"结果 SHA-256：`{r['run_results_sha256']}`。向量 SHA-256：`{r['run_vectors_sha256']}`。",'']
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        keys=list(PRIMARY);fig,ax=plt.subplots(figsize=(10.5,4.6),dpi=170)
        fig.patch.set_facecolor('#fbfaf7');ax.set_facecolor('#fbfaf7')
        for j,key in enumerate(keys):
            v=r['primary_comparisons'][key];y=3-j;mean=100*v['mean_difference'];lo,hi=[100*x for x in v['adjusted_ci9875']]
            color={'advantage':'#17775b','disadvantage':'#ba4c39','ceiling_parity':'#646e78','inconclusive':'#486a9b'}[v['adjusted_verdict']]
            ax.plot([lo,hi],[y,y],color=color,lw=4,solid_capstyle='round')
            ax.scatter([mean],[y],s=85,color=color,edgecolors='#fbfaf7',linewidths=1.5,zorder=3)
            ax.annotate(f'{mean:+.2f} pp   [{lo:+.2f}, {hi:+.2f}]',(mean,y),xytext=(0,13),textcoords='offset points',ha='center',fontsize=10,color='#293139')
        ax.axvline(0,color='#9b9b98',lw=1,ls='--',zorder=0);ax.set_yticks([3,2,1,0],[PRIMARY[k] for k in keys]);ax.tick_params(axis='y',length=0,pad=12)
        ax.set_ylim(-.55,3.6);ax.margins(x=.18);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:+g}'))
        ax.set_xlabel('DT minus live FT K3 Recall@10% (percentage points)',labelpad=10)
        ax.grid(axis='x',color='#e8e5df',zorder=0);ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#ccc7be')
        fig.suptitle('Reserved validation: where the Recall advantage holds',x=.03,y=.97,ha='left',fontsize=16,fontweight='bold',color='#222b32')
        fig.text(.03,.035,'80 cases · equal task weights within VT · paired bootstrap · 98.75% intervals for 4 prespecified comparisons',fontsize=9,color='#626967')
        fig.subplots_adjust(left=.28,right=.96,top=.82,bottom=.2);path=out/'recall_scope.png';fig.savefig(path);plt.close(fig)
        lines[6:6]=['![预定主比较的调整后区间](recall_scope.png)','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='report_written',path=str(out/'RESULTS.md'),analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest())))


if __name__=='__main__':main()
