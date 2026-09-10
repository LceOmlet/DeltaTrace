"""Fill the complete benchmark tables without changing any evaluation setting."""
import argparse
import csv
import json
from pathlib import Path

TASKS=['vt_h2_c3','vt_h4_c1','vt_h6_c1','vt_h10_c1','hotpotqa_long']
LABELS={'vt_h2_c3':'VT H2-C3','vt_h4_c1':'VT H4-C1','vt_h6_c1':'VT H6-C1','vt_h10_c1':'VT H10-C1','hotpotqa_long':'HotpotQA','VT':'VT 宏平均','all_five_tasks':'五任务宏平均'}
KEYS=TASKS+['VT','all_five_tasks']


def pc(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}'
def ci(x):return f'[{100*x[0]:+.2f}, {100*x[1]:+.2f}]'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    r=json.loads(a.analysis.read_bytes());assert r['status']=='verified_complete_benchmark' and r['case_count']==448
    out=a.analysis.parent
    def value(key,view,f):return (r['task_comparisons'][key] if key in TASKS else r['groups'][key])[view][str(f)]
    counts={t:(48 if t=='hotpotqa_long' else 100) for t in TASKS}|{'VT':400,'all_five_tasks':448}
    def ceiling(key):
        members=[key] if key in TASKS else TASKS[:4] if key=='VT' else TASKS
        return sum(r['task_comparisons'][t]['raw']['0.1']['mean_ceiling'] for t in members)/len(members)
    table=[];budgets=[]
    for key in KEYS:
        raw=value(key,'raw',.1);den=value(key,'density',.1)
        table.append(dict(dataset=key,n=counts[key],DT_raw=raw['methods']['DT_target'],FT_K3_raw=raw['methods']['FT_K3'],
            raw_difference=raw['mean_difference'],DT_sentence_mean=den['methods']['DT_target'],FT_K3_sentence_mean=den['methods']['FT_K3'],
            sentence_mean_difference=den['mean_difference'],mean_budget_ceiling=ceiling(key)))
        for view in ('raw','density'):
            for f in (.05,.1,.2):
                v=value(key,view,f)
                budgets.append(dict(dataset=key,n=counts[key],view=view,fraction=f,DT=v['methods']['DT_target'],FT_K1=v['methods']['FT_K1'],FT_K3=v['methods']['FT_K3'],
                    difference=v['mean_difference'],ci95_low=v['ci95'][0],ci95_high=v['ci95'][1],joint_exact_ceiling=v['joint_exact_ceiling']))
    for filename,rows in [('table_recall10.csv',table),('table_budgets.csv',budgets)]:
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# VT 与 HotpotQA 全量 Recall 表','',
        '**Qwen3-8B，448/448 例已完整覆盖并复核**：四组 VT 各 100 例，HotpotQA 48 例。复用完全同协议、已独立验证的 80 例，新增 368 例；额外重跑 5 个衔接控制，其 15 份 DT/FT 向量逐位一致，控制不重复计入样本数。','',
        '固定目标：VT 只解释缓存中模型生成的最终答案；HotpotQA 解释完整回答。DT 与 live FT K3 使用相同模型输入、目标 token、正文候选集和 `ceil(0.10*N_source)` token 预算。DT 保留原 content-P1 规则与原 EOS 参照。原始 token 排序和双方同样的句均值聚合同时报告。','',
        '## Recall@10% 全表','',
        '| 数据集 | n | DT 原始 | FT K3 原始 | DT 句均值 | FT K3 句均值 | 预算 Recall 上限 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in table:
        lines.append(f"| {LABELS[x['dataset']]} | {x['n']} | {pc(x['DT_raw'])} | {pc(x['FT_K3_raw'])} | {pc(x['DT_sentence_mean'])} | {pc(x['FT_K3_sentence_mean'])} | {pc(x['mean_budget_ceiling'])} |")
    lines+=['','宏平均按任务等权，n 为覆盖的样本总数。句聚合仍按 token 收费，未按整句免费补齐；预算上限由每例有效 gold 数和 token 预算计算。','',
        '## 每任务配对差值（DT−FT K3，百分点）','',
        '| 数据集 | 原始差值 | 描述性 95% 区间 | 句均值差值 | 描述性 95% 区间 |',
        '| --- | ---: | --- | ---: | --- |']
    for t in TASKS:
        raw=value(t,'raw',.1);den=value(t,'density',.1)
        lines.append(f"| {LABELS[t]} | {pp(raw['mean_difference'])} | {ci(raw['ci95'])} | {pp(den['mean_difference'])} | {ci(den['ci95'])} |")
    lines+=['','## 总体配对区间','',
        '| 范围 | 原始差值 | 描述性调整区间 | 句均值差值 | 描述性调整区间 |',
        '| --- | ---: | --- | ---: | --- |']
    for g in ('VT','HotpotQA'):
        raw=r['primary_comparisons'][g+'_raw'];den=r['primary_comparisons'][g+'_density']
        lines.append(f"| {g} | {pp(raw['mean_difference'])} | {ci(raw['adjusted_ci9875'])} | {pp(den['mean_difference'])} | {ci(den['adjusted_ci9875'])} |")
    lines+=['','**这张表包含开发样本及先前验证样本，是完整基准结果，不是另一轮独立留出验证。** 区间由 10,000 次任务内配对 bootstrap 得到；总体四项比较使用 Bonferroni 98.75% 区间，仅作完整数据的描述。独立性更强的先前保留验证及其预定结论单独保留，不用全量结果重新选择目标或排序。','',
        '## 次要预算与 FT K1','',
        '[多预算完整表](table_budgets.csv) 包含每个任务及宏平均的 5%、10%、20% 两种排序、DT、FT K1、FT K3、差值和区间。','',
        '## 复核与复现','',
        '- 样本覆盖精确等于发布缓存的全部索引；无漏例、重复计数或目标策略混用。',
        '- 已验证每个新增样本的原 prompt、gold、答案提取、EOS 参照、实际 DT/FT 目标权重与 FT 可用多跳范围，并从原始向量重算所有 Recall。80 例复用数据通过冻结的结果、向量及先前独立分析哈希绑定。',
        '- 每个任务的衔接控制均通过原始输入与三份向量的逐位一致检查；正式方法的 27 项冻结依赖不变。',
        f"- 本次新增模型操作用时合计 {r['additional_gpu_operation_seconds']:.2f} 秒，含分批加载和衔接控制；复用数据对应此前模型操作 {r['reused_gpu_operation_seconds']:.2f} 秒。实际逐操作成本及额外原目标控制保留。",'',
        '[Recall@10% CSV](table_recall10.csv) · [逐例表](per_case.csv) · [样本来源](case_origins.csv) · [完整分析](analysis.json) · [固定全量计划](../FULL_RECALL.md) · [先前 80 例保留验证](../target_scope/RESULTS.md) · [目标构造修复说明](../TARGET_CONSTRUCTION_FIX.md)','',
        '原始数据见 [全量新增批次清单](../raw/full_recall_v1/manifest.json) 和 [复用 80 例清单](../raw/target_scope_v1/manifest.json)。两者均保存确定性 gzip 和原始字节哈希，可用 `archive_results.py --restore` 分别恢复到新目录，再运行以下命令：','',
        '```bash',
        'python research/temporary/source_v2_gpu_20260910/analyze_full_recall.py \\\n  --run /path/restored-full --parent /path/restored-parent \\\n  --publication /path/original-publication --data /path/original-data \\\n  --tokenizer /path/qwen3-tokenizer.json --output /path/rechecked-full',
        'python research/temporary/source_v2_gpu_20260910/build_full_recall_report.py \\\n  --analysis /path/rechecked-full/analysis.json --plot',
        '```','']
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        labels=['VT H2-C3','VT H4-C1','VT H6-C1','VT H10-C1','HotpotQA']
        fig,axes=plt.subplots(1,2,figsize=(13,5),dpi=180,sharey=True)
        fig.patch.set_facecolor('#fbfaf7')
        for ax,view,title in zip(axes,['raw','density'],['Raw token ranking','Shared sentence-mean ranking']):
            x=np.arange(5);dt=[100*value(t,view,.1)['methods']['DT_target'] for t in TASKS];ft=[100*value(t,view,.1)['methods']['FT_K3'] for t in TASKS]
            ax.set_facecolor('#fbfaf7');ax.bar(x-.18,dt,width=.34,color='#237a65',label='DT',zorder=3);ax.bar(x+.18,ft,width=.34,color='#6684ad',label='Live FT K3',zorder=3)
            ax.scatter(x,[100*ceiling(t) for t in TASKS],s=90,marker='_',color='#bd7555',linewidths=2,label='Budget ceiling',zorder=4)
            for j in x:
                ax.text(j-.18,dt[j]+1.4,f'{dt[j]:.1f}',ha='center',fontsize=8,color='#174c3f')
                ax.text(j+.18,ft[j]+1.4,f'{ft[j]:.1f}',ha='center',fontsize=8,color='#3d5677')
            ax.set_xticks(x,labels,rotation=18,ha='right');ax.set_ylim(0,110);ax.set_title(title,fontsize=13,pad=15)
            ax.grid(axis='y',color='#e8e5df',zorder=0);ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#c8c3ba')
        axes[0].set_ylabel('Recall@10% (%)');axes[0].legend(loc='upper right',frameon=False,fontsize=9)
        fig.suptitle('Qwen3-8B · Complete VT and HotpotQA benchmark · 448 cases',x=.035,ha='left',fontweight='bold',fontsize=16)
        fig.text(.035,.025,'VT: generated final answer only · HotpotQA: full response · identical inputs and source-token budgets for both methods',fontsize=9,color='#646b68')
        fig.subplots_adjust(left=.065,right=.98,bottom=.22,top=.82,wspace=.16)
        fig.savefig(out/'recall_full.png');plt.close(fig)
        lines[6:6]=['![全量 Recall 比较](recall_full.png)','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='tables_filled',rows=len(table),unique_cases=448,path=str(out/'RESULTS.md'))))


if __name__=='__main__':main()
