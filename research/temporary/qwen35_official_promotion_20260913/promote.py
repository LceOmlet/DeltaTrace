"""Record the user-selected official profile and update manuscript pointers."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
P=ROOT/'paper/iclr2027'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,value):
    path.write_text(value.rstrip()+'\n',encoding='utf-8')
def replace(path,a,b):
    s=path.read_text(encoding='utf-8')
    if a not in s:
        assert b in s,(path,a)
        return
    assert s.count(a)==1,(path,a)
    write(path,s.replace(a,b))
def dump(path,x):write(path,json.dumps(x,ensure_ascii=False,indent=2))

old=json.loads((ROOT/'configs/clean_dt_baselines.json').read_bytes())
current=dict(schema_version=2,status='official',adopted='2026-09-13',
             qwen3=old['qwen3'],signed_output_retained=True,
             qwen35=dict(checkpoint='Qwen3.5-9B',profile='gdn-symmetric-v1',
                 entrypoint='deltatrace/profiles/official.py:make_qwen35_runner',
                 implementation='deltatrace/profiles/qwen35_gdn_symmetric.py:make_qwen35_gdn_symmetric_runner',
                 rules=dict(gdn_output_gate='symmetric',gdn_memory='average of both complete local endpoint orders',
                            scope='all 24 GDN layers',attention_PV='content1',attention_output_gate='content1',
                            QK='symmetric',MLP='symmetric',key_normalization='original'),
                 finite_FLA_calls_per_attribution=48,native_FLA_adjoint_stages_per_attribution=96,
                 finite_FA_calls_per_attribution=8,precision='original BF16 native FA/FLA',
                 additional_native_root_forwards=0,
                 quality=dict(tasks=8,cases=72,niah_cases_per_task=10,math_cases=6,morehopqa_cases=6,
                              source='paper/iclr2027/results/data/qwen35_official_quality_sources.json',
                              rise_view='signed',mas_recall_view='positive',FT_faithfulness_hops=1,FT_recall_hops=3),
                 historical_profile='clean-v1'),
             official_evaluation='experiments/official/evaluate.py',
             profile_manifest='deltatrace/profiles/sources.json',
             frozen_dependency_manifest='deltatrace/clean/sources.json')
dump(ROOT/'configs/official_dt_profiles.json',current)
dump(ROOT/'configs/active_profile.json',dict(profile='official_dt_profiles.json',goal='../docs/current_goal.md',
     qwen35_official_profile='gdn-symmetric-v1',qwen3_profile='clean-v1',
     quality_score_views={'RISE':'signed in the current paper comparison','MAS':'positive part','Recall':'positive part'},
     clean_source_manifest='../deltatrace/clean/sources.json',
     official_profile_manifest='../deltatrace/profiles/sources.json',
     official_protocol='../experiments/official/protocol.json',
     current_qwen35_quality='../paper/iclr2027/results/data/qwen35_official_quality_sources.json'))
dump(ROOT/'deltatrace/profiles/sources.json',dict(version='official-profiles-20260913',
     qwen35_default='gdn-symmetric-v1',qwen3_default='clean-v1',
     frozen_dependencies=dict(path='deltatrace/clean/sources.json',sha256=sha(ROOT/'deltatrace/clean/sources.json')),
     files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
            [ROOT/'deltatrace/profiles/official.py',ROOT/'deltatrace/profiles/qwen35_gdn_symmetric.py',ROOT/'configs/official_dt_profiles.json']},
     native_default_verification='research/temporary/qwen35_official_promotion_20260913/figure_import_verification.json'))
goal=ROOT/'docs/current_goal.md'
archived=ROOT/'docs/goal_clean_baselines_20260909.md'
if not archived.exists():archived.write_bytes(goal.read_bytes())
write(goal,"""# 当前官方实现与结果

2026-09-13，按用户要求将 GDN 对称版设为 Qwen3.5 的官方实现，并采用对应的已确认数据。

- 当前配置：[official_dt_profiles.json](../configs/official_dt_profiles.json)；默认入口：[make_qwen35_runner](../deltatrace/profiles/official.py)。
- Qwen3.5 的全部 24 个 GDN 层统一使用对称输出门和两个端点顺序的完整记忆系数平均。注意力与 MLP 规则沿用原实现。
- 当前 Qwen3.5 正文比较覆盖 8 个任务、72 例：六个 NIAH 任务各 10 例新键值模板，MATH/MoreHopQA 各 6 例固定原始样本。signed RISE、positive MAS/Recall；FT K1 忠实度、K3 Recall。
- 新版结果：[数据来源](../paper/iclr2027/results/data/qwen35_official_quality_sources.json)；[正文表](../paper/iclr2027/results/qwen35_quality_table.tex)。该数据不代表旧版 13 任务全量实验。
- Qwen3-8B 的方法及完整 1,243 例比较不变。旧 Qwen3.5 clean-v1 代码、清单和结果保留作历史复现；新运行需显式指定 `--qwen35-profile clean-v1` 才使用旧方法。
- 论文同步更新公式、示例图和质量表；Qwen3.5 正文不报告效率结果。

此前的阶段说明保存在 [2026-09-09 历史目标](goal_clean_baselines_20260909.md)，不再控制当前默认实现。
""")

replace(P/'main.tex','On Qwen3.5-9B, \\dt{} has lower MAS on most tasks and higher HotpotQA supporting-fact Recall than FlashTrace.',
        'On a separate 72-example Qwen3.5-9B comparison, \\dt{} has lower MAS on seven of eight tasks and higher Recall on all six NIAH tasks than FlashTrace.')
replace(P/'main.tex','The manuscript uses the released evaluation data and model-generated trajectories specified in the evaluation protocol.',
        'The manuscript uses the released data, fixed model-generated trajectories, and constructed NIAH variants specified in the evaluation protocol.')
intro=P/'sections/introduction.tex'
replace(intro,'(b,c) Attention and gated memory share a content/control allocation. The memory example expands retention, $T=\\alpha S$, with $S$ the previous state. The operators are schematic.',
        '(b) Attention separates content and selection. (c) Gated memory averages coefficients from the two endpoint orders of its recurrence, $m^{01}$ and $m^{10}$. The operators are schematic.')
replace(intro,'Actual Qwen3.5 scores from MoreHopQA development example 1',
        'Actual Qwen3.5 symmetric-GDN scores from fixed MoreHopQA example 1')
replace(intro,'In gated delta memory, the same principle follows stored content and the keys, queries, and gates governing retention, writing, and reading.',
        'In gated delta memory, averaging two ordered finite recurrences follows stored content and the keys, queries, and gates governing retention, writing, and reading.')
replace(intro,'On Qwen3.5-9B, \\dt{} also has lower MAS on most tasks and higher HotpotQA supporting-fact Recall than FlashTrace',
        'In the separate 72-example Qwen3.5-9B comparison, \\dt{} has lower MAS on seven of eight tasks and higher Recall on all six NIAH tasks than FlashTrace')
comp=P/'sections/computation.tex'
replace(comp,'An associative scan accumulates the retention contribution. At fixed state dimensions and chunk size, the stored token contributions and boundary states grow linearly with sequence length.',
        'An associative scan accumulates the retention contribution. The two memory orientations reuse the same captured endpoints and their coefficient outputs are averaged. At fixed state dimensions and chunk size, stored token contributions and boundary states grow linearly with sequence length.')
replace(comp,'Together, these steps apply the same attribution method to attention and hybrid models. The evaluation measures their combined latency, memory use, and throughput.',
        'Together, these steps apply the finite propagation framework to attention and hybrid models. The evaluation reports combined latency and memory for Qwen3-8B.')
replace(P/'sections/related.tex','Its rules share a content-and-control allocation with attention.',
        'Its endpoint-averaged memory rule and attention rule both account for changes in content and control.')

mechanism=P/'figures/draw_mechanism.py'
replace(mechanism,"text(11.74,2.02,'Retention: T = αS',17,MUTED,ha='center')",
        "text(11.74,2.02,'Average the two endpoint orders',17,MUTED,ha='center')")
replace(mechanism,"allocation_cards(9.65,'Stored content','Retention gate',\n                     r'$\\alpha_1\\,\\Delta S$',r'$S_0\\,\\Delta\\alpha$',r'$\\Delta T$',23)",
        "allocation_cards(9.65,'Order 0 to 1','Order 1 to 0',\n                     r'$m^{01}/2$',r'$m^{10}/2$',r'$m$',23)")
replace(mechanism,"'gdn_retention_notation':'S abbreviates S_(t-1); T=alpha*S; endpoint subscripts 0 and 1 are reference and original',",
        "'gdn_retention_notation':'The recurrence is schematic; m01 and m10 are complete memory input coefficients from opposite endpoint orders, averaged after composition.',")
replace(mechanism,"'schematic_elements':'paired finite trace; PV product; GDN retain/write/read; two equally sized local content/control allocations',",
        "'schematic_elements':'paired finite trace; PV content/control split; GDN retain/write/read with complete two-order coefficient average',")
print('Official profile, configuration pointers, and manuscript prose updated.')
