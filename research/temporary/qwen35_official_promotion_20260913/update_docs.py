"""Update current documentation without changing frozen research records."""
import csv
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
P=ROOT/'paper/iclr2027'
def write(path,text):path.write_text(text.rstrip()+'\n',encoding='utf-8')
def replace(path,a,b):
    s=path.read_text(encoding='utf-8')
    if a not in s:
        assert b in s,(path,a)
        return
    assert s.count(a)==1,(path,a)
    write(path,s.replace(a,b))

write(ROOT/'deltatrace/profiles/README.md',"""# Official attribution profiles

As of 2026-09-13, **`gdn-symmetric-v1` is the official Qwen3.5 profile**. The default factory and evaluation driver invoke it. Qwen3 retains `clean-v1`. The [active configuration](../../configs/official_dt_profiles.json) and [source manifest](sources.json) record the current entry points and identities.

```python
from deltatrace.profiles.official import make_qwen35_runner

runner = make_qwen35_runner(model, finite_fa, finite_fla)
scores, diagnostics = runner.attribute(paired_ids, mask, selection)
```

Load the reviewed directory containing `qwen35_dense_finite_runner.py` and its finite backends on `sys.path`, as for the frozen runtime. The official factory uses the already validated `make_qwen35_gdn_symmetric_runner` implementation:

- Every GDN layer uses the symmetric product rule for normalized content and its SiLU output gate.
- Every GDN memory block averages coefficients from the two orders of the existing reference/input captures, after composing each complete ordered recurrence.
- Attention PV and output gating, QK, MLP, and normalization retain their existing rules. The native forward, reference and fixed response remain the caller's inputs.

The profile rejects layer-specific attribution overrides and does not inspect task labels, query positions, gold annotations, or evaluation scores. Verified runtime overlays may pass supported execution options, such as `dynamic_shapes` and `compiler_options`, through the official factory. The original frozen runtime does not support those extra execution options.

For Qwen3.5-9B, one attribution uses 48 finite memory calls, 96 native FLA adjoint stages, and 8 finite attention calls. The two local memory orders reuse existing captures; they add no native root forward or reference endpoint.

Current quality evidence contains **8 tasks and 72 examples**: six NIAH tasks with ten novel key/value assignments each, and six fixed MATH plus six fixed MoreHopQA examples. The [source ledger](../../paper/iclr2027/results/data/qwen35_official_quality_sources.json) and [DT/FT table](../../paper/iclr2027/results/qwen35_quality_table.tex) report signed RISE, positive-part MAS and NIAH Recall at a 10% token budget. FT uses K1 for faithfulness and K3 for Recall. These subsets are distinct from the archived 13-task clean-v1 benchmark.

The [controlled experiment report](../../research/temporary/qwen35_niah_causal_20260913/RESULTS_zh.md) retains candidate selection, all regressions, the frozen validation protocol, and the single-case cost measurement. The [native integration check](../../research/temporary/qwen35_official_promotion_20260913/figure_import_verification.json) verifies that the official default matches the frozen prototype bitwise and supplies three recomputed manuscript illustrations.

Use `make_qwen35_runner(..., profile='clean-v1')` or the evaluator flag `--qwen35-profile clean-v1` to reproduce the old attribution rules. The frozen `qwen35_clean_runner.py` factory and its dependency manifest are unchanged.
""")

rootreadme=ROOT/'README.md'
replace(rootreadme,'[Method code](deltatrace/clean/)','[Method code](deltatrace/profiles/)')
replace(rootreadme,'This repository contains the method implementation for **Qwen3-8B** and **Qwen3.5-9B**, evaluation records, measured efficiency results, and the editable manuscript with its figure and table sources.',
        'This repository contains the method implementation for **Qwen3-8B** and **Qwen3.5-9B**, evaluation records, and the editable manuscript. Qwen3.5 now defaults to **GDN symmetric propagation**; the [official profile](deltatrace/profiles/README.md) and [active configuration](configs/official_dt_profiles.json) define its uniform rules and provenance.')
replace(rootreadme,'### Current comparison with task-specific VT budgets','### Qwen3-8B comparison with task-specific VT budgets')
s=rootreadme.read_text(encoding='utf-8')
start=s.find('### Hybrid-model development results')
if start>=0:
    end=s.index('### Measured efficiency',start)
    rows=list(csv.DictReader((P/'results/data/qwen35_official_quality.csv').open(encoding='utf-8')))
    by={(r['dataset'],r['method']):r for r in rows}
    tasks=[('niah_mq_q2','MQ-Q2'),('niah_mq_q4','MQ-Q4'),('niah_mq_q8','MQ-Q8'),
           ('niah_mv_v2','MV-V2'),('niah_mv_v4','MV-V4'),('niah_mv_v8','MV-V8'),('math','MATH'),('morehopqa','MoreHopQA')]
    lines=['### Qwen3.5-9B official profile results','',
      'The official `gdn-symmetric-v1` profile is compared with the same-model FlashTrace on **72 cases**: ten new key/value assignments for each NIAH task and six fixed released cases each for MATH and MoreHopQA. RISE uses signed DT ranking, MAS uses its positive part, and Recall uses a 10% eligible-token budget. FT uses K1 for RISE/MAS and K3 for Recall. These descriptive subsets are separate from the complete Qwen3 benchmark.','',
      '| Task | n | Recall FT / DT (%) | RISE FT / DT | MAS FT / DT |','| --- | ---: | ---: | ---: | ---: |']
    for task,name in tasks:
        d,f=by[task,'DT'],by[task,'FT_K1']
        rec=f"{100*float(by[task,'FT_K3']['recall']):.2f} / {100*float(d['recall']):.2f}" if task.startswith('niah') else '—'
        lines.append(f"| {name} | {d['count']} | {rec} | {float(f['rise']):.4f} / {float(d['rise']):.4f} | {float(f['mas']):.4f} / {float(d['mas']):.4f} |")
    lines+=['','DT has higher Recall on all six NIAH tasks, lower RISE on six of eight tasks, and lower MAS on seven. See the [full-precision CSV](paper/iclr2027/results/data/qwen35_official_quality.csv) and [source ledger](paper/iclr2027/results/data/qwen35_official_quality_sources.json). The older 13-task Qwen3.5 results remain archived under `qwen35_full_quality.csv` with the clean-v1 identity.','','']
    write(rootreadme,s[:start]+'\n'.join(lines)+s[end:])
replace(rootreadme,'![Paired attribution latency and allocated memory for Qwen3-8B, and serial versus batched attribution for Qwen3.5-9B.]',
        '![Paired attribution latency and allocated memory for Qwen3-8B.]')
s=rootreadme.read_text(encoding='utf-8')
s=s.replace('| Qwen3.5-9B: serial batch 1 with CPU checkpoints → real batch 2 with GPU checkpoints | 16.68 s → 12.33 s | 21.64 GB → 23.88 GB |\n','')
s=s.replace('The Qwen3.5 configuration increases throughput by **35.3%** on the recorded 16-example workload. Each row is a separate paired implementation benchmark. The Qwen3 measurements predate the full-task quality freeze, and the Qwen3.5 comparison measures a scheduling and checkpointing change.',
            'These Qwen3 measurements predate the full-task quality freeze. Historical Qwen3.5 timing fixtures describe clean-v1 and do not measure the current official profile.')
s=s.replace('reduce warm time by 5.80% on Qwen3 and 5.15% on Qwen3.5 against their respective baselines, with identical paired source vectors.',
            'reduce warm time by 5.80% on Qwen3 against its baseline, with identical paired source vectors.')
s=s.replace('The frozen method sources are in [`deltatrace/clean/`](deltatrace/clean/), version `clean-v1-20260909`:',
            'Current entry points are listed below. The frozen `clean-v1-20260909` dependencies remain in [`deltatrace/clean/`](deltatrace/clean/).')
s=s.replace('| Qwen3.5-9B | [`make_qwen35_clean_runner`](deltatrace/clean/qwen35/qwen35_clean_runner.py) | BF16, P1/content1 rules, native FA/FLA, and empty per-layer override maps |',
            '| Qwen3.5-9B | [`make_qwen35_runner`](deltatrace/profiles/official.py) | BF16, symmetric output gate and memory-order average in all GDN layers; native FA/FLA |')
s=s.replace('The [method manifest](deltatrace/clean/sources.json) records all 27 frozen dependency files.',
            'The [official profile manifest](deltatrace/profiles/sources.json) identifies the current profile. The [dependency manifest](deltatrace/clean/sources.json) retains all 27 frozen dependency files.')
s=s.replace('The earlier adapter retained on `main` uses the original development positive-score view.',
            'The current Qwen3.5 released-v1 adapter defaults to signed RISE for the official profile; the historical clean-v1 and source-v2 adapters retain positive RISE unless explicitly overridden.')
s=s.replace('historical Qwen3.5 development runs use `--evaluation-protocol released-v1 --family qwen35 --selection development16 --ft live`',
            'historical Qwen3.5 development runs use `--evaluation-protocol released-v1 --family qwen35 --qwen35-profile clean-v1 --selection development16 --ft live`')
write(rootreadme,s)

readme=ROOT/'experiments/official/README.md'
s=readme.read_text(encoding='utf-8')
if '## 当前 Qwen3.5 官方实现' not in s:
    pos=s.index('\n')+1
    text="""
## 当前 Qwen3.5 官方实现

默认 Qwen3.5 归因为 `gdn-symmetric-v1`，所有 GDN 层统一使用对称输出门和完整记忆系数的双端点顺序平均。Qwen3 沿用 clean-v1。方法版本独立于下面的评测范围协议；输出和汇总均保存方法版本及清单哈希。

```bash
python experiments/official/evaluate.py --family qwen35 --environment /path/environment.json \\
  --evaluation-protocol released-v1 --selection smoke --datasets niah_mq_q2 morehopqa \\
  --ft live --output /path/qwen35-official-smoke
```

该命令默认使用官方对称 GDN、signed RISE 和 positive MAS/Recall。`--rise-score-view` 可显式选择 signed/positive，结果记录相应口径。source-v2 的默认 RISE 仍保留原正值口径。重现历史 Qwen3.5 方法必须加 `--qwen35-profile clean-v1`，其 released-v1 默认仍为 positive RISE。

当前论文 Qwen3.5 表来自另行冻结的 72 例：[NIAH 新键值构造与验证](../../research/temporary/qwen35_niah_causal_20260913/validation_protocol.json)、[固定 MATH/MoreHopQA 子集](../../research/temporary/qwen35_niah_causal_20260913/generalization_fresh_protocol.json)。完整复现命令见[实验报告](../../research/temporary/qwen35_niah_causal_20260913/RESULTS_zh.md)。通用入口的 `paper` 选项表示原缓存全量，不会自动选择或构造这 72 例。

历史 clean-v1 13 任务数据保留，当前正文只引用 [gdn-symmetric-v1 的数据清单](../../paper/iclr2027/results/data/qwen35_official_quality_sources.json)。
"""
    write(readme,s[:pos]+text+'\n'+s[pos:])

replace(P/'results/README.md','- `data/qwen35_full_quality.csv`: full Qwen3.5-9B task means for Recall, RISE and MAS. `build_qwen35_quality.py` produces the 13-row, 74-value DT/FT table in the main text; FT uses K3 for Recall and K1 for RISE/MAS.',
        '- `data/qwen35_official_quality.csv`: current symmetric-GDN Qwen3.5 means on 8 tasks / 72 cases. `build_qwen35_quality.py` produces the 8-row, 44-value DT/FT table; FT uses K3 for Recall and K1 for RISE/MAS. `qwen35_official_cases.csv` and `qwen35_official_quality_sources.json` retain paired rows, selections and hashes.\n- `data/qwen35_full_quality.csv`: archived clean-v1 13-task means; excluded from the current manuscript table.')
replace(P/'results/README.md','python results/import_current_recovery.py\npython results/build_results.py --tables-only',
        'python results/import_current_recovery.py\npython results/import_qwen35_official.py\npython results/build_results.py --tables-only')
replace(P/'results/README.md','The case plots now compare each model with its own one-hop FT curve.',
        'The case plots compare each model with its own one-hop FT curve. Qwen3.5 scores and curves were recomputed with the official symmetric-GDN factory on the same fixed illustrative inputs; Qwen3 records retain clean-v1.')

write(P/'editorial/qwen35_quality_review.md',"""# Qwen3.5 official quality reporting

The main-text table uses the official gdn-symmetric-v1 profile on 8 tasks and 72 cases: ten novel key/value assignments on fixed templates for each of six NIAH tasks, plus six fixed released MATH and six MoreHopQA examples. All 44 displayed values come from the source ledger. FT K3 supplies Recall, FT K1 supplies RISE/MAS, and DT uses signed RISE and positive-part MAS/Recall.

The prose reports only the table's descriptive comparisons: higher NIAH Recall on 6/6 tasks, lower RISE on 6/8, and lower MAS on 7/8. It also identifies the FT-favored MQ-Q4/MQ-Q8 RISE and MQ-Q8 MAS means. No full 13-task claim, statistical-significance claim, or efficiency comparison is inferred from this table.

The memory section defines the average of two complete finite recurrence maps, and GDN output gating uses the symmetric product. Attention output gating retains its original rule. Three fixed Qwen3.5 illustrations were recomputed with the official default and their original inputs verified; they are excluded from the 72-case means. Qwen3 quality and efficiency results are unchanged.
""")
replace(P/'editorial/evidence_map.md','`deltatrace/clean/qwen35/qwen35_clean_runner.py` and decoder/GDN modules | Empty layer overrides; content-oriented output gates and GDN recurrence | Clean source definition; not a benchmark result',
        '`deltatrace/profiles/official.py`, `qwen35_gdn_symmetric.py`, and frozen decoder/GDN modules | Uniform symmetric GDN output gate; average of complete memory endpoint orders | Official source definition; not a benchmark result')
replace(P/'editorial/evidence_map.md','`results/data/qwen35_full_quality.csv` and `results/qwen35_quality_verification.json` | Qwen3.5 DT/FT Recall, RISE and MAS task means | All 13 tasks; 74 values; no efficiency or causal-mechanism inference',
        '`results/data/qwen35_official_quality.csv` and `results/qwen35_quality_verification.json` | Qwen3.5 symmetric-GDN DT/FT task means | 8 tasks, 72 cases, 44 values; no efficiency or causal-mechanism inference')
replace(P/'editorial/evidence_map.md','Figures 1--3 use stored clean-method scores and share one linear color scale across both models within each task, without recomputation or rescaling per model.',
        'Figures 1--3 use stored clean-v1 Qwen3 scores and recomputed official symmetric-GDN Qwen3.5 scores, with one linear color scale across both models within each task.')
replace(P/'paragraph_plan_zh.md','Qwen3.5 全量质量','Qwen3.5 官方对称 GDN 的 72 例质量')
replace(P/'paragraph_plan_zh.md','Qwen3.5 在正文表中汇报全量 Recall、RISE 和 MAS',
        'Qwen3.5 在正文表中汇报官方对称 GDN 的 8 任务、72 例 Recall、RISE 和 MAS')
print('Current documentation points to gdn-symmetric-v1 and its 72-case results.')
