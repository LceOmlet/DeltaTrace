"""Record the completed official promotion and visually reviewed manuscript."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import re
from pathlib import Path
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
P=ROOT/'paper/iclr2027'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_bytes())
pdf=P/'output/pdf/deltatrace-iclr2027-draft.pdf'
reader=PdfReader(pdf)
assert len(reader.pages)==17
assert not reader.metadata.get('/Author')
texts=[page.extract_text() for page in reader.pages]
table=texts[6].split('Table 4:',1)[1]
table=table[table.index('MQ-Q2'):]
rows=list(csv.DictReader((P/'results/data/qwen35_official_quality.csv').open(encoding='utf-8')))
by={(r['dataset'],r['method']):r for r in rows}
tasks=[('niah_mq_q2','MQ-Q2'),('niah_mq_q4','MQ-Q4'),('niah_mq_q8','MQ-Q8'),
       ('niah_mv_v2','MV-V2'),('niah_mv_v4','MV-V4'),('niah_mv_v8','MV-V8'),('math','MATH'),('morehopqa','MoreHopQA')]
verified=0
for i,(task,name) in enumerate(tasks):
    start=table.index(name)
    stop=table.index(tasks[i+1][1],start+len(name)) if i+1<len(tasks) else len(table)
    numbers=re.findall(r'\d+\.\d+',table[start:stop])
    expected=[]
    if task.startswith('niah'):
        expected=[f"{100*float(by[task,m]['recall']):.2f}" for m in ('FT_K3','DT')]
    expected += [f"{float(by[task,m][metric]):.4f}" for metric in ('rise','mas') for m in ('FT_K1','DT')]
    assert numbers[:len(expected)]==expected,(task,numbers,expected)
    verified+=len(expected)
assert verified==44
text='\n'.join(texts)
for stale in ('57.12','76.56','35.3%','higher HotpotQA'):
    assert stale not in text,stale
assert 'reward-level credit propagation' in texts[7]
assert 'reward-level credit propagation' not in texts[8]
log=(P/'build/main.log').read_text(encoding='utf-8')
assert not any(s in log for s in ('LaTeX Warning:','Overfull','Underfull'))
quality=load(P/'results/qwen35_quality_verification.json')
assert quality['tasks']==8 and quality['cases']==72 and quality['quality_values']==44
profile=load(ROOT/'deltatrace/profiles/sources.json')
for name,digest in profile['files'].items():assert sha(ROOT/name)==digest,name
native=load(HERE/'figure_import_verification.json')
assert native['official_default_matches_prototype_bitwise'] and native['cases']==3

receipt=load(P/'build_receipt.json')
receipt.update(created_at=datetime.now(timezone.utc).isoformat(),sha256=sha(pdf),pages=17,
    source_verification_sha256=sha(P/'source_verification.json'),
    figure_manifest_sha256=sha(P/'figures/figure_manifest.json'),
    results_verification_sha256=sha(P/'results/summary.json'),
    undefined_references=False,overfull_boxes=False,compiler_warnings=False,underfull_boxes=False,
    body_last_page=9,conclusion_last_page=8,references_pages=[9,10],appendix_pages=list(range(11,18)),
    artifact_stage='Official Qwen3.5 symmetric GDN implementation and 72-case quality comparison',
    revision='Promotes symmetric GDN as the official Qwen3.5 default; replaces current Qwen3.5 results with 8-task/72-case data; updates memory formulas and three native illustrations. Qwen3 data is unchanged; no Qwen3.5 efficiency reporting.',
    qwen35_quality=dict(table_page=7,tasks=8,cases=72,quality_values=44,
        attribution_profile='gdn-symmetric-v1',source_csv='results/data/qwen35_official_quality.csv',
        source_sha256=quality['sha256'],verification='results/qwen35_quality_verification.json',
        efficiency_reported=False),
    qwen35_native_illustrations=dict(cases=3,profile='gdn-symmetric-v1',
        prototype_identity='bitwise on the first case',independent_curves=15,independent_inputs=204))
receipt['visual_review']=dict(renderer='Poppler pdftoppm',dpi=110,reviewed_pages=list(range(1,18)),
    review_method='All final pages visually inspected in contact sheets; pages 2, 7, 8 and 13 also inspected individually.',
    checks=['44 quality values verified from the rendered PDF against the frozen CSV',
            'New memory formula and GDN gate rule match the official profile',
            'Native illustrations use the current Qwen3.5 method',
            'No clipping, overlap, overfull boxes or undefined references',
            'Conclusion fits page 8; main text including AI Use Statement fits 9 pages'],
    page_image_sha256={str(i):sha(P/f'tmp/pdfs/qwen35-official/page-{i:02}.png') for i in range(1,18)})
(P/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
promotion=dict(status='complete',adopted='2026-09-13',qwen35_default='gdn-symmetric-v1',
    qwen3_default='clean-v1',quality_tasks=8,quality_cases=72,displayed_quality_values=44,
    official_factory='deltatrace/profiles/official.py:make_qwen35_runner',
    profile_manifest_sha256=sha(ROOT/'deltatrace/profiles/sources.json'),
    evaluator_sha256=sha(ROOT/'experiments/official/evaluate.py'),
    cpu_tests=dict(command='python -m unittest discover -s experiments/official -p test_*.py',passed=32),
    native_default_prototype_bitwise=True,illustrations_recomputed=3,illustrative_curves_verified=15,
    illustrative_actual_inputs_verified=204,old_clean_sources_preserved=True,
    current_quality_source='paper/iclr2027/results/data/qwen35_official_quality_sources.json',
    current_quality_source_sha256=sha(P/'results/data/qwen35_official_quality_sources.json'),
    pdf=str(pdf.relative_to(ROOT)).replace('\\','/'),pdf_sha256=sha(pdf),pdf_pages=17,
    qwen35_quality_table_page=7,full_qwen35_benchmark_claimed=False,qwen35_efficiency_reported=False)
(HERE/'promotion_receipt.json').write_text(json.dumps(promotion,indent=2)+'\n',encoding='utf-8',newline='\n')
(HERE/'README.md').write_text('''# Qwen3.5 官方实现切换

2026-09-13，按用户要求将 `gdn-symmetric-v1` 设为官方默认，并采用对应的 8 任务、72 例数据。当前入口、配置和清单在 `deltatrace/profiles/` 与 `configs/official_dt_profiles.json`。旧 clean-v1 的代码、依赖清单、13 任务数据和示例图源均保留。

正文数据来自已冻结的六个 NIAH 任务各 10 例新键值模板，以及 MATH/MoreHopQA 各 6 例固定原始样本。FT K1 用于 RISE/MAS，K3 用于 NIAH Recall；DT 使用 signed RISE、positive MAS/Recall。所有 44 个显示值已核对到原始逐例记录，并从 PDF 再次提取核验。Qwen3 主数据未改动。

另外重算了论文原有的三个 Qwen3.5 示例：MQ-Q2 第 0 例、MoreHopQA 第 0/1 例；它们不计入 72 例正文均值。运行实际调用官方默认 factory，首例与冻结原型逐位一致。独立 CPU 审计验证了 204 次原生评分输入和 15 条曲线。代码与评分适配器的 32 项测试通过。

`raw/figures_v1/` 保存原生输出、源码、协议和验证；`figure_import_verification.json` 保存图源导入记录；`promotion_receipt.json` 保存完成状态与哈希。论文方法、双端点记忆公式、示例图和正文表已同步。最终 PDF 共 17 页，新表位于第 7 页，结论结束于第 8 页；没有 Qwen3.5 效率汇报。
''',encoding='utf-8',newline='\n')
print(json.dumps(promotion,ensure_ascii=False))
