# DeltaTrace · ICLR 2027

[main.tex](main.tex)是英文段落稿，[paragraph_plan_zh.md](paragraph_plan_zh.md)给出中文段落与9页正文的安排。

当前稿件从证据内容、选择和记忆的直觉出发，正文保留解释机制所需的公式。定理、证明、完整算子规则和实现细节集中到附录。正式实验的数值段落由独立结果文件接入。排版稿保存在`output/pdf/deltatrace-iclr2027-draft.pdf`，页数与生成记录见`build_receipt.json`。

## 官方模板

- 官网：[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)。初稿正文9页，参考文献和附录另计。
- 本机下载来源：[ICLR/Master-Template](https://github.com/ICLR/Master-Template/tree/46ed6f4c6cef5b175dde23639e77d44c3463b230/iclr2027)，提交`46ed6f4c6cef5b175dde23639e77d44c3463b230`。
- [iclr2027-official.zip](iclr2027-official.zip)保存原始压缩包；[official/](official/)保存原始样例和样式；[template_source.json](template_source.json)保存逐文件校验值。官网media域名在本机DNS解析失败，使用官方ICLR仓库的2027包。
- 实际使用的`.sty`、`.bst`、`natbib.sty`、`fancyhdr.sty`与该压缩包逐字节相同。

## 编译

带标准TeX环境时，在本目录运行：

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

本机已使用Ubuntu-22.04中的既有TeX Live环境完成编译：

```powershell
wsl.exe -d Ubuntu-22.04 -- latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

机制图和两张案例图由`figures/build_figures.py`生成；主结果表和两张效率图由`results/build_results.py`从冻结记录生成，均保留SVG、矢量PDF及PNG预览。图1展示方法机制，图2展示Qwen3-8B配对效率，图3展示长输入dense/tiled成本，图4、图5展示两模型在检索和多跳任务上的真实有符号归因，各含该模型自己的DT/FT删除曲线。两张案例图的颜色图例均在右栏删除曲线下方。

正文使用官方 GDN 对称版的 Qwen3.5-9B DT/FT质量表，包含8个任务、72例与44个Recall/RISE/MAS数值；不汇报Qwen3.5效率或开发运行日志。Qwen3-8B主表包含全部13个任务、1,243例的RISE/MAS，以及11个任务的recovery。VT四任务预算为10%/10%/20%/30%，HotpotQA采用10%正文token预算下的支持句Recall；表中列明预算并加入VT宏平均。七种方法以RISE、MAS、recovery三张表展示，完整CSV包含91行、259个指标值。数据与重建方式见[results/README.md](results/README.md)。

独立图册为`output/pdf/deltatrace-figures.pdf`，两页效率图为`output/pdf/deltatrace-efficiency.pdf`。数据对齐、色标与复现说明见[figures/README.md](figures/README.md)。正文和公式保持可编辑，官方匿名投稿格式及样式保持原样。`source_verification.json`记录模板与源码核对，`build_receipt.json`记录最终PDF与逐页检查；引用核对见`editorial/citation_review.md`。

## 写作参考

使用了[Supervisor-Skills/paper-writer](https://github.com/HKUSTDial/Supervisor-Skills/blob/main/skills/paper-writer/SKILL.md)和[intro-drafter](https://github.com/HKUSTDial/Supervisor-Skills/blob/main/skills/intro-drafter/SKILL.md)中的段落组织、论点对应证据及独立引用核验思路。行文参考[FlashTrace方法节](https://arxiv.org/html/2602.01914v4#S4)从计算对象引出操作和作用的组织方式，以具体证据情境贯穿DeltaTrace的注意力与记忆机制。公式按解释需要出现。参考文献采用ICLR官方作者—年份格式。

强化学习延伸见[独立研究备忘](research_notes/credit_assignment_extension.md)。正文以DeltaTrace的解释能力与效率为中心。
