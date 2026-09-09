# DeltaTrace · ICLR 2027

[main.tex](main.tex)是英文段落稿，[paragraph_plan_zh.md](paragraph_plan_zh.md)给出中文段落与9页正文的安排。

当前稿件包含摘要、直觉图、Introduction、完整方法主线、贡献守恒及解析算子推导、效率计算、正式评测协议、相关工作和结论。正式实验的数值段落由独立结果文件接入；本轮是模板与段落交付。排版稿保存在`output/pdf/deltatrace-iclr2027-draft.pdf`，共8页（含参考文献与附录）。

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
wsl.exe -d Ubuntu-22.04 --cd /mnt/c/Users/Chen/Documents/ChatGPT/credit/DeltaTrace/paper/iclr2027 -- latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

图1直接由TikZ绘制，正文和公式保持可编辑。稿件保留官方匿名投稿格式和行号。`source_verification.json`记录模板及源码检查，`build_receipt.json`记录PDF生成与逐页检查；独立引用核对见`editorial/citation_review.md`。

## 写作参考

使用了[Supervisor-Skills/paper-writer](https://github.com/HKUSTDial/Supervisor-Skills/blob/main/skills/paper-writer/SKILL.md)和[intro-drafter](https://github.com/HKUSTDial/Supervisor-Skills/blob/main/skills/intro-drafter/SKILL.md)中的段落组织、论点对应证据及独立引用核验思路。具体叙述服从本稿方法主线，参考文献采用ICLR官方作者—年份格式。

强化学习延伸见[独立研究备忘](research_notes/credit_assignment_extension.md)。正文以DeltaTrace的解释能力与效率为中心。
