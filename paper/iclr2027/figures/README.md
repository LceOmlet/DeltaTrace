# DeltaTrace main figure

The manuscript overview uses four panels: (a) paired original/EOS-reference
executions and a finite reverse traversal; (b) Attention; (c) Gated DeltaNet;
(d) a measured signed context example. Attention and GDN have identical panel
areas (4.20 x 4.31 drawing units), matching two-branch cards, font sizes, and
arrow weights. GDN shows its full retain/write/read structure and expands the
retention rule T = alpha S; S abbreviates the previous state.

The main example is Qwen3.5, MoreHopQA development example 1. The question asks
about the author of a play. The recorded name spans receive +1.5582 nats for
William Shakespeare and -1.1331 nats for the nearby composer William Walton.
Brackets identify the exact names whose token scores are summed. Backgrounds
retain individual token scores, including the different signs within a name.
This is an observed role contrast under the fixed EOS reference and finite
allocation, not a single-token deletion experiment.

All excerpts and the full-input strip share one linear color mapping, centered
on zero and explicitly saturated at +/-2.5 nats. The colorbar shows <= -2.5 and
>= +2.5. Saturation makes the smaller negative scores visible; no stored scores
or printed span sums are altered. Ellipses indicate omitted text. The figure
shows the requested role and the playwright identified in the stored response;
the attribution target remains the complete response plus EOS. The original
name-length arithmetic is omitted from the figure to keep the role contrast
clear. Panel (a) labels the model blocks as Attn. / GDN. The sidebar uses the
short title Answer. Its reduced height and the raised footer remove the empty
space left by the omitted arithmetic. The saturation legend, gradient and
ticks sit in the open area to the right of the requested role, above the
full-input strip.

A second complete overview, `generated/deltatrace-lookup-overview.*`, retains a
simpler positive retrieval example: bright-system -> 9153566 and billowy-method
-> 9937326 (Qwen3.5, retrieval development example 6). Its color scale spans the
full eligible-score range of both models. The manuscript includes the signed
role-contrast version. Both overviews have editable SVGs, vector PDFs, and
600-dpi PNGs (main: 8400 x 5310; lookup: 8400 x 5700). The main PDF also appears at
`output/pdf/deltatrace-overview.pdf`.

## Reproduction

Requires Python, NumPy, Matplotlib and pypdf. Run from the paper directory:

```sh
python figures/build_figures.py --overview-only
python figures/check_overview_layout.py
python verify_source.py
```

The overview-only option preserves the two existing case figure assets and
reassembles `output/pdf/deltatrace-figures.pdf` from the actual vector PDFs.
The `--cases-only` option rebuilds the retrieval and multi-hop figures while
preserving the overview assets. Without either option it rebuilds all figures.
All three manuscript figures retain their editable sources.

The data preparation checks archived report hashes, the fixed released task
cache, full actual model-input token IDs, exact decoded prompt and target,
and the recorded positive-score projection. It only decodes stored IDs:

```sh
python figures/prepare_case_data.py --audit-root /path/to/audit --index 1 --datasets morehopqa --output figures/data/overview_role_case.json
python figures/prepare_case_data.py --audit-root /path/to/audit --index 6 --datasets niah_mq_q2 --output figures/data/overview_case.json
```

`data/cases.json` contains the four original index-0 case fixtures. The case
figures now use compact paper subpanels: (a) Qwen3-8B, (b) Qwen3.5-9B, and
(c) deletion. Task descriptions appear in the manuscript captions. Global
titles and internal development identifiers are omitted from the artwork.
Answer is a normal-size annotation. Context, question and full-input strips
align across the two model columns. One quantitative colorbar sits in the right column below the deletion curve and model legend, as requested for both case figures.
The original shared linear scales are retained. Each deletion plot has four 21-point curves: DT and one-hop FT for each model, with model colors and solid/dashed method lines. The y-axis identifies normalized log-likelihood of the full response, including EOS. Each model/example uses its own identical DT/FT full-input and fully-deleted log-likelihood endpoints; the released clipping and cumulative-minimum normalization are verified directly from the saved scores. Each displayed token run is bound to its original token score.
The cases export editable SVGs, vector PDFs and 600-dpi PNGs. Standalone PDFs
also appear at `output/pdf/deltatrace-retrieval.pdf` and
`output/pdf/deltatrace-multihop.pdf`.

```sh
python figures/build_figures.py --cases-only
python figures/check_case_layout.py
python verify_source.py
```

These are illustrations of the paired development cases described in the
evaluation. The visual revisions use the archived scores and responses.

The [FlashTrace overview](https://arxiv.org/html/2602.01914v4) informed the
layered operator language, local mechanism expansion, concrete text evidence,
consistent role colors, and compact legends. Uniform blue/purple matrix cells
are schematic operators, not measured attention maps. Teal/coral backgrounds
are measured signed input contributions.
