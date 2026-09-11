# DeltaTrace main figure

## First-page Recall and time inset

`build_intro_teaser.py` produces `../results/figures/deltatrace-recall-time.*`
as a 2.70 by 3.96 inch vector PDF, editable SVG, and 400-dpi PNG. The manuscript
inserts it at its native width below the abstract, with Introduction text
wrapping on the left. It is unnumbered so the approved mechanism remains
Figure 1. All labels are 8 pt at the actual insertion size. Directional text
anchors keep the radar labels outside the outer circle; the builder rejects
text overlap, clipping, intersection with either plot or OOM marks, and less
than 2 pt clearance from the radar circle.

The radar uses all 77 reported Recall values across seven methods and eleven
tasks from the verified `all_methods.csv`. It excludes unavailable MATH and
MoreHopQA recovery and does not duplicate tasks through a macro-average axis.
Every axis shows its own endpoints: the observed extrema over all seven methods
plus 8% of their span, rounded outwards to 5 percentage points for spans below
25 points, otherwise to 10, and bounded within 0–100. All methods share the same
range within a task. Only geometry is normalized; source percentages remain
unchanged. The range policy and every raw value are in `intro_teaser_manifest.json`.
The center and outer ring denote the printed minimum and maximum, respectively;
intermediate rings divide that interval into quarters.

VT-H2/H4 and HotpotQA use 10% budgets, VT-H6 20%, and VT-H10 30%, as marked on
the figure. FT denotes the released NIAH baseline and FT K3 for VT/HotpotQA,
matching the recovery table. NIAH/VT report answer Recall; HotpotQA reports
supporting-fact Recall. These task scopes are retained in the manifest.

The timing panel reads `../../../experiments/efficiency/curve_data.json`.
It preserves all 52 successful points, local min/max error bars, missing-point
gaps and four OOM crosses. DT, FT and FT-mh are local MetaX C550 measurements;
the other curves are historical measurements. The inset labels this
mixed hardware scope, and Section 4 explains it. No timings are rescaled and no
cross-hardware speedup is claimed. The original full-size rollout assets remain
unchanged. Both panels share method colors and marker styles.

From the paper directory:

```sh
python figures/build_intro_teaser.py
python verify_source.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

## Mechanism overview

The manuscript overview uses four panels: (a) paired original/EOS-reference
executions and a finite reverse traversal; (b) Attention; (c) Gated DeltaNet;
(d) a measured signed context example. Panel (a) occupies the wider left column;
Attention is above GDN in the right column. Both operator panels have identical
areas (7.35 x 3.00 drawing units), with matching change identities and reverse
coefficient rules to the right of the forward operator. Content is blue, control
is gold, and reverse coefficients are purple. GDN shows its
full retain/write/read structure and expands the
retention rule T = alpha S; S abbreviates the previous state.

The two gray upward arrows in (a), explicitly labeled Forward, denote the
reference and original executions with the same model and fixed response.
Their output scores define Delta F. The purple downward arrow, labeled Reverse,
denotes a single reverse traversal of the finite local rules. Starting with
coefficient 1 at the scalar score produces input coefficients m_i; taking their
inner products with embedding differences Delta e_i gives A_i. Thus Delta F is
the effect being allocated, while the reverse computation composes its finite
coefficients. Appendix `app:conservation` specifies this finite chain rule.
Panel (a), the caption, and Section 2.2 follow four steps: two forward runs,
local rules from paired activations, reverse coefficient propagation, and input
token scores. Panels (b,c) pair each change identity with the reverse updates it
defines. Delta Y and Delta T label the identities; the arrows below connect each
identity to its coefficient rules. The backward outputs are M_V/M_P and
M_S/m_alpha. The figure defines endpoint subscripts 0 and 1 as reference and
original, and labels the propagated quantity as coefficients m.
Labels are at least 20.4 pt on the 14-inch canvas, or 8.0 pt at the manuscript's
5.5-inch insertion width (excluding mathematical subscripts and superscripts).

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
600-dpi PNGs (main: 8400 x 6390; lookup: 8400 x 6780). The main PDF also appears at
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
consistent role colors, and compact legends. Uniform blue/gold matrix cells
are schematic operators, not measured attention maps. Teal/coral backgrounds
are measured signed input contributions.
