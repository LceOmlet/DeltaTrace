# Figure review — 2026-09-09

The final figure set contains a method diagram and two case visualizations.
The diagram uses paired perspective planes for the original and reference
executions, one response-score change entering the model, and content,
selection, and memory paths back to the input. Attention glyphs and the
memory update explain the operations along these paths. The surname example
connects this overview to the multi-hop case and the method prose.

The method planes and internal paths are schematic. Four terminal bars are
actual Qwen3.5 source-span sums, with their original token groups, input hash,
and unrounded values recorded in `figures/figure_manifest.json`. The displayed
source subset is separate from the conservation identity over all sources.

The case figures use the first released retrieval and multi-hop development
example on Qwen3-8B and Qwen3.5-9B. The data preparation verifies the archived
numeric records, raw reports, released inputs, token IDs, exact decoded text,
fixed targets, and positive projection. Four complete model inputs and their
21-point deletion curves are preserved in `figures/data/cases.json`.

Both model columns share the color scale within each example. The scale is
centered on zero, includes the largest absolute eligible-token value across
both complete inputs, and uses a linear region of one nat on either side of
zero. Full-input strips locate the enlarged excerpts. The response cards are
answer summaries; the measured attribution target is the entire released
response plus EOS. Deletion curves retain the recorded samples without
smoothing.

Visual review covered all three pages of the standalone figure collection at
125 dpi and every manuscript page at 115 dpi. Checks include text fit,
legible labels at paper width, arrow direction, paired-plane depth, source
bar values, color legends, excerpt markers, complete figure captions, and
page flow. The generated PDF figures are vectors; editable SVGs and their
Python sources are retained. `build_receipt.json` records the final page
images and output hashes, while `verify_source.py` checks the included
figure assets and drawing-source hashes.

FlashTrace informed the perspective treatment, alignment, overview/detail
pairing, and compact legends. The method content remains DeltaTrace's
finite propagation and signed source contributions. The five existing
bibliographic entries are unchanged; their independent review remains in
`citation_review.md`.
