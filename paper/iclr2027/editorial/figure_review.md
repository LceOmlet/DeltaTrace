# Figure review — 2026-09-09

The final figure set contains a method diagram and two case visualizations.
The method figure gives the actual example most of its area. The reading
order is the exact response excerpt, a single reverse-traversal arrow, and
the measured signed input heatmap. The adjacent attention identity explains
the two roles of content and selection, using the surname and question as
examples. The conservation identity refers to all input sources. Typography,
whitespace, and one perspective sheet establish the hierarchy; no decorative
header rules or crossing paths are used.

The heatmap retains original token scores, token boundaries, source positions,
and sign. Its four text excerpts and exact response excerpt are recorded in
`figures/figure_manifest.json`. The single perspective sheet is a layout
device for the input heatmap, not a claim about an intermediate model state.
The arrow and local identity explain propagation; branch scores are not
encoded in their color or width.

The case figures use the first released retrieval and multi-hop development
example on Qwen3-8B and Qwen3.5-9B. The data preparation verifies the archived
numeric records, raw reports, released inputs, token IDs, exact decoded text,
fixed targets, and positive projection. Four complete model inputs and their
21-point deletion curves are preserved in `figures/data/cases.json`.

Both model columns share the color scale within each example. The scale is
centered on zero and linear, and includes the largest absolute eligible-token
value across both complete inputs. The main figure shares the multi-hop
scale. Full-input strips in the case figures locate the enlarged excerpts.
Their response panels use answer summaries, while the main figure quotes an
exact target excerpt. The measured attribution target is the entire released
response plus EOS. Deletion curves retain the recorded samples without smoothing.

Visual review covered all three pages of the standalone figure collection at
125 dpi and every manuscript page at 115 dpi. Checks include text fit,
legible labels at paper width, one clear arrow direction, perspective text
alignment, quantitative color legends, excerpt markers, complete captions, and
page flow. The generated PDF figures are vectors; editable SVGs and their
Python sources are retained. `build_receipt.json` records the final page
images and output hashes, while `verify_source.py` checks the included
figure assets and drawing-source hashes.

FlashTrace informed the use of a concrete example as the visual center,
perspective treatment, alignment, and compact legends. The method content
remains DeltaTrace's finite propagation and signed source contributions. The five existing
bibliographic entries are unchanged; their independent review remains in
`citation_review.md`.
