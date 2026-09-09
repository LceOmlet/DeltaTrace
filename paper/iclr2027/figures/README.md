# DeltaTrace figures

The method figure begins with an exact excerpt from the stored response and
uses the actual Qwen3.5 multi-hop input heatmap as its main visual. One arrow
connects the fixed-response score change to the signed source contributions.
The adjacent attention identity explains how carried content and changed
selection each receive credit, using the surname and the question as examples
of those roles. A single perspective sheet displays measured input scores.
The sheet is a layout device; it does not represent an intermediate model
state. The local identity and reverse arrow describe the method. Exact source
spans, the target excerpt, and the shared scale are in `figure_manifest.json`.

The two case figures use stored `clean-v1-20260909` DT results for the first
released example (index 0) of `niah_mq_q2` and `morehopqa`, on both models.
Selection is fixed by release order. No inference or new attribution run is
needed to regenerate the figures.

## Reproduction

With Python, NumPy and Matplotlib installed:

```sh
python figures/build_figures.py
```

This reads `data/cases.json` and uses `draw_mechanism.py` for the perspective
method diagram. It writes three editable SVG files, three vector
PDF assets, PNG viewing copies, `figure_manifest.json`, and the three-page
`output/pdf/deltatrace-figures.pdf` collection. The manuscript includes the
vector PDF assets through `overview.tex` and `cases.tex`.

To rebuild the fixture from the archived runs:

```sh
python figures/prepare_case_data.py --audit-root /path/to/audit
```

Preparation verifies the source numeric file, original raw reports, released
task-cache hashes, full actual model-input token IDs, exact prompt/response text,
and positive-score projection. Tokenizer JSON files decode recorded IDs only;
their pinned source URLs and byte hashes are embedded in the fixture. Model
weights are not loaded. The fixture preserves both original and UTF-8 byte
positions, so the rendered excerpts use the recorded model-input tokenization.

## Visual semantics

- Teal and coral represent positive and negative DT contributions in nats.
- Blue and purple operator terms distinguish content and selection. They
  illustrate the local allocation; they are not measured branch scores.
- Each task uses one linear color scale for both models, centered on zero.
  It extends to the largest absolute eligible-token contribution across the
  two complete inputs. The main figure uses the same multi-hop scale. Values are not
  clipped, and scores are not normalized separately per model or excerpt.
- Text excerpts are chosen for the question's evidence chain and are recorded
  as exact character spans. Ellipses mark omissions. The small strips retain
  every user-input token in original order; underlines locate the excerpts.
- The main figure quotes an exact response excerpt; case panels use answer
  summaries. Attribution explains the entire
  fixed released response plus EOS, using the original eligible-token EOS
  reference, for each model.
- Curves use the saved original 20-step `normalized_model_response` arrays.
  The x axis uses the recorded number of replaced eligible tokens. No curve
  smoothing is applied. This positive-ranking deletion view is distinct from
  the raw signed token view.
- These are four development-case illustrations. Formal aggregate comparisons
  use their complete aligned evaluation records.

FlashTrace informed the use of a concrete text heatmap as the visual center,
perspective depth, alignment, color legends, and publication layout. The DT
figure centers a fixed response, one reverse traversal, and signed input scores.
