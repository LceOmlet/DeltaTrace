# DeltaTrace figures

The method figure uses DeltaTrace's finite-change attribution, attention
content/selection allocation, gated memory, and signed source contributions.
Paired perspective planes show reference and original activations at three
functional stages: input evidence, model computation, and the fixed response.
The same surname example connects the words supplying content to the question
that guides their selection. The model planes and internal paths are schematic;
terminal source-span bars are real Qwen3.5 scores from the multi-hop case.
Their exact token groups and sums are saved in `figure_manifest.json`.

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
- Blue and purple paths in the method diagram distinguish content and
  selection. Their line widths are visual styling, not measured edge weights.
- Each task uses one symmetric logarithmic color scale for both models. The
  linear region is [-1, 1] nat; the scale extends to the largest absolute
  eligible-token contribution across the two complete inputs. Values are not
  clipped, and scores are not normalized separately per model or excerpt.
- Text excerpts are chosen for the question's evidence chain and are recorded
  as exact character spans. Ellipses mark omissions. The small strips retain
  every user-input token in original order; underlines locate the excerpts.
- The response panel is labeled as a summary. Attribution explains the entire
  fixed released response plus EOS, using the original eligible-token EOS
  reference, for each model.
- Curves use the saved original 20-step `normalized_model_response` arrays.
  The x axis uses the recorded number of replaced eligible tokens. No curve
  smoothing is applied. This positive-ranking deletion view is distinct from
  the raw signed token view.
- These are four development-case illustrations. Formal aggregate comparisons
  use their complete aligned evaluation records.

FlashTrace informed alignment, overview/detail pairing, color legends and
publication layout. Its recursive-hop mechanism, intermediate targets and
hop-difference semantics are not used in these DT figures.
