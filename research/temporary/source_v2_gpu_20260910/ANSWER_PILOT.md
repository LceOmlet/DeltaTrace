# Answer-target pilot: second hypothesis after the failed shared Recall repair

The first 80-case validation failed the shared-advantage criterion. Its
VT-H2-C3 result is retained: DT sentence-density Recall 48.11%, FT K3 74.67%.
The three single-chain VT tasks tied at the budget ceiling; HotpotQA had a
positive but uncertain difference. We do not tune further on those 80 cases.

Mechanistic hypothesis: the complete response sometimes explicitly explains
irrelevant variable chains, whereas retrieval gold labels only the queried
chain. Explaining all generated text can legitimately credit distractor facts.
Test output-target selection while retaining the frozen native model and
finite propagation rules. This changes the attribution question explicitly.

Fixed controls, each with live matched FT K1/K3:

- `full`: original full response. Match DT's seed positions to FT's existing
  non-stop, non-EOS generation aggregation. Also retain the original all-token
  DT seed as a control.
- `answer_conditioned`: same complete model input and fixed response, but
  seed only the stored final-answer span, excluding the same stop tokens.
  FT uses that same aggregation span through its existing public span API.
- `answer_only`: original prompt plus the stored generated final-answer
  substring and EOS, with the earlier generated reasoning removed. Both
  methods see the same shortened input and explain the same non-stop answer
  tokens. This separates the final-answer objective from conditioning on an
  already-written reasoning trace.

The answer substring comes only from the stored generated target's existing
token span and tokenizer offsets. No metadata outputs, reference answers,
gold spans, or document labels are read by the extractor. Gold is used only
after attribution to score the same evidence-body token Recall.

Keep the original full-prompt EOS reference for DT in all three controls.
Primary ranking remains the previously fixed sentence-density ordering, with
exactly ceil(0.10*N_source) tokens. FT receives the identical ordering. Raw
token ordering and 5%/20% budgets are diagnostics, not alternative selection
criteria. The weighted-seed implementation is a separate experimental copy;
all 27 frozen method files and the native finite attention library remain
unchanged. The original unweighted control must reproduce saved vectors.

Development: first 8 cases of VT-H2-C3 under the prior deterministic hash
ordering (unused by the first pilot), and the same 8 HotpotQA development
cases. Validation: positions 24:40 of that ordering in all five tasks, 16
per task, disjoint from the first validation and all development cases.
Save indices before new measurements. These are reserved cases from an
already audited benchmark, not a new external dataset.

Choose the target setting with the largest minimum DT-minus-matched-FT-K3
development task mean; ties use macro mean then name. Advance only when at
least one task improves and every other task improves or both methods attain
the exact per-case budget ceiling. Freeze the choice before validation.
Validation requires positive means for both the VT group and HotpotQA and a
positive lower 95% paired-bootstrap macro bound, as in the first pilot. Keep
all failed controls and charge all model calls. If the criterion fails, retain
the failure instead of selecting a different validation target.
