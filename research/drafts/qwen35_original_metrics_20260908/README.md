# Deferred, unexecuted metric runner

These files were prepared for the five-method, two-example original RISE/MAS
screen. That screen was deferred before launch when both current Qwen3.5
adaptations failed NI0 evidence recovery. No actual B2 metric execution or
equivalence result is claimed for this draft.

The budget in `configs/qwen35_original_metrics_budget_20260908.json` is historical
and deferred. Do not launch this draft by treating that budget as the current
next step. First resolve the migration target change described in
`docs/history/Qwen35共同归因失败与目标错位诊断_20260908.md`.

`author_metric_batching.py` is the proposed request scheduler; `study.py` is the
proposed driver. Their syntax has been checked, but runtime behavior remains
unverified. Paths in the archived driver are redacted reproduction placeholders.
