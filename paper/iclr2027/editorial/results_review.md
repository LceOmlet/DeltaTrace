# Results and PDF review - 2026-09-11

- The three quality tables contain all seven methods: 13 tasks for RISE/MAS and 11 for recovery. VT budgets are 10%/10%/20%/30%; HotpotQA reports supporting-fact Recall at a 10% body-token budget.
- The combined CSV has 91 rows and 259 metric values. All 224 historical values are preserved; 35 verified VT/HotpotQA values and seven VT macro values are added. Budget, metric unit, target scope and method variant are explicit in the CSV.
- The independent check verifies every formatted table cell, budget and boldface winner against source records. Source and official-template checks pass. No model calls were made during table integration.
- The final PDF has 17 pages: main text ends on page 9, references span pages 9-10, and the appendix spans pages 11-17. All pages were visually checked; the recovery table and revised appendix pages were inspected individually. No clipping, blank appendix page, compiler warnings, or overfull/underfull boxes remain.

Verification: `results/verification.json`, `source_verification.json`, and `build_receipt.json`.
