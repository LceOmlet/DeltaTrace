# Development ceiling eligibility correction, before validation

The deterministic development sample revealed a structural limitation in the
strictly-positive-per-task advancement condition: on all eight VT-H4-C1 cases,
FT K=3 and the winning DT setting both select only gold tokens. Both therefore
attain the exact token-budget ceiling on every case. Their mean Recall is
0.7083567674561464 and no method can strictly beat FT on this sample at this
budget. The development indices, candidate ranking rule, metric, budgets, and
80 reserved validation indices remain unchanged.

The ranking rule already selects `full/forward/density`. It has a positive
HotpotQA difference of 0.060137523565874276 and a VT difference of exactly zero.
It may advance only because every zero-difference task is jointly at the exact
ceiling on every development case. This correction changes eligibility, not
candidate selection. A zero difference below ceiling does not qualify.

All original negative variants remain in the development report. No validation
candidate score has been inspected. Validation superiority still requires the
conditions in RECALL_PILOT.md; ceiling ties will be reported as ties.
