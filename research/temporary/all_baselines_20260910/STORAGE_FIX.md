# Saved-tensor storage amendment before baseline quality scoring

After the numeric fix, all 20 VT technical pilots passed. HotpotQA index 0
passed Perturbation, CLP and IFR, then AttnLRP exhausted the 64 GiB GPU while
retaining its backward graph. Preserve that exception and all 23 successes.

Wrap only AttnLRP in PyTorch save_on_cpu(pin_memory=True), transferring tensors
saved for backward to host memory and restoring their original device before
use. This changes storage and transfer cost, not tensor values, forward
arithmetic, target seeds or backward rules. Do not change model training mode,
checkpointing, precision, graph, attention implementation, or any task scope.
Record a bitwise comparison against the completed VT H2-C3 AttnLRP pilot.

The exact successful pre-offload execution identity is registered in
execution_compatibility.json. Reuse those immutable 23 pilot records; newly
completed records include this amendment and compatibility-manifest hash.
Only exact registered versions may resume. All old failures remain available.
No baseline quality metric was inspected to select either technical fix.
