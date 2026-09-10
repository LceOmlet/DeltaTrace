# Saved-tensor storage amendment before baseline quality scoring

After the numeric fix, all 20 VT technical pilots passed. HotpotQA index 0
passed Perturbation, CLP and IFR, then AttnLRP exhausted the 64 GiB GPU while
retaining its backward graph. Preserve that exception and all 23 successes.

The first generic PyTorch save_on_cpu(pin_memory=True) trial did not pass the
required bitwise check (prompt-vector relative L1 difference 0.00023172).
Its pinned host allocation makes strided saved views contiguous. Preserve that
failed control and its values; do not accept a relaxed tolerance.

Use saved_tensors_hooks only around AttnLRP, copying each saved activation's
complete storage to pinned host memory and reconstructing the original shape,
strides and storage offset on its original device before backward. Model
parameter storages remain resident, avoiding redundant transfer. Each saved
activation gets its own copy; do not cache by pointer because freed GPU storage
addresses can be reused during the same forward. The required bitwise control
checks that this storage change preserves the completed pilot result. Do not change model training mode,
checkpointing, precision, graph, attention implementation, or any task scope.
Record a bitwise comparison against the completed VT H2-C3 AttnLRP pilot.

The exact successful pre-offload execution identity is registered in
execution_compatibility.json. Reuse those immutable 23 pilot records; newly
completed records include this amendment and compatibility-manifest hash.
Only exact registered versions may resume. All old failures remain available.
No baseline quality metric was inspected to select either technical fix.
