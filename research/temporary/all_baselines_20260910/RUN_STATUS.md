# Full baseline run in progress

The 25 technical pilots have passed. The full run is executing all 2,240 new
method-case pairs (five methods on 448 fixed inputs), with completed pilots reused.
DT and FT attribution are not rerun. The attribution controller is detached on
the remote server and checkpoints every completed case.

The one-off [collector](collect_when_complete.py) watches this execution and will
download the complete archive, recompute scores, build the report and run the
independent verifier. A complete comparison has not yet been produced.

- [Frozen protocol](PROTOCOL.md)
- [Numeric correction](NUMERIC_FIX.md) and [saved-tensor storage correction](STORAGE_FIX.md)
- [Successful bitwise storage control](storage_control/verification.json)
- [Execution compatibility and preserved pilot controls](execution_compatibility.json)
- [Full-run controller](run_controller.py) and [operational progress reader](progress.py)

Remote output: `/tmp/codex_source_v2_gpu_20260910_v1/all_baselines_v2`.
Remote full-run log: `/tmp/codex_source_v2_gpu_20260910_v1/all_baselines_logs_v3/full.log`.
The local `audit` and `raw` directories contain verified transfer snapshots,
so their current file counts can lag the remote execution.
