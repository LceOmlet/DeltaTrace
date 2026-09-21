# Remote environment reuse

Before any A6000/RL setup or run, read `experiments/rl/REMOTE_ENVIRONMENT.md`.
The existing runtime is already provisioned. Source
`experiments/rl/environments/a6000.env.sh` on that host and use its
`VENV_PYTHON`, model, upstream checkouts, owner receipt, and persistent caches.

Do not recreate the environment, reinstall the dependency stack, download the
checkpoint/task assets again, or clear/rebuild CUDA/Triton/Inductor caches as a
routine startup step. First inspect the recorded resource and the exact error.
Repair only a demonstrated missing or broken component, and update the receipt
afterwards. Network failures and training-interface failures are not evidence
that packages need reinstalling.

Check current GPU occupancy before choosing a device. Historical GPU indices
in receipts are not reservations. Keep environment readiness separate from
DT/RL training correctness and dataset evaluation results.
