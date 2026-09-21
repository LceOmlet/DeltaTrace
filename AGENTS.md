# RL method: one fixed plan

Before changing or testing RL credit assignment, read `experiments/rl/PLAN.md`.
It is the sole method specification accepted by the user. The RL README is
only a status/index document; environment receipts and historical results do
not define the method. Do not introduce an alternative plan in another file.

Keep the specified token-level Q^{DT}, V^{DT}, A^{DT}=Q^{DT}-V^{DT}, observation
mask, and upstream PPO objective. Every reasoning, tool-call, and final token
is a separate policy action. Do not substitute scalar/span credit, a learned
value model, observation routing, or independent reward allocation for the
specified Q/V construction. Reuse existing upstream capabilities and official
DeltaTrace interfaces; do not create shadow implementations behind them.

Check implementation changes and test claims against PLAN.md. If code differs,
correct the code against that plan; do not redefine the plan to match the code.
On an actual blocker, stop the affected implementation and report the concrete
issue to the user, as requested, without silently substituting another method.
Historical success, tensor shape checks, and attribution conservation alone
are not evidence that the accepted Q/V construction has been implemented.

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
