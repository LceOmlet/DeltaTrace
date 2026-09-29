#!/usr/bin/env bash
# Invoke the complete owner recipe; do not restate its training parameters here.
set -euo pipefail
: "${VERL_ROOT:?Use the recorded, provisioned VERL checkout}"
: "${VENV_PYTHON:?Use the recorded Python executable}"
: "${ENV_NAME:?Webshop or AppWorld}"
: "${METHOD:?dt or grpo}"

case "$ENV_NAME" in
  Webshop) recipe=examples/grpo_trainer/run_webshop.sh ;;
  AppWorld)
    echo 'AppWorld is under environment-only comparison with pristine LOOP. No training configuration is selected by this launcher.' >&2
    exit 2 ;;
  *) echo 'This experiment currently covers Webshop and AppWorld only.' >&2; exit 2 ;;
esac
case "$METHOD" in
  dt) estimator=deltatrace ;;
  grpo) estimator=grpo ;;
  *) echo 'METHOD must be dt or grpo.' >&2; exit 2 ;;
esac

cd "$VERL_ROOT"
# The owner calls python3. Resolve that name to the existing runtime, without
# intercepting modules, rewriting arguments, or duplicating data preparation.
python3() { "$VENV_PYTHON" "$@"; }

# Its first argument selects ENGINE, and its unquoted $@ also forwards all
# arguments to Hydra. An empty first argument selects its default vLLM and is
# omitted by that forwarding. Keep the owner file itself unchanged.
source "$recipe" "" \
  "algorithm.adv_estimator=$estimator" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE:-4}" \
  "$@"
