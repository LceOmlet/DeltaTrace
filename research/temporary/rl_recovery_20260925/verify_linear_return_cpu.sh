#!/bin/bash
set -e
root=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
candidate="$root/candidates/linear-return-20260925"
if [ ! -d "$candidate" ]; then
    cp -a "$root/releases/9785786" "$candidate"
fi
tar -xf "$root/receipts/rollout-major-cost/linear-return-candidate.tar" -C "$candidate"
export DT_ROOT="$candidate" DT_ENVIRONMENT_JSON="$candidate/environment.json"
source "$candidate/experiments/rl/environments/metax.env.sh"
export PYTHONPATH="$candidate/experiments/rl:$candidate/clean/qwen35:$candidate:$VERL_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$VENV_PYTHON" -m pytest --import-mode=importlib "$candidate/experiments/rl/test_reward_readout.py" "$candidate/experiments/rl/test_counterfactual.py" "$candidate/experiments/rl/test_rollout_credit.py" -q
"$VENV_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from transformers import AutoTokenizer
from reward_readout import RewardAlphabet
tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
results={}
for task,horizon in [('Sokoban',15),('Webshop',15),('AppWorld',40)]:
    a=RewardAlphabet.for_task(task,horizon)
    results[task]=dict(categories=len(a.values), labels=a.labels(), label_ids=a.label_ids(tokenizer),
                       max_query_target_tokens=a.readout_token_budget(tokenizer,horizon))
print(json.dumps(results))
(Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-major-cost/linear-return-tokenizer.json').write_text(json.dumps(results,indent=2)+'\n')
PY
