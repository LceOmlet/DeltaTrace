set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
export CUDA_VISIBLE_DEVICES=2
"$VENV_PYTHON" - <<'PY'
import subprocess,re
text=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s+2\s+\d+\s+',text.split('| Process:')[-1],re.M),text
PY
for rank in 0 1; do
 "$VENV_PYTHON" -u /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/gdn-cached-conv-interface-20261007/base-native-b8-long-v1/compare_actual_cached_conv.py --payload /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/gdn-cached-conv-interface-20261007/base-native-b8-long-v1/rank${rank}.pt --output /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/gdn-cached-conv-interface-20261007/base-native-b8-long-v1/rank${rank}-operator-result.json --official-test /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/gdn-cached-conv-interface-20261007/official-initial-states-actual-v1/test_causal_conv1d_v150.py --benchmark > /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/gdn-cached-conv-interface-20261007/base-native-b8-long-v1/rank${rank}-operator.log 2>&1
done
