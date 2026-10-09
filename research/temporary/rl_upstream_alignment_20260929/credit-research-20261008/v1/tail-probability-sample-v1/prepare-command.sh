set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
cd /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-tail-probability-sample-20261009-v1
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" test_tail_probability_statistics.py
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" prepare_tail_probability_sample.py --directory /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-tail-probability-sample-20261009-v1 --output /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-tail-probability-sample-20261009-v1/sample.json
