set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-efficiency-20261007/row-prefix-owner-fresh-v1/prepare_cpu.py
