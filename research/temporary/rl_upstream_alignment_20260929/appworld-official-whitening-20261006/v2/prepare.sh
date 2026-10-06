set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export MACA_VISIBLE_DEVICES=""
"$VENV_PYTHON" /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-official-whitening-20261006-v2/source/inspect_appworld_whitening_readiness_cpu.py --input /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-official-whitening-20261006-v2/preparation-input.json
