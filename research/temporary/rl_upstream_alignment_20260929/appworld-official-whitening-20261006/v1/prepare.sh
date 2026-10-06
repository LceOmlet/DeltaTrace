set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export MACA_VISIBLE_DEVICES=""
"$VENV_PYTHON" /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-official-whitening-20261006-v1/source/inspect_appworld_official_whitening_cpu.py --input /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-official-whitening-20261006-v1/preparation-input.json
