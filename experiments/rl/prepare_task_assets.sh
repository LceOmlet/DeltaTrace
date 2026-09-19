#!/usr/bin/env bash
set -euo pipefail

DT_ROOT="${DT_ROOT:-$PWD}"
VERL_ROOT="${VERL_ROOT:-$DT_ROOT/third_party/verl-agent2}"
WEBSHOP_ROOT="${WEBSHOP_ROOT:-$DT_ROOT/third_party/webshop}"
APPWORLD_ROOT="${APPWORLD_ROOT:-$DT_ROOT/third_party/appworld}"
VENV_PYTHON="${VENV_PYTHON:-$DT_ROOT/env/bin/python}"
APPWORLD_BIN="${APPWORLD_BIN:-$DT_ROOT/env/bin/appworld}"

for path in \
  "$WEBSHOP_ROOT/data/items_shuffle_1000.json" \
  "$WEBSHOP_ROOT/data/items_ins_v2_1000.json" \
  "$WEBSHOP_ROOT/search_engine/indexes_1k"; do
  [[ -e "$path" ]] || {
    echo "Missing official WebShop asset: $path" >&2
    echo "Clone Princeton-NLP/WebShop and run its official setup.sh -d small first." >&2
    exit 2
  }
done

mkdir -p "$VERL_ROOT/agent_system/environments/env_package/webshop/webshop/data"
ln -sfn "$WEBSHOP_ROOT/data/items_shuffle_1000.json" \
  "$VERL_ROOT/agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json"
ln -sfn "$WEBSHOP_ROOT/data/items_ins_v2_1000.json" \
  "$VERL_ROOT/agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_1000.json"
ln -sfn "$WEBSHOP_ROOT/search_engine/indexes_1k" \
  "$VERL_ROOT/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k"
ln -sfn indexes_1k \
  "$VERL_ROOT/agent_system/environments/env_package/webshop/webshop/search_engine/indexes"

if [[ ! -d "$APPWORLD_ROOT/data/tasks" ]]; then
  (
    cd "$APPWORLD_ROOT"
    "$APPWORLD_BIN" install --repo
    "$APPWORLD_BIN" download data
  )
fi

echo "official WebShop and AppWorld assets are ready"
