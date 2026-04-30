#!/bin/bash
# Evaluate all ablation results and print a summary table.
# Usage: ./scripts/ablations/eval.sh [data_root]
# Default data_root: ./data

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$(realpath "${1:-./data}")"
RESULTS_DIR="$REPO_ROOT/results"

MANIFEST="$REPO_ROOT/cfg/triangle_mapper/experiments/manifest.json"
python3 "$REPO_ROOT/scripts/ablations/generate_configs.py" > /dev/null
rm -rf "$REPO_ROOT/cfg/triangle_mapper/experiments/generated"
mapfile -t EXPERIMENTS < <(python3 -c "
import json
m = json.load(open('$MANIFEST'))
for g in m['groups']:
    for e in g['experiments']:
        print(e['name'])
")

cd "$REPO_ROOT/eval"

for exp in "${EXPERIMENTS[@]}"; do
    result_dir="$RESULTS_DIR/$exp"
    if [ ! -d "$result_dir" ]; then
        echo "Skipping $exp (no results directory)"
        continue
    fi
    echo "--- Evaluating $exp ---"
    python3 eval.py -r "$result_dir" -d "$DATA"
done

echo ""
python3 "$REPO_ROOT/eval/summarize_ablations.py" "$RESULTS_DIR"
