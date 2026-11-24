#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/infer_unbound_many.sh INPUT_DIR OUTPUT_DIR [SEEDS...]

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT="$(realpath "${1:?Missing input directory}")"
OUTPUT="$(realpath -m "${2:?Missing output directory}")"
shift 2

# Default reference seeds: 8 through 27
if [ "$#" -eq 0 ]; then
    set -- $(seq 8 27)
fi

mkdir -p "$OUTPUT"

for seed in "$@"; do
    echo "=== HierADiff seed $seed ==="

    bash "$ROOT/scripts/infer_unbound.sh" \
        "$INPUT" \
        "$seed" \
        40 \
        "$OUTPUT/seed_$seed"
done

# Rank predicted structures by original model confidence.
python - "$OUTPUT" "$@" <<'PY'
import csv
import pickle
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
rows = []

for seed in seeds:
    run = root / f"seed_{seed}"
    results_files = list(run.glob("epoch-*/all_results.pkl"))

    if len(results_files) != 1:
        raise RuntimeError(
            f"Expected one results file for seed {seed}: {results_files}"
        )

    results_file = results_files[0]
    with results_file.open("rb") as f:
        results = pickle.load(f)

    entries = [
        value for value in results.values()
        if isinstance(value, dict) and value.get("pLDDT")
    ]
    if len(entries) != 1:
        raise RuntimeError(f"Unexpected confidence data: seed {seed}")

    confidence = float(entries[0]["pLDDT"][0])

    pdbs = list(results_file.parent.glob(f"*_rs{seed}.pdb"))
    if len(pdbs) != 1:
        raise RuntimeError(f"Unexpected PDB outputs: seed {seed}")

    rows.append((seed, confidence, pdbs[0]))

rows.sort(key=lambda x: x[1], reverse=True)

ranked_dir = root / "ranked"
ranked_dir.mkdir(exist_ok=True)

with (root / "ranked_poses.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "seed", "confidence", "pdb_path"])

    for rank, (seed, confidence, source) in enumerate(rows, 1):
        target = ranked_dir / f"seed_{seed}.pdb"
        shutil.copy2(source, target)
        writer.writerow([
            rank,
            seed,
            f"{confidence:.9f}",
            f"ranked/{target.name}",
        ])
        print(f"{rank:2d}. seed={seed} confidence={confidence:.9f}")

print("MULTISEED_RANKING_SUCCESS")
PY
