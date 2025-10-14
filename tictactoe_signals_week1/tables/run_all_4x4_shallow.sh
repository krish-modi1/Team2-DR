#!/usr/bin/env bash
set -euo pipefail
# run_all_4x4_shallow.sh — 4x4 shallow layers (K=4), plies 0..3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TABLES="$ROOT/tables"
SIG="$ROOT/signals"
OUT="$ROOT/out_4x4_shallow"

mkdir -p "$OUT"/tables_out "$OUT"/plots

# 1) Brute-force JSON (plies 0..3)
python3 "$TABLES/build_json_bruteforce.py" --N 4 --K 4 --max_ply 3 --out_json "$OUT/bruteforce_N4_ply3.json"

# 2) Normalize to tensors + Peff
python3 "$TABLES/build_tables.py" --input_json "$OUT/bruteforce_N4_ply3.json" --N 4 --max_ply 4 --beta 0.5 --out_dir "$OUT/tables_out"

# 3) Extract signals + plots (no inverse MAE for N=4)
for PLY in $(seq 0 3); do
  for ROLE in X O; do
    python3 "$SIG/extract_signals.py" \
      --tables_npz "$OUT/tables_out/tables_N4.npz" \
      --ply "$PLY" --role "$ROLE" --out_dir "$OUT/plots"
  done
done

echo "[DONE] Outputs under $OUT"
