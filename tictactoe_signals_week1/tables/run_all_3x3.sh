#!/usr/bin/env bash
set -euo pipefail
# run_all_3x3.sh — Full 3x3 pipeline (K=3), all plies 0..9

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TABLES="$ROOT/tables"
SIG="$ROOT/signals"
OUT="$ROOT/out_3x3_full"

mkdir -p "$OUT"/tables_out "$OUT"/plots

# 1) Brute-force JSON (all plies 0..9)
python3 "$TABLES/build_json_bruteforce.py" --N 3 --K 3 --max_ply 9 --out_json "$OUT/bruteforce_N3_ply9.json"

# 2) Normalize to tensors + Peff
python3 "$TABLES/build_tables.py" --input_json "$OUT/bruteforce_N3_ply9.json" --N 3 --max_ply 10 --beta 0.5 --out_dir "$OUT/tables_out"

# 3) Extract signals + plots + inverse MAE check for each ply and role
for PLY in $(seq 0 9); do
  for ROLE in X O; do
    python3 "$SIG/extract_signals.py" \
      --tables_npz "$OUT/tables_out/tables_N3.npz" \
      --ply "$PLY" --role "$ROLE" --out_dir "$OUT/plots" --eval_inverse
  done
done

echo "[DONE] Outputs under $OUT"
