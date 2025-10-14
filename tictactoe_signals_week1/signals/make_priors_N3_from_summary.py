#!/usr/bin/env python3
# signals/make_priors_N3_from_summary.py
import pandas as pd, json, numpy as np, argparse
from pathlib import Path

def norm_mean_one(vals):
    arr = np.array(vals, dtype=float)
    m = arr.mean()
    return (arr / m).tolist() if m > 0 else [1.0] * len(arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True, help="out_3x3_full/signals_summary_beta0.csv")
    ap.add_argument("--out_json", required=True, help="priors_N3_beta0.json")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    out = {"N3": {}}

    for _, r in df.iterrows():
        ply  = int(r["ply"])
        role = r["role"]
        key  = f"ply_{ply}_{role}"

        # D4 三项（corner/edges/center）
        d4_corner = float(r["d4_corner"])
        d4_edges  = float(r["d4_edges"])
        # 兼容列名：如果没有 d4_center（理论上 N=3 会有），给个兜底
        d4_center = float(r.get("d4_center", 1.0))
        s = d4_corner + d4_edges + d4_center
        if s <= 0: s = 1.0
        d4_weights = {
            "corner": d4_corner / s,
            "edge":   d4_edges  / s,
            "center": d4_center / s,
        }

        rows = [float(r[f"row_{i}"]) for i in range(3)]
        cols = [float(r[f"col_{j}"]) for j in range(3)]
        shells = [float(r[f"shell_{k}"]) for k in range(3)]  # r=0..2

        out["N3"][key] = {
            "d4_weights": d4_weights,
            "row_bias": norm_mean_one(rows),
            "col_bias": norm_mean_one(cols),
            "shell_bias": norm_mean_one(shells),
            "diag_main": float(r["diag_main"]),
            "diag_anti": float(r["diag_anti"]),
        }

    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {args.out_json}")

if __name__ == "__main__":
    main()
