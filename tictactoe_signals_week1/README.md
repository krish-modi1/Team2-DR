# Week 1 · Data shaping & “sound‑wave” signals (pilot)

## Structure
- tables/build_tables.py — 构建张量（tensors）P_win, P_draw, P_eff
- signals/extract_signals.py — 提取一维“声波（sound‑wave）”信号；最小二乘逆投影（inverse projection）

## Quickstart (mock data)
```bash
cd tables
python3 build_tables.py --mock --N 3 --max_ply 4 --beta 0.5 --out_dir ../tables_out

cd ../signals
python3 extract_signals.py --tables_npz ../tables_out/tables_N3.npz --ply 0 --role X --out_dir ../plots --eval_inverse
```

Outputs:
- plots/*.png — 各家族（families）曲线
- plots/recon_*.npy — 3×3 逆投影重建结果
