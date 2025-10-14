# signals/make_priors_N4_from_N3.py
import json, numpy as np, argparse, math
from pathlib import Path

def norm_mean_one(arr):
    arr = np.array(arr, dtype=float)
    m = arr.mean()
    return (arr / m).tolist() if m>0 else [1.0]*len(arr)

def upsample_1d(v_src, tgt_len):
    # 线性插值把长度 Lsrc → Ltgt（如 3→4）
    src = np.array(v_src, dtype=float)
    x_src = np.linspace(0, 1, num=len(src))
    x_tgt = np.linspace(0, 1, num=tgt_len)
    v = np.interp(x_tgt, x_src, src)
    return v.tolist()

def extend_shells_0to2_to_0to4(v_src):
    # shells: N=3 只有 r=0..2，N=4 需要 r=0..4
    # 用线性趋势外推 r=3,4，并确保均值归一化
    v = list(v_src)
    x = np.arange(len(v))
    coeff = np.polyfit(x, v, 1)  # 简单线性
    for r in [3,4]:
        v.append(coeff[0]*r + coeff[1])
    v = [max(1e-6, float(a)) for a in v]  # 保正
    return norm_mean_one(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors_n3", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_ply", type=int, default=3, help="只迁移 4x4 浅层 0..max_ply")
    args = ap.parse_args()

    P3 = json.load(open(args.priors_n3))
    out = {"N4": {}}

    # 只迁移 plies 0..max_ply，roles X/O
    for ply in range(args.max_ply+1):
        for role in ["X","O"]:
            key3 = f"ply_{ply}_{role}"
            src = P3["N3"].get(key3)
            if not src:  # 没有就跳过
                continue

            # D4：N=4 没有“center”这个 D4 组（只有 corner/edge）
            # 我们把 N=3 的 center 权重平滑分配给 4×4 的“内圈”通过 row/col/shell 偏置体现；
            # 因此 D4 只保留 corner/edge，按原有比例归一化。
            w_corner = src["d4_weights"]["corner"]
            w_edge   = src["d4_weights"]["edge"]
            # 丢弃 center，重新归一化
            s = w_corner + w_edge
            d4_weights = {
                "corner": float(w_corner/s) if s>0 else 0.5,
                "edge":   float(w_edge/s) if s>0 else 0.5
            }

            # Row/Col：3→4 线性插值；并把均值归一化为 1
            row4 = norm_mean_one(upsample_1d(src["row_bias"], 4))
            col4 = norm_mean_one(upsample_1d(src["col_bias"], 4))

            # Shells：r=0..2 → r=0..4 线性外推；归一化为 1
            shell5 = extend_shells_0to2_to_0to4(src["shell_bias"])

            out["N4"][f"ply_{ply}_{role}"] = {
                "d4_weights": d4_weights,   # 只有 corner/edge
                "row_bias": row4,           # 长度 4
                "col_bias": col4,           # 长度 4
                "shell_bias": shell5,       # 长度 5 (r=0..4)
                # 提供对角偏置：保留原值，供你在 N=4 时酌情使用
                "diag_main": src.get("diag_main", 1.0),
                "diag_anti": src.get("diag_anti", 1.0)
            }

    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {args.out_json}")

if __name__ == "__main__":
    main()
