#!/usr/bin/env python3
# 4x4 (k=4) TicTacToe exhaustive enumeration (NO minimax) with:
# - Bitboards (two 16-bit ints for X and O)
# - D4 symmetry dedup via precomputed 8 transforms
# - Memoization
# - Optional multiprocessing over first moves
# - Heatmaps for EV / P_Xwin / P_Draw / P_Owin
#
# Examples:
#   python tictactoe_4x4_bitboard.py --all-metrics --outdir out_n4_k4 --jobs 8
#   python tictactoe_4x4_bitboard.py --metric px --out heatmap_px_n4_k4.png --jobs 8 --dump-json n4_k4.json

import argparse, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# -------- Bitboard basics (4x4) --------
# Cell index i = r*4 + c (row-major), r,c in [0..3], bit i is (1 << i)
ALL_CELLS_MASK = 0xFFFF

# 10 winning masks: 4 rows, 4 cols, 2 diagonals
WIN_MASKS = [
    0x000F, 0x00F0, 0x0F00, 0xF000,  # rows
    0x1111, 0x2222, 0x4444, 0x8888,  # cols
    0x8421, 0x1248                    # diags (\ and /)
]

def has_win(mask: int) -> bool:
    for m in WIN_MASKS:
        if (mask & m) == m:
            return True
    return False

def legal_moves(xmask: int, omask: int) -> List[int]:
    empties = ~(xmask | omask) & ALL_CELLS_MASK
    # iterate set bits
    moves = []
    m = empties
    while m:
        lsb = m & -m
        moves.append(lsb)
        m ^= lsb
    return moves

# -------- D4 (8) transforms for 4x4 grid --------
# Precompute index maps for rotations and flips.
def idx(r, c): return r*4 + c

def rot90_map():
    # (r,c) -> (c, 3-r)
    return [ idx(c, 3-r) for r in range(4) for c in range(4) ]

def rot180_map():
    # (r,c) -> (3-r, 3-c)
    return [ idx(3-r, 3-c) for r in range(4) for c in range(4) ]

def rot270_map():
    # (r,c) -> (3-c, r)
    return [ idx(3-c, r) for r in range(4) for c in range(4) ]

def flip_h_map():
    # horizontal flip: (r,c) -> (r, 3-c)
    return [ idx(r, 3-c) for r in range(4) for c in range(4) ]

def flip_v_map():
    # vertical flip: (r,c) -> (3-r, c)
    return [ idx(3-r, c) for r in range(4) for c in range(4) ]

def flip_d_main_map():
    # main-diagonal flip: (r,c) -> (c, r)
    return [ idx(c, r) for r in range(4) for c in range(4) ]

def flip_d_anti_map():
    # anti-diagonal flip: (r,c) -> (3-c, 3-r) then rotate? simpler explicit:
    # mapping (r,c) to (3-c, 3-r) is a 180-rot + main flip; compute directly:
    return [ idx(3-c, 3-r) for r in range(4) for c in range(4) ]

# identity + 3 rotations + 4 flips = 8 transforms
IDX_MAPS = [
    [idx(r,c) for r in range(4) for c in range(4)],   # identity
    rot90_map(),
    rot180_map(),
    rot270_map(),
    flip_h_map(),
    flip_v_map(),
    flip_d_main_map(),
    flip_d_anti_map()
]

def transform_mask(mask: int, mapping: List[int]) -> int:
    # Build new mask by moving bit at old position p to new position mapping[p]
    new_mask = 0
    m = mask
    while m:
        lsb = m & -m
        p = (lsb.bit_length() - 1)
        new_mask |= (1 << mapping[p])
        m ^= lsb
    return new_mask

def canonical_key(xmask: int, omask: int, player: int) -> Tuple[int,int,int]:
    # Apply all 8 transforms; take lexicographically minimal (x', o') as canonical
    best = None
    for mp in IDX_MAPS:
        x2 = transform_mask(xmask, mp)
        o2 = transform_mask(omask, mp)
        key = (x2, o2)
        if best is None or key < best:
            best = key
    return (best[0], best[1], player)

# -------- Exhaustive counting (no minimax) with memo --------
# memo: (canon_x, canon_o, player) -> (X_wins, Draws, O_wins)
MemoType = Dict[Tuple[int,int,int], Tuple[int,int,int]]

def count_outcomes(xmask: int, omask: int, player: int, memo: MemoType) -> Tuple[int,int,int]:
    # Terminal?
    if has_win(xmask):  return (1,0,0)
    if has_win(omask):  return (0,0,1)
    if (xmask | omask) == ALL_CELLS_MASK:  # draw
        return (0,1,0)

    key = canonical_key(xmask, omask, player)
    if key in memo:
        return memo[key]

    total_x = total_d = total_o = 0
    for mv in legal_moves(xmask, omask):
        if player == 1:   # X to move
            xw, d, ow = count_outcomes(xmask | mv, omask, -1, memo)
        else:             # O to move
            xw, d, ow = count_outcomes(xmask, omask | mv, 1, memo)
        total_x += xw; total_d += d; total_o += ow

    memo[key] = (total_x, total_d, total_o)
    return memo[key]

# -------- Parallel over first moves --------
def _worker_one_first_move(cell_bit: int) -> Tuple[int,int,int,int,int]:
    # Start from empty board; X plays at cell_bit first; then enumerate.
    x0, o0 = cell_bit, 0
    memo: MemoType = {}
    xw, d, ow = count_outcomes(x0, o0, -1, memo)
    return (cell_bit, xw, d, ow, len(memo))

def opening_counts_parallel(jobs: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,Dict[str,Dict[str,int]],int]:
    # 4x4 fixed: 16 cells
    all_bits = [1 << i for i in range(16)]
    x_prob = np.zeros((4,4), dtype=float)
    d_prob = np.zeros((4,4), dtype=float)
    o_prob = np.zeros((4,4), dtype=float)
    ev     = np.zeros((4,4), dtype=float)
    per_cell: Dict[str,Dict[str,int]] = {}
    memo_total = 0

    if jobs is None or jobs <= 0:
        jobs = None  # auto

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_worker_one_first_move, b) for b in all_bits]
        for fut in as_completed(futs):
            cell_bit, xw, d, ow, memo_sz = fut.result()
            idx_pos = (cell_bit.bit_length() - 1)
            r, c = divmod(idx_pos, 4)
            tot = xw + d + ow
            px = xw/tot if tot else 0.0
            pd = d/tot  if tot else 0.0
            po = ow/tot if tot else 0.0
            x_prob[r,c] = px
            d_prob[r,c] = pd
            o_prob[r,c] = po
            ev[r,c] = px - po
            per_cell[f"{r},{c}"] = {"X_wins": xw, "Draws": d, "O_wins": ow, "Total": tot}
            memo_total += memo_sz

    return ev, x_prob, d_prob, o_prob, per_cell, memo_total

# -------- Plotting --------
def plot_heatmap(matrix: np.ndarray, title: str, out_path: str, vmin=None, vmax=None):
    plt.figure(figsize=(5,5))
    plt.imshow(matrix, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    n, m = matrix.shape
    for i in range(n):
        for j in range(m):
            plt.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center")
    plt.xticks(range(m))
    plt.yticks(range(n))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=0, help="并行进程数；0或不填=自动（CPU核数）")
    ap.add_argument("--dump-json", type=str, default="", help="导出 JSON（包含四个矩阵与计数）")

    # 单指标
    ap.add_argument("--metric", type=str, choices=["ev","px","pd","po"], help="单指标模式：ev/px/pd/po")
    ap.add_argument("--out", type=str, help="单指标输出 PNG 路径")

    # 多指标一次性导出
    ap.add_argument("--all-metrics", action="store_true", help="一次性导出 EV/px/pd/po 四张图")
    ap.add_argument("--outdir", type=str, help="--all-metrics 的输出目录")
    args = ap.parse_args()

    # 参数校验
    if args.all_metrics:
        if not args.outdir:
            raise SystemExit("请提供 --outdir 目录（用于保存 EV/px/pd/po 四张图）。")
        os.makedirs(args.outdir, exist_ok=True)
    else:
        if not (args.metric and args.out):
            raise SystemExit("单指标模式需要同时提供 --metric 与 --out。或使用 --all-metrics + --outdir。")

    jobs = None if args.jobs <= 0 else args.jobs

    # 计算（4x4 固定）
    ev, px, pd, po, counts, memo_total = opening_counts_parallel(jobs)

    # 输出图
    if args.all_metrics:
        plot_heatmap(ev, "EV = P(X)-P(O), 4x4", os.path.join(args.outdir, "EV_n4_k4.png"), vmin=-1.0, vmax=1.0)
        plot_heatmap(px, "P_Xwin, 4x4",     os.path.join(args.outdir, "PX_n4_k4.png"), vmin=0.0, vmax=1.0)
        plot_heatmap(pd, "P_Draw, 4x4",     os.path.join(args.outdir, "PD_n4_k4.png"), vmin=0.0, vmax=1.0)
        plot_heatmap(po, "P_Owin, 4x4",     os.path.join(args.outdir, "PO_n4_k4.png"), vmin=0.0, vmax=1.0)
    else:
        mats = {"ev": ev, "px": px, "pd": pd, "po": po}
        ttl  = {"ev": "EV = P(X)-P(O), 4x4",
                "px": "P_Xwin, 4x4",
                "pd": "P_Draw, 4x4",
                "po": "P_Owin, 4x4"}
        if args.metric == "ev":
            plot_heatmap(mats["ev"], ttl["ev"], args.out, vmin=-1.0, vmax=1.0)
        else:
            plot_heatmap(mats[args.metric], ttl[args.metric], args.out, vmin=0.0, vmax=1.0)

    # 导出 JSON（可选）
    if args.dump_json:
        payload = {
            "n": 4, "k": 4,
            "EV": ev.tolist(),
            "P_Xwin": px.tolist(),
            "P_Draw": pd.tolist(),
            "P_Owin": po.tolist(),
            "per_cell_counts": counts,
            "memo_size_total": memo_total,
            "jobs": jobs if jobs else "auto"
        }
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
