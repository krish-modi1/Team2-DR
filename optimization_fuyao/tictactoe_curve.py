#!/usr/bin/env python3
# TicTacToe (n×n, k-in-a-row) -- Exhaustive enumeration (NO minimax) + D4 symmetry dedup
# Plot line charts: P(Xwin) vs move number (plies) for selected symmetric openings.
#
# Usage examples:
#   # 4×4, k=4，画角/边/中三条曲线（等权平均），前 10 手
#   python tictactoe_curve.py --n 4 --k 4 --series corner,edge,center --max-depth 10 --out curve_n4_k4.png
#
#   # 3×3, k=3，画角/边/中三条曲线（按路径加权）
#   python tictactoe_curve.py --n 3 --k 3 --series corner,edge,center --weighted --out curve_n3_k3_weighted.png
#
#   # 自定义开局坐标（用分号分隔多个，逗号分隔 r,c），例如 (0,0)、(0,1)、(1,1)
#   python tictactoe_curve.py --n 4 --k 4 --series 0,0;0,1;1,1 --max-depth 12 --out my_curve.png

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# -----------------------------
# Helpers for n×n bitboard
# -----------------------------

@dataclass(frozen=True)
class Config:
    n: int
    k: int
    all_mask: int
    win_masks: List[int]
    d4_maps: List[List[int]]  # 8 maps of indices

def bit_index(n: int, r: int, c: int) -> int:
    return r * n + c

def gen_all_mask(n: int) -> int:
    return (1 << (n*n)) - 1

def gen_win_masks(n: int, k: int) -> List[int]:
    """Generate all 'k-in-a-row' masks along rows/cols/diagonals/anti-diagonals."""
    masks: List[int] = []

    # Rows
    for r in range(n):
        for c0 in range(n - k + 1):
            m = 0
            for j in range(k):
                m |= 1 << bit_index(n, r, c0 + j)
            masks.append(m)

    # Cols
    for c in range(n):
        for r0 in range(n - k + 1):
            m = 0
            for i in range(k):
                m |= 1 << bit_index(n, r0 + i, c)
            masks.append(m)

    # Diagonals (\)
    for r0 in range(n):
        for c0 in range(n):
            # length of this diag segment from (r0,c0)
            length = min(n - r0, n - c0)
            for off in range(max(0, length - k + 1)):
                m = 0
                ok = True
                for t in range(k):
                    rr = r0 + off + t
                    cc = c0 + off + t
                    if rr >= n or cc >= n:
                        ok = False
                        break
                    m |= 1 << bit_index(n, rr, cc)
                if ok:
                    masks.append(m)

    # Anti-diagonals (/)
    for r0 in range(n):
        for c0 in range(n):
            length = min(n - r0, c0 + 1)
            for off in range(max(0, length - k + 1)):
                m = 0
                ok = True
                for t in range(k):
                    rr = r0 + off + t
                    cc = c0 - off - t
                    if rr >= n or cc < 0:
                        ok = False
                        break
                    m |= 1 << bit_index(n, rr, cc)
                if ok:
                    masks.append(m)

    # Remove duplicates (some generation paths may overlap)
    masks = list(sorted(set(masks)))
    return masks

def gen_d4_maps(n: int) -> List[List[int]]:
    """Generate 8 D4 transforms mapping old index -> new index."""
    idx = lambda r, c: bit_index(n, r, c)
    # Identity
    id_map = [idx(r,c) for r in range(n) for c in range(n)]
    # Rotations
    rot90  = [idx(c, n-1-r)   for r in range(n) for c in range(n)]
    rot180 = [idx(n-1-r, n-1-c) for r in range(n) for c in range(n)]
    rot270 = [idx(n-1-c, r)   for r in range(n) for c in range(n)]
    # Flips
    flip_h = [idx(r, n-1-c)   for r in range(n) for c in range(n)]  # horizontal
    flip_v = [idx(n-1-r, c)   for r in range(n) for c in range(n)]  # vertical
    flip_d = [idx(c, r)       for r in range(n) for c in range(n)]  # main diag
    flip_a = [idx(n-1-c, n-1-r) for r in range(n) for c in range(n)]# anti diag
    return [id_map, rot90, rot180, rot270, flip_h, flip_v, flip_d, flip_a]

def make_config(n: int, k: int) -> Config:
    return Config(
        n=n,
        k=k,
        all_mask=gen_all_mask(n),
        win_masks=gen_win_masks(n, k),
        d4_maps=gen_d4_maps(n)
    )

# -----------------------------
# Core game logic
# -----------------------------

def has_win(cfg: Config, mask: int) -> bool:
    for m in cfg.win_masks:
        if (mask & m) == m:
            return True
    return False

def legal_moves(cfg: Config, xmask: int, omask: int) -> List[int]:
    empties = ~(xmask | omask) & cfg.all_mask
    moves = []
    m = empties
    while m:
        lsb = m & -m
        moves.append(lsb)
        m ^= lsb
    return moves

def transform_mask(mask: int, mapping: List[int]) -> int:
    new_mask = 0
    m = mask
    while m:
        lsb = m & -m
        p = (lsb.bit_length() - 1)
        new_mask |= (1 << mapping[p])
        m ^= lsb
    return new_mask

def canonical_key(cfg: Config, xmask: int, omask: int, player: int) -> Tuple[int,int,int]:
    """Return (canon_x, canon_o, player) by trying all 8 D4 transforms and picking lexicographically minimal pair."""
    best_pair: Optional[Tuple[int,int]] = None
    for mp in cfg.d4_maps:
        x2 = transform_mask(xmask, mp)
        o2 = transform_mask(omask, mp)
        pair = (x2, o2)
        if best_pair is None or pair < best_pair:
            best_pair = pair
    return (best_pair[0], best_pair[1], player)

# memo: (canon_x, canon_o, player) -> (X_wins, Draws, O_wins)
MemoType = Dict[Tuple[int,int,int], Tuple[int,int,int]]

def count_outcomes(cfg: Config, xmask: int, omask: int, player: int, memo: MemoType) -> Tuple[int,int,int]:
    """Exhaustively count (X wins, Draws, O wins) from this state, with symmetry-dedup memoization."""
    if has_win(cfg, xmask): return (1,0,0)
    if has_win(cfg, omask): return (0,0,1)
    if (xmask | omask) == cfg.all_mask: return (0,1,0)

    key = canonical_key(cfg, xmask, omask, player)
    if key in memo:
        return memo[key]

    tx = td = to = 0
    for mv in legal_moves(cfg, xmask, omask):
        if player == 1:   # X to move
            xw, d, ow = count_outcomes(cfg, xmask | mv, omask, -1, memo)
        else:             # O to move
            xw, d, ow = count_outcomes(cfg, xmask, omask | mv, 1, memo)
        tx += xw; td += d; to += ow

    memo[key] = (tx, td, to)
    return memo[key]

# -----------------------------
# BFS over canonical states per depth
# -----------------------------

def per_depth_avg_px_for_opening(
    cfg: Config,
    start_rc: Tuple[int,int],
    max_depth: int,
    weighted: bool = False,
    shared_memo: Optional[MemoType] = None
) -> List[float]:
    """
    From opening move at (r,c), build layers of UNIQUE canonical states per ply,
    and compute average P_Xwin at each depth (unweighted or path-weighted).
    """
    r, c = start_rc
    start_bit = 1 << bit_index(cfg.n, r, c)
    # After opening by X: it's O to move, depth=1
    start_state = (start_bit, 0, -1)
    start_ck = canonical_key(cfg, *start_state)

    # curr layer: dict canonical_key -> (state, weight)
    curr: Dict[Tuple[int,int,int], Tuple[Tuple[int,int,int], int]] = { start_ck: (start_state, 1) }

    memo = shared_memo if shared_memo is not None else {}

    avg_px: List[float] = []
    depth = 1
    while curr and depth <= max_depth:
        # collect P_Xwin for each unique canonical state
        values = []
        weights = []
        for ck, (st, w) in curr.items():
            xmask, omask, player = st
            xw, d, ow = count_outcomes(cfg, xmask, omask, player, memo)
            tot = xw + d + ow
            px = (xw / tot) if tot else 0.0
            values.append(px)
            weights.append(w)

        if weighted:
            # weighted average by path multiplicity
            wsum = float(sum(weights))
            avg_px.append(float(np.dot(values, np.array(weights)) / wsum) if wsum > 0 else 0.0)
        else:
            # unweighted average over UNIQUE canonical states
            avg_px.append(float(np.mean(values)) if values else 0.0)

        # expand to next layer
        nxt: Dict[Tuple[int,int,int], Tuple[Tuple[int,int,int], int]] = {}
        for ck, (st, w) in curr.items():
            xmask, omask, player = st
            # stop at terminal states
            if has_win(cfg, xmask) or has_win(cfg, omask) or ((xmask | omask) == cfg.all_mask):
                continue
            for mv in legal_moves(cfg, xmask, omask):
                child = ((xmask | mv, omask, -1) if player == 1 else (xmask, omask | mv, 1))
                ckey = canonical_key(cfg, *child)
                if ckey in nxt:
                    # accumulate multiplicity
                    _, oldw = nxt[ckey]
                    nxt[ckey] = (child, oldw + w)
                else:
                    nxt[ckey] = (child, w)
        curr = nxt
        depth += 1

    return avg_px

# -----------------------------
# Openings selection
# -----------------------------

def parse_series_arg(n: int, s: str) -> List[Tuple[int,int]]:
    """
    Parse --series argument. Supports keywords: corner, edge, center.
    Or coordinates list like "0,0;0,1;1,1".
    """
    s = s.strip()
    if not s:
        return []

    parts = [p.strip() for p in s.split(';') if p.strip()]
    out: List[Tuple[int,int]] = []

    def add_if_in(r: int, c: int):
        if 0 <= r < n and 0 <= c < n:
            out.append((r,c))

    for p in parts:
        low = p.lower()
        if low in ("corner", "corners"):
            add_if_in(0,0)
        elif low in ("edge", "side", "edge-mid", "edge_middle"):
            add_if_in(0, n//2)  # 顶边中点（4×4 时为 (0,1)）
        elif low in ("center", "centre"):
            # 奇数 n：中心 (n//2, n//2)
            # 偶数 n：取左上那个中心格 (n//2 - 1, n//2 - 1)
            r0 = (n//2) if (n % 2 == 1) else (n//2 - 1)
            c0 = r0
            add_if_in(r0, c0)
        else:
            # coordinates "r,c"
            if ',' in p:
                rc = p.split(',')
                try:
                    r = int(rc[0]); c = int(rc[1])
                    add_if_in(r,c)
                except:
                    pass
    # 去重保持顺序
    seen = set()
    uniq = []
    for rc in out:
        if rc not in seen:
            seen.add(rc)
            uniq.append(rc)
    return uniq

# -----------------------------
# Plotting
# -----------------------------

def plot_curves(series: Dict[str, List[float]], n: int, k: int, out_path: str):
    plt.figure(figsize=(8,5))
    # x 轴按 ply（步数，从 1 开始）
    max_len = max((len(v) for v in series.values()), default=0)
    xs = list(range(1, max_len+1))
    styles = {
        "(0,0)": dict(linestyle="--", marker="o", linewidth=2.2, zorder=5),
        "(0,2)": dict(linestyle="-",  marker="s", linewidth=2.0, zorder=4),
        "(1,1)": dict(linestyle="-.", marker="^", linewidth=2.0, zorder=3),
    }

    for name, ys in series.items():
        ys_pad = ys + [np.nan]*(max_len - len(ys))
        kw = styles.get(name, {})
        plt.plot(xs, ys_pad, label=name, alpha=0.95, **kw)

    plt.xlabel("Move number (plies)")
    plt.ylabel("Average P(X win) over unique canonical states")
    plt.title(f"n={n}, k={k}: P(Xwin) vs plies for selected openings")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="board size n (default 4)")
    ap.add_argument("--k", type=int, help="in-a-row to win (default=n)")
    ap.add_argument("--series", type=str, default="corner,edge,center",
                    help="openings to plot; keywords (corner,edge,center) and/or coords 'r,c;...'; default: corner,edge,center")
    ap.add_argument("--max-depth", type=int, default=10, help="max plies to explore (default 10)")
    ap.add_argument("--weighted", action="store_true", help="use path-weighted average instead of unique-state unweighted")
    ap.add_argument("--out", type=str, required=True, help="output PNG path")
    args = ap.parse_args()

    n = args.n
    k = args.k if args.k is not None else n
    cfg = make_config(n, k)

    # 解析开局集合
    openings = parse_series_arg(n, args.series)
    if not openings:
        raise SystemExit("No openings parsed. Use --series like 'corner,edge,center' or '0,0;0,1;1,1'.")

    # 共享 memo（不同开局之间共享置换表可以加速）
    shared_memo: MemoType = {}

    # 逐个开局生成曲线
    series: Dict[str, List[float]] = {}
    for (r,c) in openings:
        label = f"({r},{c})"
        ys = per_depth_avg_px_for_opening(cfg, (r,c), max_depth=args.max_depth, weighted=args.weighted, shared_memo=shared_memo)
        series[label] = ys

    plot_curves(series, n, k, args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
