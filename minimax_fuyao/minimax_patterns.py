#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimax Next-Best-Move Pattern Analysis (TicTacToe, N×N, win length K)

What this script does
---------------------
1) For an empty board, iterate every first move (X1), then compute O's set of
   best replies under perfect play (Minimax). Save as CSV and (optionally)
   draw arrow diagrams: X1 -> {O*}.
2) Compute a short principal variation (PV) of length two plies:
   X1 -> O1 (tie-broken) -> X2 (tie-broken), save as CSV.
3) Build a transition matrix Heatmap where M[X1, O1] = 1/|best_O(X1)|
   if O1 is one of the best replies, else 0.

Usage examples
--------------
# 3×3, K=3 (classic TicTacToe)
python minimax_patterns.py --n 3 --k 3 --outdir out_3x3 --arrows center corner edge --heatmap

# 4×4, K=4 (only "one jump" best reply table; PV is still okay but deeper gets heavy)
python minimax_patterns.py --n 4 --k 4 --outdir out_4x4 --arrows none --heatmap

Notes
-----
- This uses pure Minimax with memoization. For 3×3 it's instant.
  For 4×4 we still only analyze "empty -> first move -> best replies".
- Arrow diagrams are most interpretable for 3×3; for larger N you may
  prefer the transition matrix heatmap.
"""

import argparse, math, functools, os
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- Core Game ---------------------------

@dataclass(frozen=True)
class GameSpec:
    n: int
    k: int  # in-a-row to win


def legal_moves(board: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(i for i, v in enumerate(board) if v == 0)


def apply_move(board: Tuple[int, ...], idx: int, player: int) -> Tuple[int, ...]:
    b = list(board)
    b[idx] = player
    return tuple(b)


def check_winner(board: Tuple[int, ...], spec: GameSpec) -> int:
    """
    Return +1 if X wins, -1 if O wins, 0 otherwise.
    Board encodes cells as: +1 for X, -1 for O, 0 empty.
    """
    n, k = spec.n, spec.k
    B = np.array(board, dtype=int).reshape(n, n)

    # rows & cols
    for i in range(n):
        for s in range(n - k + 1):
            if B[i, s:s + k].sum() == k:   return +1
            if B[i, s:s + k].sum() == -k:  return -1
            if B[s:s + k, i].sum() == k:   return +1
            if B[s:s + k, i].sum() == -k:  return -1

    # diagonals (all offsets that produce length >= k)
    for off in range(-(n - k), n - k + 1):
        d1 = np.diag(B, k=off)
        d2 = np.diag(np.fliplr(B), k=off)
        for line in (d1, d2):
            m = line.size
            if m < k:
                continue
            for s in range(m - k + 1):
                seg = line[s:s + k]
                ssum = seg.sum()
                if ssum == k:   return +1
                if ssum == -k:  return -1

    return 0


# --------------------------- Minimax ---------------------------

@functools.lru_cache(maxsize=None)
def minimax_val(board: Tuple[int, ...], player: int, spec: GameSpec) -> int:
    """
    Perfect-play value from state (board, player):
      returns +1 if X can force a win, -1 if O can force a win, 0 if draw.
    `player` is +1 for X to move, -1 for O to move.
    """
    w = check_winner(board, spec)
    if w != 0:
        return w
    if all(v != 0 for v in board):
        return 0

    best = -2  # worse than loss from current side's perspective
    for mv in legal_moves(board):
        child = apply_move(board, mv, player)
        v = minimax_val(child, -player, spec)   # absolute outcome (+1/0/-1)
        score = player * v                       # maximize current player's outcome
        if score > best:
            best = score
            if best == +1:
                break
    # convert back to absolute outcome
    return player * best


@functools.lru_cache(maxsize=None)
def minimax_with_argmoves(board: Tuple[int, ...], player: int, spec: GameSpec):
    """
    Return (value, best_moves) where:
      value ∈ {+1, 0, -1} is the perfect-play absolute outcome,
      best_moves is the tuple of moves (indices) that maximize current player's outcome.
    """
    w = check_winner(board, spec)
    if w != 0:
        return (w, tuple())
    if all(v != 0 for v in board):
        return (0, tuple())

    scored = []
    for mv in legal_moves(board):
        child = apply_move(board, mv, player)
        v = minimax_val(child, -player, spec)
        scored.append((mv, v))

    best_score = max(player * v for _, v in scored)
    best_moves = tuple(mv for mv, v in scored if player * v == best_score)
    return (player * best_score, best_moves)


# --------------------------- Visualization helpers ---------------------------

def categorize_position(n: int, idx: int) -> str:
    r, c = divmod(idx, n)
    corners = (r in (0, n - 1)) and (c in (0, n - 1))
    edges   = ((r in (0, n - 1)) ^ (c in (0, n - 1))) and not corners
    center  = (n % 2 == 1) and (r == n // 2) and (c == n // 2)
    if center: return "center"
    if corners: return "corner"
    if edges: return "edge"
    return "inner"


def draw_board_with_arrows(n: int, start: int, moves: List[int], path: str, title: str):
    """Draw a grid, mark X at `start`, draw arrows to each move in `moves` (O*)."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    # grid
    for i in range(n + 1):
        ax.plot([0, n], [i, i])
        ax.plot([i, i], [0, n])
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    r0, c0 = divmod(start, n)
    ax.text(c0 + 0.5, r0 + 0.5, "X", ha="center", va="center", fontsize=24)

    for mv in moves:
        r, c = divmod(mv, n)
        ax.annotate(
            "", xy=(c + 0.5, r + 0.5), xytext=(c0 + 0.5, r0 + 0.5),
            arrowprops=dict(arrowstyle="->", lw=2)
        )
        ax.text(c + 0.5, r + 0.5, "O*", ha="center", va="center", fontsize=16)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_transition_heatmap(mat: np.ndarray, n: int, out_png: str, title: str):
    """
    Plot an n^2 × n^2 transition matrix heatmap where row = X1 move, col = O best reply,
    entries sum to 1 across each row (if there are best replies).
    """
    fig = plt.figure(figsize=(6, 5), dpi=160)
    ax = plt.gca()
    im = ax.imshow(mat, origin='upper')
    ax.set_title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xlabel("O's best reply (cell index)")
    ax.set_ylabel("X first move (cell index)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    return out_png


# --------------------------- Analysis ---------------------------

def analyze_next_best_moves(n: int, k: int, outdir: str,
                            arrows: List[str],
                            make_heatmap: bool,
                            pv_depth: int = 2):
    """
    Compute:
      - best-reply table for each first move
      - arrow diagrams for selected first-move categories
      - PV table for X1 -> O1 -> X2 (pv_depth=2)
      - transition matrix heatmap
    """
    os.makedirs(outdir, exist_ok=True)
    spec = GameSpec(n, k)
    empty = tuple([0] * (n * n))

    # 1) Best replies for each first move
    rows = []
    best_reply_map: Dict[int, Tuple[int, ...]] = {}
    for mv in range(n * n):
        child = apply_move(empty, mv, +1)         # X plays first
        val_after_O, best_O = minimax_with_argmoves(child, -1, spec)
        best_reply_map[mv] = best_O
        rows.append({
            "first_move": mv,
            "first_move_row": mv // n,
            "first_move_col": mv % n,
            "first_move_type": categorize_position(n, mv),
            "num_best_O": len(best_O),
            "best_O_moves": list(best_O),
            "value_after_O_best": val_after_O  # absolute outcome with perfect play continuing
        })

    df_best = pd.DataFrame(rows).sort_values(["first_move"]).reset_index(drop=True)
    best_csv = os.path.join(outdir, f"minimax_{n}x{n}_best_replies.csv")
    df_best.to_csv(best_csv, index=False)

    # 2) Arrow diagrams (for interpretability, default to a few canonical starts)
    targets = []
    arrows = [a.lower() for a in arrows]
    if "all" in arrows:
        targets = list(range(n * n))
    else:
        # choose one representative for each requested category
        idxs = list(range(n * n))
        if "center" in arrows:
            targets += [i for i in idxs if categorize_position(n, i) == "center"]
        if "corner" in arrows:
            # pick one corner (top-left) unless 'all' is requested
            corners = [i for i in idxs if categorize_position(n, i) == "corner"]
            targets += ([corners[0]] if corners else [])
        if "edge" in arrows:
            edges = [i for i in idxs if categorize_position(n, i) == "edge"]
            targets += ([edges[0]] if edges else [])
        if "inner" in arrows:
            inners = [i for i in idxs if categorize_position(n, i) == "inner"]
            targets += ([inners[0]] if inners else [])
    # de-dup while preserving order
    seen = set()
    targets = [x for x in targets if not (x in seen or seen.add(x))]

    arrow_pngs = []
    for mv in targets:
        moves = list(best_reply_map[mv])
        title = f"{n}×{n}: After X plays {mv} (type={categorize_position(n, mv)}), O's Best Replies"
        out_png = os.path.join(outdir, f"arrows_{n}x{n}_start_{mv}.png")
        arrow_pngs.append(draw_board_with_arrows(n, mv, moves, out_png, title))

    # 3) Principal Variation (PV) for first two plies: X1 -> O1 -> X2
    pv_rows = []
    for mv in range(n * n):
        child = apply_move(empty, mv, +1)
        vO, O_moves = minimax_with_argmoves(child, -1, spec)
        if not O_moves:
            pv_rows.append({"X1": mv, "O1": None, "X2": None, "value_after_X2": vO})
            continue
        # deterministic tie-break: lowest cell index
        o1 = min(O_moves)
        s2 = apply_move(child, o1, -1)
        vX, X_moves = minimax_with_argmoves(s2, +1, spec)
        x2 = min(X_moves) if X_moves else None
        pv_rows.append({"X1": mv, "O1": o1, "X2": x2, "value_after_X2": vX})
    df_pv = pd.DataFrame(pv_rows).sort_values("X1").reset_index(drop=True)
    pv_csv = os.path.join(outdir, f"minimax_{n}x{n}_PV_first_two_plies.csv")
    df_pv.to_csv(pv_csv, index=False)

    # 4) Transition matrix heatmap (row=X1, col=O1)
    heatmap_png = None
    if make_heatmap:
        size = n * n
        M = np.zeros((size, size), dtype=float)
        for mv in range(size):
            best = best_reply_map[mv]
            if len(best) == 0:
                continue
            p = 1.0 / len(best)
            for r in best:
                M[mv, r] = p
        heatmap_png = os.path.join(outdir, f"minimax_{n}x{n}_best_reply_transition.png")
        plot_transition_heatmap(M, n, heatmap_png, f"{n}×{n} Minimax Best-Reply Transition")

    # Print summary paths
    print("Wrote:", best_csv)
    print("Wrote:", pv_csv)
    if heatmap_png:
        print("Wrote:", heatmap_png)
    for p in arrow_pngs:
        print("Wrote:", p)


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Minimax next-best-move pattern analysis for TicTacToe (N×N).")
    ap.add_argument("--n", type=int, default=3, help="board dimension (default 3)")
    ap.add_argument("--k", type=int, default=None, help="win length (default = n)")
    ap.add_argument("--outdir", type=str, default="out_patterns", help="output directory")
    ap.add_argument("--arrows", nargs="*", default=["center", "corner", "edge"],
                    help="which arrow diagrams to draw: any of {center, corner, edge, inner, all, none}")
    ap.add_argument("--heatmap", action="store_true", help="draw best-reply transition heatmap")
    args = ap.parse_args()

    n = args.n
    k = args.k if args.k is not None else n
    arrows = [] if ("none" in [a.lower() for a in args.arrows]) else args.arrows

    analyze_next_best_moves(n=n, k=k, outdir=args.outdir, arrows=arrows,
                            make_heatmap=args.heatmap, pv_depth=2)


if __name__ == "__main__":
    main()
