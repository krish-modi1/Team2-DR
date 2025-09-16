#!/usr/bin/env python3
"""
Minimax + Random-Play Heatmaps for N×N TicTacToe
Usage:
  python minimax_tictactoe.py --n 3 --k 3 --mc 0
  python minimax_tictactoe.py --n 4 --k 4 --mc 800
Outputs CSV + PNG heatmaps into current directory.
"""
import argparse, math, functools
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class GameSpec:
    n: int
    k: int

def check_winner(board: Tuple[int,...], spec: GameSpec) -> int:
    n, k = spec.n, spec.k
    B = np.array(board).reshape(n,n)
    # rows/cols
    for i in range(n):
        for s in range(n-k+1):
            if B[i, s:s+k].sum() == k: return +1
            if B[i, s:s+k].sum() == -k: return -1
            if B[s:s+k, i].sum() == k: return +1
            if B[s:s+k, i].sum() == -k: return -1
    # diagonals
    for off in range(-(n-k), n-k+1):
        d1 = np.diag(B, k=off)
        d2 = np.diag(np.fliplr(B), k=off)
        for line in (d1, d2):
            m = line.size
            if m < k: continue
            for s in range(m-k+1):
                seg = line[s:s+k]
                ssum = seg.sum()
                if ssum == k: return +1
                if ssum == -k: return -1
    return 0

def legal_moves(board): 
    return tuple(i for i,v in enumerate(board) if v==0)

def apply_move(board, idx, player):
    b = list(board); b[idx]=player; return tuple(b)

@functools.lru_cache(maxsize=None)
def minimax_value(board: Tuple[int,...], player: int, spec: GameSpec) -> int:
    w = check_winner(board, spec)
    if w!=0: return w
    if all(v!=0 for v in board): return 0
    best = -2
    for mv in legal_moves(board):
        child = apply_move(board, mv, player)
        val = minimax_value(child, -player, spec)
        score = player * val
        if score > best:
            best = score
            if best == +1: break
    return player * best

@functools.lru_cache(maxsize=None)
def random_play_counts(board: Tuple[int,...], player: int, spec: GameSpec):
    """Exact enumeration of random-play outcomes (only feasible for small boards)."""
    w = check_winner(board, spec)
    if w!=0: return (1,0,0) if w==+1 else (0,1,0)
    if all(v!=0 for v in board): return (0,0,1)
    total_x=total_o=total_t=0
    for mv in legal_moves(board):
        child = apply_move(board, mv, player)
        cx,co,ct = random_play_counts(child, -player, spec)
        total_x += cx; total_o += co; total_t += ct
    return (total_x, total_o, total_t)

def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, outfile: str):
    n = int(math.sqrt(len(df)))
    mat = df.sort_values(["row","col"])[value_col].to_numpy().reshape(n,n)
    plt.figure(figsize=(3.6,3.2), dpi=160)
    im = plt.imshow(mat, origin='upper')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(n), [str(i) for i in range(n)])
    plt.yticks(range(n), [str(i) for i in range(n)])
    for r in range(n):
        for c in range(n):
            val = mat[r,c]
            if np.isnan(val):
                s = "NaN"
            else:
                s = f"{val:.2f}"
            plt.text(c, r, s, ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return outfile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=None, help="win length (default = n)")
    ap.add_argument("--mc", type=int, default=0, help="Monte Carlo sims per move (0 = exact random-play if feasible)")
    args = ap.parse_args()
    n = args.n
    k = args.k if args.k is not None else n
    spec = GameSpec(n,k)
    empty = tuple([0]*(n*n))

    rows=[]
    for mv in range(n*n):
        child = apply_move(empty, mv, +1)
        # minimax (may be expensive for larger n)
        try:
            v = minimax_value(child, -1, spec)
        except RecursionError:
            v = float("nan")
        # random-play probabilities
        if args.mc == 0 and n <= 3:
            cx,co,ct = random_play_counts(child, -1, spec)
            tot = cx+co+ct
            wx, wo, wt = cx/tot, co/tot, ct/tot
        else:
            rng = np.random.default_rng(123+mv)
            x=o=t=0
            trials = max(1, args.mc)
            for _ in range(trials):
                state = child; pl = -1
                while True:
                    w = check_winner(state, spec)
                    if w!=0:
                        if w==+1: x+=1
                        else: o+=1
                        break
                    ms = legal_moves(state)
                    if not ms: t+=1; break
                    m2 = rng.choice(ms)
                    state = apply_move(state, m2, pl)
                    pl = -pl
            tot = x+o+t
            wx,wo,wt = x/tot, o/tot, t/tot
        rows.append({
            "n":n,"k":k,"move_index":mv,"row":mv//n,"col":mv%n,
            "minimax_value": v,
            "rand_X_win_prob": wx, "rand_O_win_prob": wo, "rand_tie_prob": wt
        })
    df = pd.DataFrame(rows).sort_values(["row","col"]).reset_index(drop=True)
    csv_path = f"./tictactoe_{n}x{n}_first_move_stats.csv"
    df.to_csv(csv_path, index=False)
    plot_heatmap(df, "rand_X_win_prob", f"{n}x{n}: X Win Probability", f"./heatmap_{n}x{n}_random_Xwin.png")
    plot_heatmap(df, "minimax_value",  f"{n}x{n}: Minimax Value", f"./heatmap_{n}x{n}_minimax.png")
    print("Wrote:", csv_path)
    print("Wrote:", f"./heatmap_{n}x{n}_random_Xwin.png")
    print("Wrote:", f"./heatmap_{n}x{n}_minimax.png")

if __name__ == "__main__":
    main()
