#!/usr/bin/env python3
# Analyze 3x3 TicTacToe (k=3) canonical layouts:
# Method 1 (Grouping): rule-based groups -> best move frequency & pattern summaries
# Method 2 (Overlay): stack per-move Pwin -> heatmaps of best-move frequency and mean Pwin
#
# Input: the JSON exported by ttt_3x3_all_layouts.py (layouts_index.json)
# Output: a folder with PNG heatmaps and JSON/CSV summaries.

import json, os, argparse, math, csv
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

N = 3

# --------------------------
# Utilities
# --------------------------

def to_tuple_board(board_list: List[List[int]]) -> Tuple[Tuple[int,...], ...]:
    return tuple(tuple(row) for row in board_list)

def count_marks(board: List[List[int]]) -> Tuple[int,int,int]:
    # returns (numX, numO, numEmpty)
    x = sum(1 for r in range(N) for c in range(N) if board[r][c] == 1)
    o = sum(1 for r in range(N) for c in range(N) if board[r][c] == -1)
    e = N*N - x - o
    return x,o,e

def two_in_row_threat(board: List[List[int]], player: int) -> int:
    """
    Count the number of 'two-in-a-row with one empty' lines for given player (1=X, -1=O).
    A simple proxy for 'immediate threat' patterns.
    """
    lines = []
    # rows
    for r in range(N):
        lines.append([(r,0),(r,1),(r,2)])
    # cols
    for c in range(N):
        lines.append([(0,c),(1,c),(2,c)])
    # diags
    lines.append([(0,0),(1,1),(2,2)])
    lines.append([(0,2),(1,1),(2,0)])
    cnt = 0
    for line in lines:
        vals = [board[r][c] for (r,c) in line]
        if vals.count(player) == 2 and vals.count(0) == 1 and vals.count(-player) == 0:
            cnt += 1
    return cnt

def center_occ(board: List[List[int]]) -> int:
    return board[1][1]

def corner_count(board: List[List[int]], player: int) -> int:
    corners = [(0,0),(0,2),(2,0),(2,2)]
    return sum(1 for (r,c) in corners if board[r][c] == player)

def edge_count(board: List[List[int]], player: int) -> int:
    edges = [(0,1),(1,0),(1,2),(2,1)]
    return sum(1 for (r,c) in edges if board[r][c] == player)

def plot_heatmap(mat: np.ndarray, title: str, path: str, vmin=None, vmax=None, annotate=True):
    plt.figure(figsize=(5,5))
    plt.imshow(mat, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    for r in range(N):
        for c in range(N):
            val = mat[r,c]
            if annotate:
                if np.isnan(val):
                    txt = "—"
                else:
                    txt = f"{val:.2f}"
                plt.text(c, r, txt, ha="center", va="center")
    plt.xticks(range(N)); plt.yticks(range(N))
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# --------------------------
# Grouping rules (Method 1)
# --------------------------

def layout_group_key(entry: Dict[str, Any]) -> Tuple:
    """
    Define a coarse grouping key for 'layouts with common traits':
    - to_move: 'X' or 'O'
    - ply bucket: 0-1, 2-3, 4-5, 6-7, 8
    - center status: -1/O, 0/empty, 1/X
    - threat signatures: (x_threats, o_threats)
    - counts: (x_corners, o_corners, x_edges, o_edges)
    """
    to_move = entry["to_move"]
    ply = entry["ply"]
    if   ply <= 1: ply_bucket = "0-1"
    elif ply <= 3: ply_bucket = "2-3"
    elif ply <= 5: ply_bucket = "4-5"
    elif ply <= 7: ply_bucket = "6-7"
    else:          ply_bucket = "8"

    board = entry["board"]
    x_th = two_in_row_threat(board, 1)
    o_th = two_in_row_threat(board, -1)
    ctr  = center_occ(board)
    xco  = corner_count(board, 1)
    oco  = corner_count(board, -1)
    xed  = edge_count(board, 1)
    oed  = edge_count(board, -1)
    return (to_move, ply_bucket, ctr, x_th, o_th, xco, oco, xed, oed)

def summarize_group(entries: List[Dict[str,Any]], outdir: str, tag: str):
    """
    For a group of layouts, produce:
    - best move frequency heatmap (each cell = fraction of layouts where it's among best moves)
    - average best-move Pwin
    - tie rate of best moves (how often multiple best)
    - overlay of mean Pwin for all available moves (not just best)
    """
    if not entries:
        return None

    freq = np.zeros((N,N), dtype=float)
    denom = 0
    best_pwins = []
    tie_counts = 0

    # Overlay: mean Pwin for all available moves
    sum_p = np.zeros((N,N), dtype=float)
    cnt_p = np.zeros((N,N), dtype=float)

    for e in entries:
        bm = e.get("best_moves", [])
        pm = e.get("per_move", {})
        # collect best moves
        if bm:
            denom += 1
            if len(bm) > 1:
                tie_counts += 1
            # best move win value
            bvals = []
            for (r,c) in bm:
                key = f"{r},{c}"
                bvals.append(pm[key]["p_current_player_win"])
            if bvals:
                best_pwins.append(float(np.mean(bvals)))
            # bump freq for all best cells
            for (r,c) in bm:
                freq[r,c] += 1.0

        # overlay all available moves
        for k, stat in pm.items():
            r,c = map(int, k.split(","))
            p = float(stat["p_current_player_win"])
            sum_p[r,c] += p
            cnt_p[r,c] += 1.0

    if denom > 0:
        freq /= denom

    mean_p = np.full((N,N), np.nan, dtype=float)
    for r in range(N):
        for c in range(N):
            if cnt_p[r,c] > 0:
                mean_p[r,c] = sum_p[r,c] / cnt_p[r,c]

    # save images
    os.makedirs(outdir, exist_ok=True)
    plot_heatmap(freq, f"[{tag}] Best-move frequency (fraction)", os.path.join(outdir, f"{tag}_bestmove_freq.png"), vmin=0.0, vmax=1.0)
    plot_heatmap(mean_p, f"[{tag}] Mean P(win) over all available moves", os.path.join(outdir, f"{tag}_mean_pwin.png"), vmin=0.0, vmax=1.0)

    # summary json
    summary = {
        "tag": tag,
        "num_layouts": len(entries),
        "bestmove_freq_matrix": freq.tolist(),
        "mean_pwin_matrix": mean_p.tolist(),
        "avg_bestmove_pwin": float(np.mean(best_pwins)) if best_pwins else None,
        "tie_rate_among_best": float(tie_counts/denom) if denom>0 else None
    }
    with open(os.path.join(outdir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

# --------------------------
# Overlay method (Method 2)
# --------------------------

def overlay_all(entries: List[Dict[str,Any]], outdir: str, tag: str, filter_to_move: Optional[str]=None, ply_bucket: Optional[str]=None):
    """
    Build overlays across (optionally filtered) layouts:
    - best move frequency per cell
    - mean Pwin per cell over all available moves
    """
    subset = []
    for e in entries:
        if filter_to_move and e["to_move"] != filter_to_move:
            continue
        if ply_bucket:
            ply = e["ply"]
            if   ply <= 1: b = "0-1"
            elif ply <= 3: b = "2-3"
            elif ply <= 5: b = "4-5"
            elif ply <= 7: b = "6-7"
            else:          b = "8"
            if b != ply_bucket:
                continue
        subset.append(e)

    return summarize_group(subset, outdir, tag)

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="layouts_index.json produced by ttt_3x3_all_layouts.py")
    ap.add_argument("--outdir", type=str, required=True, help="output directory for analysis")
    ap.add_argument("--max_groups", type=int, default=12, help="dump up to this many groups (most populous)")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    layouts: List[Dict[str,Any]] = raw["layouts"]

    # Filter non-terminal (where per_move exists)
    non_terminal = [e for e in layouts if e.get("per_move")]

    # ------------- Method 1: Group by rule-based key -------------
    groups: Dict[Tuple, List[Dict[str,Any]]] = {}
    for e in non_terminal:
        k = layout_group_key(e)
        groups.setdefault(k, []).append(e)

    # sort groups by size
    group_items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    os.makedirs(args.outdir, exist_ok=True)

    # write a CSV index of groups
    with open(os.path.join(args.outdir, "groups_index.csv"), "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["rank","size","to_move","ply_bucket","center","x_threats","o_threats","x_corners","o_corners","x_edges","o_edges"])
        for rank,(gk, glist) in enumerate(group_items, start=1):
            w.writerow([rank, len(glist), *gk])

    # summarize top-K groups
    for rank,(gk, glist) in enumerate(group_items[:args.max_groups], start=1):
        tag = f"group{rank}_tm{gk[0]}_ply{gk[1]}_ctr{gk[2]}_xT{gk[3]}_oT{gk[4]}"
        summarize_group(glist, args.outdir, tag)

    # ------------- Method 2: Overlays across global/subsets -------------
    # Global overlay (all non-terminal)
    overlay_all(non_terminal, args.outdir, tag="ALL_non_terminal")

    # Separate overlays by to_move
    overlay_all(non_terminal, args.outdir, tag="X_to_move", filter_to_move="X")
    overlay_all(non_terminal, args.outdir, tag="O_to_move", filter_to_move="O")

    # By ply buckets
    for b in ["0-1","2-3","4-5","6-7","8"]:
        overlay_all(non_terminal, args.outdir, tag=f"PLY_{b}", ply_bucket=b)

    # Also dump a slim JSON with just (canonical key -> best moves), handy for quick lookups
    slim = {}
    for e in non_terminal:
        ck = e["canonical_key"]
        key = f"{ck['xmask']}-{ck['omask']}-{ck['player']}"
        slim[key] = {
            "to_move": e["to_move"],
            "ply": e["ply"],
            "best_moves": e.get("best_moves", []),
        }
    with open(os.path.join(args.outdir, "best_moves_index_slim.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)

    print("Done. See:", args.outdir)

if __name__ == "__main__":
    main()
