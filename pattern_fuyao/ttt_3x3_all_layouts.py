#!/usr/bin/env python3
# 3x3 TicTacToe (k=3) — Export ALL unique canonical layouts:
# - Exhaustive enumeration (NO minimax) from empty board
# - D4 symmetry dedup (8 transforms) to get UNIQUE layouts
# - For each NON-terminal layout, compute per-empty-cell:
#       P_win(current_player | move at cell)  under exhaustive uniform continuation
# - Save one PNG per layout with X/O placed and per-cell win-rates overlaid
# - Save a JSON mapping layout -> best next move(s) and per-cell stats
#
# Usage:
#   python ttt_3x3_all_layouts.py --outdir out_n3k3_layouts --json layouts_index.json
#
# Notes:
# - 完全穷举 + 记忆化（置换表）用于复用子树计数；结果是“全路径意义”的真实胜率。
# - 终局布局也会入库（图只显示棋面；无可落子与胜率文本），JSON中 best_moves 为空。

import os, json, argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 3x3 bitboard + D4 utilities
# -----------------------------

N = 3
ALL_MASK = (1 << (N*N)) - 1

def idx(r:int, c:int) -> int:
    return r * N + c

# 3-in-a-row masks
WIN_MASKS = [
    # rows
    (1<<idx(0,0)) | (1<<idx(0,1)) | (1<<idx(0,2)),
    (1<<idx(1,0)) | (1<<idx(1,1)) | (1<<idx(1,2)),
    (1<<idx(2,0)) | (1<<idx(2,1)) | (1<<idx(2,2)),
    # cols
    (1<<idx(0,0)) | (1<<idx(1,0)) | (1<<idx(2,0)),
    (1<<idx(0,1)) | (1<<idx(1,1)) | (1<<idx(2,1)),
    (1<<idx(0,2)) | (1<<idx(1,2)) | (1<<idx(2,2)),
    # diags
    (1<<idx(0,0)) | (1<<idx(1,1)) | (1<<idx(2,2)),
    (1<<idx(0,2)) | (1<<idx(1,1)) | (1<<idx(2,0)),
]

def has_win(mask:int) -> bool:
    for m in WIN_MASKS:
        if (mask & m) == m:
            return True
    return False

def legal_moves(xmask:int, omask:int) -> List[int]:
    empties = ~(xmask | omask) & ALL_MASK
    moves = []
    m = empties
    while m:
        lsb = m & -m
        moves.append(lsb)
        m ^= lsb
    return moves

# D4 transforms (index maps)
def rot90_map():   return [ idx(c, 2-r)   for r in range(N) for c in range(N) ]
def rot180_map():  return [ idx(2-r,2-c)  for r in range(N) for c in range(N) ]
def rot270_map():  return [ idx(2-c, r)   for r in range(N) for c in range(N) ]
def flip_h_map():  return [ idx(r, 2-c)   for r in range(N) for c in range(N) ]
def flip_v_map():  return [ idx(2-r, c)   for r in range(N) for c in range(N) ]
def flip_d_map():  return [ idx(c, r)     for r in range(N) for c in range(N) ]  # main diag
def flip_a_map():  return [ idx(2-c,2-r)  for r in range(N) for c in range(N) ]  # anti diag

IDX_MAPS = [
    [idx(r,c) for r in range(N) for c in range(N)],
    rot90_map(), rot180_map(), rot270_map(),
    flip_h_map(), flip_v_map(), flip_d_map(), flip_a_map()
]

def transform_mask(mask:int, mapping:List[int]) -> int:
    new_mask = 0
    m = mask
    while m:
        lsb = m & -m
        p = (lsb.bit_length() - 1)
        new_mask |= (1 << mapping[p])
        m ^= lsb
    return new_mask

def canonical_pair(xmask:int, omask:int) -> Tuple[int,int]:
    """Return lexicographically minimal (x',o') among 8 D4 transforms."""
    best = None
    for mp in IDX_MAPS:
        x2 = transform_mask(xmask, mp)
        o2 = transform_mask(omask, mp)
        pair = (x2, o2)
        if best is None or pair < best:
            best = pair
    return best

def canonical_key(xmask:int, omask:int, player:int) -> Tuple[int,int,int]:
    x2,o2 = canonical_pair(xmask, omask)
    return (x2, o2, player)

def popcount(x:int) -> int:
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")

def player_to_move_from_masks(xmask:int, omask:int) -> int:
    # X moves if counts equal; O moves if X has exactly one more
    px, po = popcount(xmask), popcount(omask)
    return 1 if px == po else -1

# -----------------------------
# Exhaustive counting (no minimax) with memo
# -----------------------------

# memo[(canon_x, canon_o, player)] = (X_wins, Draws, O_wins)
MemoType = Dict[Tuple[int,int,int], Tuple[int,int,int]]

def count_outcomes(xmask:int, omask:int, player:int, memo:MemoType) -> Tuple[int,int,int]:
    """Exhaustively count (X wins, Draws, O wins) from this state."""
    if has_win(xmask): return (1,0,0)
    if has_win(omask): return (0,0,1)
    if (xmask | omask) == ALL_MASK: return (0,1,0)

    key = canonical_key(xmask, omask, player)
    if key in memo:
        return memo[key]

    tx = td = to = 0
    for mv in legal_moves(xmask, omask):
        if player == 1:   # X to move
            xw, d, ow = count_outcomes(xmask | mv, omask, -1, memo)
        else:             # O to move
            xw, d, ow = count_outcomes(xmask, omask | mv, 1, memo)
        tx += xw; td += d; to += ow

    memo[key] = (tx, td, to)
    return memo[key]

# -----------------------------
# Enumerate ALL unique layouts by BFS with D4 dedup
# -----------------------------

def enumerate_unique_layouts() -> Dict[Tuple[int,int,int], Tuple[int,int,int]]:
    """
    Returns a dict:
      canon_key -> representative_state (xmask, omask, player)
    starting from empty board BFS, dedup by canonical key.
    """
    start = (0, 0, 1)  # empty, X to move
    start_key = canonical_key(*start)

    seen: Dict[Tuple[int,int,int], Tuple[int,int,int]] = { start_key: start }
    frontier = [start]

    while frontier:
        new_frontier = []
        for (xmask, omask, player) in frontier:
            # do not expand terminal
            if has_win(xmask) or has_win(omask) or ((xmask|omask) == ALL_MASK):
                continue
            for mv in legal_moves(xmask, omask):
                child = ((xmask | mv, omask, -1) if player == 1 else (xmask, omask | mv, 1))
                ck = canonical_key(*child)
                if ck not in seen:
                    seen[ck] = child
                    new_frontier.append(child)
        frontier = new_frontier

    return seen  # includes terminal and non-terminal unique layouts

# -----------------------------
# Rendering
# -----------------------------

def board_arrays_from_masks(xmask:int, omask:int) -> np.ndarray:
    """Return a 3x3 array with 1 for X, -1 for O, 0 for empty."""
    A = np.zeros((N,N), dtype=int)
    for p in range(N*N):
        r,c = divmod(p, N)
        bit = 1 << p
        if xmask & bit: A[r,c] = 1
        elif omask & bit: A[r,c] = -1
    return A

def render_layout_png(path:str, xmask:int, omask:int, player:int,
                      move_pwin: Dict[Tuple[int,int], float]):
    """
    Draw a 3x3 board:
      - Place X/O for occupied cells
      - For each legal empty cell, overlay the current player's win-rate (0..1)
    """
    A = board_arrays_from_masks(xmask, omask)

    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax.set_xlim(-0.5, N-0.5); ax.set_ylim(-0.5, N-0.5)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.invert_yaxis()  # so (0,0) at top-left visually

    # optional: background heat for available moves (mask others as NaN)
    heat = np.full((N,N), np.nan, dtype=float)
    for (r,c), p in move_pwin.items():
        heat[r,c] = p
    im = ax.imshow(heat, vmin=0.0, vmax=1.0, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"P({'X' if player==1 else 'O'} win | move)")

    # grid lines
    for t in range(N+1):
        ax.axhline(t-0.5, lw=1, color='black')
        ax.axvline(t-0.5, lw=1, color='black')

    # draw X/O markers
    for r in range(N):
        for c in range(N):
            if A[r,c] == 1:
                ax.text(c, r, 'X', ha='center', va='center', fontsize=28, fontweight='bold')
            elif A[r,c] == -1:
                ax.text(c, r, 'O', ha='center', va='center', fontsize=28, fontweight='bold')

    # overlay percentages for legal empties
    for (r,c), p in move_pwin.items():
        ax.text(c, r, f"{p*100:.0f}%", ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    ax.set_title(f"To move: {'X' if player==1 else 'O'}")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main export
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="directory to save per-layout PNGs")
    ap.add_argument("--json", type=str, required=True, help="output JSON file (index of layouts and best moves)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Enumerate all UNIQUE layouts
    unique_layouts = enumerate_unique_layouts()
    print(f"Unique canonical layouts (including terminal): {len(unique_layouts)}")

    # 2) Prepare memo for exhaustive counts
    memo: MemoType = {}
    # cache for child state counts
    counts_cache: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}

    # 3) Iterate layouts, compute per-move win rates and best next move(s)
    index = []  # list of dicts to be saved in JSON

    # For stable order, sort by (num moves, xmask, omask, player)
    def key_sort(item):
        (cx, co, pl), (xmask, omask, player) = item
        return (popcount(xmask)+popcount(omask), cx, co, 0 if player==1 else 1)

    for canon_key, state in sorted(unique_layouts.items(), key=key_sort):
        xmask, omask, player = state

        # Determine legal moves
        moves = legal_moves(xmask, omask)

        # Per-cell probability map for current player
        move_pwin: Dict[Tuple[int,int], float] = {}
        move_stats: Dict[str, Dict[str, int]] = {}  # "r,c" -> {xw,d,ow,total,win_for_current}

        if moves and not (has_win(xmask) or has_win(omask)):
            # compute for each move
            best_val = -1.0
            for mv in moves:
                p = (mv.bit_length() - 1)
                r, c = divmod(p, N)
                child = ((xmask | mv, omask, -1) if player == 1 else (xmask, omask | mv, 1))
                ckey_canon = canonical_key(*child)
                if ckey_canon in counts_cache:
                    xw, d, ow = counts_cache[ckey_canon]
                else:
                    xw, d, ow = count_outcomes(*child, memo)
                    counts_cache[ckey_canon] = (xw, d, ow)
                tot = xw + d + ow
                if tot == 0:
                    pwin = 0.0
                    win_num = 0
                else:
                    win_num = xw if player == 1 else ow
                    pwin = win_num / tot

                move_pwin[(r,c)] = pwin
                move_stats[f"{r},{c}"] = {
                    "x_wins": int(xw), "draws": int(d), "o_wins": int(ow),
                    "total": int(tot),
                    "current_player_win": int(win_num),
                    "p_current_player_win": float(pwin)
                }
                if pwin > best_val:
                    best_val = pwin

            # best moves (handle ties)
            best_moves = [ [r,c] for (r,c),pv in move_pwin.items() if abs(pv - best_val) < 1e-12 ]
        else:
            best_moves = []  # terminal or no legal moves

        # 4) Save image
        fname = f"layout_moves_{popcount(xmask)+popcount(omask):02d}ply_{canon_key[0]:05d}_{canon_key[1]:05d}_{'X' if player==1 else 'O'}.png"
        # Shorter: use sequential numbering to avoid huge names
        # but including ply in filename helps browsing.
        img_path = os.path.join(args.outdir, fname)
        render_layout_png(img_path, xmask, omask, player, move_pwin)

        # 5) Append JSON entry
        board_arr = [[0]*N for _ in range(N)]
        for p in range(N*N):
            r,c = divmod(p, N)
            bit = 1 << p
            board_arr[r][c] = 1 if (xmask & bit) else (-1 if (omask & bit) else 0)

        index.append({
            "canonical_key": {"xmask": canon_key[0], "omask": canon_key[1], "player": ("X" if player==1 else "O")},
            "ply": popcount(xmask) + popcount(omask),
            "board": board_arr,
            "to_move": "X" if player==1 else "O",
            "best_moves": best_moves,  # list of [r,c]
            "per_move": move_stats,    # dict keyed by "r,c"
            "image": os.path.basename(img_path)
        })

    # 6) Save JSON
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({
            "n": 3, "k": 3,
            "total_unique_layouts": len(unique_layouts),
            "note": "Layouts are unique under D4 symmetry; probabilities are exhaustive full-path win-rates for the current player.",
            "layouts": index
        }, f, ensure_ascii=False, indent=2)

    print(f"Done. PNGs in: {args.outdir}")
    print(f"Index JSON: {args.json}")

if __name__ == "__main__":
    main()
