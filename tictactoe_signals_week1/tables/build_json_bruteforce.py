#!/usr/bin/env python3
"""
build_json_bruteforce.py
Generate P_win/P_draw JSON by brute force (no symmetry reduction).
Definition:
- For each ply p and role r ∈ {X,O}, we average over ALL legal boards with exactly p moves,
  where it's r's turn to play (X moves at even p).
- For each empty cell c in those boards, we compute the probability that r eventually
  WINS or DRAWS if r first plays at c and thereafter BOTH players pick uniformly
  at random among legal moves until terminal.
- We then average these probabilities over the enumerated boards, ignoring boards
  where cell c is not empty.
- Output schema matches build_tables.py loader:
{
  "win":  {"X": {"ply_0": [[...NxN...], ...], "ply_1": ...}, "O": {...}},
  "draw": {"X": {...}, "O": {...}}
}

This is EXPENSIVE for large N/p. Practical for 3x3 (full plies). For 4x4, keep p small.
"""

import argparse, json, math, itertools, functools, time
from collections import defaultdict

X, O, E = 1, 2, 0

def all_win_lines(N:int, K:int):
    """All K-in-a-row winning line index lists on an NxN board."""
    lines = []
    # rows
    for r in range(N):
        for c in range(N-K+1):
            lines.append([r*N + (c+i) for i in range(K)])
    # cols
    for c in range(N):
        for r in range(N-K+1):
            lines.append([(r+i)*N + c for i in range(K)])
    # diag down-right
    for r in range(N-K+1):
        for c in range(N-K+1):
            lines.append([(r+i)*N + (c+i) for i in range(K)])
    # diag up-right
    for r in range(K-1, N):
        for c in range(N-K+1):
            lines.append([(r-i)*N + (c+i) for i in range(K)])
    # Deduplicate (can happen for K=N on 3x3 corners overlap? keep unique)
    uniq = []
    seen = set()
    for L in lines:
        t = tuple(L)
        if t not in seen:
            seen.add(t); uniq.append(L)
    return uniq

def winner(board, lines):
    """Return X or O if someone has K-in-a-row; return None if not terminal; return 'draw' if full with no winner."""
    # check wins
    for L in lines:
        vals = {board[i] for i in L}
        if len(vals)==1 and next(iter(vals)) in (X,O):
            return next(iter(vals))
    # full -> draw
    if all(v!=E for v in board):
        return 'draw'
    return None

def legal_moves(board):
    return [i for i,v in enumerate(board) if v==E]

def next_player(turn):
    return O if turn==X else X

def evaluate_random_uniform(board, turn, lines, root_player, cache):
    """
    Return (p_win, p_draw) for ROOT player's perspective, assuming from this node onward
    BOTH sides choose uniformly at random among legal moves until terminal.
    This is exact enumeration, not Monte Carlo.
    """
    key = (tuple(board), turn, root_player)
    if key in cache:
        return cache[key]

    term = winner(board, lines)
    if term is not None:
        if term == 'draw':
            res = (0.0, 1.0)
        else:
            res = (1.0, 0.0) if term == root_player else (0.0, 0.0)
        cache[key] = res
        return res

    moves = legal_moves(board)
    if not moves:
        # no moves but not caught by winner -> draw safeguard
        res = (0.0, 1.0)
        cache[key] = res
        return res

    # Uniform over moves
    p_acc_win, p_acc_draw = 0.0, 0.0
    prob_each = 1.0/len(moves)
    for m in moves:
        board[m] = turn
        w, d = evaluate_random_uniform(board, next_player(turn), lines, root_player, cache)
        board[m] = E
        p_acc_win += prob_each * w
        p_acc_draw += prob_each * d
    cache[key] = (p_acc_win, p_acc_draw)
    return cache[key]

def enumerate_boards_exact_ply(N:int, p:int, lines):
    """
    Generate all legal boards with exactly p moves, no prior winner before p (i.e., build forward ensuring legality).
    X starts; so turn at ply p is X if p even else O.
    """
    boards = []
    init = [E]*(N*N)
    order = list(range(N*N))  # fixed cell order to avoid permutations leading to same position with different sequences? 
    # However, we want NO symmetry reduction and DO want distinct positions by placement cells, but
    # identical final boards via different orderings should count once. We'll generate by combinations + assignment of X/O pattern (p counts).
    # Construct all subsets of size p and fill with alternating X/O following play order, but different permutations yield same final board; 
    # We instead use recursion placing moves sequentially with turn alternation; but memo final board set to dedup by final layout only.
    used = set()
    def rec(board, turn, moves_done, last_winner):
        if last_winner is not None:
            return
        if moves_done == p:
            t = tuple(board)
            if t not in used:
                used.add(t)
                boards.append(list(board))
            return
        for idx in range(N*N):
            if board[idx]==E:
                board[idx]=turn
                w = winner(board, lines)
                rec(board, next_player(turn), moves_done+1, w)
                board[idx]=E
    rec(init, X, 0, None)
    return boards

def average_maps(N:int, K:int, max_ply:int):
    lines = all_win_lines(N,K)
    result_win = { "X": {}, "O": {} }
    result_draw= { "X": {}, "O": {} }

    for p in range(max_ply+1):
        # Whose turn at ply p (X starts at p=0)
        turn_at_p = X if p%2==0 else O
        boards = enumerate_boards_exact_ply(N, p, lines)

        # prepare accumulators per role (we compute maps for both roles at this ply, per user request)
        for role_name, role in [("X", X), ("O", O)]:
            win_map = [[0.0 for _ in range(N)] for __ in range(N)]
            draw_map= [[0.0 for _ in range(N)] for __ in range(N)]
            count_map=[[0   for _ in range(N)] for __ in range(N)]

            for B in boards:
                # Skip terminal boards (shouldn't occur due to construction), but be safe
                if winner(B, lines) is not None:
                    continue
                # It's "role" to move? If not, we still compute the map under the counterfactual "if it's role's turn now"?
                # The user's prior tensor uses role as an index alongside ply, so we interpret as: at ply p, for the role who would move at p, that's direct;
                # for the opposite role, it's the next ply. But to keep it simple and symmetric, we compute 'role moves now' by evaluating outcome from this board 
                # assuming 'role' moves next REGARDLESS of parity, i.e., treat as separate slice (counterfactual). 
                # Implement by evaluating after forcing 'role' to play first move on the board.
                cache = {}
                for cell in range(N*N):
                    if B[cell]==E:
                        # root player is 'role'; first action is role plays at 'cell'
                        B[cell] = role
                        w,d = evaluate_random_uniform(B, next_player(role), lines, root_player=role, cache=cache)
                        B[cell] = E
                        y,x = divmod(cell, N)
                        win_map[y][x] += w
                        draw_map[y][x]+= d
                        count_map[y][x]+= 1

            # average over all boards where the cell was legal
            for y in range(N):
                for x in range(N):
                    c = count_map[y][x]
                    if c>0:
                        win_map[y][x] /= c
                        draw_map[y][x]/= c
                    else:
                        win_map[y][x] = 0.0
                        draw_map[y][x]= 0.0

            result_win[role_name][f"ply_{p}"] = win_map
            result_draw[role_name][f"ply_{p}"] = draw_map

        print(f"[ply {p}] boards={len(boards)} (N={N}, K={K})")

    return { "win": result_win, "draw": result_draw }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--K", type=int, default=None, help="K-in-a-row (default N)")
    ap.add_argument("--max_ply", type=int, default=4)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()
    K = args.K if args.K is not None else args.N

    t0 = time.time()
    data = average_maps(args.N, K, args.max_ply)
    with open(args.out_json, "w") as f:
        json.dump(data, f)
    dt = time.time()-t0
    print(f"[OK] Wrote {args.out_json} in {dt:.2f}s")

if __name__ == "__main__":
    main()
