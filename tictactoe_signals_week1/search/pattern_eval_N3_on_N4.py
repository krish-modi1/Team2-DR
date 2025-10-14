#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pattern_eval_N3_on_N4.py
用 3×3 的先验/规律对 4×4 局面打分 -> 选 best move，
并和 4×4 的 P_eff（来自 bruteforce JSON）对齐评估。

示例见文件末尾注释。
"""

import argparse, json, math, itertools, random
from typing import List, Tuple, Dict, Optional

# ---------- 基础常量 ----------
X, O, E = 1, 2, 0

# ---------- 纯 Python Spearman（避免依赖 scipy） ----------
def _spearmanr_py(xs, ys):
    """纯 Python Spearman：平均名次处理 ties，返回 rho 或 None"""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    def rank_avg(vs):
        idx = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0]*len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j+1 < len(vs) and vs[idx[j+1]] == vs[idx[i]]:
                j += 1
            avg = (i+1 + j+1)/2.0
            for k in range(i, j+1):
                ranks[idx[k]] = avg
            i = j+1
        return ranks
    rx, ry = rank_avg(xs), rank_avg(ys)
    mx = sum(rx)/len(rx); my = sum(ry)/len(ry)
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(len(rx)))
    denx = (sum((r-mx)**2 for r in rx))**0.5
    deny = (sum((r-my)**2 for r in ry))**0.5
    if denx == 0 or deny == 0:
        return None
    return num/(denx*deny)

# ---------- 几何辅助 ----------
def is_corner(y:int,x:int,N:int)->bool:
    return (y,x) in [(0,0),(0,N-1),(N-1,0),(N-1,N-1)]

def is_edge(y:int,x:int,N:int)->bool:
    return (y in (0,N-1)) or (x in (0,N-1))

def manhattan_shell_geom(y:int,x:int,N:int)->float:
    """几何中心 L1 半径（偶数 N 时中心在半格）"""
    cy = N/2 - 0.5
    cx = N/2 - 0.5
    return abs(y - cy) + abs(x - cx)

def shell_weight_from_float(shells:List[float], r:float)->float:
    """对连续半径做线性插值到离散 shells"""
    rmax = len(shells)-1
    if r<=0: return float(shells[0])
    if r>=rmax: return float(shells[rmax])
    lo = int(math.floor(r)); hi = lo+1; t = r-lo
    return float((1-t)*shells[lo] + t*shells[hi])

# ---------- 读取 3×3 先验 ----------
def load_priors_n3(path:str)->Dict:
    with open(path,"r") as f:
        return json.load(f)

# ---------- 3×3 → 4×4 的“形状扩展” ----------
def _norm_mean_one(v):
    m = sum(v)/len(v) if v else 1.0
    return [x/m if m>0 else 1.0 for x in v]

def expand_rows_cols_shells_from_N3_to_N4(p3_slice:dict)->Dict:
    # 行/列：把 N3 的中线复制一份 -> 长度 4
    rows3 = p3_slice["row_bias"]; cols3 = p3_slice["col_bias"]; shells3 = p3_slice["shell_bias"]
    rows4 = [rows3[0], rows3[1], rows3[1], rows3[2]]
    cols4 = [cols3[0], cols3[1], cols3[1], cols3[2]]
    # 壳：粗映射到 0..4（N3 的最外一层映到 N4 的 r=3/4）
    idx_map = [0,1,2,2,3]
    shells4 = [ shells3[min(i, len(shells3)-1)] for i in idx_map ]
    return {
        "row_bias": _norm_mean_one(rows4),
        "col_bias": _norm_mean_one(cols4),
        "shell_bias": _norm_mean_one(shells4),
        # D4：N4 的“内格”用 N3 的 center 作为基准（也可设为 1.0）
        "d4_weights": {
            "corner": float(p3_slice["d4_weights"].get("corner",1.0)),
            "edge":   float(p3_slice["d4_weights"].get("edge",1.0)),
            "inner":  float(p3_slice["d4_weights"].get("center",1.0)),
        }
    }

# ---------- 用 N3 先验对 4×4 的单步打分 ----------
def score_move_from_N3(y:int,x:int, board:List[List[int]], ply:int, role_name:str, P3:Dict)->float:
    key3 = f"ply_{min(ply,8)}_{role_name}"
    if "N3" not in P3 or key3 not in P3["N3"]:
        return 1.0
    p3 = P3["N3"][key3]
    p4 = expand_rows_cols_shells_from_N3_to_N4(p3)

    N = 4
    # D4 权重
    if is_corner(y,x,N):
        w_d4 = p4["d4_weights"]["corner"]
    elif is_edge(y,x,N):
        w_d4 = p4["d4_weights"]["edge"]
    else:
        w_d4 = p4["d4_weights"]["inner"]   # N4 内格

    # 行/列/壳
    w_row = p4["row_bias"][y]
    w_col = p4["col_bias"][x]
    r = manhattan_shell_geom(y,x,N)
    w_shell = shell_weight_from_float(p4["shell_bias"], r)

    return float(w_d4 * w_row * w_col * w_shell)

def rank_moves_from_N3(board:List[List[int]], role:int, ply:int, P3:Dict)->List[Tuple[Tuple[int,int], float]]:
    role_name = "X" if role==X else "O"
    legal = [(y,x) for y in range(4) for x in range(4) if board[y][x]==E]
    scored = [((y,x), score_move_from_N3(y,x,board,ply,role_name,P3)) for (y,x) in legal]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored

# ---------- 读取 4×4 的 bruteforce JSON（胜/和表） ----------
def load_tables_from_json(path:str)->Dict:
    with open(path,"r") as f:
        return json.load(f)  # {"win":{"X":{"ply_0":[[...]]...},"O":...},"draw":{...}}

def peff_map(tables:Dict, ply:int, role:int, beta:float)->List[List[float]]:
    role_name = "X" if role==X else "O"
    key = f"ply_{ply}"
    W = tables["win"][role_name][key]
    D = tables["draw"][role_name][key]
    N = len(W)
    M = [[0.0]*N for _ in range(N)]
    for y in range(N):
        for x in range(N):
            M[y][x] = float(W[y][x] + beta * D[y][x])
    return M

# ---------- 文本/字符串到 4×4 棋盘 ----------
def parse_board_4x4(s:str)->List[List[int]]:
    """
    输入 16 字符：每个为 'X','O','.' 或 'x','o'
    也支持用 4 行、以 '/' 或 空格 分隔。
    """
    s = s.strip().replace("/"," ").replace("|"," ").replace(","," ")
    tokens = [t for t in s.split() if t]
    if len(tokens)==4 and all(len(t)==4 for t in tokens):
        flat = "".join(tokens)
    else:
        flat = "".join(ch for ch in s if ch in "XOxo.")
        if len(flat)!=16:
            raise ValueError("Board string needs 16 cells using X/O/. (e.g. '................')")
    board = []
    for i in range(0,16,4):
        row = []
        for ch in flat[i:i+4]:
            if ch in "Xx": row.append(X)
            elif ch in "Oo": row.append(O)
            else: row.append(E)
        row = row[:4]
        board.append(row)
    if len(board)!=4:
        raise ValueError("Expect 4 rows.")
    return board

def count_ply(board)->int:
    return sum(1 for y in range(4) for x in range(4) if board[y][x]!=E)

# ---------- 单局面：best move + 对齐 P_eff（含并列容差） ----------
def predict_single(args):
    P3 = load_priors_n3(args.priors_n3)
    tables = load_tables_from_json(args.bruteforce_json)

    board = parse_board_4x4(args.board)
    role = X if args.role.upper()=="X" else O
    ply  = count_ply(board)

    ranked = rank_moves_from_N3(board, role, ply, P3)
    (mv, s_top) = ranked[0]
    y,x = mv

    M = peff_map(tables, ply, role, args.beta)
    peff_at_mv = M[y][x]

    # 允许并列 + 容差
    eps = float(args.tol)
    legal = [(yy,xx) for yy in range(4) for xx in range(4) if board[yy][xx]==E]
    max_val = max(M[yy][xx] for (yy,xx) in legal)
    argmaxes = [(yy,xx) for (yy,xx) in legal if abs(M[yy][xx] - max_val) <= eps]
    hit = any(mv == a for a in argmaxes)

    print(f"[Input] ply={ply} role={args.role} board={args.board}")
    print(f"[N3→4x4] best_move={mv}  score={s_top:.6f}")
    print(f"[4x4 P_eff] beta={args.beta:.2f}  P_eff(best_move)={peff_at_mv:.6f}  oracle_max={max_val:.6f}  argmaxes={argmaxes}")
    print(f"[Match] hit_top1_or_tie={hit}")

# ---------- 胜负/合法性 ----------
def winner(board_flat, lines):
    for L in lines:
        vals = {board_flat[i] for i in L}
        if len(vals)==1 and next(iter(vals)) in (X,O):
            return next(iter(vals))
    if all(v!=E for v in board_flat): return 'draw'
    return None

def all_win_lines(N:int,K:int):
    lines=[]
    for r in range(N):                # rows
        for c in range(N-K+1):
            lines.append([r*N + (c+i) for i in range(K)])
    for c in range(N):                # cols
        for r in range(N-K+1):
            lines.append([(r+i)*N + c for i in range(K)])
    for r in range(N-K+1):            # diag \
        for c in range(N-K+1):
            lines.append([(r+i)*N + (c+i) for i in range(K)])
    for r in range(K-1, N):           # diag /
        for c in range(N-K+1):
            lines.append([(r-i)*N + (c+i) for i in range(K)])
    # dedup
    seen=set(); uniq=[]
    for L in lines:
        t=tuple(L)
        if t not in seen: seen.add(t); uniq.append(L)
    return uniq

def enumerate_boards_exact_ply(N:int, p:int, lines)->List[List[List[int]]]:
    boards=[]
    init=[E]*(N*N); used=set()
    def rec(board, turn, moves_done, last_winner):
        if last_winner is not None: return
        if moves_done==p:
            t=tuple(board)
            if t not in used: used.add(t); boards.append(list(board))
            return
        for idx in range(N*N):
            if board[idx]==E:
                board[idx]=turn
                w = winner(board, lines)
                rec(board, O if turn==X else X, moves_done+1, w)
                board[idx]=E
    rec(init, X, 0, None)
    # 转成 4×4 矩阵
    out=[]
    for B in boards:
        grid=[[E]*4 for _ in range(4)]
        for i,v in enumerate(B):
            y,x=divmod(i,4)
            grid[y][x]=v
        out.append(grid)
    return out

# ---------- 批量评估（含并列容差 + 纯 Python Spearman） ----------
def eval_bulk(args):
    P3 = load_priors_n3(args.priors_n3)
    tables = load_tables_from_json(args.bruteforce_json)
    lines = all_win_lines(4,4)

    rng = random.Random(123)
    total=0; hit=0
    abs_err_sum=0.0
    spear_vals=[]

    for p in range(args.max_ply+1):
        boards = enumerate_boards_exact_ply(4, p, lines)
        if args.sample_per_ply and len(boards)>args.sample_per_ply:
            boards = rng.sample(boards, args.sample_per_ply)
        for B in boards:
            flat = [v for row in B for v in row]
            if winner(flat, lines) is not None:
                continue
            role = X if p%2==0 else O
            ranked = rank_moves_from_N3(B, role, p, P3)
            mv_pred,_ = ranked[0]

            M = peff_map(tables, p, role, args.beta)
            legal = [(y,x) for y in range(4) for x in range(4) if B[y][x]==E]
            max_val = max(M[y][x] for (y,x) in legal)
            eps = float(args.tol)
            argmaxes = [(y,x) for (y,x) in legal if abs(M[y][x] - max_val) <= eps]

            hit += (mv_pred in argmaxes)
            total += 1
            abs_err_sum += abs(M[mv_pred[0]][mv_pred[1]] - max_val)

            # Spearman 排序相关（用我们自己的实现）
            pred_scores = [s for _,s in ranked]
            peff_scores = [M[mv[0]][mv[1]] for mv,_ in ranked]
            rho = _spearmanr_py(pred_scores, peff_scores)
            if rho is not None and not math.isnan(rho):
                spear_vals.append(rho)

    print(f"[Eval] beta={args.beta}  max_ply={args.max_ply}  sampled_per_ply={args.sample_per_ply or 'ALL'}  tol={args.tol}")
    print(f"  Top1 match (with ties) = {hit}/{total} = {hit/total:.3f}")
    print(f"  MAE(P_eff@pred vs oracle_max) = {abs_err_sum/total:.6f}")
    if spear_vals:
        print(f"  Spearman corr (mean over boards) = {sum(spear_vals)/len(spear_vals):.3f}")
    else:
        print("  Spearman corr: n/a")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors_n3", type=str, default="priors_N3_beta0.json")
    ap.add_argument("--bruteforce_json", type=str, default="out_4x4_shallow/bruteforce_N4_ply3.json")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=1e-9, help="并列容差，判等用")

    # 单局面
    ap.add_argument("--board", type=str, help="16 chars with X/O/., or 4 tokens like '.... .... .... ....'")
    ap.add_argument("--role", type=str, choices=["X","O"], help="who plays now")

    # 批量评估
    ap.add_argument("--eval_bulk", action="store_true")
    ap.add_argument("--max_ply", type=int, default=3)
    ap.add_argument("--sample_per_ply", type=int, default=0)

    args = ap.parse_args()

    if args.eval_bulk:
        eval_bulk(args)
    else:
        if not (args.board and args.role):
            raise SystemExit("单局面模式需要 --board 和 --role，例如 --board '................' --role X")
        predict_single(args)

if __name__ == "__main__":
    main()

"""
示例：
1) 单局面（空盘，X 走，beta=0.0）
   python search/pattern_eval_N3_on_N4.py \
     --priors_n3 priors_N3_beta0.json \
     --bruteforce_json out_4x4_shallow/bruteforce_N4_ply3.json \
     --beta 0.0 \
     --board '................' \
     --role X

2) 批量评估（p<=3，随机每层抽 300 个局面；并列容差 tol=1e-9）
   python search/pattern_eval_N3_on_N4.py \
     --priors_n3 priors_N3_beta0.json \
     --bruteforce_json out_4x4_shallow/bruteforce_N4_ply3.json \
     --beta 0.0 \
     --eval_bulk --max_ply 3 --sample_per_ply 300

注：
- 如果想只对比 win（不含 draw），可在 peff_map() 中改为仅返回 W。
- 若你后续有更深 ply 的 4×4 bruteforce JSON，把 --max_ply 和路径改一下即可。
"""
