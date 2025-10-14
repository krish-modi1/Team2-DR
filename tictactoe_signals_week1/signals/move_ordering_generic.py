#!/usr/bin/env python3
"""
move_ordering_generic.py
给合法落子打分并排序，用于 N=3 / N=4。需要配套的先验 JSON：
- N=3: priors_N3_beta0.json
- N=4: priors_N4_beta0.json
"""

import json, math
from typing import List, Tuple, Dict

# ---------- 基础几何分组 ----------
def is_corner(y: int, x: int, N: int) -> bool:
    return (y, x) in [(0, 0), (0, N-1), (N-1, 0), (N-1, N-1)]

def is_edge(y: int, x: int, N: int) -> bool:
    if is_corner(y, x, N):
        return False
    return (y == 0) or (y == N-1) or (x == 0) or (x == N-1)

def is_center(y: int, x: int, N: int) -> bool:
    # 仅奇数 N 有唯一中心
    return (N % 2 == 1) and (y == N // 2) and (x == N // 2)

def manhattan_shell(y: int, x: int, N: int) -> float:
    """
    用几何中心计算 L1 半径。
    - 奇数 N：几何中心与格点重合（例如 N=3 的 (1,1)），结果等同于旧定义。
    - 偶数 N：几何中心在半格（例如 N=4 的 (1.5,1.5)），
      四个内格 (1,1),(1,2),(2,1),(2,2) 的半径将完全对称（都为 1.0）。
    """
    cy = N / 2 - 0.5
    cx = N / 2 - 0.5
    return abs(y - cy) + abs(x - cx)

def _shell_weight_from_float_radius(p_slice: dict, r_float: float) -> float:
    """
    将连续半径 r_float 线性插值到离散的 shell_bias（索引 0..r_max）。
    例如 r=1.3 -> 在 shell[1] 与 shell[2] 之间线性插值。
    """
    shells = p_slice["shell_bias"]
    r_max = len(shells) - 1
    if r_float <= 0:
        return float(shells[0])
    if r_float >= r_max:
        return float(shells[r_max])

    lo = int(math.floor(r_float))
    hi = lo + 1
    t = r_float - lo
    return float((1.0 - t) * shells[lo] + t * shells[hi])

# ---------- 先验加载 ----------
def load_priors(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

# ---------- 单步打分 ----------
def score_move(
    y: int,
    x: int,
    ply: int,
    role: str,          # 'X' 或 'O'
    priors: Dict,
    N: int,
    mode: str = "product",  # 'product' 或 'linear'
) -> float:
    """
    返回对落子 (y,x) 的先验分数；分数越高，越应该优先探索/抽样。
    """
    root = f"N{N}"
    key  = f"ply_{ply}_{role}"
    if root not in priors or key not in priors[root]:
        return 1.0  # 找不到对应切片就给中性分

    p = priors[root][key]

    # D4 权重：N=3 有 center；N=4 只有 corner/edge（内部格子 D4=1.0，由行/列/壳区分）
    if is_center(y, x, N):
        w_d4 = p["d4_weights"].get("center", 1.0)
    elif is_corner(y, x, N):
        w_d4 = p["d4_weights"].get("corner", 1.0)
    elif is_edge(y, x, N):
        w_d4 = p["d4_weights"].get("edge", 1.0)
    else:
        # N=4 的内部格子：不施加 D4 偏置，让 row/col/shell 发挥作用
        w_d4 = 1.0

    # 行/列/壳（这些在生成时已做“均值=1”的归一化）
    w_row = p["row_bias"][y]
    w_col = p["col_bias"][x]
    r_float = manhattan_shell(y, x, N)
    w_shell = _shell_weight_from_float_radius(p, r_float)

    if mode == "product":
        score = w_d4 * w_row * w_col * w_shell
    else:
        # 简单线性加权（可按需微调）
        a, b, c, d = 0.4, 0.2, 0.2, 0.2
        score = a * w_d4 + b * w_row + c * w_col + d * w_shell

    return float(score)

# ---------- 批量排序 ----------
def rank_moves(
    legal_moves: List[Tuple[int, int]],
    ply: int,
    role: str,
    priors: Dict,
    N: int,
    top_k: int | None = None,
    anneal: float = 1.0,   # 退火系数，<1 可在后期降低先验影响
) -> List[Tuple[Tuple[int, int], float]]:
    """
    返回 [((y,x), score), ...]，按分数降序排列。
    """
    scored = []
    for (y, x) in legal_moves:
        s = score_move(y, x, ply, role, priors, N)
        s *= anneal  # 可选：让先验随 ply 衰减 -> 在外面算个 alpha(ply) 传进来
        scored.append(((y, x), s))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k] if top_k else scored

# ---------- 加权随机选择（用于 rollout） ----------
def select_move_weighted(
    legal_moves: List[Tuple[int, int]],
    ply: int,
    role: str,
    priors: Dict,
    N: int,
    anneal: float = 1.0,
):
    """
    按分数当作权重做一次抽样，返回一个 (y,x)。
    """
    import random
    scored = rank_moves(legal_moves, ply, role, priors, N, top_k=None, anneal=anneal)
    moves, weights = zip(*scored)
    # 防止极端情况（全 0）：加个极小值
    eps = 1e-9
    weights = [w + eps for w in weights]
    return random.choices(moves, weights=weights, k=1)[0]
