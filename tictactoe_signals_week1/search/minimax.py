#!/usr/bin/env python3
"""
search/minimax.py
Alpha-Beta (Negamax 形式) + 先验排序（rank_moves）

依赖你的 State 最小接口：
- state.legal_moves() -> List[(y,x)]
- state.play(y, x)    -> 新 State（不改原对象）
- state.is_terminal() -> bool
- state.side_to_move  -> 1 或 "X" 表示 X 行棋；其他表示 O
- state.move_count    -> int（已下手数）
- state.result()      -> "X"/"O"/"draw"（仅在终局时可用）
"""

from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass
import math

# —— 先验（使用纯 4×4 数据生成的版本） ——
from signals.move_ordering_generic import load_priors, rank_moves
PRIORS   = load_priors("priors_N4_beta0_hybrid.json")
BOARD_N  = 4

def role_of(state) -> str:
    return "X" if getattr(state, "side_to_move", 1) in (1, "X") else "O"

def evaluate(state) -> float:
    """终局：X 胜=+1，O 胜=-1，和棋=0；非终局默认 0（可按需替换成更强启发式）。"""
    if state.is_terminal():
        res = getattr(state, "result", lambda: None)()
        if res == "X": return 1.0
        if res == "O": return -1.0
        return 0.0
    return 0.0

@dataclass
class SearchConfig:
    depth: int = 6
    use_ordering: bool = True
    beam_k: int | None = None  # 仅探索前 k 手（可选）

def alpha_beta(state, cfg: SearchConfig, alpha: float = -math.inf, beta: float = math.inf) -> float:
    """Negamax 形式 alpha–beta，返回对当前行棋方的估值。"""
    if cfg.depth <= 0 or state.is_terminal():
        return evaluate(state)

    ply  = getattr(state, "move_count", 0)
    role = role_of(state)

    legal = state.legal_moves()
    if not legal:
        return evaluate(state)

    # 先验排序（高分优先；可选 beam_k 剪枝）
    if cfg.use_ordering:
        ordered = rank_moves(legal, ply, role, PRIORS, BOARD_N, top_k=cfg.beam_k)
        moves = [mv for (mv, _) in ordered]
    else:
        moves = legal

    best = -math.inf
    for (y, x) in moves:
        child = state.play(y, x)
        score = -alpha_beta(child, SearchConfig(cfg.depth - 1, cfg.use_ordering, cfg.beam_k), -beta, -alpha)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # 剪枝
    return best

def best_move(state, cfg: SearchConfig) -> Tuple[int, int]:
    """返回当前局面的最佳一步（按 alpha-beta 评估 argmax）。"""
    ply  = getattr(state, "move_count", 0)
    role = role_of(state)
    legal = state.legal_moves()
    assert legal, "No legal moves."

    ordered = rank_moves(legal, ply, role, PRIORS, BOARD_N, top_k=cfg.beam_k) if cfg.use_ordering else [(m, 0.0) for m in legal]
    alpha, beta = -math.inf, math.inf
    best_sc, best_mv = -math.inf, ordered[0][0]

    for (y, x), _ in ordered:
        sc = -alpha_beta(state.play(y, x), SearchConfig(cfg.depth - 1, cfg.use_ordering, cfg.beam_k), -beta, -alpha)
        if sc > best_sc:
            best_sc, best_mv = sc, (y, x)
        if sc > alpha:
            alpha = sc
    return best_mv
