#!/usr/bin/env python3
"""
search/mcts.py — MCTS with PUCT, local RNG, and (deterministic or prior-weighted) rollouts.

State interface expected:
- state.legal_moves() -> List[Tuple[int,int]]
- state.play(y, x)    -> New State (pure)
- state.is_terminal() -> bool
- state.side_to_move  -> 1 or "X" for X-to-move; anything else for O
- state.move_count    -> int (plies played)
- state.result()      -> "X"/"O"/"draw" (only when terminal)

PYTHONPATH=. python - <<'PY'
from search.mcts import MCTS, MCTSConfig
from tictactoe_4x4_bitboard import GameState
cfg = MCTSConfig(
    sims=4000, c_puct=0.9,
    dirichlet_alpha=0.0, dirichlet_eps=0.0,
    deterministic_rollout=True,
    order_in_expand=True,
    beam_k=6,
    seed=1018
)
m = MCTS(GameState.initial(), cfg); m.run()
print("MCTS(hybrid, top6) Best:", m.best_move())
PY

"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import math, random

from signals.move_ordering_generic import load_priors, rank_moves

# —— 先验（使用纯 4×4 数据生成的版本） ——
PRIORS  = load_priors("priors_N4_beta0_hybrid.json")
BOARD_N = 4

def role_of(state) -> str:
    return "X" if getattr(state, "side_to_move", 1) in (1, "X") else "O"

def terminal_value(state) -> float:
    res = getattr(state, "result", lambda: None)()
    if res == "X": return 1.0
    if res == "O": return -1.0
    return 0.0

@dataclass
class MCTSConfig:
    sims: int = 3000
    c_puct: float = 1.0
    # Root Dirichlet noise (AlphaGo style). Set alpha=eps=0 to disable.
    dirichlet_alpha: float = 0.0
    dirichlet_eps: float = 0.0
    # Rollout options
    rollout_max_len: int = 80
    use_prior_rollout: bool = True          # use priors during rollout
    deterministic_rollout: bool = False     # if True, rollout picks prior Top-1 deterministically
    prior_anneal: float = 1.0               # scale prior influence in rollout (if weighted)
    # Expansion options
    order_in_expand: bool = True            # use priors to order expansion
    beam_k: Optional[int] = None            # expand only top-k moves (by prior)
    # Reproducibility
    seed: Optional[int] = None              # local RNG seed (e.g., 1018 for fixed runs)

@dataclass
class Node:
    state: any
    parent: Optional["Node"] = None
    move_from_parent: Optional[Tuple[int,int]] = None

    children: Dict[Tuple[int,int], "Node"] = field(default_factory=dict)

    # Stats
    N: int = 0
    W: float = 0.0

    # Priors on actions at this node
    P: Dict[Tuple[int,int], float] = field(default_factory=dict)  # prior prob for each action
    Q0: float = 0.0  # optional prior value baseline (unused; keep 0)

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else self.Q0

class MCTS:
    def __init__(self, root_state, cfg: MCTSConfig):
        self.root = Node(root_state)
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)  # independent RNG
        self._ensure_expanded(self.root, is_root=True)

    # ---- Weighted choice using local RNG ----
    @staticmethod
    def _normalize(weights: List[float]) -> List[float]:
        s = sum(weights)
        if s <= 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / s for w in weights]

    def _weighted_choice(self, items: List, weights: List[float]):
        weights = self._normalize(weights)
        r = self.rng.random()
        acc = 0.0
        for item, w in zip(items, weights):
            acc += w
            if r <= acc:
                return item
        return items[-1]

    # ---- Dirichlet noise using local RNG ----
    def _dirichlet(self, k: int, alpha: float) -> List[float]:
        xs = [self.rng.gammavariate(alpha, 1.0) for _ in range(k)]
        return self._normalize(xs)

    # ---- Selection: PUCT ----
    def select(self) -> Node:
        node = self.root
        while not node.state.is_terminal():
            if not node.children:
                self._ensure_expanded(node)
                break
            sqrtNp = math.sqrt(max(1, node.N))
            best_val, best_child = None, None
            for mv, child in node.children.items():
                Pmv = node.P.get(mv, 0.0)
                Q = child.Q()
                U = self.cfg.c_puct * Pmv * sqrtNp / (1 + child.N)
                val = Q + U
                if (best_val is None) or (val > best_val):
                    best_val, best_child = val, child
            node = best_child
        return node

    # ---- Expansion: children + priors P from rank_moves ----
    def _ensure_expanded(self, node: Node, is_root: bool = False):
        if node.state.is_terminal() or node.children:
            return
        legal = node.state.legal_moves()
        if not legal:
            return

        ply  = getattr(node.state, "move_count", 0)
        role = role_of(node.state)

        ordered = rank_moves(legal, ply, role, PRIORS, BOARD_N, top_k=self.cfg.beam_k) \
                  if self.cfg.order_in_expand else [(mv, 1.0) for mv in legal]
        if not ordered:
            return

        moves, scores = zip(*ordered)
        probs = self._normalize(list(scores))

        # Root noise
        if is_root and self.cfg.dirichlet_alpha > 0 and self.cfg.dirichlet_eps > 0 and len(moves) > 1:
            noise = self._dirichlet(len(moves), self.cfg.dirichlet_alpha)
            eps = self.cfg.dirichlet_eps
            probs = [(1 - eps) * p + eps * n for p, n in zip(probs, noise)]
            probs = self._normalize(probs)

        node.P = {}
        for mv, p in zip(moves, probs):
            if mv not in node.children:
                node.children[mv] = Node(node.state.play(*mv), parent=node, move_from_parent=mv)
            node.P[mv] = float(p)

    # ---- Rollout: deterministic (Top-1) or prior-weighted ----
    def rollout(self, state) -> float:
        s = state
        steps = self.cfg.rollout_max_len
        ply = getattr(s, "move_count", 0)
        while (not s.is_terminal()) and steps > 0:
            legal = s.legal_moves()
            if not legal:
                break
            role = role_of(s)

            if self.cfg.deterministic_rollout:
                mv = rank_moves(legal, ply, role, PRIORS, BOARD_N, top_k=1)[0][0]
            elif self.cfg.use_prior_rollout:
                ordered = rank_moves(legal, ply, role, PRIORS, BOARD_N)
                ms, ws = zip(*ordered)
                ws = [max(1e-12, (w ** self.cfg.prior_anneal)) for w in ws]
                mv = self._weighted_choice(list(ms), list(ws))
            else:
                mv = self._weighted_choice(legal, [1.0] * len(legal))

            s = s.play(*mv)
            ply += 1
            steps -= 1

        return terminal_value(s) if s.is_terminal() else 0.0

    # ---- Backprop ----
    def backprop(self, node: Node, value: float):
        n = node
        v = value
        while n is not None:
            n.N += 1
            n.W += v
            v = -v   # switch perspective
            n = n.parent

    # ---- Main loop ----
    def run(self):
        for _ in range(self.cfg.sims):
            leaf = self.select()
            if not leaf.state.is_terminal():
                self._ensure_expanded(leaf)
                if leaf.children:
                    # Prefer high prior child to start rollout
                    mv = max(leaf.P.items(), key=lambda kv: kv[1])[0]
                    leaf = leaf.children[mv]
            v = self.rollout(leaf.state)
            self.backprop(leaf, v)

    def best_move(self) -> Tuple[int,int]:
        if not self.root.children:
            self._ensure_expanded(self.root, is_root=True)
            if not self.root.children:
                raise RuntimeError("No legal moves at root.")
        mv, child = max(self.root.children.items(), key=lambda kv: kv[1].N)
        return mv
