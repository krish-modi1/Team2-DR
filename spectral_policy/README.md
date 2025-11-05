# Pattern-Driven Spectral Policies for Tic‑Tac‑Toe (N×N)

This repo explores a math‑first, interpretable alternative to neural networks for board games like Tic‑Tac‑Toe. We build a deterministic policy using spectral features (DCT coefficients) and linear regression. It trains in milliseconds, uses very little data, and plays competitively vs random and light MCTS baselines. The same idea generalizes from 3×3 to 4×4 and arbitrary N.

Beginners can run the examples as-is; researchers can inspect the full mathematical formulation and experiment toggles (symmetry, label sources, baselines).

## What we built

- Interpretable spectral features: 2D DCT‑II of the X/O occupancy grids.
- A single linear scorer s(ϕ) = wᵀϕ trained by ridge regression (closed form).
- Practical policy: immediate win/block safety, else argmax s(ϕ) over legal moves.
- Labeling:
  - 3×3: exhaustive optimal moves from an optimal decision tree.
  - 4×4 and general N: Monte Carlo rollouts vs Random for move value targets.
- Optional symmetry: average features across all 8 dihedral symmetries (D4).
- Baselines: Random and small MCTS/UCT opponents to calibrate strength.
- NxN engine and CLI to run experiments at arbitrary sizes.

## Why this matters

- Data‑efficient: closed‑form fit from small datasets; no backprop, no GPU.
- Fast: microseconds per move; easily embedded in search for stronger play.
- Interpretable: weights on frequency patterns; step‑by‑step move explanations.
- Generalizable: the same spectral basis transfers across board sizes.

## Files and roles

- `spectral_policy_3x3.py` — 3×3 policy trained on optimal labels; CV over λ, visualizations, W/D/L vs Random/Perfect; optional `--sym-avg`.
- `spectral_policy_4x4.py` — 4×4 policy trained on MC labels; evaluation vs Random and MCTS; optional `--sym-avg`.
- `spectral_policy_nxn.py` — generalized NxN trainer/evaluator; `--labels mc|optimal` (optimal for N=3), Random/MCTS baselines, optional `--sym-avg`.
- `optimal_decision_tree.json` — exhaustive optimal data for 3×3 (required by 3×3 optimal mode).

You may also have `analytical_patterns_3x3.py` (analytical hand‑crafted features); this compares cleanly with the spectral approach.

## Mathematical formulation

### Board encoding

- Let the board be flattened to length N² with values: 1 (X), −1 (O), 0 (empty).
- Build two binary grids X and O in ℝ^{N×N}: X[r,c]=1 if X occupies (r,c), O likewise for O.

### Spectral features (DCT‑II)

We use an orthonormal 1D DCT‑II matrix C∈ℝ^{N×N}, with

$$C_{u,x} = \alpha_u \cos\left(\frac{\pi(2x+1)u}{2N}\right),\quad \alpha_0 = \sqrt{\tfrac{1}{N}},\; \alpha_{u>0} = \sqrt{\tfrac{2}{N}}.$$

The 2D DCT is computed by separability: \(\mathrm{DCT2}(A) = C A C^T\).

Our feature map for a (post‑move) board state is

$$\phi \in \mathbb{R}^{2N^2} = \big[\mathrm{vec}(\mathrm{DCT2}(X));\; \mathrm{vec}(\mathrm{DCT2}(O))\big].$$

Optional symmetry averaging (invariance under D4) replaces ϕ by the average of DCT features across the 8 dihedral transforms of the board.

### Labels

- 3×3 optimal: for each non‑terminal state and legal move, label 1 if the move is optimal, 0 otherwise (from the tree).
- MC rollouts (any N): for each state and legal move, place the move and estimate

$$y = \mathbb{E}[\text{outcome}] \in [-1,1], \quad \text{win}=+1,\; \text{draw}=0,\; \text{loss}=-1,$$

by averaging random playouts to termination vs a Random opponent.

### Model and training

We fit a single linear scorer for post‑move features:

$$s(\phi) = w^\top \phi.$$

Ridge regression (closed form):

$$\min_w \; \|Xw - y\|_2^2 + \lambda \|w\|_2^2 \quad \Rightarrow \quad w = (X^\top X + \lambda I)^{-1} X^\top y.$$

For 3×3 we can select λ by K‑fold cross‑validation on state‑level accuracy (does argmax s pick an optimal move for that state?).

### Policy

- Safety (optional): check any immediate winning move; if none, block any opponent immediate win.
- Otherwise choose the legal move with the highest score s(ϕ) using features from the hypothetical post‑move board.

Note: If you prefer a “pure function” with no hand‑crafted heuristics, add `--no-safety` to disable immediate win/block. This runs the learned spectral scorer alone.

## How to run (Windows PowerShell)

From the project folder:

- 3×3 (optimal labels), compare symmetry:
```powershell
& "C:\Users\ASUS\anaconda3\python.exe" spectral_policy_3x3.py --games-random 50 --games-perfect 30 --seed 42
& "C:\Users\ASUS\anaconda3\python.exe" spectral_policy_3x3.py --sym-avg --games-random 50 --games-perfect 30 --seed 42
```

- 4×4 (MC labels), with/without symmetry and MCTS baseline:
```powershell
& "C:\Users\ASUS\anaconda3\python.exe" spectral_policy_4x4.py --samples 400 --rollouts 6 --games-random 50 --games-mcts 20 --mcts-iters 150 --seed 42
& "C:\Users\ASUS\anaconda3\python.exe" spectral_policy_4x4.py --sym-avg --samples 400 --rollouts 6 --games-random 50 --games-mcts 20 --mcts-iters 150 --seed 42
```

- General N×N (MC by default). Examples:
```powershell
# N=3 quick run (MC labels)
& "C:\Users\ASUS\anaconda3\python.exe" spectral_policy_nxn.py --n 3 --samples 200 --rollouts 6 --games-random 20 --seed 42

# N=3 with optimal labels (needs optimal_decision_tree.json)
& "C:\Users\ASUS\anaconda3\python.exe" spectral_policy_nxn.py --n 3 --labels optimal --games-random 50 --seed 42

# N=4 with MCTS baseline
& "C:\Users\ASUS\anaconda3\python.exe" spectral_policy_nxn.py --n 4 --samples 600 --rollouts 8 --games-random 100 --games-mcts 20 --mcts-iters 200 --seed 42

# N=5 exploratory
& "C:\Users\ASUS\anaconda3\python.exe" spectral_policy_nxn.py --n 5 --samples 300 --rollouts 12 --games-random 50 --games-mcts 20 --mcts-iters 150 --seed 42
```

Common flags:
- `--sym-avg` enforces symmetry invariance by feature averaging.
- `--no-safety` disables immediate win/block safety overrides to evaluate the bare linear scorer.

### Interpreting the 3×3 visualizer

`spectral_policy_3x3.py` can print a per-cell score grid and the top feature contributions for the selected move when you enable visualization (omit `--no-examples` or use `--show-game random|perfect`).

- The score grid shows s(ϕ) for each legal move; higher is better.
- The “Top feature contributions” list shows, for the chosen move, the largest magnitude terms of `w[i] * ϕ[i]` and their raw values `w[i]` and `ϕ[i]`.
- Feature order is `X[0,0]..X[2,2], O[0,0]..O[2,2]` which correspond to DCT-II coefficients (low to high frequency) for X then O channels.

Tip: Use `--topk 8` to see more contributors; use `--sym-avg` to view symmetry‑averaged features.

## What we found (selected results)

Numbers below are representative runs with seed 42; use higher game counts for tighter estimates.

- 3×3 (optimal labels)
  - Standard features:
    - CV acc ~0.516 (state‑level), tie‑aware ~0.623
    - Vs Random (30): W:29 D:1 L:0
    - Vs Perfect (30 alt): W:0 D:30 L:0 (draws as expected vs perfect play)
  - Symmetry‑averaged features:
    - CV acc ~0.507, tie‑aware ~0.638
    - Vs Random (30): W:29 D:1 L:0
    - Vs Perfect (30 alt): W:0 D:30 L:0
  - Takeaway: symmetry averaging changes little in 3×3 (patterns already symmetric; invariance removes directional cues but also reduces variance).

- 4×4 (MC labels, samples=400, rollouts=6)
  - Standard:
    - Vs Random (25): W:17 D:8 L:0
    - Vs MCTS@150 iters (10): W:10 D:0 L:0
  - Sym‑avg:
    - Vs Random (25): W:20 D:5 L:0
    - Vs MCTS@150 iters (10): W:8 D:2 L:0
  - Takeaway: both modes are competitive; small differences are within typical evaluation variance at these match counts.

- NxN quick checks (MC labels)
  - N=3: Vs Random (10): W:9 D:1 L:0
  - N=4: Vs Random (10): W:3 D:7 L:0; Vs MCTS@100 (5): W:5 D:0 L:0

### Safety vs no‑safety (trade‑offs)

Safety adds two hand‑coded checks: take an immediate win if available; otherwise block the opponent’s immediate win. Disabling it (`--no-safety`) evaluates the learned linear policy alone. Expect a classic precision–recall style trade‑off:

- With safety ON: more conservative, high draw rate, avoids blunders and near‑term losses.
- With safety OFF: more aggressive, often increases wins, but can also increase losses.

Observed quick runs (seed 42; your numbers will vary slightly):

- 3×3 vs Perfect (10 alt):
  - safety ON: W:0 D:10 L:0
  - safety OFF: W:0 D:5 L:5
- 3×3 vs Random (10):
  - safety ON: W:9 D:1 L:0
  - safety OFF: W:9 D:0 L:1
- 4×4 vs MCTS@150 (10): safety ON/OFF both achieved W:10 D:0 L:0 in a representative run (variance across small samples is expected).

Recommendation: keep safety ON when benchmarking basic strength and fairness; try `--no-safety` to evaluate the pure function approximator.

## Why symmetry averaging had small impact

- Tic‑Tac‑Toe patterns are inherently symmetric; the DCT basis already encodes global structure.
- Symmetry averaging enforces exact invariance but discards orientation; that information is often non‑critical for these tasks, so effects are modest.
- As a linear model, the scorer has limited capacity. It generalizes well with or without explicit invariance; augmentation or canonicalization may help more than averaging.

## Interpretable “analytical” patterns (optional)

You can define human‑readable pattern functions (e.g., open‑two, open‑three, center/corner/edge) as explicit polynomials of the indicator variables x_i,o_i,e_i. Example for a line L=(i,j,k):

- Immediate X win: \(\sum_L x_i x_j x_k\)
- X open‑two: \(\sum_L (x_i x_j e_k + x_i x_k e_j + x_j x_k e_i)\)
- O open‑two (threat): same with o’s.

Combine these with spectral features or use them standalone; fit weights with ridge/Lasso for sparse, readable policies.

## Limits and when to add search or nonlinearity

- Larger boards and deeper tactics benefit from a bit of search (MCTS/alpha‑beta). Our linear spectral score is an efficient evaluation for such search.
- For very complex games (e.g., Go), deep models with richer nonlinearity + search outperform linear policies. Still, the spectral approach is a strong, data‑efficient baseline and a great teaching tool.

## Suggested next steps

- Compare symmetry strategies: averaging vs augmentation vs canonicalization.
- Add a small set of line‑threat counts (open‑k features) and test with sparse regression.
- Add a `--compare-sym` option to run standard vs sym‑avg within a single invocation and print a compact table.
- Extend to Connect Four (gravity) and Othello/Reversi (add mobility/stability features), re‑use DCT as the spectral backbone.

## Reproducibility and performance

- All scripts accept `--seed` for reproducibility.
- Ridge solve is closed‑form with tiny matrices (≤ 2N² features); training is instantaneous.
- Inference is a handful of matrix multiplies per legal move.

## Credits
## Troubleshooting and tips

- “Python was not found”: use your Anaconda Python explicitly, e.g.
  `& "C:\\Users\\ASUS\\anaconda3\\python.exe" spectral_policy_3x3.py ...`
- Missing `optimal_decision_tree.json`: the 3×3 optimal‑label modes require this file (present in this repo). If you renamed or moved it, put it back in the project root or pass an absolute path if the script supports it. For NxN with `--labels mc`, the file is not needed.
- Long runs: reduce `--games-*`, `--samples`, and/or `--rollouts` for quicker smoke tests; increase for stable estimates.
- MCTS strength: use higher `--mcts-iters` for a stronger opponent; note that runtime grows roughly linearly with iterations.

## API quick reference (choose_move contract)

- Input: a `TicTacToe` game object with fields `board` (length N², values {1,0,−1}) and `current_player` (1=X or −1=O).
- Output: an integer move index in `[0, N²)` corresponding to the chosen empty cell.
- Error modes: raises if no legal moves (should not happen because callers check `is_terminal()`); otherwise deterministic given the same `w`, `sym_avg`, `safety`, and RNG seed for any stochastic opponents used in evaluation.

Edge cases handled:
- Empty board, nearly full board, and one‑move‑to‑win/loss are supported.
- With `--sym-avg`, feature invariance ensures rotated/flipped boards map to the same representation (within numeric precision).

- DCT‑II basis and separability are standard; we adopt the orthonormal variant for stability.
- Dihedral group (D4) symmetries provide natural invariances for square grids.
- MCTS opponent is a minimal UCT implementation used for calibration.
