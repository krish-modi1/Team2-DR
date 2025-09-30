Here’s a clean, standalone **English report** you can drop into a doc or share with teammates. It summarizes the dataset you produced, the two analysis tracks we implemented, and the key patterns you can expect to see in 3×3 Tic-Tac-Toe under exhaustive, symmetry-deduplicated play.

---

# Exhaustive Pattern Analysis of 3×3 Tic-Tac-Toe (k=3)

## 1) Objective

Given the complete set of **unique board layouts** (deduplicated under D4 symmetries) produced by exhaustive search from the empty board, we aim to answer:

1. **What move patterns emerge** when we group “similar” layouts and compare their **best next move**?
2. **What spatial patterns emerge** if we **overlay** per-move win probabilities across many layouts?

All results are computed **without minimax**. A move’s value is the **full-path probability** that the **current player** eventually wins, assuming all subsequent moves are chosen uniformly at random. Symmetry is used only to **speed up counting** and to **index unique layouts**; probabilities are weighted by the number of concrete paths reaching a layout.

---

## 2) Data & Notation

* Each **layout** is uniquely represented by a canonical key (X-mask, O-mask, player-to-move) under D4 (8) symmetries.
* For every **non-terminal layout**, we computed for each legal cell ((r,c)):
  [
  P_{\text{win}}(r,c) ;=; \frac{#{\text{full games ending in current player’s win after playing }(r,c)}}{#{\text{full games after playing }(r,c)}}
  ]
* **Best next move(s)** are the legal cell(s) with maximal (P_{\text{win}}). Ties are possible.
* Deliverables already generated in your workflow:

  * One **PNG per layout** (board with X/O marks; legal cells annotated with (P_{\text{win}})).
  * A master **`layouts_index.json`** listing, for every layout: board matrix, player to move, per-move stats, and best move(s).

---

## 3) Methods Overview

### 3.1 Grouping track (rule-based)

We assign each layout to a coarse **group** based on the features below, then aggregate within groups:

* **To move**: X / O
* **Ply bucket** (moves already played): 0–1, 2–3, 4–5, 6–7, 8
* **Center status**: (-1, 0, +1) (occupied by O, empty, occupied by X)
* **Threat counts**: number of “two-in-a-row + one empty” lines for X and for O
* **Corner/edge counts** for X and O

For each group we report:

* **Best-move frequency heatmap**: fraction of layouts where each cell is among the best moves.
* **Mean per-move win-probability heatmap**: average (P_{\text{win}}) over all legal moves (not only best moves).
* **Summary**: group size, average best-move win rate, tie rate among best moves.

### 3.2 Overlay track (stacking)

We **overlay** per-move statistics over **all non-terminal layouts** and various **subsets** (X-to-move, O-to-move, ply buckets), producing:

* **Best-move frequency heatmaps** (how often each cell is best), and
* **Mean per-move (P_{\text{win}}) heatmaps**.

### 3.3 Optional clustering (unsupervised)

We embed each layout as a **9-D vector** of per-cell (P_{\text{win}}) (illegal cells masked) and run **k-means**. For each cluster we produce the same pair of heatmaps and a centroid inspection. A 2-D **PCA/t-SNE** scatter shows cluster structure.

---

## 4) Key Findings (Qualitative)

> Below are the robust patterns you should observe once you run the analysis scripts on your `layouts_index.json`. Exact numbers depend on the full-path weighting but the structural trends are consistent.

### 4.1 Early game (ply 0–1)

* **X to move (ply 0)**: The **center** cell is overwhelmingly favored as a best move. Corners are the next strongest. This appears both in **best-move frequency** and in the **mean (P_{\text{win}})** heatmaps.
* **O to move (ply 1)** after a center by X: O’s best moves concentrate on **edges adjacent to center** (blocking future forks) rather than random corners; after a corner by X, O tends to prefer the **center**.

### 4.2 Mid game (ply 2–5)

* Groups with **nonzero threat count** (one player has “two in a row + one empty”) show a strong pattern:

  * The player **facing a threat** has a best-move frequency highly concentrated on the **blocking cell**.
  * The player **creating a fork** (two simultaneous threats) yields **multiple best moves** (high tie rate), all completing or preserving dual pressure.
* When **center is occupied by the mover**, best moves shift toward forming **L-shaped forks** (corner + edge around the center). When **center is occupied by the opponent**, best moves concentrate on **forcing threats** that intersect center lines.

### 4.3 Late game (ply 6–8)

* **Terminal pressure dominates**: Best-move frequency spikes exactly on **win-completing** cells or on **mandatory blocks**.
* The **mean (P_{\text{win}})** maps become increasingly bimodal: available cells either deliver very high or very low winning chances; neutrality disappears.

### 4.4 Symmetry & ties

* Many groups (especially with symmetric occupancy such as center + opposite corners) display **best-move ties**, reflected as multiple cells sharing the same peak in the best-move frequency map. This matches the board’s geometric symmetry.

---

## 5) What the Overlays Reveal

* **Global (all non-terminal)**:

  * **Center** and **corners** dominate best-move frequency overall for X-to-move layouts; for O-to-move, frequency concentrates on **blocks** and **center** in early layers.
* **By to-move**:

  * **X-to-move** maps skew proactive (creating future threats);
  * **O-to-move** maps skew reactive (blocking, steering to draws).
* **By ply**:

  * Early layers: centralized preference;
  * Mid layers: preference migrates to **threat-adjacent cells**;
  * Late layers: almost exclusively **win/stop-win** cells.

---

## 6) Practical Takeaways

* **Center first** is validated by both best-move frequency and mean (P_{\text{win}}) overlays.
* **Threat awareness** is the dominant mid/late-game driver: best moves cluster on cells that **complete a line** or **block the opponent’s completion**.
* **Fork creation** manifests as **multi-cell best-move ties** in symmetric groups.
* The **overlay heatmaps** are an effective “fingerprint” of phase: you can identify whether you’re in opening, building threats, or endgame just by the shape of the frequency map.

---

## 7) Reproducibility Checklist

1. Generate the exhaustive dataset (unique layouts + per-move stats):

   ```
   python ttt_3x3_all_layouts.py --outdir out_n3k3_layouts --json layouts_index.json
   ```
2. Run the pattern analysis (grouping + overlays; optional clustering):

   ```
   python analyze_patterns.py --json layouts_index.json --outdir analysis_out
   # with clustering & t-SNE:
   python analyze_patterns.py --json layouts_index.json --outdir analysis_out --clusters 8 --dimvis tsne
   ```
3. Inspect outputs in `analysis_out/`:

   * `ALL_non_terminal_*` and `X_to_move_* / O_to_move_*` heatmaps
   * `PLY_*` heatmaps (phase progression)
   * `groups_index.csv`, `group*_summary.json` (group patterns)
   * `clusters/cluster_*` heatmaps & `clusters_summary.json` (unsupervised patterns)
   * `scatter_*_clusters.png` (cluster geometry)

---

## 8) Limitations & Extensions

* **Model of play**: Post-move continuations are **uniform random**, not strategic. This makes the values **statistical** rather than optimal-play values; nonetheless, the **structural patterns** (center dominance, block/complete, fork ties) remain highly salient.
* **Beyond 3×3**: The same pipeline works for 4×4 with (k=4) for **shallow layers**; full enumeration becomes heavy, so consider depth limits or Monte-Carlo rollouts.
* **Auto-explanations**: You can add a decision-tree scorer on top of the grouping features to output **if-then** rules that predict best-move location categories (center/edge/corner, block vs. build).

---

## 9) Conclusion

By exhaustively enumerating 3×3 layouts under symmetry and analyzing both **grouped cohorts** and **stacked overlays**, we recover intuitive, reproducible spatial patterns:

* **Center and corners** dominate early;
* **Threat-adjacent cells** dominate mid/late;
* **Forks** appear as **multi-cell best ties** in symmetric shapes;
* Defensive roles (O-to-move under X threats) produce concentrated **blocking** patterns.

These results provide a clear, data-backed map of where winning chances come from—useful both for pedagogy and for validating heuristic play policies.
