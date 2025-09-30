#!/usr/bin/env python3
# Analyze patterns from 3x3 TicTacToe layouts_index.json
# - Method 1: Rule-based grouping + overlays
# - Method 2: Unsupervised clustering on 9-D per-move P(win) vectors
#
# Input: layouts_index.json from ttt_3x3_all_layouts.py
# Output: heatmaps (PNG) + summaries (JSON/CSV) + 2D scatter (PNG)
#
# Usage examples:
#   python analyze_patterns.py --json layouts_index.json --outdir analysis_out
#   python analyze_patterns.py --json layouts_index.json --outdir analysis_out --clusters 8 --dimvis tsne
#
# Notes:
# - Works even if scikit-learn is not installed (falls back to numpy k-means & PCA).
# - Handles NaN in per-move Pwin (illegal cells) by masking for mean / replacing with-1 for clustering.

import os, json, csv, argparse, math, random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# -------- Optional sklearn (with graceful fallback) --------
HAVE_SK = True
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA as SKPCA
    from sklearn.manifold import TSNE
except Exception:
    HAVE_SK = False

N = 3  # 3x3

# --------------------------
# Utilities & plotting
# --------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def plot_heatmap(mat: np.ndarray, title: str, path: str, vmin=None, vmax=None, annotate=True):
    plt.figure(figsize=(5,5))
    plt.imshow(mat, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    if annotate:
        for r in range(N):
            for c in range(N):
                val = mat[r,c]
                txt = "—" if (isinstance(val,float) and np.isnan(val)) else f"{val:.2f}"
                plt.text(c, r, txt, ha="center", va="center")
    plt.xticks(range(N)); plt.yticks(range(N))
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def ply_bucket(ply: int) -> str:
    if   ply <= 1: return "0-1"
    elif ply <= 3: return "2-3"
    elif ply <= 5: return "4-5"
    elif ply <= 7: return "6-7"
    else:          return "8"

def two_in_row_threat(board: List[List[int]], player: int) -> int:
    # 2-in-line + 1 empty as simple threat proxy
    lines = []
    for r in range(N): lines.append([(r,0),(r,1),(r,2)])
    for c in range(N): lines.append([(0,c),(1,c),(2,c)])
    lines.append([(0,0),(1,1),(2,2)])
    lines.append([(0,2),(1,1),(2,0)])
    cnt = 0
    for line in lines:
        vals = [board[r][c] for (r,c) in line]
        if vals.count(player)==2 and vals.count(0)==1 and vals.count(-player)==0:
            cnt += 1
    return cnt

def center_val(board: List[List[int]]) -> int:
    return board[1][1]

def corner_count(board: List[List[int]], player: int) -> int:
    corners = [(0,0),(0,2),(2,0),(2,2)]
    return sum(1 for (r,c) in corners if board[r][c]==player)

def edge_count(board: List[List[int]], player: int) -> int:
    edges = [(0,1),(1,0),(1,2),(2,1)]
    return sum(1 for (r,c) in edges if board[r][c]==player)

def layout_group_key(entry: Dict[str, Any]) -> Tuple:
    b = entry["board"]
    return (
        entry["to_move"],           # 'X' or 'O'
        ply_bucket(entry["ply"]),   # step bucket
        center_val(b),              # -1,0,1
        two_in_row_threat(b, 1),    # X threats
        two_in_row_threat(b, -1),   # O threats
        corner_count(b, 1), corner_count(b,-1),
        edge_count(b, 1), edge_count(b, -1)
    )

def summarize_group(entries: List[Dict[str,Any]], outdir: str, tag: str) -> Dict[str,Any]:
    if not entries:
        return {}
    freq = np.zeros((N,N), dtype=float)  # best-move frequency
    denom = 0
    best_vals = []
    ties = 0

    sum_p = np.zeros((N,N), dtype=float) # overlay mean Pwin for all legal moves
    cnt_p = np.zeros((N,N), dtype=float)

    for e in entries:
        pm = e.get("per_move", {})
        bm = e.get("best_moves", [])
        if bm:
            denom += 1
            if len(bm)>1: ties += 1
            vals = []
            for (r,c) in bm:
                key=f"{r},{c}"
                vals.append(float(pm[key]["p_current_player_win"]))
                freq[r,c]+=1.0
            if vals: best_vals.append(float(np.mean(vals)))
        for k,st in pm.items():
            r,c = map(int,k.split(","))
            p = float(st["p_current_player_win"])
            sum_p[r,c]+=p; cnt_p[r,c]+=1.0

    if denom>0: freq/=denom
    mean_p = np.full((N,N), np.nan, dtype=float)
    for r in range(N):
        for c in range(N):
            if cnt_p[r,c]>0:
                mean_p[r,c]=sum_p[r,c]/cnt_p[r,c]

    ensure_dir(outdir)
    plot_heatmap(freq, f"[{tag}] Best-move frequency", os.path.join(outdir, f"{tag}_bestmove_freq.png"), vmin=0.0, vmax=1.0)
    plot_heatmap(mean_p, f"[{tag}] Mean P(win) over legal moves", os.path.join(outdir, f"{tag}_mean_pwin.png"), vmin=0.0, vmax=1.0)

    summ = {
        "tag": tag,
        "num_layouts": len(entries),
        "avg_bestmove_pwin": float(np.mean(best_vals)) if best_vals else None,
        "tie_rate": float(ties/denom) if denom>0 else None
    }
    with open(os.path.join(outdir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    return summ

# --------------------------
# Vectorization for clustering
# --------------------------

def vectorize_entry(entry: Dict[str,Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (vec9, mask9) where vec9[c] = Pwin if legal else NaN; mask True where valid.
    We'll fill NaN with -1.0 for k-means space so 'illegal' becomes a distinct code.
    """
    v = np.full(9, np.nan, dtype=float)
    m = np.zeros(9, dtype=bool)
    pm = entry.get("per_move", {})
    for k,st in pm.items():
        r,c = map(int,k.split(","))
        p = float(st["p_current_player_win"])
        v[r*3+c] = p
        m[r*3+c] = True
    return v, m

def fallback_pca(X: np.ndarray, ncomp=2) -> np.ndarray:
    # zero-mean
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    # fill remaining NaN with 0
    Xc = np.where(np.isnan(Xc), 0.0, Xc)
    # SVD
    U,S,VT = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :ncomp] * S[:ncomp]
    return Z

def fallback_kmeans(X: np.ndarray, k: int, n_init: int = 8, iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    # simple Lloyd's algorithm
    rng = np.random.default_rng(0)
    best_inertia = None
    best_cent = None
    best_lab = None
    n = X.shape[0]
    for _ in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        C = X[idx].copy()
        for _it in range(iters):
            # assign
            d2 = ((X[:,None,:]-C[None,:,:])**2).sum(axis=2)
            lab = d2.argmin(axis=1)
            # update
            C_new = np.zeros_like(C)
            for j in range(k):
                sel = (lab==j)
                if sel.any():
                    C_new[j] = np.nanmean(X[sel], axis=0)
                else:
                    # re-seed empty cluster
                    C_new[j] = X[rng.integers(0,n)]
            if np.allclose(C_new, C, atol=1e-6): 
                C = C_new; break
            C = C_new
        # inertia
        inertia = 0.0
        for j in range(k):
            sel = (lab==j)
            if sel.any():
                diff = X[sel]-C[j]
                inertia += np.nansum(diff*diff)
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia; best_cent = C.copy(); best_lab = lab.copy()
    return best_lab, best_cent

# --------------------------
# Main analysis
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="layouts_index.json from ttt_3x3_all_layouts.py")
    ap.add_argument("--outdir", type=str, required=True, help="output directory")
    ap.add_argument("--clusters", type=int, default=8, help="k for k-means (default 8)")
    ap.add_argument("--dimvis", type=str, default="pca", choices=["none","pca","tsne"], help="2D visualization method")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_groups", type=int, default=12)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load layouts
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    layouts: List[Dict[str,Any]] = data["layouts"]

    # keep only non-terminal (have per_move)
    nt = [e for e in layouts if e.get("per_move")]

    # ---------- Method 1: rule-based groups ----------
    groups: Dict[Tuple, List[Dict[str,Any]]] = {}
    for e in nt:
        k = layout_group_key(e)
        groups.setdefault(k, []).append(e)

    # index groups
    group_items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    with open(os.path.join(args.outdir, "groups_index.csv"), "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["rank","size","to_move","ply_bucket","center","x_threats","o_threats","x_corners","o_corners","x_edges","o_edges"])
        for rank,(gk, glist) in enumerate(group_items, start=1):
            w.writerow([rank, len(glist), *gk])

    # summarize top-K groups
    for rank,(gk, glist) in enumerate(group_items[:args.max_groups], start=1):
        tag = f"group{rank}_tm{gk[0]}_ply{gk[1]}_ctr{gk[2]}_xT{gk[3]}_oT{gk[4]}"
        summarize_group(glist, args.outdir, tag)

    # Global overlays
    summarize_group(nt, args.outdir, "ALL_non_terminal")
    summarize_group([e for e in nt if e["to_move"]=="X"], args.outdir, "X_to_move")
    summarize_group([e for e in nt if e["to_move"]=="O"], args.outdir, "O_to_move")
    for b in ["0-1","2-3","4-5","6-7","8"]:
        summarize_group([e for e in nt if ply_bucket(e["ply"])==b], args.outdir, f"PLY_{b}")

    # ---------- Method 2: clustering ----------
    # Build feature matrix (n_samples, 9)
    vecs = []
    masks = []
    keep_idx = []
    for i,e in enumerate(nt):
        v,m = vectorize_entry(e)
        vecs.append(v)
        masks.append(m)
        keep_idx.append(i)
    V = np.vstack(vecs)                    # NaN where illegal
    M = np.vstack(masks).astype(bool)
    # For clustering space, replace NaN with -1 to indicate "illegal/unavailable"
    X = np.where(M, V, -1.0)

    k = max(2, int(args.clusters))
    if HAVE_SK:
        km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        centers = centers.reshape(k,9)
    else:
        labels, centers = fallback_kmeans(X, k=k, n_init=12, iters=200)

    # Per-cluster overlays
    clusters_dir = os.path.join(args.outdir, "clusters")
    ensure_dir(clusters_dir)
    cluster_summaries = []
    for c in range(k):
        idxs = [i for i,l in enumerate(labels) if l==c]
        subset = [nt[i] for i in idxs]
        tag = f"cluster_{c}_n{len(subset)}"
        summ = summarize_group(subset, clusters_dir, tag)
        # save centroid heatmaps: best freq proxy = centroid values where >=0; mean pwin = mask positive entries
        cen = centers[c].copy()
        cen_mat = cen.reshape(3,3)
        # centroid is in feature space with -1 at illegal; map to NaN for display
        cen_disp = np.where(cen_mat<0, np.nan, cen_mat)
        plot_heatmap(cen_disp, f"[{tag}] Centroid P(win) (feature space)", os.path.join(clusters_dir, f"{tag}_centroid_pwin.png"), vmin=0.0, vmax=1.0)
        # record
        cluster_summaries.append({
            "cluster": c,
            "size": len(subset),
            "avg_bestmove_pwin": summ.get("avg_bestmove_pwin") if summ else None,
            "tie_rate": summ.get("tie_rate") if summ else None
        })

    with open(os.path.join(clusters_dir, "clusters_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_summaries, f, ensure_ascii=False, indent=2)

    # 2D visualization
    if args.dimvis != "none":
        if args.dimvis == "tsne" and HAVE_SK:
            Z = TSNE(n_components=2, init="pca", random_state=args.seed, perplexity=max(5, min(30, X.shape[0]//10))).fit_transform(X)
        else:
            if HAVE_SK:
                Z = SKPCA(n_components=2, random_state=args.seed).fit_transform(X)
            else:
                Z = fallback_pca(X, ncomp=2)

        # scatter colored by cluster
        plt.figure(figsize=(6,5))
        for c in range(k):
            sel = (labels==c)
            plt.scatter(Z[sel,0], Z[sel,1], s=12, label=f"C{c}")
        plt.title(f"{args.dimvis.upper()} of layouts (k={k})")
        plt.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"scatter_{args.dimvis}_clusters.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # scatter colored by to_move
        plt.figure(figsize=(6,5))
        selX = np.array([e["to_move"]=="X" for e in nt])
        plt.scatter(Z[selX,0], Z[selX,1], s=12, label="X to move")
        plt.scatter(Z[~selX,0], Z[~selX,1], s=12, label="O to move")
        plt.title(f"{args.dimvis.upper()} by to-move")
        plt.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"scatter_{args.dimvis}_to_move.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # Also dump a CSV of cluster membership
    with open(os.path.join(args.outdir, "cluster_membership.csv"), "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["idx","cluster","ply","to_move","image"])
        for i,e in enumerate(nt):
            w.writerow([i, int(labels[i]), e["ply"], e["to_move"], e.get("image","")])

    print("Done. Outputs in:", args.outdir)

if __name__ == "__main__":
    main()
