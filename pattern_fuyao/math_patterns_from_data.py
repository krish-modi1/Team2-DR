#!/usr/bin/env python3
# Pure data → math patterns for 3x3 TicTacToe move-win heatmaps
# Input: layouts_index.json (from exhaustive pipeline)
# Output: PNG heatmaps + JSON summaries in --outdir
#
# Patterns extracted:
# 1) D4 (dihedral) group decomposition: energy in irreps A1, A2, B1, B2, E
# 2) PCA/SVD eigenpatterns (no sklearn needed) on 9-D move-win vectors
# 3) Low-rank approximation curve (variance captured vs rank)
# 4) Orbit (center/edge/corner) mean/variance stats
# 5) E-subspace principal directions (two orthogonal "nontrivial" patterns)
#
# Run:
#   python math_patterns_from_data.py --json layouts_index.json --outdir math_out

import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

N = 3

# ---------- load dataset (non-terminal entries with per-move pwin) ----------
def load_matrix(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = [e for e in data["layouts"] if e.get("per_move")]
    # build H: (#samples, 9) move-win vector; illegal cells = NaN
    H = []
    metas = []
    for e in entries:
        v = np.full(9, np.nan, dtype=float)
        for k, st in e["per_move"].items():
            r, c = map(int, k.split(","))
            v[r*3+c] = float(st["p_current_player_win"])
        H.append(v)
        metas.append({"ply": e["ply"], "to_move": e["to_move"]})
    H = np.vstack(H) if H else np.zeros((0,9))
    return H, metas

# ---------- D4 group (8 transforms) on 3x3 indices ----------
def idx(r,c): return r*3 + c
def rot90(p):  r,c = divmod(p,3); return idx(c, 2-r)
def rot180(p): r,c = divmod(p,3); return idx(2-r, 2-c)
def rot270(p): r,c = divmod(p,3); return idx(2-c, r)
def flip_h(p): r,c = divmod(p,3); return idx(r, 2-c)   # horizontal mirror
def flip_v(p): r,c = divmod(p,3); return idx(2-r, c)   # vertical mirror
def flip_d(p): r,c = divmod(p,3); return idx(c, r)     # main diag
def flip_a(p): r,c = divmod(p,3); return idx(2-c, 2-r) # anti diag

# Permutation matrices (9x9) for the 8 group elements, in fixed order:
# e, r90, r180, r270, fh, fv, fd, fa
def perm_matrix(mapper):
    P = np.zeros((9,9), dtype=float)
    for p in range(9):
        P[mapper(p), p] = 1.0
    return P

G = []
G.append(("e",    perm_matrix(lambda p: p)))
G.append(("r90",  perm_matrix(rot90)))
G.append(("r180", perm_matrix(rot180)))
G.append(("r270", perm_matrix(rot270)))
G.append(("fh",   perm_matrix(flip_h)))
G.append(("fv",   perm_matrix(flip_v)))
G.append(("fd",   perm_matrix(flip_d)))
G.append(("fa",   perm_matrix(flip_a)))

# Conjugacy classes in our ordering (sizes): e(1), r180(1), {r90,r270}(2), {fh,fv}(2), {fd,fa}(2)
cls_indices = {
    "e":   [0],
    "r180":[2],
    "r90r270":[1,3],
    "axes":[4,5],   # horizontal/vertical mirrors
    "diags":[6,7],  # diagonal mirrors
}

# Character table for D4 irreps  (order aligned with classes above):
# classes:      e   r180  r90/270  axes  diags
# sizes:        1     1      2       2      2
irreps = {
    "A1": [ 1,   1,     1,      1,     1 ],
    "A2": [ 1,   1,     1,     -1,    -1 ],
    "B1": [ 1,   1,    -1,      1,    -1 ],
    "B2": [ 1,   1,    -1,     -1,     1 ],
    "E" : [ 2,  -2,     0,      0,     0 ],
}
dim_irrep = {"A1":1, "A2":1, "B1":1, "B2":1, "E":2}
group_size = 8

# Build projection operators P_irrep = (d/|G|) * sum_g chi(g) R(g)
def build_projectors():
    Ps = {}
    # map class characters back to each element
    for name, chi in irreps.items():
        P = np.zeros((9,9), dtype=float)
        # add class by class with their character
        # e
        for i in cls_indices["e"]:
            P += chi[0] * G[i][1]
        # r180
        for i in cls_indices["r180"]:
            P += chi[1] * G[i][1]
        # r90,r270
        for i in cls_indices["r90r270"]:
            P += chi[2] * G[i][1]
        # axis mirrors
        for i in cls_indices["axes"]:
            P += chi[3] * G[i][1]
        # diagonal mirrors
        for i in cls_indices["diags"]:
            P += chi[4] * G[i][1]
        P *= (dim_irrep[name] / group_size)
        Ps[name] = P
    return Ps

P_ir = build_projectors()  # dict of 9x9 projectors

# ---------- utilities ----------
def vec2img(v): return v.reshape(3,3)
def img2vec(M): return M.reshape(9)

def save_heatmap(M, title, path, vmin=0.0, vmax=1.0, annotate=True):
    plt.figure(figsize=(4.2,4.2))
    plt.imshow(M, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    if annotate:
        for r in range(3):
            for c in range(3):
                x = M[r,c]
                if np.isnan(x):
                    txt = "—"
                else:
                    txt = f"{x:.2f}"
                plt.text(c, r, txt, ha="center", va="center")
    plt.xticks(range(3)); plt.yticks(range(3))
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# pairwise-NaN aware mean
def nanmean2d(A):
    return np.nanmean(A, axis=0)

# SVD (PCA) with NaN: simple mean-imputation per coordinate
def pca_with_nan(H, k=5):
    # column means ignoring NaN
    col_mean = np.nanmean(H, axis=0)
    X = np.where(np.isnan(H), col_mean[None,:], H)
    # center
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U,S,VT = np.linalg.svd(Xc, full_matrices=False)
    # components = rows of VT
    comps = VT[:k]          # (k,9)
    explained = (S[:k]**2) / (S**2).sum()
    return mu.flatten(), comps, explained

def project_energy(H, P):
    # Energy of projection: ||(H P)||^2 / ||H||^2
    # Use mean-imputed version for stability (same as PCA)
    col_mean = np.nanmean(H, axis=0)
    X = np.where(np.isnan(H), col_mean[None,:], H)
    Xc = X - X.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc @ P, ord='fro')**2
    den = np.linalg.norm(Xc, ord='fro')**2 + 1e-12
    return num / den

def orbit_masks():
    # D4 orbits on 3x3: center, edges, corners
    center = np.zeros(9, dtype=bool); center[idx(1,1)] = True
    edges = np.zeros(9, dtype=bool)
    for (r,c) in [(0,1),(1,0),(1,2),(2,1)]: edges[idx(r,c)]=True
    corners = np.zeros(9, dtype=bool)
    for (r,c) in [(0,0),(0,2),(2,0),(2,2)]: corners[idx(r,c)]=True
    return center, edges, corners

def rank_approx_curve(H, out_png, max_k=9):
    col_mean = np.nanmean(H, axis=0)
    X = np.where(np.isnan(H), col_mean[None,:], H)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U,S,VT = np.linalg.svd(Xc, full_matrices=False)
    total = (S**2).sum()
    xs = []; ys = []
    acc = 0.0
    for k in range(1, max_k+1):
        acc = (S[:k]**2).sum() / total
        xs.append(k); ys.append(acc)
    plt.figure(figsize=(4.6,3.6))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Rank k")
    plt.ylabel("Explained variance")
    plt.title("Low-rank approximation curve")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    H, metas = load_matrix(args.json)
    if H.shape[0] == 0:
        print("No non-terminal entries with per-move data.")
        return

    # --- global mean heatmap (data-only) ---
    mean9 = nanmean2d(H)           # (9,)
    save_heatmap(vec2img(mean9), "Mean P(win) over all layouts", os.path.join(args.outdir, "mean_pwin.png"))

    # --- D4 irrep energy decomposition ---
    energies = {}
    for name,P in P_ir.items():
        energies[name] = project_energy(H, P)
    with open(os.path.join(args.outdir, "d4_irrep_energy.json"), "w", encoding="utf-8") as f:
        json.dump(energies, f, indent=2)
    # bar chart
    plt.figure(figsize=(5.0,3.2))
    ks = list(energies.keys()); vs = [energies[k] for k in ks]
    plt.bar(ks, vs)
    plt.title("Energy fraction by D4 irreps")
    plt.ylim(0,1); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "d4_irrep_energy.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- PCA / eigenpatterns ---
    mu, comps, explained = pca_with_nan(H, k=5)
    save_heatmap(vec2img(mu), "PCA mean (imputed)", os.path.join(args.outdir, "pca_mean.png"))
    for i in range(comps.shape[0]):
        # components can be positive/negative; rescale to [-1,1] display
        C = vec2img(comps[i])
        vmax = np.max(np.abs(C)) + 1e-12
        save_heatmap(C, f"PC{i+1} (explained {explained[i]*100:.1f}%)",
                     os.path.join(args.outdir, f"pc{i+1}.png"),
                     vmin=-vmax, vmax=+vmax, annotate=False)
    # low-rank curve
    rank_approx_curve(H, os.path.join(args.outdir, "lowrank_curve.png"))

    # --- Orbit stats (center / edges / corners) ---
    center_mask, edge_mask, corner_mask = orbit_masks()
    col_mean = np.nanmean(H, axis=0)
    X = np.where(np.isnan(H), col_mean[None,:], H)
    means = {
        "center_mean": float(np.mean(X[:, center_mask])),
        "edge_mean":   float(np.mean(X[:, edge_mask])),
        "corner_mean": float(np.mean(X[:, corner_mask])),
    }
    vars_ = {
        "center_var": float(np.var(X[:, center_mask])),
        "edge_var":   float(np.var(X[:, edge_mask])),
        "corner_var": float(np.var(X[:, corner_mask])),
    }
    with open(os.path.join(args.outdir, "orbit_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"means":means, "vars":vars_}, f, indent=2)

    # --- E-subspace principal directions (within E) ---
    # Stack projections of samples onto E, then SVD inside that subspace
    P_E = P_ir["E"]
    XE = (X - X.mean(axis=0, keepdims=True)) @ P_E          # (n,9)
    # Projected covariance SVD:
    Ue, Se, VTe = np.linalg.svd(XE, full_matrices=False)
    # Two leading orthonormal directions in E-subspace:
    E1 = vec2img(VTe[0])
    E2 = vec2img(VTe[1]) if VTe.shape[0] > 1 else np.zeros((3,3))
    v1 = np.max(np.abs(E1)) + 1e-12
    v2 = np.max(np.abs(E2)) + 1e-12
    save_heatmap(E1, "E-subspace dir1", os.path.join(args.outdir, "E_dir1.png"), vmin=-v1, vmax=+v1, annotate=False)
    save_heatmap(E2, "E-subspace dir2", os.path.join(args.outdir, "E_dir2.png"), vmin=-v2, vmax=+v2, annotate=False)

    # --- Also export a compact summary README ---
    with open(os.path.join(args.outdir, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("Math-patterns summary (data-only)\n")
        f.write("=================================\n")
        f.write(f"Samples: {X.shape[0]}\n")
        f.write("\n[D4 irrep energy fractions]\n")
        for k,v in energies.items():
            f.write(f"  {k}: {v:.3f}\n")
        f.write("\n[PCA explained variance]\n")
        for i,e in enumerate(explained, start=1):
            f.write(f"  PC{i}: {e:.3f}\n")
        f.write("\n[Orbit means]\n")
        for k,v in means.items(): f.write(f"  {k}: {v:.4f}\n")
        f.write("\n[Orbit variances]\n")
        for k,v in vars_.items(): f.write(f"  {k}: {v:.4f}\n")
        f.write("\nSee PNGs for eigenpatterns and E-subspace directions.\n")

    print("Done. Outputs in:", args.outdir)

if __name__ == "__main__":
    main()
