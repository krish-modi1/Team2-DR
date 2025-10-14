#!/usr/bin/env python3
"""
signals/extract_signals.py
Compute 1D "sound-wave" signals from a 2D grid and provide a simple inverse (signals -> grid).
Families:
- D4 orbits (center / edges / corners)
- Row/column profiles (N + N curves)
- Main & anti diagonals (2 curves)
- Concentric rings / Manhattan shells (r = 0..r_max)
We concatenate all signals into one vector s = A @ vec(grid). For inverse, use pinv(A).
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def d4_orbit_ids(N:int):
    """
    Returns a list of index lists: [orbit0_cells, orbit1_cells, ...].
    Orbits: center (if N odd), edges, corners, and remaining by D4 symmetry.
    For Week 1 we implement three coarse groups for odd N: center / edges / corners.
    For even N, we approximate groups by: inner-edges, outer-edges, corners.
    """
    idxs = np.arange(N*N).reshape(N,N)
    orbits = []
    # corners
    corners = [idxs[0,0], idxs[0,-1], idxs[-1,0], idxs[-1,-1]]
    orbits.append(corners)
    # edges (non-corner)
    edges = []
    for i in range(N):
        for j in range(N):
            if (i in (0, N-1) or j in (0, N-1)) and (i,j) not in [(0,0),(0,N-1),(N-1,0),(N-1,N-1)]:
                edges.append(idxs[i,j])
    if edges:
        orbits.append(edges)
    # center (if odd)
    if N % 2 == 1:
        orbits.append([idxs[N//2, N//2]])
    return orbits

def row_profiles(N:int):
    idxs = np.arange(N*N).reshape(N,N)
    return [list(idxs[i,:]) for i in range(N)]

def col_profiles(N:int):
    idxs = np.arange(N*N).reshape(N,N)
    return [list(idxs[:,j]) for j in range(N)]

def diagonals(N:int):
    idxs = np.arange(N*N).reshape(N,N)
    main = [idxs[i,i] for i in range(N)]
    anti = [idxs[i,N-1-i] for i in range(N)]
    return [main, anti]

def manhattan_shells(N:int):
    idxs = np.arange(N*N).reshape(N,N)
    cy = (N-1)//2; cx = (N-1)//2
    shells = {}
    for y in range(N):
        for x in range(N):
            r = abs(y-cy)+abs(x-cx)
            shells.setdefault(r, []).append(idxs[y,x])
    return [cells for r,cells in sorted(shells.items(), key=lambda kv: kv[0])]

def groups_to_matrix(groups, N:int):
    """
    Build a measurement matrix G of shape [len(groups), N*N] where each row averages a group.
    Entry G[g, i] = 1/|group| if i in group else 0.
    """
    m = len(groups); A = np.zeros((m, N*N), dtype=np.float32)
    for g, cells in enumerate(groups):
        w = 1.0/len(cells)
        for i in cells:
            A[g, i] = w
    return A

def build_design_matrix(N:int):
    # Construct each family matrix, then stack vertically
    mats = []
    mats.append(groups_to_matrix(d4_orbit_ids(N), N))
    mats.append(groups_to_matrix(row_profiles(N), N))
    mats.append(groups_to_matrix(col_profiles(N), N))
    mats.append(groups_to_matrix(diagonals(N), N))
    mats.append(groups_to_matrix(manhattan_shells(N), N))
    A = np.vstack(mats)
    return A

def forward_signals(grid:np.ndarray) -> (np.ndarray, np.ndarray):
    """
    grid: (N,N)
    returns: s (signals concat), A (design matrix)
    """
    N = grid.shape[0]
    A = build_design_matrix(N)
    x = grid.reshape(-1)
    s = A @ x
    return s, A

def inverse_from_signals(s:np.ndarray, A:np.ndarray, N:int, lam:float=1e-6):
    """
    Solve x_hat = pinv(A) s with small Tikhonov regularization: (A^T A + lam I)^{-1} A^T s
    """
    At = A.T
    M = At @ A + lam * np.eye(At.shape[0], dtype=A.dtype)
    x_hat = np.linalg.solve(M, At @ s)
    return x_hat.reshape(N,N)

def plot_family_curves(s:np.ndarray, N:int, out_dir:Path, stem:str):
    """
    s is concatenated as [D4 | rows | cols | diags | shells]. We split to plot line curves.
    """
    idx = 0
    parts = {}
    # D4
    d4_len = len(d4_orbit_ids(N)); parts["d4"] = s[idx:idx+d4_len]; idx += d4_len
    # rows
    parts["rows"] = s[idx:idx+N]; idx += N
    # cols
    parts["cols"] = s[idx:idx+N]; idx += N
    # diags
    parts["diags"] = s[idx:idx+2]; idx += 2
    # shells
    shells_len = len(manhattan_shells(N)); parts["shells"] = s[idx:idx+shells_len]; idx += shells_len

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in parts.items():
        plt.figure()
        plt.plot(np.arange(len(arr)), arr, marker='o')
        plt.title(f"{name} — N={N}")
        plt.xlabel("index")
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_{name}.png", dpi=160)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables_npz", type=str, required=True, help="path to npz from build_tables.py")
    ap.add_argument("--ply", type=int, default=0)
    ap.add_argument("--role", type=str, default="X", choices=["X","O"])
    ap.add_argument("--out_dir", type=str, default="plots")
    ap.add_argument("--beta_override", type=float, default=None, help="if set, recompute P_eff with this beta")
    ap.add_argument("--eval_inverse", action="store_true", help="compute inverse reconstruction MAE on 3x3")
    args = ap.parse_args()

    data = np.load(args.tables_npz, allow_pickle=True)
    P_eff = data["P_eff"]
    beta = data["beta"].item() if "beta" in data else None
    if args.beta_override is not None:
        P_eff = data["P_win"] + args.beta_override * data["P_draw"]
        beta = args.beta_override

    # select slice
    N = P_eff.shape[0]
    ridx = 0 if args.role=="X" else 1
    grid = P_eff[:,:,args.ply, ridx]

    out_dir = Path(args.out_dir)
    s, A = forward_signals(grid)
    plot_family_curves(s, N, out_dir, stem=f"N{N}_ply{args.ply}_{args.role}")

    if args.eval_inverse and N==3:
        recon = inverse_from_signals(s, A, N, lam=1e-6)
        mae = np.mean(np.abs(recon - grid))
        print(f"[inverse] N=3 MAE={mae:.4f}  (beta={beta})")
        np.save(out_dir / f"recon_N{N}_ply{args.ply}_{args.role}.npy", recon)

if __name__ == "__main__":
    main()
