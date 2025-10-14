#!/usr/bin/env python3
"""
tables/build_tables.py
Week 1 pilot: normalize artifacts into tensors P_win[N×N, ply, role] and P_draw[N×N, ply, role].
Role ∈ {"X","O"} encoded as role index 0/1.
Also export P_eff = P_win + beta * P_draw.
"""

import argparse
import json
import numpy as np
from pathlib import Path

ROLES = {"X":0, "O":1}

def load_maps(input_path:str, N:int, max_ply:int):
    """
    Expected input JSON (example schema):
    {
      "win": { "X": { "ply_0": [[...NxN...], ...], "ply_1": [...], ... },
               "O": { "ply_0": [[...NxN...], ...], ... } },
      "draw":{ "X": { "ply_0": [[...NxN...], ...], ... },
               "O": { "ply_0": [[...NxN...], ...], ... } }
    }
    Each ply entry may be a single NxN matrix or a list of length 1; we accept both.
    If you don't have real data yet, pass --mock to generate smooth mock tables.
    """
    with open(input_path, "r") as f:
        raw = json.load(f)
    P_win = np.zeros((N, N, max_ply, 2), dtype=np.float32)
    P_draw= np.zeros((N, N, max_ply, 2), dtype=np.float32)

    for kind, target in [("win", P_win), ("draw", P_draw)]:
        for role_name, rdict in raw[kind].items():
            ridx = ROLES[role_name]
            for p in range(max_ply):
                key = f"ply_{p}"
                grid = rdict.get(key, None)
                if grid is None:
                    # If missing, carry last known or zeros
                    grid = target[:,:,max(0,p-1),ridx] if p>0 else np.zeros((N,N), dtype=np.float32)
                elif isinstance(grid, list) and isinstance(grid[0], list):
                    grid = np.array(grid, dtype=np.float32)
                else:
                    grid = np.array(grid, dtype=np.float32)
                target[:,:,p,ridx] = grid
    return P_win, P_draw

def make_mock(N:int, max_ply:int, seed:int=42):
    """
    Generates smooth mock maps for quick plumbing & plotting.
    """
    rng = np.random.default_rng(seed)
    # Base radial bowl + role-dependent bias + ply decay
    yy, xx = np.mgrid[0:N, 0:N]
    cx = (N-1)/2.0; cy = (N-1)/2.0
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2)/(np.sqrt(2)*cx if N>1 else 1.0)
    base = 0.6 - 0.3*r
    P_win = np.zeros((N,N,max_ply,2), dtype=np.float32)
    P_draw= np.zeros((N,N,max_ply,2), dtype=np.float32)
    for p in range(max_ply):
        decay = np.exp(-0.15*p)
        noise = 0.02*rng.normal(size=(N,N))
        # X a bit corner-loving, O a bit edge-loving
        bias_X = 0.04*((xx%2==0)&(yy%2==0)) - 0.02*((xx==cx)|(yy==cy))
        bias_O = -bias_X
        P_win[:,:,p,0] = np.clip(base + bias_X, 0, 1)*decay + noise
        P_win[:,:,p,1] = np.clip(base + bias_O, 0, 1)*decay + noise
        # Draw probability complementary-ish + noise
        P_draw[:,:,p,0] = np.clip(0.3 + 0.2*r, 0, 1)*(1.0 - 0.5*decay) + 0.02*rng.normal(size=(N,N))
        P_draw[:,:,p,1] = P_draw[:,:,p,0]  # symmetric for mock
    P_win = np.clip(P_win, 0, 1); P_draw = np.clip(P_draw, 0, 1)
    return P_win, P_draw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=3, help="board size")
    ap.add_argument("--max_ply", type=int, default=5, help="max ply depth (inclusive upper bound is max_ply-1)")
    ap.add_argument("--beta", type=float, default=0.5, help="draw weight β in P_eff = P_win + β * P_draw")
    ap.add_argument("--input_json", type=str, default=None, help="path to JSON maps; omit when using --mock")
    ap.add_argument("--mock", action="store_true", help="use internally generated mock maps")
    ap.add_argument("--out_dir", type=str, default="tables_out", help="output folder for npz tensors")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mock:
        P_win, P_draw = make_mock(args.N, args.max_ply)
    else:
        assert args.input_json is not None, "Provide --input_json or use --mock"
        P_win, P_draw = load_maps(args.input_json, args.N, args.max_ply)

    P_eff = P_win + args.beta * P_draw

    np.savez_compressed(Path(args.out_dir) / f"tables_N{args.N}.npz",
                        P_win=P_win, P_draw=P_draw, P_eff=P_eff, beta=args.beta)
    print(f"[OK] Saved tensors to {Path(args.out_dir) / f'tables_N{args.N}.npz'}")

if __name__ == "__main__":
    main()
