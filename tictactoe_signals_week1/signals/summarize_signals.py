#!/usr/bin/env python3
import argparse, numpy as np, csv

def d4_orbit_ids(N:int):
    idxs = np.arange(N*N).reshape(N,N)
    orbits = []
    corners = [idxs[0,0], idxs[0,-1], idxs[-1,0], idxs[-1,-1]]; orbits.append(corners)
    edges = []
    for i in range(N):
        for j in range(N):
            if (i in (0,N-1) or j in (N-1,0)) and (i,j) not in [(0,0),(0,N-1),(N-1,0),(N-1,N-1)]:
                edges.append(idxs[i,j])
    if edges: orbits.append(edges)
    if N%2==1: orbits.append([idxs[N//2,N//2]])
    return orbits
def row_profiles(N): idxs=np.arange(N*N).reshape(N,N); return [list(idxs[i,:]) for i in range(N)]
def col_profiles(N): idxs=np.arange(N*N).reshape(N,N); return [list(idxs[:,j]) for j in range(N)]
def diagonals(N):
    idxs=np.arange(N*N).reshape(N,N)
    main=[idxs[i,i] for i in range(N)]; anti=[idxs[i,N-1-i] for i in range(N)]
    return [main,anti]
def manhattan_shells(N):
    idxs=np.arange(N*N).reshape(N,N)
    cy=(N-1)//2; cx=(N-1)//2
    shells={}
    for y in range(N):
        for x in range(N):
            r=abs(y-cy)+abs(x-cx)
            shells.setdefault(r,[]).append(idxs[y,x])
    return [cells for _,cells in sorted(shells.items(), key=lambda kv: kv[0])]
def groups_to_matrix(groups,N):
    m=len(groups); A=np.zeros((m,N*N),dtype=np.float32)
    for g,cells in enumerate(groups):
        w=1.0/len(cells)
        for i in cells: A[g,i]=w
    return A
def split_signals(s,N):
    idx=0; d={}
    d4_len=len(d4_orbit_ids(N)); d["d4"]=s[idx:idx+d4_len]; idx+=d4_len
    d["rows"]=s[idx:idx+N]; idx+=N
    d["cols"]=s[idx:idx+N]; idx+=N
    d["diags"]=s[idx:idx+2]; idx+=2
    shells_len=len(manhattan_shells(N)); d["shells"]=s[idx:idx+shells_len]; idx+=shells_len
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tables_npz", required=True)
    ap.add_argument("--out_csv", default="signals_summary.csv")
    ap.add_argument("--beta_override", type=float, default=None)
    args=ap.parse_args()

    data=np.load(args.tables_npz, allow_pickle=True)
    if args.beta_override is None:
        P_eff=data["P_eff"]
    else:
        P_eff=data["P_win"] + args.beta_override*data["P_draw"]

    N,_,max_ply,roles=P_eff.shape
    A=np.vstack([
        groups_to_matrix(d4_orbit_ids(N),N),
        groups_to_matrix(row_profiles(N),N),
        groups_to_matrix(col_profiles(N),N),
        groups_to_matrix(diagonals(N),N),
        groups_to_matrix(manhattan_shells(N),N),
    ])

    shells_len=len(manhattan_shells(N))

    with open(args.out_csv,"w",newline="") as f:
        w=csv.writer(f)
        header=["ply","role",
                "d4_corner","d4_edges"]+(["d4_center"] if N%2==1 else []) + \
                [f"row_{i}" for i in range(N)] + [f"col_{j}" for j in range(N)] + \
                ["diag_main","diag_anti"] + [f"shell_{r}" for r in range(shells_len)] + \
                ["center_minus_edges","edges_minus_corners","shells_slope"]
        w.writerow(header)

        for ply in range(max_ply):
            for ridx,role in enumerate(["X","O"]):
                grid=P_eff[:,:,ply,ridx]; x=grid.reshape(-1); s=A@x
                parts=split_signals(s,N)
                if N%2==1:
                    corner, edges, center = parts["d4"][0], parts["d4"][1], parts["d4"][2]
                    center_minus_edges = center - edges
                    edges_minus_corners = edges - corner
                else:
                    # 偶数 N 的 D4 只有 corner/edge，没有 center
                    center_minus_edges = 0.0
                    edges_minus_corners = 0.0

                shells = parts["shells"]
                xs = np.arange(len(shells))
                slope = float(np.polyfit(xs, shells, 1)[0])

                row = [ply, role] + \
                    ([corner, edges, center] if N%2==1 else [parts["d4"][0], parts["d4"][1]]) + \
                    list(parts["rows"]) + list(parts["cols"]) + \
                    list(parts["diags"]) + list(shells) + \
                    [center_minus_edges, edges_minus_corners, slope]

                w.writerow(row)

    print(f"[OK] wrote {args.out_csv}")

if __name__=="__main__":
    main()
