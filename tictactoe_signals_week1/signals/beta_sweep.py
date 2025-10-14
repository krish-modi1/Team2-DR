#!/usr/bin/env python3
import argparse, numpy as np, csv

def d4_orbit_ids(N:int):
    idxs = np.arange(N*N).reshape(N,N)
    orbits = []
    corners = [idxs[0,0], idxs[0,-1], idxs[-1,0], idxs[-1,-1]]
    orbits.append(corners)
    edges = []
    for i in range(N):
        for j in range(N):
            if (i in (0, N-1) or j in (0, N-1)) and (i,j) not in [(0,0),(0,N-1),(N-1,0),(N-1,N-1)]:
                edges.append(idxs[i,j])
    if edges: orbits.append(edges)
    if N%2==1: orbits.append([idxs[N//2, N//2]])
    return orbits

def row_profiles(N:int):
    idxs = np.arange(N*N).reshape(N,N); return [list(idxs[i,:]) for i in range(N)]
def col_profiles(N:int):
    idxs = np.arange(N*N).reshape(N,N); return [list(idxs[:,j]) for j in range(N)]
def diagonals(N:int):
    idxs = np.arange(N*N).reshape(N,N)
    main=[idxs[i,i] for i in range(N)]; anti=[idxs[i,N-1-i] for i in range(N)]
    return [main,anti]
def manhattan_shells(N:int):
    idxs = np.arange(N*N).reshape(N,N)
    cy=(N-1)//2; cx=(N-1)//2
    shells={}
    for y in range(N):
        for x in range(N):
            r=abs(y-cy)+abs(x-cx)
            shells.setdefault(r,[]).append(idxs[y,x])
    return [cells for _,cells in sorted(shells.items(), key=lambda kv: kv[0])]
def groups_to_matrix(groups,N:int):
    import numpy as np
    m=len(groups); A=np.zeros((m,N*N),dtype=np.float32)
    for g,cells in enumerate(groups):
        w=1.0/len(cells)
        for i in cells: A[g,i]=w
    return A
def build_A(N:int):
    import numpy as np
    mats=[
        groups_to_matrix(d4_orbit_ids(N),N),
        groups_to_matrix(row_profiles(N),N),
        groups_to_matrix(col_profiles(N),N),
        groups_to_matrix(diagonals(N),N),
        groups_to_matrix(manhattan_shells(N),N),
    ]
    return np.vstack(mats)
def inverse_mae(grid, lam:float=1e-6):
    import numpy as np
    N=grid.shape[0]; A=build_A(N); x=grid.reshape(-1); s=A@x
    At=A.T; M=At@A + lam*np.eye(At.shape[0],dtype=A.dtype)
    xhat=np.linalg.solve(M, At@s)
    recon=xhat.reshape(N,N)
    return float(np.mean(np.abs(recon-grid)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tables_npz", required=True)
    ap.add_argument("--beta_grid", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--out_csv", default="beta_sweep.csv")
    args=ap.parse_args()

    data=np.load(args.tables_npz, allow_pickle=True)
    P_win=data["P_win"]; P_draw=data["P_draw"]
    betas=[float(b) for b in args.beta_grid.split(",")]
    N,_,max_ply,roles=P_win.shape
    rows=[]
    best_overall=(1e9,None,None,None)

    for beta in betas:
        P_eff = P_win + beta*P_draw
        for ply in range(max_ply):
            for ridx,role in enumerate(["X","O"]):
                grid=P_eff[:,:,ply,ridx]
                mae=inverse_mae(grid)
                rows.append({"beta":beta,"ply":ply,"role":role,"mae":mae})
                if mae<best_overall[0]:
                    best_overall=(mae,beta,ply,role)

    with open(args.out_csv,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["beta","ply","role","mae"])
        w.writeheader(); w.writerows(rows)

    best_by_slice={}
    for r in rows:
        key=(r["ply"], r["role"])
        if key not in best_by_slice or r["mae"]<best_by_slice[key]["mae"]:
            best_by_slice[key]=r

    print("[Best per (ply,role)]")
    for k,v in sorted(best_by_slice.items()):
        print(f"ply={k[0]:2d} role={k[1]}  beta={v['beta']:.2f}  mae={v['mae']:.4f}")
    print("\n[Best overall] mae=%.4f  beta=%.2f  at ply=%d role=%s" % best_overall)
    print(f"[CSV] wrote {args.out_csv}")

if __name__=="__main__":
    main()
