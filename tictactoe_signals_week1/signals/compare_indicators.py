# signals/compare_indicators.py
import pandas as pd
from pathlib import Path

n3 = pd.read_csv("out_3x3_full/signals_summary_beta0.csv")
n4 = pd.read_csv("out_4x4_shallow/signals_summary_beta0.csv")

def summarize(df, name):
    bins = [(0,3,"early"), (4,6,"mid"), (7,9,"late")]
    out = []
    for lo,hi,label in bins:
        sub = df[(df.ply>=lo)&(df.ply<=hi)]
        if sub.empty: continue
        g = sub.groupby("role")[["center_minus_edges","edges_minus_corners","shells_slope"]].mean()
        g["phase"] = label
        g["board"] = name
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

s3 = summarize(n3,"N3")
s4 = summarize(n4,"N4")
both = pd.concat([s3,s4], ignore_index=True)
print(both.to_string(index=False))
