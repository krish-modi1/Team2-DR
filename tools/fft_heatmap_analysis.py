#!/usr/bin/env python3
"""
fft_heatmap_analysis.py

Load a heatmap (CSV with row/col/value) or an image (PNG/JPG), compute 2D FFT,
save the shifted magnitude spectrum image and a radial frequency profile CSV.

Usage examples:
  python tools/fft_heatmap_analysis.py --input minimax/tictactoe_3x3_first_move_stats.csv --value-col rand_X_win_prob --outdir tools/fft_out
  python tools/fft_heatmap_analysis.py --input path/to/heatmap_3x3_random_Xwin.png --outdir tools/fft_out

Outputs:
  - spectrum.png      : log-magnitude of shifted 2D FFT
  - spectrum.npy      : raw (shifted) complex FFT array
  - radial_profile.csv: two columns freq_radius, mean_magnitude

"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import math


def load_csv_matrix(path: Path, value_col: str = None):
    df = pd.read_csv(path)
    # try to detect row/col columns
    if {"row","col","move_index"}.intersection(df.columns):
        if "row" in df.columns and "col" in df.columns:
            n = int(math.sqrt(len(df)))
            if value_col is None:
                # pick first numeric column that's not row/col
                candidates = [c for c in df.columns if c not in ("row","col")]
                value_col = candidates[0]
            mat = np.full((n,n), np.nan, dtype=float)
            for _, r in df.iterrows():
                rr = int(r["row"]) ; cc = int(r["col"]) ; mat[rr,cc] = float(r[value_col])
            return mat, value_col
        elif "move_index" in df.columns:
            L = len(df)
            n = int(math.sqrt(L))
            if value_col is None:
                candidates = [c for c in df.columns if c != "move_index"]
                value_col = candidates[0]
            vals = df[value_col].to_numpy().astype(float)
            return vals.reshape(n,n), value_col
    # fallback: try reshape by length
    arr = df.select_dtypes(include=[float,int]).iloc[:,0].to_numpy()
    n = int(math.sqrt(len(arr)))
    return arr.reshape(n,n), (value_col or df.columns[0])


def load_image_matrix(path: Path):
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=float)
    # normalize to 0..1
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
    return arr


def compute_fft(mat: np.ndarray):
    # Replace NaNs
    mat = np.array(mat, dtype=float)
    if np.isnan(mat).any():
        mat = np.nan_to_num(mat, nan=0.0)
    # subtract mean to reduce DC dominance
    mat = mat - np.mean(mat)
    F = np.fft.fft2(mat)
    Fsh = np.fft.fftshift(F)
    mag = np.abs(Fsh)
    return Fsh, mag


def plot_and_save_spectrum(mag: np.ndarray, out_png: Path, cmap: str = "inferno"):
    plt.figure(figsize=(5,5), dpi=160)
    # use log scale for visibility
    img = np.log1p(mag)
    plt.imshow(img, origin='lower', cmap=cmap, aspect='equal')
    plt.axis('off')
    plt.title('log(1+|FFT|)')
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def radial_profile(mag: np.ndarray, center=None, nbins=100):
    h, w = mag.shape
    if center is None:
        cx, cy = (w//2, h//2)
    else:
        cx, cy = center
    y, x = np.indices((h, w))
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    r_flat = r.flatten()
    mag_flat = mag.flatten()
    maxr = r_flat.max()
    bins = np.linspace(0, maxr, nbins+1)
    inds = np.digitize(r_flat, bins)
    radial_mean = []
    radial_r = []
    for i in range(1, len(bins)):
        sel = (inds == i)
        if not np.any(sel):
            radial_mean.append(0.0)
        else:
            radial_mean.append(mag_flat[sel].mean())
        radial_r.append((bins[i-1] + bins[i]) / 2.0)
    return np.array(radial_r), np.array(radial_mean)


def main():
    ap = argparse.ArgumentParser(description="FFT analysis of heatmaps (CSV or image)")
    ap.add_argument("--input", required=True, help="input file: CSV (row/col/value) or PNG/JPG image")
    ap.add_argument("--value-col", default=None, help="(for CSV) column name to use as value; default: auto-detect")
    ap.add_argument("--outdir", default="tools/fft_out", help="output directory")
    ap.add_argument("--nbins", type=int, default=100, help="radial profile bins")
    args = ap.parse_args()

    p = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if p.suffix.lower() in ('.csv',):
        mat, vcol = load_csv_matrix(p, args.value_col)
        print(f"Loaded CSV matrix, value_col={vcol}, shape={mat.shape}")
    else:
        mat = load_image_matrix(p)
        print(f"Loaded image matrix, shape={mat.shape}")

    Fsh, mag = compute_fft(mat)

    np.save(outdir / "spectrum_shifted.npy", Fsh)
    plot_and_save_spectrum(mag, outdir / "spectrum.png")

    r, m = radial_profile(mag, nbins=args.nbins)
    pd.DataFrame({"radius": r, "mean_magnitude": m}).to_csv(outdir / "radial_profile.csv", index=False)

    print("Wrote:")
    print((outdir / "spectrum.png").resolve())
    print((outdir / "spectrum_shifted.npy").resolve())
    print((outdir / "radial_profile.csv").resolve())


if __name__ == '__main__':
    main()
