#!/usr/bin/env python3
"""FFT analysis for state-index probability series by position.

Source data: `exhaustive-analysis-krish/state_position_plots/state_index_mapping.csv`
Each position/outcome column forms a 1D signal after sorting by state_index.

Outputs per position/outcome:
  - fft_spectrum_{position}_{outcome}.csv: frequency index, normalized frequency, magnitude
  - fft_spectrum_{position}_{outcome}.png: log-magnitude plot vs frequency index
Aggregate:
  - dominant_frequencies.csv listing top components per series (excluding DC)

Example:
  python tools/fft_position_series_analysis.py \
    --csv exhaustive-analysis-krish/state_position_plots/state_index_mapping.csv \
    --outdir tools/fft_position_out
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTCOME_COLUMNS = ["p_x_win", "p_o_win", "p_draw"]


def load_position_series(df: pd.DataFrame, position: int, column: str) -> np.ndarray:
    """Return the probability series for a position/outcome sorted by state_index."""
    subset = df[df["position"] == position].sort_values("state_index")
    seq = subset[column].to_numpy(dtype=float)
    if seq.size == 0:
        raise ValueError(f"No data for position {position} column {column}")
    return np.nan_to_num(seq, nan=np.nanmean(seq) if np.isnan(seq).any() else 0.0)


def compute_fft(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, magnitude) for rfft of the demeaned series."""
    demeaned = series - np.mean(series)
    fft_vals = np.fft.rfft(demeaned)
    mags = np.abs(fft_vals) / series.size
    freqs = np.fft.rfftfreq(series.size, d=1.0)
    return freqs, mags


def save_spectrum_csv(freqs: np.ndarray, mags: np.ndarray, out_csv: Path) -> None:
    df = pd.DataFrame({
        "frequency_index": np.arange(freqs.size),
        "normalized_frequency": freqs,
        "magnitude": mags,
        "log_magnitude": np.log1p(mags)
    })
    df.to_csv(out_csv, index=False)


def plot_spectrum(freqs: np.ndarray, mags: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(6, 3.4), dpi=160)
    plt.plot(np.arange(freqs.size), np.log1p(mags), color="#336699", linewidth=1.4)
    plt.scatter(np.arange(freqs.size), np.log1p(mags), color="#6699cc", s=10)
    plt.title(title, fontsize=11)
    plt.xlabel("Frequency index")
    plt.ylabel("log(1 + magnitude)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def summarize_peaks(freqs: np.ndarray, mags: np.ndarray, topk: int = 5) -> List[Dict[str, float]]:
    """Return list of top-k frequency components excluding DC."""
    if mags.size <= 1:
        return []
    idxs = np.argsort(mags[1:])[::-1] + 1  # skip DC at index 0
    top_entries = []
    for idx in idxs[:topk]:
        top_entries.append({
            "frequency_index": int(idx),
            "normalized_frequency": float(freqs[idx]),
            "magnitude": float(mags[idx])
        })
    return top_entries


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def analyze_positions(csv_path: Path, outdir: Path, positions: List[int]) -> None:
    df = pd.read_csv(csv_path)
    ensure_outdir(outdir)

    summary_rows: List[Dict[str, float]] = []

    for pos in positions:
        for col in OUTCOME_COLUMNS:
            series = load_position_series(df, pos, col)
            freqs, mags = compute_fft(series)

            base = f"position{pos}_{col}"
            save_spectrum_csv(freqs, mags, outdir / f"fft_spectrum_{base}.csv")
            plot_spectrum(freqs, mags, outdir / f"fft_spectrum_{base}.png",
                          title=f"Position {pos} {col} | N={series.size}")

            peaks = summarize_peaks(freqs, mags)
            for rank, peak in enumerate(peaks, start=1):
                summary_rows.append({
                    "position": pos,
                    "outcome": col,
                    "rank": rank,
                    "frequency_index": peak["frequency_index"],
                    "normalized_frequency": peak["normalized_frequency"],
                    "magnitude": peak["magnitude"],
                    "series_length": series.size
                })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(outdir / "dominant_frequencies.csv", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FFT analysis for position probability trends")
    ap.add_argument("--csv", default="exhaustive-analysis-krish/state_position_plots/state_index_mapping.csv",
                    help="Input CSV with columns position,state_index,p_x_win,p_o_win,p_draw")
    ap.add_argument("--positions", nargs="*", type=int,
                    help="Optional subset of board positions (0-8). Default: all.")
    ap.add_argument("--outdir", default="tools/fft_position_out",
                    help="Directory to store spectra and summaries")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.positions:
        positions = args.positions
    else:
        positions = list(range(9))

    analyze_positions(csv_path, Path(args.outdir), positions)


if __name__ == "__main__":
    main()
