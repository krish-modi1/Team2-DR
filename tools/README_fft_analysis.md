# FFT Heatmap Analysis

This small utility loads a heatmap (CSV or image), computes a 2D Fourier transform, and
exports a log-magnitude spectrum image plus a radial frequency profile.

Usage examples:

```powershell
python tools/fft_heatmap_analysis.py --input minimax/tictactoe_3x3_first_move_stats.csv --value-col rand_X_win_prob --outdir tools/fft_out
python tools/fft_heatmap_analysis.py --input minimax/heatmap_3x3_random_Xwin.png --outdir tools/fft_out
```

Outputs will be written to the given outdir:
- `spectrum.png` - log(1+|FFT|) visualization
- `spectrum_shifted.npy` - raw shifted FFT complex array
- `radial_profile.csv` - radial mean magnitude per frequency radius

Notes:
- For CSV inputs the script expects either `row`/`col` columns or a `move_index` column.
- For image inputs the script converts to grayscale and normalizes to [0,1].

## Position Trend FFT (1D)

To analyze the “position vs probability” scatter series (see
`exhaustive-analysis-krish/state_position_plots/position_*_*.png`), use:

```powershell
python tools/fft_position_series_analysis.py --csv exhaustive-analysis-krish/state_position_plots/state_index_mapping.csv --outdir tools/fft_position_out
```

Each position (0-8) and outcome (`p_x_win`, `p_o_win`, `p_draw`) gets:

- `fft_spectrum_position{pos}_{outcome}.csv`
- `fft_spectrum_position{pos}_{outcome}.png`

The script also produces `dominant_frequencies.csv` summarizing top components (excluding DC).
