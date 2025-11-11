# FFT Pattern Analysis Report



## 1. Scope and Data

- **Primary focus**: one-dimensional FFTs generated in `tools/fft_position_out/` from `state_index_mapping.csv` (position-vs-probability scatter trends).

- **Objective**: quantify how smooth or oscillatory each board position’s probability series is, identify symmetry patterns, and highlight any notable higher-frequency behaviour.

## 2. Method Summary

| Dataset | Pre-processing | FFT variant | Artefacts |
| --- | --- | --- | --- |
| Position trend series (`state_index_mapping.csv`) | sort by `state_index`, fill NaNs with column mean, subtract global mean | 1D rFFT (`numpy.fft.rfft`) | `fft_spectrum_position{k}_{outcome}.{csv,png}`, `dominant_frequencies.csv` in `tools/fft_position_out/` |
| Heatmap images (layer 0, P_X / P_O / P_draw) | grayscale, normalise to [0,1], mean removal | 2D FFT (`numpy.fft.fftshift`) | `spectrum.png`, `spectrum_shifted.npy`, `radial_profile.csv` in `tools/fft_out_fuyao/` |

For reporting we rely mainly on the position-series outputs; heatmap spectra serve as supporting evidence about spatial smoothness.

## 3. Key Statistics from `dominant_frequencies.csv`

The table below lists the magnitude of the strongest non-zero frequency component (frequency index 1) for each position and outcome. Values are normalised by series length.

| Position | Type | `p_x_win` | `p_o_win` | `p_draw` | Series length |
| --- | --- | --- | --- | --- | --- |
| 0 | Corner | 0.1652 | 0.1060 | 0.0597 | 1818 |
| 1 | Edge   | 0.1542 | 0.0954 | 0.0591 | 1753 |
| 2 | Corner | 0.1652 | 0.1045 | 0.0612 | 1818 |
| 3 | Edge   | 0.1542 | 0.0953 | 0.0592 | 1753 |
| 4 | Center | **0.1813** | **0.1270** | 0.0577 | 1883 |
| 5 | Edge   | 0.1542 | 0.0969 | 0.0577 | 1753 |
| 6 | Corner | 0.1652 | 0.1050 | 0.0607 | 1818 |
| 7 | Edge   | 0.1542 | 0.0976 | 0.0570 | 1753 |
| 8 | Corner | 0.1652 | 0.1053 | 0.0606 | 1818 |

**Observations**

1. **Single dominant mode.** Every series is governed by the first non-zero frequency (index 1; normalised frequency ≈5×10⁻⁴). After mean subtraction, this component describes the broad rise/fall visible in the scatter plots. Higher harmonics are at least 2–3× smaller.
2. **Symmetry captured in magnitudes.** Corners (0,2,6,8) share virtually identical spectra; edges (1,3,5,7) likewise cluster tightly. This matches tic-tac-toe rotational/reflection symmetry and confirms the dataset respects it numerically.
3. **Centre stands out.** Position 4 has a markedly stronger low-frequency amplitude (≈10–15 % higher than others), meaning the centre’s win/loss probability shifts more decisively across the state catalogue. This quantitative gap backs gameplay intuition about the centre’s strategic leverage.
4. **Draw series are flatter.** `p_draw` magnitudes are roughly half of the corresponding win magnitudes, indicating draw probabilities vary more gently with state ordering.

## 4. Higher-Frequency Behaviour

Using the top-5 entries in `dominant_frequencies.csv` per series:

- Peak indices beyond 1 rarely exceed magnitude 0.04 (≈25 % of the primary component).
- Draw sequences occasionally show bumps around indices 8–50 (normalised frequency ≈0.004–0.028). These correspond to small oscillations where alternating states toggle draw likelihood, but they do not dominate.
- A few edge positions register notable components near indices 11–13 or 31–32, hinting at layer-specific structures (e.g., when certain mid-game layers introduce alternating advantages). Follow-up could tag these states to inspect concrete board patterns.

## 5. Visual Evidence

Log-magnitude plots (`fft_spectrum_position*k*_{outcome}.png`) illustrate the steep decay after the first spike. Example: `tools/fft_position_out/fft_spectrum_position8_p_draw.png` shows the index 1 peak towering over a near-flat noise floor, reinforcing the “smooth trend” narrative.

For comparison, the 2D spectra (`tools/fft_out_fuyao/`) also concentrate energy in low frequencies. Radial profiles at radius ≈4 px contain 6–9 k mean magnitude, dropping below 800 by radius ≈35. This indicates that when we eventually analyse raw probability grids, most information will still live in coarse spatial structures.

## 6. Interpretation for Gameplay Analytics

- The FFT confirms the position trends can be approximated by very low-order models (e.g., cubic splines or even single sine components). This could compress the dataset or simplify visualisations without losing fidelity.
- Symmetric positions are spectrally indistinguishable, so future plots can group corners/edges/centre to reduce redundancy.
- Detecting deviations from symmetry (e.g., when using biased policies or limited rollouts) should be easy: any significant additional peaks or magnitude disparities would signal asymmetry.




