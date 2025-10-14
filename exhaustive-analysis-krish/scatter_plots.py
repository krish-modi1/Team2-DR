#!/usr/bin/env python3
"""
Generate separate scatter plots for P(X wins), P(O wins), and P(draws)
against game layer from a CSV file.

Usage:
    python3 scatter_plots.py probabilities.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_scatter(df, y_col, color, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['layer'], df[y_col], color=color, alpha=0.6, edgecolor='k')
    plt.xlabel('Layer (Game Depth)', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.title(f'{title} vs Game Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(-0.5, df['layer'].max() + 0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved plot: {output_path}')

def main():
    if len(sys.argv) < 2:
        print('Usage: python scatter_plots.py probabilities.csv')
        sys.exit(1)

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # Ensure output directory exists
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # Plot definitions
    plots = [
        ('P(X wins)', 'red', 'P(X wins)', os.path.join(output_dir, 'scatter_PX_wins.png')),
        ('P(O wins)', 'blue', 'P(O wins)', os.path.join(output_dir, 'scatter_PO_wins.png')),
        ('P(draws)', 'green', 'P(draws)', os.path.join(output_dir, 'scatter_P_draws.png')),
    ]

    for col, color, title, out_path in plots:
        plot_scatter(df, col, color, title, out_path)

if __name__ == '__main__':
    main()
