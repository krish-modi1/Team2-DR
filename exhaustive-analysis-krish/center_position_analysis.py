"""
Center Position Analysis: Exhaustive Tic-Tac-Toe Probabilities
Generates line plots showing P(X wins) for all game paths maintaining X at position 4.

Sorting methodology:
- Layer 2: All states sorted by P(X wins) ascending
- Layer 3+: States grouped by parent, within-group sorted by P(X wins) ascending
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_and_prepare_data(csv_path='ttt_3x3_exhaustive_probabilities.csv'):
    """Load exhaustive probabilities and prepare center position analysis"""
    df = pd.read_csv(csv_path)

    center_df = df[df['state'].str[4] == 'X'].copy().reset_index(drop=True)

    print(f"Loaded {len(center_df)} center position states")

    return df, center_df

def build_sorted_plot_data(df, center_df):
    """Build plot data with proper sorting by layer"""

    plot_data = []

    # Layer 0
    layer_0 = df[df['layer'] == 0].iloc[0]
    plot_data.append({
        'layer': 0,
        'state': '.....',
        'px': layer_0['P(X wins)'],
        'x_order': 0
    })

    # Layer 1
    layer_1 = center_df[center_df['layer'] == 1]
    if len(layer_1) > 0:
        plot_data.append({
            'layer': 1,
            'state': layer_1.iloc[0]['state'],
            'px': layer_1.iloc[0]['P(X wins)'],
            'x_order': 0
        })

    # Layer 2: Sort by P(X wins) ascending
    layer_2 = center_df[center_df['layer'] == 2].sort_values('P(X wins)').reset_index(drop=True)
    for idx, (_, row) in enumerate(layer_2.iterrows()):
        plot_data.append({
            'layer': 2,
            'state': row['state'],
            'px': row['P(X wins)'],
            'x_order': idx
        })

    # Layers 3+: Group by parent, sort within groups
    for layer in range(3, 9):
        layer_data = center_df[center_df['layer'] == layer].copy()
        if len(layer_data) == 0:
            break

        parent_layer = layer - 1
        parent_plot_data = [d for d in plot_data if d['layer'] == parent_layer]
        parent_states_ordered = [d['state'] for d in sorted(parent_plot_data, key=lambda x: x['x_order'])]

        parent_to_children = defaultdict(list)

        for _, child_row in layer_data.iterrows():
            child_state = child_row['state']
            child_board = [1 if c == 'X' else (-1 if c == 'O' else 0) for c in child_state]

            for parent_state in parent_states_ordered:
                parent_board = [1 if c == 'X' else (-1 if c == 'O' else 0) for c in parent_state]

                is_valid_child = False
                for pos in range(9):
                    if child_board[pos] != 0:
                        test_board = child_board.copy()
                        test_board[pos] = 0
                        if test_board == parent_board:
                            is_valid_child = True
                            break

                if is_valid_child:
                    parent_to_children[parent_state].append({
                        'state': child_state,
                        'px': child_row['P(X wins)']
                    })
                    break

        x_order_counter = 0
        for parent_state in parent_states_ordered:
            children = parent_to_children.get(parent_state, [])
            children.sort(key=lambda x: x['px'])

            for child in children:
                plot_data.append({
                    'layer': layer,
                    'state': child['state'],
                    'px': child['px'],
                    'x_order': x_order_counter
                })
                x_order_counter += 1

    return pd.DataFrame(plot_data)

def create_line_plots(plot_df, output_file='center_position_lines.png'):
    """Create 3x3 grid of line plots (not histograms)"""

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Center Position Opening: P(X wins) by Layer', fontsize=14, fontweight='bold')

    axes_flat = axes.flatten()

    for plot_idx in range(9):
        ax = axes_flat[plot_idx]
        layer = plot_idx

        layer_plot_data = plot_df[plot_df['layer'] == layer].sort_values('x_order')

        if len(layer_plot_data) == 0:
            ax.text(0.5, 0.5, f'Layer {layer}: No data', ha='center', va='center')
            ax.set_title(f'Layer {layer}')
            continue

        x = layer_plot_data['x_order'].values
        y = layer_plot_data['px'].values

        ax.plot(x, y, 'o-', color='steelblue', linewidth=1.5, markersize=4)
        ax.fill_between(x, y, alpha=0.2, color='steelblue')

        ax.set_xlabel('State Index', fontsize=9)
        ax.set_ylabel('P(X wins)', fontsize=9)
        ax.set_title(f'Layer {layer} ({len(layer_plot_data)} states)', fontsize=11, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        min_p, max_p, mean_p = y.min(), y.max(), y.mean()
        stats_text = f'Min: {min_p:.3f}\nMax: {max_p:.3f}\nMean: {mean_p:.3f}'
        ax.text(0.98, 0.02, stats_text, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                transform=ax.transAxes, family='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == '__main__':
    df, center_df = load_and_prepare_data()
    plot_df = build_sorted_plot_data(df, center_df)

    print("\nPlot data summary:")
    for layer in sorted(plot_df['layer'].unique()):
        count = len(plot_df[plot_df['layer'] == layer])
        print(f"  Layer {layer}: {count}")

    create_line_plots(plot_df)
