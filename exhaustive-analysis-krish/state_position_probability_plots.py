#!/usr/bin/env python3
"""
State-wise Position Probability Analysis for Tic-Tac-Toe

This script creates probability distribution plots where states are ordered by probability
rather than layer. Generates a universal state index mapping CSV for consistency.

Usage:
    python state_position_probability_plots.py positional_probabilities.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os

class StatePositionProbabilityAnalyzer:
    def __init__(self, json_file, n=3):
        """
        Initialize analyzer with positional probabilities JSON
        
        Args:
            json_file: Path to positional_probabilities.json
            n: Board size (default 3 for 3x3)
        """
        self.json_file = json_file
        self.n = n
        self.data = None
        self.position_states = defaultdict(list)
        self.state_index_mappings = {}  # Per-position state index mappings
        
    def load_data(self):
        """Load positional probabilities from JSON"""
        try:
            with open(self.json_file, 'r') as f:
                self.data = json.load(f)
            
            print("Loaded positional probabilities")
            print(f"  Total states: {len(self.data['states'])}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def extract_position_state_data(self):
        """Extract (state, probability) pairs for each position"""
        print("\nExtracting state-probability pairs for each position...")
        
        for state_key, state_data in self.data['states'].items():
            layer = state_data['layer']
            
            for pos_key, pos_data in state_data['positions'].items():
                position = int(pos_key)
                
                self.position_states[position].append({
                    'state_key': state_key,
                    'layer': layer,
                    'p_x_win': pos_data['p_x_win'],
                    'p_o_win': pos_data['p_o_win'],
                    'p_draw': pos_data['p_draw']
                })
        
        # Sort by probability for each position
        for position in self.position_states:
            self.position_states[position].sort(key=lambda x: x['p_x_win'])
        
        print(f"Extracted data for {len(self.position_states)} positions")
        for position in sorted(self.position_states.keys()):
            count = len(self.position_states[position])
            print(f"  Position {position}: {count} unique states")
    
    def create_state_index_csv(self, output_dir='state_position_plots'):
        """Create universal state index mapping CSV for all positions"""
        os.makedirs(output_dir, exist_ok=True)
        print("\nCreating state index mapping CSV...")
        
        all_mappings = []
        
        for position in sorted(self.position_states.keys()):
            states_data = self.position_states[position]
            
            for idx, state_info in enumerate(states_data):
                all_mappings.append({
                    'position': position,
                    'state_index': idx,
                    'state_key': state_info['state_key'],
                    'layer': state_info['layer'],
                    'p_x_win': state_info['p_x_win'],
                    'p_o_win': state_info['p_o_win'],
                    'p_draw': state_info['p_draw']
                })
            
            # Store per-position mapping
            self.state_index_mappings[position] = {
                idx: state_info['state_key'] 
                for idx, state_info in enumerate(states_data)
            }
        
        df = pd.DataFrame(all_mappings)
        csv_path = os.path.join(output_dir, 'state_index_mapping.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"  Saved state index mapping CSV: {csv_path}")
        print(f"  Total entries: {len(all_mappings)}")
        
        return csv_path
    
    def get_position_type(self, position):
        """Get position type (corner/edge/center)"""
        if position in [0, 2, 6, 8]:
            return "corner"
        elif position == 4:
            return "center"
        else:
            return "edge"
    
    def plot_position_state_probabilities_separate(self, output_dir='state_position_plots'):
        """
        Create 3 separate plots per position (one for each outcome type)
        States are ordered by probability value
        """
        os.makedirs(output_dir, exist_ok=True)
        print("\nGenerating state-probability plots (ordered by probability)...")
        
        for position in sorted(self.position_states.keys()):
            states_data = self.position_states[position]
            
            if not states_data:
                continue
            
            row, col = position // self.n, position % self.n
            pos_type = self.get_position_type(position)
            
            # Extract data (already sorted by p_x_win)
            num_states = len(states_data)
            state_indices = list(range(num_states))
            
            p_x_wins = [s['p_x_win'] for s in states_data]
            p_o_wins = [s['p_o_win'] for s in states_data]
            p_draws = [s['p_draw'] for s in states_data]
            layers = [s['layer'] for s in states_data]
            
            # Color by layer
            unique_layers = sorted(set(layers))
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
            layer_to_color = {layer: colors[i] for i, layer in enumerate(unique_layers)}
            point_colors = [layer_to_color[layer] for layer in layers]
            
            # Plot 1: P(X wins)
            fig, ax = plt.subplots(figsize=(16, 6))
            
            scatter = ax.scatter(state_indices, p_x_wins, c=point_colors, 
                               alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
            ax.plot(state_indices, p_x_wins, 'k-', alpha=0.15, linewidth=1)
            
            ax.axhline(np.mean(p_x_wins), color='red', linestyle='--', 
                      linewidth=2.5, label=f'Mean: {np.mean(p_x_wins):.3f}', zorder=10)
            ax.axhline(np.median(p_x_wins), color='darkred', linestyle=':', 
                      linewidth=2.5, label=f'Median: {np.median(p_x_wins):.3f}', zorder=10)
            
            ax.set_xlabel('State Index (ordered by P(X wins))', fontsize=13, fontweight='bold')
            ax.set_ylabel('P(X wins)', fontsize=13, fontweight='bold')
            ax.set_title(f'Position {position} ({row},{col}) - {pos_type.upper()}\n'
                        f'X Win Probabilities Across {num_states} Unique States',
                        fontsize=15, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax.set_ylim(-0.05, 1.05)
            
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                      norm=plt.Normalize(vmin=min(unique_layers), 
                                                        vmax=max(unique_layers)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Layer (Game Depth)', fontsize=12, fontweight='bold')
            
            stats_text = f'Statistics:\n'
            stats_text += f'Mean: {np.mean(p_x_wins):.4f}\n'
            stats_text += f'Std:  {np.std(p_x_wins):.4f}\n'
            stats_text += f'Min:  {np.min(p_x_wins):.4f}\n'
            stats_text += f'Max:  {np.max(p_x_wins):.4f}'
            
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'position_{position}_X_wins.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 2: P(O wins) - sort by O win probability
            sorted_o_indices = sorted(range(num_states), key=lambda i: p_o_wins[i])
            sorted_o_wins = [p_o_wins[i] for i in sorted_o_indices]
            sorted_o_colors = [point_colors[i] for i in sorted_o_indices]
            
            fig, ax = plt.subplots(figsize=(16, 6))
            
            scatter = ax.scatter(range(num_states), sorted_o_wins, c=sorted_o_colors, 
                               alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
            ax.plot(range(num_states), sorted_o_wins, 'k-', alpha=0.15, linewidth=1)
            
            ax.axhline(np.mean(p_o_wins), color='blue', linestyle='--', 
                      linewidth=2.5, label=f'Mean: {np.mean(p_o_wins):.3f}', zorder=10)
            ax.axhline(np.median(p_o_wins), color='darkblue', linestyle=':', 
                      linewidth=2.5, label=f'Median: {np.median(p_o_wins):.3f}', zorder=10)
            
            ax.set_xlabel('State Index (ordered by P(O wins))', fontsize=13, fontweight='bold')
            ax.set_ylabel('P(O wins)', fontsize=13, fontweight='bold')
            ax.set_title(f'Position {position} ({row},{col}) - {pos_type.upper()}\n'
                        f'O Win Probabilities Across {num_states} Unique States',
                        fontsize=15, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax.set_ylim(-0.05, 1.05)
            
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                      norm=plt.Normalize(vmin=min(unique_layers), 
                                                        vmax=max(unique_layers)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Layer (Game Depth)', fontsize=12, fontweight='bold')
            
            stats_text = f'Statistics:\n'
            stats_text += f'Mean: {np.mean(p_o_wins):.4f}\n'
            stats_text += f'Std:  {np.std(p_o_wins):.4f}\n'
            stats_text += f'Min:  {np.min(p_o_wins):.4f}\n'
            stats_text += f'Max:  {np.max(p_o_wins):.4f}'
            
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'position_{position}_O_wins.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 3: P(draws) - sort by draw probability
            sorted_d_indices = sorted(range(num_states), key=lambda i: p_draws[i])
            sorted_draws = [p_draws[i] for i in sorted_d_indices]
            sorted_d_colors = [point_colors[i] for i in sorted_d_indices]
            
            fig, ax = plt.subplots(figsize=(16, 6))
            
            scatter = ax.scatter(range(num_states), sorted_draws, c=sorted_d_colors, 
                               alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
            ax.plot(range(num_states), sorted_draws, 'k-', alpha=0.15, linewidth=1)
            
            ax.axhline(np.mean(p_draws), color='green', linestyle='--', 
                      linewidth=2.5, label=f'Mean: {np.mean(p_draws):.3f}', zorder=10)
            ax.axhline(np.median(p_draws), color='darkgreen', linestyle=':', 
                      linewidth=2.5, label=f'Median: {np.median(p_draws):.3f}', zorder=10)
            
            ax.set_xlabel('State Index (ordered by P(draws))', fontsize=13, fontweight='bold')
            ax.set_ylabel('P(draws)', fontsize=13, fontweight='bold')
            ax.set_title(f'Position {position} ({row},{col}) - {pos_type.upper()}\n'
                        f'Draw Probabilities Across {num_states} Unique States',
                        fontsize=15, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax.set_ylim(-0.05, 1.05)
            
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                      norm=plt.Normalize(vmin=min(unique_layers), 
                                                        vmax=max(unique_layers)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Layer (Game Depth)', fontsize=12, fontweight='bold')
            
            stats_text = f'Statistics:\n'
            stats_text += f'Mean: {np.mean(p_draws):.4f}\n'
            stats_text += f'Std:  {np.std(p_draws):.4f}\n'
            stats_text += f'Min:  {np.min(p_draws):.4f}\n'
            stats_text += f'Max:  {np.max(p_draws):.4f}'
            
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'position_{position}_draws.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved position {position} (3 separate plots)")
    
    def plot_layer_separated_view(self, output_dir='state_position_plots'):
        """
        Separate scatter plots per layer for each position
        Shows how state-probability relationship changes across layers
        """
        layer_dir = os.path.join(output_dir, 'layer_separated')
        os.makedirs(layer_dir, exist_ok=True)
        print("\nGenerating layer-separated views...")
        
        for position in sorted(self.position_states.keys()):
            states_data = self.position_states[position]
            
            if not states_data:
                continue
            
            row, col = position // self.n, position % self.n
            pos_type = self.get_position_type(position)
            
            # Group by layer
            layer_data = defaultdict(list)
            for state in states_data:
                layer_data[state['layer']].append(state)
            
            # Sort within each layer by probability
            for layer in layer_data:
                layer_data[layer].sort(key=lambda x: x['p_x_win'])
            
            # Create subplot grid
            layers = sorted(layer_data.keys())
            n_layers = len(layers)
            n_cols = 3
            n_rows = (n_layers + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            fig.suptitle(f'Position {position} ({row},{col}) - {pos_type.upper()}\n'
                        f'P(X wins) by State, Separated by Layer',
                        fontsize=14, fontweight='bold')
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, layer in enumerate(layers):
                r, c = idx // n_cols, idx % n_cols
                ax = axes[r, c]
                
                layer_states = layer_data[layer]
                num_states = len(layer_states)
                
                x_wins = [s['p_x_win'] for s in layer_states]
                
                ax.scatter(range(num_states), x_wins, alpha=0.7, s=40, color='red', 
                          edgecolors='black', linewidth=0.5)
                ax.plot(range(num_states), x_wins, 'k-', alpha=0.3, linewidth=1)
                ax.axhline(np.mean(x_wins), color='darkred', linestyle='--', linewidth=2)
                
                ax.set_title(f'Layer {layer} ({num_states} states)', fontsize=11, fontweight='bold')
                ax.set_xlabel('State Index (by P(X wins))', fontsize=10)
                ax.set_ylabel('P(X wins)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
                
                ax.text(0.02, 0.98, f'Mean={np.mean(x_wins):.3f}\nStd={np.std(x_wins):.3f}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Hide unused subplots
            for idx in range(n_layers, n_rows * n_cols):
                r, c = idx // n_cols, idx % n_cols
                axes[r, c].axis('off')
            
            plt.tight_layout()
            
            output_path = os.path.join(layer_dir, f'position_{position}_layer_separated.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved position {position} layer-separated plot")
    
    def generate_pattern_analysis_report(self, output_dir='state_position_plots'):
        """Generate text report analyzing patterns found"""
        print("\nGenerating pattern analysis report...")
        
        report_path = os.path.join(output_dir, 'pattern_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("STATE-POSITION PROBABILITY PATTERN ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            for position in sorted(self.position_states.keys()):
                states_data = self.position_states[position]
                
                if not states_data:
                    continue
                
                pos_type = self.get_position_type(position)
                row, col = position // self.n, position % self.n
                
                p_x_wins = [s['p_x_win'] for s in states_data]
                
                f.write(f"POSITION {position} ({row},{col}) - {pos_type.upper()}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total unique states: {len(states_data)}\n\n")
                
                f.write("P(X wins) Statistics:\n")
                f.write(f"  Mean:       {np.mean(p_x_wins):.6f}\n")
                f.write(f"  Median:     {np.median(p_x_wins):.6f}\n")
                f.write(f"  Std Dev:    {np.std(p_x_wins):.6f}\n")
                f.write(f"  Min:        {np.min(p_x_wins):.6f}\n")
                f.write(f"  Max:        {np.max(p_x_wins):.6f}\n")
                f.write(f"  Range:      {np.max(p_x_wins) - np.min(p_x_wins):.6f}\n\n")
                
                min_idx = np.argmin(p_x_wins)
                max_idx = np.argmax(p_x_wins)
                
                f.write("Extreme States:\n")
                f.write(f"  Worst state for X: {states_data[min_idx]['state_key']}\n")
                f.write(f"    Layer: {states_data[min_idx]['layer']}, P(X wins): {p_x_wins[min_idx]:.6f}\n\n")
                f.write(f"  Best state for X:  {states_data[max_idx]['state_key']}\n")
                f.write(f"    Layer: {states_data[max_idx]['layer']}, P(X wins): {p_x_wins[max_idx]:.6f}\n\n")
                
                layer_variance = {}
                for state in states_data:
                    layer = state['layer']
                    if layer not in layer_variance:
                        layer_variance[layer] = []
                    layer_variance[layer].append(state['p_x_win'])
                
                f.write("Variance by Layer:\n")
                for layer in sorted(layer_variance.keys()):
                    var = np.var(layer_variance[layer])
                    f.write(f"  Layer {layer}: variance = {var:.6f} ({len(layer_variance[layer])} states)\n")
                
                f.write("\n" + "=" * 70 + "\n\n")
        
        print(f"  Saved pattern analysis report")
    
    def run(self):
        """Run complete analysis"""
        print("State-Position Probability Analysis")
        print("=" * 50)
        
        if not self.load_data():
            return False
        
        self.extract_position_state_data()
        
        self.create_state_index_csv()
        self.plot_position_state_probabilities_separate()
        self.plot_layer_separated_view()
        self.generate_pattern_analysis_report()
        
        print("\nComplete!")
        print("\nGenerated:")
        print("  - 27 probability plots (9 positions x 3 outcomes)")
        print("  - 9 layer-separated plots")
        print("  - 1 state index mapping CSV")
        print("  - 1 pattern analysis report")
        print("\nOutput in: state_position_plots/")
        
        return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python state_position_probability_plots.py positional_probabilities.json")
        print("\nCreates state-wise probability plots ordered by probability value.")
        print("Generates universal state index mapping CSV for consistency.")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        sys.exit(1)
    
    analyzer = StatePositionProbabilityAnalyzer(json_file)
    success = analyzer.run()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
