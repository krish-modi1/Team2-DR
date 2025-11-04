#!/usr/bin/env python3
"""
Per-Position Histogram and KDE Analysis for Tic-Tac-Toe

This script generates:
1. Per-position histograms and KDE plots (aggregated across all layers)
2. Per-position per-layer histograms and KDE plots (9 positions × 9 layers)

Usage:
    python position_kde_analysis.py positional_probabilities.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
import os

class PositionKDEAnalyzer:
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
        
        # Storage for position data
        # position_data[position] = {'p_x_win': [], 'p_o_win': [], 'p_draw': []}
        self.position_data = defaultdict(lambda: {
            'p_x_win': [],
            'p_o_win': [],
            'p_draw': []
        })
        
        # Storage for position-layer data
        # position_layer_data[position][layer] = {'p_x_win': [], 'p_o_win': [], 'p_draw': []}
        self.position_layer_data = defaultdict(lambda: defaultdict(lambda: {
            'p_x_win': [],
            'p_o_win': [],
            'p_draw': []
        }))
        
    def load_data(self):
        """Load positional probabilities from JSON"""
        try:
            with open(self.json_file, 'r') as f:
                self.data = json.load(f)
            
            print(f"✓ Loaded positional probabilities")
            print(f"  Total states: {len(self.data['states'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def extract_position_data(self):
        """Extract probability values organized by position and layer"""
        print("\nExtracting position and layer data...")
        
        for state_key, state_data in self.data['states'].items():
            layer = state_data['layer']
            
            for pos_key, pos_data in state_data['positions'].items():
                position = int(pos_key)
                
                # Aggregate across all layers
                self.position_data[position]['p_x_win'].append(pos_data['p_x_win'])
                self.position_data[position]['p_o_win'].append(pos_data['p_o_win'])
                self.position_data[position]['p_draw'].append(pos_data['p_draw'])
                
                # Per layer
                self.position_layer_data[position][layer]['p_x_win'].append(pos_data['p_x_win'])
                self.position_layer_data[position][layer]['p_o_win'].append(pos_data['p_o_win'])
                self.position_layer_data[position][layer]['p_draw'].append(pos_data['p_draw'])
        
        print(f"✓ Extracted data for {len(self.position_data)} positions")
        
        # Print summary
        for position in sorted(self.position_data.keys()):
            count = len(self.position_data[position]['p_x_win'])
            print(f"  Position {position}: {count} occurrences across all layers")
    
    def get_position_type(self, position):
        """Get position type (corner/edge/center)"""
        if position in [0, 2, 6, 8]:
            return "corner"
        elif position == 4:
            return "center"
        else:
            return "edge"
    
    def plot_position_histogram_kde_aggregate(self, output_dir='position_kde_plots'):
        """
        Plot 1: Per-position histograms and KDE (aggregated across all layers)
        Creates 9 plots, one for each position
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[1/2] Generating per-position aggregate histograms and KDE...")
        
        for position in sorted(self.position_data.keys()):
            data = self.position_data[position]
            
            if not data['p_x_win']:
                continue
            
            row, col = position // self.n, position % self.n
            pos_type = self.get_position_type(position)
            count = len(data['p_x_win'])
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(f'Position {position} ({row},{col}) - {pos_type.upper()}\n'
                        f'{count} total occurrences across all layers',
                        fontsize=14, fontweight='bold')
            
            # P(X wins)
            axes[0].hist(data['p_x_win'], bins=25, alpha=0.6, color='red', 
                        edgecolor='black', density=True, label='Histogram')
            
            if len(data['p_x_win']) > 1:
                try:
                    kde_x = stats.gaussian_kde(data['p_x_win'])
                    x_range = np.linspace(0, 1, 200)
                    axes[0].plot(x_range, kde_x(x_range), 'r-', linewidth=2, label='KDE')
                    axes[0].fill_between(x_range, kde_x(x_range), alpha=0.3, color='red')
                except:
                    pass
            
            axes[0].set_xlabel('P(X wins)', fontsize=11)
            axes[0].set_ylabel('Density', fontsize=11)
            axes[0].set_title('X Win Probabilities', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].axvline(np.mean(data['p_x_win']), color='darkred', 
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_x_win"]):.3f}')
            axes[0].legend()
            axes[0].set_xlim(0, 1)
            
            # P(O wins)
            axes[1].hist(data['p_o_win'], bins=25, alpha=0.6, color='blue', 
                        edgecolor='black', density=True, label='Histogram')
            
            if len(data['p_o_win']) > 1:
                try:
                    kde_o = stats.gaussian_kde(data['p_o_win'])
                    x_range = np.linspace(0, 1, 200)
                    axes[1].plot(x_range, kde_o(x_range), 'b-', linewidth=2, label='KDE')
                    axes[1].fill_between(x_range, kde_o(x_range), alpha=0.3, color='blue')
                except:
                    pass
            
            axes[1].set_xlabel('P(O wins)', fontsize=11)
            axes[1].set_ylabel('Density', fontsize=11)
            axes[1].set_title('O Win Probabilities', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].axvline(np.mean(data['p_o_win']), color='darkblue', 
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_o_win"]):.3f}')
            axes[1].legend()
            axes[1].set_xlim(0, 1)
            
            # P(draws)
            axes[2].hist(data['p_draw'], bins=25, alpha=0.6, color='green', 
                        edgecolor='black', density=True, label='Histogram')
            
            if len(data['p_draw']) > 1:
                try:
                    kde_d = stats.gaussian_kde(data['p_draw'])
                    x_range = np.linspace(0, 1, 200)
                    axes[2].plot(x_range, kde_d(x_range), 'g-', linewidth=2, label='KDE')
                    axes[2].fill_between(x_range, kde_d(x_range), alpha=0.3, color='green')
                except:
                    pass
            
            axes[2].set_xlabel('P(draws)', fontsize=11)
            axes[2].set_ylabel('Density', fontsize=11)
            axes[2].set_title('Draw Probabilities', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].axvline(np.mean(data['p_draw']), color='darkgreen', 
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_draw"]):.3f}')
            axes[2].legend()
            axes[2].set_xlim(0, 1)
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f'position_{position}_aggregate_hist_kde.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved position {position} aggregate plot")
    
    def plot_position_layer_histogram_kde(self, output_dir='position_layer_kde_plots'):
        """
        Plot 2: Per-position per-layer histograms and KDE
        Creates 9 positions × 9 layers = 81 plots total
        Each plot shows histogram + KDE for that specific position-layer combination
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[2/2] Generating per-position per-layer histograms and KDE...")
        
        total_plots = 0
        
        for position in sorted(self.position_layer_data.keys()):
            # Create subdirectory for this position
            pos_dir = os.path.join(output_dir, f'position_{position}')
            os.makedirs(pos_dir, exist_ok=True)
            
            row, col = position // self.n, position % self.n
            pos_type = self.get_position_type(position)
            
            for layer in sorted(self.position_layer_data[position].keys()):
                data = self.position_layer_data[position][layer]
                
                if not data['p_x_win'] or len(data['p_x_win']) < 2:
                    continue
                
                count = len(data['p_x_win'])
                
                # Create figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                fig.suptitle(f'Position {position} ({row},{col}) - {pos_type.upper()} | Layer {layer}\n'
                            f'{count} occurrences at this layer',
                            fontsize=14, fontweight='bold')
                
                # P(X wins)
                axes[0].hist(data['p_x_win'], bins=min(20, count//2), alpha=0.6, color='red', 
                            edgecolor='black', density=True, label='Histogram')
                
                if len(data['p_x_win']) > 2:
                    try:
                        kde_x = stats.gaussian_kde(data['p_x_win'], bw_method='scott')
                        x_range = np.linspace(0, 1, 200)
                        axes[0].plot(x_range, kde_x(x_range), 'r-', linewidth=2, label='KDE')
                        axes[0].fill_between(x_range, kde_x(x_range), alpha=0.3, color='red')
                    except:
                        pass
                
                axes[0].set_xlabel('P(X wins)', fontsize=11)
                axes[0].set_ylabel('Density', fontsize=11)
                axes[0].set_title('X Win Probabilities', fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                axes[0].axvline(np.mean(data['p_x_win']), color='darkred', 
                               linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_x_win"]):.3f}')
                axes[0].legend()
                axes[0].set_xlim(0, 1)
                
                # P(O wins)
                axes[1].hist(data['p_o_win'], bins=min(20, count//2), alpha=0.6, color='blue', 
                            edgecolor='black', density=True, label='Histogram')
                
                if len(data['p_o_win']) > 2:
                    try:
                        kde_o = stats.gaussian_kde(data['p_o_win'], bw_method='scott')
                        x_range = np.linspace(0, 1, 200)
                        axes[1].plot(x_range, kde_o(x_range), 'b-', linewidth=2, label='KDE')
                        axes[1].fill_between(x_range, kde_o(x_range), alpha=0.3, color='blue')
                    except:
                        pass
                
                axes[1].set_xlabel('P(O wins)', fontsize=11)
                axes[1].set_ylabel('Density', fontsize=11)
                axes[1].set_title('O Win Probabilities', fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].axvline(np.mean(data['p_o_win']), color='darkblue', 
                               linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_o_win"]):.3f}')
                axes[1].legend()
                axes[1].set_xlim(0, 1)
                
                # P(draws)
                axes[2].hist(data['p_draw'], bins=min(20, count//2), alpha=0.6, color='green', 
                            edgecolor='black', density=True, label='Histogram')
                
                if len(data['p_draw']) > 2:
                    try:
                        kde_d = stats.gaussian_kde(data['p_draw'], bw_method='scott')
                        x_range = np.linspace(0, 1, 200)
                        axes[2].plot(x_range, kde_d(x_range), 'g-', linewidth=2, label='KDE')
                        axes[2].fill_between(x_range, kde_d(x_range), alpha=0.3, color='green')
                    except:
                        pass
                
                axes[2].set_xlabel('P(draws)', fontsize=11)
                axes[2].set_ylabel('Density', fontsize=11)
                axes[2].set_title('Draw Probabilities', fontweight='bold')
                axes[2].grid(True, alpha=0.3)
                axes[2].axvline(np.mean(data['p_draw']), color='darkgreen', 
                               linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_draw"]):.3f}')
                axes[2].legend()
                axes[2].set_xlim(0, 1)
                
                plt.tight_layout()
                
                output_path = os.path.join(pos_dir, f'position_{position}_layer_{layer}_hist_kde.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                total_plots += 1
        
        print(f"  ✓ Generated {total_plots} position-layer plots")
    
    def generate_summary_grid(self, output_dir='position_kde_plots'):
        """Generate a summary grid showing all 9 positions in one image"""
        print(f"\nGenerating summary grid...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle('Position-wise P(X wins) KDE Comparison\nAcross All Layers', 
                    fontsize=16, fontweight='bold')
        
        for position in range(9):
            row, col = position // 3, position % 3
            ax = axes[row, col]
            
            if position not in self.position_data:
                ax.axis('off')
                continue
            
            data = self.position_data[position]
            pos_type = self.get_position_type(position)
            
            # Plot KDE
            if len(data['p_x_win']) > 2:
                try:
                    kde = stats.gaussian_kde(data['p_x_win'])
                    x_range = np.linspace(0, 1, 200)
                    ax.fill_between(x_range, kde(x_range), alpha=0.5, color='red')
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
                except:
                    ax.hist(data['p_x_win'], bins=20, alpha=0.6, color='red', density=True)
            
            ax.axvline(np.mean(data['p_x_win']), color='darkred', 
                      linestyle='--', linewidth=2)
            
            ax.set_title(f'Pos {position} ({pos_type})\nμ={np.mean(data["p_x_win"]):.3f}', 
                        fontweight='bold')
            ax.set_xlabel('P(X wins)', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'all_positions_kde_grid.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved summary grid")
    
    def run(self):
        """Run complete analysis"""
        print("Position-wise Histogram and KDE Analysis")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return False
        
        # Extract data
        self.extract_position_data()
        
        # Generate plots
        self.plot_position_histogram_kde_aggregate()
        self.plot_position_layer_histogram_kde()
        self.generate_summary_grid()
        
        print(f"\n✅ Complete!")
        print(f"\n📊 Generated plots:")
        print(f"  • 9 aggregate position plots: position_kde_plots/")
        print(f"  • 81 position-layer plots: position_layer_kde_plots/position_X/")
        print(f"  • 1 summary grid: position_kde_plots/all_positions_kde_grid.png")
        
        return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python position_kde_analysis.py positional_probabilities.json")
        print("\nGenerates:")
        print("  1. Per-position histograms + KDE (aggregated across layers)")
        print("  2. Per-position per-layer histograms + KDE (9×9 = 81 plots)")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        sys.exit(1)
    
    # Create analyzer and run
    analyzer = PositionKDEAnalyzer(json_file)
    success = analyzer.run()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
