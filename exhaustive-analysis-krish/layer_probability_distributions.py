#!/usr/bin/env python3
"""
Layer-by-Layer Probability Distribution Analysis

This script reads positional_probabilities.json and creates probability distribution
visualizations for each game layer (depth/turn count).

For each layer, it generates:
1. Histograms of P(X wins), P(O wins), and P(draws) for all available moves
2. Box plots showing distribution statistics
3. Kernel density estimation plots for smooth probability curves
4. Summary statistics per layer

Usage:
    python layer_probability_distributions.py positional_probabilities.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

class LayerProbabilityAnalyzer:
    def __init__(self, json_file):
        """
        Initialize analyzer with positional probabilities JSON
        
        Args:
            json_file: Path to positional_probabilities.json
        """
        self.json_file = json_file
        self.data = None
        self.layer_probabilities = defaultdict(lambda: {
            'p_x_win': [],
            'p_o_win': [],
            'p_draw': [],
            'states_count': 0,
            'moves_count': 0
        })
        
    def load_data(self):
        """Load positional probabilities from JSON"""
        try:
            with open(self.json_file, 'r') as f:
                self.data = json.load(f)
            
            print(f"Loaded positional probabilities")
            print(f"  Total states: {len(self.data['states'])}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def extract_layer_probabilities(self):
        """Extract probability values organized by layer"""
        print("\nExtracting probabilities by layer...")
        
        for state_key, state_data in self.data['states'].items():
            layer = state_data['layer']
            
            self.layer_probabilities[layer]['states_count'] += 1
            
            # Extract all position probabilities for this state
            for pos_key, pos_data in state_data['positions'].items():
                self.layer_probabilities[layer]['p_x_win'].append(pos_data['p_x_win'])
                self.layer_probabilities[layer]['p_o_win'].append(pos_data['p_o_win'])
                self.layer_probabilities[layer]['p_draw'].append(pos_data['p_draw'])
                self.layer_probabilities[layer]['moves_count'] += 1
        
        print(f"Extracted probabilities for {len(self.layer_probabilities)} layers")
        
        # Print summary
        print("\nLayer Summary:")
        print(f"{'Layer':<6} {'States':<8} {'Moves':<8} {'Avg P(X)':<10} {'Avg P(O)':<10} {'Avg P(D)':<10}")
        print("-" * 65)
        
        for layer in sorted(self.layer_probabilities.keys()):
            data = self.layer_probabilities[layer]
            avg_x = np.mean(data['p_x_win']) if data['p_x_win'] else 0
            avg_o = np.mean(data['p_o_win']) if data['p_o_win'] else 0
            avg_d = np.mean(data['p_draw']) if data['p_draw'] else 0
            
            print(f"{layer:<6} {data['states_count']:<8} {data['moves_count']:<8} "
                  f"{avg_x:<10.3f} {avg_o:<10.3f} {avg_d:<10.3f}")
    
    def plot_layer_histograms(self, output_dir='layer_distributions'):
        """Create histogram plots for each layer"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating layer histograms...")
        
        for layer in sorted(self.layer_probabilities.keys()):
            data = self.layer_probabilities[layer]
            
            if not data['p_x_win']:  # Skip if no data
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f'Layer {layer} Probability Distributions\n'
                        f'{data["states_count"]} states, {data["moves_count"]} moves',
                        fontsize=14, fontweight='bold')
            
            # P(X wins) histogram
            axes[0].hist(data['p_x_win'], bins=30, color='red', alpha=0.7, edgecolor='black')
            axes[0].set_xlabel('P(X wins)', fontsize=11)
            axes[0].set_ylabel('Frequency', fontsize=11)
            axes[0].set_title('X Win Probabilities', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].axvline(np.mean(data['p_x_win']), color='darkred', 
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_x_win"]):.3f}')
            axes[0].legend()
            
            # P(O wins) histogram
            axes[1].hist(data['p_o_win'], bins=30, color='blue', alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('P(O wins)', fontsize=11)
            axes[1].set_ylabel('Frequency', fontsize=11)
            axes[1].set_title('O Win Probabilities', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].axvline(np.mean(data['p_o_win']), color='darkblue', 
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_o_win"]):.3f}')
            axes[1].legend()
            
            # P(draws) histogram
            axes[2].hist(data['p_draw'], bins=30, color='green', alpha=0.7, edgecolor='black')
            axes[2].set_xlabel('P(draws)', fontsize=11)
            axes[2].set_ylabel('Frequency', fontsize=11)
            axes[2].set_title('Draw Probabilities', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].axvline(np.mean(data['p_draw']), color='darkgreen', 
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["p_draw"]):.3f}')
            axes[2].legend()
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'layer_{layer}_histograms.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved layer {layer} histograms")
    
    def plot_layer_boxplots(self, output_dir='layer_distributions'):
        """Create box plots comparing distributions across layers"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating comparison box plots...")
        
        layers = sorted(self.layer_probabilities.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle('Probability Distributions Across All Layers', 
                    fontsize=14, fontweight='bold')
        
        # Prepare data for box plots
        x_data = [self.layer_probabilities[layer]['p_x_win'] for layer in layers]
        o_data = [self.layer_probabilities[layer]['p_o_win'] for layer in layers]
        d_data = [self.layer_probabilities[layer]['p_draw'] for layer in layers]
        
        # P(X wins) box plot
        bp1 = axes[0].boxplot(x_data, labels=layers, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightcoral')
        axes[0].set_xlabel('Layer', fontsize=11)
        axes[0].set_ylabel('P(X wins)', fontsize=11)
        axes[0].set_title('X Win Probability by Layer', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # P(O wins) box plot
        bp2 = axes[1].boxplot(o_data, labels=layers, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightblue')
        axes[1].set_xlabel('Layer', fontsize=11)
        axes[1].set_ylabel('P(O wins)', fontsize=11)
        axes[1].set_title('O Win Probability by Layer', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # P(draws) box plot
        bp3 = axes[2].boxplot(d_data, labels=layers, patch_artist=True)
        for patch in bp3['boxes']:
            patch.set_facecolor('lightgreen')
        axes[2].set_xlabel('Layer', fontsize=11)
        axes[2].set_ylabel('P(draws)', fontsize=11)
        axes[2].set_title('Draw Probability by Layer', fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'all_layers_boxplots.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved comparison box plots")
    
    def plot_layer_kde(self, output_dir='layer_distributions'):
        """Create kernel density estimation plots for each layer"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating KDE plots...")
        
        for layer in sorted(self.layer_probabilities.keys()):
            data = self.layer_probabilities[layer]
            
            if not data['p_x_win'] or len(data['p_x_win']) < 2:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot KDE for each probability type
            try:
                sns.kdeplot(data=data['p_x_win'], color='red', fill=True, 
                           alpha=0.5, label='P(X wins)', ax=ax)
                sns.kdeplot(data=data['p_o_win'], color='blue', fill=True, 
                           alpha=0.5, label='P(O wins)', ax=ax)
                sns.kdeplot(data=data['p_draw'], color='green', fill=True, 
                           alpha=0.5, label='P(draws)', ax=ax)
            except:
                print(f"Skipping KDE for layer {layer} (insufficient data variation)")
                plt.close()
                continue
            
            ax.set_xlabel('Probability', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'Layer {layer} Probability Density\n'
                        f'{data["states_count"]} states, {data["moves_count"]} moves',
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'layer_{layer}_kde.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved layer {layer} KDE plot")
    
    def plot_summary_statistics(self, output_dir='layer_distributions'):
        """Plot summary statistics across layers"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating summary statistics plots...")
        
        layers = sorted(self.layer_probabilities.keys())
        
        # Calculate statistics for each layer
        stats = {
            'layers': layers,
            'mean_x': [],
            'std_x': [],
            'mean_o': [],
            'std_o': [],
            'mean_d': [],
            'std_d': []
        }
        
        for layer in layers:
            data = self.layer_probabilities[layer]
            stats['mean_x'].append(np.mean(data['p_x_win']))
            stats['std_x'].append(np.std(data['p_x_win']))
            stats['mean_o'].append(np.mean(data['p_o_win']))
            stats['std_o'].append(np.std(data['p_o_win']))
            stats['mean_d'].append(np.mean(data['p_draw']))
            stats['std_d'].append(np.std(data['p_draw']))
        
        # Plot mean probabilities with error bars
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Mean probabilities
        axes[0].errorbar(stats['layers'], stats['mean_x'], yerr=stats['std_x'], 
                        marker='o', label='P(X wins)', color='red', capsize=5, linewidth=2)
        axes[0].errorbar(stats['layers'], stats['mean_o'], yerr=stats['std_o'], 
                        marker='s', label='P(O wins)', color='blue', capsize=5, linewidth=2)
        axes[0].errorbar(stats['layers'], stats['mean_d'], yerr=stats['std_d'], 
                        marker='^', label='P(draws)', color='green', capsize=5, linewidth=2)
        
        axes[0].set_xlabel('Layer', fontsize=12)
        axes[0].set_ylabel('Mean Probability', fontsize=12)
        axes[0].set_title('Mean Probabilities by Layer (with Std Dev)', 
                         fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Standard deviations
        axes[1].plot(stats['layers'], stats['std_x'], marker='o', 
                    label='Std(X wins)', color='red', linewidth=2)
        axes[1].plot(stats['layers'], stats['std_o'], marker='s', 
                    label='Std(O wins)', color='blue', linewidth=2)
        axes[1].plot(stats['layers'], stats['std_d'], marker='^', 
                    label='Std(draws)', color='green', linewidth=2)
        
        axes[1].set_xlabel('Layer', fontsize=12)
        axes[1].set_ylabel('Standard Deviation', fontsize=12)
        axes[1].set_title('Probability Variance by Layer', 
                         fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'summary_statistics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved summary statistics plot")
    
    def generate_statistics_report(self, output_dir='layer_distributions'):
        """Generate detailed statistics report"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating statistics report...")
        
        report_path = os.path.join(output_dir, 'layer_statistics_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("LAYER-BY-LAYER PROBABILITY DISTRIBUTION ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            for layer in sorted(self.layer_probabilities.keys()):
                data = self.layer_probabilities[layer]
                
                f.write(f"LAYER {layer}\n")
                f.write("-" * 70 + "\n")
                f.write(f"States: {data['states_count']}\n")
                f.write(f"Total moves: {data['moves_count']}\n\n")
                
                f.write("P(X wins) Statistics:\n")
                f.write(f"  Mean:   {np.mean(data['p_x_win']):.6f}\n")
                f.write(f"  Median: {np.median(data['p_x_win']):.6f}\n")
                f.write(f"  Std:    {np.std(data['p_x_win']):.6f}\n")
                f.write(f"  Min:    {np.min(data['p_x_win']):.6f}\n")
                f.write(f"  Max:    {np.max(data['p_x_win']):.6f}\n\n")
                
                f.write("P(O wins) Statistics:\n")
                f.write(f"  Mean:   {np.mean(data['p_o_win']):.6f}\n")
                f.write(f"  Median: {np.median(data['p_o_win']):.6f}\n")
                f.write(f"  Std:    {np.std(data['p_o_win']):.6f}\n")
                f.write(f"  Min:    {np.min(data['p_o_win']):.6f}\n")
                f.write(f"  Max:    {np.max(data['p_o_win']):.6f}\n\n")
                
                f.write("P(draws) Statistics:\n")
                f.write(f"  Mean:   {np.mean(data['p_draw']):.6f}\n")
                f.write(f"  Median: {np.median(data['p_draw']):.6f}\n")
                f.write(f"  Std:    {np.std(data['p_draw']):.6f}\n")
                f.write(f"  Min:    {np.min(data['p_draw']):.6f}\n")
                f.write(f"  Max:    {np.max(data['p_draw']):.6f}\n\n")
                
                f.write("=" * 70 + "\n\n")
        
        print(f"  Saved statistics report")
    
    def run(self, output_dir='layer_distributions'):
        """Run complete layer-by-layer analysis"""
        print("Layer-by-Layer Probability Distribution Analysis")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return False
        
        # Extract probabilities by layer
        self.extract_layer_probabilities()
        
        # Generate all visualizations
        self.plot_layer_histograms(output_dir)
        self.plot_layer_boxplots(output_dir)
        self.plot_layer_kde(output_dir)
        self.plot_summary_statistics(output_dir)
        self.generate_statistics_report(output_dir)
        
        print(f"\nComplete! All visualizations saved in '{output_dir}/'")
        print(f"\nGenerated:")
        print(f"  • Individual layer histograms")
        print(f"  • Comparison box plots")
        print(f"  • Kernel density estimation plots")
        print(f"  • Summary statistics plots")
        print(f"  • Detailed statistics report")
        
        return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python layer_probability_distributions.py positional_probabilities.json")
        print("\nThis script creates probability distribution visualizations for each game layer.")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        sys.exit(1)
    
    # Optional: specify output directory
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'layer_distributions'
    
    # Create analyzer and run
    analyzer = LayerProbabilityAnalyzer(json_file)
    success = analyzer.run(output_dir)
    
    if success:
        print(f"\nUse these visualizations to:")
        print(f"  • Understand how probabilities evolve through game progression")
        print(f"  • Identify layers with highest/lowest variance")
        print(f"  • Detect strategic patterns and critical turning points")
        print(f"  • Compare first-player vs second-player advantages by depth")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
