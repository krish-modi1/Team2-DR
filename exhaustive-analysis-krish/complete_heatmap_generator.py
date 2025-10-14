#!/usr/bin/env python3
"""
Generate Next-Move Probability Heatmaps from Complete Tic-Tac-Toe Game Tree JSON

This script reads the original tic_tac_toe_3x3_complete_tree.json file and generates
heatmaps showing win probabilities if the next move is played at each empty position.

Usage:
    python complete_heatmap_generator.py tic_tac_toe_3x3_complete_tree.json

Output Structure:
    next_move_heatmaps/
    ├── P_X_wins/layer_N/ - P(X wins) if current player plays at each position
    ├── P_O_wins/layer_N/ - P(O wins) if current player plays at each position  
    └── P_draws/layer_N/  - P(draws) if current player plays at each position
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import defaultdict

class CompleteTreeHeatmapGenerator:
    def __init__(self, json_file, n=3):
        """
        Initialize heatmap generator with complete game tree JSON
        
        Args:
            json_file: Path to tic_tac_toe_3x3_complete_tree.json
            n: Board size (default 3)
        """
        self.json_file = json_file
        self.n = n
        self.tree_data = None
        self.game_tree = None
        self.base_output_dir = "next_move_heatmaps"
        
    def load_game_tree(self):
        """Load the complete game tree from JSON"""
        try:
            with open(self.json_file, 'r') as f:
                self.tree_data = json.load(f)
            
            self.game_tree = self.tree_data['tree']
            print(f"✓ Loaded complete game tree with {len(self.game_tree)} states")
            
            # Validate structure
            if 'n' in self.tree_data:
                self.n = self.tree_data['n']
            
            return True
            
        except Exception as e:
            print(f"Error loading game tree: {e}")
            return False
    
    def create_directory_structure(self):
        """Create output directory structure"""
        outcome_types = ['P_X_wins', 'P_O_wins', 'P_draws']
        
        # Find all layers in the tree
        layers = set()
        for state_key, node_data in self.game_tree.items():
            state_list = eval(state_key)
            depth = sum(1 for x in state_list if x != 0)
            layers.add(depth)
        
        # Create directories
        for outcome_type in outcome_types:
            for layer in sorted(layers):
                layer_dir = os.path.join(self.base_output_dir, outcome_type, f'layer_{layer}')
                os.makedirs(layer_dir, exist_ok=True)
        
        print(f"✓ Created directory structure for {len(outcome_types)} outcomes and {len(layers)} layers")
    
    def state_key_to_board(self, state_key):
        """Convert state key to 2D board array"""
        state_list = eval(state_key)  # Convert string representation to list
        return np.array(state_list).reshape(self.n, self.n)
    
    def board_to_state_key(self, board):
        """Convert 2D board array to state key string"""
        flat_board = board.flatten().tolist()
        return str(flat_board)
    
    def get_state_depth(self, state_key):
        """Get depth (number of moves) of a state"""
        state_list = eval(state_key)
        return sum(1 for x in state_list if x != 0)
    
    def calculate_probabilities_for_state(self, state_key):
        """
        Calculate probabilities for a state using exhaustive tree analysis
        
        This implements the correct mathematical approach:
        P(outcome | state) = (1/n) × Σ P(outcome | state_after_move_i)
        where n = number of valid moves
        """
        if state_key not in self.game_tree:
            return {'p_x_win': 0.0, 'p_o_win': 0.0, 'p_draw': 1.0}
        
        node_data = self.game_tree[state_key]
        
        # Terminal states have deterministic outcomes
        if node_data['terminal']:
            result = node_data['result']
            if result == 1:  # X wins
                return {'p_x_win': 1.0, 'p_o_win': 0.0, 'p_draw': 0.0}
            elif result == -1:  # O wins
                return {'p_x_win': 0.0, 'p_o_win': 1.0, 'p_draw': 0.0}
            else:  # Draw
                return {'p_x_win': 0.0, 'p_o_win': 0.0, 'p_draw': 1.0}
        
        # Non-terminal: average over all possible moves
        total_x_win = 0.0
        total_o_win = 0.0
        total_draw = 0.0
        num_moves = len(node_data['moves'])
        
        if num_moves == 0:
            return {'p_x_win': 0.0, 'p_o_win': 0.0, 'p_draw': 1.0}
        
        for move, next_state_list in node_data['moves']:
            next_state_key = str(next_state_list)
            next_probs = self.calculate_probabilities_for_state(next_state_key)
            
            total_x_win += next_probs['p_x_win']
            total_o_win += next_probs['p_o_win']
            total_draw += next_probs['p_draw']
        
        return {
            'p_x_win': total_x_win / num_moves,
            'p_o_win': total_o_win / num_moves,
            'p_draw': total_draw / num_moves
        }
    
    def get_next_move_probabilities(self, state_key):
        """
        Calculate probabilities for each possible next move from current state
        
        Returns three matrices showing probabilities if next move played at each position
        """
        if state_key not in self.game_tree:
            return None, None, None
        
        node_data = self.game_tree[state_key]
        
        # Skip terminal states
        if node_data['terminal']:
            return None, None, None
        
        current_board = self.state_key_to_board(state_key)
        current_player = node_data['player']
        
        # Initialize probability matrices
        prob_matrix_x = np.full((self.n, self.n), np.nan)
        prob_matrix_o = np.full((self.n, self.n), np.nan)
        prob_matrix_d = np.full((self.n, self.n), np.nan)
        
        # For each possible move in the game tree
        for move, next_state_list in node_data['moves']:
            next_state_key = str(next_state_list)
            
            # Calculate probabilities for the next state
            next_probs = self.calculate_probabilities_for_state(next_state_key)
            
            # Convert move index to board coordinates
            row = move // self.n
            col = move % self.n
            
            # Store probabilities in matrices
            prob_matrix_x[row, col] = next_probs['p_x_win']
            prob_matrix_o[row, col] = next_probs['p_o_win']
            prob_matrix_d[row, col] = next_probs['p_draw']
        
        return prob_matrix_x, prob_matrix_o, prob_matrix_d
    
    def create_heatmap(self, state_key, prob_matrix, outcome_type):
        """Create a single heatmap for a state and outcome type"""
        
        if prob_matrix is None:
            return None
        
        current_board = self.state_key_to_board(state_key)
        node_data = self.game_tree[state_key]
        current_player = node_data['player']
        layer = self.get_state_depth(state_key)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Color scheme based on outcome type
        if outcome_type == 'P_X_wins':
            cmap = 'Reds'
            color_label = 'P(X wins)'
        elif outcome_type == 'P_O_wins':
            cmap = 'Blues'
            color_label = 'P(O wins)'
        else:  # P_draws
            cmap = 'Greens'
            color_label = 'P(draws)'
        
        # Create heatmap (mask NaN values for occupied positions)
        masked_probs = np.ma.masked_where(np.isnan(prob_matrix), prob_matrix)
        im = ax.imshow(masked_probs, cmap=cmap, vmin=0, vmax=1, alpha=0.9)
        
        # Add board state overlay and probability text
        for i in range(self.n):
            for j in range(self.n):
                cell_value = current_board[i, j]
                prob_value = prob_matrix[i, j]
                
                if cell_value == 1:  # X piece (already played)
                    ax.add_patch(plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                             fill=True, color='lightcoral', alpha=0.8))
                    ax.text(j, i, 'X', ha='center', va='center',
                           fontsize=28, fontweight='bold', color='darkred')
                           
                elif cell_value == -1:  # O piece (already played)
                    ax.add_patch(plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                             fill=True, color='lightblue', alpha=0.8))
                    ax.text(j, i, 'O', ha='center', va='center',
                           fontsize=28, fontweight='bold', color='darkblue')
                           
                else:  # Empty position - show probability if next move played here
                    if not np.isnan(prob_value):
                        ax.text(j, i, f'{prob_value:.3f}', ha='center', va='center',
                               fontsize=12, fontweight='bold', color='white',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor='black', alpha=0.8))
        
        # Customize plot
        player_name = 'X' if current_player == 1 else 'O'
        ax.set_title(f'Layer {layer}: {color_label} if {player_name} plays next move\n'
                    f'State: {state_key}', 
                    fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.n))
        ax.set_xticklabels(range(self.n))
        ax.set_yticklabels(range(self.n))
        
        # Add grid
        ax.set_xticks(np.arange(-.5, self.n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.n, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{color_label} if {player_name} plays here', 
                      rotation=270, labelpad=25)
        
        plt.tight_layout()
        
        # Save the heatmap
        # Create safe filename by replacing problematic characters
        safe_state = state_key.replace('[', '').replace(']', '').replace(', ', '_').replace(' ', '')
        filename = f'state_{safe_state}_{player_name}_nextmove.png'
        output_path = os.path.join(self.base_output_dir, outcome_type, 
                                  f'layer_{layer}', filename)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_all_heatmaps(self):
        """Generate heatmaps for all non-terminal states in the game tree"""
        print("Generating next-move probability heatmaps from complete game tree...")
        
        outcome_types = ['P_X_wins', 'P_O_wins', 'P_draws']
        
        # Find all non-terminal states
        non_terminal_states = []
        for state_key, node_data in self.game_tree.items():
            if not node_data['terminal']:
                non_terminal_states.append(state_key)
        
        print(f"Found {len(non_terminal_states)} non-terminal states to process")
        
        generated_count = 0
        total_expected = len(non_terminal_states) * len(outcome_types)
        
        # Group by layer for better progress tracking
        states_by_layer = defaultdict(list)
        for state_key in non_terminal_states:
            layer = self.get_state_depth(state_key)
            states_by_layer[layer].append(state_key)
        
        for layer in sorted(states_by_layer.keys()):
            layer_states = states_by_layer[layer]
            print(f"\nProcessing Layer {layer}: {len(layer_states)} states")
            
            for i, state_key in enumerate(layer_states):
                try:
                    # Get probabilities for all possible next moves
                    prob_x, prob_o, prob_d = self.get_next_move_probabilities(state_key)
                    
                    if prob_x is not None:  # Valid state with moves
                        # Generate heatmap for each outcome type
                        prob_matrices = {
                            'P_X_wins': prob_x,
                            'P_O_wins': prob_o,
                            'P_draws': prob_d
                        }
                        
                        for outcome_type, prob_matrix in prob_matrices.items():
                            output_path = self.create_heatmap(state_key, prob_matrix, outcome_type)
                            if output_path:
                                generated_count += 1
                    
                    # Progress update
                    if (i + 1) % 20 == 0:
                        progress = (i + 1) / len(layer_states) * 100
                        print(f"  Layer {layer} progress: {i+1}/{len(layer_states)} ({progress:.1f}%)")
                        
                except Exception as e:
                    print(f"  Error processing state {state_key}: {e}")
                    continue
        
        print(f"\nGenerated {generated_count} heatmaps total")
        print(f"   (Expected ~{total_expected}, actual {generated_count})")
        
        return generated_count
    
    def generate_summary_statistics(self):
        """Generate summary statistics about the heatmaps"""
        print("\nGenerating summary statistics...")
        
        stats_by_layer = defaultdict(lambda: {
            'total_states': 0,
            'avg_moves_per_state': 0,
            'total_moves': 0
        })
        
        for state_key, node_data in self.game_tree.items():
            if not node_data['terminal']:
                layer = self.get_state_depth(state_key)
                num_moves = len(node_data['moves'])
                
                stats_by_layer[layer]['total_states'] += 1
                stats_by_layer[layer]['total_moves'] += num_moves
        
        # Calculate averages
        for layer, stats in stats_by_layer.items():
            if stats['total_states'] > 0:
                stats['avg_moves_per_state'] = stats['total_moves'] / stats['total_states']
        
        # Print summary
        print("\nSummary by Layer:")
        print(f"{'Layer':<6} {'States':<8} {'Total Moves':<12} {'Avg Moves/State':<15}")
        print("-" * 50)
        
        for layer in sorted(stats_by_layer.keys()):
            stats = stats_by_layer[layer]
            print(f"{layer:<6} {stats['total_states']:<8} {stats['total_moves']:<12} {stats['avg_moves_per_state']:<15.1f}")
        
        return stats_by_layer
    
    def run(self):
        """Run the complete heatmap generation process"""
        print("Complete Tic-Tac-Toe Tree Heatmap Generator")
        print("=" * 50)
        
        # Load game tree
        if not self.load_game_tree():
            return False
        
        # Create directory structure
        self.create_directory_structure()
        
        # Generate summary statistics
        self.generate_summary_statistics()
        
        # Generate all heatmaps
        generated_count = self.generate_all_heatmaps()
        
        print(f"\nComplete! Generated {generated_count} heatmaps")
        print(f"\nOutput structure:")
        print(f"  {self.base_output_dir}/")
        print(f"  ├── P_X_wins/layer_N/    (heatmaps showing P(X wins) for each next move)")
        print(f"  ├── P_O_wins/layer_N/    (heatmaps showing P(O wins) for each next move)")
        print(f"  └── P_draws/layer_N/     (heatmaps showing P(draws) for each next move)")
        
        print(f"\nEach heatmap shows:")
        print(f"  • Current board state (X and O pieces fixed)")
        print(f"  • Probability values if next move played at each empty position")
        print(f"  • Color intensity indicates probability magnitude")
        print(f"  • Use these to identify optimal moves visually!")
        
        return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python complete_heatmap_generator.py tic_tac_toe_3x3_complete_tree.json")
        print("\nThis script generates probability heatmaps from the complete game tree.")
        print("Each heatmap shows win probabilities if the next move is played at each position.")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        sys.exit(1)
    
    # Create generator and run
    generator = CompleteTreeHeatmapGenerator(json_file)
    success = generator.run()
    
    if success:
        print("\nHeatmaps ready for analysis!")
        print("Browse the layer folders to explore strategic patterns")
        print("Higher probability values indicate better moves")
    else:
        print("\nHeatmap generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()