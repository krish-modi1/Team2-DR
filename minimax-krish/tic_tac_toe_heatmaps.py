import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from collections import defaultdict

class TicTacToeHeatmapGenerator:
    """
    Generates various heatmaps and visualizations from 3x3 Tic-Tac-Toe minimax analysis data.
    """
    
    def __init__(self, states_file="tic_tac_toe_3x3_states.json", 
                 moves_file="tic_tac_toe_3x3_moves.json"):
        """
        Initialize the heatmap generator by loading analysis data.
        
        Args:
            states_file (str): Path to the states JSON file
            moves_file (str): Path to the moves JSON file
        """
        self.states_file = states_file
        self.moves_file = moves_file
        self.states_data = {}
        self.moves_data = {}
        self.load_data()
    
    def load_data(self):
        """
        Load the minimax analysis data from JSON files.
        """
        try:
            print(f"Loading states data from {self.states_file}...")
            with open(self.states_file, 'r') as f:
                self.states_data = json.load(f)
            print(f"Loaded {len(self.states_data)} unique board states.")
            
            print(f"Loading moves data from {self.moves_file}...")
            with open(self.moves_file, 'r') as f:
                self.moves_data = json.load(f)
            print(f"Loaded move scores for {len(self.moves_data)} board positions.")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find data file: {e}")
            print("Please run the main minimax analysis first to generate the JSON files.")
            raise
    
    def create_position_value_heatmap(self):
        """
        Create a heatmap showing the average minimax score for each board position
        across all game states where that position was a valid move.
        
        Returns:
            np.ndarray: 3x3 array of average position values
        """
        print("Generating position value heatmap...")
        
        # Initialize arrays to track scores and counts for each position
        position_scores = np.zeros((3, 3))
        position_counts = np.zeros((3, 3))
        
        # Aggregate scores for each position across all states
        for state_hash, moves_list in self.moves_data.items():
            for move_data in moves_list:
                row, col = move_data['move']
                score = move_data['score']
                
                # Add to running totals
                position_scores[row, col] += score
                position_counts[row, col] += 1
        
        # Calculate average scores (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            average_scores = np.divide(position_scores, position_counts, 
                                     out=np.zeros_like(position_scores), 
                                     where=position_counts!=0)
        
        return average_scores
    
    def create_opening_move_heatmap(self):
        """
        Create a heatmap specifically for opening moves from the empty board.
        
        Returns:
            np.ndarray: 3x3 array of opening move scores
        """
        print("Generating opening move heatmap...")
        
        # Empty board hash (all zeros)
        empty_board_hash = str(tuple([0] * 9))
        opening_scores = np.zeros((3, 3))
        
        if empty_board_hash in self.moves_data:
            moves_list = self.moves_data[empty_board_hash]
            for move_data in moves_list:
                row, col = move_data['move']
                score = move_data['score']
                opening_scores[row, col] = score
        else:
            print("Warning: No opening moves data found!")
        
        return opening_scores
    
    def create_move_probability_heatmap(self, state_hash=None):
        """
        Create a probability heatmap for moves in a specific state.
        Converts minimax scores to probabilities using softmax transformation.
        
        Args:
            state_hash (str): Hash of the board state to analyze. If None, uses empty board.
            
        Returns:
            np.ndarray: 3x3 array of move probabilities
        """
        if state_hash is None:
            state_hash = str(tuple([0] * 9))  # Empty board
        
        print(f"Generating move probability heatmap for state: {state_hash[:20]}...")
        
        probabilities = np.zeros((3, 3))
        
        if state_hash in self.moves_data:
            moves_list = self.moves_data[state_hash]
            scores = [move_data['score'] for move_data in moves_list]
            
            # Apply softmax to convert scores to probabilities
            # Subtract max for numerical stability
            scores_array = np.array(scores)
            exp_scores = np.exp(scores_array - np.max(scores_array))
            softmax_probs = exp_scores / np.sum(exp_scores)
            
            # Map probabilities back to board positions
            for i, move_data in enumerate(moves_list):
                row, col = move_data['move']
                probabilities[row, col] = softmax_probs[i]
        
        return probabilities
    
    def plot_heatmap(self, data, title, filename=None, cmap='viridis', 
                     show_values=True, vmin=None, vmax=None):
        """
        Plot a heatmap with proper formatting and labels.
        
        Args:
            data (np.ndarray): 3x3 data array to plot
            title (str): Title for the plot
            filename (str): If provided, save plot to this file
            cmap (str): Colormap to use
            show_values (bool): Whether to show values in cells
            vmin, vmax (float): Color scale limits
        """
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        ax = sns.heatmap(data, 
                        annot=show_values,
                        fmt='.3f' if show_values else '',
                        cmap=cmap,
                        cbar_kws={'label': 'Score/Probability'},
                        square=True,
                        linewidths=0.5,
                        vmin=vmin,
                        vmax=vmax)
        
        # Customize appearance
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Column', fontsize=12)
        plt.ylabel('Row', fontsize=12)
        
        # Add position labels
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(['0', '1', '2'])
        ax.set_yticklabels(['0', '1', '2'])
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {filename}")
        
        plt.show()
    
    def analyze_position_patterns(self):
        """
        Analyze and print patterns discovered in position values.
        """
        print("\n" + "="*60)
        print("POSITION PATTERN ANALYSIS")
        print("="*60)
        
        # Get position values
        position_values = self.create_position_value_heatmap()
        
        print("Average minimax scores by position:")
        print(position_values)
        
        # Find best and worst positions
        max_score = np.max(position_values)
        min_score = np.min(position_values[position_values > 0])  # Exclude zeros
        
        max_positions = np.where(position_values == max_score)
        min_positions = np.where(position_values == min_score)
        
        print(f"\nBest positions (score {max_score:.3f}):")
        for i in range(len(max_positions[0])):
            row, col = max_positions[0][i], max_positions[1][i]
            print(f"  Position ({row},{col}): {self.get_position_name(row, col)}")
        
        print(f"\nWorst positions (score {min_score:.3f}):")
        for i in range(len(min_positions[0])):
            row, col = min_positions[0][i], min_positions[1][i]
            print(f"  Position ({row},{col}): {self.get_position_name(row, col)}")
        
        # Analyze position types
        center_score = position_values[1, 1]
        corner_scores = [position_values[0, 0], position_values[0, 2], 
                        position_values[2, 0], position_values[2, 2]]
        edge_scores = [position_values[0, 1], position_values[1, 0], 
                      position_values[1, 2], position_values[2, 1]]
        
        print(f"\nPosition type analysis:")
        print(f"  Center (1,1): {center_score:.3f}")
        print(f"  Corners average: {np.mean(corner_scores):.3f}")
        print(f"  Edges average: {np.mean(edge_scores):.3f}")
        
        return position_values
    
    def get_position_name(self, row, col):
        """
        Get a descriptive name for a board position.
        
        Args:
            row, col (int): Position coordinates
            
        Returns:
            str: Descriptive name
        """
        if row == 1 and col == 1:
            return "Center"
        elif (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return "Corner"
        else:
            return "Edge"
    
    def analyze_opening_moves(self):
        """
        Analyze and visualize opening move preferences.
        """
        print("\n" + "="*60)
        print("OPENING MOVE ANALYSIS")
        print("="*60)
        
        opening_scores = self.create_opening_move_heatmap()
        
        # Find all opening moves and their scores
        opening_moves = []
        for i in range(3):
            for j in range(3):
                if opening_scores[i, j] != 0:
                    opening_moves.append(((i, j), opening_scores[i, j]))
        
        # Sort by score (descending)
        opening_moves.sort(key=lambda x: x[1], reverse=True)
        
        print("Opening moves ranked by minimax score:")
        for i, ((row, col), score) in enumerate(opening_moves):
            position_name = self.get_position_name(row, col)
            print(f"  {i+1}. Position ({row},{col}) - {position_name}: Score = {score}")
        
        # Identify optimal opening moves
        best_score = opening_moves[0][1] if opening_moves else 0
        optimal_moves = [move for move, score in opening_moves if score == best_score]
        
        print(f"\nOptimal opening moves (score {best_score}):")
        for row, col in optimal_moves:
            position_name = self.get_position_name(row, col)
            print(f"  Position ({row},{col}) - {position_name}")
        
        return opening_scores
    
    def generate_all_heatmaps(self):
        """
        Generate and display all heatmap visualizations.
        """
        print("Generating all heatmap visualizations...")
        
        # 1. Position value heatmap
        position_values = self.create_position_value_heatmap()
        self.plot_heatmap(position_values, 
                         "3x3 Tic-Tac-Toe: Average Position Values",
                         "position_values_heatmap.png",
                         cmap='RdYlBu_r')
        
        # 2. Opening move heatmap
        opening_scores = self.create_opening_move_heatmap()
        self.plot_heatmap(opening_scores,
                         "3x3 Tic-Tac-Toe: Opening Move Scores", 
                         "opening_moves_heatmap.png",
                         cmap='RdYlGn')
        
        # 3. Opening move probabilities
        opening_probs = self.create_move_probability_heatmap()
        self.plot_heatmap(opening_probs,
                         "3x3 Tic-Tac-Toe: Opening Move Probabilities",
                         "opening_probabilities_heatmap.png", 
                         cmap='plasma',
                         vmin=0, vmax=1)
        
        print("All heatmaps generated successfully!")
    
    def create_statistical_summary(self):
        """
        Create a comprehensive statistical summary of the analysis.
        """
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        # Count game outcomes
        x_wins = sum(1 for state in self.states_data.values() 
                    if state.get('terminal', False) and state.get('winner') == 1)
        o_wins = sum(1 for state in self.states_data.values() 
                    if state.get('terminal', False) and state.get('winner') == -1)
        draws = sum(1 for state in self.states_data.values() 
                   if state.get('terminal', False) and state.get('winner') == 0)
        terminal_states = x_wins + o_wins + draws
        
        print(f"Total unique board states: {len(self.states_data)}")
        print(f"Terminal states: {terminal_states}")
        print(f"  - X wins: {x_wins} ({100*x_wins/terminal_states:.1f}%)")
        print(f"  - O wins: {o_wins} ({100*o_wins/terminal_states:.1f}%)")
        print(f"  - Draws: {draws} ({100*draws/terminal_states:.1f}%)")
        
        # Depth analysis
        depths = [state['depth'] for state in self.states_data.values()]
        print(f"\nGame depth statistics:")
        print(f"  - Maximum depth: {max(depths)}")
        print(f"  - Average depth: {np.mean(depths):.2f}")
        print(f"  - Most common depth: {max(set(depths), key=depths.count)}")
        
        return {
            'total_states': len(self.states_data),
            'x_wins': x_wins,
            'o_wins': o_wins, 
            'draws': draws,
            'max_depth': max(depths),
            'avg_depth': np.mean(depths)
        }


def main():
    """
    Main function to generate all heatmaps and analysis.
    """
    print("3x3 Tic-Tac-Toe Heatmap and Probability Analysis")
    print("=" * 50)
    
    try:
        # Initialize the heatmap generator
        generator = TicTacToeHeatmapGenerator()
        
        # Generate statistical summary
        stats = generator.create_statistical_summary()
        
        # Analyze position patterns
        position_values = generator.analyze_position_patterns()
        
        # Analyze opening moves
        opening_scores = generator.analyze_opening_moves()
        
        # Generate all heatmap visualizations
        generator.generate_all_heatmaps()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Key findings:")
        print(f"1. Total game states analyzed: {stats['total_states']}")
        print(f"2. With perfect play: {stats['draws']} draws, {stats['x_wins']} X wins, {stats['o_wins']} O wins")
        print(f"3. Best opening position: Center with score {position_values[1,1]:.3f}")
        print(f"4. Average corner value: {np.mean([position_values[0,0], position_values[0,2], position_values[2,0], position_values[2,2]]):.3f}")
        print(f"5. Average edge value: {np.mean([position_values[0,1], position_values[1,0], position_values[1,2], position_values[2,1]]):.3f}")
        print("\nAll heatmaps saved as PNG files for presentation use.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("\nMake sure to run the main minimax analysis first to generate the required JSON files.")


if __name__ == "__main__":
    main()