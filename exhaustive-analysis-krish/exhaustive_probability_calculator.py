#!/usr/bin/env python3
"""
Exhaustive Game Tree Probability Calculator

This script calculates exact win/loss/draw probabilities for each (game_state, player_to_move)
combination using pure mathematical analysis of the exhaustive game tree structure.

Mathematical Foundation:
P(outcome | state, player) = (1/n) x Σ P(outcome | state_after_move_i)
where n = number of valid moves from current state

This represents the mathematical expectation over ALL possible game continuations,
giving fractional probabilities for intermediate game states.

Key Format: "[game_state]_[player_to_move]"
- game_state: list like [0,0,0,0,0,0,0,0,0] where 0=empty, 1=X, -1=O
- player_to_move: 1=X, -1=O

Usage:
    calculator = ExhaustiveTreeProbabilityCalculator('your_game_tree.json')
    calculator.load_game_tree()
    probabilities = calculator.calculate_all_probabilities()
    calculator.add_metadata()
    calculator.save_probabilities_json('output_probabilities.json')
"""

import json
from collections import defaultdict

class ExhaustiveTreeProbabilityCalculator:
    """
    Calculate exact probabilities from exhaustive game tree using combinatorial analysis.
    
    Mathematical Foundation:
    P(outcome | state, player) = Mathematical expectation over all possible game paths
                                from (state, player_to_move) combination
    """
    
    def __init__(self, game_tree_json_file):
        """
        Initialize calculator with exhaustive game tree JSON file
        
        Args:
            game_tree_json_file: Path to JSON file containing complete game tree
        """
        self.tree_file = game_tree_json_file
        self.tree_data = None
        self.game_tree = None
        self.n = None
        self.probabilities = {}  # Will store: {state_player_key: {p_x_win, p_o_win, p_draw}}
        
    def load_game_tree(self):
        """Load the exhaustive game tree from JSON file"""
        try:
            with open(self.tree_file, 'r') as f:
                self.tree_data = json.load(f)
            
            self.n = self.tree_data['n']
            self.game_tree = self.tree_data['tree']
            
            print(f"✓ Loaded {self.n}x{self.n} exhaustive game tree")
            print(f"✓ Total states in tree: {len(self.game_tree):,}")
            return True
            
        except FileNotFoundError:
            print(f"❌ Game tree file '{self.tree_file}' not found!")
            return False
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Invalid game tree format: {e}")
            return False
    
    def state_to_key(self, state_list, player_to_move):
        """
        Convert game state and player to unique key
        
        Args:
            state_list: List representing board [0=empty, 1=X, -1=O]
            player_to_move: 1=X, -1=O
            
        Returns:
            String key in format "[state_list]_[player]"
        """
        return f"{state_list}_{player_to_move}"
    
    def key_to_state_player(self, key):
        """
        Extract state and player from key
        
        Args:
            key: String key in format "[state_list]_[player]"
            
        Returns:
            (state_list, player) tuple
        """
        parts = key.rsplit('_', 1)
        state_str = parts[0]
        player = int(parts[1])
        state = eval(state_str)  # Convert string representation back to list
        return state, player
    
    def calculate_all_probabilities(self):
        """
        Calculate probabilities for all (state, player_to_move) combinations
        
        Algorithm:
        1. Process states by depth (deepest first)
        2. Terminal states have deterministic outcomes
        3. Non-terminal states: P(outcome) = average over all possible moves
        
        Returns:
            Dictionary mapping state-player keys to probability dictionaries
        """
        print("Calculating exhaustive tree probabilities...")
        print("Method: Combinatorial analysis - equal weight to all possible moves")
        
        # Group states by depth for processing order
        states_by_depth = defaultdict(list)
        
        for state_key, node_data in self.game_tree.items():
            state_list = eval(state_key)
            depth = sum(1 for x in state_list if x != 0)
            player_to_move = node_data['player']
            full_key = self.state_to_key(state_list, player_to_move)
            states_by_depth[depth].append((full_key, state_key, node_data))
        
        # Process from deepest (terminal) to shallowest (root)
        for depth in sorted(states_by_depth.keys(), reverse=True):
            print(f"Processing depth {depth}: {len(states_by_depth[depth])} states")
            
            for full_key, state_key, node_data in states_by_depth[depth]:
                self._calculate_state_probability(full_key, node_data)
        
        print(f"✓ Calculated probabilities for {len(self.probabilities):,} state-player combinations")
        return self.probabilities
    
    def _calculate_state_probability(self, full_key, node_data):
        """
        Calculate probability for a single (state, player) combination
        
        Args:
            full_key: Unique key for this state-player combination
            node_data: Node data from game tree
            
        Returns:
            Dictionary with p_x_win, p_o_win, p_draw probabilities
        """
        if full_key in self.probabilities:
            return self.probabilities[full_key]
        
        # Base case: Terminal states have deterministic outcomes
        if node_data['terminal']:
            result = node_data['result']
            if result == 1:  # X wins
                probs = {'p_x_win': 1.0, 'p_o_win': 0.0, 'p_draw': 0.0}
            elif result == -1:  # O wins
                probs = {'p_x_win': 0.0, 'p_o_win': 1.0, 'p_draw': 0.0}
            else:  # Draw (result == 0)
                probs = {'p_x_win': 0.0, 'p_o_win': 0.0, 'p_draw': 1.0}
            
            self.probabilities[full_key] = probs
            return probs
        
        # Recursive case: Non-terminal states
        # Mathematical expectation over all possible moves
        total_x_win = 0.0
        total_o_win = 0.0
        total_draw = 0.0
        num_moves = len(node_data['moves'])
        
        current_player = node_data['player']
        
        for move, next_state_list in node_data['moves']:
            next_state_key = str(next_state_list)
            
            if next_state_key in self.game_tree:
                # After this move, it's opponent's turn
                next_player = -current_player
                next_full_key = self.state_to_key(next_state_list, next_player)
                
                # Get or calculate probabilities for next state
                if next_full_key in self.probabilities:
                    next_probs = self.probabilities[next_full_key]
                else:
                    next_node_data = self.game_tree[next_state_key]
                    next_probs = self._calculate_state_probability(next_full_key, next_node_data)
                
                # Accumulate probabilities
                total_x_win += next_probs['p_x_win']
                total_o_win += next_probs['p_o_win']
                total_draw += next_probs['p_draw']
        
        # Calculate mathematical expectation (uniform distribution over moves)
        if num_moves > 0:
            probs = {
                'p_x_win': total_x_win / num_moves,
                'p_o_win': total_o_win / num_moves,
                'p_draw': total_draw / num_moves
            }
        else:
            # Should not happen in well-formed tree
            probs = {'p_x_win': 0.0, 'p_o_win': 0.0, 'p_draw': 1.0}
        
        self.probabilities[full_key] = probs
        return probs
    
    def add_metadata(self):
        """Add useful metadata to each probability entry"""
        print("Adding metadata to probability entries...")
        
        enhanced_probabilities = {}
        
        for full_key, probs in self.probabilities.items():
            state_list, player_to_move = self.key_to_state_player(full_key)
            state_key = str(state_list)
            
            if state_key in self.game_tree:
                node_data = self.game_tree[state_key]
                
                # Create enhanced entry with metadata
                enhanced_entry = probs.copy()
                enhanced_entry['metadata'] = {
                    'game_state': state_list,
                    'player_to_move': 'X' if player_to_move == 1 else 'O',
                    'depth': sum(1 for x in state_list if x != 0),
                    'is_terminal': node_data['terminal'],
                    'num_valid_moves': len(node_data['moves']) if not node_data['terminal'] else 0
                }
                
                if node_data['terminal']:
                    result = node_data['result']
                    if result == 1:
                        enhanced_entry['metadata']['terminal_outcome'] = 'X_wins'
                    elif result == -1:
                        enhanced_entry['metadata']['terminal_outcome'] = 'O_wins'
                    else:
                        enhanced_entry['metadata']['terminal_outcome'] = 'draw'
                
                enhanced_probabilities[full_key] = enhanced_entry
        
        self.probabilities = enhanced_probabilities
        print(f"✓ Added metadata to {len(self.probabilities):,} entries")
    
    def save_probabilities_json(self, output_filename):
        """
        Save calculated probabilities to JSON file and CSV file
        
        Args:
            output_filename: Path for output JSON file (CSV will have same name with .csv extension)
            
        Returns:
            Tuple of (json_filename, csv_filename)
        """
        import csv
        import os
        
        # Find starting position probabilities
        empty_state = [0] * (self.n * self.n)
        start_key = self.state_to_key(empty_state, 1)  # X starts first
        start_probs = self.probabilities.get(start_key, {})
        
        # Calculate summary statistics
        total_states = len(self.probabilities)
        x_favored = sum(1 for p in self.probabilities.values() 
                       if p.get('p_x_win', 0) > max(p.get('p_o_win', 0), p.get('p_draw', 0)))
        o_favored = sum(1 for p in self.probabilities.values() 
                       if p.get('p_o_win', 0) > max(p.get('p_x_win', 0), p.get('p_draw', 0)))
        draw_favored = sum(1 for p in self.probabilities.values() 
                          if p.get('p_draw', 0) >= max(p.get('p_x_win', 0), p.get('p_o_win', 0)))
        
        # Prepare JSON data
        output_data = {
            'n': self.n,
            'description': f'Exhaustive tree probabilities for {self.n}x{self.n} tic-tac-toe',
            'mathematical_method': 'Combinatorial analysis: P(outcome|state,player) = average over all possible moves',
            'key_format': '[game_state]_[player_to_move] where player: 1=X, -1=O',
            'cell_encoding': '0=empty, 1=X, -1=O',
            'total_state_player_combinations': total_states,
            'summary_statistics': {
                'starting_position': {
                    'key': start_key,
                    'state': empty_state,
                    'player_to_move': 'X',
                    'probabilities': start_probs
                },
                'distribution': {
                    'states_favoring_x': x_favored,
                    'states_favoring_o': o_favored, 
                    'states_favoring_draw': draw_favored
                }
            },
            'probabilities': self.probabilities
        }
        
        # Save JSON file
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Create CSV filename
        base_name = os.path.splitext(output_filename)[0]
        csv_filename = base_name + '.csv'
        
        # Prepare CSV data
        csv_rows = []
        
        for full_key, prob_data in self.probabilities.items():
            state_list, player_to_move = self.key_to_state_player(full_key)
            
            # Calculate layer (depth) - number of moves made
            layer = sum(1 for x in state_list if x != 0)
            
            # Convert state to string representation
            state_str = ''.join(str(x) for x in state_list)
            
            # Convert player to string
            to_move = 'X' if player_to_move == 1 else 'O'
            
            # Extract probabilities
            p_x_wins = prob_data['p_x_win']
            p_o_wins = prob_data['p_o_win']
            p_draws = prob_data['p_draw']
            
            csv_rows.append([
                state_str,      # state
                to_move,        # to_move
                layer,          # layer
                p_x_wins,       # P(X wins)
                p_o_wins,       # P(O wins)
                p_draws         # P(draws)
            ])
        
        # Sort CSV rows by layer, then by state for better organization
        csv_rows.sort(key=lambda row: (row[2], row[0]))  # Sort by layer, then state
        
        # Write CSV file
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['state', 'to_move', 'layer', 'P(X wins)', 'P(O wins)', 'P(draws)'])
            
            # Write data rows
            writer.writerows(csv_rows)
        
        print(f"✓ Saved probabilities to {output_filename}")
        print(f"✓ Saved CSV data to {csv_filename}")
        print(f"  Total state-player combinations: {total_states:,}")
        print(f"  JSON file size: ~{len(json.dumps(output_data)) / 1024 / 1024:.1f} MB")
        print(f"  CSV rows: {len(csv_rows):,} (plus header)")
        
        return output_filename, csv_filename
    
    def get_best_move(self, current_state, current_player):
        """
        Find best move for current player using calculated probabilities
        
        Args:
            current_state: List representing current board state
            current_player: 1=X, -1=O
            
        Returns:
            (best_move_info, current_probabilities, all_move_analysis) tuple
        """
        current_key = self.state_to_key(current_state, current_player)
        
        if current_key not in self.probabilities:
            print(f"State-player combination not found: {current_key}")
            return None, None, None
        
        current_state_key = str(current_state)
        if current_state_key not in self.game_tree:
            print(f"State not found in game tree: {current_state_key}")
            return None, None, None
        
        node_data = self.game_tree[current_state_key]
        
        # If terminal, no moves possible
        if node_data['terminal']:
            return None, self.probabilities[current_key], None
        
        # Analyze all possible moves
        move_analysis = []
        best_move = None
        best_win_prob = -1
        
        for move, next_state_list in node_data['moves']:
            next_player = -current_player
            next_key = self.state_to_key(next_state_list, next_player)
            
            if next_key in self.probabilities:
                next_probs = self.probabilities[next_key]
                
                # Get win probability for current player after this move
                if current_player == 1:  # X
                    win_prob = next_probs['p_x_win']
                else:  # O
                    win_prob = next_probs['p_o_win']
                
                move_info = {
                    'move_position': move,
                    'coordinates': (move // self.n, move % self.n),
                    'resulting_state': next_state_list,
                    'win_probability': win_prob,
                    'full_probabilities': next_probs
                }
                move_analysis.append(move_info)
                
                if win_prob > best_win_prob:
                    best_win_prob = win_prob
                    best_move = move_info
        
        # Sort moves by win probability (best first)
        move_analysis.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return best_move, self.probabilities[current_key], move_analysis
    
    def print_sample_analysis(self, num_samples=10):
        """Print sample of calculated probabilities for inspection"""
        print(f"\nSample Probability Analysis ({num_samples} examples):")
        print("-" * 85)
        print(f"{'State':<15} {'Player':<8} {'P(X win)':<10} {'P(O win)':<10} {'P(Draw)':<10} {'Depth':<6}")
        print("-" * 85)
        
        # Sort by depth for better understanding
        sorted_items = sorted(self.probabilities.items(), 
                             key=lambda x: x[1].get('metadata', {}).get('depth', 0))
        
        count = 0
        for full_key, prob_data in sorted_items:
            if count >= num_samples:
                break
            
            state_list, player = self.key_to_state_player(full_key)
            
            # Create compact state representation
            compact_state = ''.join('.' if x == 0 else ('X' if x == 1 else 'O') 
                                   for x in state_list)
            
            player_str = 'X' if player == 1 else 'O'
            depth = prob_data.get('metadata', {}).get('depth', 0)
            
            print(f"{compact_state:<15} {player_str:<8} {prob_data['p_x_win']:<10.3f} "
                  f"{prob_data['p_o_win']:<10.3f} {prob_data['p_draw']:<10.3f} {depth:<6}")
            
            count += 1
    
    def analyze_position(self, current_state, current_player):
        """
        Comprehensive analysis of a specific position
        
        Args:
            current_state: List representing board state
            current_player: 1=X, -1=O
        """
        print(f"\n{'='*60}")
        print(f"POSITION ANALYSIS: {'X' if current_player == 1 else 'O'} to move")
        print(f"{'='*60}")
        
        # Display board
        print("Current board:")
        for i in range(self.n):
            row = ""
            for j in range(self.n):
                val = current_state[i * self.n + j]
                if val == 0: row += ". "
                elif val == 1: row += "X "
                else: row += "O "
            print(row)
        print()
        
        # Get analysis
        best_move, current_probs, move_analysis = self.get_best_move(current_state, current_player)
        
        if current_probs:
            print("Current position probabilities:")
            print(f"P(X wins) = {current_probs['p_x_win']:.3f}")
            print(f"P(O wins) = {current_probs['p_o_win']:.3f}")
            print(f"P(Draw) = {current_probs['p_draw']:.3f}")
            print()
        
        if best_move:
            print(f"Best move: position {best_move['move_position']} {best_move['coordinates']}")
            print(f"Win probability after best move: {best_move['win_probability']:.3f}")
            print()
            
            if move_analysis:
                print("All move options ranked by win probability:")
                for i, move_info in enumerate(move_analysis):
                    pos = move_info['move_position']
                    coords = move_info['coordinates']
                    win_prob = move_info['win_probability']
                    print(f"  {i+1}. Position {pos} {coords}: P(win) = {win_prob:.3f}")


def main():
    """Example usage of the probability calculator"""
    print("Exhaustive Game Tree Probability Calculator")
    print("=" * 50)
    print("Calculates exact win/loss/draw probabilities for each")
    print("(game_state, player_to_move) combination using mathematical")
    print("analysis of the complete game tree structure.")
    print()
    
    # Get input files
    input_file = input("Enter game tree JSON filename: ").strip()
    if not input_file:
        print("No filename provided!")
        return
    
    output_file = input("Enter output probabilities filename (default: probabilities.json): ").strip()
    if not output_file:
        output_file = "probabilities.json"
    
    # Initialize and run calculator
    calculator = ExhaustiveTreeProbabilityCalculator(input_file)
    
    if not calculator.load_game_tree():
        return
    
    print(f"\nCalculating probabilities...")
    probabilities = calculator.calculate_all_probabilities()
    
    print(f"Adding metadata...")
    calculator.add_metadata()
    
    print(f"Saving results...")
    calculator.save_probabilities_json(output_file)
    
    # Show sample analysis
    calculator.print_sample_analysis(15)
    
    print(f"\n✅ SUCCESS!")
    print(f"Created probability lookup table with {len(probabilities):,} entries")
    print(f"Use this for optimal move selection:")
    print(f"best_move = argmax P(current_player_wins | state_after_move)")


if __name__ == "__main__":
    main()