#!/usr/bin/env python3
"""
Extract and Store Positional Probabilities from Complete Game Tree

This script reads the tic_tac_toe_3x3_complete_tree.json and calculates the probability
values for each position in each state, storing them in a structured JSON format for
easy access later without regenerating heatmaps.

Output format:
{
    "metadata": {
        "n": 3,
        "total_states": 5478,
        "generation_timestamp": "2025-10-14T16:54:00",
        "description": "Positional probabilities for each state-position combination"
    },
    "states": {
        "[0, 0, 0, 0, 0, 0, 0, 0, 0]": {
            "layer": 0,
            "player_to_move": "X",
            "positions": {
                "0": {"p_x_win": 0.585, "p_o_win": 0.288, "p_draw": 0.127},
                "1": {"p_x_win": 0.536, "p_o_win": 0.336, "p_draw": 0.128},
                ...
            }
        },
        ...
    }
}

Usage:
    python extract_positional_probabilities.py tic_tac_toe_3x3_complete_tree.json
"""

import json
import sys
import os
from datetime import datetime
from collections import defaultdict

class PositionalProbabilityExtractor:
    def __init__(self, json_file, n=3):
        """
        Initialize extractor with game tree JSON
        
        Args:
            json_file: Path to tic_tac_toe_3x3_complete_tree.json
            n: Board size (default 3)
        """
        self.json_file = json_file
        self.n = n
        self.tree_data = None
        self.game_tree = None
        self.probability_cache = {}  # Cache for calculated probabilities
        self.positional_data = {}
        
    def load_game_tree(self):
        """Load the complete game tree from JSON"""
        try:
            with open(self.json_file, 'r') as f:
                self.tree_data = json.load(f)
            
            self.game_tree = self.tree_data['tree']
            print(f"✓ Loaded complete game tree with {len(self.game_tree)} states")
            
            if 'n' in self.tree_data:
                self.n = self.tree_data['n']
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading game tree: {e}")
            return False
    
    def get_state_depth(self, state_key):
        """Get depth (number of moves) of a state"""
        state_list = eval(state_key)
        return sum(1 for x in state_list if x != 0)
    
    def calculate_probabilities_for_state(self, state_key):
        """
        Calculate probabilities for a state using exhaustive tree analysis
        Uses memoization for efficiency
        """
        # Check cache first
        if state_key in self.probability_cache:
            return self.probability_cache[state_key]
        
        if state_key not in self.game_tree:
            result = {'p_x_win': 0.0, 'p_o_win': 0.0, 'p_draw': 1.0}
            self.probability_cache[state_key] = result
            return result
        
        node_data = self.game_tree[state_key]
        
        # Terminal states have deterministic outcomes
        if node_data['terminal']:
            result_value = node_data['result']
            if result_value == 1:  # X wins
                result = {'p_x_win': 1.0, 'p_o_win': 0.0, 'p_draw': 0.0}
            elif result_value == -1:  # O wins
                result = {'p_x_win': 0.0, 'p_o_win': 1.0, 'p_draw': 0.0}
            else:  # Draw
                result = {'p_x_win': 0.0, 'p_o_win': 0.0, 'p_draw': 1.0}
            
            self.probability_cache[state_key] = result
            return result
        
        # Non-terminal: average over all possible moves
        total_x_win = 0.0
        total_o_win = 0.0
        total_draw = 0.0
        num_moves = len(node_data['moves'])
        
        if num_moves == 0:
            result = {'p_x_win': 0.0, 'p_o_win': 0.0, 'p_draw': 1.0}
            self.probability_cache[state_key] = result
            return result
        
        for move, next_state_list in node_data['moves']:
            next_state_key = str(next_state_list)
            next_probs = self.calculate_probabilities_for_state(next_state_key)
            
            total_x_win += next_probs['p_x_win']
            total_o_win += next_probs['p_o_win']
            total_draw += next_probs['p_draw']
        
        result = {
            'p_x_win': total_x_win / num_moves,
            'p_o_win': total_o_win / num_moves,
            'p_draw': total_draw / num_moves
        }
        
        self.probability_cache[state_key] = result
        return result
    
    def extract_positional_probabilities(self):
        """
        Extract positional probabilities for all states
        
        For each state, calculate probabilities if next move played at each empty position
        """
        print("Extracting positional probabilities from game tree...")
        
        total_states = len(self.game_tree)
        processed = 0
        
        # Group by layer for progress tracking
        states_by_layer = defaultdict(list)
        for state_key in self.game_tree.keys():
            layer = self.get_state_depth(state_key)
            states_by_layer[layer].append(state_key)
        
        for layer in sorted(states_by_layer.keys()):
            layer_states = states_by_layer[layer]
            print(f"Processing Layer {layer}: {len(layer_states)} states")
            
            for state_key in layer_states:
                node_data = self.game_tree[state_key]
                
                # Skip terminal states (no moves available)
                if node_data['terminal']:
                    processed += 1
                    continue
                
                # Extract positional probabilities
                current_player = node_data['player']
                player_name = 'X' if current_player == 1 else 'O'
                
                # Initialize state entry
                state_entry = {
                    'layer': layer,
                    'player_to_move': player_name,
                    'is_terminal': False,
                    'positions': {}
                }
                
                # For each possible move, get the resulting probabilities
                for move, next_state_list in node_data['moves']:
                    next_state_key = str(next_state_list)
                    
                    # Calculate probabilities for next state
                    next_probs = self.calculate_probabilities_for_state(next_state_key)
                    
                    # Store probabilities for this position
                    state_entry['positions'][str(move)] = {
                        'p_x_win': round(next_probs['p_x_win'], 6),
                        'p_o_win': round(next_probs['p_o_win'], 6),
                        'p_draw': round(next_probs['p_draw'], 6),
                        'next_state': next_state_list,
                        'row': move // self.n,
                        'col': move % self.n
                    }
                
                # Add to positional data
                self.positional_data[state_key] = state_entry
                processed += 1
                
                # Progress update
                if processed % 100 == 0:
                    progress = (processed / total_states) * 100
                    print(f"  Progress: {processed}/{total_states} ({progress:.1f}%)")
        
        print(f"✓ Extracted positional probabilities for {len(self.positional_data)} non-terminal states")
    
    def generate_summary_statistics(self):
        """Generate summary statistics about the positional data"""
        print("\nGenerating summary statistics...")
        
        stats = {
            'total_states': len(self.positional_data),
            'states_by_layer': defaultdict(int),
            'avg_positions_per_layer': defaultdict(lambda: {'count': 0, 'total': 0}),
            'total_position_probability_entries': 0
        }
        
        for state_key, state_data in self.positional_data.items():
            layer = state_data['layer']
            num_positions = len(state_data['positions'])
            
            stats['states_by_layer'][layer] += 1
            stats['avg_positions_per_layer'][layer]['count'] += 1
            stats['avg_positions_per_layer'][layer]['total'] += num_positions
            stats['total_position_probability_entries'] += num_positions
        
        # Calculate averages
        avg_positions = {}
        for layer, data in stats['avg_positions_per_layer'].items():
            if data['count'] > 0:
                avg_positions[layer] = data['total'] / data['count']
        
        # Print summary
        print("\nSummary Statistics:")
        print("-" * 60)
        print(f"Total non-terminal states: {stats['total_states']}")
        print(f"Total position-probability entries: {stats['total_position_probability_entries']}")
        print()
        print(f"{'Layer':<6} {'States':<8} {'Avg Positions/State':<20}")
        print("-" * 40)
        
        for layer in sorted(stats['states_by_layer'].keys()):
            count = stats['states_by_layer'][layer]
            avg_pos = avg_positions.get(layer, 0)
            print(f"{layer:<6} {count:<8} {avg_pos:<20.1f}")
        
        return stats
    
    def save_to_json(self, output_file='positional_probabilities.json'):
        """Save positional probabilities to JSON file"""
        print(f"\nSaving to {output_file}...")
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics()
        
        # Prepare output structure
        output_data = {
            'metadata': {
                'n': self.n,
                'source_file': self.json_file,
                'total_states': len(self.positional_data),
                'total_position_entries': summary_stats['total_position_probability_entries'],
                'generation_timestamp': datetime.now().isoformat(),
                'description': 'Positional probabilities showing P(X wins), P(O wins), P(draws) if next move played at each position',
                'probability_calculation': 'Exhaustive tree analysis: P(outcome|state) = average over all possible game paths',
                'states_by_layer': dict(summary_stats['states_by_layer'])
            },
            'states': self.positional_data
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Calculate file size
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"✓ Saved positional probabilities to {output_file}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  States: {len(self.positional_data)}")
        print(f"  Position entries: {summary_stats['total_position_probability_entries']}")
    
    def print_sample_data(self, num_samples=3):
        """Print sample entries from the positional data"""
        print(f"\nSample Positional Probability Data:")
        print("=" * 60)
        
        # Get a few sample states
        sample_states = list(self.positional_data.items())[:num_samples]
        
        for state_key, state_data in sample_states:
            print(f"\nState: {state_key}")
            print(f"Layer: {state_data['layer']}, Player to move: {state_data['player_to_move']}")
            print(f"Positions available: {len(state_data['positions'])}")
            print(f"Position probabilities:")
            
            for position, probs in list(state_data['positions'].items())[:3]:  # Show first 3 positions
                print(f"  Position {position} (row={probs['row']}, col={probs['col']}): "
                      f"P(X)={probs['p_x_win']:.3f}, P(O)={probs['p_o_win']:.3f}, P(D)={probs['p_draw']:.3f}")
    
    def run(self, output_file='positional_probabilities.json'):
        """Run the complete extraction process"""
        print("Positional Probability Extractor")
        print("=" * 50)
        
        # Load game tree
        if not self.load_game_tree():
            return False
        
        # Extract positional probabilities
        self.extract_positional_probabilities()
        
        # Save to JSON
        self.save_to_json(output_file)
        
        # Print sample data
        self.print_sample_data()
        
        print(f"\n✅ Complete! Positional probabilities saved to {output_file}")
        print(f"\n🎯 Usage:")
        print(f"   Load this JSON to quickly access probabilities for any state-position")
        print(f"   No need to recalculate or regenerate heatmaps")
        print(f"   Perfect for building game engines or analysis tools")
        
        return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python extract_positional_probabilities.py tic_tac_toe_3x3_complete_tree.json")
        print("\nThis script extracts and stores positional probabilities from the complete game tree.")
        print("Output: positional_probabilities.json")
        print("\nThe output JSON contains:")
        print("  • For each non-terminal state")
        print("  • For each empty position")
        print("  • P(X wins), P(O wins), P(draws) if next move played there")
        print("  • Easy lookup format for building applications")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        sys.exit(1)
    
    # Get optional output filename
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'positional_probabilities.json'
    
    # Create extractor and run
    extractor = PositionalProbabilityExtractor(json_file)
    success = extractor.run(output_file)
    
    if success:
        print(f"\n📊 JSON Structure:")
        print(f"{{")
        print(f"  'metadata': {{ ... }},")
        print(f"  'states': {{")
        print(f"    '[0,0,0,0,0,0,0,0,0]': {{")
        print(f"      'layer': 0,")
        print(f"      'player_to_move': 'X',")
        print(f"      'positions': {{")
        print(f"        '0': {{'p_x_win': 0.585, 'p_o_win': 0.288, 'p_draw': 0.127, ...}},")
        print(f"        '1': {{'p_x_win': 0.536, 'p_o_win': 0.336, 'p_draw': 0.128, ...}},")
        print(f"        ...")
        print(f"      }}")
        print(f"    }},")
        print(f"    ...")
        print(f"  }}")
        print(f"}}")
    else:
        print("\n❌ Extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()