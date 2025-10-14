#!/usr/bin/env python3
"""
Tic-Tac-Toe Game Tree Visualizer

This script reads a JSON file containing a complete tic-tac-toe game tree
and provides various visualization and exploration tools.

Usage:
    python game_tree_visualizer.py

Features:
- Load and analyze game tree statistics
- Display ASCII tree structure 
- Find and show specific game paths
- Interactive tree exploration
- Works with any nxn tic-tac-toe game tree

Author: Generated for tic-tac-toe analysis
"""

import json
from collections import deque, defaultdict

class GameTreeVisualizer:
    def __init__(self, json_filename):
        """Initialize the visualizer with a game tree JSON file"""
        self.json_filename = json_filename
        self.data = None
        self.n = None
        self.tree = None
        
    def load_json(self):
        """Load the game tree from JSON file"""
        try:
            with open(self.json_filename, 'r') as f:
                self.data = json.load(f)
            self.n = self.data['n']
            self.tree = self.data['tree']
            print(f"✓ Loaded {self.n}x{self.n} tic-tac-toe game tree")
            print(f"✓ Total states: {len(self.tree)}")
            return True
        except FileNotFoundError:
            print(f"❌ File {self.json_filename} not found!")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {self.json_filename}")
            return False
    
    def state_key_to_tuple(self, state_key):
        """Convert string representation back to tuple"""
        return tuple(eval(state_key))
    
    def format_board(self, state_tuple, compact=False):
        """Format board state for display"""
        if compact:
            # Single line format
            result = ""
            for val in state_tuple:
                if val == 0:
                    result += "."
                elif val == 1:
                    result += "X"
                else:
                    result += "O"
            return result
        else:
            # Multi-line format
            lines = []
            for i in range(self.n):
                line = ""
                for j in range(self.n):
                    val = state_tuple[i * self.n + j]
                    if val == 0:
                        line += "."
                    elif val == 1:
                        line += "X"
                    else:
                        line += "O"
                    if j < self.n - 1:
                        line += " "
                lines.append(line)
            return "\n".join(lines)
    
    def get_state_depth(self, state_tuple):
        """Get the depth (number of moves) of a state"""
        return sum(1 for x in state_tuple if x != 0)
    
    def get_state_info(self, state_key):
        """Get formatted info about a state"""
        state_tuple = self.state_key_to_tuple(state_key)
        node_data = self.tree[state_key]
        
        info = {
            'board': self.format_board(state_tuple),
            'compact_board': self.format_board(state_tuple, compact=True),
            'depth': self.get_state_depth(state_tuple),
            'player': 'X' if node_data['player'] == 1 else 'O',
            'terminal': node_data['terminal'],
            'result': node_data['result'],
            'num_moves': len(node_data['moves']) if not node_data['terminal'] else 0
        }
        
        if info['terminal']:
            if info['result'] == 1:
                info['outcome'] = "X WINS"
            elif info['result'] == -1:
                info['outcome'] = "O WINS"
            else:
                info['outcome'] = "DRAW"
        
        return info
    
    def show_statistics(self):
        """Display comprehensive statistics about the game tree"""
        if not self.load_json():
            return
        
        stats = self.data['statistics']
        
        print("\n" + "="*50)
        print(f"📊 GAME TREE STATISTICS ({self.n}x{self.n})")
        print("="*50)
        
        print(f"🎯 Total unique states: {stats['total_states']:,}")
        print(f"🏁 Terminal states: {sum(stats['terminal_outcomes'].values()):,}")
        print(f"🔄 Non-terminal states: {stats['total_states'] - sum(stats['terminal_outcomes'].values()):,}")
        
        print("\n📈 States by depth (number of moves):")
        for depth in sorted(stats['states_by_moves'].keys(), key=int):
            count = stats['states_by_moves'][depth]
            bar = "█" * min(50, count // max(1, max(stats['states_by_moves'].values()) // 50))
            print(f"  {depth:2}: {count:4} states {bar}")
        
        print("\n🏆 Terminal outcomes:")
        outcomes = stats['terminal_outcomes']
        total_terminal = sum(outcomes.values())
        for outcome, count in outcomes.items():
            pct = (count / total_terminal * 100) if total_terminal > 0 else 0
            print(f"  {outcome:8}: {count:3} ({pct:5.1f}%)")
    
    def show_tree_structure(self, max_depth=3, max_children=3):
        """Display ASCII tree structure"""
        if not self.load_json():
            return
        
        print(f"\n🌳 TREE STRUCTURE (depth ≤ {max_depth}, showing ≤ {max_children} children)")
        print("="*60)
        
        start_key = str([0] * (self.n * self.n))
        self._print_subtree(start_key, "", True, max_depth, max_children, 0)
    
    def _print_subtree(self, state_key, prefix, is_last, max_depth, max_children, current_depth):
        """Recursively print tree structure"""
        if current_depth > max_depth or state_key not in self.tree:
            return
        
        info = self.get_state_info(state_key)
        
        # Print current node
        connector = "└── " if is_last else "├── "
        status = f"[{info['outcome']}]" if info['terminal'] else f"[{info['player']} to move]"
        
        print(f"{prefix}{connector}{info['compact_board']} {status}")
        
        if not info['terminal'] and current_depth < max_depth:
            node_data = self.tree[state_key]
            children = node_data['moves'][:max_children]  # Limit children shown
            
            for i, (move, next_state_list) in enumerate(children):
                next_key = str(next_state_list)
                is_last_child = (i == len(children) - 1)
                
                new_prefix = prefix + ("    " if is_last else "│   ")
                
                # Show the move being made
                row, col = move // self.n, move % self.n
                move_prefix = prefix + ("    " if is_last else "│   ")
                move_connector = "└─move─> " if is_last_child else "├─move─> "
                print(f"{move_prefix}{move_connector}pos {move} ({row},{col})")
                
                self._print_subtree(next_key, new_prefix, is_last_child, max_depth, max_children, current_depth + 1)
            
            if len(node_data['moves']) > max_children:
                extra = len(node_data['moves']) - max_children
                ellipsis_prefix = prefix + ("    " if is_last else "│   ")
                print(f"{ellipsis_prefix}⋮ ({extra} more children)")
    
    def find_game_path(self, target_depth=None, target_outcome=None):
        """Find and display a specific game path"""
        if not self.load_json():
            return
        
        print(f"\n🎯 FINDING GAME PATH")
        print("="*40)
        
        # Find a suitable target state
        target_key = None
        for state_key, node_data in self.tree.items():
            if not node_data['terminal']:
                continue
            
            state_tuple = self.state_key_to_tuple(state_key)
            depth = self.get_state_depth(state_tuple)
            result = node_data['result']
            
            depth_match = (target_depth is None or depth == target_depth)
            outcome_match = (target_outcome is None or 
                           (target_outcome == 'X' and result == 1) or
                           (target_outcome == 'O' and result == -1) or
                           (target_outcome == 'draw' and result == 0))
            
            if depth_match and outcome_match:
                target_key = state_key
                break
        
        if not target_key:
            print("❌ No suitable target state found")
            return
        
        # Find path from root to target
        path = self._find_path_to_state(target_key)
        if not path:
            print("❌ No path found to target")
            return
        
        target_info = self.get_state_info(target_key)
        print(f"🎯 Target: {target_info['outcome']} in {target_info['depth']} moves")
        print(f"📍 Path length: {len(path)} states")
        
        print(f"\n📝 MOVE SEQUENCE:")
        for i, state_key in enumerate(path):
            info = self.get_state_info(state_key)
            
            if i == 0:
                print(f"\nMove 0 (Start): Empty board")
            else:
                # Find the move that led to this state
                prev_state_key = path[i-1]
                prev_node = self.tree[prev_state_key]
                
                for move, next_state_list in prev_node['moves']:
                    if str(next_state_list) == state_key:
                        row, col = move // self.n, move % self.n
                        prev_info = self.get_state_info(prev_state_key)
                        print(f"\nMove {i}: {prev_info['player']} plays position {move} (row {row}, col {col})")
                        break
            
            print(info['board'])
            
            if info['terminal']:
                print(f"🏁 Game Over: {info['outcome']}")
    
    def _find_path_to_state(self, target_key):
        """BFS to find path from root to target state"""
        start_key = str([0] * (self.n * self.n))
        
        if start_key == target_key:
            return [start_key]
        
        queue = deque([(start_key, [start_key])])
        visited = set()
        
        while queue:
            current_key, path = queue.popleft()
            
            if current_key in visited:
                continue
            visited.add(current_key)
            
            if current_key == target_key:
                return path
            
            if current_key in self.tree:
                node_data = self.tree[current_key]
                if not node_data['terminal']:
                    for move, next_state_list in node_data['moves']:
                        next_key = str(next_state_list)
                        if next_key not in visited:
                            queue.append((next_key, path + [next_key]))
        
        return None
    
    def interactive_explorer(self):
        """Interactive exploration of the game tree"""
        if not self.load_json():
            return
        
        print(f"\n🎮 INTERACTIVE GAME TREE EXPLORER")
        print("="*45)
        print("Commands: number (make move), 'b' (back), 'r' (restart), 'q' (quit)")
        
        start_key = str([0] * (self.n * self.n))
        path_stack = [start_key]
        
        while True:
            current_key = path_stack[-1]
            
            if current_key not in self.tree:
                print("❌ Invalid state! Restarting...")
                path_stack = [start_key]
                continue
            
            info = self.get_state_info(current_key)
            
            print(f"\n{'='*30}")
            print(f"📍 Position (depth {info['depth']}):")
            print(info['board'])
            
            if info['terminal']:
                print(f"\n🏁 GAME OVER: {info['outcome']}")
                print("Press Enter to restart or 'q' to quit...")
                choice = input().strip().lower()
                if choice == 'q':
                    break
                else:
                    path_stack = [start_key]
                    continue
            
            print(f"\n🎯 {info['player']} to move")
            print(f"📊 Available moves: {info['num_moves']}")
            
            node_data = self.tree[current_key]
            print("\nOptions:")
            for i, (move, next_state_list) in enumerate(node_data['moves']):
                row, col = move // self.n, move % self.n
                next_info = self.get_state_info(str(next_state_list))
                preview = next_info['compact_board']
                print(f"  {i}: pos {move} ({row},{col}) → {preview}")
            
            choice = input(f"\nEnter choice (0-{len(node_data['moves'])-1}, 'b', 'r', 'q'): ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'r':
                path_stack = [start_key]
                print("🔄 Restarted to beginning")
            elif choice == 'b':
                if len(path_stack) > 1:
                    path_stack.pop()
                    print("⬅️ Moved back one step")
                else:
                    print("❌ Already at the beginning")
            else:
                try:
                    move_idx = int(choice)
                    if 0 <= move_idx < len(node_data['moves']):
                        _, next_state_list = node_data['moves'][move_idx]
                        next_key = str(next_state_list)
                        path_stack.append(next_key)
                    else:
                        print("❌ Invalid move number!")
                except ValueError:
                    print("❌ Invalid input! Use numbers, 'b', 'r', or 'q'")


def main():
    """Main function to demonstrate the visualizer"""
    print("🎯 TIC-TAC-TOE GAME TREE VISUALIZER")
    print("="*40)
    
    filename = input("Enter JSON filename (or press Enter for 'game_tree_2x2.json'): ").strip()
    if not filename:
        filename = 'game_tree_2x2.json'
    
    viz = GameTreeVisualizer(filename)
    
    while True:
        print("\n📋 AVAILABLE FUNCTIONS:")
        print("1. Show statistics")
        print("2. Show tree structure")  
        print("3. Find game path")
        print("4. Interactive explorer")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            viz.show_statistics()
        elif choice == '2':
            max_depth = input("Max depth to show (default 3): ").strip()
            max_children = input("Max children per node (default 3): ").strip()
            
            max_depth = int(max_depth) if max_depth.isdigit() else 3
            max_children = int(max_children) if max_children.isdigit() else 3
            
            viz.show_tree_structure(max_depth, max_children)
        elif choice == '3':
            outcome = input("Target outcome (X/O/draw, or Enter for any): ").strip()
            depth = input("Target depth (or Enter for any): ").strip()
            
            outcome = outcome if outcome in ['X', 'O', 'draw'] else None
            depth = int(depth) if depth.isdigit() else None
            
            viz.find_game_path(target_depth=depth, target_outcome=outcome)
        elif choice == '4':
            viz.interactive_explorer()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please select 1-5.")


if __name__ == "__main__":
    main()