import numpy as np
import json
from collections import defaultdict

class TicTacToe:
    """
    3x3 Tic-Tac-Toe game representation and logic.
    
    Board representation:
    - 3x3 numpy array
    - 1 represents X (maximizing player)
    - -1 represents O (minimizing player) 
    - 0 represents empty cell
    """
    
    def __init__(self, size=3):
        """
        Initialize a new 3x3 Tic-Tac-Toe game.
        
        Args:
            size (int): Board size (always 3 for this implementation)
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # X starts first (maximizing player)

    def copy(self):
        """
        Create a deep copy of the current game state.
        Essential for minimax to explore moves without affecting original state.
        
        Returns:
            TicTacToe: Deep copy of current game
        """
        new_game = TicTacToe(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def get_valid_moves(self):
        """
        Get all legal moves (empty positions) on the current board.
        
        Returns:
            list: List of (row, col) tuples representing valid moves
        """
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:  # Empty cell
                    moves.append((i, j))
        return moves

    def make_move(self, row, col):
        """
        Make a move on the board and switch to the next player.
        
        Args:
            row (int): Row index (0-2)
            col (int): Column index (0-2)
            
        Returns:
            bool: True if move was successful, False if invalid
        """
        if self.board[row, col] == 0:  # Cell is empty
            self.board[row, col] = self.current_player
            self.current_player *= -1  # Switch player (1 -> -1 or -1 -> 1)
            return True
        return False

    def undo_move(self, row, col):
        """
        Undo a previously made move. Critical for minimax backtracking.
        
        Args:
            row (int): Row index of move to undo
            col (int): Column index of move to undo
        """
        self.board[row, col] = 0  # Clear the cell
        self.current_player *= -1  # Switch back to previous player

    def check_winner(self):
        """
        Check if the game has ended and determine the winner.
        
        Returns:
            int: 1 if X wins, -1 if O wins, 0 if draw, None if game continues
        """
        # Check all rows for a winner
        for i in range(self.size):
            if abs(sum(self.board[i, :])) == self.size:
                # All cells in row have same non-zero value
                return self.board[i, 0] if self.board[i, 0] != 0 else None
        
        # Check all columns for a winner
        for j in range(self.size):
            if abs(sum(self.board[:, j])) == self.size:
                # All cells in column have same non-zero value
                return self.board[0, j] if self.board[0, j] != 0 else None
        
        # Check main diagonal (top-left to bottom-right)
        main_diag_sum = sum(self.board[i, i] for i in range(self.size))
        if abs(main_diag_sum) == self.size:
            return self.board[0, 0] if self.board[0, 0] != 0 else None
        
        # Check anti-diagonal (top-right to bottom-left)
        anti_diag_sum = sum(self.board[i, self.size-1-i] for i in range(self.size))
        if abs(anti_diag_sum) == self.size:
            return self.board[0, self.size-1] if self.board[0, self.size-1] != 0 else None
        
        # Check for draw (no empty cells left)
        if len(self.get_valid_moves()) == 0:
            return 0  # Draw
        
        return None  # Game continues

    def is_terminal(self):
        """
        Check if the game is in a terminal state (ended).
        
        Returns:
            bool: True if game has ended, False otherwise
        """
        return self.check_winner() is not None

    def get_board_hash(self):
        """
        Generate a hashable representation of the board state.
        Used for memoization and state storage.
        
        Returns:
            tuple: Hashable representation of board state
        """
        return tuple(self.board.flatten())

    def display(self):
        """
        Display the current board state in a human-readable format.
        Useful for debugging and visualization.
        """
        symbols = {1: 'X', -1: 'O', 0: '.'}
        print("Current Board:")
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()


class MinimaxSolver:
    """
    Complete Minimax solver for 3x3 Tic-Tac-Toe.
    
    This class implements the minimax algorithm with the following enhancements:
    - Memoization for performance optimization
    - Complete state tracking for analysis
    - Move scoring for heatmap generation
    """
    
    def __init__(self, game):
        """
        Initialize the Minimax solver.
        
        Args:
            game (TicTacToe): The game instance to solve
        """
        self.game = game
        self.memo = {}  # Memoization cache: {(state_hash, is_maximizing): score}
        self.all_states = {}  # Complete state storage: {state_hash: state_info}
        self.move_scores = defaultdict(list)  # Move scores for each state

    def minimax(self, game_state, depth=0, maximizing_player=True):
        """
        Core minimax algorithm implementation with memoization and state tracking.
        
        Args:
            game_state (TicTacToe): Current game state
            depth (int): Current search depth
            maximizing_player (bool): True if maximizing player's turn, False otherwise
            
        Returns:
            int: Minimax score for this position
        """
        # Generate unique key for this position and player turn
        state_hash = game_state.get_board_hash()
        memo_key = (state_hash, maximizing_player)
        
        # Check if we've already computed this position
        if memo_key in self.memo:
            return self.memo[memo_key]

        # Check if game has ended (terminal node)
        winner = game_state.check_winner()
        if winner is not None:
            # Calculate terminal score based on winner and depth
            if winner == 1:  # X wins (maximizing player)
                score = 10 - depth  # Prefer faster wins
            elif winner == -1:  # O wins (minimizing player)
                score = -10 + depth  # Prefer slower losses
            else:  # Draw
                score = 0
            
            # Store terminal state information
            self.all_states[state_hash] = {
                'board': game_state.board.copy(),
                'score': score,
                'depth': depth,
                'terminal': True,
                'winner': winner
            }
            
            # Cache result and return
            self.memo[memo_key] = score
            return score

        # Non-terminal state: evaluate all possible moves
        valid_moves = game_state.get_valid_moves()
        
        if maximizing_player:
            # Maximizing player (X) wants highest score
            max_eval = float('-inf')
            best_move = None
            
            for move in valid_moves:
                row, col = move
                
                # Make the move
                game_state.make_move(row, col)
                
                # Recursively evaluate resulting position
                eval_score = self.minimax(game_state, depth + 1, False)
                
                # Undo the move (backtrack)
                game_state.undo_move(row, col)
                
                # Record move score for heatmap generation
                self.move_scores[state_hash].append({
                    'move': move,
                    'score': eval_score,
                    'player': 1
                })
                
                # Update best score and move
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            
            # Store state information
            self.all_states[state_hash] = {
                'board': game_state.board.copy(),
                'score': max_eval,
                'depth': depth,
                'terminal': False,
                'best_move': best_move,
                'player': 1
            }
            
            # Cache and return result
            self.memo[memo_key] = max_eval
            return max_eval
            
        else:
            # Minimizing player (O) wants lowest score
            min_eval = float('inf')
            best_move = None
            
            for move in valid_moves:
                row, col = move
                
                # Make the move
                game_state.make_move(row, col)
                
                # Recursively evaluate resulting position
                eval_score = self.minimax(game_state, depth + 1, True)
                
                # Undo the move (backtrack)
                game_state.undo_move(row, col)
                
                # Record move score for heatmap generation
                self.move_scores[state_hash].append({
                    'move': move,
                    'score': eval_score,
                    'player': -1
                })
                
                # Update best score and move
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            
            # Store state information
            self.all_states[state_hash] = {
                'board': game_state.board.copy(),
                'score': min_eval,
                'depth': depth,
                'terminal': False,
                'best_move': best_move,
                'player': -1
            }
            
            # Cache and return result
            self.memo[memo_key] = min_eval
            return min_eval

    def solve_completely(self):
        """
        Solve the entire game tree starting from empty board.
        This performs exhaustive analysis of all possible 3x3 Tic-Tac-Toe games.
        
        Returns:
            int: Optimal score with perfect play from both sides
        """
        print("Starting exhaustive Minimax analysis of 3x3 Tic-Tac-Toe...")
        game_copy = self.game.copy()
        optimal_score = self.minimax(game_copy, maximizing_player=True)
        print(f"Analysis complete!")
        print(f"Optimal result with perfect play: {optimal_score}")
        print(f"Total unique board states: {len(self.all_states)}")
        return optimal_score

    def get_best_move(self, game_state):
        """
        Get the best move for the current game state.
        
        Args:
            game_state (TicTacToe): Current game state
            
        Returns:
            tuple: (row, col) of best move, or None if no move found
        """
        state_hash = game_state.get_board_hash()
        if state_hash in self.all_states:
            return self.all_states[state_hash].get('best_move')
        return None

    def save_analysis_to_json(self, states_filename="tic_tac_toe_3x3_states.json", 
                             moves_filename="tic_tac_toe_3x3_moves.json"):
        """
        Save complete analysis results to JSON files for future reference.
        
        Args:
            states_filename (str): Filename for states data
            moves_filename (str): Filename for moves data
        """
        print(f"Saving analysis results to JSON files...")
        
        # Prepare states data for JSON serialization
        states_data = {}
        for state_hash, state_info in self.all_states.items():
            states_data[str(state_hash)] = {
                'board': state_info['board'].tolist(),  # Convert numpy array to list
                'score': int(state_info['score']),  # Ensure integer type
                'depth': int(state_info['depth']),
                'terminal': bool(state_info['terminal']),
                'best_move': state_info.get('best_move'),
                'player': int(state_info.get('player', 0)) if state_info.get('player') is not None else None,
                'winner': int(state_info.get('winner', 0)) if state_info.get('winner') is not None else None
            }
        
        # Prepare moves data for JSON serialization
        moves_data = {}
        for state_hash, moves_list in self.move_scores.items():
            moves_data[str(state_hash)] = [
                {
                    'move': list(move_data['move']),  # Convert tuple to list
                    'score': int(move_data['score']),
                    'player': int(move_data['player'])
                }
                for move_data in moves_list
            ]
        
        # Save to JSON files
        with open(states_filename, 'w') as f:
            json.dump(states_data, f, indent=2)
        
        with open(moves_filename, 'w') as f:
            json.dump(moves_data, f, indent=2)
        
        print(f"States data saved to: {states_filename}")
        print(f"Moves data saved to: {moves_filename}")

    def print_analysis_summary(self):
        """
        Print a comprehensive summary of the analysis results.
        """
        print("\n" + "="*60)
        print("3x3 TIC-TAC-TOE MINIMAX ANALYSIS SUMMARY")
        print("="*60)
        
        # Count different types of states
        terminal_states = 0
        x_wins = 0
        o_wins = 0
        draws = 0
        
        for state_info in self.all_states.values():
            if state_info['terminal']:
                terminal_states += 1
                winner = state_info.get('winner')
                if winner == 1:
                    x_wins += 1
                elif winner == -1:
                    o_wins += 1
                else:
                    draws += 1
        
        print(f"Total unique board positions: {len(self.all_states)}")
        print(f"Terminal positions: {terminal_states}")
        print(f"  - X wins: {x_wins}")
        print(f"  - O wins: {o_wins}")
        print(f"  - Draws: {draws}")
        print(f"Non-terminal positions: {len(self.all_states) - terminal_states}")
        
        # Analysis of opening moves
        initial_hash = tuple([0] * 9)  # Empty board
        if initial_hash in self.move_scores:
            print(f"\nOpening move analysis:")
            moves_data = self.move_scores[initial_hash]
            print(f"All first moves for X:")
            for move_data in sorted(moves_data, key=lambda x: x['score'], reverse=True):
                row, col = move_data['move']
                score = move_data['score']
                print(f"  Position ({row},{col}): Score = {score}")
        
        print("="*60)


def main():
    """
    Main function to run the complete 3x3 Tic-Tac-Toe analysis.
    """
    print("3x3 Tic-Tac-Toe Minimax Analysis")
    print("=" * 40)
    
    # Initialize game and solver
    game = TicTacToe(3)
    solver = MinimaxSolver(game)
    
    # Perform complete analysis
    optimal_score = solver.solve_completely()
    
    # Print detailed summary
    solver.print_analysis_summary()
    
    # Save results to JSON files
    solver.save_analysis_to_json()
    
    print(f"\nAnalysis complete! All data saved for heatmap generation.")
    print(f"With perfect play, the game result is: {optimal_score}")
    if optimal_score == 0:
        print("This confirms that 3x3 Tic-Tac-Toe is a draw with optimal play.")
    elif optimal_score > 0:
        print("X (first player) has a winning advantage.")
    else:
        print("O (second player) has a winning advantage.")


if __name__ == "__main__":
    main()