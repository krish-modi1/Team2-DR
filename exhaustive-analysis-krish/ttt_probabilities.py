from collections import defaultdict
import csv

class TicTacToeGameTree:
    def __init__(self, n: int):
        self.n = n
        self.board_size = n * n
        self.game_tree = {}
        self.state_count = defaultdict(int)
        self.terminal_states = defaultdict(int)

    @staticmethod
    def board_to_tuple(board):
        return tuple(board)

    @staticmethod
    def tuple_to_board(board_tuple):
        return list(board_tuple)

    def check_winner(self, board):
        n = self.n
        for i in range(n):
            s = sum(board[i*n + j] for j in range(n))
            if s == n: return 1
            if s == -n: return -1
        for j in range(n):
            s = sum(board[i*n + j] for i in range(n))
            if s == n: return 1
            if s == -n: return -1
        s = sum(board[i*n + i] for i in range(n))
        if s == n: return 1
        if s == -n: return -1
        s = sum(board[i*n + (n - 1 - i)] for i in range(n))
        if s == n: return 1
        if s == -n: return -1
        return 0

    def is_terminal(self, board):
        w = self.check_winner(board)
        if w != 0:
            return True, w
        if 0 not in board:
            return True, 0
        return False, None

    def get_valid_moves(self, board):
        return [i for i, v in enumerate(board) if v == 0]

    def make_move(self, board, pos, player):
        b = list(board)
        b[pos] = player
        return b

    def generate_game_tree(self):
        start_board = [0] * self.board_size
        start_state = self.board_to_tuple(start_board)
        stack = [(start_state, 1)]
        visited = set()

        while stack:
            state, player = stack.pop()
            if state in visited:
                continue
            visited.add(state)

            board = self.tuple_to_board(state)
            terminal, result = self.is_terminal(board)

            if terminal:
                self.game_tree[state] = {
                    'moves': [],
                    'terminal': True,
                    'result': result,
                    'player': player
                }
                if result == 1:
                    self.terminal_states['X_wins'] += 1
                elif result == -1:
                    self.terminal_states['O_wins'] += 1
                else:
                    self.terminal_states['draws'] += 1
            else:
                children = []
                for mv in self.get_valid_moves(board):
                    child_board = self.make_move(board, mv, player)
                    child_state = self.board_to_tuple(child_board)
                    children.append((mv, child_state))
                    stack.append((child_state, -player))

                self.game_tree[state] = {
                    'moves': children,
                    'terminal': False,
                    'result': None,
                    'player': player
                }

            ply = sum(1 for x in board if x != 0)
            self.state_count[ply] += 1

        return self.game_tree

    def compute_exhaustive_probabilities(self):
        """
        Compute P(X wins), P(O wins), P(draw) for each state under exhaustive play.
        For each state, we uniformly explore all possible continuations.
        """
        probabilities = {}

        def compute_probs(state):
            if state in probabilities:
                return probabilities[state]

            node = self.game_tree[state]

            if node['terminal']:
                result = node['result']
                if result == 1:  # X wins
                    probs = (1.0, 0.0, 0.0)
                elif result == -1:  # O wins
                    probs = (0.0, 1.0, 0.0)
                else:  # Draw
                    probs = (0.0, 0.0, 1.0)
            else:
                # Non-terminal: average over all possible moves
                moves = node['moves']
                n_moves = len(moves)

                px_total = 0.0
                po_total = 0.0
                pd_total = 0.0

                for mv, child_state in moves:
                    px, po, pd = compute_probs(child_state)
                    px_total += px
                    po_total += po
                    pd_total += pd

                probs = (px_total / n_moves, po_total / n_moves, pd_total / n_moves)

            probabilities[state] = probs
            return probs

        # Compute probabilities for all states
        for state in self.game_tree:
            compute_probs(state)

        return probabilities

    def save_probabilities_to_csv(self, probabilities, filename):
        """
        Save state probabilities to CSV with columns:
        state, to_move, layer, P(X wins), P(O wins), P(draws)
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['state', 'to_move', 'layer', 'P(X wins)', 'P(O wins)', 'P(draws)'])

            for state, probs in probabilities.items():
                board = list(state)
                node = self.game_tree[state]

                # Convert state to string representation
                state_str = ''.join('.' if x == 0 else ('X' if x == 1 else 'O') for x in board)

                # Player to move
                player = node['player']
                to_move = 'X' if player == 1 else 'O'

                # Layer (ply)
                layer = sum(1 for x in board if x != 0)

                # Probabilities
                px, po, pd = probs

                writer.writerow([state_str, to_move, layer, f'{px:.10f}', f'{po:.10f}', f'{pd:.10f}'])

        print(f"Probabilities saved to {filename}")

if __name__ == '__main__':
    n = 3

    print(f"Generating complete {n}x{n} Tic-Tac-Toe game tree...")
    game = TicTacToeGameTree(n)
    game.generate_game_tree()

    print(f"Total states: {len(game.game_tree)}")
    print(f"Terminal states: X wins={game.terminal_states['X_wins']}, "
          f"O wins={game.terminal_states['O_wins']}, draws={game.terminal_states['draws']}")

    print("\nComputing exhaustive probabilities...")
    probabilities = game.compute_exhaustive_probabilities()

    print("\nSaving to CSV...")
    output_file = f'ttt_{n}x{n}_exhaustive_probabilities.csv'
    game.save_probabilities_to_csv(probabilities, output_file)

    print(f"\nComplete! {len(probabilities)} states with probabilities saved to {output_file}")

    # Display sample results
    print("\nSample results (first 10 non-terminal states):")
    count = 0
    for state, probs in probabilities.items():
        node = game.game_tree[state]
        if not node['terminal'] and count < 10:
            board = list(state)
            state_str = ''.join('.' if x == 0 else ('X' if x == 1 else 'O') for x in board)
            layer = sum(1 for x in board if x != 0)
            px, po, pd = probs
            print(f"  {state_str} (layer {layer}): P(X)={px:.4f}, P(O)={po:.4f}, P(draw)={pd:.4f}")
            count += 1
