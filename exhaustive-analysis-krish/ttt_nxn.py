from collections import defaultdict, deque
import json

class TicTacToeGameTree:
    def __init__(self, n: int):
        self.n = n
        self.board_size = n * n
        self.game_tree = {}                 # state_tuple -> {moves, terminal, result, player}
        self.state_count = defaultdict(int) # ply -> count
        self.terminal_states = defaultdict(int)  # 'X_wins','O_wins','draws'

    @staticmethod
    def board_to_tuple(board):
        return tuple(board)

    @staticmethod
    def tuple_to_board(board_tuple):
        return list(board_tuple)

    def print_board(self, board):
        for i in range(self.n):
            row = []
            for j in range(self.n):
                v = board[i*self.n + j]
                row.append('.' if v == 0 else ('X' if v == 1 else 'O'))
            print(' '.join(row))
        print()

    def check_winner(self, board):
        n = self.n
        # rows
        for i in range(n):
            s = sum(board[i*n + j] for j in range(n))
            if s == n: return 1
            if s == -n: return -1
        # cols
        for j in range(n):
            s = sum(board[i*n + j] for i in range(n))
            if s == n: return 1
            if s == -n: return -1
        # main diag
        s = sum(board[i*n + i] for i in range(n))
        if s == n: return 1
        if s == -n: return -1
        # anti diag
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

        stack = [(start_state, 1)]  # (state, player_to_move) with X=1, O=-1
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
                    'result': result,   # 1=X wins, -1=O wins, 0=draw
                    'player': player    # player to move if it weren’t terminal
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

    def print_statistics(self):
        print("="*50)
        print(f"COMPLETE {self.n}x{self.n} TIC-TAC-TOE GAME TREE STATISTICS")
        print("="*50)
        print(f"Board size: {self.n}x{self.n} = {self.board_size} cells")
        print(f"Total unique game states: {len(self.game_tree)}")

        print("\nStates by number of moves (ply):")
        for k in sorted(self.state_count):
            print(f"  {k} moves: {self.state_count[k]} states")

        print("\nTerminal state outcomes:")
        print(f"  X wins: {self.terminal_states['X_wins']}")
        print(f"  O wins: {self.terminal_states['O_wins']}")
        print(f"  Draws: {self.terminal_states['draws']}")
        print(f"  Total terminal states: {sum(self.terminal_states.values())}")

        total_cells = self.board_size
        theoretical_max = sum(3**i for i in range(total_cells + 1))
        pct = 100.0 * len(self.game_tree) / theoretical_max
        print(f"\nTheoretical max states (sum_{total_cells} 3^i): {theoretical_max}")
        print(f"Actual states: {len(self.game_tree)} ({pct:.2f}% of theoretical max)")

    def find_path_to_state(self, target_board, max_depth=10**9):
        start = self.board_to_tuple([0] * self.board_size)
        target = self.board_to_tuple(target_board)
        if target not in self.game_tree:
            return None

        q = deque([(start, [])])
        seen = set()
        while q:
            s, path = q.popleft()
            if s == target:
                return path
            if s in seen:
                continue
            seen.add(s)
            node = self.game_tree.get(s)
            if node and not node['terminal']:
                for mv, nxt in node['moves']:
                    q.append((nxt, path + [mv]))
        return None

    def save_tree_to_file(self, filename):
        json_tree = {}
        for state_tuple, data in self.game_tree.items():
            key = list(state_tuple)
            node = {
                'terminal': data['terminal'],
                'result': data['result'],
                'player': data['player'],
            }
            if data['terminal']:
                node['moves'] = []
            else:
                node['moves'] = [(mv, list(nxt)) for mv, nxt in data['moves']]
            json_tree[str(key)] = node

        payload = {
            'n': self.n,
            'statistics': {
                'total_states': len(self.game_tree),
                'states_by_moves': dict(self.state_count),
                'terminal_outcomes': dict(self.terminal_states)
            },
            'tree': json_tree
        }
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Game tree saved to {filename}")


if __name__ == '__main__':
    # Dynamically choose n here
    n = 3  # change to any n, but note combinatorial explosion for n>3
    game = TicTacToeGameTree(n)
    game.generate_game_tree()
    game.print_statistics()

    # Example: save the tree
    out_file = f'tic_tac_toe_{n}x{n}_complete_tree.json'
    game.save_tree_to_file(out_file)

    # Example: demonstrate a terminal path on 3x3
    if n == 3:
        # find any X-win terminal and show the path
        for state, node in game.game_tree.items():
            if node['terminal'] and node['result'] == 1:
                board = game.tuple_to_board(state)
                print("\nExample terminal (X wins):")
                game.print_board(board)
                path = game.find_path_to_state(board)
                print("Path (flattened indices):", path)
                break
