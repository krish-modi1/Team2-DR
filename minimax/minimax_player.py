# In a new file named 'minimax_player.py'
import math

class MinimaxPlayer:
    def __init__(self, letter):
        self.letter = letter # The player's letter ('X' or 'O')

    def get_move(self, game):
        # We'll use a minimax algorithm to find the best move
        if len(game.available_moves()) == 9:
            # For the very first move, just pick a random square
            import random
            square = random.choice(game.available_moves())
        else:
            # Call the minimax algorithm to get the best square
            square = self.minimax(game, self.letter)['square']
        return square

    def minimax(self, state, player):
        # The 'state' is the current game object
        # The 'player' is the letter of the current player ('X' or 'O')
        max_player = self.letter  # The AI player
        other_player = 'O' if player == 'X' else 'X'

        # --- BASE CASE: Check if the previous move was a winning move ---
        # This is the condition that stops the recursion
        if state.current_winner == other_player:
            # We need to return the score and the square
            return {'square': None,
                    'score': 1 * (len(state.available_moves()) + 1) if other_player == max_player else -1 * (
                                len(state.available_moves()) + 1)}

        elif not state.available_moves(): # It's a tie
            return {'square': None, 'score': 0}

        # --- RECURSIVE STEP ---
        if player == max_player:
            # We want to maximize the score
            best = {'square': None, 'score': -math.inf}
        else:
            # We want to minimize the score
            best = {'square': None, 'score': math.inf}

        for possible_move in state.available_moves():
            # 1. Make a move (on an imaginary board)
            state.make_move(possible_move, player)

            # 2. Use recursion to simulate the game after that move
            #    by calling minimax for the *other player*
            sim_score = self.minimax(state, other_player)

            # 3. CRITICAL: Undo the move to explore other possibilities
            state.board[possible_move[0]][possible_move[1]] = ' '
            state.current_winner = None
            
            # This is just for the recursive call, we don't want to affect the real board
            sim_score['square'] = possible_move

            # 4. Update the 'best' dictionary if necessary
            if player == max_player:  # Maximizing player
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:  # Minimizing player
                if sim_score['score'] < best['score']:
                    best = sim_score
        
        return best

        # Add this class to 'minimax_player.py'

class HumanPlayer:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square_str = input(self.letter + '\'s turn. Input move (row col): ')
            try:
                # Try to parse the input "0 1" into a tuple (0, 1)
                row, col = [int(s) for s in square_str.split()]
                if (row, col) in game.available_moves():
                    valid_square = True
                    val = (row, col)
                else:
                    print('Invalid square. Try again.')
            except (ValueError, IndexError):
                print('Invalid input format. Please use "row col" (e.g., "0 2").')
        return val