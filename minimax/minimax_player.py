# In a new file named 'minimax_player.py'
import math

class MinimaxPlayer:
    def __init__(self, letter):
        self.letter = letter # The player's letter ('X' or 'O')

    def get_move(self, game):
        # We'll use a minimax algorithm to find the best move
        # print(f"AI ({self.letter}) evaluating move...")
        # On an empty board, many opening moves are strategically identical
        if len(game.available_moves()) == game.size * game.size:
            # For the very first move, just pick a random square
            import random
            square = random.choice(game.available_moves())
            print(f"AI ({self.letter}) chooses random opening move: {square}")
        else:
            # Call the minimax algorithm to get the best square
            result = self.minimax(game, self.letter)
            square = result['square']
            print(f"AI ({self.letter}) chooses move: {square} with score: {result['score']}")
        return square

    
    def minimax(self, state, player, alpha=-math.inf, beta=math.inf):
        # Debug print for recursion depth and player
        # print(f"Minimax called for player {player}, alpha={alpha}, beta={beta}, available moves={len(state.available_moves())}")
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
            sim_score = self.minimax(state, other_player, alpha, beta)

            # 3. CRITICAL: Undo the move to explore other possibilities
            state.board[possible_move[0]][possible_move[1]] = ' '
            state.current_winner = None
            
            # This is just for the recursive call, we don't want to affect the real board
            sim_score['square'] = possible_move

            # 4. Update the 'best' dictionary if necessary
            if player == max_player:  # Maximizing player
                if sim_score['score'] > best['score']:
                    best = sim_score
                alpha = max(alpha, best['score'])
            else:  # Minimizing player
                if sim_score['score'] < best['score']:
                    best = sim_score
                beta = min(beta, best['score'])
            
            # Alpha-beta pruning
            if beta <= alpha:
                break
        
        return best
    
    def get_move_with_scores(self, game):
        # If the board is empty, handle it as a special case
        if len(game.available_moves()) == game.size * game.size:
            import random
            move = random.choice(game.available_moves())
            # Return a neutral score for all moves on an empty board
            scores = {m: 0 for m in game.available_moves()}
            return move, scores

        # Otherwise, calculate the score for every possible move
        scores = {}
        for possible_move in game.available_moves():
            # 1. Make the move
            game.make_move(possible_move, self.letter)
            # 2. Get the score for the resulting board from the opponent's view
            score = self.minimax(game, 'O' if self.letter == 'X' else 'X')['score']
            # 3. UNDO THE MOVE
            game.board[possible_move[0]][possible_move[1]] = ' '
            game.current_winner = None
            scores[possible_move] = score

        # Find the best move from the calculated scores
        best_move = max(scores, key=scores.get)
        return best_move, scores

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