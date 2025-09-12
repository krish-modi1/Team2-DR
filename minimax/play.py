# In a new file named 'play.py'

import time
from tictactoe_game import TicTacToe
from minimax_player import MinimaxPlayer, HumanPlayer

def play(game, x_player, o_player, print_game=True):
    """Manages the main game loop"""

    if print_game:
        game.print_board()

    letter = 'X'  # Starting letter

    # Loop as long as there are available moves
    while game.available_moves():
        if letter == 'O': # Human's turn
            square = o_player.get_move(game)
        else: # AI's turn
            square = x_player.get_move(game)
            time.sleep(0.8) # Add a small delay to make the AI seem like it's 'thinking'

        # Make the move
        if game.make_move(square, letter):
            if print_game:
                print(f'\n{letter} makes a move to square {square}')
                game.print_board()
                print('') # Empty line for spacing

            if game.current_winner:
                if print_game:
                    print(f'{letter} wins!')
                return letter  # Ends the loop and returns the winner

            # Switch turns
            letter = 'O' if letter == 'X' else 'X'

    if print_game:
        print('It\'s a tie!')
    return None

if __name__ == '__main__':
    # Setup the game with an AI ('X') and a Human ('O')
    ai_player = MinimaxPlayer('X')
    human_player = HumanPlayer('O')
    game_instance = TicTacToe()
    
    # Start the game
    play(game_instance, ai_player, human_player, print_game=True)