# In a new file named 'play.py'

import time
from tictactoe_game import TicTacToe
from minimax_player import MinimaxPlayer, HumanPlayer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_heatmap(game, scores):
    # Create a data grid with the same dimensions as the board
    board_size = game.size
    data = np.full((board_size, board_size), np.nan) # Fill with NaN (Not a Number) initially

    # Populate the grid with scores for available moves
    for move, score in scores.items():
        row, col = move
        data[row][col] = score

    # Generate the heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(data,
                annot=True,          # Write the data value in each cell
                fmt=".2f",           # Format as a float with 2 decimal places
                cmap='viridis',      # Color scheme
                cbar=False,          # Do not show the color bar
                linewidths=.5,
                linecolor='black',
                annot_kws={"size": 14}) # Make font size larger
    
    plt.title("Minimax Move Scores (AI is 'X')")
    plt.show()

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
            square, scores = x_player.get_move_with_scores(game)

            # Generate the heatmap before the AI moves
            if print_game:
                print("AI is thinking... Here are its move scores:")
                generate_heatmap(game, scores)

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