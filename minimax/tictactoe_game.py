# In a file named 'tictactoe_game.py'

class TicTacToe:
    def __init__(self):
        # We'll use a 2D list to represent the 3x3 board
        # ' ' represents an empty square
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_winner = None # Keep track of the winner

    def print_board(self):
        # The board is a list of lists. Let's print the rows.
        for row in self.board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        # Returns a list of (row, col) tuples for all empty squares
        moves = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == ' ':
                    moves.append((r, c))
        return moves

    def make_move(self, square, letter):
        # square is a tuple (row, col), letter is 'X' or 'O'
        if self.board[square[0]][square[1]] == ' ':
            self.board[square[0]][square[1]] = letter
            if self.check_winner(square, letter):
                self.current_winner = letter
            return True # The move was valid
        return False # The move was invalid
    
# This is a method *inside* the TicTacToe class

    def check_winner(self, square, letter):
        # square is the (row, col) of the most recent move
        # letter is the player ('X' or 'O') who made the move

        # 1. Check the row
        row_ind = square[0]
        if all([cell == letter for cell in self.board[row_ind]]):
            return True

        # 2. Check the column
        col_ind = square[1]
        column = [self.board[r][col_ind] for r in range(3)]
        if all([cell == letter for cell in column]):
            return True

        # 3. Check the diagonals (only if the move is on a diagonal)
        # The two diagonals are (0,0), (1,1), (2,2) and (0,2), (1,1), (2,0)
        if square[0] == square[1]: # Move is on the top-left to bottom-right diagonal
            diag1 = [self.board[i][i] for i in range(3)]
            if all([cell == letter for cell in diag1]):
                return True
        
        if square[0] + square[1] == 2: # Move is on the top-right to bottom-left diagonal
            diag2 = [self.board[i][2-i] for i in range(3)]
            if all([cell == letter for cell in diag2]):
                return True
        
        # If all checks fail
        return False