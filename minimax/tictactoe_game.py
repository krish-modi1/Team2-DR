# In a file named 'tictactoe_game.py'

class TicTacToe:
    def __init__(self, size=3):
           # We'll use a 2D list to represent the board
           # ' ' represents an empty square
           self.size = size
           self.board = [[' ' for _ in range(size)] for _ in range(size)]
           self.current_winner = None # Keep track of the winner

    def print_board(self):
        # The board is a list of lists. Let's print the rows.
        for row in self.board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        # Returns a list of (row, col) tuples for all empty squares
        moves = []
        for r in range(self.size):
            for c in range(self.size):
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
        row_ind, col_ind = square # Unpack the move coordinates

        # 1. Check the row
        if all([cell == letter for cell in self.board[row_ind]]):
            return True

        # 2. Check the column
        column = [self.board[r][col_ind] for r in range(self.size)]
        if all([cell == letter for cell in column]):
            return True

        # 3. Check the diagonals (only if the move is on a diagonal)
        # Top-left to bottom-right diagonal (where row == col)
        if row_ind == col_ind:
            diag1 = [self.board[i][i] for i in range(self.size)]
            if all([cell == letter for cell in diag1]):
                return True
        
        # Top-right to bottom-left diagonal (where row + col == N - 1)
        if row_ind + col_ind == self.size - 1:
            diag2 = [self.board[i][self.size - 1 - i] for i in range(self.size)]
            if all([cell == letter for cell in diag2]):
                return True
        
        # If all checks fail
        return False