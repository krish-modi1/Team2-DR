# tictactoe_4x4_bitboard.py
from typing import List, Tuple, Optional
import copy

# X=1 先手，O=-1，空=0
X, O, E = 1, -1, 0

class GameState:
    def __init__(self, board: Optional[list[list[int]]] = None, side_to_move: int = X, move_count: int = 0):
        self.board = board or [[E]*4 for _ in range(4)]
        self.side_to_move = side_to_move
        self.move_count = move_count

    @staticmethod
    def initial() -> "GameState":
        return GameState()

    def legal_moves(self) -> List[Tuple[int,int]]:
        if self.is_terminal(): return []
        return [(y,x) for y in range(4) for x in range(4) if self.board[y][x]==E]

    def play(self, y: int, x: int) -> "GameState":
        assert self.board[y][x]==E, "illegal move"
        nb = copy.deepcopy(self.board)
        nb[y][x] = self.side_to_move
        return GameState(nb, -self.side_to_move, self.move_count+1)

    def is_terminal(self) -> bool:
        lines=[]
        for i in range(4):
            lines.append(self.board[i])
            lines.append([self.board[r][i] for r in range(4)])
        lines.append([self.board[i][i] for i in range(4)])
        lines.append([self.board[i][3-i] for i in range(4)])
        for line in lines:
            s=sum(line)
            if s==4*X or s==4*O: return True
        return all(self.board[y][x]!=E for y in range(4) for x in range(4))

    def result(self) -> str:
        lines=[]
        for i in range(4):
            lines.append(self.board[i])
            lines.append([self.board[r][i] for r in range(4)])
        lines.append([self.board[i][i] for i in range(4)])
        lines.append([self.board[i][3-i] for i in range(4)])
        for line in lines:
            s=sum(line)
            if s==4*X: return "X"
            if s==4*O: return "O"
        return "draw"
