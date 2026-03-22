import numpy as np

class Board:
    def __init__(self, height: int = 20, width: int = 10):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=np.int8)

    def clone(self):
         new_board = Board(self.height, self.width)
         new_board.grid = self.grid.copy()
         return new_board

    def check_collision(self, piece, top: int, left: int) -> bool:
        for r, row in enumerate(piece):
            for c, value in enumerate(row):
                if value == 0:
                    continue

                br = top + r
                bc = left + c

                if bc < 0 or bc >= self.width or br >= self.height:
                    return True

                if br >= 0 and self.grid[br, bc] == 1:
                    return True

        return False

    def place_piece(self, piece, top: int, left: int):
        for r, row in enumerate(piece):
            for c, value in enumerate(row):
                if value == 0:
                    continue

                br = top + r
                bc = left + c
                if br >= 0:
                    self.grid[br, bc] = 1

    def clear_lines(self) -> int:
        full_rows = [r for r in range(self.height) if np.all(self.grid[r] == 1)]
        num_cleared = len(full_rows)

        if num_cleared > 0:
            new_grid = np.zeros_like(self.grid)
            remaining = np.delete(self.grid, full_rows, axis=0)

            if remaining.shape[0] > 0:
                new_grid[-remaining.shape[0]:] = remaining

            self.grid = new_grid

        return num_cleared

    def is_game_over(self) -> bool:
        return bool(np.any(self.grid[0] == 1))
