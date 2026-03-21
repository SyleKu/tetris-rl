from dataclasses import dataclass

from tetris_rl.env.features import aggregate_height, bumpiness, holes, max_height
from tetris_rl.env.pieces import PIECES

@dataclass
class HeuristicWeights:
    lines_cleared: float = 1.0
    holes: float = -0.7
    bumpiness: float = -0.2
    aggregate_height: float = -0.3
    max_height_: float = -0.2

def score_board(grid, lines_cleared: int, weights: HeuristicWeights) -> float:
    return(
        weights.lines_cleared * lines_cleared
        + weights.holes * holes(grid)
        + weights.bumpiness * bumpiness(grid)
        + weights.aggregate_height * aggregate_height(grid)
        + weights.max_height_ * max_height(grid)
    )

class HeuristicAgent:
    def __init__(self, weights: HeuristicWeights | None = None):
        self.weights = weights or HeuristicWeights()

    def select_action(self, env) -> int:
        piece_name = env.current_piece_name
        variants = PIECES[piece_name]

        best_score = float("-inf")
        best_action = None

        for rotation_idx, piece in enumerate(variants):
            piece_width = len(piece[0])

            for column in range(env.width):

                # skip piece if it doesn't fit
                if column + piece_width > env.width:
                    continue

                board_copy = env.board.clone()

                # check spawn collision
                if board_copy.check_collision(piece, -len(piece), column):
                    continue

                # calculate drop
                row = -len(piece)
                while not board_copy.check_collision(piece, row + 1, column):
                    row += 1

                board_copy.place_piece(piece, row, column)
                lines_cleared = board_copy.cleared_lines()

                score = score_board(
                    board_copy.grid,
                    lines_cleared=lines_cleared,
                    weights=self.weights,
                )

                action = rotation_idx * env.width + column

                if score > best_score:
                    best_score = score
                    best_action = action

        if best_action is None:
            return 0

        return best_action


