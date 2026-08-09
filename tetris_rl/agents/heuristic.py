from dataclasses import dataclass

from tetris_rl.env.features import aggregate_height, bumpiness, holes, max_height
from tetris_rl.env.pieces import PIECES

@dataclass
class HeuristicWeights:
    lines_cleared: float = 1.0
    holes: float = -0.7
    bumpiness: float = -0.2
    aggregate_height: float = -0.3
    max_height: float = -0.2

def score_board(grid, lines_cleared: int, weights: HeuristicWeights) -> float:
    return(
        weights.lines_cleared * lines_cleared
        + weights.holes * holes(grid)
        + weights.bumpiness * bumpiness(grid)
        + weights.aggregate_height * aggregate_height(grid)
        + weights.max_height * max_height(grid)
    )

def board_potential(grid, weights: HeuristicWeights) -> float:
    """State-only board score Phi(s) used as the potential for reward shaping.

    This is ``score_board`` with the transition-dependent ``lines_cleared`` term
    removed, so it is a pure function of the board state. Used by Experiment H's
    potential-based reward shaping (``F = gamma * Phi(s') - Phi(s)``), which
    injects the heuristic's board-quality knowledge as a dense, policy-invariant
    signal (Ng et al., 1999). On an empty board every feature is zero, so
    ``Phi = 0``.
    """
    return score_board(grid, lines_cleared=0, weights=weights)

class HeuristicAgent:
    def __init__(self, weights: HeuristicWeights | None = None):
        self.weights = weights or HeuristicWeights()

    def select_action(self, env) -> int:
        valid_actions = env.get_valid_actions()

        best_score = float("-inf")
        best_action = None

        for rotation_idx, column in valid_actions:
            piece = PIECES[env.current_piece_name][rotation_idx]

            board_copy = env.board.clone()

            row = -len(piece)
            while not board_copy.check_collision(piece, row + 1, column):
                row += 1

            board_copy.place_piece(piece, row, column)
            lines_cleared = board_copy.clear_lines()

            score = score_board(
                board_copy.grid,
                lines_cleared=lines_cleared,
                weights=self.weights,
            )

            if score > best_score:
                best_score = score
                # encode into the env's fixed action space
                best_action = env.encode_action(rotation_idx, column)

        if best_action is None:
            return 0

        return best_action
