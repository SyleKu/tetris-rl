from dataclasses import dataclass

from tetris_rl.env.features import aggregate_height, bumpiness, holes, max_height

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
