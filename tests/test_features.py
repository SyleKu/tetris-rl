import numpy as np

from tetris_rl.env.features import (
    column_heights,
    aggregate_height,
    max_height,
    holes,
    bumpiness
)

def test_column_heights_empty_grid():
    grid = np.zeros((4, 4), dtype=np.int8)
    heights = column_heights(grid)

    assert np.array_equal(heights, np.array([0, 0, 0, 0]))

def test_column_heights_simple_case():
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ]
        , dtype=np.int8,
    )

    heights = column_heights(grid)

    assert np.array_equal(heights, np.array([2, 3, 1, 0]))

def test_aggregate_height():
    grid = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )

    assert aggregate_height(grid) == 4

def test_max_height():
    grid = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )

    assert max_height(grid) == 3

def test_holes_counts_empty_cells_below_blocks():
    grid = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )

    assert holes(grid) == 1

def test_holes_zero_when_no_gaps_below_blocks():
    grid = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 1],
        ],
        dtype=np.int8,
    )

    assert holes(grid) == 0

def test_bumpiness():
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=np.int8,
    )

    # heights = [2, 3, 1, 0]
    # bumpiness = [2-3] + [3-1] + [1-0] = 1 + 2 + 1 = 4
    assert bumpiness(grid) == 4
