import numpy as np

def column_heights(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    heights = np.zeros_like(width, dtype=np.int32)

    for col in range(width):
        filled = np.where(grid[:, col] == 1)[0]
        if len(filled) > 0:
            heights[col] = height - filled[0]

    return heights

def aggregate_height(grid: np.ndarray) -> int:
    return int(np.sum(column_heights(grid)))

def max_height(grid: np.ndarray) -> int:
    return int(np.max(column_heights(grid)))

def holes(grid: np.ndarray) -> int:
    h, w = grid.shape
    total = 0
    for col in range(w):
        block_seen = False
        for row in range(h):
            if grid[row, col] == 1:
                block_seen = True
            elif block_seen and grid[row, col] == 0:
                total += 1

    return total

def bumpiness(grid: np.ndarray) -> int:
    heights = column_heights(grid)
    return int(np.sum(np.abs(np.diff(heights))))
