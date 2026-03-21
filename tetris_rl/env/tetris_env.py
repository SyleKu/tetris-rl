import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tetris_rl.env.board import Board
from tetris_rl.env.features import aggregate_height, bumpiness, holes
from tetris_rl.env.pieces import PIECES

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, height=20, width=10, render_mode=None):
        super().__init__()
        self.height = height
        self.width = width
        self.render_mode = render_mode

        self.board = Board(height=height, width=width)
        self.piece_names = list(PIECES.keys())
        self.current_piece_name = None
        self.current_piece = None

        self.max_rotations = 4
        self.max_actions = self.max_rotations * self.width
        self.action_space = spaces.Discrete(self.max_actions)

        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(6,),
            dtype=np.float32,
        )

    def _sample_piece(self):
        name = random.choice(self.piece_names)
        variants = PIECES[name]
        return name, variants

    def _get_observation(self):
        grid = self.board.grid
        obs = np.array(
            [
                aggregate_height(grid),
                holes(grid),
                bumpiness(grid),
                np.max(grid.sum(axis=1)),
                self.board.is_game_over(),
                len(PIECES[self.current_piece_name]),
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = Board(height=self.height, width=self.width)
        self.current_piece_name, self.current_piece = self._sample_piece()
        return self._get_observation(), {}

    def decode_action(self, action: int):
        rotation = action // self.width
        column = action % self.width
        return rotation, column

    def _drop_height(self, piece, column: int):
        row = -len(piece)
        while not self.board.check_collision(piece, row + 1, column):
            row += 1
        return row

    @staticmethod
    def piece_one_hot(self, piece_name: str, piece_names: list[str]) -> np.ndarray:
        vec = np.zeros(len(piece_names), dtype=np.float32)
        vec[piece_names.index(piece_name)] = 1.0
        return vec

    def step(self, action: int):
        rotation_idx, column = self.decode_action(action)
        variants = PIECES[self.current_piece_name]
        piece = variants[rotation_idx % len(variants)]

        if column + len(piece[0]) > self.width:
            reward = -2.0
            terminated = False
            return self._get_observation(), reward, terminated, False, {"invalid_action": True}

        if self.board.check_collision(piece, -len(piece), column):
            reward = -10.0
            terminated = True
            return self._get_observation(), reward, terminated, False, {"game_over": True}

        row = self._drop_height(piece, column)
        self.board.place_piece(piece, row, column)
        lines = self.board.clear_lines()

        grid = self.board.grid
        reward = (
            1.0 * lines
            - 0.05 * aggregate_height(grid)
            - 0.2 * holes(grid)
            - 0.05 * bumpiness(grid)
        )

        terminated = self.board.is_game_over()

        if terminated:
            reward -= 5.0

        self.current_piece_name, self.current_piece = self._sample_piece()

        obs = self._get_observation()
        info = {"lines_cleared": lines}

        if terminated:
            info["game_over"] = True

        return obs, reward, terminated, False, info
