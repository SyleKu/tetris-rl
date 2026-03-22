import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tetris_rl.env.board import Board
from tetris_rl.env.features import aggregate_height, bumpiness, holes, max_height
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

        # fixed maximum number of valid placements across all places
        self.max_actions = self._compute_max_actions()
        self.action_space = spaces.Discrete(self.max_actions)

        # 5 scalar features + one-hot current place
        obs_dim = 5 + len(self.piece_names)
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _compute_max_actions(self) -> int:
        max_actions = 0
        for piece_name, variants in PIECES.items():
            count = 0
            for piece in variants:
                piece_width = len(piece[0])
                count += max(0, self.width - piece_width + 1)
            max_actions = max(max_actions, count)
        return max_actions

    def _sample_piece(self):
        name = random.choice(self.piece_names)
        variants = PIECES[name]
        return name, variants

    def piece_one_hot(self, piece_name: str) -> np.ndarray:
        vec = np.zeros(len(self.piece_names), dtype=np.float32)
        vec[self.piece_names.index(piece_name)] = 1.0
        return vec

    def _get_observation(self):
        grid = self.board.grid
        scalar_features = np.array(
            [
                aggregate_height(grid),
                holes(grid),
                bumpiness(grid),
                max_height(grid),
                float(self.board.is_game_over())
            ],
            dtype=np.float32,
        )

        piece_features = self.piece_one_hot(self.current_piece_name)
        return np.concatenate([scalar_features, piece_features]).astype(np.float32)

    def _enumerate_valid_actions(self):
        if self.board.is_game_over():
            return []

        variants = PIECES[self.current_piece_name]
        valid_actions = []

        for rotation_idx, piece in enumerate(variants):
            piece_width = len(piece[0])

            for column in range(self.width - piece_width + 1):
                # if piece cannot even spawn here, skip
                if self.board.check_collision(piece, -len(piece), column):
                    continue

                valid_actions.append((rotation_idx, column))

        return valid_actions

    def get_valid_actions(self):
        return self._enumerate_valid_actions()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = Board(height=self.height, width=self.width)
        self.current_piece_name, self.current_piece = self._sample_piece()
        return self._get_observation(), {}

    def _drop_height(self, piece, column: int):
        row = -len(piece)
        while not self.board.check_collision(piece, row + 1, column):
            row += 1
        return row

    def step(self, action: int):
        # Early game-over check
        if self.board.is_game_over():
            reward = -10.0
            terminated = True
            truncated = False

            return (
                self._get_observation(),
                reward,
                terminated,
                truncated,
                {"game_over": True}
            )

        valid_actions = self._enumerate_valid_actions()

        # no valid placement left -> game over
        if len(valid_actions) == 0:
            reward = -10.0
            terminated = True
            truncated = False
            return (
                self._get_observation(),
                reward,
                terminated,
                truncated,
                {"invalid_action": True}
            )

        # map agent action into valid placement list
        action_idx = int(action) % len(valid_actions)
        rotation_idx, column = valid_actions[action_idx]

        piece = PIECES[self.current_piece_name][rotation_idx]

        row = self._drop_height(piece, column)
        self.board.place_piece(piece, row, column)
        lines = self.board.clear_lines()

        grid = self.board.grid

        reward = (
                5.0 * lines
                - 0.02 * aggregate_height(grid)
                - 0.1 * holes(grid)
                - 0.02 * bumpiness(grid)
        )

        terminated = self.board.is_game_over()
        truncated = False

        if terminated:
            reward -= 5.0

        self.current_piece_name, self.current_piece = self._sample_piece()

        obs = self._get_observation()
        info = {"lines_cleared": lines}

        if terminated:
            info["game_over"] = True

        return obs, reward, terminated, truncated, info
