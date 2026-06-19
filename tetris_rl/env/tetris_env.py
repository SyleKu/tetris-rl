import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tetris_rl.config import ObservationConfig, RewardConfig
from tetris_rl.env.board import Board
from tetris_rl.env.features import aggregate_height, bumpiness, column_heights, holes
from tetris_rl.env.pieces import PIECES

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        height=20,
        width=10,
        render_mode=None,
        reward_config: RewardConfig | None = None,
        observation_config: ObservationConfig | None = None,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.render_mode = render_mode
        self.reward_config = reward_config or RewardConfig()
        self.observation_config = observation_config or ObservationConfig()

        self.board = Board(height=height, width=width)
        self.piece_names = list(PIECES.keys())
        self.current_piece_name = None
        self.current_piece = None
        self.next_piece_name = None

        # Fixed action <-> placement mapping.
        #
        # An action integer decodes to (rotation_idx, column) via
        # divmod(action, width). The mapping never changes, so a given action
        # integer always denotes the same geometric placement regardless of the
        # board state or which piece is current. Placements that are illegal for
        # the current piece (rotation index out of range, off-board column, or
        # cannot spawn) are reported by ``action_masks`` and must be filtered by
        # a masking-capable policy (MaskablePPO). This replaces the previous
        # ``action % len(valid_actions)`` scheme, whose meaning shifted every
        # step and was effectively unlearnable.
        self.max_rotations = max(len(variants) for variants in PIECES.values())
        self.max_actions = self.max_rotations * self.width
        self.action_space = spaces.Discrete(self.max_actions)

        # Observation dimension depends on which optional components are enabled
        # (see ObservationConfig). Base = flattened grid + current-piece one-hot.
        obs_dim = self.height * self.width + len(self.piece_names)
        if self.observation_config.include_next_piece:
            obs_dim += len(self.piece_names)
        if self.observation_config.include_engineered_features:
            # per-column heights + holes + bumpiness
            obs_dim += self.width + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _decode_action(self, action: int) -> tuple[int, int]:
        rotation_idx, column = divmod(int(action), self.width)
        return rotation_idx, column

    def _sample_piece(self):
        idx = self.np_random.integers(len(self.piece_names))
        return self.piece_names[idx]

    def piece_one_hot(self, piece_name: str) -> np.ndarray:
        vec = np.zeros(len(self.piece_names), dtype=np.float32)
        vec[self.piece_names.index(piece_name)] = 1.0
        return vec

    def _engineered_features(self) -> np.ndarray:
        """Normalized board-quality features (heights, holes, bumpiness) in [0, 1]."""
        grid = self.board.grid

        heights = column_heights(grid).astype(np.float32) / self.height

        # Safe upper bounds keep each scalar within [0, 1].
        holes_norm = holes(grid) / max(1, (self.height - 1) * self.width)
        bumpiness_norm = bumpiness(grid) / max(1, (self.width - 1) * self.height)

        scalars = np.array([holes_norm, bumpiness_norm], dtype=np.float32)
        return np.clip(np.concatenate([heights, scalars]), 0.0, 1.0)

    def _get_observation(self):
        parts = [
            self.board.grid.flatten().astype(np.float32),
            self.piece_one_hot(self.current_piece_name),
        ]
        if self.observation_config.include_next_piece:
            parts.append(self.piece_one_hot(self.next_piece_name))
        if self.observation_config.include_engineered_features:
            parts.append(self._engineered_features())
        return np.concatenate(parts).astype(np.float32)

    def _is_valid_placement(self, rotation_idx: int, column: int, piece_name: str | None = None) -> bool:
        if piece_name is None:
            piece_name = self.current_piece_name

        variants = PIECES[piece_name]
        if rotation_idx < 0 or rotation_idx >= len(variants):
            return False

        piece = variants[rotation_idx]
        piece_width = len(piece[0])
        if column < 0 or column + piece_width > self.width:
            return False

        # piece must be able to spawn at the top of this column
        return not self.board.check_collision(piece, -len(piece), column)

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the fixed action set; True where the placement is legal.

        Consumed by MaskablePPO (and by mask-aware evaluation). On a dead board
        every action is masked out.
        """
        mask = np.zeros(self.max_actions, dtype=bool)
        if self.board.is_game_over():
            return mask

        for action in range(self.max_actions):
            rotation_idx, column = self._decode_action(action)
            if self._is_valid_placement(rotation_idx, column):
                mask[action] = True
        return mask

    def _enumerate_valid_actions(self, piece_name: str | None = None):
        """List of legal ``(rotation_idx, column)`` placements (used by the heuristic)."""
        if self.board.is_game_over():
            return []

        if piece_name is None:
            piece_name = self.current_piece_name

        variants = PIECES[piece_name]
        valid_actions = []

        for rotation_idx, piece in enumerate(variants):
            piece_width = len(piece[0])
            for column in range(self.width - piece_width + 1):
                if self.board.check_collision(piece, -len(piece), column):
                    continue
                valid_actions.append((rotation_idx, column))

        return valid_actions

    def get_valid_actions(self):
        return self._enumerate_valid_actions()

    def encode_action(self, rotation_idx: int, column: int) -> int:
        """Inverse of :meth:`_decode_action`."""
        return rotation_idx * self.width + column

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = Board(height=self.height, width=self.width)
        self.current_piece_name = self._sample_piece()
        self.next_piece_name = self._sample_piece()
        return self._get_observation(), {}

    def _drop_height(self, piece, column: int):
        row = -len(piece)
        while not self.board.check_collision(piece, row + 1, column):
            row += 1
        return row

    def _get_spawn_position(self, piece):
        spawn_row = -len(piece)
        spawn_col = (self.width - len(piece[0])) // 2
        return spawn_row, spawn_col

    def _can_spawn_piece(self, piece_name: str) -> bool:
        # the next piece can spawn if *any* of its rotations fits at the top
        variants = PIECES[piece_name]
        for piece in variants:
            for column in range(self.width - len(piece[0]) + 1):
                if not self.board.check_collision(piece, -len(piece), column):
                    return True
        return False

    def step(self, action: int):
        rc = self.reward_config

        # Stepping on an already-dead board.
        if self.board.is_game_over():
            return (
                self._get_observation(),
                -rc.game_over_penalty,
                True,
                False,
                {"game_over": True},
            )

        rotation_idx, column = self._decode_action(action)

        # Illegal placement. With action masking this is never reached; an
        # unmasked policy (DQN) ends the episode with a penalty, which keeps the
        # action semantics consistent and avoids infinite loops.
        if not self._is_valid_placement(rotation_idx, column):
            return (
                self._get_observation(),
                -rc.invalid_action_penalty,
                True,
                False,
                {"invalid_action": True},
            )

        piece = PIECES[self.current_piece_name][rotation_idx]

        grid_before = self.board.grid.copy()
        height_before = aggregate_height(grid_before)
        holes_before = holes(grid_before)
        bumpiness_before = bumpiness(grid_before)

        row = self._drop_height(piece, column)
        self.board.place_piece(piece, row, column)
        lines = self.board.clear_lines()

        grid_after = self.board.grid.copy()
        height_after = aggregate_height(grid_after)
        holes_after = holes(grid_after)
        bumpiness_after = bumpiness(grid_after)

        delta_height = height_before - height_after
        delta_holes = holes_before - holes_after
        delta_bumpiness = bumpiness_before - bumpiness_after

        reward = (
            rc.line_clear * lines
            + rc.valid_move
            + rc.delta_height * delta_height
            + rc.delta_holes * delta_holes
            + rc.delta_bumpiness * delta_bumpiness
        )

        # The previewed next piece becomes current; draw a fresh preview.
        upcoming_piece_name = self.next_piece_name
        terminated = not self._can_spawn_piece(upcoming_piece_name)
        truncated = False

        if terminated:
            reward -= rc.spawn_fail_penalty

        self.current_piece_name = upcoming_piece_name
        self.next_piece_name = self._sample_piece()

        obs = self._get_observation()
        info = {"lines_cleared": lines}
        if terminated:
            info["game_over"] = True

        return obs, reward, terminated, truncated, info
