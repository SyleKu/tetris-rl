import numpy as np

from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.env.pieces import PIECES


def test_reset_returns_observation_and_info():
    env = TetrisEnv()
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

def test_reset_observation_has_expected_shape():
    env = TetrisEnv()
    obs, _ = env.reset()

    assert obs.shape == (6,)
    assert obs.dtype == np.float32

def test_action_space_matches_expected_size():
    env = TetrisEnv()

    assert (env.max_actions == env.max_rotations * env.width)
    assert env.action_space.n == env.max_actions

def test_decode_action_returns_rotation_and_column():
    env = TetrisEnv(width=10)

    rotation, column = env.decode_action(23)

    assert rotation == 2
    assert column == 3

def test_step_returns_gymnasium_tuple():
    env = TetrisEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, float)
    assert isinstance(terminated, float)
    assert isinstance(info, dict)

def test_invalid_actions_sets_info_flag():
    env = TetrisEnv(width=4)
    env.reset()

    # Force a piece with width 4, so placing it starting at column 1 is invalid
    env.current_piece_name = "I"

    # action = rotation * width + column
    # rotation 0 -> horizontal I piece = [[1,1,1,1]]
    # column 1 -> invalid because 1 + 4 > 4
    action = 0 * env.width + 1

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert reward == -2.0
    assert terminated is False
    assert truncated is False
    assert info.get("invalid_action") is True

def test_game_over_when_spawn_position_is_blocked():
    env = TetrisEnv(height=6, width=6)
    env.reset()

    env.current_piece_name = "O"
    env.board.grid[0, 0] = 1
    env.board.grid[0, 1] = 1

    action = 0 # rotation 0, column 0

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert terminated is True
    assert truncated is False
    assert info.get("game_over") is True

def test_reset_clears_board():
    env = TetrisEnv()
    env.reset()

    env.board.grid[0, 0] = 1
    assert env.board.grid.sum() == 1

    env.reset()
    assert env.board.grid.sum() == 0

def test_step_info_contains_lines_cleared_for_valid_move():
    env = TetrisEnv(height=4, width=4)
    env.reset()

    # Prepare board so an O piece clears the bottom two rows at columns 2 and 3
    env.board.grid[2] = [1, 1, 0, 0]
    env.board.grid[3] = [1, 1, 0, 0]

    env.current_piece_name = "O"

    action = 2 # rotation 0, column 2

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert "lines_cleared" in info
    assert info["lines_cleared"] == 2

def test_current_piece_after_reset_is_valid():
    env = TetrisEnv()
    env.reset()

    assert env.current_piece_name in PIECES
