import numpy as np
from gymnasium.spaces import Discrete

from tetris_rl.env.tetris_env import TetrisEnv

def test_reset_returns_observation_and_info():
    env = TetrisEnv()
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

def test_reset_observation_has_expected_shape():
    env = TetrisEnv()
    obs, _ = env.reset()

    expected_dim = env.height * env.width + len(env.piece_names)
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32

def test_action_space_matches_expected_size():
    env = TetrisEnv()

    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == env.max_actions
    assert env.max_actions > 0

def test_get_valid_actions_returns_nonempty_list_after_reset():
    env = TetrisEnv()
    env.reset()

    valid_actions = env.get_valid_actions()

    assert isinstance(valid_actions, list)
    assert len(valid_actions) > 0

    for action in valid_actions:
        assert isinstance(action, tuple)
        assert len(action) == 2

def test_step_returns_gymnasium_tuple():
    env = TetrisEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_reset_clears_board():
    env = TetrisEnv()
    env.reset()

    env.board.grid[0, 0] = 1
    assert env.board.grid.sum() == 1

    env.reset()
    assert env.board.grid.sum() == 0

def test_get_valid_actions_are_within_piece_bounds():
    env = TetrisEnv(width=4)
    env.reset()

    valid_actions = env.get_valid_actions()

    for rotation_idx, column in valid_actions:
        piece = env.current_piece[rotation_idx]
        piece_width = len(piece[0])

        assert column >= 0
        assert column + piece_width <= env.width

def test_step_handles_action_larger_than_valid_action_list():
    env = TetrisEnv(width=4)
    env.reset()

    valid_actions = env.get_valid_actions()
    assert len(valid_actions) > 0

    # intentionally choose a very large action index
    large_index = 999

    obs, reward, terminated, truncated, info = env.step(large_index)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_step_returns_game_over_when_no_valid_actions_exist():
    env = TetrisEnv(height=4, width=4)
    env.reset()

    # Fill board completely so no piece can spawn anywhere
    env.board.grid[:, :] = 1

    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(obs, np.ndarray)
    assert reward == -10.0
    assert terminated is True
    assert truncated is False
    assert info.get("game_over") is True
