import numpy as np
from gymnasium.spaces import Box, Discrete

from tetris_rl.env.pieces import PIECES
from tetris_rl.env.tetris_env import TetrisEnv

# =========================
# BASIC RESET TESTS
# =========================

def test_reset_returns_observation_and_info():
    env = TetrisEnv()
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

def test_observation_space_is_box():
    env = TetrisEnv()

    assert isinstance(env.observation_space, Box)

def test_reset_observation_shapes():
    env = TetrisEnv()
    obs, _ = env.reset()

    expected_dim = env.height * env.width + len(env.piece_names)

    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32

# =========================
# BOARD TESTS
# =========================

def test_board_is_empty_after_reset():
    env = TetrisEnv()
    obs, _ = env.reset()

    assert np.sum(env.board.grid) == 0

def test_board_contains_only_binary_values():
    env = TetrisEnv()
    env.reset()

    board = env.board.grid
    unique_values = np.unique(board)

    assert np.all(np.isin(unique_values, [0.0, 1.0]))

def test_reset_clears_board():
    env = TetrisEnv()
    env.reset()

    env.board.grid[0, 0] = 1
    assert env.board.grid.sum() == 1

    env.reset()
    assert env.board.grid.sum() == 0

# =========================
# PIECE TESTS
# =========================

def test_piece_one_hot_is_encoded_in_observation():
    env = TetrisEnv()
    obs, _ = env.reset()

    piece_vec = obs[-len(env.piece_names):]

    assert np.sum(piece_vec) == 1.0
    assert np.count_nonzero(piece_vec) == 1

def test_piece_vector_size_in_observation():
    env = TetrisEnv()
    obs, _ = env.reset()

    piece_vec = obs[-len(env.piece_names):]
    assert len(piece_vec) == len(env.piece_names)

# =========================
# ACTION SPACE TESTS
# =========================

def test_action_space_is_discrete():
    env = TetrisEnv()
    obs, _ = env.reset()

    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == env.max_actions
    assert env.max_actions > 0

def test_get_valid_actions_returns_list():
    env = TetrisEnv()
    env.reset()

    actions = env.get_valid_actions()

    assert isinstance(actions, list)
    assert len(actions) > 0

def test_valid_actions_are_well_formed():
    env = TetrisEnv()
    env.reset()

    actions = env.get_valid_actions()

    for action in actions:
        assert isinstance(action, tuple)
        assert len(action) == 2

def test_valid_actions_respect_board_width():
    env = TetrisEnv(width=4)
    env.reset()

    actions = env.get_valid_actions()

    for rotation_idx, column in actions:
        piece = PIECES[env.current_piece_name][rotation_idx]
        piece_width = len(piece[0])

        assert column >= 0
        assert column + piece_width <= env.width

# =========================
# STEP TESTS
# =========================

def test_step_returns_correct_tuple():
    env = TetrisEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_step_observation_shape():
    env = TetrisEnv()
    env.reset()

    obs, _, _, _, _ = env.step(0)

    expected_dim = env.height * env.width + len(env.piece_names)
    assert obs.shape == (expected_dim,)

def test_steps_handles_large_action_index():
    env = TetrisEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(999)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_step_changes_board_state():
    env = TetrisEnv()
    env.reset()

    board_before = env.board.grid.copy()

    env.step(0)

    board_after = env.board.grid

    assert not np.array_equal(board_before, board_after)

# =========================
# GAME OVER TESTS
# =========================

def test_game_over_when_no_valid_actions():
    env = TetrisEnv(height=4, width=4)
    env.reset()

    env.board.grid[:, :] = 1 # completely full board

    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(obs, np.ndarray)
    assert reward == -10.0
    assert terminated is True
    assert truncated is False
    assert info.get("game_over") is True

def test_step_after_game_over():
    env = TetrisEnv(height=4, width=4)
    env.reset()

    # force game over
    env.board.grid[:, :] = 1

    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(obs, np.ndarray)
    assert reward == -10.0
    assert terminated is True
    assert truncated is False
    assert info.get("game_over") is True
