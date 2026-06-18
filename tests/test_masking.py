import numpy as np

from tetris_rl.agents.heuristic import HeuristicAgent
from tetris_rl.config import RewardConfig
from tetris_rl.env.pieces import PIECES
from tetris_rl.env.tetris_env import TetrisEnv

# =========================
# ENCODE / DECODE TESTS
# =========================

def test_encode_decode_round_trip():
    env = TetrisEnv()
    env.reset()

    for rotation_idx in range(env.max_rotations):
        for column in range(env.width):
            action = env.encode_action(rotation_idx, column)

            assert 0 <= action < env.max_actions
            assert env._decode_action(action) == (rotation_idx, column)

def test_action_space_size_matches_encoding():
    env = TetrisEnv()

    # The fixed encoding spans max_rotations * width placements.
    assert env.max_actions == env.max_rotations * env.width
    assert env.action_space.n == env.max_actions

# =========================
# ACTION MASK TESTS
# =========================

def test_action_masks_shape_and_dtype():
    env = TetrisEnv()
    env.reset(seed=0)

    mask = env.action_masks()

    assert mask.shape == (env.max_actions,)
    assert mask.dtype == bool

def test_action_masks_match_enumerated_valid_actions():
    env = TetrisEnv()
    env.reset(seed=0)

    mask = env.action_masks()
    masked = set(np.flatnonzero(mask).tolist())
    enumerated = {
        env.encode_action(rotation_idx, column)
        for rotation_idx, column in env.get_valid_actions()
    }

    assert masked == enumerated
    assert len(masked) > 0

def test_action_masks_empty_on_game_over():
    env = TetrisEnv(height=4, width=4)
    env.reset()

    env.board.grid[:, :] = 1  # completely full -> game over

    mask = env.action_masks()

    assert not mask.any()

def test_every_masked_action_is_a_valid_placement():
    env = TetrisEnv()
    env.reset(seed=0)

    mask = env.action_masks()

    for action in np.flatnonzero(mask):
        rotation_idx, column = env._decode_action(int(action))
        assert env._is_valid_placement(rotation_idx, column)

# =========================
# INVALID ACTION TESTS
# =========================

def test_invalid_action_terminates_with_penalty():
    env = TetrisEnv(reward_config=RewardConfig(invalid_action_penalty=7.0))
    env.reset(seed=0)

    mask = env.action_masks()
    invalid_action = int(np.flatnonzero(~mask)[0])

    obs, reward, terminated, truncated, info = env.step(invalid_action)

    assert terminated is True
    assert truncated is False
    assert reward == -7.0
    assert info.get("invalid_action") is True

def test_valid_masked_action_is_accepted():
    env = TetrisEnv()
    env.reset(seed=0)

    mask = env.action_masks()
    valid_action = int(np.flatnonzero(mask)[0])

    obs, reward, terminated, truncated, info = env.step(valid_action)

    # A single legal placement on an empty board never ends the episode early
    # and is never reported as an invalid action.
    assert terminated is False
    assert "invalid_action" not in info
    assert "lines_cleared" in info

# =========================
# HEURISTIC / MASK CONSISTENCY
# =========================

def test_heuristic_action_is_always_in_mask():
    env = TetrisEnv()
    env.reset(seed=0)
    agent = HeuristicAgent()

    for _ in range(25):
        mask = env.action_masks()
        if not mask.any():
            break

        action = agent.select_action(env)
        assert mask[action]

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

# =========================
# REWARD CONFIG TESTS
# =========================

def _reward_for_clearing_one_line(line_clear: float) -> tuple[float, int]:
    """Drop a horizontal I across a width-4 board to clear exactly one line."""
    env = TetrisEnv(width=4, height=6, reward_config=RewardConfig(line_clear=line_clear))
    env.reset(seed=0)
    env.current_piece_name = "I"

    rotation_idx = next(
        i for i, v in enumerate(PIECES["I"]) if len(v) == 1 and len(v[0]) == 4
    )
    action = env.encode_action(rotation_idx, 0)

    _, reward, _, _, info = env.step(action)
    return reward, info["lines_cleared"]

def test_reward_uses_configured_line_clear_weight():
    reward_small, lines_small = _reward_for_clearing_one_line(50.0)
    reward_big, lines_big = _reward_for_clearing_one_line(1000.0)

    assert lines_small == 1
    assert lines_big == 1

    # Board, piece, seed and deltas are identical across both runs, so the only
    # difference in reward is the line-clear weight applied to the single line.
    assert reward_big - reward_small == 950.0
