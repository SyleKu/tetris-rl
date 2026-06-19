import numpy as np

from tetris_rl.config import ObservationConfig
from tetris_rl.env.tetris_env import TetrisEnv

# =========================
# DEFAULT (FLAT) OBSERVATION
# =========================

def test_default_observation_is_flat_grid_plus_piece():
    env = TetrisEnv()
    obs, _ = env.reset(seed=0)

    expected = env.height * env.width + len(env.piece_names)
    assert obs.shape == (expected,)
    assert env.observation_space.shape == (expected,)
    assert obs.dtype == np.float32

# =========================
# ENRICHED OBSERVATION (EXPERIMENT G)
# =========================

def _enriched_env():
    cfg = ObservationConfig(include_engineered_features=True, include_next_piece=True)
    return TetrisEnv(observation_config=cfg)

def test_enriched_observation_dimension():
    env = _enriched_env()
    obs, _ = env.reset(seed=0)

    n = len(env.piece_names)
    # grid + current one-hot + next one-hot + heights + holes + bumpiness
    expected = env.height * env.width + n + n + env.width + 2

    assert obs.shape == (expected,)
    assert env.observation_space.shape == (expected,)

def test_engineered_features_only_dimension():
    env = TetrisEnv(observation_config=ObservationConfig(include_engineered_features=True))
    obs, _ = env.reset(seed=0)

    n = len(env.piece_names)
    expected = env.height * env.width + n + env.width + 2
    assert obs.shape == (expected,)

def test_next_piece_only_dimension():
    env = TetrisEnv(observation_config=ObservationConfig(include_next_piece=True))
    obs, _ = env.reset(seed=0)

    n = len(env.piece_names)
    expected = env.height * env.width + n + n
    assert obs.shape == (expected,)

def test_enriched_observation_stays_within_unit_box():
    env = _enriched_env()
    obs, _ = env.reset(seed=0)

    for _ in range(20):
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

        mask = env.action_masks()
        if not mask.any():
            break
        action = int(np.flatnonzero(mask)[0])
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

def test_next_piece_one_hot_is_present_and_valid():
    env = TetrisEnv(observation_config=ObservationConfig(include_next_piece=True))
    obs, _ = env.reset(seed=0)

    n = len(env.piece_names)
    grid_size = env.height * env.width
    current = obs[grid_size:grid_size + n]
    next_piece = obs[grid_size + n:grid_size + 2 * n]

    assert current.sum() == 1.0 and np.count_nonzero(current) == 1
    assert next_piece.sum() == 1.0 and np.count_nonzero(next_piece) == 1

def test_engineered_heights_reflect_board_state():
    env = TetrisEnv(observation_config=ObservationConfig(include_engineered_features=True))
    env.reset(seed=0)

    n = len(env.piece_names)
    grid_size = env.height * env.width

    # Empty board -> all engineered features are zero.
    obs_empty = env._get_observation()
    assert np.all(obs_empty[grid_size + n:] == 0.0)

    # A filled bottom-left cell -> first column height becomes positive.
    env.board.grid[-1, 0] = 1
    obs = env._get_observation()
    heights = obs[grid_size + n:grid_size + n + env.width]
    assert heights[0] > 0.0
