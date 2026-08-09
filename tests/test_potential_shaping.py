"""Tests for Experiment H: potential-based reward shaping from the heuristic.

The shaping term added to the reward is the Ng et al. (1999) potential-based
form ``F = potential_coef * (gamma * Phi(s') - Phi(s))`` where ``Phi`` is the
heuristic board score evaluated as a pure *state* function (no line-clear term).
Because it is potential-based it is policy-invariant: it densifies the learning
signal without changing the optimal policy.
"""

import numpy as np

from tetris_rl.agents.heuristic import HeuristicWeights, board_potential, score_board
from tetris_rl.config import DQN_EXPH, PPO_EXPH, RewardConfig
from tetris_rl.env.tetris_env import TetrisEnv


# =========================
# POTENTIAL FUNCTION
# =========================

def test_board_potential_is_state_only_heuristic_score():
    """Phi(grid) must equal the heuristic score with zero lines cleared."""
    weights = HeuristicWeights()
    grid = np.zeros((6, 4), dtype=np.int8)
    grid[5] = [1, 1, 0, 1]
    grid[4, 0] = 1

    assert board_potential(grid, weights) == score_board(
        grid, lines_cleared=0, weights=weights
    )


def test_board_potential_empty_board_is_zero():
    grid = np.zeros((20, 10), dtype=np.int8)

    assert board_potential(grid, HeuristicWeights()) == 0.0


def test_board_potential_penalizes_holes():
    """A board with a covered hole scores strictly lower than a clean one."""
    weights = HeuristicWeights()

    clean = np.zeros((6, 4), dtype=np.int8)
    clean[5, 0] = 1  # single block, no hole

    holey = np.zeros((6, 4), dtype=np.int8)
    holey[4, 0] = 1  # block floating above an empty cell -> one hole below

    assert board_potential(holey, weights) < board_potential(clean, weights)


# =========================
# SHAPING IN THE ENV
# =========================

def test_shaping_off_by_default():
    """Existing experiments (F/G) must be unaffected: shaping is opt-in."""
    assert RewardConfig().potential_shaping is False


def _step_O_at_origin(reward_config):
    """Drop an O at column 0 on an empty 20x10 board; return (reward, grid_after)."""
    env = TetrisEnv(reward_config=reward_config)
    env.reset(seed=0)
    env.current_piece_name = "O"

    action = env.encode_action(0, 0)
    _, reward, terminated, _, info = env.step(action)

    assert terminated is False
    assert info["lines_cleared"] == 0
    return reward, env.board.grid.copy()


def test_shaping_adds_potential_difference():
    """reward_on - reward_off == coef * (gamma * Phi(s') - Phi(s)).

    The transition (empty board -> O at col 0) is identical for both configs, so
    every non-shaping reward term cancels and only the potential term remains.
    Phi(s) on the empty board is 0, so the difference reduces to
    coef * gamma * Phi(grid_after).
    """
    weights = HeuristicWeights()
    off = RewardConfig()
    on = RewardConfig(potential_shaping=True, potential_coef=1.0, potential_gamma=0.99)

    reward_off, grid_off = _step_O_at_origin(off)
    reward_on, grid_on = _step_O_at_origin(on)

    # Same deterministic transition -> identical resulting board.
    assert np.array_equal(grid_off, grid_on)

    phi_before = 0.0  # empty board
    phi_after = board_potential(grid_on, weights)
    expected = on.potential_coef * (on.potential_gamma * phi_after - phi_before)

    assert np.isclose(reward_on - reward_off, expected)


def test_shaping_coefficient_scales_the_term():
    """Doubling potential_coef doubles the shaping contribution."""
    base = RewardConfig(potential_shaping=True, potential_coef=1.0)
    doubled = RewardConfig(potential_shaping=True, potential_coef=2.0)
    off = RewardConfig()

    reward_off, _ = _step_O_at_origin(off)
    reward_1x, _ = _step_O_at_origin(base)
    reward_2x, _ = _step_O_at_origin(doubled)

    term_1x = reward_1x - reward_off
    term_2x = reward_2x - reward_off

    assert np.isclose(term_2x, 2.0 * term_1x)


# =========================
# EXPERIMENT H CONFIGS
# =========================

def test_exph_configs_enable_shaping_on_flat_observation():
    for cfg in (DQN_EXPH, PPO_EXPH):
        assert cfg.experiment == "expH"
        assert cfg.reward.potential_shaping is True
        # Experiment H isolates the reward change: observation is the flat
        # Experiment F observation, so any gain is attributable to shaping.
        assert cfg.observation.include_engineered_features is False
        assert cfg.observation.include_next_piece is False


def test_exph_reuses_expf_hyperparameters():
    from tetris_rl.config import DQN_EXPF, PPO_EXPF

    assert DQN_EXPH.hyperparams == DQN_EXPF.hyperparams
    assert PPO_EXPH.hyperparams == PPO_EXPF.hyperparams
    assert DQN_EXPH.checkpoint_prefix() == "dqn_expH_1000000"
    assert PPO_EXPH.checkpoint_prefix() == "ppo_expH_1000000"
