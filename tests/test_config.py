from dataclasses import FrozenInstanceError

import pytest

from tetris_rl.config import (
    DQN_EXPF,
    PPO_EXPF,
    REWARD_PRESETS,
    RewardConfig,
    TrainConfig,
)

# =========================
# CHECKPOINT NAMING TESTS
# =========================

def test_checkpoint_prefix_format():
    config = TrainConfig(algo="ppo", experiment="expF", total_timesteps=1_000_000)

    assert config.checkpoint_prefix() == "ppo_expF_1000000"

def test_checkpoint_path_includes_prefix_seed_and_dir():
    config = TrainConfig(algo="dqn", experiment="expF", total_timesteps=500)

    path = config.checkpoint_path(seed=2, checkpoint_dir="/tmp/ckpts")

    assert path == "/tmp/ckpts/dqn_expF_500_seed2"

# =========================
# REWARD CONFIG TESTS
# =========================

def test_reward_config_is_frozen():
    reward = RewardConfig()

    with pytest.raises(FrozenInstanceError):
        reward.line_clear = 1.0  # type: ignore[misc]

def test_reward_presets_are_reward_configs():
    assert REWARD_PRESETS  # non-empty
    for name, preset in REWARD_PRESETS.items():
        assert isinstance(preset, RewardConfig), name

def test_strong_penalty_preset_overrides_defaults():
    preset = REWARD_PRESETS["expD_strong_penalty"]

    assert preset.delta_holes == 0.3
    assert preset.delta_height == 0.1

# =========================
# CONCRETE TRAIN CONFIG TESTS
# =========================

def test_expf_configs_target_experiment_f():
    assert DQN_EXPF.experiment == "expF"
    assert PPO_EXPF.experiment == "expF"
    assert DQN_EXPF.algo == "dqn"
    assert PPO_EXPF.algo == "ppo"

def test_expf_configs_carry_hyperparameters():
    assert DQN_EXPF.hyperparams
    assert PPO_EXPF.hyperparams
    assert DQN_EXPF.checkpoint_prefix() == "dqn_expF_1000000"
    assert PPO_EXPF.checkpoint_prefix() == "ppo_expF_1000000"
