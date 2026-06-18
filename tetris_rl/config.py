"""Central configuration for environment rewards and training runs.

Experiments in this project are a sequence of controlled changes (see
``docs/research_log.md``). Previously the reward weights and hyperparameters
lived as inline magic numbers in the env/training scripts, so only the *latest*
experiment survived in code. Externalizing them here means a past experiment can
be reproduced by instantiating the matching config instead of doing git
archaeology.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """Reward-shaping coefficients used by :class:`TetrisEnv`.

    The functional form is::

        reward = line_clear * lines_cleared
               + valid_move
               + delta_height    * (height_before    - height_after)
               + delta_holes     * (holes_before     - holes_after)
               + delta_bumpiness * (bumpiness_before - bumpiness_after)

    plus ``-spawn_fail_penalty`` when the next piece cannot spawn after a
    placement, ``-game_over_penalty`` when a step is taken on an already-dead
    board, and ``-invalid_action_penalty`` when an unmasked policy selects an
    illegal placement (only reachable without action masking, e.g. DQN).

    Defaults reproduce the "Experiment C/D" reward (Experiment D kept C's
    reward and only changed the observation).
    """

    line_clear: float = 50.0
    valid_move: float = 0.1
    delta_height: float = 0.02
    delta_holes: float = 0.1
    delta_bumpiness: float = 0.02

    spawn_fail_penalty: float = 5.0
    game_over_penalty: float = 10.0
    invalid_action_penalty: float = 10.0


# Named reward presets. Earlier experiments (A/B) used absolute board penalties
# rather than the delta-based form the env now implements, so only the
# delta-based experiments are expressible as presets here.
REWARD_PRESETS: dict[str, RewardConfig] = {
    # Experiment C: line clearing made dominant, auxiliary deltas reduced.
    "expC": RewardConfig(),
    # Experiment D: identical reward to C (the change in D was the observation).
    "expD": RewardConfig(),
    # An example tuning direction from the README "Future Work" (stronger
    # discouragement of holes/height); kept here as a starting point, not a
    # validated result.
    "expD_strong_penalty": RewardConfig(delta_holes=0.3, delta_height=0.1),
}


@dataclass
class TrainConfig:
    """Everything needed to reproduce a training run."""

    algo: str  # "dqn" or "ppo"
    experiment: str = "expD"
    total_timesteps: int = 1_000_000
    seeds: tuple[int, ...] = (0, 1, 2)
    reward: RewardConfig = field(default_factory=RewardConfig)
    hyperparams: dict = field(default_factory=dict)

    def checkpoint_prefix(self) -> str:
        """Prefix shared by every seed's checkpoint, e.g. ``ppo_expF_1000000``."""
        return f"{self.algo}_{self.experiment}_{self.total_timesteps}"

    def checkpoint_path(self, seed: int, checkpoint_dir: str = "./results/checkpoints") -> str:
        return f"{checkpoint_dir}/{self.checkpoint_prefix()}_seed{seed}"


# --- Concrete configs for the runs reported in the README / research log ------
# Experiment F (action masking + fixed encoding) reuses the Experiment D reward;
# only the action space / training tooling changed, so the reward defaults below
# are still the "expD" reward (see research_log.md).

DQN_EXPF = TrainConfig(
    algo="dqn",
    experiment="expF",
    hyperparams=dict(
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.15,
        exploration_final_eps=0.02,
    ),
)

PPO_EXPF = TrainConfig(
    algo="ppo",
    experiment="expF",
    hyperparams=dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
)
