"""Single entry point that runs both training pipelines (DQN then PPO).

Equivalent to running these two back to back:

    python -m tetris_rl.training.train_dqn
    python -m tetris_rl.training.train_ppo

Each uses its module default config (currently Experiment G), so this writes
``dqn_expG_*`` and ``ppo_expG_*`` checkpoints across all configured seeds.

Run from the repo root:

    python -m tetris_rl.training.train_all
"""

from tetris_rl.training.train_dqn import train as train_dqn
from tetris_rl.training.train_ppo import train as train_ppo


def train_all():
    print("\n########## DQN ##########")
    train_dqn()

    print("\n########## PPO ##########")
    train_ppo()

    print("\nAll training runs complete.")


if __name__ == "__main__":
    train_all()
