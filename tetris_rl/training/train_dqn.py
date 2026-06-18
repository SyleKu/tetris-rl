import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from tetris_rl.config import DQN_EXPF, TrainConfig
from tetris_rl.env.tetris_env import TetrisEnv

# NOTE: stable-baselines3 has no action-masking-capable DQN, so DQN trains on
# the env without a mask. The fixed action<->placement mapping still makes the
# action semantics consistent (a big improvement over the old modulo scheme),
# but the agent must learn to avoid illegal placements, which the env penalizes
# and terminates on. MaskablePPO (train_ppo.py) is the recommended setup.


def make_env(seed, reward_config):
    env = TetrisEnv(reward_config=reward_config)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train(config: TrainConfig = DQN_EXPF):
    os.makedirs("./results/checkpoints", exist_ok=True)
    os.makedirs("./results/tb/dqn", exist_ok=True)

    for seed in config.seeds:
        print(f"\n=== Training DQN ({config.experiment}) with seed={seed} ===")

        env = make_env(seed, config.reward)

        model = DQN(
            "MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=f"./results/tb/dqn/seed_{seed}/",
            seed=seed,
            **config.hyperparams,
        )

        model.learn(total_timesteps=config.total_timesteps)

        save_path = config.checkpoint_path(seed)
        model.save(save_path)
        print(f"Saved model to: {save_path}.zip")

        env.close()


if __name__ == "__main__":
    train()
