import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from tetris_rl.config import PPO_EXPF, TrainConfig
from tetris_rl.env.tetris_env import TetrisEnv


def _mask_fn(env):
    return env.action_masks()


def make_env(seed, reward_config):
    env = TetrisEnv(reward_config=reward_config)
    env = ActionMasker(env, _mask_fn)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train(config: TrainConfig = PPO_EXPF):
    os.makedirs("./results/checkpoints", exist_ok=True)
    os.makedirs("./results/tb/ppo", exist_ok=True)

    for seed in config.seeds:
        print(f"\n=== Training MaskablePPO ({config.experiment}) with seed={seed} ===")

        env = make_env(seed, config.reward)

        model = MaskablePPO(
            "MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=f"./results/tb/ppo/seed_{seed}/",
            seed=seed,
            device="cpu",  # recommended for MLP policies on small, non-image observations
            **config.hyperparams,
        )

        model.learn(total_timesteps=config.total_timesteps)

        save_path = config.checkpoint_path(seed)
        model.save(save_path)
        print(f"Saved model to: {save_path}.zip")

        env.close()


if __name__ == "__main__":
    train()
