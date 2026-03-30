import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from tetris_rl.env.tetris_env import TetrisEnv

TOTAL_TIMESTEPS = 1_000_000
SEEDS = [0, 1, 2]

def make_env(seed):
    env = TetrisEnv()
    env.reset(seed=seed)
    env = Monitor(env)
    return env

def train():
    os.makedirs("./results/checkpoints", exist_ok=True)
    os.makedirs("./results/tb/ppo", exist_ok=True)

    for seed in SEEDS:
        print(f"\n=== Training PPO (Experiment D) with seed={seed} ===")

        env = make_env(seed)

        model = PPO(
            'MlpPolicy',
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"./results/tb/ppo/seed_{seed}/",
            seed=seed,
            device="cpu", # recommended for PPO + MLP/CNN non-image-ish small models
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        save_path = f"./results/checkpoints/ppo_expD_{TOTAL_TIMESTEPS}_seed{seed}"
        model.save(save_path)
        print(f"Saved model to: {save_path}.zip")

        env.close()

if __name__ == "__main__":
    train()
