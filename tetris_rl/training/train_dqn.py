import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from tetris_rl.env.tetris_env import TetrisEnv

TOTAL_TIMESTEPS = 300_000
SEEDS = [0, 1, 2]

def make_env(seed):
    env = TetrisEnv()
    env.reset(seed=seed)
    env = Monitor(env)
    return env

def train():
    os.makedirs("./results/checkpoints", exist_ok=True)
    os.makedirs("./results/tb/dqn", exist_ok=True)

    for seed in SEEDS:
        print(f"\n=== Training DQN (Experiment D) with seed={seed} ===")

        env = make_env(seed)

        model = DQN(
            "MlpPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.15,
            exploration_final_eps=0.02,
            verbose=1,
            tensorboard_log=f"./results/tb/dqn/seed_{seed}/",
            seed=seed,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        save_path = f"./results/checkpoints/dqn_expD_{TOTAL_TIMESTEPS}_seed{seed}"
        model.save(save_path)
        print(f"Saved model to: {save_path}.zip")

        env.close()

if __name__ == "__main__":
    train()
