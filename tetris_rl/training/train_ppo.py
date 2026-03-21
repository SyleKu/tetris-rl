from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from tetris_rl.env.tetris_env import TetrisEnv

def main():
    env = Monitor(TetrisEnv())

    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./results/tb/ppo/",
    )

    model.learn(total_timesteps=300_000)
    model.save("./results/checkpoints/ppo_tetris")

if __name__ == "__main__":
    main()
