from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from tetris_rl.env.tetris_env import TetrisEnv

def main():
    env = Monitor(TetrisEnv())

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        tensorboard_log="./results/tb/dqn/",
    )

    model.learn(total_timesteps=300_000)
    model.save("./results/checkpoints/dqn_tetris")

if __name__ == "__main__":
    main()
