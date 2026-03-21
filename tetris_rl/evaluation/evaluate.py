import numpy as np
from stable_baselines3 import DQN

from tetris_rl.env.tetris_env import TetrisEnv

def evaluate(model_path: str, episodes: int = 20):
    env = TetrisEnv()
    model = DQN.load(model_path)

    rewards = []
    lines = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        total_lines = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_lines += info.get('lines_cleared', 0)

        rewards.append(total_reward)
        lines.append(total_lines)

    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average lines: {np.mean(lines):.2f}")

def main():
    evaluate(model_path="./results/checkpoints/dqn_tetris")

if __name__ == "__main__":
    main()